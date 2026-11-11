"""Named external I/O without policy weights or transport."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from lerobot_engine.channel_mapping import ChannelMapping
from lerobot_engine.adapters.wall_x import validate_checkpoint


def write_mapping(directory, mapping):
    (directory / "cyclo_io_mapping.json").write_text(json.dumps(mapping))


def robot():
    groups = {
        "follower_body": {"role": "follower", "msg_type": "sensor_msgs/msg/JointState", "topic": "/joints",
                          "joint_names": [f"j{i}" for i in range(19)]},
        "follower_arm": {"role": "follower", "parent": "follower_body", "joint_names": [f"j{i}" for i in range(16)]},
        "follower_head": {"role": "follower", "parent": "follower_body", "joint_names": ["j16", "j17"]},
        "follower_lift": {"role": "follower", "parent": "follower_body", "joint_names": ["j18"]},
    }
    commands = {name: {"msg_type": "trajectory_msgs/msg/JointTrajectory", "joint_names": groups[f"follower_{name}"]["joint_names"]}
                for name in ("arm", "head", "lift")}
    commands["mobile"] = {"msg_type": "geometry_msgs/msg/Twist", "joint_names": ["linear_x", "linear_y", "angular_z"]}
    return SimpleNamespace(_config={"joint_groups": groups, "sensors": {"odom": {}}}, _action_groups=commands,
                           get_joint_names=lambda group: groups[group]["joint_names"])


def config(state, action):
    return SimpleNamespace(input_features={"observation.state": {"shape": [state]}},
                           output_features={"action": {"shape": [action]}})


def metadata(state=None, action=None):
    return {"version": 1, "state_names": state or [f"j{i}" for i in range(16)],
            "action_names": action or [f"j{i}" for i in range(16)]}


def test_22_channel_robot_16_channel_policy_and_wall_limit(tmp_path):
    bot = robot()
    mapping = ChannelMapping(bot, config(16, 16), metadata(), [])
    assert mapping.action_keys == ["arm"]
    assert mapping.state_sources == ("joint:follower_cyclo_input_0",)
    assert len(mapping.joint_views["follower_cyclo_input_0"]["joint_names"]) == 16
    validate_checkpoint(vars(config(16, 16)), tmp_path)
    np.testing.assert_array_equal(mapping.state((np.arange(16),)), np.arange(16))


def test_independent_state_selection_and_action_reordering():
    names = [f"j{i}" for i in reversed(range(16))]
    mapping = ChannelMapping(robot(), config(3, 16), metadata(["j17", "linear_y", "j0"], names), [])
    assert mapping.state_sources == ("joint:follower_cyclo_input_0", "sensor:odom.linear_velocity")
    np.testing.assert_array_equal(mapping.state(([17., 0.], [100., 200., 300.])), [17., 200., 0.])
    prediction = np.array([list(reversed(range(16))), list(reversed(range(16, 32)))])
    original = prediction.copy()
    np.testing.assert_array_equal(mapping.action(prediction), np.arange(32).reshape(2, 16))
    np.testing.assert_array_equal(prediction, original)


@pytest.mark.parametrize("state_dim,action_dim", [(17, 16), (15, 16), (16, 17), (16, 15)])
def test_mapped_dimensions_must_match_checkpoint(state_dim, action_dim):
    with pytest.raises(ValueError, match="checkpoint requires"):
        ChannelMapping(robot(), config(state_dim, action_dim), metadata(), [])


@pytest.mark.parametrize("change,reason", [
    ({"state_names": ["unknown"]}, "Unknown state"),
    ({"action_names": ["unknown"]}, "Unknown action"),
    ({"action_names": ["j0"]}, "Partial action group"),
    ({"state_names": ["j0", "j0"]}, "duplicate"),
    ({"action_names": []}, "nonempty"),
])
def test_bad_mapping_rejected(change, reason):
    data = {**metadata(), **change}
    with pytest.raises(ValueError, match=reason):
        ChannelMapping(robot(), config(len(data["state_names"]), max(1, len(data["action_names"]))), data, [])


def test_legacy_exact_dimensions_only(caplog):
    mapping = ChannelMapping(robot(), config(22, 22), None, ["arm", "head", "lift", "mobile"])
    assert mapping.action_keys == ["arm", "head", "lift", "mobile"]
    assert "equal dimensions do not prove" in caplog.text
    for state, action in [(16, 22), (22, 16), (24, 22)]:
        with pytest.raises(ValueError, match="checkpoint requires"):
            ChannelMapping(robot(), config(state, action), None, ["arm", "head", "lift", "mobile"])


def test_ambiguous_source_and_invalid_values_rejected():
    bot = robot()
    bot._config["joint_groups"]["follower_other"] = dict(bot._config["joint_groups"]["follower_body"], topic="/other")
    with pytest.raises(ValueError, match="Ambiguous"):
        ChannelMapping(bot, config(16, 16), metadata(), [])
    mapping = ChannelMapping(robot(), config(16, 16), metadata(), [])
    with pytest.raises(ValueError, match="finite"):
        mapping.state((np.full(16, np.nan),))
    for values in (np.zeros((1, 22)), np.full((1, 16), np.inf)):
        with pytest.raises(ValueError, match="Postprocessed action"):
            mapping.action(values)


def test_relative_processor_prefix_contract():
    class RelativeActionsProcessorStep:
        enabled = True

        def get_config(self):
            return {"exclude_joints": [], "action_names": None}

    proc = SimpleNamespace(steps=[RelativeActionsProcessorStep()])
    mapping = ChannelMapping(robot(), config(16, 16), metadata(), [])
    mapping.validate_processors(proc)
    mapping = ChannelMapping(robot(), config(16, 16), metadata(action=[f"j{i}" for i in reversed(range(16))]), [])
    with pytest.raises(ValueError, match="prefix mismatch"):
        mapping.validate_processors(proc)


@pytest.mark.parametrize("labels", [list(reversed(range(16))), list(range(15)), list(range(17))])
@pytest.mark.parametrize("exclude_joints", [[], ["j0"]])
def test_relative_processor_rejects_inconsistent_channel_labels(labels, exclude_joints):
    class RelativeActionsProcessorStep:
        enabled = True

        def get_config(self):
            return {"exclude_joints": exclude_joints, "action_names": [f"j{i}" for i in labels]}

    mapping = ChannelMapping(robot(), config(16, 16), metadata(), [])
    with pytest.raises(ValueError, match="action names differ"):
        mapping.validate_processors(SimpleNamespace(steps=[RelativeActionsProcessorStep()]))


@pytest.mark.parametrize("labelled", [True, False])
def test_relative_exclusions_require_matching_processor_labels(labelled):
    data = metadata(state=["j18"] + [f"j{i}" for i in range(1, 16)])

    class RelativeActionsProcessorStep:
        enabled = True

        def get_config(self):
            return {"exclude_joints": ["J0"], "action_names": data["action_names"] if labelled else None}

    mapping = ChannelMapping(robot(), config(16, 16), data, [])
    proc = SimpleNamespace(steps=[RelativeActionsProcessorStep()])
    if labelled:
        mapping.validate_processors(proc)
    else:
        with pytest.raises(ValueError, match="prefix mismatch"):
            mapping.validate_processors(proc)


def test_disabled_relative_processor_does_not_constrain_channel_order():
    class RelativeActionsProcessorStep:
        enabled = False

        def get_config(self):
            raise AssertionError("Disabled relative transform must not be inspected")

    mapping = ChannelMapping(robot(), config(16, 16), metadata(action=[f"j{i}" for i in reversed(range(16))]), [])
    mapping.validate_processors(SimpleNamespace(steps=[RelativeActionsProcessorStep()]))


def test_mapping_reads_again_after_file_changes(tmp_path):
    from cyclo_lerobot_io.mapping import read_mapping
    write_mapping(tmp_path, metadata())
    first = ChannelMapping(robot(), config(16, 16), read_mapping(tmp_path), [])
    write_mapping(tmp_path, metadata(action=[f"j{i}" for i in reversed(range(16))]))
    second = ChannelMapping(robot(), config(16, 16), read_mapping(tmp_path), [])
    assert first.action_names != second.action_names


def test_real_engine_callbacks_subset_and_cached_load(tmp_path, monkeypatch):
    import json
    from unittest import mock
    import torch
    from test_temporal_engine import RobotClient, robot_client_impl, engine_module

    bots = []

    def make_robot(robot_type, **kwargs):
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            bot = RobotClient(robot_type, **kwargs)
        bot._config["cameras"] = {}
        names = (bot.get_joint_names("follower_arm_left") + bot.get_joint_names("follower_arm_right"))
        start = bot.start_observation_subscriptions

        def subscribe(**selection):
            with mock.patch.object(robot_client_impl, "ROS2Subscriber", return_value=SimpleNamespace(close=lambda: None)):
                start(**selection)
            # Unused head/lift/odom never arrive. ROS names are deliberately reversed.
            bot._update_joint("follower_upper_body", SimpleNamespace(name=list(reversed(names)),
                              position=list(reversed(range(16))), velocity=[], effort=[]))

        bot.start_observation_subscriptions = subscribe
        bots.append(bot)
        return bot

    with mock.patch.object(RobotClient, "_init_subscriptions"):
        reference = RobotClient("ffw_sg2_rev1")
    names = reference.get_joint_names("follower_arm_left") + reference.get_joint_names("follower_arm_right")
    reference.close()
    data = {"version": 1, "state_names": names, "action_names": names}
    write_mapping(tmp_path, data)
    checkpoint = {"type": "act", "input_features": {"observation.state": {"shape": [16]}},
                  "output_features": {"action": {"shape": [16]}}}
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    cfg = SimpleNamespace(type="act", input_features={"observation.state": SimpleNamespace(shape=(16,))},
                          output_features={"action": SimpleNamespace(shape=(16,))})
    observed = []

    def predict(batch):
        observed.append(batch["observation.state"].clone())
        return torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)

    policy = SimpleNamespace(config=cfg, predict_action_chunk=predict, reset=mock.Mock())
    class Processor:
        steps = []

        def __call__(self, batch):
            return batch

    engine = engine_module.LeRobotEngine()
    engine._load_policy_assets = mock.Mock(return_value=(policy, Processor(), lambda a: a * 10))
    monkeypatch.setitem(engine._init_robot.__globals__, "RobotClient", make_robot)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = SimpleNamespace(model_path=str(tmp_path), robot_type="ffw_sg2_rev1")
    try:
        for reverse in (False, True):
            if reverse:
                write_mapping(tmp_path, {**data, "action_names": names[::-1]})
            loaded = engine.load_policy(request)
            assert loaded["success"], loaded
            assert loaded["action_keys"] == ["arm_left", "arm_right"]
            result = engine.get_action_chunk(SimpleNamespace(task_instruction="test"))
            assert result["success"], result
            expected = np.arange(16) * 10
            np.testing.assert_array_equal(result["action_chunk"], expected[::-1] if reverse else expected)
            torch.testing.assert_close(observed[-1], torch.arange(16, dtype=torch.float32).reshape(1, 16))
            assert bots[-1]._subscription_selection["sensors"] == frozenset()
            assert bots[-1]._subscription_selection["joint_groups"] == {"follower_cyclo_input_0"}
        engine._load_policy_assets.assert_called_once()
        assert bots[0]._closed
        (tmp_path / "cyclo_io_mapping.json").write_text("invalid json")
        failed = engine.load_policy(request)
        assert not failed["success"] and "Invalid channel metadata" in failed["message"]
        assert not engine.is_ready and engine._channel_mapping is None
        assert all(bot._closed for bot in bots)
    finally:
        engine.cleanup()
