import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rldx_mapping", ROOT / "rldx_engine/mapping.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RobotMapping = module.RobotMapping


def fixture():
    metadata = {"state_names": ["j2", "j1", "linear_x"], "action_names": ["j2", "j1"]}
    robot = {
        "cameras": {"head": {}, "wrist": {"rotation_deg": 270}, "unused": {}},
        "joint_groups": {"follower_arm": {"role": "follower", "joint_names": ["j1", "j2"]}},
        "sensors": {"odom": {}},
    }
    actions = {"arm": {"joint_names": ["j1", "j2"]}}
    cameras = {"head": "head", "wrist": "wrist"}
    return metadata, robot, actions, cameras


def test_explicit_names_and_only_required_cameras():
    mapping = RobotMapping(*fixture())
    snapshot = {
        "images": {"head": np.zeros((4, 6, 3), np.uint8), "wrist": np.zeros((3, 8, 3), np.uint8)},
        "joint_positions": {"follower_rldx_input_0": [2., 1.]},
        "sensors": {"odom": {"linear_velocity": [3., 0., 0.]}},
    }
    obs = mapping.observation(snapshot, "pick", "task")
    assert obs["video"]["head"].shape == (1, 1, 4, 6, 3)
    assert obs["video"]["wrist"].shape == (1, 1, 8, 3, 3)
    np.testing.assert_array_equal(obs["state"]["joint_position"], [[[2., 1., 3.]]])
    assert obs["language"]["task"] == [["pick"]]
    assert mapping.required["camera_names"] == ["head", "wrist"]
    assert mapping.joint_views == {
        "follower_rldx_input_0": {"parent": "follower_arm", "joint_names": ["j2", "j1"]}}
    assert mapping.required["joint_groups"] == ["follower_rldx_input_0"]
    np.testing.assert_array_equal(mapping.action({"joint_position": np.array([[[20., 10.]]])}, 1), [[10., 20.]])


@pytest.mark.parametrize("names", [["j1", "j1"], ["j1"], ["j1", "unknown"]])
def test_reject_action_mismatch(names):
    meta, robot, actions, cams = fixture()
    meta["action_names"] = names
    with pytest.raises(ValueError):
        RobotMapping(meta, robot, actions, cams)


def test_reject_missing_state_and_nonfinite_actions():
    meta, robot, actions, cams = fixture()
    meta["state_names"] = ["missing"]
    with pytest.raises(ValueError, match="cannot provide"):
        RobotMapping(meta, robot, actions, cams)
    mapping = RobotMapping(*fixture())
    for chunk in (np.zeros((2, 1, 2)), np.full((1, 1, 2), np.nan), np.zeros((1, 2, 2))):
        with pytest.raises(ValueError, match="Invalid RLDX"):
            mapping.action({"joint_position": chunk}, 1)


def test_checkpoint_contract_rejects_unsupported_models_before_gpu(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT))
    from rldx_engine import RLDXEngine
    (tmp_path / "cyclo_input_metadata.json").write_text(json.dumps({"robot_type": "test"}))
    engine = RLDXEngine()
    for options, text in (({"use_memory": True}, "memory/physics"),
                          ({"use_motion": True}, "motion"),
                          ({"use_physics": True}, "physics"),
                          ({"rtc_inference_mode": "guided"}, "RTC"),
                          ({"video_length": 4}, "PT-IMG")):
        config = {"model_type": "RLDX-1", "video_length": 1, **options}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = engine.load_policy(SimpleNamespace(model_path=str(tmp_path), robot_type="test"))
        assert result["success"] is False and text in result["message"]
        assert not engine.is_ready


def test_cleanup_is_repeatable(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from rldx_engine import RLDXEngine
    engine = RLDXEngine()
    closed = []
    engine.robot = SimpleNamespace(close=lambda: closed.append(True))
    engine.policy = object()
    engine.cleanup()
    engine.cleanup()
    assert closed == [True]
    assert not engine.is_ready


def test_engine_load_warmup_repeat_clear_reload(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType
    monkeypatch.syspath_prepend(str(ROOT))
    from rldx_engine import RLDXEngine

    meta, config, actions, cameras = fixture()
    meta.update(robot_type="test", cameras=cameras, fps=15)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "RLDX-1", "video_length": 1}))
    (tmp_path / "cyclo_input_metadata.json").write_text(json.dumps(meta))
    (tmp_path / "processor").mkdir()
    (tmp_path / "processor/processor_config.json").write_text("{}")
    clients, policies = [], []

    class Robot:
        def __init__(self, robot_type, *, defer_subscriptions):
            assert robot_type == "test" and defer_subscriptions
            self._config, self._action_groups = config, actions
            self.camera_names = list(config["cameras"])
            self.closed = False
            self.joint_views = None
            clients.append(self)

        def configure_joint_views(self, views):
            assert views == {
                "follower_rldx_input_0": {"parent": "follower_arm", "joint_names": ["j2", "j1"]}}
            self.joint_views = views

        def start_observation_subscriptions(self, **required):
            assert required["camera_names"] == ["head", "wrist"]
            assert required["joint_groups"] == list(self.joint_views)

        def wait_for_ready(self, **options):
            return True

        def get_required_input_snapshot(self, sources, *, max_age_s):
            assert max_age_s == 1.0 and "camera:unused" not in sources
            assert "joint:follower_arm" not in sources
            assert "joint:follower_rldx_input_0" in sources
            return {"images": {"head": np.zeros((4, 6, 3), np.uint8), "wrist": np.zeros((3, 8, 3), np.uint8)},
                    "joint_positions": {"follower_rldx_input_0": [2., 1.]},
                    "sensors": {"odom": {"linear_velocity": [3., 0., 0.]}}}

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self, **options):
            assert options["strict"]
            assert clients[-1].joint_views is not None
            self.validator = SimpleNamespace(expected_state_dims={"joint_position": 3},
                                             expected_action_dims={"joint_position": 2})
            self.calls, self.resets = 0, 0
            policies.append(self)

        def get_modality_config(self):
            return {k: SimpleNamespace(modality_keys=v, delta_indices=list(range(2)) if k == "action" else [0])
                    for k, v in {"video": ["head", "wrist"], "state": ["joint_position"],
                                 "action": ["joint_position"], "language": ["task"]}.items()}

        def reset(self):
            self.resets += 1

        def get_action(self, observation, options):
            self.calls += 1
            assert observation["language"]["task"] == [["pick"]]
            np.testing.assert_array_equal(observation["state"]["joint_position"], [[[2., 1., 3.]]])
            return {"joint_position": np.asarray([[[20., 10.], [21., 11.]]], np.float32)}, {}

    for name, attrs in {
        "robot_client": {"RobotClient": Robot},
        "robot_client.camera_mapping": {"resolve_camera_feature_sources": lambda names, _: {n: n for n in names}},
        "rldx.data.embodiment_tags": {"EmbodimentTag": {"GENERAL_EMBODIMENT": "general_embodiment"}},
        "rldx.policy.rldx_policy": {"RLDXPolicy": Policy},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    engine = RLDXEngine()
    request = SimpleNamespace(model_path=str(tmp_path), robot_type="test", task_instruction="pick")
    for _ in range(2):
        loaded = engine.load_policy(request)
        assert loaded["success"], loaded
        assert policies[-1].calls == 1  # LOAD warmup, not a robot command
        for _ in range(3):
            result = engine.get_action_chunk(request)
            assert result["success"] and result["chunk_size"] == result["action_dim"] == 2
            np.testing.assert_array_equal(result["action_chunk"], [10., 20., 11., 21.])
        engine.cleanup()
        assert clients[-1].closed and not engine.is_ready
