"""Real RobotClient callbacks feed RLDX state without ROS transport or weights."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def connection(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root.parents[1] / "sdk/robot_client/tests"))
    from rldx_engine import RLDXEngine
    from rldx_engine.mapping import RobotMapping
    from test_initial_pose_sync import RobotClient, robot_client_impl

    with mock.patch.object(RobotClient, "_init_subscriptions"):
        robot = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    robot._config = {"cameras": {}, "sensors": {}, "joint_groups": {
        name: {"role": "follower", "joint_names": names, "topic": f"/{name}",
               "msg_type": "sensor_msgs/msg/JointState"}
        for name, names in {"arm": ["j1", "j2", "unused"], "hand": ["grip"],
                            "unneeded": ["other"]}.items()}}
    actions = {"arm": {"joint_names": ["j1", "j2"]}, "hand": {"joint_names": ["grip"]}}
    mapping = RobotMapping({"state_names": ["j2", "grip", "j1"],
                            "action_names": ["j2", "grip", "j1"]}, robot._config, actions, {})
    callbacks = {}

    def subscribe(*, topic, msg_type, callback):
        assert msg_type == "sensor_msgs/msg/JointState"
        callbacks[topic] = callback
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(robot_client_impl, "ROS2Subscriber", subscribe)
    robot.configure_joint_views(mapping.joint_views)
    robot.start_observation_subscriptions(**mapping.required)
    assert set(callbacks) == {"/arm", "/hand"}
    engine = RLDXEngine()
    engine.robot, engine.mapping = robot, mapping
    engine.language_key, engine.max_age_s, engine.horizon = "task", .5, 1
    engine.policy = mock.Mock()
    engine.policy.get_action.return_value = ({"joint_position": np.array([[[20., 30., 10.]]])}, {})

    def feed(group, names, positions, received_s):
        with mock.patch.object(robot_client_impl.time, "monotonic", return_value=received_s):
            callbacks[f"/{group}"](SimpleNamespace(name=names, position=positions, velocity=[], effort=[]))

    def predict(now):
        with mock.patch.object(robot_client_impl.time, "monotonic", return_value=now):
            return engine.get_action_chunk(SimpleNamespace(task_instruction="pick"))

    yield SimpleNamespace(robot=robot, mapping=mapping, engine=engine, feed=feed, predict=predict)
    engine.cleanup()


def test_joint_message_order_does_not_change_model_order(connection):
    c = connection
    c.feed("hand", ["grip"], [3.], 10.)
    for names, values in ((["j1", "j2"], [1., 2.]), (["j2", "j1"], [2., 1.]),
                          (["unused", "j2", "j1"], [99., 2., 1.])):
        c.feed("arm", names, values, 10.)
        result = c.predict(10.1)
        assert result["success"], result
        observation = c.engine.policy.get_action.call_args.args[0]
        np.testing.assert_array_equal(observation["state"]["joint_position"], [[[2., 3., 1.]]])
        np.testing.assert_array_equal(result["action_chunk"], [10., 20., 30.])
        np.testing.assert_array_equal(c.robot._joint_positions["arm"], values)
    assert c.engine.policy.get_action.call_count == 3
    assert not c.robot._command_publishers


@pytest.mark.parametrize("names,values", [
    (["j2", "other"], [2., 3.]),
    (["j1", "j2", "j1"], [1., 2., 3.]),
    (["j1", "j2"], [1.]),
    ([], [1., 2.]),
    ([], []),
    (["j1", "j2"], [1., float("nan")]),
    (["j1", "j2"], [float("inf"), 2.]),
])
def test_bad_message_blocks_inference_until_valid_replacement(connection, names, values):
    c = connection
    c.feed("hand", ["grip"], [3.], 10.)
    c.feed("arm", ["j1", "j2"], [1., 2.], 10.)
    assert c.predict(10.1)["success"]
    c.engine.policy.get_action.reset_mock()
    c.feed("arm", names, values, 10.2)
    result = c.predict(10.3)
    assert not result["success"] and "Missing input source: joint:" in result["message"]
    c.engine.policy.get_action.assert_not_called()
    c.feed("arm", ["j2", "j1"], [4., 5.], 10.4)
    assert c.predict(10.4)["success"]
    observation = c.engine.policy.get_action.call_args.args[0]
    np.testing.assert_array_equal(observation["state"]["joint_position"], [[[4., 3., 5.]]])


def test_other_group_updates_do_not_refresh_stale_joint_view(connection):
    c = connection
    c.feed("arm", ["j2", "j1"], [2., 1.], 10.)
    c.feed("hand", ["grip"], [3.], 11.)
    result = c.predict(11.1)
    assert not result["success"] and "stale" in result["message"]
    c.engine.policy.get_action.assert_not_called()
