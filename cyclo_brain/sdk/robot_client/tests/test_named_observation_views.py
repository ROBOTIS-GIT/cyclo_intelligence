"""Selected named state views preserve reception identity and command isolation."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import test_initial_pose_sync as sync_tests
from test_initial_pose_sync import RobotClient, robot_client_impl
from test_observation_capture import specification, ReceptionHistory, InputAssembler


@pytest.fixture
def client():
    with mock.patch.object(RobotClient, "_init_subscriptions"):
        bot = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    bot._config = {"cameras": {}, "sensors": {}, "joint_groups": {
        "body": {"role": "follower", "msg_type": "sensor_msgs/msg/JointState", "topic": "/state",
                 "joint_names": ["a", "b", "unused"]}}}
    bot.configure_joint_views({"selected": {"parent": "body", "joint_names": ["b", "a"]}})
    bot._joint_children = {"body": ["selected"]}
    yield bot
    bot.close()


def feed(client, names, positions, stamp):
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=stamp):
        client._update_joint("body", SimpleNamespace(name=names, position=positions, velocity=[], effort=[]))


def test_named_view_order_missing_unused_joints_and_history(client):
    spec = specification("joint:selected", (-1., 0.))
    history = ReceptionHistory(spec)
    client.attach_observation_capture(history)
    feed(client, ["a", "b"], [1., 2.], 10.)
    feed(client, ["b", "a"], [4., 3.], 11.)
    values = InputAssembler(spec, {"stack": np.stack}).assemble(history, anchor_s=11.)
    np.testing.assert_array_equal(values["input"], [[2., 1.], [4., 3.]])
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=11.1):
        snapshot = client.get_required_input_snapshot({"joint:selected"}, max_age_s=.2)
    np.testing.assert_array_equal(snapshot["joint_positions"]["selected"], [4., 3.])
    assert snapshot["reception_monotonic_timestamps"]["joint:selected"] == 11.
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=12.):
        with pytest.raises(ValueError, match="stale"):
            client.get_required_input_snapshot({"joint:selected"}, max_age_s=.2)


@pytest.mark.parametrize("names,values", [(["a"], [1.]), (["a", "b"], [1.]),
                                         (["a", "b", "a"], [1., 2., 3.]),
                                         (["a", "b"], [1., float("nan")]), ([], [])])
def test_bad_named_message_invalidates_latest_view(client, names, values):
    feed(client, ["a", "b"], [1., 2.], 10.)
    feed(client, names, values, 11.)
    with pytest.raises(ValueError, match="Missing input"):
        client.get_required_input_snapshot({"joint:selected"})
    assert client._input_received_monotonic["joint:selected"] == 10.


def test_only_selected_groups_receive_prediction_hold_and_slow_start():
    helper = sync_tests.InitialPoseSyncCommandTest()
    bot = helper._make_client("ffw_sg2_rev1")
    helper._seed_joint_state(bot)
    keys = ["arm_left", "arm_right"]
    values = np.arange(16, dtype=np.float64)
    np.testing.assert_array_equal(bot.publish_action_with_receipt(values, keys), values)
    bot.publish_current_pose_hold(keys)
    bot.publish_initial_pose_sync(values, keys)
    for key in bot._action_groups:
        assert len(bot._command_publishers[f"leader_{key}"].messages) == (3 if key in keys else 0)
