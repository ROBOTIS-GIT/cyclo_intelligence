"""Deferred model subscriptions avoid unused middleware callbacks and decode work."""

from types import SimpleNamespace
from copy import deepcopy
from unittest import mock

import numpy as np
import pytest

from test_initial_pose_sync import RobotClient, robot_client_impl


CONFIG = {
    "cameras": {name: {"topic": f"/{name}", "msg_type": "sensor_msgs/msg/CompressedImage"}
                for name in ("eye", "unused_eye")},
    "joint_groups": {
        "parent": {"topic": "/state", "msg_type": "sensor_msgs/msg/JointState", "joint_names": ["a", "b"]},
        "child": {"parent": "parent", "joint_names": ["b"]},
        "unused": {"topic": "/unused_state", "msg_type": "sensor_msgs/msg/JointState", "joint_names": ["c"]},
    },
    "sensors": {"odom": {"topic": "/odom", "msg_type": "nav_msgs/msg/Odometry"}},
}


@pytest.fixture
def subscriptions():
    with mock.patch.object(robot_client_impl, "_build_runtime_config", return_value=CONFIG), \
            mock.patch.object(robot_client_impl, "ROS2Subscriber") as subscribe:
        subscribe.side_effect = lambda **kwargs: mock.Mock(**kwargs)
        yield subscribe


def test_default_client_keeps_all_configured_subscriptions(subscriptions):
    client = RobotClient("ffw_sg2_rev1")
    try:
        assert {c.kwargs["topic"] for c in subscriptions.call_args_list} == {
            "/eye", "/unused_eye", "/state", "/unused_state", "/odom",
        }
    finally:
        client.close()


def test_deferred_client_subscribes_only_declared_sources_and_synthetic_parent_once(subscriptions):
    client = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    try:
        subscriptions.assert_not_called()
        sink = SimpleNamespace(sources=frozenset({"joint:child"}), record=mock.Mock())
        client.attach_observation_capture(sink)
        client.start_observation_subscriptions(camera_names=["eye"], joint_groups=["parent", "child"],
                                               sensor_names=[])
        assert [c.kwargs["topic"] for c in subscriptions.call_args_list] == ["/eye", "/state"]
        callbacks = {c.kwargs["topic"]: c.kwargs["callback"] for c in subscriptions.call_args_list}
        callbacks["/state"](SimpleNamespace(name=["a", "b"], position=[1., 2.], velocity=[], effort=[]))
        assert client._joint_positions["child"].tolist() == [2.]
        assert client._joint_positions.keys() == {"parent", "child"}
        assert any(c.args[0] == "joint:child" for c in sink.record.call_args_list)
        pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        with mock.patch.object(robot_client_impl.cv2, "imdecode", return_value=pixels) as decode:
            callbacks["/eye"](SimpleNamespace(data=b"image"))
            decode.assert_called_once()
        assert client._images.keys() == {"eye"}
        assert client.get_missing_observations() == []
        with pytest.raises(RuntimeError, match="fresh deferred"):
            client.start_observation_subscriptions(camera_names=[], joint_groups=[], sensor_names=[])
        assert subscriptions.call_count == 2
    finally:
        client.close()


def test_empty_selection_creates_no_subscription_or_decode_work(subscriptions):
    client = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    client.start_observation_subscriptions(camera_names=[], joint_groups=[], sensor_names=[])
    subscriptions.assert_not_called()
    assert client.get_missing_observations() == []
    client.close()


def test_selection_cannot_omit_history_or_enable_a_disabled_modality(subscriptions):
    client = RobotClient("ffw_sg2_rev1", defer_subscriptions=True, subscribe_images=False)
    try:
        with pytest.raises(ValueError, match="Invalid observation"):
            client.start_observation_subscriptions(camera_names=["eye"], joint_groups=[], sensor_names=[])
        client.attach_observation_capture(SimpleNamespace(sources=frozenset({"joint:child"}), record=mock.Mock()))
        with pytest.raises(ValueError, match="omits captured"):
            client.start_observation_subscriptions(camera_names=[], joint_groups=[], sensor_names=[])
        subscriptions.assert_not_called()
        client.start_observation_subscriptions(camera_names=[], joint_groups=["child"], sensor_names=[])
        assert [c.kwargs["topic"] for c in subscriptions.call_args_list] == ["/state"]
    finally:
        client.close()


def test_partial_subscription_failure_closes_resources_and_prevents_reuse(subscriptions):
    first = mock.Mock()
    subscriptions.side_effect = [first, RuntimeError("network unavailable")]
    client = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    with pytest.raises(RuntimeError, match="network unavailable"):
        client.start_observation_subscriptions(camera_names=["eye"], joint_groups=["child"], sensor_names=[])
    first.close.assert_called_once()
    assert client._closed and not client._subscribers


def test_shared_physical_topic_has_one_subscription_and_preserves_all_views(subscriptions):
    config = deepcopy(CONFIG)
    config["joint_groups"]["other_parent"] = dict(config["joint_groups"]["parent"])
    config["joint_groups"]["other_child"] = {"parent": "other_parent", "joint_names": ["a"]}
    with mock.patch.object(robot_client_impl, "_build_runtime_config", return_value=config):
        client = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    try:
        sink = SimpleNamespace(sources=frozenset({"joint:parent", "joint:other_parent",
                                                "joint:child", "joint:other_child"}), record=mock.Mock())
        client.attach_observation_capture(sink)
        client.start_observation_subscriptions(camera_names=[], sensor_names=[],
                                               joint_groups=["parent", "other_parent", "child", "other_child"])
        assert subscriptions.call_count == 1
        callback = subscriptions.call_args.kwargs["callback"]
        message = SimpleNamespace(name=["a", "b"], position=[1., 2.], velocity=[3., 4.], effort=[5., 6.])
        callback(message)
        for parent in ("parent", "other_parent"):
            assert client.get_joint_positions(parent).tolist() == [1., 2.]
            assert client._joint_velocities[parent].tolist() == [3., 4.]
            assert client._joint_efforts[parent].tolist() == [5., 6.]
        assert client.get_joint_positions("child").tolist() == [2.]
        assert client.get_joint_positions("other_child").tolist() == [1.]
        assert client._joint_velocities["other_child"].tolist() == [3.]
        assert {call.args[0] for call in sink.record.call_args_list} == sink.sources
        assert sink.record.call_count == 4
        # Public snapshots cannot mutate another logical view or a retained sample.
        values = client.get_joint_positions("parent")
        values[:] = 99
        assert client.get_joint_positions("other_parent").tolist() == [1., 2.]
        message.position = [7., 8.]
        callback(message)
        assert client.get_joint_positions("other_parent").tolist() == [7., 8.]
        assert client.get_joint_positions("child").tolist() == [8.]
        subscribers = list(client._state_subscribers)
        client.set_state_subscription(False)
        for subscriber in subscribers:
            subscriber.close.assert_called_once()
        subscriptions.reset_mock()
        client.set_state_subscription(True)
        assert subscriptions.call_count == 1
    finally:
        client.close()


def test_joint_subscription_dedup_keeps_distinct_message_types_separate(subscriptions):
    config = deepcopy(CONFIG)
    config["joint_groups"]["different_type"] = {
        "topic": "/state", "msg_type": "custom_msgs/msg/JointState", "joint_names": ["a"],
    }
    with mock.patch.object(robot_client_impl, "_build_runtime_config", return_value=config):
        client = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    try:
        client.start_observation_subscriptions(camera_names=[], sensor_names=[],
                                               joint_groups=["parent", "different_type"])
        assert subscriptions.call_count == 2
        assert {call.kwargs["msg_type"] for call in subscriptions.call_args_list} == {
            "sensor_msgs/msg/JointState", "custom_msgs/msg/JointState",
        }
    finally:
        client.close()
