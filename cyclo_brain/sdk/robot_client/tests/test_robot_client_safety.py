#!/usr/bin/env python3

"""Focused fail-closed tests for RobotClient's real-robot boundaries."""

import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot_client"
    / "robot_client.py"
)


def _load_robot_client_module():
    """Load RobotClient without requiring ROS/Zenoh/OpenCV on the test host."""
    stubs = {
        "cv2": types.ModuleType("cv2"),
        "zenoh_ros2_sdk": types.ModuleType("zenoh_ros2_sdk"),
        "schema": types.ModuleType("schema"),
    }
    stubs["zenoh_ros2_sdk"].ROS2Publisher = object
    stubs["zenoh_ros2_sdk"].ROS2Subscriber = object
    stubs["zenoh_ros2_sdk"].get_message_class = lambda _name: object
    stubs["schema"].get_image_topics = lambda _section: {}
    stubs["schema"].get_state_groups = lambda _section: {}
    stubs["schema"].get_action_groups = lambda _section: {}
    stubs["schema"].load_robot_section = lambda _robot_type: {}

    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(
            "robot_client_safety_test_target",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


robot_client_module = _load_robot_client_module()
RobotClient = robot_client_module.RobotClient


def _command_client():
    client = RobotClient.__new__(RobotClient)
    # These focused tests bypass __init__; keep __del__ from touching transport.
    client._closed = True
    client._action_groups = {
        "mobile": {
            "msg_type": "geometry_msgs/msg/Twist",
            "joint_names": [],
        },
        "arm": {
            "msg_type": "trajectory_msgs/msg/JointTrajectory",
            "joint_names": ["joint_1", "joint_2"],
        },
    }
    client._action_keys = ["mobile", "arm"]
    client._command_publishers = {
        "leader_mobile": object(),
        "leader_arm": object(),
    }
    client._command_joint_names = {
        "leader_mobile": [],
        "leader_arm": ["joint_1", "joint_2"],
    }
    client._cmd_vel_linear_deadband = 0.0
    client._cmd_vel_angular_deadband = 0.0
    client._published = []
    client._publish_twist = lambda publisher, values: client._published.append(
        ("twist", publisher, np.asarray(values).copy())
    )
    client._publish_joint_trajectory = (
        lambda publisher, names, values: client._published.append(
            ("trajectory", publisher, list(names), np.asarray(values).copy())
        )
    )
    return client


def _observation_client():
    client = RobotClient.__new__(RobotClient)
    # These focused tests bypass __init__; keep __del__ from touching transport.
    client._closed = True
    client._lock = threading.Lock()
    client._config = {
        "cameras": {"head": {}, "left_wrist": {}},
        "joint_groups": {"arm": {}},
        "sensors": {"odom": {}, "optional_sensor": {}},
    }
    client._images = {}
    client._image_timestamps = {}
    client._joint_positions = {}
    client._joint_timestamps = {}
    client._sensors = {}
    client._sensor_timestamps = {}
    return client


class PublishActionSafetyTest(unittest.TestCase):
    def test_valid_vector_is_fully_partitioned_and_published(self):
        client = _command_client()

        client.publish_action(np.array([0.1, 0.2, 0.3, 1.0, 2.0]))

        self.assertEqual([entry[0] for entry in client._published], ["twist", "trajectory"])
        np.testing.assert_array_equal(client._published[0][2], [0.1, 0.2, 0.3])
        self.assertEqual(client._published[1][2], ["joint_1", "joint_2"])
        np.testing.assert_array_equal(client._published[1][3], [1.0, 2.0])

    def test_unknown_key_fails_before_any_publish(self):
        client = _command_client()

        with self.assertRaisesRegex(ValueError, "unknown action key"):
            client.publish_action([0.1, 0.2, 0.3, 1.0], ["mobile", "unknown"])

        self.assertEqual(client._published, [])

    def test_unavailable_later_publisher_fails_before_any_publish(self):
        client = _command_client()
        del client._command_publishers["leader_arm"]

        with self.assertRaisesRegex(RuntimeError, "publisher unavailable"):
            client.publish_action([0.1, 0.2, 0.3, 1.0, 2.0])

        self.assertEqual(client._published, [])

    def test_short_or_extra_values_fail_before_any_publish(self):
        for values in ([0.0] * 4, [0.0] * 6):
            with self.subTest(width=len(values)):
                client = _command_client()
                with self.assertRaisesRegex(ValueError, "action width mismatch"):
                    client.publish_action(values)
                self.assertEqual(client._published, [])

    def test_non_finite_values_fail_before_any_publish(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid):
                client = _command_client()
                values = [0.0, 0.0, 0.0, 0.0, invalid]
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    client.publish_action(values)
                self.assertEqual(client._published, [])

    def test_alias_collision_and_empty_selection_are_rejected(self):
        client = _command_client()
        with self.assertRaisesRegex(ValueError, "duplicate action key"):
            client.publish_action([0.0] * 6, ["mobile", "odometry"])
        self.assertEqual(client._published, [])

        with self.assertRaisesRegex(ValueError, "at least one"):
            client.publish_action([], [])
        self.assertEqual(client._published, [])


class ObservationFreshnessSafetyTest(unittest.TestCase):
    def test_unused_camera_does_not_block_but_required_camera_is_checked(self):
        client = _observation_client()
        client._images = {"head": object()}
        client._image_timestamps = {"head": 99.9}
        client._joint_positions = {"arm": np.zeros(2)}
        client._joint_timestamps = {"arm": 99.9}
        client._sensors = {"odom": {}}
        client._sensor_timestamps = {"odom": 99.9}
        client.validate_observation_freshness(0.5, now_s=100, camera_names=["head"])
        with self.assertRaisesRegex(RuntimeError, "missing camera:left_wrist"):
            client.validate_observation_freshness(0.5, now_s=100)
        client._image_timestamps["head"] = 98
        with self.assertRaisesRegex(RuntimeError, "stale camera:head"):
            client.validate_observation_freshness(0.5, now_s=100, camera_names=["head"])

    def test_all_configured_policy_observations_are_fresh(self):
        client = _observation_client()
        client._images = {"head": object(), "left_wrist": object()}
        client._image_timestamps = {"head": 99.90, "left_wrist": 99.95}
        client._joint_positions = {"arm": np.zeros(2)}
        client._joint_timestamps = {"arm": 99.85}
        client._sensors = {"odom": {"position": np.zeros(3)}}
        client._sensor_timestamps = {"odom": 99.81}

        # The unrelated optional sensor is intentionally not a policy gate.
        client.validate_observation_freshness(0.2, now_s=100.0)

    def test_missing_observations_and_timestamps_are_reported_together(self):
        client = _observation_client()
        client._images = {"head": object()}
        client._image_timestamps = {"head": 100.0}
        client._joint_positions = {"arm": np.zeros(2)}

        with self.assertRaises(RuntimeError) as caught:
            client.validate_observation_freshness(0.5, now_s=100.0)

        message = str(caught.exception)
        self.assertIn("missing camera:left_wrist", message)
        self.assertIn("missing timestamp for joint:arm", message)
        self.assertIn("missing odom", message)

    def test_stale_future_and_invalid_timestamps_fail_closed(self):
        client = _observation_client()
        client._images = {"head": object(), "left_wrist": object()}
        client._image_timestamps = {"head": 98.0, "left_wrist": 101.0}
        client._joint_positions = {"arm": np.zeros(2)}
        client._joint_timestamps = {"arm": 99.9}
        client._sensors = {"odom": {"position": np.zeros(3)}}
        client._sensor_timestamps = {"odom": np.nan}

        with self.assertRaises(RuntimeError) as caught:
            client.validate_observation_freshness(0.5, now_s=100.0)

        message = str(caught.exception)
        self.assertIn("stale camera:head (2.000s old)", message)
        self.assertIn("future timestamp for camera:left_wrist", message)
        self.assertIn("invalid timestamp for sensor:odom", message)

    def test_invalid_threshold_is_rejected(self):
        client = _observation_client()
        for max_age in (-0.1, np.nan, np.inf, "bad"):
            with self.subTest(max_age=max_age):
                with self.assertRaisesRegex(ValueError, "max_age_s"):
                    client.validate_observation_freshness(max_age, now_s=100.0)


if __name__ == "__main__":
    unittest.main()
