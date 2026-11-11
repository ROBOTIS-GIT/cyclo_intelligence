#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, **kwargs) -> None:
        self.messages.append(kwargs)


class FakeMessage:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


zenoh_stub = types.ModuleType("zenoh_ros2_sdk")
zenoh_stub.ROS2Publisher = object
zenoh_stub.ROS2Subscriber = object
zenoh_stub.get_message_class = lambda _name: FakeMessage
sys.modules.setdefault("zenoh_ros2_sdk", zenoh_stub)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot_client"
    / "robot_client.py"
)
ROBOT_CONFIG_PATH = MODULE_PATH.parents[4] / "shared" / "shared" / "robot_configs"
sys.path.insert(0, str(ROBOT_CONFIG_PATH))
spec = importlib.util.spec_from_file_location("robot_client_impl", MODULE_PATH)
robot_client_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robot_client_impl)
RobotClient = robot_client_impl.RobotClient


class InitialPoseSyncCommandTest(unittest.TestCase):
    def test_saved_pose_uses_named_targets_and_zero_terminal_derivatives(self):
        for robot_type in ("ffw_sg2_rev1", "ffw_sh5_rev1", "omy_f3m"):
            with self.subTest(robot_type=robot_type):
                client = self._make_client(robot_type)
                self._seed_joint_state(client)
                target = {name: 0.123 for name in client._joint_positions_by_name}
                client.publish_named_pose(target, duration_s=5.)
                for key, group in client._action_groups.items():
                    msg = client._command_publishers[f"leader_{key}"].messages[-1]
                    if group["msg_type"] == "geometry_msgs/msg/Twist":
                        self.assertEqual(msg["linear"].x, 0.)
                        self.assertEqual(msg["angular"].z, 0.)
                    else:
                        point = msg["points"][0]
                        np.testing.assert_allclose(point.positions, [target[n] for n in group["joint_names"]])
                        np.testing.assert_array_equal(point.velocities, np.zeros(len(group["joint_names"])))
                        self.assertEqual(point.time_from_start.sec, 5)

    def test_saved_pose_rejects_stale_or_invalid_state_before_publication(self):
        client = self._make_client("ffw_sg2_rev1")
        self._seed_joint_state(client)
        target = dict(client._joint_positions_by_name)
        name = next(iter(target))
        client._joint_position_timestamps_by_name[name] -= 10
        with self.assertRaisesRegex(RuntimeError, "stale"):
            client.publish_named_pose(target)
        assert all(not publisher.messages for publisher in client._command_publishers.values())
        self._seed_joint_state(client)
        client._joint_positions_by_name[name] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            client.publish_named_pose(target)
        assert all(not publisher.messages for publisher in client._command_publishers.values())

    def test_expired_step_stops_twist_and_preserves_position_target_and_receipt(self):
        client = self._make_client("ffw_sg2_rev1")
        keys = list(client._action_keys)
        action = np.ones(self._action_dimension(client, keys))
        original = action.copy()
        receipt = client.publish_action_with_receipt(action, keys, zero_twist=True)
        expected = []
        offset = 0
        for key in keys:
            cfg = client._action_groups[key]
            messages = client._command_publishers[f"leader_{key}"].messages
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                width = 3
                expected.extend([0.] * width)
                assert messages[-1]["linear"].x == 0.
                assert messages[-1]["linear"].y == 0.
                assert messages[-1]["angular"].z == 0.
            else:
                width = len(cfg["joint_names"])
                expected.extend(action[offset:offset + width])
            offset += width
        np.testing.assert_array_equal(receipt, expected)
        np.testing.assert_array_equal(action, original)

    def test_publication_receipt_contains_deadband_adjusted_values(self):
        client = self._make_client("ffw_sg2_rev1")
        client._cmd_vel_linear_deadband = 0.01
        client._cmd_vel_angular_deadband = 0.02
        keys = list(client._action_keys)
        action = np.full(self._action_dimension(client, keys), 0.005)
        receipt = client.publish_action_with_receipt(action, keys)
        offset = 0
        expected = []
        for key in keys:
            cfg = client._action_groups[key]
            width = 3 if cfg["msg_type"] == "geometry_msgs/msg/Twist" else len(cfg["joint_names"])
            expected.extend([0.] * width if cfg["msg_type"] == "geometry_msgs/msg/Twist" else action[offset:offset + width])
            offset += width
        np.testing.assert_array_equal(receipt, expected)
        assert all(len(p.messages) == 1 for p in client._command_publishers.values())

    def test_receipt_validates_every_publisher_before_sending_anything(self):
        client = self._make_client("ffw_sg2_rev1")
        keys = list(client._action_keys)
        action = np.ones(self._action_dimension(client, keys))
        client._command_publishers.pop(f"leader_{keys[-1]}")
        with self.assertRaisesRegex(RuntimeError, "publisher unavailable"):
            client.publish_action_with_receipt(action, keys)
        assert all(not p.messages for p in client._command_publishers.values())

    def test_partial_publish_failure_never_returns_a_successful_receipt(self):
        client = self._make_client("ffw_sg2_rev1")
        keys = list(client._action_keys)
        action = np.ones(self._action_dimension(client, keys))
        failing = client._command_publishers[f"leader_{keys[-1]}"]
        failing.publish = mock.Mock(side_effect=RuntimeError("transport failed"))
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            client.publish_action_with_receipt(action, keys)
        assert client._command_publishers[f"leader_{keys[0]}"].messages

    def test_input_snapshot_is_independent_and_converts_bgr_to_rgb(self):
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            client = RobotClient("ffw_sg2_rev1")
        client._images["eye"] = np.array([[[1, 2, 3]]], dtype=np.uint8)
        client._joint_positions["arm"] = np.array([0.5], dtype=np.float32)
        client._sensors["odom"] = {"linear_velocity": [1., 0., 0.]}
        client._image_timestamps["eye"] = 123.
        snapshot = client.get_input_snapshot()
        np.testing.assert_array_equal(snapshot["images"]["eye"], [[[3, 2, 1]]])
        snapshot["images"]["eye"][:] = 9
        snapshot["joint_positions"]["arm"][:] = 9
        snapshot["sensors"]["odom"]["linear_velocity"][0] = 9
        assert client._images["eye"][0, 0, 0] == 1
        assert client._joint_positions["arm"][0] == 0.5
        assert client._sensors["odom"]["linear_velocity"][0] == 1
        assert snapshot["reception_wall_timestamps"]["images"]["eye"] == 123.
        assert snapshot["captured_monotonic_s"] > 0
        client.close()

    def test_readiness_can_select_required_inputs_without_model_specific_names(self):
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            client = RobotClient("ffw_sg2_rev1")
        client._config = {
            "cameras": {"required_eye": {}, "unused_eye": {}},
            "joint_groups": {"follower_arm": {}, "unused_joint": {}},
            "sensors": {"odom": {}},
        }
        required = dict(camera_names=["required_eye"],
                        joint_groups=["follower_arm"], sensor_names=["odom"])
        client._images["required_eye"] = np.zeros((2, 2, 3))
        self.assertEqual(client.get_missing_observations(**required),
                         ["joint:follower_arm", "sensor:odom"])
        self.assertFalse(client.wait_for_ready(timeout=0, **required))
        client._joint_positions["follower_arm"] = np.ones(2)
        client._sensors["odom"] = {"linear_velocity": [0, 0, 0]}
        self.assertTrue(client.wait_for_ready(timeout=0, **required))
        self.assertFalse(client._all_ready())
        self.assertEqual(client._get_missing(), ["camera:unused_eye", "joint:unused_joint"])
        client.close()

    def _make_client(self, robot_type: str) -> RobotClient:
        section = robot_client_impl.robot_schema.load_robot_section(robot_type)
        action_groups = robot_client_impl.robot_schema.get_action_groups(section)
        client = RobotClient.__new__(RobotClient)
        client._config = robot_client_impl._build_runtime_config(section)
        client._action_groups = action_groups
        client._action_keys = sorted(action_groups)
        client._command_publishers = {}
        client._command_joint_names = {}
        client._joint_positions_by_name = {}
        client._joint_position_timestamps_by_name = {}
        client._joint_positions = {}
        client._joint_velocities = {}
        client._joint_efforts = {}
        client._joint_timestamps = {}
        client._joint_children = {}
        client._lock = threading.Lock()
        client._cmd_vel_linear_deadband = 0.0
        client._cmd_vel_angular_deadband = 0.0
        client._initial_pose_sync_state_max_age_s = 1.0
        client._closed = True
        for key, cfg in action_groups.items():
            publisher_key = f"leader_{key}"
            client._command_publishers[publisher_key] = FakePublisher()
            client._command_joint_names[publisher_key] = list(
                cfg.get("joint_names", [])
            )
        return client

    def test_joint_state_max_age_environment_override_and_fallback(self) -> None:
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            with mock.patch.dict(
                os.environ,
                {"INITIAL_POSE_SYNC_STATE_MAX_AGE_S": "2.5"},
            ):
                client = RobotClient("omy_f3m")
                self.assertEqual(client._initial_pose_sync_state_max_age_s, 2.5)

            for invalid in ("0", "-1", "nan", "invalid"):
                with self.subTest(invalid=invalid), mock.patch.dict(
                    os.environ,
                    {"INITIAL_POSE_SYNC_STATE_MAX_AGE_S": invalid},
                ):
                    client = RobotClient("omy_f3m")
                    self.assertEqual(client._initial_pose_sync_state_max_age_s, 1.0)

    def test_subscription_options_select_observation_groups(self) -> None:
        subscribers = []

        def make_subscriber(**kwargs):
            subscribers.append(kwargs)
            return SimpleNamespace(close=lambda: None)

        with mock.patch.object(robot_client_impl, "ROS2Subscriber", make_subscriber):
            client = RobotClient(
                "ffw_sg2_rev1",
                subscribe_images=False,
                subscribe_state=True,
                subscribe_sensors=False,
            )

        self.assertEqual([sub["topic"] for sub in subscribers], ["/joint_states"])
        self.assertTrue(client._all_ready() is False)
        self.assertEqual(
            set(client._get_missing()),
            {
                "joint:follower_upper_body",
                "joint:follower_arm_left",
                "joint:follower_arm_right",
                "joint:follower_head",
                "joint:follower_lift",
            },
        )

    def test_disabling_all_observations_creates_no_subscribers(self) -> None:
        subscribers = []

        def make_subscriber(**kwargs):
            subscribers.append(kwargs)
            return SimpleNamespace(close=lambda: None)

        with mock.patch.object(robot_client_impl, "ROS2Subscriber", make_subscriber):
            client = RobotClient(
                "ffw_sg2_rev1",
                subscribe_images=False,
                subscribe_state=False,
                subscribe_sensors=False,
            )

        self.assertEqual(subscribers, [])
        self.assertTrue(client._all_ready())
        self.assertEqual(client._get_missing(), [])

    def test_state_subscription_can_follow_robot_publish_mode(self) -> None:
        subscribers = []

        def make_subscriber(**kwargs):
            subscriber = SimpleNamespace(kwargs=kwargs, closed=False)

            def close():
                subscriber.closed = True

            subscriber.close = close
            subscribers.append(subscriber)
            return subscriber

        with mock.patch.object(robot_client_impl, "ROS2Subscriber", make_subscriber):
            client = RobotClient(
                "ffw_sg2_rev1",
                subscribe_images=False,
                subscribe_state=False,
                subscribe_sensors=False,
            )
            client.set_state_subscription(True)
            self.assertEqual(len(subscribers), 1)
            self.assertEqual(subscribers[0].kwargs["topic"], "/joint_states")
            self.assertEqual(client._state_subscribers, subscribers)

            client.set_state_subscription(False)

        self.assertTrue(subscribers[0].closed)
        self.assertEqual(client._state_subscribers, [])
        self.assertEqual(client._subscribers, [])

    def test_optional_sensors_do_not_block_observation_readiness(self) -> None:
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            client = RobotClient(
                "ffw_sh5_rev1",
                subscribe_images=False,
                subscribe_state=False,
                subscribe_sensors=True,
            )

        self.assertTrue(client._config["sensors"])
        self.assertTrue(client._all_ready())
        self.assertEqual(client._get_missing(), [])

    @staticmethod
    def _action_dimension(client: RobotClient, action_keys: list[str]) -> int:
        total = 0
        for key in action_keys:
            cfg = client._action_groups[key]
            total += (
                3
                if cfg["msg_type"] == "geometry_msgs/msg/Twist"
                else len(cfg["joint_names"])
            )
        return total

    @staticmethod
    def _seed_joint_state(client: RobotClient) -> None:
        value = 0.0
        received_at = time.monotonic()
        for cfg in client._action_groups.values():
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                continue
            for name in reversed(cfg["joint_names"]):
                client._joint_positions_by_name[name] = value
                client._joint_position_timestamps_by_name[name] = received_at
                value += 0.1

    def test_supported_robot_layouts_publish_all_position_groups_and_zero_mobile(self):
        for robot_type in ("ffw_sg2_rev1", "ffw_sh5_rev1", "omy_f3m"):
            with self.subTest(robot_type=robot_type):
                client = self._make_client(robot_type)
                action_keys = list(client._action_keys)
                self._seed_joint_state(client)
                action = np.arange(
                    self._action_dimension(client, action_keys),
                    dtype=np.float64,
                )

                client.publish_initial_pose_sync(
                    action,
                    action_keys,
                    duration_s=5.5,
                )

                for key in action_keys:
                    publisher = client._command_publishers[f"leader_{key}"]
                    self.assertEqual(len(publisher.messages), 1)
                    message = publisher.messages[0]
                    cfg = client._action_groups[key]
                    if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                        self.assertEqual(message["linear"].x, 0.0)
                        self.assertEqual(message["linear"].y, 0.0)
                        self.assertEqual(message["angular"].z, 0.0)
                    else:
                        point = message["points"][0]
                        self.assertEqual(point.time_from_start.sec, 5)
                        self.assertEqual(point.time_from_start.nanosec, 500_000_000)
                        self.assertEqual(message["joint_names"], cfg["joint_names"])

    def test_current_pose_hold_uses_joint_names_in_config_order(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        joint_names = client._action_groups["arm"]["joint_names"]
        reversed_names = list(reversed(joint_names))
        client._update_joint(
            "follower_arm",
            SimpleNamespace(
                name=reversed_names,
                position=[float(index + 1) for index in range(len(reversed_names))],
                velocity=[],
                effort=[],
            ),
        )

        client.publish_current_pose_hold(action_keys, duration_s=0.1)

        message = client._command_publishers["leader_arm"].messages[0]
        expected = [client._joint_positions_by_name[name] for name in joint_names]
        np.testing.assert_allclose(message["points"][0].positions, expected)
        self.assertEqual(message["points"][0].time_from_start.nanosec, 100_000_000)

    def test_missing_current_joint_state_prevents_any_sync_command(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        action = np.zeros(self._action_dimension(client, action_keys))

        with self.assertRaisesRegex(RuntimeError, "current joint state unavailable"):
            client.publish_initial_pose_sync(action, action_keys, duration_s=5.0)

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_stale_current_joint_state_prevents_any_sync_command(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        self._seed_joint_state(client)
        stale_at = time.monotonic() - 1.1
        client._joint_position_timestamps_by_name = {
            name: stale_at for name in client._joint_positions_by_name
        }
        action = np.zeros(self._action_dimension(client, action_keys))

        with self.assertRaisesRegex(RuntimeError, "current joint state stale"):
            client.publish_initial_pose_sync(action, action_keys, duration_s=5.0)

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_one_stale_joint_blocks_only_its_group_not_other_holds_or_twists(self):
        client = self._make_client("ffw_sg2_rev1")
        action_keys = list(client._action_keys)
        self._seed_joint_state(client)
        stale_name = next(iter(client._joint_positions_by_name))
        client._joint_position_timestamps_by_name[stale_name] = time.monotonic() - 1.1

        with self.assertRaisesRegex(RuntimeError, stale_name):
            client.publish_current_pose_hold(action_keys, duration_s=0.1)

        for key, group in client._action_groups.items():
            messages = client._command_publishers[f"leader_{key}"].messages
            self.assertEqual(len(messages), 0 if stale_name in group.get("joint_names", []) else 1)

    def test_twist_failure_does_not_skip_position_holds_or_later_twists(self):
        client = self._make_client("ffw_sg2_rev1")
        self._seed_joint_state(client)
        mobile = next(k for k, group in client._action_groups.items()
                      if group["msg_type"] == "geometry_msgs/msg/Twist")
        client._action_groups["second_base"] = dict(client._action_groups[mobile])
        client._command_publishers["leader_second_base"] = FakePublisher()
        failing = client._command_publishers[f"leader_{mobile}"]
        failing.publish = mock.Mock(side_effect=RuntimeError("transport failed"))
        with self.assertRaisesRegex(RuntimeError, mobile):
            client.publish_current_pose_hold(list(client._action_groups))
        failing.publish.assert_called_once()
        for key, publisher in client._command_publishers.items():
            if key != f"leader_{mobile}":
                self.assertEqual(len(publisher.messages), 1, key)

    def test_missing_publishers_do_not_skip_other_groups(self):
        for missing_key in ("arm_left", "mobile"):
            with self.subTest(key=missing_key):
                client = self._make_client("ffw_sg2_rev1")
                self._seed_joint_state(client)
                client._command_publishers.pop(f"leader_{missing_key}")
                with self.assertRaisesRegex(RuntimeError, f"{missing_key}.*publisher unavailable"):
                    client.publish_current_pose_hold()
                self.assertTrue(all(len(p.messages) == 1 for p in client._command_publishers.values()))

    def test_multiple_hold_failures_are_aggregated_and_retry_can_succeed(self):
        client = self._make_client("ffw_sg2_rev1")
        self._seed_joint_state(client)
        failing = client._command_publishers["leader_arm_left"]
        original_publish = failing.publish
        failing.publish = mock.Mock(side_effect=RuntimeError("arm transport failed"))
        bad_name = client._action_groups["arm_right"]["joint_names"][0]
        client._joint_positions_by_name[bad_name] = float("nan")
        with self.assertRaises(RuntimeError) as result:
            client.publish_current_pose_hold()
        self.assertIn("arm_left: arm transport failed", str(result.exception))
        self.assertIn("arm_right: current joint state contains non-finite", str(result.exception))
        for key in ("head", "lift", "mobile"):
            self.assertEqual(len(client._command_publishers[f"leader_{key}"].messages), 1)
        failing.publish = original_publish
        self._seed_joint_state(client)
        self.assertIsNone(client.publish_current_pose_hold())

    def test_missing_joint_state_does_not_block_other_position_groups(self):
        client = self._make_client("ffw_sg2_rev1")
        self._seed_joint_state(client)
        name = client._action_groups["arm_left"]["joint_names"][0]
        del client._joint_positions_by_name[name]
        with self.assertRaisesRegex(RuntimeError, name):
            client.publish_current_pose_hold()
        for key, publisher in client._command_publishers.items():
            self.assertEqual(len(publisher.messages), 0 if key == "leader_arm_left" else 1)

    def test_partial_initial_sync_failure_uses_independent_hold_recovery(self):
        client = self._make_client("ffw_sg2_rev1")
        self._seed_joint_state(client)
        keys = list(client._action_keys)
        failing = client._command_publishers["leader_arm_left"]
        failing.publish = mock.Mock(side_effect=RuntimeError("arm transport failed"))
        with self.assertRaisesRegex(RuntimeError, "arm transport failed"):
            client.publish_initial_pose_sync(np.ones(self._action_dimension(client, keys)), keys)
        self.assertEqual(failing.publish.call_count, 2)
        for key in ("arm_right", "head", "lift"):
            point = client._command_publishers[f"leader_{key}"].messages[-1]["points"][0]
            self.assertEqual(point.time_from_start.nanosec, 100_000_000)
        mobile = client._command_publishers["leader_mobile"].messages[-1]
        self.assertEqual(mobile["linear"].x, 0.)

    def test_invalid_action_layout_is_rejected_before_publish(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        self._seed_joint_state(client)
        expected_dim = self._action_dimension(client, action_keys)

        invalid_actions = [
            np.zeros(expected_dim - 1),
            np.zeros(expected_dim + 1),
            np.full(expected_dim, np.nan),
            np.full(expected_dim, np.inf),
        ]
        for action in invalid_actions:
            with self.subTest(size=len(action)):
                with self.assertRaises(ValueError):
                    client.publish_initial_pose_sync(
                        action,
                        action_keys,
                        duration_s=5.0,
                    )

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_unknown_action_key_is_rejected_before_publish(self):
        client = self._make_client("omy_f3m")
        self._seed_joint_state(client)

        with self.assertRaisesRegex(ValueError, "unknown action key"):
            client.publish_initial_pose_sync(
                np.zeros(1),
                ["missing_action_group"],
                duration_s=5.0,
            )

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_embedded_inference_request_matches_native_service_field_order(self):
        messages_path = MODULE_PATH.parent / "messages" / "__init__.py"
        messages_spec = importlib.util.spec_from_file_location(
            "robot_client_messages_impl",
            messages_path,
        )
        messages = importlib.util.module_from_spec(messages_spec)
        messages_spec.loader.exec_module(messages)

        native_request = (
            MODULE_PATH.parents[4] / "interfaces" / "srv" / "InferenceCommand.srv"
        ).read_text(encoding="utf-8").split("---", maxsplit=1)[0]

        def serialized_fields(definition: str) -> list[str]:
            fields = []
            for raw_line in definition.splitlines():
                line = raw_line.split("#", maxsplit=1)[0].strip()
                if not line or "=" in line:
                    continue
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_/\[\]]*\s+[a-z][a-z0-9_]*", line):
                    fields.append(line)
            return fields

        self.assertEqual(
            serialized_fields(messages.INFERENCE_COMMAND_REQUEST_DEF),
            serialized_fields(native_request),
        )


if __name__ == "__main__":
    unittest.main()
