#!/usr/bin/env python3

import sys
import types
import unittest
import importlib.util
from unittest import mock
from pathlib import Path


robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
robot_client_stub.__path__ = []
sys.modules.setdefault("robot_client", robot_client_stub)

CAMERA_MAPPING_PATH = (
    Path(__file__).resolve().parents[3]
    / "sdk"
    / "robot_client"
    / "robot_client"
    / "camera_mapping.py"
)
camera_mapping_spec = importlib.util.spec_from_file_location(
    "robot_client.camera_mapping",
    CAMERA_MAPPING_PATH,
)
camera_mapping = importlib.util.module_from_spec(camera_mapping_spec)
sys.modules[camera_mapping_spec.name] = camera_mapping
camera_mapping_spec.loader.exec_module(camera_mapping)

ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

spec = importlib.util.spec_from_file_location(
    "lerobot_engine.io_mapping",
    ENGINE_DIR / "io_mapping.py",
)
io_mapping = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = io_mapping
spec.loader.exec_module(io_mapping)
IoMappingMixin = io_mapping.IoMappingMixin


class IoMappingCameraTest(unittest.TestCase):
    def setUp(self):
        # These tests isolate camera/readiness wiring; channel resolution has
        # real-layout coverage in test_channel_mapping.py.
        self.mapping = mock.Mock(action_keys=["arm"], state_names=("a",), joint_views={})
        for patcher in (
            mock.patch.object(io_mapping, "ChannelMapping", return_value=self.mapping),
            mock.patch("cyclo_lerobot_io.mapping.read_mapping", return_value=None),
            mock.patch.object(IoMappingMixin, "_loaded_model_path", "/test", create=True),
            mock.patch.object(IoMappingMixin, "_preprocessor", None, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_load_checks_only_policy_cameras_and_consumed_state(self):
        for ready in (False, True):
            with self.subTest(ready=ready):
                robot = mock.Mock()
                robot.camera_names = ["required_eye", "unused_eye"]
                robot._config = {
                    "joint_groups": {"follower_arm": {"role": "follower"}},
                    "sensors": {"odom": {}},
                }
                robot.wait_for_ready.return_value = ready
                robot.get_missing_observations.return_value = ["joint:follower_arm"]
                engine = IoMappingMixin()
                engine._policy = types.SimpleNamespace(config=types.SimpleNamespace(type="act"))
                engine._policy_image_keys = lambda: {"observation.images.required_eye"}
                engine._observation_plan = lambda _: (None, types.SimpleNamespace(required_observations={
                    "camera_names": ["required_eye"], "joint_groups": ["follower_arm"], "sensor_names": ["odom"],
                }))
                with mock.patch.object(io_mapping, "RobotClient", return_value=robot) as construct:
                    if ready:
                        engine._init_robot("some_robot")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "joint:follower_arm"):
                            engine._init_robot("some_robot")
                construct.assert_called_once_with("some_robot", defer_subscriptions=True)
                robot.start_observation_subscriptions.assert_called_once_with(
                    camera_names=["required_eye"], joint_groups=["follower_arm"], sensor_names=["odom"],
                )
                robot.wait_for_ready.assert_called_once_with(
                    timeout=10.0, camera_names=["required_eye"],
                    joint_groups=["follower_arm"], sensor_names=["odom"],
                )

    def test_adapter_layout_validation_runs_before_waiting_for_observations(self):
        from lerobot_engine.adapters import AdapterDefinition

        robot = mock.Mock()
        robot.camera_names = []
        robot._config = {"joint_groups": {"follower_arm": {"role": "follower"}}}
        config = types.SimpleNamespace(type="custom")
        validate = mock.Mock(side_effect=ValueError("custom model layout mismatch"))
        engine = IoMappingMixin()
        engine._policy = types.SimpleNamespace(config=config)
        engine._adapter_definition = AdapterDefinition(layout_validator=validate)
        engine._policy_image_keys = lambda: set()
        with mock.patch.object(io_mapping, "RobotClient", return_value=robot):
            with self.assertRaisesRegex(ValueError, "custom model layout mismatch"):
                engine._init_robot("some_robot")
        validate.assert_called_once_with(config, robot, ["arm"], ["arm"], layout=self.mapping)
        robot.wait_for_ready.assert_not_called()
        robot.start_observation_subscriptions.assert_not_called()

    def test_readiness_uses_declared_sources_not_all_robot_config_modalities(self):
        robot = mock.Mock()
        robot.camera_names = ["head"]
        robot._config = {"joint_groups": {"follower_arm": {"role": "follower"},
                                          "follower_unused": {"role": "follower"}},
                         "sensors": {"odom": {}}}
        engine = IoMappingMixin()
        engine._policy = types.SimpleNamespace(config=types.SimpleNamespace(type="custom"))
        engine._policy_image_keys = lambda: {"observation.images.head"}
        required = {"camera_names": [], "joint_groups": ["follower_arm"], "sensor_names": []}
        engine._observation_plan = mock.Mock(return_value=(None, types.SimpleNamespace(required_observations=required)))
        with mock.patch.object(io_mapping, "RobotClient", return_value=robot):
            engine._init_robot("some_robot")
        engine._observation_plan.assert_called_once_with(False)
        robot.wait_for_ready.assert_called_once_with(timeout=10.0, **required)
        robot.start_observation_subscriptions.assert_called_once_with(**required)

    def test_missing_checkpoint_image_metadata_keeps_legacy_default_keys(self):
        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                ["cam_left_head", "cam_left_wrist"],
                set(),
            ),
            {
                "cam_left_head": "observation.images.cam_left_head",
                "cam_left_wrist": "observation.images.cam_left_wrist",
            },
        )

    def test_maps_rgb_prefixed_cameras_to_policy_keys(self):
        robot_cameras = [
            "rgb.cam_left_head",
            "rgb.cam_right_head",
            "rgb.cam_left_wrist",
            "rgb.cam_right_wrist",
        ]
        policy_keys = {
            "observation.images.cam_left_head",
            "observation.images.cam_right_head",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        }

        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(robot_cameras, policy_keys),
            {
                "rgb.cam_left_head": "observation.images.cam_left_head",
                "rgb.cam_right_head": "observation.images.cam_right_head",
                "rgb.cam_left_wrist": "observation.images.cam_left_wrist",
                "rgb.cam_right_wrist": "observation.images.cam_right_wrist",
            },
        )

    def test_keeps_exact_camera_key_preferred(self):
        robot_cameras = ["rgb.cam_left_head"]
        policy_keys = {
            "observation.images.rgb.cam_left_head",
            "observation.images.cam_left_head",
        }

        with self.assertRaisesRegex(RuntimeError, "matched multiple model keys"):
            IoMappingMixin._resolve_camera_mappings(robot_cameras, policy_keys)

        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                robot_cameras,
                {"observation.images.rgb.cam_left_head"},
            ),
            {"rgb.cam_left_head": "observation.images.rgb.cam_left_head"},
        )


if __name__ == "__main__":
    unittest.main()
