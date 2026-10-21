#!/usr/bin/env python3

import sys
import types
import unittest
import importlib.util
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path


robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_client_stub)

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


class IoMappingCameraAliasTest(unittest.TestCase):
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

        with self.assertRaisesRegex(RuntimeError, "Missing camera mappings"):
            IoMappingMixin._resolve_camera_mappings(robot_cameras, policy_keys)

        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                robot_cameras,
                {"observation.images.rgb.cam_left_head"},
            ),
            {"rgb.cam_left_head": "observation.images.rgb.cam_left_head"},
        )

    def test_maps_legacy_single_head_policy_key_to_left_head_camera(self):
        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                [
                    "cam_left_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
                {
                    "observation.images.rgb.cam_head",
                    "observation.images.cam_wrist_left",
                    "observation.images.cam_wrist_right",
                },
            ),
            {
                "cam_left_head": "observation.images.rgb.cam_head",
                "cam_left_wrist": "observation.images.cam_wrist_left",
                "cam_right_wrist": "observation.images.cam_wrist_right",
            },
        )


class ActIoMappingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = IoMappingMixin()
        self.engine._loaded_model_path = self.tmp.name
        self.engine._has_mobile_state = False
        self.engine._state_modalities = ["arm_left", "arm_right"]
        self.engine._action_keys = ["arm_left", "arm_right"]
        self.engine._policy = SimpleNamespace(config=SimpleNamespace(
            input_features={"observation.state": SimpleNamespace(shape=(16,))},
            output_features={"action": SimpleNamespace(shape=(8,))},
        ))
        self.engine._robot = SimpleNamespace(
            _config={"joint_groups": {
                "follower_" + side: {"role": "follower", "joint_names": list(range(8))}
                for side in ("arm_left", "arm_right")
            }},
            _action_groups={
                side: {"msg_type": "trajectory_msgs/msg/JointTrajectory", "joint_names": list(range(8))}
                for side in ("arm_left", "arm_right")
            },
        )

    def write_mapping(self, **overrides):
        mapping = {"robot_type": "ffw_sg2_rev1",
                   "state_modalities": ["arm_left", "arm_right"],
                   "action_keys": ["arm_right"]}
        mapping.update(overrides)
        (Path(self.tmp.name) / "cyclo_inference.json").write_text(json.dumps(mapping))

    def test_subset_action_requires_explicit_mapping(self):
        with self.assertRaisesRegex(ValueError, "action dimension mismatch"):
            self.engine._configure_act_io("ffw_sg2_rev1")

    def test_right_only_action_keeps_both_state_arms(self):
        self.write_mapping()
        self.engine._configure_act_io("ffw_sg2_rev1")
        self.assertEqual(self.engine._state_modalities, ["arm_left", "arm_right"])
        self.assertEqual(self.engine._action_keys, ["arm_right"])

    def test_right_only_state_is_not_left_prefix(self):
        self.engine._policy.config.input_features["observation.state"].shape = (8,)
        self.write_mapping(state_modalities=["arm_right"])
        self.engine._configure_act_io("ffw_sg2_rev1")
        self.assertEqual(self.engine._state_modalities, ["arm_right"])

    def test_preserves_explicit_training_order(self):
        self.write_mapping(state_modalities=["arm_right", "arm_left"])
        self.engine._configure_act_io("ffw_sg2_rev1")
        self.assertEqual(self.engine._state_modalities, ["arm_right", "arm_left"])

    def test_rejects_unknown_duplicate_or_wrong_size_mapping(self):
        for values in (["missing"], ["arm_right", "arm_right"], ["arm_left", "arm_right"]):
            with self.subTest(values=values):
                self.write_mapping(action_keys=values)
                with self.assertRaises(ValueError):
                    self.engine._configure_act_io("ffw_sg2_rev1")

    def test_rejects_wrong_robot(self):
        self.write_mapping(robot_type="another_robot")
        with self.assertRaisesRegex(ValueError, "robot_type"):
            self.engine._configure_act_io("ffw_sg2_rev1")


if __name__ == "__main__":
    unittest.main()
