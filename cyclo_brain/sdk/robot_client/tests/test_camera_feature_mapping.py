#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot_client"
    / "camera_mapping.py"
)
spec = importlib.util.spec_from_file_location("camera_mapping", MODULE_PATH)
camera_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_mapping)

resolve_camera_feature_sources = camera_mapping.resolve_camera_feature_sources


class CameraFeatureSourceMappingTest(unittest.TestCase):
    def test_maps_behavior_10k_dataset_keys_to_robot_sources(self):
        self.assertEqual(
            resolve_camera_feature_sources(
                [
                    "observation.images.rgb.cam_left_head",
                    "observation.images.rgb.cam_right_head",
                    "observation.images.rgb.cam_left_wrist",
                    "observation.images.rgb.cam_right_wrist",
                ],
                [
                    "cam_left_head",
                    "cam_right_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
            ),
            {
                "observation.images.rgb.cam_left_head": "cam_left_head",
                "observation.images.rgb.cam_right_head": "cam_right_head",
                "observation.images.rgb.cam_left_wrist": "cam_left_wrist",
                "observation.images.rgb.cam_right_wrist": "cam_right_wrist",
            },
        )

    def test_maps_legacy_swapped_keys_to_dataset_canonical_sources(self):
        self.assertEqual(
            resolve_camera_feature_sources(
                [
                    "observation.images.cam_head_left",
                    "observation.images.cam_head_right",
                    "observation.images.cam_wrist_left",
                    "observation.images.cam_wrist_right",
                ],
                [
                    "cam_left_head",
                    "cam_right_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
            ),
            {
                "observation.images.cam_head_left": "cam_left_head",
                "observation.images.cam_head_right": "cam_right_head",
                "observation.images.cam_wrist_left": "cam_left_wrist",
                "observation.images.cam_wrist_right": "cam_right_wrist",
            },
        )

    def test_maps_legacy_camera_keys_without_cam_prefix(self):
        self.assertEqual(
            resolve_camera_feature_sources(
                [
                    "observation.images.head_left",
                    "observation.images.wrist_right",
                ],
                ["cam_left_head", "cam_right_wrist"],
            ),
            {
                "observation.images.head_left": "cam_left_head",
                "observation.images.wrist_right": "cam_right_wrist",
            },
        )

    def test_keeps_custom_camera_names_without_cyclo_semantics(self):
        self.assertEqual(
            resolve_camera_feature_sources(
                ["observation.images.front"],
                ["front"],
            ),
            {"observation.images.front": "front"},
        )

    def test_rejects_two_model_keys_for_one_robot_source(self):
        with self.assertRaisesRegex(RuntimeError, "matched multiple model keys"):
            resolve_camera_feature_sources(
                [
                    "observation.images.rgb.cam_left_head",
                    "observation.images.cam_left_head",
                ],
                ["cam_left_head"],
            )


if __name__ == "__main__":
    unittest.main()
