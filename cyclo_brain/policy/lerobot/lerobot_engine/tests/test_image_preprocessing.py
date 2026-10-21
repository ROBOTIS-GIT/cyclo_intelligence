#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "image_preprocessing.py"
spec = importlib.util.spec_from_file_location("image_preprocessing", MODULE_PATH)
image_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_preprocessing)


class ImagePreprocessingTest(unittest.TestCase):
    def test_extracts_per_policy_image_resize_targets(self):
        features = {
            "observation.state": SimpleNamespace(shape=(22,)),
            "observation.images.cam_head_left": SimpleNamespace(shape=(3, 720, 1280)),
            "observation.images.cam_wrist_left": SimpleNamespace(shape=(3, 640, 480)),
        }

        targets = image_preprocessing.infer_image_resize_targets(features)

        self.assertEqual(
            targets,
            {
                "observation.images.cam_head_left": (1280, 720),
                "observation.images.cam_wrist_left": (480, 640),
            },
        )

    def test_rotates_wrist_image_from_640x480_to_480x640(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        rotated = image_preprocessing.apply_rotation(image, 270)

        self.assertEqual(rotated.shape, (640, 480, 3))

    def test_letterbox_head_preserves_aspect_ratio_and_zero_padding(self):
        tensor = image_preprocessing.prepare_letterbox_tensor(
            np.full((720, 1280, 3), 255, dtype=np.uint8), target_size=(384, 384)
        ).numpy()
        self.assertEqual(tensor.shape, (1, 3, 384, 384))
        np.testing.assert_allclose(tensor[:, :, :84], 0)
        np.testing.assert_allclose(tensor[:, :, 84:300], 1, atol=3e-7)
        np.testing.assert_allclose(tensor[:, :, 300:], 0)

    def test_letterbox_rotated_wrist_is_padded_horizontally(self):
        tensor = image_preprocessing.prepare_letterbox_tensor(
            np.full((480, 640, 3), 255, dtype=np.uint8),
            rotation_deg=270, target_size=(384, 384),
        ).numpy()
        np.testing.assert_allclose(tensor[:, :, :, :48], 0)
        np.testing.assert_allclose(tensor[:, :, :, 48:336], 1, atol=3e-7)
        np.testing.assert_allclose(tensor[:, :, :, 336:], 0)


if __name__ == "__main__":
    unittest.main()
