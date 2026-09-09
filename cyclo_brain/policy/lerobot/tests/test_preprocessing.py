#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest

import numpy as np
import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

constants_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.constants",
    ENGINE_DIR / "constants.py",
)
constants = importlib.util.module_from_spec(constants_spec)
sys.modules[constants_spec.name] = constants
constants_spec.loader.exec_module(constants)

image_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.image_preprocessing",
    ENGINE_DIR / "image_preprocessing.py",
)
image_preprocessing = importlib.util.module_from_spec(image_spec)
sys.modules[image_spec.name] = image_preprocessing
image_spec.loader.exec_module(image_preprocessing)

preprocessing_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.preprocessing",
    ENGINE_DIR / "preprocessing.py",
)
preprocessing = importlib.util.module_from_spec(preprocessing_spec)
sys.modules[preprocessing_spec.name] = preprocessing
preprocessing_spec.loader.exec_module(preprocessing)

PreprocessingMixin = preprocessing.PreprocessingMixin
STATE_KEY = constants.STATE_KEY


class FakeRobot:
    _config = {"cameras": {}}

    def __init__(self, positions):
        self._positions = positions

    def get_images(self, format="rgb"):
        return {"unused": np.zeros((2, 2, 3), dtype=np.uint8)}

    def get_joint_positions(self):
        return {"follower_arm": self._positions}


class Preprocessor(PreprocessingMixin):
    def __init__(self, positions, expected):
        self._robot = FakeRobot(positions)
        self._cameras = {}
        self._state_modalities = ["arm"]
        self._image_preprocessing = None
        self._device = torch.device("cpu")
        feature = SimpleNamespace(shape=(expected,))
        config = SimpleNamespace(input_features={STATE_KEY: feature})
        self._policy = SimpleNamespace(config=config)

    def _fail(self, message):
        return {"error": message}


class PreprocessingTest(unittest.TestCase):
    def camera_preprocessor(self, operations):
        preprocessor = Preprocessor([1.0, 2.0], expected=2)
        key = "observation.images.head"
        preprocessor._cameras = {"head": key}
        preprocessor._robot._config = {"cameras": {"head": {"rotation_deg": 270}}}
        image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
        preprocessor._robot.get_images = lambda format: {"head": image}
        preprocessor._image_preprocessing = image_preprocessing.ImagePreprocessing(
            {"backend": "torch", "operations": operations},
            {key: {"shape": [3, 4, 4]}},
        )
        return preprocessor, key, image

    def test_real_observation_path_preserves_native_rotated_size(self):
        preprocessor, key, image = self.camera_preprocessor([{"type": "identity"}])
        batch = preprocessor._build_observation("pick")
        expected = torch.from_numpy(np.rot90(image).copy()).float().div(255).permute(2, 0, 1).unsqueeze(0)
        torch.testing.assert_close(batch[key], expected, rtol=0, atol=0)
        self.assertEqual(batch["task"], ["pick"])

    def test_camera_transform_error_returns_failure_not_partial_batch(self):
        preprocessor, key, _ = self.camera_preprocessor([{"type": "center_crop", "size": [100, 100]}])
        result = preprocessor._build_observation("pick")
        self.assertIn("Camera preprocessing failed for head", result["error"])
        self.assertNotIn(key, result)

    def test_diffusion_stack_validation_runs_on_processed_sizes(self):
        preprocessor, key, _ = self.camera_preprocessor([{"type": "identity"}])
        preprocessor._policy.config.type = "diffusion"
        preprocessor._cameras["wrist"] = "observation.images.wrist"
        with self.assertRaisesRegex(ValueError, "equal sizes"):
            preprocessor._validate_camera_shapes({key: torch.zeros(1, 3, 8, 12),
                "observation.images.wrist": torch.zeros(1, 3, 12, 8)})
        preprocessor._validate_camera_shapes({key: torch.zeros(1, 3, 4, 4),
            "observation.images.wrist": torch.zeros(1, 3, 4, 4)})

    def test_groot_does_not_recheck_removed_raw_camera_keys(self):
        preprocessor, key, _ = self.camera_preprocessor([{"type": "identity"}])
        preprocessor._policy.config.type = "groot"
        # GR00T removes raw camera keys during packing; don't inspect them afterwards.
        preprocessor._validate_camera_shapes({"video": object()})

    def test_xvla_stack_requires_equal_sizes_only_without_internal_resize(self):
        preprocessor, key, _ = self.camera_preprocessor([{"type": "identity"}])
        preprocessor._policy.config.type = "xvla"
        preprocessor._policy.config.resize_imgs_with_padding = None
        wrist = "observation.images.wrist"
        preprocessor._cameras["wrist"] = wrist
        batch = {key: torch.zeros(1, 3, 8, 12), wrist: torch.zeros(1, 3, 12, 8)}
        with self.assertRaisesRegex(ValueError, "xvla cameras must have equal sizes"):
            preprocessor._validate_camera_shapes(batch)
        preprocessor._policy.config.resize_imgs_with_padding = (224, 224)
        preprocessor._validate_camera_shapes(batch)
        preprocessor._policy.config.resize_imgs_with_padding = None
        batch[wrist] = torch.zeros(1, 3, 8, 12)
        preprocessor._validate_camera_shapes(batch)

    def test_pads_short_state_to_policy_shape(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=4)

        batch = preprocessor._build_observation("task")

        np.testing.assert_allclose(
            batch[STATE_KEY].numpy(),
            np.asarray([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_truncates_long_state_to_policy_shape(self):
        preprocessor = Preprocessor([1.0, 2.0, 3.0, 4.0], expected=2)

        batch = preprocessor._build_observation("task")

        np.testing.assert_allclose(
            batch[STATE_KEY].numpy(),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
