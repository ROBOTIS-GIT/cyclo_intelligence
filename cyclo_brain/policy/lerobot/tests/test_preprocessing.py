#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest import mock

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
    def test_input_plan_is_compiled_once_for_repeated_predictions(self):
        preprocessor = Preprocessor([1., 2.], expected=2)
        with mock.patch.object(preprocessing, "latest_input_plan", wraps=preprocessing.latest_input_plan) as compile_plan:
            for _ in range(10):
                preprocessor._build_observation("task")
        self.assertEqual(compile_plan.call_count, 1)

    def test_step_prefers_readiness_checked_selected_snapshot(self):
        preprocessor = Preprocessor([1., 2.], expected=2)

        class SelectiveRobot(FakeRobot):
            def get_input_snapshot(self):
                raise AssertionError("must not copy all inputs while waiting")

            def get_required_input_snapshot(self, sources, *, max_age_s, after_s):
                assert sources == {"joint:follower_arm"}
                assert max_age_s == 1. and after_s == 9.
                self.calls += 1
                if self.calls < 3:
                    raise ValueError("joint:follower_arm: stale")
                return {"images": {}, "joint_positions": {"follower_arm": np.ones(2)},
                        "sensors": {}, "reception_monotonic_timestamps": {"joint:follower_arm": 10.}}

        preprocessor._robot = SelectiveRobot([1., 2.])
        preprocessor._robot.calls = 0
        with mock.patch.object(preprocessing.time, "monotonic", return_value=10.1), \
                mock.patch.object(preprocessing.time, "sleep"):
            batch = preprocessor._build_observation("task", require_received=True, observation_after_s=9.)
        self.assertIn(STATE_KEY, batch)
        self.assertEqual(preprocessor._robot.calls, 3)

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

    def test_step_waits_for_each_required_source_before_transforming(self):
        preprocessor, key, image = self.camera_preprocessor([{"type": "identity"}])

        class SnapshotRobot(FakeRobot):
            def get_input_snapshot(self):
                return next(snapshots)

        preprocessor._robot = SnapshotRobot([1., 2.])
        preprocessor._robot._config = {"cameras": {"head": {"rotation_deg": 270}}}

        def snapshot(camera_stamp, joint_stamp):
            return {"images": {"head": image}, "joint_positions": {"follower_arm": [1., 2.]},
                    "sensors": {}, "reception_monotonic_timestamps": {
                        "camera:head": camera_stamp, "joint:follower_arm": joint_stamp}}

        # A fresh joint alone is not enough. Unused cameras/sensors are not required.
        snapshots = iter([snapshot(9., 10.1), snapshot(10.1, 9.), snapshot(10.2, 10.2)])
        with mock.patch.object(preprocessing.time, "monotonic", return_value=10.3), \
                mock.patch.object(preprocessing.time, "sleep") as sleep, \
                mock.patch.object(preprocessor, "_transform_image", wraps=preprocessor._transform_image) as transform:
            batch = preprocessor._build_observation("pick", require_received=True, observation_after_s=10.)
        self.assertIn(key, batch)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(transform.call_count, 1)
        self.assertEqual(batch["task"], ["pick"])

    def test_step_missing_timestamps_fail_before_model_preprocessing(self):
        preprocessor = Preprocessor([1., 2.], expected=2)
        with mock.patch.object(preprocessor, "_transform_state") as transform:
            result = preprocessor._build_observation("pick", require_received=True)
        self.assertIn("timestamped RobotClient", result["error"])
        transform.assert_not_called()

    def test_step_stale_snapshot_has_bounded_wait_and_explicit_source_error(self):
        preprocessor = Preprocessor([1., 2.], expected=2)

        class SnapshotRobot(FakeRobot):
            def get_input_snapshot(self):
                return {"images": {}, "joint_positions": {"follower_arm": [1., 2.]}, "sensors": {},
                        "reception_monotonic_timestamps": {"joint:follower_arm": 5.}}

        preprocessor._robot = SnapshotRobot([1., 2.])
        with mock.patch.object(preprocessing.time, "monotonic", side_effect=[10., 10., 11.]), \
                mock.patch.object(preprocessing.time, "sleep") as sleep, \
                mock.patch.object(preprocessor, "_transform_state") as transform:
            result = preprocessor._build_observation("pick", require_received=True)
        self.assertIn("joint:follower_arm: stale", result["error"])
        self.assertEqual(sleep.call_count, 1)
        transform.assert_not_called()


if __name__ == "__main__":
    unittest.main()
