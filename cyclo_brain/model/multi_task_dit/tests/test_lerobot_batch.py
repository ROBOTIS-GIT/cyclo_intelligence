"""Tests for Cyclo's LeRobot-to-MultiTaskDiT batch boundary."""

import unittest

import torch

from cyclo_brain.model.multi_task_dit import (
    CYCLO_SG2_CAMERA_KEYS,
    canonicalize_dataset_stats,
    canonicalize_training_batch,
)


def _batch() -> dict:
    result = {
        "observation.state": torch.zeros(2, 1, 22),
        "action": torch.zeros(2, 16, 22),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        "task": torch.zeros(2, dtype=torch.long),
        "observation.images.rgb.cam_right_head": torch.zeros(2, 3, 256, 256),
    }
    for index, key in enumerate(CYCLO_SG2_CAMERA_KEYS):
        result[key] = torch.full((2, 3, 256, 256), index + 1, dtype=torch.uint8)
    return result


class CycloLeRobotBatchTest(unittest.TestCase):
    def test_batch_adds_history_scales_uint8_and_overrides_bad_task(self):
        source = _batch()
        result = canonicalize_training_batch(source, task_instruction="pick up the jelly bag")

        self.assertEqual(result["observation.state"].shape, (2, 1, 22))
        self.assertEqual(result["action"].shape, (2, 16, 22))
        self.assertIs(result["action_is_pad"], source["action_is_pad"])
        self.assertEqual(result["task"], ["pick up the jelly bag"] * 2)
        self.assertNotIn("observation.images.rgb.cam_right_head", result)
        for index, key in enumerate(CYCLO_SG2_CAMERA_KEYS):
            self.assertEqual(result[key].shape, (2, 1, 3, 256, 256))
            self.assertEqual(result[key].dtype, torch.float32)
            torch.testing.assert_close(
                result[key],
                torch.full_like(result[key], (index + 1) / 255.0),
            )

    def test_missing_camera_is_rejected(self):
        source = _batch()
        source.pop(CYCLO_SG2_CAMERA_KEYS[1])
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            canonicalize_training_batch(source)

    def test_identical_episode_duplicated_stats_are_collapsed(self):
        stats = {
            key: {
                name: torch.arange(3, dtype=torch.float32)[:, None, None].expand(3, 4, 1).clone()
                for name in ("mean", "std", "min", "max")
            }
            for key in CYCLO_SG2_CAMERA_KEYS
        }
        result = canonicalize_dataset_stats(stats)
        for key in CYCLO_SG2_CAMERA_KEYS:
            for name in ("mean", "std", "min", "max"):
                self.assertEqual(result[key][name].shape, (3, 1, 1))

    def test_non_identical_duplicated_stats_are_rejected(self):
        stats = {
            key: {
                name: torch.zeros(3, 2, 1)
                for name in ("mean", "std", "min", "max")
            }
            for key in CYCLO_SG2_CAMERA_KEYS
        }
        stats[CYCLO_SG2_CAMERA_KEYS[0]]["mean"][0, 1, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "non-identical duplicated mean"):
            canonicalize_dataset_stats(stats)


if __name__ == "__main__":
    unittest.main()
