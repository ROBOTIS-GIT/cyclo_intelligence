#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.rlt_stage1_training_cli import (  # noqa: E402
    _FeatureCache,
    _FeatureCacheWriter,
    build_parser,
)


class RLTStage1TrainingCLITests(unittest.TestCase):
    def test_feature_cache_pads_variable_backbone_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            writer = _FeatureCacheWriter(root)
            writer.append(
                {
                    "tokens": torch.randn(2, 3, 8),
                    "token_valid": torch.ones(2, 3, dtype=torch.bool),
                    "image_token": torch.tensor(
                        [[True, True, False], [True, False, False]]
                    ),
                }
            )
            writer.append(
                {
                    "tokens": torch.randn(1, 5, 8),
                    "token_valid": torch.ones(1, 5, dtype=torch.bool),
                    "image_token": torch.tensor([[True, True, True, False, False]]),
                }
            )
            writer.finalize({"test": True})

            cache = _FeatureCache(root)
            tokens, valid, image = cache.batch([2, 0], torch.device("cpu"))

            self.assertEqual(tuple(tokens.shape), (2, 5, 8))
            self.assertEqual(tuple(valid.shape), (2, 5))
            self.assertEqual(tuple(image.shape), (2, 5))
            self.assertEqual(cache.sample_count, 3)
            self.assertEqual(cache.max_selected_tokens, 3)
            self.assertFalse(bool(valid[1, 3:].any()))

    def test_parser_accepts_multiple_selected_datasets(self) -> None:
        args = build_parser().parse_args(
            [
                "--groot-checkpoint",
                "/workspace/model/groot/showroom_groot",
                "--dataset-root",
                "/workspace/lerobot/a_v21",
                "--dataset-root",
                "/workspace/lerobot/b_v21",
                "--output-dir",
                "/workspace/checkpoint/rlt/stage1/run",
                "--job-id",
                "run",
                "--steps",
                "100",
                "--batch-size",
                "2",
                "--save-freq",
                "10",
            ]
        )

        self.assertEqual(len(args.dataset_root), 2)
        self.assertEqual(args.steps, 100)


if __name__ == "__main__":
    unittest.main()
