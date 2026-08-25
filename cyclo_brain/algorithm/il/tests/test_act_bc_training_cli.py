from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cyclo_brain.algorithm.il.act_bc.training_cli import build_parser, config_from_args


class ACTBCTrainingCLITest(unittest.TestCase):
    def test_repeated_roots_and_success_groups_remain_ordered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(
                [
                    "--dataset-root",
                    str(root / "epoch0"),
                    "--success-episodes",
                    "0,2",
                    "--dataset-root",
                    str(root / "epoch1"),
                    "--success-episodes",
                    "1",
                    "--output-dir",
                    str(root / "output"),
                    "--steps",
                    "100",
                    "--batch-size",
                    "8",
                    "--save-freq",
                    "20",
                    "--device",
                    "cpu",
                ]
            )
            config = config_from_args(args)
            self.assertEqual(
                [selection.root.name for selection in config.selections],
                ["epoch0", "epoch1"],
            )
            self.assertEqual(
                [selection.success_episodes for selection in config.selections],
                [(0, 2), (1,)],
            )
            self.assertEqual(config.chunk_size, 30)
            self.assertEqual(config.batch_size, 8)

    def test_root_and_success_repeat_counts_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(
                [
                    "--dataset-root",
                    str(root / "epoch0"),
                    "--success-episodes",
                    "0",
                    "--dataset-root",
                    str(root / "epoch1"),
                    "--output-dir",
                    str(root / "output"),
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--save-freq",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            with self.assertRaisesRegex(ValueError, "same number"):
                config_from_args(args)

    def test_episodes_alias_accepts_unlabeled_demonstration_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(
                [
                    "--dataset-root",
                    str(root / "unlabeled"),
                    "--episodes",
                    "0,1,2",
                    "--output-dir",
                    str(root / "output"),
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--save-freq",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            config = config_from_args(args)
            self.assertEqual(config.selections[0].success_episodes, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
