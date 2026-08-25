"""Tests for success-only LeRobot episode discovery."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from cyclo_brain.model.multi_task_dit.success_dataset import (
    classify_episode_outcome_rows,
    discover_episode_outcomes,
)


def _row(
    episode_index: object,
    length: int,
    outcome: bool,
) -> dict[str, object]:
    return {
        "episode_index": episode_index,
        "length": length,
        "stats/episode_success/min": torch.tensor([outcome]),
        "stats/episode_success/max": np.asarray([outcome]),
        "stats/episode_success/mean": [float(outcome)],
        "stats/episode_success/count": (length,),
    }


class EpisodeOutcomeRowsTest(unittest.TestCase):
    def test_unwraps_values_sorts_ids_and_counts_frames(self):
        rows = [_row(9, 7, True), _row(np.asarray([2]), 3, False), _row(4, 5, True)]

        result = classify_episode_outcome_rows(rows)

        self.assertEqual(result.success_episodes, (4, 9))
        self.assertEqual(result.failure_episodes, (2,))
        self.assertEqual(result.success_episode_count, 2)
        self.assertEqual(result.failure_episode_count, 1)
        self.assertEqual(result.total_episode_count, 3)
        self.assertEqual(result.success_frames, 12)
        self.assertEqual(result.failure_frames, 3)
        self.assertEqual(result.total_frames, 15)

    def test_rejects_mixed_labels(self):
        row = _row(0, 3, True)
        row["stats/episode_success/min"] = [False]
        with self.assertRaisesRegex(ValueError, "mixed success labels"):
            classify_episode_outcome_rows([row])

    def test_rejects_contradictory_mean(self):
        row = _row(0, 3, True)
        row["stats/episode_success/mean"] = [0.5]
        with self.assertRaisesRegex(ValueError, "mean 0.5 contradicts"):
            classify_episode_outcome_rows([row])

    def test_rejects_count_length_mismatch(self):
        row = _row(0, 3, False)
        row["stats/episode_success/count"] = [2]
        with self.assertRaisesRegex(ValueError, "count 2 does not match length 3"):
            classify_episode_outcome_rows([row])

    def test_rejects_duplicate_episode_id(self):
        with self.assertRaisesRegex(ValueError, "duplicate episode_index 0"):
            classify_episode_outcome_rows([_row(0, 3, True), _row(0, 3, True)])

    def test_rejects_non_singleton_stats(self):
        row = _row(0, 3, True)
        row["stats/episode_success/max"] = [True, True]
        with self.assertRaisesRegex(ValueError, "exactly one value"):
            classify_episode_outcome_rows([row])

    def test_rejects_empty_metadata(self):
        with self.assertRaisesRegex(ValueError, "contains no episodes"):
            classify_episode_outcome_rows([])


class EpisodeOutcomeParquetTest(unittest.TestCase):
    def test_discovers_recursively_in_deterministic_order(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for chunk, row in (("chunk-001", _row(6, 4, False)), ("chunk-000", _row(1, 2, True))):
                destination = root / "meta" / "episodes" / chunk / "file-000.parquet"
                destination.parent.mkdir(parents=True)
                plain_row = {
                    key: value.tolist() if isinstance(value, (np.ndarray, torch.Tensor)) else value
                    for key, value in row.items()
                }
                pq.write_table(pa.Table.from_pylist([plain_row]), destination)

            result = discover_episode_outcomes(root)

        self.assertEqual(result.success_episodes, (1,))
        self.assertEqual(result.failure_episodes, (6,))
        self.assertEqual(result.total_frames, 6)


if __name__ == "__main__":
    unittest.main()
