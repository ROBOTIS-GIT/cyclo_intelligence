from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cyclo_brain.algorithm.rl.label_lerobot_v3 import (
    _patch_episode_statistics,
    _patch_global_statistics,
    _restore_source_dataset_info,
    _validate_labeled_dataset,
)


class LabelLeRobotV3Test(unittest.TestCase):
    def test_source_metadata_extensions_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "meta").mkdir(parents=True)
            (destination / "meta").mkdir(parents=True)
            source_info = {
                "codebase_version": "v3.0",
                "fps": 15,
                "annotation_path": (
                    "annotations/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.json"
                ),
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": ["joint"],
                        "fps": 15,
                    }
                },
            }
            generated_info = {
                "codebase_version": "v3.0",
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": ["joint"],
                    },
                    "episode_success": {
                        "dtype": "bool",
                        "shape": [1],
                        "names": None,
                        "fps": 15,
                    },
                },
            }
            (source / "meta" / "info.json").write_text(
                json.dumps(source_info), encoding="utf-8"
            )
            (destination / "meta" / "info.json").write_text(
                json.dumps(generated_info), encoding="utf-8"
            )

            _restore_source_dataset_info(source, destination)

            restored = json.loads(
                (destination / "meta" / "info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(restored["annotation_path"], source_info["annotation_path"])
            self.assertEqual(restored["features"]["action"], source_info["features"]["action"])
            self.assertEqual(restored["features"]["episode_success"]["dtype"], "bool")

    def test_success_statistics_match_cyclo_v3_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "codebase_version": "v3.0",
                        "fps": 15,
                        "total_episodes": 2,
                        "total_frames": 3,
                        "features": {
                            "episode_success": {
                                "dtype": "bool",
                                "shape": [1],
                                "names": None,
                                "fps": 15,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            pq.write_table(
                pa.table({"episode_success": pa.array([True, True, True])}),
                root / "data" / "chunk-000" / "file-000.parquet",
            )
            pq.write_table(
                pa.table(
                    {
                        "episode_index": pa.array([0, 1], type=pa.int64()),
                        "length": pa.array([2, 1], type=pa.int64()),
                    }
                ),
                root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            )

            self.assertEqual(_patch_episode_statistics(root, True), 2)
            _patch_global_statistics(root, True)

            self.assertEqual(
                _validate_labeled_dataset(root, expected_success=True),
                (2, 3),
            )
            episodes = pq.read_table(
                root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            ).to_pydict()
            self.assertEqual(
                episodes["stats/episode_success/count"],
                [[2], [1]],
            )


if __name__ == "__main__":
    unittest.main()
