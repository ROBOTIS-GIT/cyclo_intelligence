#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.rlt_stage1_dataset import (  # noqa: E402
    CAMERA_KEYS,
    LANGUAGE_KEY,
    RLTStage1LeRobotV21Source,
)


STATE_NAMES = [
    *(f"arm_l_joint{i}" for i in range(1, 8)),
    "gripper_l_joint1",
    *(f"arm_r_joint{i}" for i in range(1, 8)),
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
    "linear_x",
    "linear_y",
    "angular_z",
]


class RLTStage1DatasetTests(unittest.TestCase):
    def _make_dataset(self, root: Path, *, version: str = "v2.1") -> None:
        meta = root / "meta"
        meta.mkdir(parents=True)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [22],
                "names": STATE_NAMES,
            }
        }
        for key in CAMERA_KEYS:
            features[f"observation.images.rgb.{key}"] = {
                "dtype": "video",
                "shape": [4, 5, 3],
            }
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": version,
                    "chunks_size": 1000,
                    "data_path": (
                        "data/chunk-{episode_chunk:03d}/"
                        "episode_{episode_index:06d}.parquet"
                    ),
                    "video_path": (
                        "videos/chunk-{episode_chunk:03d}/{video_key}/"
                        "episode_{episode_index:06d}.mp4"
                    ),
                    "features": features,
                }
            ),
            encoding="utf-8",
        )
        (meta / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 2, "tasks": ["grasp"]})
            + "\n",
            encoding="utf-8",
        )
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "grasp the jelly bag"}) + "\n",
            encoding="utf-8",
        )
        parquet = root / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"test")
        for key in CAMERA_KEYS:
            video = (
                root
                / "videos/chunk-000"
                / f"observation.images.rgb.{key}"
                / "episode_000000.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"test")

    def test_streams_only_stage1_observation_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_dataset(root)
            state = np.arange(44, dtype=np.float32).reshape(2, 22)

            source = RLTStage1LeRobotV21Source(
                root,
                parquet_reader=lambda _path: {
                    "observation.state": state,
                    "task_index": [0, 0],
                },
                video_reader=lambda _path: iter(
                    [
                        np.zeros((4, 5, 3), dtype=np.uint8),
                        np.ones((4, 5, 3), dtype=np.uint8),
                    ]
                ),
            )

            batches = list(source.iter_batches(2))

            self.assertEqual(len(source), 2)
            self.assertEqual(len(batches), 1)
            observation = batches[0]
            self.assertEqual(set(observation), {"video", "state", "language"})
            self.assertEqual(
                tuple(observation["video"]["cam_left_head"].shape),
                (2, 1, 4, 5, 3),
            )
            self.assertEqual(tuple(observation["state"]["arm_left"].shape), (2, 1, 8))
            self.assertEqual(tuple(observation["state"]["odometry"].shape), (2, 1, 3))
            self.assertEqual(
                observation["language"][LANGUAGE_KEY],
                [["grasp the jelly bag"], ["grasp the jelly bag"]],
            )

    def test_rejects_lerobot_v30_before_video_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_dataset(root, version="v3.0")

            with self.assertRaisesRegex(ValueError, "v2.1"):
                RLTStage1LeRobotV21Source(root)


if __name__ == "__main__":
    unittest.main()
