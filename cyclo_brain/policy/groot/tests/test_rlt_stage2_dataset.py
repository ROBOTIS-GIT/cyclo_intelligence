#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.rlt_stage2_dataset import (  # noqa: E402
    CAMERA_KEYS,
    RLTStage2DatasetConfig,
    RLTStage2FeatureReplay,
    RLTStage2LeRobotV21Source,
    RLTStage2LeRobotV30Source,
    materialize_rlt_stage2_replay,
    rlt_stage2_window_starts,
)
from cyclo_brain.algorithm.rl.rlt import RLTStage2Spec  # noqa: E402


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


def _spec() -> RLTStage2Spec:
    return RLTStage2Spec(
        reference_contract_fingerprint="a" * 64,
        rl_token_artifact_fingerprint="b" * 64,
        rl_token_dim=8,
        proprio_dim=19,
        reference_horizon=16,
        chunk_length=10,
        action_dim=19,
        action_hz=15.0,
        action_normalization_id="test-normalized/v1",
        action_codec_id="test-recorder-order/v1",
        model_domain="normalized",
        schema_version=1,
    )


class _FakeExtractor:
    def __init__(self) -> None:
        self.extracted_samples = 0

    def extract(self, observation):
        left = torch.from_numpy(observation["state"]["arm_left"][:, -1])
        right = torch.from_numpy(observation["state"]["arm_right"][:, -1])
        odometry = torch.from_numpy(observation["state"]["odometry"][:, -1])
        proprio = torch.cat((left, right, odometry), dim=-1).float()
        batch = proprio.shape[0]
        z_rl = proprio[:, :8].clone()
        reference = proprio[:, None, :].expand(batch, 16, 19).clone()
        self.extracted_samples += batch
        return {
            "z_rl": z_rl,
            "proprio": proprio,
            "reference_actions": reference,
        }

    def normalize_actions(self, action_groups, state_groups):
        del state_groups
        return np.concatenate(
            [action_groups[key] for key in ("arm_left", "arm_right", "odometry")],
            axis=-1,
        ).astype(np.float32)


class RLTStage2DatasetTests(unittest.TestCase):
    def _make_dataset(
        self,
        root: Path,
        *,
        fps: float = 15.0,
        include_outcomes: bool = True,
        episodes: tuple[tuple[int, int, bool], ...] = (
            (0, 13, True),
            (1, 12, False),
        ),
    ) -> None:
        meta = root / "meta"
        meta.mkdir(parents=True)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [22],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [22],
                "names": STATE_NAMES,
            },
        }
        if include_outcomes:
            features["episode_success"] = {"dtype": "bool", "shape": [1]}
        for key in CAMERA_KEYS:
            features[f"observation.images.rgb.{key}"] = {
                "dtype": "video",
                "shape": [4, 5, 3],
            }
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v2.1",
                    "fps": fps,
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
            "".join(
                json.dumps(
                    {
                        "episode_index": index,
                        "length": length,
                        "tasks": ["grasp"],
                        "episode_success": successful,
                    }
                )
                + "\n"
                for index, length, successful in episodes
            ),
            encoding="utf-8",
        )
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "grasp the jelly bag"}) + "\n",
            encoding="utf-8",
        )
        for index, _length, _successful in episodes:
            parquet = root / f"data/chunk-000/episode_{index:06d}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_bytes(b"test")
            for camera in CAMERA_KEYS:
                video = (
                    root
                    / "videos/chunk-000"
                    / f"observation.images.rgb.{camera}"
                    / f"episode_{index:06d}.mp4"
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"test")

    def _make_v30_dataset(self, root: Path, *, include_outcomes: bool = True) -> None:
        meta = root / "meta"
        (meta / "episodes/chunk-000").mkdir(parents=True)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [22],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [22],
                "names": STATE_NAMES,
            },
        }
        if include_outcomes:
            features["episode_success"] = {"dtype": "bool", "shape": [1]}
        for key in CAMERA_KEYS:
            features[f"observation.images.rgb.{key}"] = {
                "dtype": "video",
                "shape": [3, 4, 5],
            }
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": 15,
                    "total_episodes": 2,
                    "total_frames": 25,
                    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    "features": features,
                }
            ),
            encoding="utf-8",
        )
        for relative in (
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
        for key in CAMERA_KEYS:
            path = root / f"videos/observation.images.rgb.{key}/chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")

    @staticmethod
    def _v30_rows(path: Path, required, optional):
        del required, optional
        if path.name == "tasks.parquet":
            return [{"task_index": 0, "task": "grasp the jelly bag"}]
        rows = []
        cursor = 0
        for index, length, successful in ((0, 13, True), (1, 12, False)):
            row = {
                "episode_index": index,
                "length": length,
                "tasks": ["grasp"],
                "stats/episode_success/mean": [float(successful)],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
            }
            for key in CAMERA_KEYS:
                prefix = f"videos/observation.images.rgb.{key}"
                row[f"{prefix}/chunk_index"] = 0
                row[f"{prefix}/file_index"] = 0
                row[f"{prefix}/from_timestamp"] = cursor / 15
                row[f"{prefix}/to_timestamp"] = (cursor + length) / 15
            rows.append(row)
            cursor += length
        return rows

    @staticmethod
    def _v30_slice(_path, columns, start, stop, expected_rows):
        if expected_rows != 25:
            raise AssertionError(expected_rows)
        episode_index = [0] * 13 + [1] * 12
        frame_index = [*range(13), *range(12)]
        state = np.arange(25 * 22, dtype=np.float32).reshape(25, 22)
        values = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "observation.state": state,
            "action": state + 10000,
            "task_index": [0] * 25,
            "episode_success": [True] * 13 + [False] * 12,
        }
        return {name: values[name][start:stop] for name in columns}

    @staticmethod
    def _v30_video(_path, start_frame, length, fps):
        if fps != 15.0:
            raise AssertionError(fps)
        return iter(
            np.full((4, 5, 3), start_frame + frame, dtype=np.uint8)
            for frame in range(length)
        )

    @staticmethod
    def _episode_from_path(path: Path) -> int:
        return int(path.stem.removeprefix("episode_"))

    def _parquet_reader(self, path: Path):
        episode = self._episode_from_path(path)
        length, successful = ((13, True), (12, False))[episode]
        base = episode * 1000
        state = np.arange(length * 22, dtype=np.float32).reshape(length, 22) + base
        action = state + 10000
        return {
            "observation.state": state,
            "action": action,
            "task_index": [0] * length,
            "episode_success": [successful] * length,
        }

    def _video_reader(self, path: Path):
        episode = self._episode_from_path(path)
        length = (13, 12)[episode]
        return iter(
            np.full((4, 5, 3), frame, dtype=np.uint8)
            for frame in range(length)
        )

    def test_window_starts_include_unaligned_terminal_chunk(self) -> None:
        self.assertEqual(rlt_stage2_window_starts(9), ())
        self.assertEqual(rlt_stage2_window_starts(10), (0,))
        self.assertEqual(rlt_stage2_window_starts(11), (0, 1))
        self.assertEqual(rlt_stage2_window_starts(12), (0, 2))
        self.assertEqual(rlt_stage2_window_starts(13), (0, 2, 3))

    def test_materializes_discounted_terminal_rewards_and_exact_10x19_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_dataset(root)
            source = RLTStage2LeRobotV21Source(
                root,
                parquet_reader=self._parquet_reader,
                video_reader=self._video_reader,
            )
            extractor = _FakeExtractor()
            replay = materialize_rlt_stage2_replay(
                (source,),
                extractor=extractor,
                spec=_spec(),
                output_root=Path(temporary) / "replay",
                config=RLTStage2DatasetConfig(discount=0.99),
                feature_batch_size=2,
            )

            self.assertEqual(len(replay), 5)
            self.assertEqual(extractor.extracted_samples, 8)
            self.assertEqual(replay.metadata["usable_episode_count"], 2)
            self.assertEqual(replay.metadata["success_episode_count"], 1)
            self.assertEqual(replay.metadata["failure_episode_count"], 1)
            self.assertEqual(
                replay.metadata["reference_extraction"],
                {"seed": 0, "feature_batch_size": 2},
            )
            self.assertEqual(len(replay.metadata["dataset_snapshots"]), 1)
            self.assertEqual(
                len(replay.metadata["dataset_snapshot_fingerprint"]),
                64,
            )
            self.assertEqual(tuple(replay.tensors["executed_actions"].shape), (5, 10, 19))
            self.assertEqual(tuple(replay.tensors["z_rl"].shape), (5, 8))
            torch.testing.assert_close(
                replay.tensors["reward"].flatten(),
                torch.tensor([0.0, 0.0, 0.99**9, 0.0, 0.0]),
            )
            torch.testing.assert_close(
                replay.tensors["bootstrap_discount"].flatten(),
                torch.tensor([0.99**10, 0.99**10, 0.0, 0.99**10, 0.0]),
            )
            self.assertEqual(
                replay.metadata["row_provenance"][2]["start_frame_index"], 3
            )
            self.assertTrue(replay.metadata["row_provenance"][2]["terminal"])
            # Recorder head/lift values are excluded; odometry occupies the
            # final three positions in the normalized 19-D action.
            first_raw = self._parquet_reader(
                root / "data/chunk-000/episode_000000.parquet"
            )["action"][0]
            expected = np.concatenate((first_raw[:16], first_raw[19:22]))
            np.testing.assert_array_equal(
                replay.tensors["executed_actions"][0, 0].numpy(), expected
            )

            restored = RLTStage2FeatureReplay.load(
                Path(temporary) / "replay", expected_spec=_spec()
            )
            batch = restored.batch([4, 0], device="cpu")
            batch.validate(_spec())
            self.assertEqual(tuple(batch.reference_actions.shape), (2, 10, 19))

    def test_dataset_snapshot_changes_when_consumed_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_dataset(root)
            source = RLTStage2LeRobotV21Source(
                root,
                parquet_reader=self._parquet_reader,
                video_reader=self._video_reader,
            )
            first = source.content_snapshot()
            video = (
                root
                / "videos/chunk-000"
                / "observation.images.rgb.cam_left_head"
                / "episode_000000.mp4"
            )
            video.write_bytes(b"changed")
            second = source.content_snapshot()

            self.assertNotEqual(
                first["content_fingerprint"],
                second["content_fingerprint"],
            )

    def test_outcome_gate_counts_only_episodes_with_complete_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            episodes = ((0, 9, True), (1, 12, False))
            self._make_dataset(root, episodes=episodes)
            episode_contract = {
                index: (length, successful)
                for index, length, successful in episodes
            }

            def parquet_reader(path: Path):
                episode = self._episode_from_path(path)
                length, successful = episode_contract[episode]
                state = np.arange(
                    length * 22, dtype=np.float32
                ).reshape(length, 22)
                return {
                    "observation.state": state,
                    "action": state + 10000,
                    "task_index": [0] * length,
                    "episode_success": [successful] * length,
                }

            def video_reader(_path: Path):
                self.fail("outcome validation must run before video decoding")

            source = RLTStage2LeRobotV21Source(
                root,
                parquet_reader=parquet_reader,
                video_reader=video_reader,
            )
            with self.assertRaisesRegex(
                ValueError,
                "both success and failure episodes with complete 10-step chunks",
            ):
                materialize_rlt_stage2_replay(
                    (source,),
                    extractor=_FakeExtractor(),
                    spec=_spec(),
                    output_root=Path(temporary) / "replay",
                )

    def test_rejects_wrong_fps_and_missing_outcomes_before_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "wrong-fps"
            self._make_dataset(root, fps=10.0)
            with self.assertRaisesRegex(ValueError, "15 Hz"):
                RLTStage2LeRobotV21Source(root)

            missing = Path(temporary) / "missing-outcome"
            self._make_dataset(missing, include_outcomes=False)
            with self.assertRaisesRegex(ValueError, "episode_success"):
                RLTStage2LeRobotV21Source(missing)

    def test_native_v30_materialization_matches_chunk_reward_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_v30_dataset(root)
            source = RLTStage2LeRobotV30Source(
                root,
                parquet_rows_reader=self._v30_rows,
                parquet_slice_reader=self._v30_slice,
                video_segment_reader=self._v30_video,
            )
            replay = materialize_rlt_stage2_replay(
                (source,),
                extractor=_FakeExtractor(),
                spec=_spec(),
                output_root=Path(temporary) / "replay",
                config=RLTStage2DatasetConfig(discount=0.99),
                feature_batch_size=2,
            )

            self.assertEqual(len(replay), 5)
            self.assertEqual(replay.metadata["success_episode_count"], 1)
            self.assertEqual(replay.metadata["failure_episode_count"], 1)
            self.assertEqual(
                tuple(replay.tensors["executed_actions"].shape),
                (5, 10, 19),
            )
            torch.testing.assert_close(
                replay.tensors["reward"].flatten(),
                torch.tensor([0.0, 0.0, 0.99**9, 0.0, 0.0]),
            )
            snapshot_files = {
                row["relative_path"]
                for row in replay.metadata["dataset_snapshots"][0]["files"]
            }
            self.assertIn("meta/tasks.parquet", snapshot_files)
            self.assertIn("data/chunk-000/file-000.parquet", snapshot_files)
            self.assertEqual(
                sum(path.endswith("file-000.mp4") for path in snapshot_files),
                3,
            )

    def test_native_v30_rejects_missing_outcome_and_bad_video_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing"
            self._make_v30_dataset(root, include_outcomes=False)
            with self.assertRaisesRegex(ValueError, "episode_success"):
                RLTStage2LeRobotV30Source(
                    root,
                    parquet_rows_reader=self._v30_rows,
                    parquet_slice_reader=self._v30_slice,
                    video_segment_reader=self._v30_video,
                )

            malformed = Path(temporary) / "malformed"
            self._make_v30_dataset(malformed)

            def bad_rows(path, required, optional):
                rows = self._v30_rows(path, required, optional)
                if path.name != "tasks.parquet":
                    rows[0][
                        "videos/observation.images.rgb.cam_left_head/to_timestamp"
                    ] = 12 / 15
                return rows

            with self.assertRaisesRegex(ValueError, "episode metadata"):
                RLTStage2LeRobotV30Source(
                    malformed,
                    parquet_rows_reader=bad_rows,
                    parquet_slice_reader=self._v30_slice,
                    video_segment_reader=self._v30_video,
                )


if __name__ == "__main__":
    unittest.main()
