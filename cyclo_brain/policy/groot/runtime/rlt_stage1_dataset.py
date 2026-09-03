#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only LeRobot v2.1 observations for GR00T RLT Stage 1.

Stage 1 only reconstructs frozen GR00T backbone tokens.  It therefore needs
the three cameras, robot state and language, but deliberately never reads
actions, rewards or success labels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


LANGUAGE_KEY = "annotation.human.task_description"
CAMERA_KEYS = ("cam_left_head", "cam_left_wrist", "cam_right_wrist")
STATE_GROUP_NAMES = {
    "arm_left": (
        "arm_l_joint1",
        "arm_l_joint2",
        "arm_l_joint3",
        "arm_l_joint4",
        "arm_l_joint5",
        "arm_l_joint6",
        "arm_l_joint7",
        "gripper_l_joint1",
    ),
    "arm_right": (
        "arm_r_joint1",
        "arm_r_joint2",
        "arm_r_joint3",
        "arm_r_joint4",
        "arm_r_joint5",
        "arm_r_joint6",
        "arm_r_joint7",
        "gripper_r_joint1",
    ),
    "odometry": ("linear_x", "linear_y", "angular_z"),
}


class RLTStage1DatasetError(ValueError):
    """Raised when a dataset cannot satisfy the frozen-GR00T contract."""


@dataclass(frozen=True)
class _Episode:
    index: int
    length: int
    tasks: tuple[str, ...]


ParquetReader = Callable[[Path], Mapping[str, Sequence[Any]]]
VideoReader = Callable[[Path], Iterator[np.ndarray]]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RLTStage1DatasetError(f"Cannot read dataset metadata: {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RLTStage1DatasetError(f"Cannot read dataset metadata: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RLTStage1DatasetError(f"Dataset metadata is empty or invalid: {path}")
    return rows


def _default_parquet_reader(path: Path) -> Mapping[str, Sequence[Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["observation.state", "task_index"],
    )
    return {name: table[name].to_pylist() for name in table.column_names}


def _default_video_reader(path: Path) -> Iterator[np.ndarray]:
    import av

    container = av.open(str(path), mode="r")
    try:
        streams = tuple(container.streams.video)
        if len(streams) != 1:
            raise RLTStage1DatasetError(
                f"RLT Stage 1 expects one video stream in {path}"
            )
        for frame in container.decode(streams[0]):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


class RLTStage1LeRobotV21Source:
    """Stream deterministic GR00T observations from one LeRobot v2.1 root."""

    def __init__(
        self,
        root: str | Path,
        *,
        parquet_reader: ParquetReader | None = None,
        video_reader: VideoReader | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        if self.root.is_symlink() or not self.root.is_dir():
            raise RLTStage1DatasetError(
                f"RLT Stage 1 dataset must be a real directory: {self.root}"
            )
        self._parquet_reader = parquet_reader or _default_parquet_reader
        self._video_reader = video_reader or _default_video_reader

        info = _read_json(self.root / "meta/info.json")
        if not isinstance(info, dict) or info.get("codebase_version") != "v2.1":
            raise RLTStage1DatasetError(
                "GR00T RL Token Training requires a LeRobot v2.1 dataset"
            )
        self._data_pattern = str(info.get("data_path", ""))
        self._video_pattern = str(info.get("video_path", ""))
        self._chunk_size = int(info.get("chunks_size", 0))
        if not self._data_pattern or not self._video_pattern or self._chunk_size < 1:
            raise RLTStage1DatasetError("LeRobot v2.1 path metadata is incomplete")

        features = info.get("features")
        if not isinstance(features, dict):
            raise RLTStage1DatasetError("LeRobot v2.1 features metadata is missing")
        state_feature = features.get("observation.state")
        state_names = state_feature.get("names") if isinstance(state_feature, dict) else None
        if not isinstance(state_names, list) or len(set(state_names)) != len(state_names):
            raise RLTStage1DatasetError("LeRobot observation.state names are invalid")
        try:
            self._state_indices = {
                group: tuple(state_names.index(name) for name in names)
                for group, names in STATE_GROUP_NAMES.items()
            }
        except ValueError as error:
            raise RLTStage1DatasetError(
                "LeRobot state does not contain the SG2 arm/gripper/odometry fields"
            ) from error

        self._camera_features: dict[str, str] = {}
        for output_key in CAMERA_KEYS:
            matches = [
                key
                for key, value in features.items()
                if key.endswith(f".{output_key}")
                and isinstance(value, dict)
                and value.get("dtype") == "video"
            ]
            if len(matches) != 1:
                raise RLTStage1DatasetError(
                    f"LeRobot dataset must contain exactly one {output_key} video"
                )
            self._camera_features[output_key] = matches[0]

        task_rows = _read_jsonl(self.root / "meta/tasks.jsonl")
        try:
            self._tasks = {
                int(row["task_index"]): str(row["task"]) for row in task_rows
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RLTStage1DatasetError("LeRobot task metadata is invalid") from error

        episode_rows = _read_jsonl(self.root / "meta/episodes.jsonl")
        try:
            episodes = tuple(
                _Episode(
                    index=int(row["episode_index"]),
                    length=int(row["length"]),
                    tasks=tuple(str(value) for value in row.get("tasks", ())),
                )
                for row in episode_rows
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RLTStage1DatasetError("LeRobot episode metadata is invalid") from error
        self._episodes = tuple(sorted(episodes, key=lambda item: item.index))
        if any(episode.length < 1 for episode in self._episodes):
            raise RLTStage1DatasetError("LeRobot episodes must contain at least one frame")
        self.frame_count = sum(episode.length for episode in self._episodes)

    def __len__(self) -> int:
        return self.frame_count

    def _safe_file(self, relative: str) -> Path:
        candidate = self.root / relative
        if candidate.is_symlink():
            raise RLTStage1DatasetError(f"Dataset file must not be a symlink: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise RLTStage1DatasetError(
                f"Dataset file is missing or escapes its root: {candidate}"
            ) from error
        if not resolved.is_file():
            raise RLTStage1DatasetError(f"Dataset path is not a file: {resolved}")
        return resolved

    def _episode_paths(self, episode: _Episode) -> tuple[Path, dict[str, Path]]:
        chunk = episode.index // self._chunk_size
        values = {"episode_chunk": chunk, "episode_index": episode.index}
        try:
            parquet = self._safe_file(self._data_pattern.format(**values))
            videos = {
                output_key: self._safe_file(
                    self._video_pattern.format(
                        **values,
                        video_key=feature_key,
                    )
                )
                for output_key, feature_key in self._camera_features.items()
            }
        except (KeyError, ValueError) as error:
            raise RLTStage1DatasetError("LeRobot path template is invalid") from error
        return parquet, videos

    def iter_batches(self, batch_size: int) -> Iterator[dict[str, dict[str, Any]]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("RLT Stage 1 batch_size must be positive")
        pending: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]] = []
        for episode in self._episodes:
            parquet_path, video_paths = self._episode_paths(episode)
            columns = self._parquet_reader(parquet_path)
            if set(columns) != {"observation.state", "task_index"}:
                raise RLTStage1DatasetError("LeRobot parquet columns disagree")
            state = np.asarray(columns["observation.state"], dtype=np.float32)
            task_indices = tuple(int(value) for value in columns["task_index"])
            if state.ndim != 2 or state.shape[0] != episode.length:
                raise RLTStage1DatasetError("LeRobot state row count disagrees with episode")
            if len(task_indices) != episode.length or not np.isfinite(state).all():
                raise RLTStage1DatasetError("LeRobot task/state rows are invalid")

            with ExitStack() as stack:
                video_iters: dict[str, Iterator[np.ndarray]] = {}
                for key, path in video_paths.items():
                    iterator = iter(self._video_reader(path))
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        stack.callback(close)
                    video_iters[key] = iterator

                for frame_index in range(episode.length):
                    try:
                        images = {
                            key: np.asarray(next(iterator), dtype=np.uint8)
                            for key, iterator in video_iters.items()
                        }
                    except StopIteration as error:
                        raise RLTStage1DatasetError(
                            "LeRobot video is shorter than its episode"
                        ) from error
                    if any(image.ndim != 3 or image.shape[-1] != 3 for image in images.values()):
                        raise RLTStage1DatasetError("LeRobot videos must contain RGB frames")
                    try:
                        language = self._tasks[task_indices[frame_index]]
                    except KeyError as error:
                        raise RLTStage1DatasetError("LeRobot task_index is unknown") from error
                    states = {
                        group: state[frame_index, list(indices)]
                        for group, indices in self._state_indices.items()
                    }
                    pending.append((images, states, language))
                    if len(pending) == batch_size:
                        yield self._collate(pending)
                        pending = []
        if pending:
            yield self._collate(pending)

    @staticmethod
    def _collate(
        samples: Sequence[tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]],
    ) -> dict[str, dict[str, Any]]:
        return {
            "video": {
                key: np.stack([sample[0][key] for sample in samples])[:, None, ...]
                for key in CAMERA_KEYS
            },
            "state": {
                key: np.stack([sample[1][key] for sample in samples])
                .astype(np.float32, copy=False)[:, None, :]
                for key in STATE_GROUP_NAMES
            },
            "language": {
                LANGUAGE_KEY: [[sample[2]] for sample in samples],
            },
        }


__all__ = [
    "CAMERA_KEYS",
    "LANGUAGE_KEY",
    "RLTStage1DatasetError",
    "RLTStage1LeRobotV21Source",
]
