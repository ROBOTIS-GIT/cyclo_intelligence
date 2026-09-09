#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Materialize immutable PI-RLT Stage-2 transitions from LeRobot v2.1/v3.0.

The expensive GR00T forward pass is an extraction step, not part of the RL
optimizer.  This module streams the three recorded cameras, asks one frozen
GR00T instance for its token representation and 16-step reference action, and
immediately compresses those values with the frozen Stage-1 RL-token encoder.
Only detached, normalized tensors are retained for Action-MLP/twin-Q training.

The active showroom contract is intentionally narrow: 15 Hz, three cameras,
left arm/gripper (8), right arm/gripper (8), and odometry (3).  Head and lift
columns may be present in the recorder dataset but never enter the 19-D RLT
action/state vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from cyclo_brain.algorithm.common import (
    atomic_json_save,
    atomic_torch_save,
    canonical_json_sha256,
    file_sha256,
)
from cyclo_brain.algorithm.rl.rlt import (
    RLTStage2Batch,
    RLTStage2Spec,
    stage2_spec_fingerprint,
)
from cyclo_brain.model.common import (
    GROOT_REFERENCE_ACTION_HORIZON,
    RLT_ACTION_DIM,
    RLT_ACTION_HORIZON,
)

from .rlt_adapter import _rec_to_dtype
from .rlt_lerobot_v30 import (
    read_dataset_json,
    read_dataset_jsonl,
    ParquetRowsReader,
    ParquetSliceReader,
    RLTLeRobotV30Layout,
    VideoSegmentReader,
    lerobot_codebase_version,
)
from .rlt_stage1_dataset import CAMERA_KEYS, LANGUAGE_KEY, STATE_GROUP_NAMES


ACTION_GROUP_NAMES = STATE_GROUP_NAMES
RLT_CHUNK_LENGTH = RLT_ACTION_HORIZON
RLT_REFERENCE_HORIZON = GROOT_REFERENCE_ACTION_HORIZON
_REPLAY_FORMAT = "cyclo.groot.rlt.stage2_feature_replay/v1"
_REPLAY_MANIFEST_FORMAT = "cyclo.groot.rlt.stage2_feature_replay_manifest/v1"
_DATASET_SNAPSHOT_FORMAT = "cyclo.groot.rlt.dataset_snapshot/v1"


class RLTStage2DatasetError(ValueError):
    """Raised when recorded data cannot satisfy the active RLT contract."""


@dataclass(frozen=True)
class RLTStage2DatasetConfig:
    """Fixed window/reward contract used by the current PI-RLT integration."""

    chunk_length: int = RLT_CHUNK_LENGTH
    stride: int = 2
    discount: float = 0.99
    expected_fps: float = 15.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.chunk_length, "chunk_length"),
            (self.stride, "stride"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"RLT Stage 2 {name} must be a positive integer")
        if self.chunk_length != RLT_CHUNK_LENGTH:
            raise ValueError(
                f"RLT Stage 2 requires {RLT_CHUNK_LENGTH}-action chunks"
            )
        for value, name in (
            (self.discount, "discount"),
            (self.expected_fps, "expected_fps"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"RLT Stage 2 {name} must be finite and positive")
        if float(self.discount) > 1.0:
            raise ValueError("RLT Stage 2 discount must be in (0, 1]")
        object.__setattr__(self, "discount", float(self.discount))
        object.__setattr__(self, "expected_fps", float(self.expected_fps))


@dataclass(frozen=True)
class _EpisodeInfo:
    index: int
    length: int
    tasks: tuple[str, ...]
    metadata_success: bool | None


@dataclass(frozen=True)
class RLTStage2RawEpisode:
    """One validated episode without decoded video frames."""

    dataset_root: Path
    episode_index: int
    length: int
    successful: bool
    task_indices: tuple[int, ...]
    state_groups: Mapping[str, np.ndarray]
    action_groups: Mapping[str, np.ndarray]


ParquetReader = Callable[[Path], Mapping[str, Sequence[Any]]]
VideoReader = Callable[[Path], Iterator[np.ndarray]]


def _read_json(path: Path) -> Any:
    return read_dataset_json(path, RLTStage2DatasetError)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_dataset_jsonl(path, RLTStage2DatasetError)


def _default_parquet_reader(path: Path) -> Mapping[str, Sequence[Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["observation.state", "action", "task_index", "episode_success"],
    )
    return {name: table[name].to_pylist() for name in table.column_names}


def _default_video_reader(path: Path) -> Iterator[np.ndarray]:
    import av

    container = av.open(str(path), mode="r")
    try:
        streams = tuple(container.streams.video)
        if len(streams) != 1:
            raise RLTStage2DatasetError(
                f"RLT Stage 2 expects one video stream in {path}"
            )
        for frame in container.decode(streams[0]):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def rlt_stage2_window_starts(
    episode_length: int,
    *,
    chunk_length: int = RLT_CHUNK_LENGTH,
    stride: int = 2,
) -> tuple[int, ...]:
    """Return stride-aligned windows plus the exact terminal window.

    Anchoring the final window ensures the sparse terminal reward is present
    even when ``episode_length - chunk_length`` is not stride aligned.
    """

    for value, name in (
        (episode_length, "episode_length"),
        (chunk_length, "chunk_length"),
        (stride, "stride"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"RLT Stage 2 {name} must be a non-negative integer")
    if chunk_length < 1 or stride < 1:
        raise ValueError("RLT Stage 2 chunk_length and stride must be positive")
    if episode_length < chunk_length:
        return ()
    terminal = episode_length - chunk_length
    starts = list(range(0, terminal + 1, stride))
    if starts[-1] != terminal:
        starts.append(terminal)
    return tuple(starts)


class RLTStage2LeRobotV21Source:
    """Read one immutable LeRobot v2.1 replay root episode by episode."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_fps: float = 15.0,
        parquet_reader: ParquetReader | None = None,
        video_reader: VideoReader | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        if self.root.is_symlink() or not self.root.is_dir():
            raise RLTStage2DatasetError(
                f"RLT Stage 2 dataset must be a real directory: {self.root}"
            )
        self._parquet_reader = parquet_reader or _default_parquet_reader
        self._video_reader = video_reader or _default_video_reader

        info = _read_json(self.root / "meta/info.json")
        if not isinstance(info, dict) or info.get("codebase_version") != "v2.1":
            raise RLTStage2DatasetError("GR00T RLT Stage 2 requires LeRobot v2.1")
        try:
            self.fps = float(info["fps"])
        except (KeyError, TypeError, ValueError) as error:
            raise RLTStage2DatasetError("LeRobot v2.1 fps metadata is invalid") from error
        if not math.isclose(self.fps, float(expected_fps), rel_tol=0.0, abs_tol=1e-6):
            raise RLTStage2DatasetError(
                f"RLT Stage 2 requires {float(expected_fps):g} Hz data; got {self.fps:g}"
            )
        self._data_pattern = str(info.get("data_path", ""))
        self._video_pattern = str(info.get("video_path", ""))
        try:
            self._chunk_size = int(info.get("chunks_size", 0))
        except (TypeError, ValueError) as error:
            raise RLTStage2DatasetError("LeRobot v2.1 chunk metadata is invalid") from error
        if not self._data_pattern or not self._video_pattern or self._chunk_size < 1:
            raise RLTStage2DatasetError("LeRobot v2.1 path metadata is incomplete")

        features = info.get("features")
        if not isinstance(features, Mapping):
            raise RLTStage2DatasetError("LeRobot v2.1 features metadata is missing")
        success_feature = features.get("episode_success")
        if not isinstance(success_feature, Mapping) or success_feature.get("dtype") != "bool":
            raise RLTStage2DatasetError(
                "RLT Stage 2 requires a boolean episode_success feature"
            )
        self._state_indices = self._group_indices(
            features.get("observation.state"), "observation.state"
        )
        self._action_indices = self._group_indices(features.get("action"), "action")

        self._camera_features: dict[str, str] = {}
        for camera in CAMERA_KEYS:
            matches = [
                key
                for key, value in features.items()
                if str(key).endswith(f".{camera}")
                and isinstance(value, Mapping)
                and value.get("dtype") == "video"
            ]
            if len(matches) != 1:
                raise RLTStage2DatasetError(
                    f"LeRobot dataset must contain exactly one {camera} video"
                )
            self._camera_features[camera] = str(matches[0])

        try:
            self._tasks = {
                int(row["task_index"]): str(row["task"])
                for row in _read_jsonl(self.root / "meta/tasks.jsonl")
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RLTStage2DatasetError("LeRobot task metadata is invalid") from error

        parsed_episodes: list[_EpisodeInfo] = []
        try:
            for row in _read_jsonl(self.root / "meta/episodes.jsonl"):
                raw_success = row.get("episode_success")
                if raw_success is not None and not isinstance(raw_success, bool):
                    raise TypeError("episode_success")
                parsed_episodes.append(
                    _EpisodeInfo(
                        index=int(row["episode_index"]),
                        length=int(row["length"]),
                        tasks=tuple(str(value) for value in row.get("tasks", ())),
                        metadata_success=raw_success,
                    )
                )
        except (KeyError, TypeError, ValueError) as error:
            raise RLTStage2DatasetError("LeRobot episode metadata is invalid") from error
        self._episodes = tuple(sorted(parsed_episodes, key=lambda value: value.index))
        if (
            not self._episodes
            or len({episode.index for episode in self._episodes}) != len(self._episodes)
            or any(episode.length < 1 for episode in self._episodes)
        ):
            raise RLTStage2DatasetError("LeRobot episode indices/lengths are invalid")

    @staticmethod
    def _group_indices(feature: Any, label: str) -> dict[str, tuple[int, ...]]:
        names = feature.get("names") if isinstance(feature, Mapping) else None
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise RLTStage2DatasetError(f"LeRobot {label} names are invalid")
        try:
            return {
                group: tuple(names.index(name) for name in group_names)
                for group, group_names in ACTION_GROUP_NAMES.items()
            }
        except ValueError as error:
            raise RLTStage2DatasetError(
                f"LeRobot {label} lacks the SG2 arm/gripper/odometry fields"
            ) from error

    def __len__(self) -> int:
        return sum(episode.length for episode in self._episodes)

    def content_snapshot(self) -> dict[str, Any]:
        """Hash every dataset file that can affect Stage-2 materialization."""

        files = {
            self._safe_file("meta/info.json"),
            self._safe_file("meta/episodes.jsonl"),
            self._safe_file("meta/tasks.jsonl"),
        }
        for episode in self._episodes:
            parquet, videos = self._paths(episode)
            files.add(parquet)
            files.update(videos.values())
        records = []
        total_size = 0
        resolved_root = self.root.resolve(strict=True)
        for path in sorted(files):
            before = path.stat()
            digest = file_sha256(path)
            after = path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"RLT Stage 2 dataset changed while hashing: {path}"
                )
            relative = path.relative_to(resolved_root).as_posix()
            records.append(
                {
                    "relative_path": relative,
                    "byte_count": before.st_size,
                    "sha256": digest,
                }
            )
            total_size += before.st_size
        core = {
            "format": _DATASET_SNAPSHOT_FORMAT,
            "file_count": len(records),
            "total_byte_count": total_size,
            "files": records,
        }
        return {
            **core,
            "content_fingerprint": _canonical_fingerprint(core),
        }

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def _safe_file(self, relative: str) -> Path:
        candidate = self.root / relative
        if candidate.is_symlink():
            raise RLTStage2DatasetError(f"Dataset file must not be a symlink: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise RLTStage2DatasetError(
                f"Dataset file is missing or escapes its root: {candidate}"
            ) from error
        if not resolved.is_file():
            raise RLTStage2DatasetError(f"Dataset path is not a file: {resolved}")
        return resolved

    def _paths(self, episode: _EpisodeInfo) -> tuple[Path, dict[str, Path]]:
        values = {
            "episode_chunk": episode.index // self._chunk_size,
            "episode_index": episode.index,
        }
        try:
            parquet = self._safe_file(self._data_pattern.format(**values))
            videos = {
                camera: self._safe_file(
                    self._video_pattern.format(
                        **values,
                        video_key=feature_key,
                    )
                )
                for camera, feature_key in self._camera_features.items()
            }
        except (KeyError, ValueError) as error:
            raise RLTStage2DatasetError("LeRobot path template is invalid") from error
        return parquet, videos

    def iter_episodes(self) -> Iterator[RLTStage2RawEpisode]:
        expected_columns = {
            "observation.state",
            "action",
            "task_index",
            "episode_success",
        }
        for metadata in self._episodes:
            parquet, _videos = self._paths(metadata)
            columns = self._parquet_reader(parquet)
            if set(columns) != expected_columns:
                raise RLTStage2DatasetError("LeRobot parquet columns disagree")
            state = np.asarray(columns["observation.state"], dtype=np.float32)
            action = np.asarray(columns["action"], dtype=np.float32)
            task_indices = tuple(int(value) for value in columns["task_index"])
            success_values = tuple(columns["episode_success"])
            if (
                state.ndim != 2
                or action.ndim != 2
                or state.shape[0] != metadata.length
                or action.shape[0] != metadata.length
                or len(task_indices) != metadata.length
                or len(success_values) != metadata.length
                or not np.isfinite(state).all()
                or not np.isfinite(action).all()
            ):
                raise RLTStage2DatasetError(
                    f"LeRobot episode {metadata.index} row tensors are invalid"
                )
            if any(type(value) not in {bool, np.bool_} for value in success_values):
                raise RLTStage2DatasetError("episode_success rows must be boolean")
            outcomes = {bool(value) for value in success_values}
            if len(outcomes) != 1:
                raise RLTStage2DatasetError("episode_success must be constant per episode")
            successful = outcomes.pop()
            if (
                metadata.metadata_success is not None
                and metadata.metadata_success != successful
            ):
                raise RLTStage2DatasetError(
                    "episode_success parquet and metadata values disagree"
                )
            unknown_tasks = sorted(set(task_indices).difference(self._tasks))
            if unknown_tasks:
                raise RLTStage2DatasetError("LeRobot task_index is unknown")
            yield RLTStage2RawEpisode(
                dataset_root=self.root,
                episode_index=metadata.index,
                length=metadata.length,
                successful=successful,
                task_indices=task_indices,
                state_groups={
                    group: state[:, indices].astype(np.float32, copy=False)
                    for group, indices in self._state_indices.items()
                },
                action_groups={
                    group: action[:, indices].astype(np.float32, copy=False)
                    for group, indices in self._action_indices.items()
                },
            )

    def iter_observation_batches(
        self,
        episode: RLTStage2RawEpisode,
        frame_indices: Sequence[int],
        batch_size: int,
    ) -> Iterator[tuple[tuple[int, ...], dict[str, dict[str, Any]]]]:
        if episode.dataset_root != self.root:
            raise ValueError("RLT episode belongs to another dataset root")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("RLT feature batch_size must be positive")
        selected = tuple(sorted(set(int(value) for value in frame_indices)))
        if any(value < 0 or value >= episode.length for value in selected):
            raise ValueError("RLT requested frame index is outside its episode")
        if not selected:
            return
        metadata = next(
            value for value in self._episodes if value.index == episode.episode_index
        )
        _parquet, video_paths = self._paths(metadata)
        selected_set = set(selected)
        pending: list[tuple[int, dict[str, np.ndarray], dict[str, np.ndarray], str]] = []
        with ExitStack() as stack:
            iterators: dict[str, Iterator[np.ndarray]] = {}
            for camera, path in video_paths.items():
                iterator = iter(self._video_reader(path))
                close = getattr(iterator, "close", None)
                if callable(close):
                    stack.callback(close)
                iterators[camera] = iterator
            for frame_index in range(episode.length):
                try:
                    images = {
                        camera: np.asarray(next(iterator), dtype=np.uint8)
                        for camera, iterator in iterators.items()
                    }
                except StopIteration as error:
                    raise RLTStage2DatasetError(
                        "LeRobot video is shorter than its episode"
                    ) from error
                if any(
                    image.ndim != 3 or image.shape[-1] != 3
                    for image in images.values()
                ):
                    raise RLTStage2DatasetError("LeRobot videos must contain RGB frames")
                if frame_index not in selected_set:
                    continue
                states = {
                    group: values[frame_index]
                    for group, values in episode.state_groups.items()
                }
                language = self._tasks[episode.task_indices[frame_index]]
                pending.append((frame_index, images, states, language))
                if len(pending) == batch_size:
                    yield self._collate(pending)
                    pending = []
            sentinel = object()
            if any(next(iterator, sentinel) is not sentinel for iterator in iterators.values()):
                raise RLTStage2DatasetError("LeRobot video is longer than its episode")
        if pending:
            yield self._collate(pending)

    @staticmethod
    def _collate(
        samples: Sequence[
            tuple[int, dict[str, np.ndarray], dict[str, np.ndarray], str]
        ],
    ) -> tuple[tuple[int, ...], dict[str, dict[str, Any]]]:
        return (
            tuple(sample[0] for sample in samples),
            {
                "video": {
                    camera: np.stack([sample[1][camera] for sample in samples])[
                        :, None, ...
                    ]
                    for camera in CAMERA_KEYS
                },
                "state": {
                    group: np.stack([sample[2][group] for sample in samples])
                    .astype(np.float32, copy=False)[:, None, :]
                    for group in STATE_GROUP_NAMES
                },
                "language": {
                    LANGUAGE_KEY: [[sample[3]] for sample in samples],
                },
            },
        )


class RLTStage2LeRobotV30Source:
    """Read one immutable LeRobot v3 replay root without rewriting it."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_fps: float = 15.0,
        parquet_rows_reader: ParquetRowsReader | None = None,
        parquet_slice_reader: ParquetSliceReader | None = None,
        video_segment_reader: VideoSegmentReader | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        self._layout = RLTLeRobotV30Layout(
            self.root,
            camera_keys=CAMERA_KEYS,
            expected_fps=expected_fps,
            require_outcomes=True,
            parquet_rows_reader=parquet_rows_reader,
            parquet_slice_reader=parquet_slice_reader,
            video_segment_reader=video_segment_reader,
        )
        self.fps = self._layout.fps
        self._state_indices = RLTStage2LeRobotV21Source._group_indices(
            self._layout.features.get("observation.state"),
            "observation.state",
        )
        self._action_indices = RLTStage2LeRobotV21Source._group_indices(
            self._layout.features.get("action"),
            "action",
        )

    def __len__(self) -> int:
        return self._layout.total_frames

    @property
    def episode_count(self) -> int:
        return len(self._layout.episodes)

    def content_snapshot(self) -> dict[str, Any]:
        """Hash every v3 metadata/data/video file consumed by RLT."""

        records = []
        total_size = 0
        resolved_root = self.root.resolve(strict=True)
        for path in self._layout.consumed_files:
            before = path.stat()
            digest = file_sha256(path)
            after = path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"RLT Stage 2 dataset changed while hashing: {path}"
                )
            records.append(
                {
                    "relative_path": path.relative_to(resolved_root).as_posix(),
                    "byte_count": before.st_size,
                    "sha256": digest,
                }
            )
            total_size += before.st_size
        core = {
            "format": _DATASET_SNAPSHOT_FORMAT,
            "file_count": len(records),
            "total_byte_count": total_size,
            "files": records,
        }
        return {**core, "content_fingerprint": _canonical_fingerprint(core)}

    def iter_episodes(self) -> Iterator[RLTStage2RawEpisode]:
        for metadata in self._layout.episodes:
            columns = self._layout.read_episode_columns(
                metadata,
                (
                    "observation.state",
                    "action",
                    "task_index",
                    "episode_success",
                ),
            )
            state = np.asarray(columns["observation.state"], dtype=np.float32)
            action = np.asarray(columns["action"], dtype=np.float32)
            task_indices = tuple(int(value) for value in columns["task_index"])
            success_values = tuple(columns["episode_success"])
            if (
                state.ndim != 2
                or action.ndim != 2
                or state.shape[0] != metadata.length
                or action.shape[0] != metadata.length
                or len(task_indices) != metadata.length
                or len(success_values) != metadata.length
                or not np.isfinite(state).all()
                or not np.isfinite(action).all()
            ):
                raise RLTStage2DatasetError(
                    f"LeRobot episode {metadata.index} row tensors are invalid"
                )
            if any(type(value) not in {bool, np.bool_} for value in success_values):
                raise RLTStage2DatasetError("episode_success rows must be boolean")
            outcomes = {bool(value) for value in success_values}
            if len(outcomes) != 1:
                raise RLTStage2DatasetError(
                    "episode_success must be constant per episode"
                )
            successful = outcomes.pop()
            if metadata.metadata_success != successful:
                raise RLTStage2DatasetError(
                    "episode_success parquet and metadata values disagree"
                )
            if set(task_indices).difference(self._layout.tasks):
                raise RLTStage2DatasetError("LeRobot task_index is unknown")
            yield RLTStage2RawEpisode(
                dataset_root=self.root,
                episode_index=metadata.index,
                length=metadata.length,
                successful=successful,
                task_indices=task_indices,
                state_groups={
                    group: state[:, indices].astype(np.float32, copy=False)
                    for group, indices in self._state_indices.items()
                },
                action_groups={
                    group: action[:, indices].astype(np.float32, copy=False)
                    for group, indices in self._action_indices.items()
                },
            )

    def iter_observation_batches(
        self,
        episode: RLTStage2RawEpisode,
        frame_indices: Sequence[int],
        batch_size: int,
    ) -> Iterator[tuple[tuple[int, ...], dict[str, dict[str, Any]]]]:
        if episode.dataset_root != self.root:
            raise ValueError("RLT episode belongs to another dataset root")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("RLT feature batch_size must be positive")
        selected = tuple(sorted(set(int(value) for value in frame_indices)))
        if any(value < 0 or value >= episode.length for value in selected):
            raise ValueError("RLT requested frame index is outside its episode")
        if not selected:
            return
        metadata = self._layout.episodes[episode.episode_index]
        selected_set = set(selected)
        pending: list[tuple[int, dict[str, np.ndarray], dict[str, np.ndarray], str]] = []
        with ExitStack() as stack:
            iterators: dict[str, Iterator[np.ndarray]] = {}
            for camera in CAMERA_KEYS:
                iterator = iter(self._layout.iter_video(metadata, camera))
                close = getattr(iterator, "close", None)
                if callable(close):
                    stack.callback(close)
                iterators[camera] = iterator
            for frame_index in range(episode.length):
                try:
                    images = {
                        camera: np.asarray(next(iterator), dtype=np.uint8)
                        for camera, iterator in iterators.items()
                    }
                except StopIteration as error:
                    raise RLTStage2DatasetError(
                        "LeRobot video is shorter than its episode"
                    ) from error
                if any(
                    image.ndim != 3 or image.shape[-1] != 3
                    for image in images.values()
                ):
                    raise RLTStage2DatasetError("LeRobot videos must contain RGB frames")
                if frame_index not in selected_set:
                    continue
                states = {
                    group: values[frame_index]
                    for group, values in episode.state_groups.items()
                }
                language = self._layout.tasks[episode.task_indices[frame_index]]
                pending.append((frame_index, images, states, language))
                if len(pending) == batch_size:
                    yield RLTStage2LeRobotV21Source._collate(pending)
                    pending = []
            sentinel = object()
            if any(
                next(iterator, sentinel) is not sentinel
                for iterator in iterators.values()
            ):
                raise RLTStage2DatasetError("LeRobot video is longer than its episode")
        if pending:
            yield RLTStage2LeRobotV21Source._collate(pending)


class RLTStage2Source(Protocol):
    root: Path
    fps: float

    @property
    def episode_count(self) -> int: ...

    def content_snapshot(self) -> dict[str, Any]: ...

    def iter_episodes(self) -> Iterator[RLTStage2RawEpisode]: ...

    def iter_observation_batches(
        self,
        episode: RLTStage2RawEpisode,
        frame_indices: Sequence[int],
        batch_size: int,
    ) -> Iterator[tuple[tuple[int, ...], dict[str, dict[str, Any]]]]: ...


def open_rlt_stage2_source(
    root: str | Path,
    *,
    expected_fps: float = 15.0,
) -> RLTStage2Source:
    """Open a v2.1 or native v3 replay source without modifying it."""

    version = lerobot_codebase_version(root)
    if version == "v2.1":
        return RLTStage2LeRobotV21Source(root, expected_fps=expected_fps)
    return RLTStage2LeRobotV30Source(root, expected_fps=expected_fps)


class RLTStage2Extractor(Protocol):
    def extract(self, observation: Mapping[str, object]) -> Mapping[str, Tensor]: ...

    def normalize_actions(
        self,
        action_groups: Mapping[str, np.ndarray],
        state_groups: Mapping[str, np.ndarray],
    ) -> np.ndarray: ...


class GR00TRLTStage2Extractor:
    """Same-forward frozen GR00T/reference/RL-token extractor for Stage 2."""

    def __init__(self, policy: Any, encoder: Any) -> None:
        self.policy = policy
        self.encoder = encoder
        self.policy.model.eval().requires_grad_(False)
        self.encoder.eval().requires_grad_(False)
        processor_eval = getattr(self.policy.processor, "eval", None)
        if callable(processor_eval):
            processor_eval()
        action_keys = tuple(self.policy.modality_configs["action"].modality_keys)
        state_keys = tuple(self.policy.modality_configs["state"].modality_keys)
        expected = tuple(ACTION_GROUP_NAMES)
        if action_keys != expected or state_keys != expected:
            raise ValueError(
                "RLT Stage 2 requires GR00T arm_left/arm_right/odometry modality order"
            )

    def _prepare(self, observation: Mapping[str, object]) -> Mapping[str, Any]:
        if getattr(self.policy, "strict", False):
            self.policy.check_observation(observation)
        processed = []
        for item in self.policy._unbatch_observation(observation):
            vla_step = self.policy._to_vla_step_data(item)
            processed.append(
                self.policy.processor([{"type": "episode_step", "content": vla_step}])
            )
        collated = self.policy.collate_fn(processed)
        collated = _rec_to_dtype(
            collated,
            dtype=getattr(self.policy.model, "dtype", torch.bfloat16),
        )
        if not isinstance(collated, Mapping) or "inputs" not in collated:
            raise RuntimeError("GR00T RLT expected collator output with inputs")
        return collated

    @torch.inference_mode()
    def extract(self, observation: Mapping[str, object]) -> dict[str, Tensor]:
        collated = self._prepare(observation)
        backbone_inputs, action_inputs = self.policy.model.prepare_input(
            collated["inputs"]
        )
        backbone_output = self.policy.model.backbone(backbone_inputs)
        tokens = backbone_output["backbone_features"].detach()
        token_valid = backbone_output["backbone_attention_mask"].to(
            dtype=torch.bool
        ).detach()
        image_token = backbone_output["image_mask"].to(dtype=torch.bool).detach()
        z_rl = self.encoder(tokens, token_valid, image_token).float().detach()
        proprio = action_inputs["state"][:, -1, :RLT_ACTION_DIM].float().detach()
        prediction = self.policy.model.action_head.get_action(
            backbone_output,
            action_inputs,
            collated.get("options"),
        )
        reference = prediction["action_pred"][
            :, :RLT_REFERENCE_HORIZON, :RLT_ACTION_DIM
        ].float().detach()
        batch_size = int(tokens.shape[0])
        expected = {
            "z_rl": (batch_size, int(self.encoder.config.embedding_dim)),
            "proprio": (batch_size, RLT_ACTION_DIM),
            "reference_actions": (
                batch_size,
                RLT_REFERENCE_HORIZON,
                RLT_ACTION_DIM,
            ),
        }
        values = {
            "z_rl": z_rl,
            "proprio": proprio,
            "reference_actions": reference,
        }
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"GR00T RLT {name} does not satisfy shape {shape}")
        return {
            name: value.to(device="cpu", dtype=torch.float32).clone()
            for name, value in values.items()
        }

    def normalize_actions(
        self,
        action_groups: Mapping[str, np.ndarray],
        state_groups: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        tag = getattr(self.policy.embodiment_tag, "value", self.policy.embodiment_tag)
        normalized = self.policy.processor.state_action_processor.apply_action(
            {key: np.asarray(action_groups[key], dtype=np.float32) for key in ACTION_GROUP_NAMES},
            str(tag),
            state={
                key: np.asarray(state_groups[key], dtype=np.float32)
                for key in STATE_GROUP_NAMES
            },
            clip_outliers=False,
        )
        result = np.concatenate(
            [np.asarray(normalized[key], dtype=np.float32) for key in ACTION_GROUP_NAMES],
            axis=-1,
        )
        expected = (next(iter(action_groups.values())).shape[0], RLT_ACTION_DIM)
        if result.shape != expected or not np.isfinite(result).all():
            raise RuntimeError(
                f"GR00T RLT normalized actions have shape {result.shape}; expected {expected}"
            )
        return result


def _validate_replay_tensor(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or not value.is_floating_point()
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"RLT Stage 2 replay {name} must be detached finite {shape}")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


class RLTStage2FeatureReplay:
    """Compact CPU replay containing no image or live GR00T graph."""

    _TENSOR_NAMES = (
        "z_rl",
        "proprio",
        "reference_actions",
        "executed_actions",
        "reward",
        "bootstrap_discount",
        "next_z_rl",
        "next_proprio",
        "next_reference_actions",
    )

    def __init__(
        self,
        spec: RLTStage2Spec,
        tensors: Mapping[str, Tensor],
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        if set(tensors) != set(self._TENSOR_NAMES):
            raise ValueError("RLT Stage 2 replay tensor fields are invalid")
        count = int(tensors["z_rl"].shape[0]) if tensors["z_rl"].ndim else 0
        if count < 1:
            raise ValueError("RLT Stage 2 replay must contain at least one transition")
        shapes = {
            "z_rl": (count, spec.rl_token_dim),
            "proprio": (count, spec.proprio_dim),
            "reference_actions": (count, spec.chunk_length, spec.action_dim),
            "executed_actions": (count, spec.chunk_length, spec.action_dim),
            "reward": (count, 1),
            "bootstrap_discount": (count, 1),
            "next_z_rl": (count, spec.rl_token_dim),
            "next_proprio": (count, spec.proprio_dim),
            "next_reference_actions": (count, spec.chunk_length, spec.action_dim),
        }
        self.spec = spec
        self.spec_fingerprint = stage2_spec_fingerprint(spec)
        self.tensors = {
            name: _validate_replay_tensor(tensors[name], name=name, shape=shapes[name])
            for name in self._TENSOR_NAMES
        }
        if bool((self.tensors["bootstrap_discount"] < 0).any()) or bool(
            (self.tensors["bootstrap_discount"] > 1).any()
        ):
            raise ValueError("RLT Stage 2 replay bootstrap discounts are invalid")
        self.metadata = dict(metadata)

    def __len__(self) -> int:
        return int(self.tensors["z_rl"].shape[0])

    @property
    def average_reward(self) -> float:
        return float(self.tensors["reward"].mean().item())

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RLTStage2Batch:
        if not indices:
            raise ValueError("RLT Stage 2 replay batch is empty")
        index = torch.as_tensor(tuple(int(value) for value in indices), dtype=torch.long)
        if bool((index < 0).any()) or bool((index >= len(self)).any()):
            raise IndexError("RLT Stage 2 replay batch index is out of range")
        target = torch.device(device)
        values = {
            name: tensor.index_select(0, index).to(device=target, dtype=dtype)
            for name, tensor in self.tensors.items()
        }
        return RLTStage2Batch(spec_fingerprint=self.spec_fingerprint, **values)

    def save(self, root: str | Path) -> Path:
        directory = Path(root).expanduser().absolute()
        if directory.is_symlink():
            raise ValueError("RLT Stage 2 replay output must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "replay.pt"
        payload = {
            "format": _REPLAY_FORMAT,
            "spec": asdict(self.spec),
            "spec_fingerprint": self.spec_fingerprint,
            "metadata": self.metadata,
            "tensors": self.tensors,
        }
        atomic_torch_save(path, payload)
        digest = file_sha256(path)
        manifest = {
            "format": _REPLAY_MANIFEST_FORMAT,
            "file": path.name,
            "byte_count": path.stat().st_size,
            "sha256": digest,
            "spec_fingerprint": self.spec_fingerprint,
            "transition_count": len(self),
            "average_reward": self.average_reward,
            "metadata": self.metadata,
        }
        atomic_json_save(
            directory / "manifest.json",
            manifest,
            ensure_ascii=True,
            compact=True,
            newline=False,
        )
        return path

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_spec: RLTStage2Spec | None = None,
    ) -> "RLTStage2FeatureReplay":
        directory = Path(root).expanduser().absolute()
        manifest = _read_json(directory / "manifest.json")
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("format") != _REPLAY_MANIFEST_FORMAT
            or manifest.get("file") != "replay.pt"
        ):
            raise ValueError("RLT Stage 2 replay manifest is invalid")
        path = directory / "replay.pt"
        if path.is_symlink() or not path.is_file():
            raise ValueError("RLT Stage 2 replay artifact is missing")
        if (
            manifest.get("byte_count") != path.stat().st_size
            or manifest.get("sha256") != file_sha256(path)
        ):
            raise ValueError("RLT Stage 2 replay artifact digest disagrees")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("format") != _REPLAY_FORMAT:
            raise ValueError("RLT Stage 2 replay artifact format is invalid")
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("RLT Stage 2 replay spec is invalid")
        spec = RLTStage2Spec(**dict(raw_spec))
        fingerprint = stage2_spec_fingerprint(spec)
        if (
            payload.get("spec_fingerprint") != fingerprint
            or manifest.get("spec_fingerprint") != fingerprint
            or (expected_spec is not None and spec != expected_spec)
        ):
            raise ValueError("RLT Stage 2 replay spec disagrees")
        metadata = payload.get("metadata")
        tensors = payload.get("tensors")
        if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
            raise ValueError("RLT Stage 2 replay payload is incomplete")
        replay = cls(spec, tensors, metadata=metadata)
        manifest_reward = manifest.get("average_reward")
        if (
            manifest.get("transition_count") != len(replay)
            or isinstance(manifest_reward, bool)
            or not isinstance(manifest_reward, (int, float))
            or not math.isclose(
                float(manifest_reward),
                replay.average_reward,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or dict(manifest.get("metadata", {})) != replay.metadata
        ):
            raise ValueError("RLT Stage 2 replay manifest content disagrees")
        return replay


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(value), allow_nan=False)


def _stack_rows(values: list[Tensor], name: str) -> Tensor:
    if not values:
        raise ValueError(f"RLT Stage 2 materialization produced no {name}")
    return torch.stack(values, dim=0).to(dtype=torch.float32)


def materialize_rlt_stage2_replay(
    sources: Sequence[RLTStage2Source],
    *,
    extractor: RLTStage2Extractor,
    spec: RLTStage2Spec,
    output_root: str | Path,
    config: RLTStage2DatasetConfig | None = None,
    feature_batch_size: int = 1,
    reference_seed: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RLTStage2FeatureReplay:
    """Extract frozen features and publish one verified compact replay."""

    config = config or RLTStage2DatasetConfig()
    sources = tuple(sources)
    if not sources:
        raise ValueError("RLT Stage 2 requires at least one dataset source")
    if (
        spec.chunk_length != RLT_ACTION_HORIZON
        or spec.action_dim != RLT_ACTION_DIM
        or spec.proprio_dim != RLT_ACTION_DIM
    ):
        raise ValueError(
            "RLT Stage 2 replay requires the "
            f"{RLT_ACTION_HORIZON}x{RLT_ACTION_DIM} spec"
        )
    if not math.isclose(spec.action_hz, config.expected_fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("RLT Stage 2 replay fps disagrees with its spec")
    if (
        isinstance(feature_batch_size, bool)
        or not isinstance(feature_batch_size, int)
        or feature_batch_size < 1
    ):
        raise ValueError("RLT Stage 2 feature_batch_size must be positive")
    if (
        isinstance(reference_seed, bool)
        or not isinstance(reference_seed, int)
        or not 0 <= reference_seed < 2**63 - 4
    ):
        raise ValueError("RLT Stage 2 reference_seed is invalid")

    dataset_snapshots = [source.content_snapshot() for source in sources]
    dataset_snapshot_fingerprint = _canonical_fingerprint(
        {
            "format": "cyclo.groot.rlt.dataset_collection/v1",
            "ordered_content_fingerprints": [
                snapshot["content_fingerprint"] for snapshot in dataset_snapshots
            ],
        }
    )

    raw_episodes = [
        (source, episode)
        for source in sources
        for episode in source.iter_episodes()
    ]
    total_feature_frames = 0
    episode_starts: list[
        tuple[
            RLTStage2Source,
            RLTStage2RawEpisode,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []
    for source, episode in raw_episodes:
        starts = rlt_stage2_window_starts(
            episode.length,
            chunk_length=config.chunk_length,
            stride=config.stride,
        )
        feature_indices = tuple(
            sorted(
                set(starts).union(
                    start + config.chunk_length
                    for start in starts
                    if start + config.chunk_length < episode.length
                )
            )
        )
        episode_starts.append((source, episode, starts, feature_indices))
        total_feature_frames += len(feature_indices)
    usable_episodes = [
        (source, episode, starts, feature_indices)
        for source, episode, starts, feature_indices in episode_starts
        if starts
    ]
    if not usable_episodes:
        raise ValueError(
            "RLT Stage 2 datasets contain no complete "
            f"{RLT_CHUNK_LENGTH}-step chunks"
        )
    successes = sum(
        episode.successful for _source, episode, _starts, _features in usable_episodes
    )
    failures = len(usable_episodes) - successes
    if not successes or not failures:
        raise ValueError(
            "RLT Stage 2 replay requires both success and failure episodes "
            f"with complete {RLT_CHUNK_LENGTH}-step chunks"
        )

    rows: dict[str, list[Tensor]] = {
        name: [] for name in RLTStage2FeatureReplay._TENSOR_NAMES
    }
    random.seed(reference_seed)
    np.random.seed(reference_seed % (2**32))
    torch.manual_seed(reference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(reference_seed)
    provenance: list[dict[str, Any]] = []
    completed_features = 0
    for source, episode, starts, feature_indices in episode_starts:
        if not starts:
            continue
        features: dict[int, dict[str, Tensor]] = {}
        for indices, observation in source.iter_observation_batches(
            episode, feature_indices, feature_batch_size
        ):
            extracted = extractor.extract(observation)
            if set(extracted) != {"z_rl", "proprio", "reference_actions"}:
                raise RuntimeError("RLT Stage 2 extractor fields are invalid")
            batch_count = len(indices)
            for name, expected_tail in (
                ("z_rl", (spec.rl_token_dim,)),
                ("proprio", (spec.proprio_dim,)),
                (
                    "reference_actions",
                    (GROOT_REFERENCE_ACTION_HORIZON, spec.action_dim),
                ),
            ):
                value = extracted[name]
                expected = (batch_count, *expected_tail)
                if (
                    not isinstance(value, Tensor)
                    or tuple(value.shape) != expected
                    or value.requires_grad
                    or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all())
                ):
                    raise RuntimeError(f"RLT Stage 2 extractor {name} must have shape {expected}")
            for row_index, frame_index in enumerate(indices):
                features[frame_index] = {
                    name: extracted[name][row_index]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .clone()
                    for name in extracted
                }
            completed_features += batch_count
            if progress_callback is not None:
                progress_callback(completed_features, total_feature_frames)
        if set(features) != set(feature_indices):
            raise RuntimeError("RLT Stage 2 did not extract every requested frame")

        normalized_actions = extractor.normalize_actions(
            episode.action_groups,
            episode.state_groups,
        )
        if normalized_actions.shape != (episode.length, spec.action_dim):
            raise RuntimeError("RLT Stage 2 normalized action shape disagrees")
        for start in starts:
            current = features[start]
            stop = start + config.chunk_length
            terminal = stop == episode.length
            rows["z_rl"].append(current["z_rl"])
            rows["proprio"].append(current["proprio"])
            rows["reference_actions"].append(
                current["reference_actions"][: config.chunk_length]
            )
            rows["executed_actions"].append(
                torch.from_numpy(normalized_actions[start:stop]).clone()
            )
            reward = (
                config.discount ** (config.chunk_length - 1)
                if terminal and episode.successful
                else 0.0
            )
            rows["reward"].append(torch.tensor([reward], dtype=torch.float32))
            if terminal:
                rows["bootstrap_discount"].append(torch.zeros(1))
                rows["next_z_rl"].append(torch.zeros_like(current["z_rl"]))
                rows["next_proprio"].append(torch.zeros_like(current["proprio"]))
                rows["next_reference_actions"].append(
                    torch.zeros((spec.chunk_length, spec.action_dim))
                )
            else:
                following = features[stop]
                rows["bootstrap_discount"].append(
                    torch.tensor([config.discount ** config.chunk_length])
                )
                rows["next_z_rl"].append(following["z_rl"])
                rows["next_proprio"].append(following["proprio"])
                rows["next_reference_actions"].append(
                    following["reference_actions"][: config.chunk_length]
                )
            provenance.append(
                {
                    "dataset_root": str(source.root),
                    "episode_index": episode.episode_index,
                    "start_frame_index": start,
                    "terminal": terminal,
                    "episode_success": episode.successful,
                }
            )

    tensors = {name: _stack_rows(values, name) for name, values in rows.items()}
    if dataset_snapshots != [source.content_snapshot() for source in sources]:
        raise RuntimeError("RLT Stage 2 dataset changed during materialization")
    metadata = {
        "config": asdict(config),
        "dataset_roots": [str(source.root) for source in sources],
        "episode_count": len(raw_episodes),
        "usable_episode_count": len(usable_episodes),
        "success_episode_count": successes,
        "failure_episode_count": failures,
        "skipped_short_episode_count": sum(
            not starts for _source, _episode, starts, _features in episode_starts
        ),
        "feature_frame_count": total_feature_frames,
        "transition_count": len(provenance),
        "row_provenance": provenance,
        "reward_contract": "terminal_success_plus_one_discounted_within_chunk/v1",
        "action_codec": "arm_left_8+arm_right_8+odometry_3/v1",
        "dataset_snapshots": dataset_snapshots,
        "dataset_snapshot_fingerprint": dataset_snapshot_fingerprint,
        "reference_extraction": {
            "seed": reference_seed,
            "feature_batch_size": feature_batch_size,
        },
    }
    replay = RLTStage2FeatureReplay(spec, tensors, metadata=metadata)
    replay.save(output_root)
    return RLTStage2FeatureReplay.load(output_root, expected_spec=spec)


__all__ = [
    "ACTION_GROUP_NAMES",
    "GR00TRLTStage2Extractor",
    "RLTStage2DatasetConfig",
    "RLTStage2DatasetError",
    "RLTStage2FeatureReplay",
    "RLTStage2LeRobotV21Source",
    "RLTStage2LeRobotV30Source",
    "RLTStage2RawEpisode",
    "RLTStage2Source",
    "materialize_rlt_stage2_replay",
    "open_rlt_stage2_source",
    "rlt_stage2_window_starts",
]
