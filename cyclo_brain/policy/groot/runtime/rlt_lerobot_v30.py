#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Safe, read-only access to Cyclo LeRobot v3.0 aggregate datasets.

LeRobot v3 stores several episodes in one parquet/video shard.  RLT still
consumes one episode at a time, so this module validates the compact episode
index and exposes exact per-episode parquet and video slices without creating
or rewriting a second dataset.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_dataset_json(path: Path, error_type: type[ValueError]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(f"Cannot read dataset metadata: {path}") from error


def read_dataset_jsonl(path: Path, error_type: type[ValueError]) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(f"Cannot read dataset metadata: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise error_type(f"Dataset metadata is empty or invalid: {path}")
    return rows


class RLTLeRobotV30Error(ValueError):
    """Raised when a v3 dataset cannot satisfy the immutable RLT contract."""


ParquetRowsReader = Callable[
    [Path, Sequence[str], Sequence[str]], Sequence[Mapping[str, Any]]
]
ParquetSliceReader = Callable[
    [Path, Sequence[str], int, int, int], Mapping[str, Sequence[Any]]
]
VideoSegmentReader = Callable[[Path, int, int, float], Iterator[np.ndarray]]


@dataclass(frozen=True)
class RLTLeRobotV30VideoSegment:
    path: Path
    start_frame: int
    length: int


@dataclass(frozen=True)
class RLTLeRobotV30Episode:
    index: int
    length: int
    tasks: tuple[str, ...]
    metadata_success: bool | None
    data_path: Path
    data_start: int
    data_stop: int
    data_shard_length: int
    videos: Mapping[str, RLTLeRobotV30VideoSegment]


def _default_parquet_rows(
    path: Path,
    required: Sequence[str],
    optional: Sequence[str],
) -> Sequence[Mapping[str, Any]]:
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
    except Exception as error:  # noqa: BLE001 - dependency validation boundary
        raise RLTLeRobotV30Error(f"Cannot inspect LeRobot v3 parquet: {path}") from error
    missing = set(required).difference(names)
    if missing:
        raise RLTLeRobotV30Error(
            f"LeRobot v3 parquet is missing columns {sorted(missing)}: {path}"
        )
    columns = [*required, *(name for name in optional if name in names)]
    try:
        return pq.read_table(path, columns=columns).to_pylist()
    except Exception as error:  # noqa: BLE001 - dependency validation boundary
        raise RLTLeRobotV30Error(f"Cannot read LeRobot v3 parquet: {path}") from error


def _default_parquet_slice(
    path: Path,
    columns: Sequence[str],
    start: int,
    stop: int,
    expected_rows: int,
) -> Mapping[str, Sequence[Any]]:
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        row_count = int(parquet.metadata.num_rows)
    except Exception as error:  # noqa: BLE001 - dependency validation boundary
        raise RLTLeRobotV30Error(f"Cannot inspect LeRobot v3 data: {path}") from error
    missing = set(columns).difference(names)
    if missing:
        raise RLTLeRobotV30Error(
            f"LeRobot v3 data is missing columns {sorted(missing)}: {path}"
        )
    if row_count != expected_rows or start < 0 or stop <= start or stop > row_count:
        raise RLTLeRobotV30Error(f"LeRobot v3 episode frame slice is invalid: {path}")

    values: dict[str, list[Any]] = {name: [] for name in columns}
    cursor = 0
    try:
        for batch in parquet.iter_batches(batch_size=2048, columns=list(columns)):
            batch_stop = cursor + batch.num_rows
            overlap_start = max(start, cursor)
            overlap_stop = min(stop, batch_stop)
            if overlap_start < overlap_stop:
                selected = batch.slice(
                    overlap_start - cursor,
                    overlap_stop - overlap_start,
                )
                for name in columns:
                    column_index = selected.schema.get_field_index(name)
                    if column_index < 0:
                        raise RLTLeRobotV30Error(
                            f"Parquet batch is missing required column: {name}"
                        )
                    values[name].extend(selected.column(column_index).to_pylist())
            cursor = batch_stop
            if cursor >= stop:
                break
    except Exception as error:  # noqa: BLE001 - dependency validation boundary
        raise RLTLeRobotV30Error(f"Cannot read LeRobot v3 data: {path}") from error
    if any(len(column) != stop - start for column in values.values()):
        raise RLTLeRobotV30Error(f"LeRobot v3 episode frame slice is incomplete: {path}")
    return values


def _default_video_segment_reader(
    path: Path,
    start_frame: int,
    length: int,
    fps: float,
) -> Iterator[np.ndarray]:
    import av

    container = av.open(str(path), mode="r")
    try:
        streams = tuple(container.streams.video)
        if len(streams) != 1:
            raise RLTLeRobotV30Error(
                f"RLT expects one video stream in LeRobot v3 shard: {path}"
            )
        stream = streams[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        if not math.isclose(rate, fps, rel_tol=0.0, abs_tol=1e-6):
            raise RLTLeRobotV30Error(
                f"LeRobot v3 video FPS disagrees with dataset metadata: {path}"
            )
        start_seconds = start_frame / fps
        container.seek(
            int(start_seconds / float(stream.time_base)),
            backward=True,
            stream=stream,
        )
        expected = start_frame
        stop = start_frame + length
        for frame in container.decode(stream):
            if frame.pts is None:
                raise RLTLeRobotV30Error(f"LeRobot v3 video frame has no PTS: {path}")
            frame_index = round(float(frame.pts * stream.time_base) * fps)
            if frame_index < start_frame:
                continue
            if frame_index >= stop:
                break
            if frame_index != expected:
                raise RLTLeRobotV30Error(
                    f"LeRobot v3 video frame sequence is discontinuous: {path}"
                )
            expected += 1
            yield frame.to_ndarray(format="rgb24")
        if expected != stop:
            raise RLTLeRobotV30Error(
                f"LeRobot v3 video is shorter than its episode slice: {path}"
            )
    finally:
        container.close()


def lerobot_codebase_version(root: str | Path) -> str:
    path = Path(root).expanduser().absolute()
    try:
        value = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RLTLeRobotV30Error(f"Cannot read LeRobot metadata: {path}") from error
    raw = str(value.get("codebase_version") if isinstance(value, Mapping) else "")
    normalized = raw.strip().lower().removeprefix("v")
    if normalized.startswith("2.1"):
        return "v2.1"
    if normalized.startswith("3"):
        return "v3.0"
    raise RLTLeRobotV30Error(f"Unsupported LeRobot codebase_version: {raw or 'missing'}")


class RLTLeRobotV30Layout:
    """Validated episode index over one immutable LeRobot v3 root."""

    def __init__(
        self,
        root: str | Path,
        *,
        camera_keys: Sequence[str],
        expected_fps: float | None = None,
        require_outcomes: bool = False,
        parquet_rows_reader: ParquetRowsReader | None = None,
        parquet_slice_reader: ParquetSliceReader | None = None,
        video_segment_reader: VideoSegmentReader | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        if self.root.is_symlink() or not self.root.is_dir():
            raise RLTLeRobotV30Error(
                f"LeRobot v3 dataset must be a real directory: {self.root}"
            )
        self._resolved_root = self.root.resolve(strict=True)
        self._rows_reader = parquet_rows_reader or _default_parquet_rows
        self._slice_reader = parquet_slice_reader or _default_parquet_slice
        self._video_reader = video_segment_reader or _default_video_segment_reader

        info_path = self._safe_file("meta/info.json")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RLTLeRobotV30Error(
                f"Cannot read LeRobot v3 metadata: {info_path}"
            ) from error
        if not isinstance(info, Mapping) or info.get("codebase_version") != "v3.0":
            raise RLTLeRobotV30Error("RLT LeRobot v3 source requires codebase_version v3.0")
        self.info = dict(info)
        try:
            self.fps = float(info["fps"])
            self.total_episodes = int(info["total_episodes"])
            self.total_frames = int(info["total_frames"])
        except (KeyError, TypeError, ValueError) as error:
            raise RLTLeRobotV30Error("LeRobot v3 counts/FPS metadata is invalid") from error
        if (
            not math.isfinite(self.fps)
            or self.fps <= 0
            or self.total_episodes < 1
            or self.total_frames < 1
        ):
            raise RLTLeRobotV30Error("LeRobot v3 counts/FPS metadata is invalid")
        if expected_fps is not None and not math.isclose(
            self.fps, float(expected_fps), rel_tol=0.0, abs_tol=1e-6
        ):
            raise RLTLeRobotV30Error(
                f"RLT requires {float(expected_fps):g} Hz data; got {self.fps:g}"
            )

        features = info.get("features")
        if not isinstance(features, Mapping):
            raise RLTLeRobotV30Error("LeRobot v3 features metadata is missing")
        self.features = dict(features)
        outcome = features.get("episode_success")
        if require_outcomes and not (
            isinstance(outcome, Mapping) and outcome.get("dtype") == "bool"
        ):
            raise RLTLeRobotV30Error(
                "RLT Stage 2 requires a boolean episode_success feature"
            )

        self.camera_features: dict[str, str] = {}
        for camera in camera_keys:
            matches = [
                str(key)
                for key, value in features.items()
                if str(key).endswith(f".{camera}")
                and isinstance(value, Mapping)
                and value.get("dtype") == "video"
            ]
            if len(matches) != 1:
                raise RLTLeRobotV30Error(
                    f"LeRobot v3 must contain exactly one {camera} video"
                )
            self.camera_features[str(camera)] = matches[0]

        self._data_pattern = self._path_pattern(info.get("data_path"), "data_path")
        self._video_pattern = self._path_pattern(info.get("video_path"), "video_path")
        self.tasks = self._read_tasks()
        self.episodes = self._read_episodes(require_outcomes=require_outcomes)
        self.consumed_files = self._consumed_files(info_path)

    @staticmethod
    def _path_pattern(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RLTLeRobotV30Error(f"LeRobot v3 {label} is invalid")
        return value

    def _safe_file(self, relative: str | Path) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RLTLeRobotV30Error(f"Unsafe LeRobot v3 path: {relative_path}")
        candidate = self.root / relative_path
        cursor = self.root
        for part in relative_path.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RLTLeRobotV30Error(
                    f"LeRobot v3 paths must not use symbolic links: {cursor}"
                )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self._resolved_root)
        except (OSError, ValueError) as error:
            raise RLTLeRobotV30Error(
                f"LeRobot v3 file is missing or escapes its root: {candidate}"
            ) from error
        if not resolved.is_file():
            raise RLTLeRobotV30Error(f"LeRobot v3 path is not a file: {resolved}")
        return resolved

    def _format_file(self, pattern: str, **values: int | str) -> Path:
        try:
            relative = pattern.format(**values)
        except (KeyError, ValueError) as error:
            raise RLTLeRobotV30Error("LeRobot v3 path template is invalid") from error
        return self._safe_file(relative)

    def _metadata_files(self) -> tuple[Path, ...]:
        root = self.root / "meta/episodes"
        if root.is_symlink() or not root.is_dir():
            raise RLTLeRobotV30Error("LeRobot v3 episode metadata is missing or unsafe")
        paths = sorted(root.glob("chunk-*/file-*.parquet"))
        if not paths or len(paths) > 10_000:
            raise RLTLeRobotV30Error("LeRobot v3 episode metadata shard count is invalid")
        return tuple(
            self._safe_file(path.relative_to(self.root))
            for path in paths
        )

    def _read_tasks(self) -> dict[int, str]:
        path = self._safe_file("meta/tasks.parquet")
        rows = self._rows_reader(path, ("task_index", "task"), ())
        tasks: dict[int, str] = {}
        try:
            for row in rows:
                index = self._integer(row.get("task_index"), "task_index")
                task = str(row["task"])
                if index in tasks or not task:
                    raise ValueError("duplicate or empty task")
                tasks[index] = task
        except (KeyError, TypeError, ValueError) as error:
            raise RLTLeRobotV30Error("LeRobot v3 task metadata is invalid") from error
        if not tasks:
            raise RLTLeRobotV30Error("LeRobot v3 task metadata is empty")
        return tasks

    @staticmethod
    def _integer(value: Any, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(label)
        parsed = int(value)
        if parsed < 0 or isinstance(value, float) and not value.is_integer():
            raise ValueError(label)
        return parsed

    @staticmethod
    def _scalar(value: Any) -> Any:
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        return value

    def _read_episodes(self, *, require_outcomes: bool) -> tuple[RLTLeRobotV30Episode, ...]:
        camera_columns: list[str] = []
        for feature in self.camera_features.values():
            prefix = f"videos/{feature}"
            camera_columns.extend(
                (
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                    f"{prefix}/to_timestamp",
                )
            )
        required = (
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            *camera_columns,
        )
        raw_rows: list[Mapping[str, Any]] = []
        for path in self._metadata_files():
            raw_rows.extend(
                self._rows_reader(
                    path,
                    required,
                    ("tasks", "stats/episode_success/mean"),
                )
            )
        if len(raw_rows) != self.total_episodes:
            raise RLTLeRobotV30Error("LeRobot v3 episode count disagrees with metadata")

        parsed: list[dict[str, Any]] = []
        try:
            for row in raw_rows:
                index = self._integer(row.get("episode_index"), "episode_index")
                length = self._integer(row.get("length"), "length")
                data_chunk = self._integer(row.get("data/chunk_index"), "data chunk")
                data_file = self._integer(row.get("data/file_index"), "data file")
                data_from = self._integer(row.get("dataset_from_index"), "data start")
                data_to = self._integer(row.get("dataset_to_index"), "data stop")
                if length < 1 or data_to - data_from != length:
                    raise ValueError("episode length")
                raw_tasks = row.get("tasks", ())
                tasks = (
                    tuple(str(value) for value in raw_tasks if str(value))
                    if isinstance(raw_tasks, (list, tuple))
                    else ((str(raw_tasks),) if raw_tasks else ())
                )
                raw_success = self._scalar(row.get("stats/episode_success/mean"))
                metadata_success: bool | None = None
                if raw_success is not None:
                    numeric = float(raw_success)
                    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
                        raise ValueError("episode_success")
                    metadata_success = numeric >= 0.5
                if require_outcomes and metadata_success is None:
                    raise ValueError("episode_success")

                videos: dict[str, dict[str, Any]] = {}
                for camera, feature in self.camera_features.items():
                    prefix = f"videos/{feature}"
                    chunk = self._integer(row.get(f"{prefix}/chunk_index"), "video chunk")
                    file = self._integer(row.get(f"{prefix}/file_index"), "video file")
                    start_s = float(row[f"{prefix}/from_timestamp"])
                    stop_s = float(row[f"{prefix}/to_timestamp"])
                    if (
                        not math.isfinite(start_s)
                        or not math.isfinite(stop_s)
                        or start_s < 0.0
                        or stop_s <= start_s
                    ):
                        raise ValueError("video timestamps")
                    start_frame = round(start_s * self.fps)
                    stop_frame = round(stop_s * self.fps)
                    if stop_frame - start_frame != length:
                        raise ValueError("video duration")
                    videos[camera] = {
                        "chunk": chunk,
                        "file": file,
                        "start": start_frame,
                        "stop": stop_frame,
                    }
                parsed.append(
                    {
                        "index": index,
                        "length": length,
                        "tasks": tasks,
                        "metadata_success": metadata_success,
                        "data_chunk": data_chunk,
                        "data_file": data_file,
                        "data_from": data_from,
                        "data_to": data_to,
                        "videos": videos,
                    }
                )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RLTLeRobotV30Error("LeRobot v3 episode metadata is invalid") from error

        parsed.sort(key=lambda value: value["index"])
        if [row["index"] for row in parsed] != list(range(self.total_episodes)):
            raise RLTLeRobotV30Error(
                "LeRobot v3 episode indices are incomplete or non-contiguous"
            )
        cursor = 0
        for row in parsed:
            if row["data_from"] != cursor:
                raise RLTLeRobotV30Error("LeRobot v3 dataset frame ranges are not contiguous")
            cursor = row["data_to"]
        if cursor != self.total_frames:
            raise RLTLeRobotV30Error("LeRobot v3 total_frames disagrees with episodes")

        data_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in parsed:
            data_groups[(row["data_chunk"], row["data_file"])].append(row)
        data_locations: dict[int, tuple[Path, int, int, int]] = {}
        for (chunk, file), rows in data_groups.items():
            rows.sort(key=lambda value: value["data_from"])
            shard_start = rows[0]["data_from"]
            expected = shard_start
            for row in rows:
                if row["data_from"] != expected:
                    raise RLTLeRobotV30Error(
                        "LeRobot v3 data shard frame ranges are not contiguous"
                    )
                expected = row["data_to"]
            path = self._format_file(
                self._data_pattern,
                chunk_index=chunk,
                file_index=file,
            )
            shard_length = expected - shard_start
            for row in rows:
                data_locations[row["index"]] = (
                    path,
                    row["data_from"] - shard_start,
                    row["data_to"] - shard_start,
                    shard_length,
                )

        video_locations: dict[int, dict[str, RLTLeRobotV30VideoSegment]] = defaultdict(dict)
        for camera, feature in self.camera_features.items():
            groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for row in parsed:
                video = row["videos"][camera]
                groups[(video["chunk"], video["file"])].append(row)
            for (chunk, file), rows in groups.items():
                rows.sort(key=lambda value: value["videos"][camera]["start"])
                expected = 0
                for row in rows:
                    video = row["videos"][camera]
                    if video["start"] != expected:
                        raise RLTLeRobotV30Error(
                            "LeRobot v3 video shard episode ranges are not contiguous"
                        )
                    expected = video["stop"]
                path = self._format_file(
                    self._video_pattern,
                    video_key=feature,
                    chunk_index=chunk,
                    file_index=file,
                )
                for row in rows:
                    video = row["videos"][camera]
                    video_locations[row["index"]][camera] = RLTLeRobotV30VideoSegment(
                        path=path,
                        start_frame=video["start"],
                        length=row["length"],
                    )

        episodes = []
        for row in parsed:
            data_path, data_start, data_stop, shard_length = data_locations[row["index"]]
            episodes.append(
                RLTLeRobotV30Episode(
                    index=row["index"],
                    length=row["length"],
                    tasks=row["tasks"],
                    metadata_success=row["metadata_success"],
                    data_path=data_path,
                    data_start=data_start,
                    data_stop=data_stop,
                    data_shard_length=shard_length,
                    videos=dict(video_locations[row["index"]]),
                )
            )
        return tuple(episodes)

    def _consumed_files(self, info_path: Path) -> tuple[Path, ...]:
        files = {
            info_path,
            self._safe_file("meta/tasks.parquet"),
            *self._metadata_files(),
        }
        for episode in self.episodes:
            files.add(episode.data_path)
            files.update(segment.path for segment in episode.videos.values())
        return tuple(sorted(files))

    def read_episode_columns(
        self,
        episode: RLTLeRobotV30Episode,
        columns: Sequence[str],
    ) -> Mapping[str, Sequence[Any]]:
        requested = tuple(dict.fromkeys(columns))
        internal = tuple(dict.fromkeys((*requested, "episode_index", "frame_index")))
        values = self._slice_reader(
            episode.data_path,
            internal,
            episode.data_start,
            episode.data_stop,
            episode.data_shard_length,
        )
        if set(values) != set(internal):
            raise RLTLeRobotV30Error("LeRobot v3 parquet columns disagree")
        try:
            episode_indices = tuple(int(value) for value in values["episode_index"])
            frame_indices = tuple(int(value) for value in values["frame_index"])
        except (TypeError, ValueError) as error:
            raise RLTLeRobotV30Error("LeRobot v3 frame indices are invalid") from error
        if episode_indices != (episode.index,) * episode.length:
            raise RLTLeRobotV30Error("LeRobot v3 episode frame slice leaked")
        if frame_indices != tuple(range(episode.length)):
            raise RLTLeRobotV30Error("LeRobot v3 frame indices are not contiguous")
        return {name: values[name] for name in requested}

    def iter_video(
        self,
        episode: RLTLeRobotV30Episode,
        camera: str,
    ) -> Iterator[np.ndarray]:
        try:
            segment = episode.videos[camera]
        except KeyError as error:
            raise RLTLeRobotV30Error(f"LeRobot v3 episode has no {camera} video") from error
        return self._video_reader(
            segment.path,
            segment.start_frame,
            segment.length,
            self.fps,
        )


__all__ = [
    "RLTLeRobotV30Episode",
    "RLTLeRobotV30Error",
    "RLTLeRobotV30Layout",
    "RLTLeRobotV30VideoSegment",
    "lerobot_codebase_version",
]
