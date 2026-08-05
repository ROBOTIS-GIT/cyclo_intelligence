# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sidecar Parquet reader for recording format v2 frame timestamps.

VideoRecorder writes one Parquet per camera with columns:
``frame_index`` (int32), ``header_stamp_ns`` (int64), ``recv_ns`` (int64).
This module loads that file and produces helpers to map a synced grid
of target timestamps (from LeRobot resampling) to MP4 frame indices.
Camera synchronization uses ``header_stamp_ns`` by default so transport
delay does not shift the chosen image frame; ``recv_ns`` is only a
fallback for malformed/legacy sidecars with no header stamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pyarrow.parquet as pq


@dataclass
class FrameTimestamps:
    """In-memory view of one camera's sidecar Parquet."""

    camera: str
    frame_index: np.ndarray  # shape=(N,), int32
    header_stamp_ns: np.ndarray  # shape=(N,), int64
    recv_ns: np.ndarray  # shape=(N,), int64

    @property
    def num_frames(self) -> int:
        return int(self.frame_index.shape[0])

    def effective_time_source(self, time_source: str = "header") -> str:
        """Return the actual timestamp column used for grid alignment."""
        if time_source == "header":
            if self.header_stamp_ns.size and np.any(self.header_stamp_ns > 0):
                return "header"
            return "recv"
        if time_source == "recv":
            return "recv"
        raise ValueError(
            f"time_source must be 'header' or 'recv', got {time_source!r}"
        )

    def stamp_ns_for_grid(self, time_source: str = "header") -> np.ndarray:
        """Return timestamp column used to align camera frames to a grid."""
        effective = self.effective_time_source(time_source)
        if effective == "header":
            return self.header_stamp_ns
        if effective == "recv":
            return self.recv_ns
        raise AssertionError(f"unexpected time_source: {effective!r}")

    def map_to_grid(
        self,
        grid_ns: Iterable[int],
        time_source: str = "header",
    ) -> np.ndarray:
        """Return MP4 frame indices aligned to a target timestamp grid.

        For each ``t`` in ``grid_ns`` returns the largest MP4 frame index
        whose chosen stamp is ``<= t`` (causal — never use a frame from
        the future). If ``t`` precedes every recorded frame the result
        is 0 (clamp). Returns shape ``(len(grid_ns),)`` int64.

        Args:
            grid_ns: target grid timestamps in nanoseconds.
            time_source: ``"header"`` (publisher clock from
                ``msg.header.stamp``, default for camera alignment) or
                ``"recv"`` (subscriber clock, fallback/debug path).
        """
        grid = np.asarray(grid_ns, dtype=np.int64)
        stamps = self.stamp_ns_for_grid(time_source)
        if stamps.size == 0:
            return np.zeros(grid.shape, dtype=np.int64)
        # The sidecar guarantees frame_index monotonicity but not stamp
        # monotonicity — recv times are monotonic in practice but a
        # rebooted clock would break searchsorted's contract. Use a
        # sorted copy when needed; numpy sort is O(N log N) once per
        # camera per episode, negligible.
        if stamps.size > 1 and not np.all(np.diff(stamps) >= 0):
            order = np.argsort(stamps)
            sorted_stamps = stamps[order]
            sorted_indices = np.arange(stamps.size)[order]
        else:
            sorted_stamps = stamps
            sorted_indices = np.arange(stamps.size)
        pos = np.searchsorted(sorted_stamps, grid, side="right") - 1
        np.clip(pos, 0, sorted_stamps.size - 1, out=pos)
        return sorted_indices[pos].astype(np.int64)


def _optional_stamp(stamps: np.ndarray, index: int, *, positive_only: bool) -> Optional[int]:
    if index < 0 or index >= int(stamps.size):
        return None
    value = int(stamps[index])
    if positive_only and value <= 0:
        return None
    return value


def build_frame_reuse_report(
    indices: Iterable[int],
    grid_ns: Iterable[int],
    frame_timestamps: FrameTimestamps,
    *,
    episode_index: int,
    camera: str,
    fps: int,
    time_source: str = "header",
) -> Dict[str, Any]:
    """Build compressed metadata for target frames that reused a source frame."""
    index_array = np.asarray(indices, dtype=np.int64)
    grid = np.asarray(grid_ns, dtype=np.int64)
    total_target_frames = int(index_array.size)
    total_source_frames = int(frame_timestamps.num_frames)
    effective_time_source = frame_timestamps.effective_time_source(time_source)
    stamps = frame_timestamps.stamp_ns_for_grid(time_source)

    grid_for_count = grid[:total_target_frames]
    clamped_before_first_count = 0
    if stamps.size and grid_for_count.size:
        clamped_before_first_count = int(
            np.count_nonzero(grid_for_count < int(np.min(stamps)))
        )

    runs: List[Dict[str, Any]] = []
    reused_target_frames = 0
    run_start: Optional[int] = None
    run_source: Optional[int] = None

    def finish_run(end_frame: int) -> None:
        nonlocal run_start, run_source, reused_target_frames
        if run_start is None or run_source is None:
            return
        source_pos = int(run_source)
        source_frame_index = source_pos
        if 0 <= source_pos < int(frame_timestamps.frame_index.size):
            source_frame_index = int(frame_timestamps.frame_index[source_pos])
        count = int(end_frame - run_start + 1)
        reused_target_frames += count
        runs.append({
            "target_start_frame": int(run_start),
            "target_end_frame": int(end_frame),
            "count": count,
            "source_frame_index": source_frame_index,
            "source_header_stamp_ns": _optional_stamp(
                frame_timestamps.header_stamp_ns,
                source_pos,
                positive_only=True,
            ),
            "source_recv_ns": _optional_stamp(
                frame_timestamps.recv_ns,
                source_pos,
                positive_only=False,
            ),
        })
        run_start = None
        run_source = None

    for target_frame in range(1, total_target_frames):
        source_pos = int(index_array[target_frame])
        reused = source_pos == int(index_array[target_frame - 1])
        if reused:
            if run_start is None:
                run_start = target_frame
                run_source = source_pos
            elif run_source != source_pos:
                finish_run(target_frame - 1)
                run_start = target_frame
                run_source = source_pos
        elif run_start is not None:
            finish_run(target_frame - 1)
    if run_start is not None:
        finish_run(total_target_frames - 1)

    reuse_ratio = (
        float(reused_target_frames) / float(total_target_frames)
        if total_target_frames > 0 else 0.0
    )
    return {
        "episode_index": int(episode_index),
        "camera": str(camera),
        "target_fps": int(fps),
        "time_source": effective_time_source,
        "total_target_frames": total_target_frames,
        "total_source_frames": total_source_frames,
        "reused_target_frames": int(reused_target_frames),
        "reuse_ratio": reuse_ratio,
        "clamped_before_first_count": int(clamped_before_first_count),
        "runs": runs,
    }


def load_frame_timestamps(parquet_path: Path, camera: str) -> FrameTimestamps:
    """Load the sidecar parquet for ``camera``.

    Raises ``FileNotFoundError`` if the parquet is missing. Sorting by
    ``frame_index`` is asserted — recording writes monotonic indices and
    callers rely on that for searchsorted to be correct.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"frame_timestamps not found: {parquet_path}")
    # This is always one sidecar file. Avoid pq.read_table(), which routes
    # through PyArrow's dataset layer and has crashed in aarch64 wheels.
    table = pq.ParquetFile(parquet_path).read(
        columns=["frame_index", "header_stamp_ns", "recv_ns"],
    )
    frame_index = table.column("frame_index").to_numpy().astype(np.int64)
    header_stamp_ns = table.column("header_stamp_ns").to_numpy().astype(np.int64)
    recv_ns = table.column("recv_ns").to_numpy().astype(np.int64)
    if frame_index.size > 1 and not np.all(np.diff(frame_index) >= 0):
        raise ValueError(
            f"{parquet_path} frame_index column must be non-decreasing"
        )
    return FrameTimestamps(
        camera=camera,
        frame_index=frame_index.astype(np.int32),
        header_stamp_ns=header_stamp_ns,
        recv_ns=recv_ns,
    )
