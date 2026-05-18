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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

    def map_to_grid(
        self,
        grid_ns: Iterable[int],
        time_source: str = "recv",
        strategy: str = "previous",
    ) -> np.ndarray:
        """Return MP4 frame indices aligned to a target timestamp grid.

        For each ``t`` in ``grid_ns`` returns an MP4 frame index selected
        from the chosen timestamp source. ``strategy="previous"`` returns
        the largest timestamp ``<= t`` (causal). ``strategy="nearest"``
        returns whichever timestamp has the smallest absolute delta to
        ``t``. If ``t`` falls outside the recorded range, both strategies
        clamp to the first/last frame. Returns shape ``(len(grid_ns),)``
        int64.

        Args:
            grid_ns: target grid timestamps in nanoseconds.
            time_source: ``"recv"`` (subscriber clock, default — matches
                MCAP ``log_time`` semantics) or ``"header"`` (publisher
                clock from ``msg.header.stamp``).
            strategy: ``"previous"`` for causal lookup or ``"nearest"``
                for offline timestamp alignment.
        """
        grid = np.asarray(grid_ns, dtype=np.int64)
        if time_source == "recv":
            stamps = self.recv_ns
        elif time_source == "header":
            stamps = self.header_stamp_ns
        else:
            raise ValueError(
                f"time_source must be 'recv' or 'header', got {time_source!r}"
            )
        if strategy not in {"previous", "nearest"}:
            raise ValueError(
                f"strategy must be 'previous' or 'nearest', got {strategy!r}"
            )
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
        if strategy == "previous":
            pos = np.searchsorted(sorted_stamps, grid, side="right") - 1
            np.clip(pos, 0, sorted_stamps.size - 1, out=pos)
            return sorted_indices[pos].astype(np.int64)

        right = np.searchsorted(sorted_stamps, grid, side="left")
        right_clipped = np.clip(right, 0, sorted_stamps.size - 1)
        left_clipped = np.clip(right - 1, 0, sorted_stamps.size - 1)
        left_delta = np.abs(grid - sorted_stamps[left_clipped])
        right_delta = np.abs(sorted_stamps[right_clipped] - grid)
        # Deterministic tie-break: prefer the previous frame when the
        # target lies exactly halfway between two camera frames.
        pos = np.where(right_delta < left_delta, right_clipped, left_clipped)
        return sorted_indices[pos].astype(np.int64)


def load_frame_timestamps(parquet_path: Path, camera: str) -> FrameTimestamps:
    """Load the sidecar parquet for ``camera``.

    Raises ``FileNotFoundError`` if the parquet is missing. Sorting by
    ``frame_index`` is asserted — recording writes monotonic indices and
    callers rely on that for searchsorted to be correct.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"frame_timestamps not found: {parquet_path}")
    table = pq.read_table(parquet_path)
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
