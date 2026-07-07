# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-camera raw-spool recorder for recording format v2.

Recording-time work is intentionally tiny: the ROS callback copies the
CompressedImage payload into a bounded queue, and the camera video worker
appends those JPEG bytes to ``videos/<cam>.mjpeg.tmp``. ffmpeg and
Parquet writes stay off the video hot path so pipe backpressure and
pyarrow warm-up cannot make the camera queue backlog.

On STOP, writers are drained and the raw MJPEG spool is left on disk as
``videos/<cam>.mjpeg.tmp`` so the service response is not blocked by
ffmpeg. The background transcoder later remuxes it into
``videos/<cam>.mp4`` with ``-c:v copy`` before doing H.264 work. A
Parquet sidecar (``videos/<cam>_timestamps.parquet``) still tracks
``header.stamp`` (publisher clock) and ``recv_ns`` (subscriber clock)
for every frame written. LeRobot resampling maps the synced grid to MP4
frame indices using ``header_stamp_ns`` by default so transport delay
does not shift image selection; ``recv_ns`` stays available for
diagnostics and legacy fallback.

Subscriptions are created once at ``__init__`` (= when the robot_type is
first selected) and persist until ``close()``. Episode boundaries toggle
a ``_recording_active`` gate that the ROS callback checks before
enqueuing a frame, so creating subs no longer fires on every START.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


_SUB_QOS = QoSProfile(
    depth=200,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

_DEFAULT_QUEUE_MAX = 256
_QUEUE_MAX_ENV = "CYCLO_VIDEO_RECORDER_QUEUE_MAX"
_DIAGNOSTICS_ENV = "CYCLO_VIDEO_RECORDER_DIAGNOSTICS"
_REMUX_WORKERS_ENV = "CYCLO_VIDEO_RECORDER_REMUX_WORKERS"

_SOFT_CALLBACK_QUEUE_FRAMES_ENV = "CYCLO_VIDEO_RECORDER_SOFT_CALLBACK_QUEUE_FRAMES"
_SOFT_CALLBACK_QUEUE_MB_ENV = "CYCLO_VIDEO_RECORDER_SOFT_CALLBACK_QUEUE_MB"
_SOFT_RAW_QUEUE_FRAMES_ENV = "CYCLO_VIDEO_RECORDER_SOFT_RAW_QUEUE_FRAMES"
_SOFT_RAW_QUEUE_MB_ENV = "CYCLO_VIDEO_RECORDER_SOFT_RAW_QUEUE_MB"
_SOFT_METADATA_QUEUE_ROWS_ENV = "CYCLO_VIDEO_RECORDER_SOFT_METADATA_QUEUE_ROWS"
_TIMESTAMP_FUTURE_WARNING_S_ENV = "CYCLO_VIDEO_TIMESTAMP_FUTURE_WARNING_S"
_TIMESTAMP_PAST_WARNING_S_ENV = "CYCLO_VIDEO_TIMESTAMP_PAST_WARNING_S"
_CAMERA_MONITOR_STALL_WINDOW_S_ENV = "CYCLO_CAMERA_MONITOR_STALL_WINDOW_S"
_CAMERA_MONITOR_STALL_RATIO_ENV = "CYCLO_CAMERA_MONITOR_STALL_RATIO"
_CAMERA_MONITOR_SLOW_RATIO_ENV = "CYCLO_CAMERA_MONITOR_SLOW_RATIO"
_CAMERA_MONITOR_EMA_ALPHA_ENV = "CYCLO_CAMERA_MONITOR_EMA_ALPHA"
_CAMERA_MONITOR_MIN_BASELINE_HZ_ENV = "CYCLO_CAMERA_MONITOR_MIN_BASELINE_HZ"
_CAMERA_MONITOR_WARMUP_S_ENV = "CYCLO_CAMERA_MONITOR_WARMUP_S"
_CAMERA_MONITOR_RATE_WINDOW_S_ENV = "CYCLO_CAMERA_MONITOR_RATE_WINDOW_S"

_DEFAULT_SOFT_CALLBACK_QUEUE_MB = 128
_DEFAULT_SOFT_RAW_QUEUE_MB = 256
_DEFAULT_SOFT_METADATA_QUEUE_ROWS = 16_384
_DEFAULT_TIMESTAMP_FUTURE_WARNING_S = 0.5
_DEFAULT_TIMESTAMP_PAST_WARNING_S = 2.0
_DEFAULT_CAMERA_MONITOR_STALL_WINDOW_S = 2.0
_DEFAULT_CAMERA_MONITOR_STALL_RATIO = 0.2
_DEFAULT_CAMERA_MONITOR_SLOW_RATIO = 0.6
_DEFAULT_CAMERA_MONITOR_EMA_ALPHA = 0.2
_DEFAULT_CAMERA_MONITOR_MIN_BASELINE_HZ = 1.0
_DEFAULT_CAMERA_MONITOR_WARMUP_S = 3.0
_DEFAULT_CAMERA_MONITOR_RATE_WINDOW_S = 1.0
_PRESSURE_WARN_INTERVAL_NS = 1_000_000_000


def _resolve_queue_max() -> int:
    raw = os.environ.get(_QUEUE_MAX_ENV)
    if raw is None:
        return _DEFAULT_QUEUE_MAX
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_QUEUE_MAX


def _resolve_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _resolve_positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _resolve_diagnostics_mode() -> str:
    raw = os.environ.get(_DIAGNOSTICS_ENV, "summary").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "detailed", "full"}:
        return "detailed"
    return "summary"


def _resolve_diagnostics_enabled() -> bool:
    return _resolve_diagnostics_mode() == "detailed"


def _resolve_remux_workers(camera_count: int) -> int:
    default = max(1, min(camera_count, 4))
    return min(camera_count, _resolve_positive_int_env(_REMUX_WORKERS_ENV, default))


_JPEG_SOI = b"\xff\xd8"

# Trailing SOI lets ffmpeg's image2pipe/mjpeg demuxer finalize the last
# real frame without muxing a visible synthetic frame.
_JPEG_SENTINEL = _JPEG_SOI

_TIMESTAMP_SCHEMA = pa.schema([
    ("frame_index", pa.int32()),
    ("header_stamp_ns", pa.int64()),
    ("recv_ns", pa.int64()),
])

_DIAGNOSTICS_SCHEMA = pa.schema([
    ("frame_index", pa.int32()),
    ("header_stamp_ns", pa.int64()),
    ("callback_enter_ns", pa.int64()),
    ("header_read_ns", pa.int64()),
    ("bytes_copy_start_ns", pa.int64()),
    ("bytes_copy_done_ns", pa.int64()),
    ("recv_ns", pa.int64()),
    ("enqueue_start_ns", pa.int64()),
    ("enqueue_done_ns", pa.int64()),
    ("video_dequeue_ns", pa.int64()),
    ("raw_dequeue_ns", pa.int64()),
    ("raw_write_start_ns", pa.int64()),
    ("raw_write_done_ns", pa.int64()),
    ("metadata_enqueue_start_ns", pa.int64()),
    ("metadata_enqueue_done_ns", pa.int64()),
    ("metadata_dequeue_ns", pa.int64()),
    ("timestamp_flush_start_ns", pa.int64()),
    ("timestamp_flush_done_ns", pa.int64()),
    ("diagnostics_flush_start_ns", pa.int64()),
    ("diagnostics_flush_done_ns", pa.int64()),
    ("queue_size_before", pa.int32()),
    ("queue_size_after", pa.int32()),
    ("metadata_queue_size_before", pa.int32()),
    ("metadata_queue_size_after", pa.int32()),
    ("frame_size_bytes", pa.int32()),
])


@dataclass
class _CameraStream:
    name: str
    topic: str

    # Persistent — created in __init__/reconfigure, lives until close().
    subscription: Optional[object] = None
    queue: Queue = field(default_factory=Queue)
    raw_queue: Queue = field(default_factory=Queue)
    metadata_queue: Queue = field(default_factory=Queue)
    state_lock: threading.Lock = field(default_factory=threading.Lock)

    # Per-episode — populated by start_episode, cleared by stop_episode.
    mp4_path: Optional[Path] = None
    raw_path: Optional[Path] = None
    sidecar_path: Optional[Path] = None
    diagnostics_path: Optional[Path] = None
    stats_path: Optional[Path] = None
    raw_fd: Optional[int] = None
    writer: Optional[pq.ParquetWriter] = None
    diagnostics_writer: Optional[pq.ParquetWriter] = None
    writer_lock: threading.Lock = field(default_factory=threading.Lock)
    diagnostics_writer_lock: threading.Lock = field(default_factory=threading.Lock)
    worker: Optional[threading.Thread] = None
    raw_worker: Optional[threading.Thread] = None
    metadata_worker: Optional[threading.Thread] = None

    frames_received: int = 0
    frames_written: int = 0
    frames_metadata_written: int = 0
    frames_remuxed: int = 0
    frames_dropped_queue: int = 0
    frames_dropped_invalid: int = 0
    raw_write_error: Optional[str] = None
    metadata_error: Optional[str] = None
    remux_error: Optional[str] = None
    first_recv_ns: Optional[int] = None
    last_recv_ns: Optional[int] = None
    callback_queued_bytes: int = 0
    raw_queued_bytes: int = 0
    max_callback_queue_items: int = 0
    max_raw_queue_items: int = 0
    max_metadata_queue_items: int = 0
    max_callback_queue_bytes: int = 0
    max_raw_queue_bytes: int = 0
    max_enqueue_wait_ns: int = 0
    max_raw_write_ns: int = 0
    max_metadata_flush_ns: int = 0
    pressure_warning_count: int = 0
    last_pressure_warn_monotonic_ns: int = 0
    timestamp_warning_count: int = 0
    timestamp_future_warning_count: int = 0
    timestamp_past_warning_count: int = 0
    max_header_ahead_ns: int = 0
    max_header_behind_ns: int = 0
    last_timestamp_warn_monotonic_ns: int = 0
    remux_duration_ns: int = 0

    # Persistent live monitor state. This is intentionally separate from
    # per-episode stats so idle camera health checks do not contaminate
    # saved recorder diagnostics.
    monitor_subscribe_start_ns: int = 0
    monitor_message_count: int = 0
    monitor_last_recv_ns: int = 0
    monitor_last_tick_ns: int = 0
    monitor_last_count_snapshot: int = 0
    monitor_rate_hz: float = 0.0
    monitor_ema_hz: float = 0.0
    monitor_ema_initialised: bool = False
    monitor_status: int = 0
    monitor_rate_status: int = 0
    monitor_timestamp_status: int = 0
    monitor_timestamp_skew_ns: int = 0


class VideoRecorder:
    """Manages raw MJPEG spools + MP4 remux + sidecars for every camera."""

    def __init__(
        self,
        node: Node,
        cameras: Dict[str, str],
        callback_group: Optional[CallbackGroup] = None,
        ffmpeg_bin: str = "ffmpeg",
        framerate_hint: int = 30,
    ) -> None:
        self._node = node
        self._cameras_spec = dict(cameras)
        self._cb_group = callback_group or ReentrantCallbackGroup()
        self._ffmpeg_bin = shutil.which(ffmpeg_bin) or ffmpeg_bin
        self._ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
        self._framerate_hint = framerate_hint
        self._queue_max = _resolve_queue_max()
        self._soft_callback_queue_frames = _resolve_positive_int_env(
            _SOFT_CALLBACK_QUEUE_FRAMES_ENV, self._queue_max,
        )
        self._soft_callback_queue_bytes = (
            _resolve_positive_int_env(
                _SOFT_CALLBACK_QUEUE_MB_ENV, _DEFAULT_SOFT_CALLBACK_QUEUE_MB,
            )
            * 1024
            * 1024
        )
        self._soft_raw_queue_frames = _resolve_positive_int_env(
            _SOFT_RAW_QUEUE_FRAMES_ENV, self._queue_max,
        )
        self._soft_raw_queue_bytes = (
            _resolve_positive_int_env(
                _SOFT_RAW_QUEUE_MB_ENV, _DEFAULT_SOFT_RAW_QUEUE_MB,
            )
            * 1024
            * 1024
        )
        self._soft_metadata_queue_rows = _resolve_positive_int_env(
            _SOFT_METADATA_QUEUE_ROWS_ENV, _DEFAULT_SOFT_METADATA_QUEUE_ROWS,
        )
        self._timestamp_future_warning_ns = int(
            _resolve_positive_float_env(
                _TIMESTAMP_FUTURE_WARNING_S_ENV,
                _DEFAULT_TIMESTAMP_FUTURE_WARNING_S,
            ) * 1_000_000_000
        )
        self._timestamp_past_warning_ns = int(
            _resolve_positive_float_env(
                _TIMESTAMP_PAST_WARNING_S_ENV,
                _DEFAULT_TIMESTAMP_PAST_WARNING_S,
            ) * 1_000_000_000
        )
        self._camera_monitor_stall_window_ns = int(
            _resolve_positive_float_env(
                _CAMERA_MONITOR_STALL_WINDOW_S_ENV,
                _DEFAULT_CAMERA_MONITOR_STALL_WINDOW_S,
            ) * 1_000_000_000
        )
        self._camera_monitor_stall_ratio = _resolve_positive_float_env(
            _CAMERA_MONITOR_STALL_RATIO_ENV,
            _DEFAULT_CAMERA_MONITOR_STALL_RATIO,
        )
        self._camera_monitor_slow_ratio = _resolve_positive_float_env(
            _CAMERA_MONITOR_SLOW_RATIO_ENV,
            _DEFAULT_CAMERA_MONITOR_SLOW_RATIO,
        )
        self._camera_monitor_ema_alpha = min(
            1.0,
            _resolve_positive_float_env(
                _CAMERA_MONITOR_EMA_ALPHA_ENV,
                _DEFAULT_CAMERA_MONITOR_EMA_ALPHA,
            ),
        )
        self._camera_monitor_min_baseline_hz = _resolve_positive_float_env(
            _CAMERA_MONITOR_MIN_BASELINE_HZ_ENV,
            _DEFAULT_CAMERA_MONITOR_MIN_BASELINE_HZ,
        )
        self._camera_monitor_warmup_ns = int(
            _resolve_positive_float_env(
                _CAMERA_MONITOR_WARMUP_S_ENV,
                _DEFAULT_CAMERA_MONITOR_WARMUP_S,
            ) * 1_000_000_000
        )
        self._camera_monitor_rate_window_ns = int(
            _resolve_positive_float_env(
                _CAMERA_MONITOR_RATE_WINDOW_S_ENV,
                _DEFAULT_CAMERA_MONITOR_RATE_WINDOW_S,
            ) * 1_000_000_000
        )
        self._diagnostics_mode = _resolve_diagnostics_mode()
        self._diagnostics_enabled = self._diagnostics_mode == "detailed"

        self._streams: Dict[str, _CameraStream] = {}
        self._recording_active = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._callback_cond = threading.Condition()
        self._active_callbacks = 0

        self._build_streams(self._cameras_spec)
        self._node.get_logger().info(
            f"VideoRecorder: configured {len(self._streams)} camera(s), "
            f"soft_callback_queue={self._soft_callback_queue_frames} frames/"
            f"{self._soft_callback_queue_bytes // (1024 * 1024)} MiB, "
            f"soft_raw_queue={self._soft_raw_queue_frames} frames/"
            f"{self._soft_raw_queue_bytes // (1024 * 1024)} MiB, "
            f"backend=raw-spool, diagnostics={self._diagnostics_mode}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode_dir: Path) -> None:
        """Open raw spools/parquet writers and start per-camera workers."""
        with self._lifecycle_lock:
            if self._recording_active.is_set():
                raise RuntimeError("VideoRecorder already recording an episode")
            videos_dir = Path(episode_dir) / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)

            for cam_name, stream in self._streams.items():
                self._drain_queue(stream.queue)
                self._drain_queue(stream.raw_queue)
                self._drain_queue(stream.metadata_queue)

                stream.mp4_path = videos_dir / f"{cam_name}.mp4"
                stream.raw_path = videos_dir / f"{cam_name}.mjpeg.tmp"
                stream.sidecar_path = videos_dir / f"{cam_name}_timestamps.parquet"
                stream.diagnostics_path = videos_dir / f"{cam_name}_diagnostics.parquet"
                stream.stats_path = videos_dir / f"{cam_name}_recorder_stats.json"
                self._reset_stream_state(stream)

                if stream.raw_path.exists():
                    stream.raw_path.unlink()
                stream.raw_fd = os.open(
                    stream.raw_path,
                    os.O_CREAT | os.O_TRUNC | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                    0o644,
                )
                stream.writer = pq.ParquetWriter(
                    stream.sidecar_path, _TIMESTAMP_SCHEMA, compression="zstd",
                )
                if self._diagnostics_enabled:
                    stream.diagnostics_writer = pq.ParquetWriter(
                        stream.diagnostics_path,
                        _DIAGNOSTICS_SCHEMA,
                        compression="zstd",
                    )

                stream.metadata_worker = threading.Thread(
                    target=self._metadata_worker_loop, args=(stream,),
                    name=f"video-{cam_name}-metadata", daemon=True,
                )
                stream.raw_worker = threading.Thread(
                    target=self._raw_worker_loop, args=(stream,),
                    name=f"video-{cam_name}-raw", daemon=True,
                )
                stream.worker = threading.Thread(
                    target=self._video_worker_loop, args=(stream,),
                    name=f"video-{cam_name}", daemon=True,
                )
                stream.metadata_worker.start()
                stream.raw_worker.start()
                stream.worker.start()
                self._node.get_logger().info(
                    f"VideoRecorder: {cam_name} <- {stream.topic} -> "
                    f"{stream.mp4_path.name} (raw spool)"
                )

            self._recording_active.set()

    def stop_episode(self) -> Dict[str, Dict[str, object]]:
        """Drain workers, close sidecars, leave remux work to the background."""
        with self._lifecycle_lock:
            if not self._recording_active.is_set():
                return {}

            self._recording_active.clear()
            self._wait_for_active_callbacks()
            streams = list(self._streams.values())
            stats: Dict[str, Dict[str, object]] = {}

            try:
                for stream in streams:
                    self._enqueue_stop_sentinel(stream)
                for stream in streams:
                    self._join_worker(stream, timeout=10.0, phase="dispatcher drain")

                for stream in streams:
                    self._enqueue_raw_stop_sentinel(stream)
                for stream in streams:
                    self._join_raw_worker(stream, timeout=10.0, phase="raw drain")

                for stream in streams:
                    self._enqueue_metadata_stop_sentinel(stream)
                for stream in streams:
                    self._join_metadata_worker(
                        stream, timeout=10.0, phase="metadata drain",
                    )
            finally:
                for stream in streams:
                    self._close_raw_fd(stream)
                for stream in streams:
                    self._close_writer(stream)
                    self._close_diagnostics_writer(stream)
                for stream in streams:
                    self._cleanup_empty_raw_spool(stream)

                for stream in streams:
                    self._write_stats_json(stream)
                    stats[stream.name] = self._public_stats(stream)
                    self._node.get_logger().info(
                        f"VideoRecorder: {stream.name} stats {stats[stream.name]}"
                    )
                    stream.worker = None
                    stream.raw_worker = None
                    stream.metadata_worker = None
                    stream.mp4_path = None
                    stream.raw_path = None
                    stream.sidecar_path = None
                    stream.diagnostics_path = None
                    stream.stats_path = None

            return stats

    def reconfigure(self, cameras: Dict[str, str]) -> None:
        with self._lifecycle_lock:
            if self._recording_active.is_set():
                raise RuntimeError("Cannot reconfigure while recording")
            self._teardown_subscriptions()
            self._cameras_spec = dict(cameras)
            self._build_streams(self._cameras_spec)

    def close(self) -> None:
        if self._recording_active.is_set():
            try:
                self.stop_episode()
            except Exception as exc:  # pragma: no cover - defensive
                self._node.get_logger().error(
                    f"VideoRecorder.close: stop_episode raised: {exc!r}"
                )
        self._teardown_subscriptions()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_streams(self, cameras: Dict[str, str]) -> None:
        for cam_name, topic in cameras.items():
            stream = _CameraStream(
                name=cam_name,
                topic=topic,
                queue=Queue(),
                raw_queue=Queue(),
                metadata_queue=Queue(),
            )
            self._reset_camera_monitor_state(stream, time.monotonic_ns())
            stream.subscription = self._node.create_subscription(
                CompressedImage,
                topic,
                lambda msg, s=stream: self._on_frame(s, msg),
                _SUB_QOS,
                callback_group=self._cb_group,
            )
            self._streams[cam_name] = stream
            self._node.get_logger().info(
                f"VideoRecorder: {cam_name} <- {topic} subscribed (idle)"
            )

    @staticmethod
    def _drain_queue(queue: Queue) -> None:
        try:
            while True:
                queue.get_nowait()
        except Empty:
            return

    def _enqueue_stop_sentinel(self, stream: _CameraStream) -> None:
        stream.queue.put(None)

    def _enqueue_metadata_stop_sentinel(self, stream: _CameraStream) -> None:
        stream.metadata_queue.put(None)

    def _enqueue_raw_stop_sentinel(self, stream: _CameraStream) -> None:
        stream.raw_queue.put(None)

    def _join_worker(
        self, stream: _CameraStream, *, timeout: float, phase: str,
    ) -> None:
        worker = stream.worker
        if worker is None:
            return
        worker.join(timeout=timeout)
        if worker.is_alive():
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} worker did not exit during "
                f"{phase} join ({timeout:.1f}s)"
            )

    def _join_metadata_worker(
        self, stream: _CameraStream, *, timeout: float, phase: str,
    ) -> None:
        worker = stream.metadata_worker
        if worker is None:
            return
        worker.join(timeout=timeout)
        if worker.is_alive():
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} metadata worker did not exit during "
                f"{phase} join ({timeout:.1f}s)"
            )

    def _join_raw_worker(
        self, stream: _CameraStream, *, timeout: float, phase: str,
    ) -> None:
        worker = stream.raw_worker
        if worker is None:
            return
        worker.join(timeout=timeout)
        if worker.is_alive():
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} raw worker did not exit during "
                f"{phase} join ({timeout:.1f}s)"
            )

    def _reset_stream_state(self, stream: _CameraStream) -> None:
        with stream.state_lock:
            stream.frames_received = 0
            stream.frames_written = 0
            stream.frames_metadata_written = 0
            stream.frames_remuxed = 0
            stream.frames_dropped_queue = 0
            stream.frames_dropped_invalid = 0
            stream.raw_write_error = None
            stream.metadata_error = None
            stream.remux_error = None
            stream.first_recv_ns = None
            stream.last_recv_ns = None
            stream.callback_queued_bytes = 0
            stream.raw_queued_bytes = 0
            stream.max_callback_queue_items = 0
            stream.max_raw_queue_items = 0
            stream.max_metadata_queue_items = 0
            stream.max_callback_queue_bytes = 0
            stream.max_raw_queue_bytes = 0
            stream.max_enqueue_wait_ns = 0
            stream.max_raw_write_ns = 0
            stream.max_metadata_flush_ns = 0
            stream.pressure_warning_count = 0
            stream.last_pressure_warn_monotonic_ns = 0
            stream.timestamp_warning_count = 0
            stream.timestamp_future_warning_count = 0
            stream.timestamp_past_warning_count = 0
            stream.max_header_ahead_ns = 0
            stream.max_header_behind_ns = 0
            stream.last_timestamp_warn_monotonic_ns = 0
            stream.remux_duration_ns = 0

    def _reset_camera_monitor_state(
        self, stream: _CameraStream, now_ns: Optional[int] = None,
    ) -> None:
        if now_ns is None:
            now_ns = time.monotonic_ns()
        with stream.state_lock:
            stream.monitor_subscribe_start_ns = now_ns
            stream.monitor_message_count = 0
            stream.monitor_last_recv_ns = 0
            stream.monitor_last_tick_ns = 0
            stream.monitor_last_count_snapshot = 0
            stream.monitor_rate_hz = 0.0
            stream.monitor_ema_hz = 0.0
            stream.monitor_ema_initialised = False
            stream.monitor_status = 0
            stream.monitor_rate_status = 0
            stream.monitor_timestamp_status = 0
            stream.monitor_timestamp_skew_ns = 0

    def _wait_for_active_callbacks(self) -> None:
        with self._callback_cond:
            while self._active_callbacks > 0:
                self._callback_cond.wait(timeout=0.1)

    def _close_raw_fd(self, stream: _CameraStream) -> None:
        fd = stream.raw_fd
        stream.raw_fd = None
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            pass

    def _close_writer(self, stream: _CameraStream) -> None:
        if stream.writer is None:
            return
        with stream.writer_lock:
            writer = stream.writer
            if writer is None:
                return
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - defensive
                self._node.get_logger().error(
                    f"VideoRecorder: {stream.name} parquet close failed: {exc!r}"
                )
            stream.writer = None

    def _close_diagnostics_writer(self, stream: _CameraStream) -> None:
        if stream.diagnostics_writer is None:
            return
        with stream.diagnostics_writer_lock:
            writer = stream.diagnostics_writer
            if writer is None:
                return
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - defensive
                self._node.get_logger().error(
                    f"VideoRecorder: {stream.name} diagnostics parquet close failed: {exc!r}"
                )
            stream.diagnostics_writer = None

    def _now_ns(self) -> int:
        try:
            return int(self._node.get_clock().now().nanoseconds)
        except AttributeError:
            return time.perf_counter_ns()

    @staticmethod
    def _queue_size(queue: Queue) -> int:
        try:
            return int(queue.qsize())
        except (AttributeError, NotImplementedError):
            return -1

    def _pressure_warning(self, stream: _CameraStream, message: str) -> None:
        now_ns = time.monotonic_ns()
        should_log = False
        with stream.state_lock:
            stream.pressure_warning_count += 1
            if (
                now_ns - stream.last_pressure_warn_monotonic_ns
                >= _PRESSURE_WARN_INTERVAL_NS
            ):
                stream.last_pressure_warn_monotonic_ns = now_ns
                should_log = True
        if should_log:
            self._node.get_logger().warn(message)

    def _note_timestamp_skew(
        self, stream: _CameraStream, header_minus_recv_ns: int,
    ) -> None:
        future_threshold_ns = int(getattr(
            self,
            "_timestamp_future_warning_ns",
            int(_DEFAULT_TIMESTAMP_FUTURE_WARNING_S * 1_000_000_000),
        ))
        past_threshold_ns = int(getattr(
            self,
            "_timestamp_past_warning_ns",
            int(_DEFAULT_TIMESTAMP_PAST_WARNING_S * 1_000_000_000),
        ))
        kind: Optional[str] = None
        if header_minus_recv_ns > future_threshold_ns:
            kind = "future"
        elif -header_minus_recv_ns > past_threshold_ns:
            kind = "past"
        if kind is None:
            return

        now_ns = time.monotonic_ns()
        should_log = False
        with stream.state_lock:
            stream.timestamp_warning_count += 1
            if kind == "future":
                stream.timestamp_future_warning_count += 1
                stream.max_header_ahead_ns = max(
                    stream.max_header_ahead_ns,
                    header_minus_recv_ns,
                )
            else:
                stream.timestamp_past_warning_count += 1
                stream.max_header_behind_ns = max(
                    stream.max_header_behind_ns,
                    -header_minus_recv_ns,
                )
            if (
                now_ns - stream.last_timestamp_warn_monotonic_ns
                >= _PRESSURE_WARN_INTERVAL_NS
            ):
                stream.last_timestamp_warn_monotonic_ns = now_ns
                should_log = True

        if should_log:
            skew_s = header_minus_recv_ns / 1_000_000_000.0
            self._node.get_logger().warning(
                f"VideoRecorder: {stream.name} camera timestamp skew "
                f"header_minus_recv={skew_s:.3f}s"
            )

    def _camera_timestamp_status(self, header_minus_recv_ns: int) -> int:
        future_threshold_ns = int(getattr(
            self,
            "_timestamp_future_warning_ns",
            int(_DEFAULT_TIMESTAMP_FUTURE_WARNING_S * 1_000_000_000),
        ))
        past_threshold_ns = int(getattr(
            self,
            "_timestamp_past_warning_ns",
            int(_DEFAULT_TIMESTAMP_PAST_WARNING_S * 1_000_000_000),
        ))
        if header_minus_recv_ns > future_threshold_ns:
            return 2
        if -header_minus_recv_ns > past_threshold_ns:
            return 2
        return 0

    def _note_camera_monitor_sample(
        self,
        stream: _CameraStream,
        *,
        recv_monotonic_ns: int,
        header_minus_recv_ns: int,
    ) -> None:
        timestamp_status = self._camera_timestamp_status(header_minus_recv_ns)
        with stream.state_lock:
            stream.monitor_message_count += 1
            stream.monitor_last_recv_ns = recv_monotonic_ns
            stream.monitor_timestamp_skew_ns = header_minus_recv_ns
            stream.monitor_timestamp_status = timestamp_status

    def _note_callback_enqueue(
        self,
        stream: _CameraStream,
        *,
        frame_size: int,
        enqueue_wait_ns: int,
        queue_items_after: int,
    ) -> None:
        warning: Optional[str] = None
        with stream.state_lock:
            stream.callback_queued_bytes += frame_size
            stream.max_enqueue_wait_ns = max(
                stream.max_enqueue_wait_ns, enqueue_wait_ns,
            )
            if queue_items_after >= 0:
                stream.max_callback_queue_items = max(
                    stream.max_callback_queue_items, queue_items_after,
                )
            stream.max_callback_queue_bytes = max(
                stream.max_callback_queue_bytes, stream.callback_queued_bytes,
            )
            if (
                (
                    queue_items_after >= self._soft_callback_queue_frames
                    and queue_items_after >= 0
                )
                or stream.callback_queued_bytes >= self._soft_callback_queue_bytes
            ):
                warning = (
                    f"VideoRecorder: {stream.name} callback queue pressure "
                    f"items={queue_items_after} bytes={stream.callback_queued_bytes}"
                )
        if warning is not None:
            self._pressure_warning(stream, warning)

    def _note_callback_dequeue(self, stream: _CameraStream, frame_size: int) -> None:
        with stream.state_lock:
            stream.callback_queued_bytes = max(
                0, stream.callback_queued_bytes - frame_size,
            )

    def _note_raw_enqueue(
        self,
        stream: _CameraStream,
        frame_size: int,
        *,
        queue_items_after: int,
    ) -> None:
        warning: Optional[str] = None
        with stream.state_lock:
            stream.raw_queued_bytes += frame_size
            if queue_items_after >= 0:
                stream.max_raw_queue_items = max(
                    stream.max_raw_queue_items, queue_items_after,
                )
            stream.max_raw_queue_bytes = max(
                stream.max_raw_queue_bytes, stream.raw_queued_bytes,
            )
            if (
                (
                    queue_items_after >= self._soft_raw_queue_frames
                    and queue_items_after >= 0
                )
                or stream.raw_queued_bytes >= self._soft_raw_queue_bytes
            ):
                warning = (
                    f"VideoRecorder: {stream.name} raw queue pressure "
                    f"items={queue_items_after} bytes={stream.raw_queued_bytes}"
                )
        if warning is not None:
            self._pressure_warning(stream, warning)

    def _note_raw_dequeue(self, stream: _CameraStream, frame_size: int) -> None:
        with stream.state_lock:
            stream.raw_queued_bytes = max(0, stream.raw_queued_bytes - frame_size)

    def _note_metadata_enqueue(
        self, stream: _CameraStream, *, queue_items_after: int,
    ) -> None:
        warning: Optional[str] = None
        with stream.state_lock:
            if queue_items_after >= 0:
                stream.max_metadata_queue_items = max(
                    stream.max_metadata_queue_items, queue_items_after,
                )
                if queue_items_after >= self._soft_metadata_queue_rows:
                    warning = (
                        f"VideoRecorder: {stream.name} metadata queue pressure "
                        f"items={queue_items_after}"
                    )
        if warning is not None:
            self._pressure_warning(stream, warning)

    def _teardown_subscriptions(self) -> None:
        for stream in self._streams.values():
            if stream.subscription is not None:
                try:
                    self._node.destroy_subscription(stream.subscription)
                except Exception:  # pragma: no cover - destroy is best-effort
                    pass
                stream.subscription = None
        self._streams.clear()

    def _on_frame(self, stream: _CameraStream, msg: CompressedImage) -> None:
        callback_enter_ns = self._now_ns()
        recv_monotonic_ns = time.monotonic_ns()
        header_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
        header_read_ns = self._now_ns()
        recv_ns = header_read_ns
        header_minus_recv_ns = header_ns - recv_ns
        self._note_camera_monitor_sample(
            stream,
            recv_monotonic_ns=recv_monotonic_ns,
            header_minus_recv_ns=header_minus_recv_ns,
        )

        with self._callback_cond:
            if not self._recording_active.is_set():
                return
            self._active_callbacks += 1
        try:
            with stream.state_lock:
                stream.frames_received += 1
            bytes_copy_start_ns = self._now_ns()
            data = bytes(msg.data)
            bytes_copy_done_ns = self._now_ns()
            if len(data) < 2 or data[:2] != _JPEG_SOI:
                with stream.state_lock:
                    stream.frames_dropped_invalid += 1
                return
            self._note_timestamp_skew(stream, header_minus_recv_ns)
            queue_size_before = self._queue_size(stream.queue)
            enqueue_start_ns = self._now_ns()
            diagnostics = {
                "callback_enter_ns": callback_enter_ns,
                "header_read_ns": header_read_ns,
                "bytes_copy_start_ns": bytes_copy_start_ns,
                "bytes_copy_done_ns": bytes_copy_done_ns,
                "enqueue_start_ns": enqueue_start_ns,
                "enqueue_done_ns": enqueue_start_ns,
                "queue_size_before": queue_size_before,
                "queue_size_after": queue_size_before,
                "frame_size_bytes": len(data),
            }
            queue_items_after = (
                -1 if queue_size_before < 0 else queue_size_before + 1
            )
            self._note_callback_enqueue(
                stream,
                frame_size=len(data),
                enqueue_wait_ns=(
                    self._now_ns() - diagnostics["enqueue_start_ns"]
                ),
                queue_items_after=queue_items_after,
            )
            stream.queue.put((data, header_ns, recv_ns, diagnostics))
            diagnostics["enqueue_done_ns"] = self._now_ns()
            diagnostics["queue_size_after"] = queue_items_after
        finally:
            with self._callback_cond:
                self._active_callbacks -= 1
                if self._active_callbacks <= 0:
                    self._callback_cond.notify_all()

    def _video_worker_loop(self, stream: _CameraStream) -> None:
        next_index = 0
        while True:
            item = stream.queue.get()
            if item is None:
                return

            video_dequeue_ns = self._now_ns()
            data, header_ns, recv_ns, diagnostics = item
            self._note_callback_dequeue(stream, len(data))
            with stream.state_lock:
                if stream.first_recv_ns is None:
                    stream.first_recv_ns = recv_ns
                stream.last_recv_ns = recv_ns

            metadata = {
                "frame_index": next_index,
                "header_stamp_ns": header_ns,
                "recv_ns": recv_ns,
                "callback_enter_ns": diagnostics["callback_enter_ns"],
                "header_read_ns": diagnostics["header_read_ns"],
                "bytes_copy_start_ns": diagnostics["bytes_copy_start_ns"],
                "bytes_copy_done_ns": diagnostics["bytes_copy_done_ns"],
                "enqueue_start_ns": diagnostics["enqueue_start_ns"],
                "enqueue_done_ns": diagnostics.get(
                    "enqueue_done_ns", diagnostics["enqueue_start_ns"]
                ),
                "video_dequeue_ns": video_dequeue_ns,
                "queue_size_before": diagnostics["queue_size_before"],
                "queue_size_after": diagnostics.get(
                    "queue_size_after", diagnostics["queue_size_before"]
                ),
                "frame_size_bytes": diagnostics["frame_size_bytes"],
            }
            raw_queue_size_before = self._queue_size(stream.raw_queue)
            raw_queue_items_after = (
                -1 if raw_queue_size_before < 0 else raw_queue_size_before + 1
            )
            self._note_raw_enqueue(
                stream, len(data), queue_items_after=raw_queue_items_after,
            )
            stream.raw_queue.put((data, metadata))
            next_index += 1

    def _raw_worker_loop(self, stream: _CameraStream) -> None:
        try:
            while True:
                item = stream.raw_queue.get()
                if item is None:
                    self._write_raw_sentinel(stream)
                    return
                raw_dequeue_ns = self._now_ns()
                data, metadata = item
                self._note_raw_dequeue(stream, len(data))
                fd = stream.raw_fd
                if fd is None:
                    with stream.state_lock:
                        stream.raw_write_error = "raw fd closed before write"
                    self._node.get_logger().error(
                        f"VideoRecorder: {stream.name} raw fd closed before write"
                    )
                    return
                try:
                    raw_write_start_ns = self._now_ns()
                    self._write_all(fd, data)
                    raw_write_done_ns = self._now_ns()
                except OSError as exc:
                    with stream.state_lock:
                        stream.raw_write_error = repr(exc)
                    self._node.get_logger().error(
                        f"VideoRecorder: {stream.name} raw spool write failed: {exc!r}"
                    )
                    return
                with stream.state_lock:
                    stream.max_raw_write_ns = max(
                        stream.max_raw_write_ns,
                        raw_write_done_ns - raw_write_start_ns,
                    )

                metadata_queue_size_before = self._queue_size(stream.metadata_queue)
                metadata_enqueue_start_ns = self._now_ns()
                metadata.update({
                    "raw_dequeue_ns": raw_dequeue_ns,
                    "raw_write_start_ns": raw_write_start_ns,
                    "raw_write_done_ns": raw_write_done_ns,
                    "metadata_enqueue_start_ns": metadata_enqueue_start_ns,
                    "metadata_enqueue_done_ns": metadata_enqueue_start_ns,
                    "metadata_queue_size_before": metadata_queue_size_before,
                    "metadata_queue_size_after": (
                        -1 if metadata_queue_size_before < 0
                        else metadata_queue_size_before + 1
                    ),
                })
                metadata_queue_items_after = (
                    -1 if metadata_queue_size_before < 0
                    else metadata_queue_size_before + 1
                )
                self._note_metadata_enqueue(
                    stream, queue_items_after=metadata_queue_items_after,
                )
                stream.metadata_queue.put(metadata)
                with stream.state_lock:
                    stream.frames_written += 1
        finally:
            self._close_raw_fd(stream)

    def _write_raw_sentinel(self, stream: _CameraStream) -> None:
        fd = stream.raw_fd
        if fd is None:
            return
        try:
            self._write_all(fd, _JPEG_SENTINEL)
        except OSError:
            pass

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        total = 0
        while total < len(view):
            written = os.write(fd, view[total:])
            if written <= 0:
                raise OSError("os.write returned 0 bytes")
            total += written

    def _metadata_worker_loop(self, stream: _CameraStream) -> None:
        BATCH = 128
        rows: list[dict] = []

        def flush() -> None:
            if not rows:
                return
            timestamp_flush_start_ns = self._now_ns()
            table = pa.table(
                {
                    "frame_index": pa.array(
                        [row["frame_index"] for row in rows], type=pa.int32(),
                    ),
                    "header_stamp_ns": pa.array(
                        [row["header_stamp_ns"] for row in rows], type=pa.int64(),
                    ),
                    "recv_ns": pa.array(
                        [row["recv_ns"] for row in rows], type=pa.int64(),
                    ),
                },
                schema=_TIMESTAMP_SCHEMA,
            )
            writer = stream.writer
            if writer is not None:
                with stream.writer_lock:
                    if stream.writer is writer:
                        writer.write_table(table)
            timestamp_flush_done_ns = self._now_ns()

            diagnostics_flush_start_ns = timestamp_flush_done_ns
            diagnostics_flush_done_ns = diagnostics_flush_start_ns
            diagnostics_writer = stream.diagnostics_writer
            if diagnostics_writer is not None:
                diag_table = self._build_diagnostics_table(
                    rows,
                    timestamp_flush_start_ns,
                    timestamp_flush_done_ns,
                    diagnostics_flush_start_ns,
                    diagnostics_flush_done_ns,
                )
                diagnostics_flush_start_ns = self._now_ns()
                with stream.diagnostics_writer_lock:
                    if stream.diagnostics_writer is diagnostics_writer:
                        diagnostics_writer.write_table(diag_table)
                diagnostics_flush_done_ns = self._now_ns()

            flush_ns = max(
                timestamp_flush_done_ns - timestamp_flush_start_ns,
                diagnostics_flush_done_ns - diagnostics_flush_start_ns,
            )
            with stream.state_lock:
                stream.max_metadata_flush_ns = max(
                    stream.max_metadata_flush_ns, flush_ns,
                )
                stream.frames_metadata_written += len(rows)
            rows.clear()

        try:
            while True:
                try:
                    item = stream.metadata_queue.get(timeout=1.0)
                except Empty:
                    flush()
                    continue
                if item is None:
                    flush()
                    return
                item["metadata_dequeue_ns"] = self._now_ns()
                rows.append(item)
                if len(rows) >= BATCH:
                    flush()
        except Exception as exc:  # noqa: BLE001
            with stream.state_lock:
                stream.metadata_error = repr(exc)
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} metadata worker failed: {exc!r}"
            )

    def _build_diagnostics_table(
        self,
        rows: list[dict],
        timestamp_flush_start_ns: int,
        timestamp_flush_done_ns: int,
        diagnostics_flush_start_ns: int,
        diagnostics_flush_done_ns: int,
    ):
        return pa.table(
            {
                "frame_index": pa.array(
                    [row["frame_index"] for row in rows], type=pa.int32(),
                ),
                "header_stamp_ns": pa.array(
                    [row["header_stamp_ns"] for row in rows], type=pa.int64(),
                ),
                "callback_enter_ns": pa.array(
                    [row["callback_enter_ns"] for row in rows], type=pa.int64(),
                ),
                "header_read_ns": pa.array(
                    [row["header_read_ns"] for row in rows], type=pa.int64(),
                ),
                "bytes_copy_start_ns": pa.array(
                    [row["bytes_copy_start_ns"] for row in rows], type=pa.int64(),
                ),
                "bytes_copy_done_ns": pa.array(
                    [row["bytes_copy_done_ns"] for row in rows], type=pa.int64(),
                ),
                "recv_ns": pa.array(
                    [row["recv_ns"] for row in rows], type=pa.int64(),
                ),
                "enqueue_start_ns": pa.array(
                    [row["enqueue_start_ns"] for row in rows], type=pa.int64(),
                ),
                "enqueue_done_ns": pa.array(
                    [row["enqueue_done_ns"] for row in rows], type=pa.int64(),
                ),
                "video_dequeue_ns": pa.array(
                    [row["video_dequeue_ns"] for row in rows], type=pa.int64(),
                ),
                "raw_dequeue_ns": pa.array(
                    [row["raw_dequeue_ns"] for row in rows], type=pa.int64(),
                ),
                "raw_write_start_ns": pa.array(
                    [row["raw_write_start_ns"] for row in rows], type=pa.int64(),
                ),
                "raw_write_done_ns": pa.array(
                    [row["raw_write_done_ns"] for row in rows], type=pa.int64(),
                ),
                "metadata_enqueue_start_ns": pa.array(
                    [row["metadata_enqueue_start_ns"] for row in rows],
                    type=pa.int64(),
                ),
                "metadata_enqueue_done_ns": pa.array(
                    [row["metadata_enqueue_done_ns"] for row in rows],
                    type=pa.int64(),
                ),
                "metadata_dequeue_ns": pa.array(
                    [row["metadata_dequeue_ns"] for row in rows], type=pa.int64(),
                ),
                "timestamp_flush_start_ns": pa.array(
                    [timestamp_flush_start_ns] * len(rows), type=pa.int64(),
                ),
                "timestamp_flush_done_ns": pa.array(
                    [timestamp_flush_done_ns] * len(rows), type=pa.int64(),
                ),
                "diagnostics_flush_start_ns": pa.array(
                    [diagnostics_flush_start_ns] * len(rows), type=pa.int64(),
                ),
                "diagnostics_flush_done_ns": pa.array(
                    [diagnostics_flush_done_ns] * len(rows), type=pa.int64(),
                ),
                "queue_size_before": pa.array(
                    [row["queue_size_before"] for row in rows], type=pa.int32(),
                ),
                "queue_size_after": pa.array(
                    [row["queue_size_after"] for row in rows], type=pa.int32(),
                ),
                "metadata_queue_size_before": pa.array(
                    [row["metadata_queue_size_before"] for row in rows],
                    type=pa.int32(),
                ),
                "metadata_queue_size_after": pa.array(
                    [row["metadata_queue_size_after"] for row in rows],
                    type=pa.int32(),
                ),
                "frame_size_bytes": pa.array(
                    [row["frame_size_bytes"] for row in rows], type=pa.int32(),
                ),
            },
            schema=_DIAGNOSTICS_SCHEMA,
        )

    def _stream_stats_snapshot(self, stream: _CameraStream) -> dict:
        with stream.state_lock:
            return {
                "name": stream.name,
                "topic": stream.topic,
                "diagnostics_mode": self._diagnostics_mode,
                "frames_received": stream.frames_received,
                "frames_written": stream.frames_written,
                "frames_metadata_written": stream.frames_metadata_written,
                "frames_remuxed": stream.frames_remuxed,
                "frames_dropped_queue": stream.frames_dropped_queue,
                "frames_dropped_invalid": stream.frames_dropped_invalid,
                "raw_write_error": stream.raw_write_error,
                "metadata_error": stream.metadata_error,
                "remux_error": stream.remux_error,
                "first_recv_ns": stream.first_recv_ns,
                "last_recv_ns": stream.last_recv_ns,
                "callback_queue_bytes_current": stream.callback_queued_bytes,
                "raw_queue_bytes_current": stream.raw_queued_bytes,
                "max_callback_queue_items": stream.max_callback_queue_items,
                "max_raw_queue_items": stream.max_raw_queue_items,
                "max_metadata_queue_items": stream.max_metadata_queue_items,
                "max_callback_queue_bytes": stream.max_callback_queue_bytes,
                "max_raw_queue_bytes": stream.max_raw_queue_bytes,
                "max_enqueue_wait_ns": stream.max_enqueue_wait_ns,
                "max_raw_write_ns": stream.max_raw_write_ns,
                "max_metadata_flush_ns": stream.max_metadata_flush_ns,
                "pressure_warning_count": stream.pressure_warning_count,
                "timestamp_warning_count": stream.timestamp_warning_count,
                "timestamp_future_warning_count": (
                    stream.timestamp_future_warning_count
                ),
                "timestamp_past_warning_count": (
                    stream.timestamp_past_warning_count
                ),
                "max_header_ahead_ns": stream.max_header_ahead_ns,
                "max_header_behind_ns": stream.max_header_behind_ns,
                "remux_duration_ns": stream.remux_duration_ns,
                "soft_callback_queue_frames": self._soft_callback_queue_frames,
                "soft_callback_queue_bytes": self._soft_callback_queue_bytes,
                "soft_raw_queue_frames": self._soft_raw_queue_frames,
                "soft_raw_queue_bytes": self._soft_raw_queue_bytes,
                "soft_metadata_queue_rows": self._soft_metadata_queue_rows,
            }

    @staticmethod
    def _remux_status_from_snapshot(snapshot: dict) -> str:
        if snapshot.get("remux_error"):
            return "failed"
        frames_written = int(snapshot.get("frames_written") or 0)
        frames_remuxed = int(snapshot.get("frames_remuxed") or 0)
        if frames_written <= 0:
            return "not_required"
        if frames_remuxed >= frames_written:
            return "done"
        return "pending"

    def _public_stats(self, stream: _CameraStream) -> Dict[str, object]:
        snapshot = self._stream_stats_snapshot(stream)
        return {
            "frames_received": int(snapshot["frames_received"]),
            "frames_written": int(snapshot["frames_written"]),
            "frames_metadata_written": int(snapshot["frames_metadata_written"]),
            "frames_dropped_queue": int(snapshot["frames_dropped_queue"]),
            "frames_dropped_invalid": int(snapshot["frames_dropped_invalid"]),
            "frames_remuxed": int(snapshot["frames_remuxed"]),
            "pressure_warning_count": int(snapshot["pressure_warning_count"]),
            "timestamp_warning_count": int(snapshot["timestamp_warning_count"]),
            "timestamp_future_warning_count": int(
                snapshot["timestamp_future_warning_count"]
            ),
            "timestamp_past_warning_count": int(
                snapshot["timestamp_past_warning_count"]
            ),
            "max_header_ahead_ns": int(snapshot["max_header_ahead_ns"]),
            "max_header_behind_ns": int(snapshot["max_header_behind_ns"]),
            "max_callback_queue_items": int(snapshot["max_callback_queue_items"]),
            "max_raw_queue_items": int(snapshot["max_raw_queue_items"]),
            "max_metadata_queue_items": int(snapshot["max_metadata_queue_items"]),
            "max_callback_queue_bytes": int(snapshot["max_callback_queue_bytes"]),
            "max_raw_queue_bytes": int(snapshot["max_raw_queue_bytes"]),
            "raw_write_error": snapshot["raw_write_error"],
            "metadata_error": snapshot["metadata_error"],
            "remux_error": snapshot["remux_error"],
            "remux_status": self._remux_status_from_snapshot(snapshot),
        }

    def camera_monitor_snapshot(self) -> Dict[str, list]:
        now_ns = time.monotonic_ns()
        stall_window_ns = int(getattr(
            self,
            "_camera_monitor_stall_window_ns",
            int(_DEFAULT_CAMERA_MONITOR_STALL_WINDOW_S * 1_000_000_000),
        ))
        warmup_ns = int(getattr(
            self,
            "_camera_monitor_warmup_ns",
            int(_DEFAULT_CAMERA_MONITOR_WARMUP_S * 1_000_000_000),
        ))
        slow_ratio = float(getattr(
            self,
            "_camera_monitor_slow_ratio",
            _DEFAULT_CAMERA_MONITOR_SLOW_RATIO,
        ))
        stall_ratio = float(getattr(
            self,
            "_camera_monitor_stall_ratio",
            _DEFAULT_CAMERA_MONITOR_STALL_RATIO,
        ))
        ema_alpha = float(getattr(
            self,
            "_camera_monitor_ema_alpha",
            _DEFAULT_CAMERA_MONITOR_EMA_ALPHA,
        ))
        min_baseline_hz = float(getattr(
            self,
            "_camera_monitor_min_baseline_hz",
            _DEFAULT_CAMERA_MONITOR_MIN_BASELINE_HZ,
        ))
        rate_window_ns = int(getattr(
            self,
            "_camera_monitor_rate_window_ns",
            int(_DEFAULT_CAMERA_MONITOR_RATE_WINDOW_S * 1_000_000_000),
        ))

        snapshot: Dict[str, list] = {
            "names": [],
            "topics": [],
            "rates_hz": [],
            "baseline_hz": [],
            "seconds_since_last": [],
            "status": [],
            "timestamp_skew_s": [],
            "timestamp_status": [],
        }

        for stream in self._streams.values():
            with stream.state_lock:
                count_now = stream.monitor_message_count
                last_recv_ns = stream.monitor_last_recv_ns
                last_tick_ns = stream.monitor_last_tick_ns
                last_count_snapshot = stream.monitor_last_count_snapshot
                subscribe_start_ns = stream.monitor_subscribe_start_ns
                rate_hz = stream.monitor_rate_hz
                ema_hz = stream.monitor_ema_hz
                ema_initialised = stream.monitor_ema_initialised
                rate_status = stream.monitor_rate_status
                timestamp_status = stream.monitor_timestamp_status
                timestamp_skew_ns = stream.monitor_timestamp_skew_ns

            should_update_rate = last_tick_ns == 0 or (
                now_ns > last_tick_ns
                and (now_ns - last_tick_ns) >= rate_window_ns
            )
            if should_update_rate and last_tick_ns != 0 and now_ns > last_tick_ns:
                dt_s = (now_ns - last_tick_ns) / 1_000_000_000.0
                if dt_s > 0.0:
                    rate_hz = max(
                        0.0,
                        (count_now - last_count_snapshot) / dt_s,
                    )

            in_warmup = (
                subscribe_start_ns != 0
                and now_ns >= subscribe_start_ns
                and (now_ns - subscribe_start_ns) < warmup_ns
            )
            no_recent_msg = False
            if last_recv_ns != 0:
                no_recent_msg = (
                    now_ns > last_recv_ns
                    and (now_ns - last_recv_ns) > stall_window_ns
                )
            elif subscribe_start_ns != 0:
                no_recent_msg = (
                    now_ns > subscribe_start_ns
                    and (now_ns - subscribe_start_ns) > stall_window_ns
                )

            stalled = False
            slow = False
            if not in_warmup:
                effective_baseline = ema_hz if ema_hz > min_baseline_hz else 0.0
                if no_recent_msg:
                    stalled = True
                elif should_update_rate and effective_baseline > 0.0:
                    if rate_hz < effective_baseline * stall_ratio:
                        stalled = True
                    elif rate_hz < effective_baseline * slow_ratio:
                        slow = True

                if should_update_rate and not stalled and not slow and rate_hz > 0.0:
                    if not ema_initialised:
                        ema_hz = rate_hz
                        ema_initialised = True
                    else:
                        ema_hz = (ema_alpha * rate_hz) + (
                            (1.0 - ema_alpha) * ema_hz
                        )

            if stalled:
                rate_status = 2
            elif slow:
                rate_status = 1
            elif should_update_rate:
                rate_status = 0
            combined_status = max(rate_status, timestamp_status)
            seconds_since_last = (
                -1.0
                if last_recv_ns == 0
                else max(0.0, (now_ns - last_recv_ns) / 1_000_000_000.0)
            )

            with stream.state_lock:
                if should_update_rate:
                    stream.monitor_last_tick_ns = now_ns
                    stream.monitor_last_count_snapshot = count_now
                stream.monitor_rate_hz = rate_hz
                stream.monitor_ema_hz = ema_hz
                stream.monitor_ema_initialised = ema_initialised
                stream.monitor_rate_status = rate_status
                stream.monitor_status = combined_status

            snapshot["names"].append(stream.name)
            snapshot["topics"].append(stream.topic)
            snapshot["rates_hz"].append(float(rate_hz))
            snapshot["baseline_hz"].append(float(ema_hz))
            snapshot["seconds_since_last"].append(float(seconds_since_last))
            snapshot["status"].append(int(combined_status))
            snapshot["timestamp_skew_s"].append(
                float(timestamp_skew_ns / 1_000_000_000.0)
            )
            snapshot["timestamp_status"].append(int(timestamp_status))

        return snapshot

    def recording_warnings(self) -> list[str]:
        warnings: list[str] = []
        future_threshold_s = (
            int(getattr(
                self,
                "_timestamp_future_warning_ns",
                int(_DEFAULT_TIMESTAMP_FUTURE_WARNING_S * 1_000_000_000),
            ))
            / 1_000_000_000.0
        )
        past_threshold_s = (
            int(getattr(
                self,
                "_timestamp_past_warning_ns",
                int(_DEFAULT_TIMESTAMP_PAST_WARNING_S * 1_000_000_000),
            ))
            / 1_000_000_000.0
        )
        for stream in self._streams.values():
            with stream.state_lock:
                future_count = stream.timestamp_future_warning_count
                past_count = stream.timestamp_past_warning_count
            if future_count > 0:
                warnings.append(
                    f"{stream.name}: camera header timestamp is more than "
                    f"{future_threshold_s:g}s ahead of receive time."
                )
            if past_count > 0:
                warnings.append(
                    f"{stream.name}: camera header timestamp is more than "
                    f"{past_threshold_s:g}s behind receive time."
                )
        return warnings

    def _write_stats_json(self, stream: _CameraStream) -> None:
        if self._diagnostics_mode == "off" or stream.stats_path is None:
            return
        payload = self._stream_stats_snapshot(stream)
        payload.update({
            "remux_status": self._remux_status_from_snapshot(payload),
            "mp4_path": str(stream.mp4_path) if stream.mp4_path else None,
            "raw_path": str(stream.raw_path) if stream.raw_path else None,
            "sidecar_path": str(stream.sidecar_path) if stream.sidecar_path else None,
            "diagnostics_path": (
                str(stream.diagnostics_path)
                if self._diagnostics_enabled and stream.diagnostics_path
                else None
            ),
        })
        try:
            stream.stats_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self._node.get_logger().warning(
                f"VideoRecorder: {stream.name} stats json write failed: {exc!r}"
            )

    def _cleanup_empty_raw_spool(self, stream: _CameraStream) -> None:
        raw_path = stream.raw_path
        if raw_path is None:
            return
        with stream.state_lock:
            frames_written = stream.frames_written
        if frames_written > 0:
            return
        try:
            if raw_path.exists():
                raw_path.unlink()
        except OSError:
            pass

    def _remux_streams(self, streams: list[_CameraStream]) -> None:
        active_streams = [stream for stream in streams if stream.frames_written > 0]
        if not active_streams:
            for stream in streams:
                self._remux_raw_spool(stream)
            return
        max_workers = _resolve_remux_workers(len(active_streams))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._remux_raw_spool, stream): stream
                for stream in active_streams
            }
            for future in as_completed(futures):
                stream = futures[future]
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    with stream.state_lock:
                        stream.remux_error = repr(exc)
                    self._node.get_logger().error(
                        f"VideoRecorder: {stream.name} remux raised: {exc!r}"
                    )
        for stream in streams:
            if stream.frames_written <= 0:
                self._remux_raw_spool(stream)

    def _remux_raw_spool(self, stream: _CameraStream) -> None:
        raw_path = stream.raw_path
        mp4_path = stream.mp4_path
        if raw_path is None or mp4_path is None:
            return
        with stream.state_lock:
            frames_written = stream.frames_written
        if frames_written <= 0:
            try:
                if raw_path.exists():
                    raw_path.unlink()
            except OSError:
                pass
            return
        if not raw_path.exists():
            with stream.state_lock:
                stream.remux_error = "raw spool missing"
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} raw spool missing: {raw_path}"
            )
            return

        tmp_mp4 = mp4_path.with_name(f"{mp4_path.stem}.remuxing.mp4")
        if tmp_mp4.exists():
            try:
                tmp_mp4.unlink()
            except OSError:
                pass
        fps = self._estimate_framerate(stream)
        cmd = [
            self._ffmpeg_bin,
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-framerate", f"{fps:.6f}",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-i", str(raw_path),
            "-c:v", "copy",
            "-an",
            "-fps_mode", "passthrough",
            "-video_track_timescale", "90000",
            str(tmp_mp4),
        ]
        start_ns = self._now_ns()
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        done_ns = self._now_ns()
        with stream.state_lock:
            stream.remux_duration_ns = done_ns - start_ns
        if result.returncode != 0:
            with stream.state_lock:
                stream.remux_error = result.stderr.decode("utf-8", errors="replace")
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} remux failed: {stream.remux_error}"
            )
            return

        frames = self._probe_frame_count(tmp_mp4)
        with stream.state_lock:
            stream.frames_remuxed = frames
            frames_written = stream.frames_written
        if frames != frames_written:
            with stream.state_lock:
                stream.remux_error = (
                    f"frame count mismatch mp4={frames} sidecar={frames_written}"
                )
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} {stream.remux_error}; "
                f"raw spool preserved at {raw_path}"
            )
            return

        os.replace(tmp_mp4, mp4_path)
        try:
            raw_path.unlink()
        except OSError as exc:
            self._node.get_logger().warning(
                f"VideoRecorder: {stream.name} raw spool cleanup failed: {exc!r}"
            )
        self._node.get_logger().info(
            f"VideoRecorder: {stream.name} remuxed {frames} frame(s) "
            f"in {(done_ns - start_ns) / 1_000_000_000.0:.3f}s"
        )

    def _estimate_framerate(self, stream: _CameraStream) -> float:
        with stream.state_lock:
            frames_written = stream.frames_written
            first_recv_ns = stream.first_recv_ns
            last_recv_ns = stream.last_recv_ns
        if (
            frames_written > 1
            and first_recv_ns is not None
            and last_recv_ns is not None
            and last_recv_ns > first_recv_ns
        ):
            duration_s = (last_recv_ns - first_recv_ns) / 1_000_000_000.0
            fps = (frames_written - 1) / duration_s
            if 1.0 <= fps <= 120.0:
                return fps
        return float(max(1, self._framerate_hint))

    def _probe_frame_count(self, path: Path) -> int:
        result = subprocess.run(
            [
                self._ffprobe_bin,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames,nb_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return -1
        for line in result.stdout.splitlines():
            text = line.strip()
            if not text or text == "N/A":
                continue
            try:
                return int(text)
            except ValueError:
                continue
        return -1
