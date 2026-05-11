# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-camera MP4 recorder for recording format v2.

Each camera gets a dedicated ffmpeg subprocess that takes the raw
CompressedImage payload (JPEG bytes) on stdin and remuxes it into an
MP4 container with ``-c:v copy`` — no decode, no re-encode. A worker
thread sits between the ROS callback and ffmpeg's stdin so the ROS
executor never blocks on pipe write/backpressure.

A Parquet sidecar (``videos/<cam>_timestamps.parquet``) tracks the
``header.stamp`` (publisher clock) and ``recv`` (subscriber clock) of
every frame written. LeRobot resampling uses ``header_stamp_ns`` to map
the synced grid to MP4 frame indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import subprocess
import threading
from typing import Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


# ROS image subscribers default to sensor data semantics: high depth,
# best-effort delivery, volatile durability. Matches camera driver
# publishers (zed_node, realsense2_camera).
_SUB_QOS = QoSProfile(
    depth=200,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

# Bounded per-camera queue. Large enough to absorb several seconds of
# bursty publishing on Jetson while ffmpeg warms up — when full we drop
# the newest frame and bump a counter, never blocking the ROS callback.
_QUEUE_MAX = 512

# JPEG SOI marker. Some sims emit corrupted payloads; we skip those so
# ffmpeg's mjpeg demuxer doesn't desync.
_JPEG_SOI = b"\xff\xd8"

_TIMESTAMP_SCHEMA = pa.schema([
    ("frame_index", pa.int32()),
    ("header_stamp_ns", pa.int64()),
    ("recv_ns", pa.int64()),
])


@dataclass
class _CameraStream:
    name: str
    topic: str
    mp4_path: Path
    sidecar_path: Path

    subscription: Optional[object] = None
    process: Optional[subprocess.Popen] = None
    queue: Queue = field(default_factory=lambda: Queue(maxsize=_QUEUE_MAX))
    writer: Optional[pq.ParquetWriter] = None
    worker: Optional[threading.Thread] = None

    frames_received: int = 0
    frames_written: int = 0
    frames_dropped_queue: int = 0
    frames_dropped_invalid: int = 0
    ffmpeg_error: Optional[str] = None


class VideoRecorder:
    """Manages MP4 + Parquet sidecar writers for every camera in an episode."""

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
        self._framerate_hint = framerate_hint

        self._streams: Dict[str, _CameraStream] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, episode_dir: Path) -> None:
        """Spin up subscribers + ffmpeg subprocesses for every camera."""
        if self._running:
            raise RuntimeError("VideoRecorder already running")
        videos_dir = Path(episode_dir) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, topic in self._cameras_spec.items():
            stream = _CameraStream(
                name=cam_name,
                topic=topic,
                mp4_path=videos_dir / f"{cam_name}.mp4",
                sidecar_path=videos_dir / f"{cam_name}_timestamps.parquet",
            )
            self._spawn_ffmpeg(stream)
            stream.writer = pq.ParquetWriter(
                stream.sidecar_path, _TIMESTAMP_SCHEMA, compression="zstd",
            )
            stream.worker = threading.Thread(
                target=self._worker_loop, args=(stream,),
                name=f"video-{cam_name}", daemon=True,
            )
            stream.worker.start()
            stream.subscription = self._node.create_subscription(
                CompressedImage,
                topic,
                lambda msg, s=stream: self._on_frame(s, msg),
                _SUB_QOS,
                callback_group=self._cb_group,
            )
            self._streams[cam_name] = stream
            self._node.get_logger().info(
                f"VideoRecorder: {cam_name} <- {topic} -> {stream.mp4_path.name}"
            )
        self._running = True

    def stop(self) -> Dict[str, Dict[str, int]]:
        """Drain queues, finalize ffmpeg + parquet writers, return stats.

        Performs the per-camera teardown phases in parallel: all
        subscribers are torn down, then all sentinels are pushed, then we
        join workers + close stdin + wait for ffmpeg across cameras
        concurrently. With 4 cameras and bounded per-phase waits, total
        stop time stays under the ROS service call deadline (~30s)
        instead of summing per-camera waits.
        """
        if not self._running:
            return {}

        streams = list(self._streams.values())

        # Phase 1: destroy subscriptions so no new frames enter the queues.
        for stream in streams:
            if stream.subscription is not None:
                self._node.destroy_subscription(stream.subscription)
                stream.subscription = None

        # Phase 2: push sentinels so each worker drains its queue and exits.
        for stream in streams:
            try:
                stream.queue.put(None, timeout=2.0)
            except Full:
                try:
                    stream.queue.get_nowait()
                except Empty:
                    pass
                try:
                    stream.queue.put_nowait(None)
                except Full:
                    pass

        # Phase 3: join workers in parallel by joining each with a short
        # poll, then proceeding to the next. Workers run concurrently in
        # OS threads so this fans out automatically.
        for stream in streams:
            if stream.worker is not None:
                stream.worker.join(timeout=10.0)
                if stream.worker.is_alive():
                    self._node.get_logger().error(
                        f"VideoRecorder: {stream.name} worker did not exit within 10s"
                    )

        # Phase 4: close ffmpeg stdin so each subprocess hits EOF and
        # finalises its MP4. The waits below sum up but each ffmpeg is
        # already draining concurrently — the first wait absorbs the
        # bulk of finalisation latency for all cameras.
        for stream in streams:
            if stream.process is not None and stream.process.stdin is not None:
                try:
                    if not stream.process.stdin.closed:
                        stream.process.stdin.close()
                except BrokenPipeError:
                    pass
        for stream in streams:
            if stream.process is not None:
                self._close_ffmpeg(stream)

        # Phase 5: finalise parquet writers + collect stats.
        stats: Dict[str, Dict[str, int]] = {}
        for stream in streams:
            if stream.writer is not None:
                try:
                    stream.writer.close()
                except Exception as exc:  # pragma: no cover - defensive
                    self._node.get_logger().error(
                        f"VideoRecorder: {stream.name} parquet close failed: {exc!r}"
                    )
                stream.writer = None
            stats[stream.name] = {
                "frames_received": stream.frames_received,
                "frames_written": stream.frames_written,
                "frames_dropped_queue": stream.frames_dropped_queue,
                "frames_dropped_invalid": stream.frames_dropped_invalid,
            }
            self._node.get_logger().info(
                f"VideoRecorder: {stream.name} stats {stats[stream.name]}"
            )

        self._streams.clear()
        self._running = False
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_ffmpeg(self, stream: _CameraStream) -> None:
        # Output dir must exist before ffmpeg opens the file — defend
        # against any third party (e.g. rosbag_recorder) that may rewrite
        # the episode dir between our mkdir in ``start`` and the moment
        # ffmpeg actually does open().
        stream.mp4_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg_bin,
            "-hide_banner", "-loglevel", "warning",
            "-y",
            # Skip the long input-probing phase: we know the stream is
            # MJPEG, one frame per packet, so feed the demuxer minimum
            # bytes / zero analyzeduration for near-zero startup latency.
            "-probesize", "32",
            "-analyzeduration", "0",
            "-fflags", "+nobuffer",
            # Use the time at which each packet arrives on the pipe as
            # its PTS. ROS image topics are variable-rate (camera FPS
            # drifts, missed publications), so any fixed ``-framerate``
            # hint produces a wrong-duration MP4 (e.g. 30fps stamp on a
            # 15Hz stream plays 2x fast). Wall-clock stamps let the
            # container record VFR with realistic per-frame timing.
            "-use_wallclock_as_timestamps", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-i", "pipe:0",
            "-c:v", "copy",
            "-an",
            # Pass packet PTS through unchanged so the container's
            # duration matches reality. ``cfr`` would force a fixed
            # rate; ``passthrough`` is the right mode for VFR sources.
            "-fps_mode", "passthrough",
            # Don't use +faststart for live capture — it forces ffmpeg to
            # buffer the entire stream in memory or a temp file so it can
            # move the moov atom to the front. Trail the moov instead
            # (the default), which keeps memory flat and lets the file
            # finalise in O(N) seek at close.
            str(stream.mp4_path),
        ]
        stream.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # Drain stderr in a side thread so it never fills its pipe and
        # so we can surface ffmpeg messages alongside the ROS log.
        threading.Thread(
            target=self._drain_stderr, args=(stream,),
            name=f"video-{stream.name}-stderr", daemon=True,
        ).start()

    def _drain_stderr(self, stream: _CameraStream) -> None:
        proc = stream.process
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            self._node.get_logger().warn(f"ffmpeg[{stream.name}]: {line}")
        proc.stderr.close()

    def _on_frame(self, stream: _CameraStream, msg: CompressedImage) -> None:
        stream.frames_received += 1
        data = bytes(msg.data)
        if len(data) < 2 or data[:2] != _JPEG_SOI:
            stream.frames_dropped_invalid += 1
            return
        header_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
        recv_ns = self._node.get_clock().now().nanoseconds
        try:
            stream.queue.put_nowait((data, header_ns, recv_ns))
        except Full:
            stream.frames_dropped_queue += 1
            if stream.frames_dropped_queue % 30 == 1:
                self._node.get_logger().warn(
                    f"VideoRecorder: {stream.name} queue full, dropped "
                    f"{stream.frames_dropped_queue} frame(s) total"
                )

    def _worker_loop(self, stream: _CameraStream) -> None:
        # Batch sidecar writes — round-trip per row is too chatty for parquet.
        BATCH = 32
        idxs: list[int] = []
        hdrs: list[int] = []
        recvs: list[int] = []
        next_index = 0

        def flush() -> None:
            if not idxs:
                return
            table = pa.table(
                {
                    "frame_index": pa.array(idxs, type=pa.int32()),
                    "header_stamp_ns": pa.array(hdrs, type=pa.int64()),
                    "recv_ns": pa.array(recvs, type=pa.int64()),
                },
                schema=_TIMESTAMP_SCHEMA,
            )
            if stream.writer is not None:
                stream.writer.write_table(table)
            idxs.clear()
            hdrs.clear()
            recvs.clear()

        while True:
            try:
                item = stream.queue.get(timeout=1.0)
            except Empty:
                flush()
                continue
            if item is None:
                flush()
                return
            data, header_ns, recv_ns = item
            proc = stream.process
            if proc is None or proc.stdin is None:
                stream.frames_dropped_queue += 1
                continue
            try:
                proc.stdin.write(data)
            except (BrokenPipeError, OSError) as exc:
                stream.ffmpeg_error = repr(exc)
                self._node.get_logger().error(
                    f"VideoRecorder: {stream.name} ffmpeg pipe broke: {exc!r}"
                )
                flush()
                return
            idxs.append(next_index)
            hdrs.append(header_ns)
            recvs.append(recv_ns)
            stream.frames_written += 1
            next_index += 1
            if len(idxs) >= BATCH:
                flush()

    def _close_ffmpeg(self, stream: _CameraStream) -> None:
        proc = stream.process
        if proc is None:
            return
        try:
            try:
                rc = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._node.get_logger().error(
                    f"VideoRecorder: {stream.name} ffmpeg did not exit; killing"
                )
                proc.kill()
                rc = proc.wait(timeout=5.0)
            if rc != 0:
                self._node.get_logger().error(
                    f"VideoRecorder: {stream.name} ffmpeg exit={rc}"
                )
        finally:
            stream.process = None
