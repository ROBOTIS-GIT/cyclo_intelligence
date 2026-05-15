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
every frame written. LeRobot resampling maps the synced grid to MP4
frame indices using ``recv_ns`` by default, matching MCAP ``log_time``
semantics; ``header_stamp_ns`` stays available for diagnostics.

Subscriptions are created once at ``__init__`` (= when the robot_type
is first selected) and persist until ``close()``. Episode boundaries
toggle a ``_recording_active`` gate that the ROS callback checks before
enqueuing a frame, so creating subs no longer fires on every START.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
import os
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
_DEFAULT_QUEUE_MAX = 256
_QUEUE_MAX_ENV = "CYCLO_VIDEO_RECORDER_QUEUE_MAX"


def _resolve_queue_max() -> int:
    raw = os.environ.get(_QUEUE_MAX_ENV)
    if raw is None:
        return _DEFAULT_QUEUE_MAX
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_QUEUE_MAX

# JPEG SOI marker. Some sims emit corrupted payloads; we skip those so
# ffmpeg's mjpeg demuxer doesn't desync.
_JPEG_SOI = b"\xff\xd8"

# Minimal valid 1x1 grayscale JPEG, written into ffmpeg's stdin right
# before close() so the demuxer can finalise the last real frame.
#
# Why: ffmpeg's image2pipe+mjpeg demuxer only commits frame N to the
# muxer when it sees frame N+1's SOI marker — it uses the next SOI as
# the byte-boundary for the previous frame. On EOF the last buffered
# frame is dropped, which makes mp4 frame_count = sidecar rows - 1.
# Feeding one extra full JPEG before close lets that real last frame
# get demuxed; this sentinel itself has no next-SOI so the demuxer
# drops it. Verified empirically (synthetic N=5/10/50/100 all recover
# to exactly N frames with this trailer).
_JPEG_SENTINEL = bytes.fromhex(
    "ffd8"                                                          # SOI
    "ffe000104a46494600010100000100010000"                          # APP0 (JFIF)
    "ffdb004300" + "01" * 64                                        # DQT
    + "ffc0000b08000100010101110000"                                # SOF0 (1x1)
    + "ffc4001f0000010501010101010100000000000000000102030405060708090a0b"  # DHT (DC)
    + "ffc40014100100000000000000000000000000000000"                # DHT (AC)
    + "ffda0008010100003f00" + "00"                                 # SOS + 1 byte ECS
    + "ffd9"                                                        # EOI
)

_TIMESTAMP_SCHEMA = pa.schema([
    ("frame_index", pa.int32()),
    ("header_stamp_ns", pa.int64()),
    ("recv_ns", pa.int64()),
])


@dataclass
class _CameraStream:
    name: str
    topic: str

    # Persistent — created in __init__/reconfigure, lives until close().
    subscription: Optional[object] = None
    queue: Queue = field(default_factory=lambda: Queue(maxsize=_DEFAULT_QUEUE_MAX))

    # Per-episode — populated by start_episode, cleared by stop_episode.
    mp4_path: Optional[Path] = None
    sidecar_path: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    writer: Optional[pq.ParquetWriter] = None
    writer_lock: threading.Lock = field(default_factory=threading.Lock)
    worker: Optional[threading.Thread] = None

    frames_received: int = 0
    frames_written: int = 0
    frames_dropped_queue: int = 0
    frames_dropped_invalid: int = 0
    ffmpeg_error: Optional[str] = None


class VideoRecorder:
    """Manages MP4 + Parquet sidecar writers for every camera in an episode.

    Subscriptions are created up-front in ``__init__`` and persist until
    ``close()`` (or a ``reconfigure()`` that swaps the camera set). Each
    episode's MP4/parquet/worker lifecycle is bracketed by
    ``start_episode`` / ``stop_episode`` — those toggle the
    ``_recording_active`` gate that the ROS callback consults to decide
    whether to enqueue a frame.
    """

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
        self._queue_max = _resolve_queue_max()

        self._streams: Dict[str, _CameraStream] = {}
        # Simple bool — read in the ROS callback on every frame, flipped
        # by start_episode/stop_episode. GIL covers the read/write so no
        # lock needed.
        self._recording_active = False

        self._build_streams(self._cameras_spec)
        self._node.get_logger().info(
            f"VideoRecorder: configured {len(self._streams)} camera(s), "
            f"queue_max={self._queue_max}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode_dir: Path) -> None:
        """Open MP4/parquet writers and worker threads for the next episode.

        Subscriptions are already live from ``__init__`` — this only spins
        up the per-episode ffmpeg/parquet/worker triple and flips the
        gate so the callback starts enqueuing frames.
        """
        if self._recording_active:
            raise RuntimeError("VideoRecorder already recording an episode")
        videos_dir = Path(episode_dir) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, stream in self._streams.items():
            # Reset per-episode state. Drain any stale items from the
            # queue — should be empty since the gate was off, but a
            # callback in flight from before the gate flipped could have
            # raced through.
            try:
                while True:
                    stream.queue.get_nowait()
            except Empty:
                pass
            stream.mp4_path = videos_dir / f"{cam_name}.mp4"
            stream.sidecar_path = videos_dir / f"{cam_name}_timestamps.parquet"
            stream.frames_received = 0
            stream.frames_written = 0
            stream.frames_dropped_queue = 0
            stream.frames_dropped_invalid = 0
            stream.ffmpeg_error = None

            self._spawn_ffmpeg(stream)
            stream.writer = pq.ParquetWriter(
                stream.sidecar_path, _TIMESTAMP_SCHEMA, compression="zstd",
            )
            stream.worker = threading.Thread(
                target=self._worker_loop, args=(stream,),
                name=f"video-{cam_name}", daemon=True,
            )
            stream.worker.start()
            self._node.get_logger().info(
                f"VideoRecorder: {cam_name} <- {stream.topic} -> {stream.mp4_path.name}"
            )

        self._recording_active = True

    def stop_episode(self) -> Dict[str, Dict[str, int]]:
        """Drain queues, finalize ffmpeg + parquet writers, return stats.

        Subscriptions stay alive for the next episode — only the
        per-episode writers/workers are torn down. The recording gate is
        flipped first so no further frames enter the queues while we
        drain.
        """
        if not self._recording_active:
            return {}

        # Close the gate first so the ROS callback stops enqueuing.
        self._recording_active = False

        streams = list(self._streams.values())
        stats: Dict[str, Dict[str, int]] = {}

        try:
            # Push sentinels so each worker drains its queue and exits.
            for stream in streams:
                self._enqueue_stop_sentinel(stream)

            # First drain window: in the healthy path workers consume the
            # sentinel, flush their final parquet batch, and return before
            # we close ffmpeg stdin.
            for stream in streams:
                self._join_worker(stream, timeout=10.0, phase="drain")
        finally:
            # The ffmpeg + parquet cleanup must always run so subprocesses
            # never leak and parquet writers always flush, even if an
            # earlier phase raised. ffmpeg without stdin.close() would
            # block forever waiting for EOF, exceeding the ~30s service
            # deadline.
            #
            # Trail one sentinel JPEG into each ffmpeg before close so
            # the mjpeg image2pipe demuxer can finalise the real last
            # frame (see _JPEG_SENTINEL docstring for the demuxer quirk
            # this avoids). The sentinel itself is dropped because no
            # further SOI follows it.
            for stream in streams:
                self._write_ffmpeg_sentinel(stream)
            for stream in streams:
                self._close_ffmpeg_stdin(stream)
            for stream in streams:
                if stream.process is not None:
                    try:
                        self._close_ffmpeg(stream)
                    except Exception as exc:  # pragma: no cover - defensive
                        self._node.get_logger().error(
                            f"VideoRecorder: {stream.name} ffmpeg close failed: {exc!r}"
                        )
            for stream in streams:
                self._join_worker(stream, timeout=2.0, phase="final")

            for stream in streams:
                self._close_writer(stream)
                stats[stream.name] = {
                    "frames_received": stream.frames_received,
                    "frames_written": stream.frames_written,
                    "frames_dropped_queue": stream.frames_dropped_queue,
                    "frames_dropped_invalid": stream.frames_dropped_invalid,
                }
                self._node.get_logger().info(
                    f"VideoRecorder: {stream.name} stats {stats[stream.name]}"
                )
                # Reset per-episode references so the next start_episode
                # spins fresh ffmpeg/worker. Subscription + queue stay.
                stream.worker = None
                stream.process = None
                stream.mp4_path = None
                stream.sidecar_path = None

        return stats

    def reconfigure(self, cameras: Dict[str, str]) -> None:
        """Swap the camera set — destroy current subs, build new ones.

        Only called when the active robot_type changes. Refuses if an
        episode is in flight; the caller (RecordingService) is responsible
        for ensuring stop_episode ran first.
        """
        if self._recording_active:
            raise RuntimeError("Cannot reconfigure while recording")
        self._teardown_subscriptions()
        self._cameras_spec = dict(cameras)
        self._build_streams(self._cameras_spec)

    def close(self) -> None:
        """Tear down all subscriptions — called on node shutdown."""
        if self._recording_active:
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
        """Create one _CameraStream + subscription per camera."""
        for cam_name, topic in cameras.items():
            stream = _CameraStream(
                name=cam_name,
                topic=topic,
                queue=Queue(maxsize=self._queue_max),
            )
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

    def _enqueue_stop_sentinel(self, stream: _CameraStream) -> None:
        """Ensure a worker sees its stop sentinel even if the queue is full."""
        try:
            stream.queue.put(None, timeout=2.0)
            return
        except Full:
            pass
        try:
            stream.queue.get_nowait()
        except Empty:
            pass
        try:
            stream.queue.put_nowait(None)
        except Full:
            self._node.get_logger().error(
                f"VideoRecorder: {stream.name} queue stayed full; "
                "worker may need ffmpeg close to exit"
            )

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

    def _write_ffmpeg_sentinel(self, stream: _CameraStream) -> None:
        if stream.process is None or stream.process.stdin is None:
            return
        try:
            if not stream.process.stdin.closed:
                stream.process.stdin.write(_JPEG_SENTINEL)
        except (BrokenPipeError, OSError):
            pass

    def _close_ffmpeg_stdin(self, stream: _CameraStream) -> None:
        if stream.process is None or stream.process.stdin is None:
            return
        try:
            if not stream.process.stdin.closed:
                stream.process.stdin.close()
        except (BrokenPipeError, OSError):
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

    def _teardown_subscriptions(self) -> None:
        for stream in self._streams.values():
            if stream.subscription is not None:
                try:
                    self._node.destroy_subscription(stream.subscription)
                except Exception:  # pragma: no cover - destroy is best-effort
                    pass
                stream.subscription = None
        self._streams.clear()

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
            # Pin the mp4 video-track timescale to 90000 (H.264 RTP
            # standard). The default ~12800Hz timebase only has ~78μs
            # resolution, so adjacent frames arriving in the same tick
            # collide and ffmpeg spams "Non-monotonic DTS ... changing
            # to N+1" warnings. 90000 also matches what
            # converter/video_sync.py writes for the final lerobot mp4,
            # so raw / transcoded / final all share one timescale.
            "-video_track_timescale", "90000",
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
        # Subscription is persistent (created in __init__) but the ROS
        # callback runs whenever a publisher emits, including between
        # episodes. Drop frames when no episode is active — we don't
        # want them in the queue.
        if not self._recording_active:
            return
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
            writer = stream.writer
            if writer is not None:
                with stream.writer_lock:
                    if stream.writer is writer:
                        writer.write_table(table)
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
