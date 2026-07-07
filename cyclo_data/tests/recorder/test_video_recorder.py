"""Resource-lifecycle tests for the recording-format-v2 VideoRecorder."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import os
import shutil
import subprocess
import sys
import threading
import types

import pytest


for mod_name in [
    "rclpy",
    "rclpy.callback_groups",
    "rclpy.node",
    "rclpy.qos",
    "sensor_msgs",
    "sensor_msgs.msg",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)


sys.modules["rclpy.callback_groups"].CallbackGroup = object
sys.modules["rclpy.callback_groups"].ReentrantCallbackGroup = object
sys.modules["rclpy.node"].Node = object
sys.modules["rclpy.qos"].DurabilityPolicy = types.SimpleNamespace(VOLATILE=1)
sys.modules["rclpy.qos"].HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
sys.modules["rclpy.qos"].ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)


class _QoSProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


sys.modules["rclpy.qos"].QoSProfile = _QoSProfile
sys.modules["sensor_msgs.msg"].CompressedImage = object

try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.int32 = lambda: "int32"
    pyarrow_stub.int64 = lambda: "int64"
    pyarrow_stub.array = lambda values, type=None: list(values)
    pyarrow_stub.schema = lambda fields: fields
    pyarrow_stub.table = lambda data, schema=None: {"data": data, "schema": schema}
    sys.modules["pyarrow"] = pyarrow_stub
    parquet_stub = types.ModuleType("pyarrow.parquet")
    parquet_stub.ParquetWriter = object
    sys.modules["pyarrow.parquet"] = parquet_stub


from cyclo_data.recorder.video_recorder import (  # noqa: E402
    _DEFAULT_QUEUE_MAX,
    _CameraStream,
    _JPEG_SENTINEL,
    _JPEG_SOI,
    _resolve_diagnostics_enabled,
    _resolve_diagnostics_mode,
    _resolve_queue_max,
    VideoRecorder,
)
import cyclo_data.recorder.video_recorder as video_recorder_module  # noqa: E402


class _Logger:
    def __init__(self, events):
        self.events = events

    def info(self, message):
        self.events.append(("info", message))

    def warn(self, message):
        self.events.append(("warn", message))

    def warning(self, message):
        self.events.append(("warning", message))

    def error(self, message):
        self.events.append(("error", message))


class _Node:
    def __init__(self, events):
        self._logger = _Logger(events)

    def get_logger(self):
        return self._logger


class _FakeStdin:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def write(self, data):
        self.events.append(("stdin.write", len(data)))

    def close(self):
        self.closed = True
        self.events.append(("stdin.close", None))


class _FakeProcess:
    def __init__(self, events):
        self.events = events
        self.stdin = _FakeStdin(events)

    def wait(self, timeout):
        self.events.append(("process.wait", timeout))
        return 0

    def kill(self):
        self.events.append(("process.kill", None))


class _FakeWorker:
    def __init__(self, events, name="worker"):
        self.events = events
        self.name = name

    def join(self, timeout):
        self.events.append((f"{self.name}.join", timeout))

    def is_alive(self):
        return False


class _FakeWriter:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append(("writer.close", None))


def _recorder_with_stream(stream):
    events = []
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder._node = _Node(events)
    recorder._recording_active = threading.Event()
    recorder._recording_active.set()
    recorder._lifecycle_lock = threading.Lock()
    recorder._callback_cond = threading.Condition()
    recorder._active_callbacks = 0
    recorder._streams = {stream.name: stream}
    recorder._ffmpeg_bin = "ffmpeg"
    recorder._ffprobe_bin = "ffprobe"
    recorder._framerate_hint = 30
    recorder._soft_callback_queue_frames = 256
    recorder._soft_callback_queue_bytes = 128 * 1024 * 1024
    recorder._soft_raw_queue_frames = 256
    recorder._soft_raw_queue_bytes = 256 * 1024 * 1024
    recorder._soft_metadata_queue_rows = 16_384
    recorder._diagnostics_mode = "summary"
    recorder._diagnostics_enabled = False
    return recorder, events


def _bare_recorder(events=None):
    if events is None:
        events = []
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder._node = _Node(events)
    recorder._soft_callback_queue_frames = 256
    recorder._soft_callback_queue_bytes = 128 * 1024 * 1024
    recorder._soft_raw_queue_frames = 256
    recorder._soft_raw_queue_bytes = 256 * 1024 * 1024
    recorder._soft_metadata_queue_rows = 16_384
    recorder._diagnostics_mode = "summary"
    recorder._diagnostics_enabled = False
    return recorder


def test_queue_max_default_env_override_and_invalid(monkeypatch):
    monkeypatch.delenv("CYCLO_VIDEO_RECORDER_QUEUE_MAX", raising=False)
    assert _resolve_queue_max() == _DEFAULT_QUEUE_MAX == 256

    monkeypatch.setenv("CYCLO_VIDEO_RECORDER_QUEUE_MAX", "7")
    assert _resolve_queue_max() == 7

    monkeypatch.setenv("CYCLO_VIDEO_RECORDER_QUEUE_MAX", "invalid")
    assert _resolve_queue_max() == _DEFAULT_QUEUE_MAX


def test_ffmpeg_trailer_is_non_decodable_soi_marker():
    assert _JPEG_SENTINEL == _JPEG_SOI
    assert len(_JPEG_SENTINEL) == 2


def test_ffmpeg_trailer_flushes_without_extra_gray_frame(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        pytest.skip(f"OpenCV/numpy unavailable: {exc}")

    frames = []
    for value in (32, 96, 224):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = value
        image[:, :, 1] = 255 - value
        image[:, :, 2] = (value * 2) % 255
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        frames.append(encoded.tobytes())

    output = tmp_path / "out.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-i",
        "pipe:0",
        "-c:v",
        "copy",
        "-an",
        "-fps_mode",
        "passthrough",
        "-video_track_timescale",
        "90000",
        str(output),
    ]
    result = subprocess.run(
        command,
        input=b"".join(frames) + _JPEG_SENTINEL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")

    capture = cv2.VideoCapture(str(output))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()

    assert len(decoded) == len(frames)
    last = decoded[-1]
    assert not np.allclose(last.mean(axis=(0, 1)), [128, 128, 128], atol=2)
    assert last.std() > 0


def test_enqueue_stop_sentinel_preserves_existing_queue_order():
    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    stream.queue.put("frame")
    recorder, _ = _recorder_with_stream(stream)

    recorder._enqueue_stop_sentinel(stream)

    assert stream.queue.get_nowait() == "frame"
    assert stream.queue.get_nowait() is None


def test_stop_episode_final_join_happens_before_writer_close():
    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    recorder, events = _recorder_with_stream(stream)
    stream.worker = _FakeWorker(events, "video_worker")
    stream.raw_worker = _FakeWorker(events, "raw_worker")
    stream.metadata_worker = _FakeWorker(events, "metadata_worker")
    stream.writer = _FakeWriter(events)

    stats = recorder.stop_episode()

    names = [event[0] for event in events]
    assert ("video_worker.join", 10.0) in events
    assert ("raw_worker.join", 10.0) in events
    assert ("metadata_worker.join", 10.0) in events
    assert names.index("video_worker.join") < names.index("writer.close")
    assert names.index("raw_worker.join") < names.index("writer.close")
    assert names.index("metadata_worker.join") < names.index("writer.close")
    assert stats["cam0"]["frames_written"] == 0
    assert stream.writer is None
    assert stream.worker is None
    assert stream.raw_worker is None
    assert stream.metadata_worker is None


def test_stop_episode_returns_before_raw_remux(tmp_path, monkeypatch):
    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    recorder, events = _recorder_with_stream(stream)
    stream.worker = _FakeWorker(events, "video_worker")
    stream.raw_worker = _FakeWorker(events, "raw_worker")
    stream.metadata_worker = _FakeWorker(events, "metadata_worker")
    stream.writer = _FakeWriter(events)
    stream.frames_written = 3
    stream.frames_metadata_written = 3
    stream.raw_path = tmp_path / "cam0.mjpeg.tmp"
    stream.mp4_path = tmp_path / "cam0.mp4"
    stream.sidecar_path = tmp_path / "cam0_timestamps.parquet"
    stream.stats_path = tmp_path / "cam0_recorder_stats.json"
    raw_path = stream.raw_path
    mp4_path = stream.mp4_path
    stats_path = stream.stats_path
    raw_path.write_bytes(b"\xff\xd8jpeg" + _JPEG_SENTINEL)

    def fail_if_called(_streams):
        raise AssertionError("STOP must not synchronously remux raw spools")

    monkeypatch.setattr(recorder, "_remux_streams", fail_if_called)

    stats = recorder.stop_episode()

    assert stats["cam0"]["remux_status"] == "pending"
    assert raw_path.exists()
    assert not mp4_path.exists()
    text = stats_path.read_text(encoding="utf-8")
    assert '"remux_status": "pending"' in text


def test_no_drop_callback_enqueues_valid_frame_under_soft_pressure():
    events = []
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder._node = _Node(events)
    recorder._recording_active = threading.Event()
    recorder._recording_active.set()
    recorder._callback_cond = threading.Condition()
    recorder._active_callbacks = 0
    recorder._soft_callback_queue_frames = 1
    recorder._soft_callback_queue_bytes = 1
    recorder._soft_raw_queue_frames = 256
    recorder._soft_raw_queue_bytes = 256 * 1024 * 1024
    recorder._soft_metadata_queue_rows = 16_384

    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    stamp = types.SimpleNamespace(sec=1, nanosec=2)
    msg = types.SimpleNamespace(
        data=bytearray(b"\xff\xd8jpeg"),
        header=types.SimpleNamespace(stamp=stamp),
    )

    recorder._on_frame(stream, msg)

    assert stream.queue.qsize() == 1
    assert stream.frames_received == 1
    assert stream.frames_dropped_queue == 0
    assert stream.pressure_warning_count >= 1
    queued = stream.queue.get_nowait()
    assert queued[0] == b"\xff\xd8jpeg"


def test_idle_callback_updates_camera_monitor_without_recording_stats():
    recorder = _bare_recorder([])
    recorder._recording_active = threading.Event()
    recorder._callback_cond = threading.Condition()
    recorder._active_callbacks = 0
    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    recorder._streams = {"cam0": stream}
    recorder._reset_camera_monitor_state(stream)
    now_ns = recorder._now_ns()
    stamp = types.SimpleNamespace(
        sec=now_ns // 1_000_000_000,
        nanosec=now_ns % 1_000_000_000,
    )
    msg = types.SimpleNamespace(
        data=bytearray(b"\xff\xd8jpeg"),
        header=types.SimpleNamespace(stamp=stamp),
    )

    recorder._on_frame(stream, msg)

    assert stream.queue.empty()
    assert stream.frames_received == 0
    assert recorder.recording_warnings() == []
    snapshot = recorder.camera_monitor_snapshot()
    assert snapshot["names"] == ["cam0"]
    assert snapshot["topics"] == ["/cam0"]
    assert snapshot["seconds_since_last"][0] >= 0
    assert snapshot["status"][0] == 0
    assert snapshot["timestamp_status"][0] == 0


def test_idle_callback_reports_camera_timestamp_skew_in_monitor_only():
    recorder = _bare_recorder([])
    recorder._timestamp_future_warning_ns = 500_000_000
    recorder._timestamp_past_warning_ns = 2_000_000_000
    recorder._recording_active = threading.Event()
    recorder._callback_cond = threading.Condition()
    recorder._active_callbacks = 0
    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    recorder._streams = {"cam0": stream}
    recorder._reset_camera_monitor_state(stream)
    future_ns = recorder._now_ns() + 2_000_000_000
    stamp = types.SimpleNamespace(
        sec=future_ns // 1_000_000_000,
        nanosec=future_ns % 1_000_000_000,
    )
    msg = types.SimpleNamespace(
        data=bytearray(b"\xff\xd8jpeg"),
        header=types.SimpleNamespace(stamp=stamp),
    )

    recorder._on_frame(stream, msg)

    assert stream.queue.empty()
    assert recorder.recording_warnings() == []
    snapshot = recorder.camera_monitor_snapshot()
    assert snapshot["status"] == [2]
    assert snapshot["timestamp_status"] == [2]
    assert snapshot["timestamp_skew_s"][0] > 0.5


def test_timestamp_skew_warnings_are_reported_in_status_and_stats():
    recorder = _bare_recorder([])
    recorder._timestamp_future_warning_ns = 500_000_000
    recorder._timestamp_past_warning_ns = 2_000_000_000
    stream = _CameraStream(name="cam0", topic="/cam0")
    recorder._streams = {"cam0": stream}

    recorder._note_timestamp_skew(stream, 1_900_000_000)
    recorder._note_timestamp_skew(stream, -43_900_000_000_000)

    warnings = recorder.recording_warnings()
    assert warnings == [
        "cam0: camera header timestamp is more than 0.5s ahead of receive time.",
        "cam0: camera header timestamp is more than 2s behind receive time.",
    ]

    stats = recorder._public_stats(stream)
    assert stats["timestamp_warning_count"] == 2
    assert stats["timestamp_future_warning_count"] == 1
    assert stats["timestamp_past_warning_count"] == 1
    assert stats["max_header_ahead_ns"] == 1_900_000_000
    assert stats["max_header_behind_ns"] == 43_900_000_000_000


def test_invalid_jpeg_is_the_only_callback_drop_path():
    events = []
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder._node = _Node(events)
    recorder._recording_active = threading.Event()
    recorder._recording_active.set()
    recorder._callback_cond = threading.Condition()
    recorder._active_callbacks = 0

    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    stamp = types.SimpleNamespace(sec=1, nanosec=2)
    msg = types.SimpleNamespace(
        data=bytearray(b"notjpeg"),
        header=types.SimpleNamespace(stamp=stamp),
    )

    recorder._on_frame(stream, msg)

    assert stream.queue.empty()
    assert stream.frames_received == 1
    assert stream.frames_dropped_invalid == 1
    assert stream.frames_dropped_queue == 0


def test_video_worker_dispatches_to_raw_queue():
    events = []
    recorder = _bare_recorder(events)

    stream = _CameraStream(name="cam0", topic="/cam0", queue=Queue())
    stream.queue.put((
        b"\xff\xd8jpeg",
        1,
        2,
        {
            "callback_enter_ns": 1,
            "header_read_ns": 2,
            "bytes_copy_start_ns": 3,
            "bytes_copy_done_ns": 4,
            "enqueue_start_ns": 5,
            "enqueue_done_ns": 6,
            "queue_size_before": 0,
            "queue_size_after": 1,
            "frame_size_bytes": 6,
        },
    ))
    stream.queue.put(None)

    recorder._video_worker_loop(stream)

    data, metadata = stream.raw_queue.get_nowait()
    assert data == b"\xff\xd8jpeg"
    assert metadata["frame_index"] == 0
    assert metadata["video_dequeue_ns"] >= metadata["enqueue_done_ns"]


def test_raw_worker_writes_spool_and_enqueues_metadata(tmp_path):
    events = []
    recorder = _bare_recorder(events)

    raw_path = tmp_path / "cam0.mjpeg.tmp"
    stream = _CameraStream(name="cam0", topic="/cam0")
    stream.raw_path = raw_path
    stream.raw_fd = os.open(raw_path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    stream.raw_queue.put((
        b"\xff\xd8jpeg",
        {
            "frame_index": 0,
            "header_stamp_ns": 1,
            "recv_ns": 2,
            "callback_enter_ns": 1,
            "header_read_ns": 2,
            "bytes_copy_start_ns": 3,
            "bytes_copy_done_ns": 4,
            "enqueue_start_ns": 5,
            "enqueue_done_ns": 6,
            "video_dequeue_ns": 7,
            "queue_size_before": 0,
            "queue_size_after": 1,
            "frame_size_bytes": 6,
        },
    ))
    stream.raw_queue.put(None)

    recorder._raw_worker_loop(stream)

    assert raw_path.read_bytes() == b"\xff\xd8jpeg" + _JPEG_SENTINEL
    assert stream.raw_fd is None
    assert stream.frames_written == 1
    metadata = stream.metadata_queue.get_nowait()
    assert metadata["frame_index"] == 0
    assert metadata["raw_dequeue_ns"] >= metadata["video_dequeue_ns"]
    assert metadata["raw_write_done_ns"] >= metadata["raw_write_start_ns"]


def test_metadata_worker_flush_holds_writer_lock(monkeypatch):
    events = []
    recorder = _bare_recorder(events)
    monkeypatch.setattr(
        video_recorder_module.pa,
        "array",
        lambda values, type=None: list(values),
    )
    monkeypatch.setattr(
        video_recorder_module.pa,
        "table",
        lambda data, schema=None: {"data": data, "schema": schema},
    )

    stream = _CameraStream(name="cam0", topic="/cam0", metadata_queue=Queue())

    class Writer:
        def write_table(self, table):
            events.append(("writer.locked", stream.writer_lock.locked()))

    class DiagnosticsWriter:
        def write_table(self, table):
            events.append((
                "diagnostics_writer.locked",
                stream.diagnostics_writer_lock.locked(),
            ))

    stream.writer = Writer()
    stream.diagnostics_writer = DiagnosticsWriter()
    stream.metadata_queue.put({
        "frame_index": 0,
        "header_stamp_ns": 1,
        "recv_ns": 2,
        "callback_enter_ns": 1,
        "header_read_ns": 2,
        "bytes_copy_start_ns": 3,
        "bytes_copy_done_ns": 4,
        "enqueue_start_ns": 5,
        "enqueue_done_ns": 6,
        "video_dequeue_ns": 7,
        "raw_dequeue_ns": 8,
        "raw_write_start_ns": 9,
        "raw_write_done_ns": 10,
        "metadata_enqueue_start_ns": 10,
        "metadata_enqueue_done_ns": 10,
        "metadata_queue_size_before": 0,
        "metadata_queue_size_after": 1,
        "queue_size_before": 0,
        "queue_size_after": 1,
        "frame_size_bytes": 6,
    })
    stream.metadata_queue.put(None)

    recorder._metadata_worker_loop(stream)

    assert ("writer.locked", True) in events
    assert ("diagnostics_writer.locked", True) in events
    assert stream.frames_metadata_written == 1


def test_remux_success_replaces_mp4_and_deletes_raw(tmp_path, monkeypatch):
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder._node = _Node([])
    recorder._ffmpeg_bin = "ffmpeg"
    recorder._ffprobe_bin = "ffprobe"
    recorder._framerate_hint = 30

    raw_path = tmp_path / "cam0.mjpeg.tmp"
    raw_path.write_bytes(b"\xff\xd8jpeg" + _JPEG_SENTINEL)
    mp4_path = tmp_path / "cam0.mp4"
    stream = _CameraStream(name="cam0", topic="/cam0")
    stream.raw_path = raw_path
    stream.mp4_path = mp4_path
    stream.frames_written = 1

    def fake_run(cmd, stdout, stderr, check=False, **kwargs):
        Path(cmd[-1]).write_bytes(b"mp4")
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(video_recorder_module.subprocess, "run", fake_run)
    monkeypatch.setattr(recorder, "_probe_frame_count", lambda path: 1)

    recorder._remux_raw_spool(stream)

    assert mp4_path.read_bytes() == b"mp4"
    assert not raw_path.exists()
    assert stream.remux_error is None


def test_remux_mismatch_preserves_raw_spool(tmp_path, monkeypatch):
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder._node = _Node([])
    recorder._ffmpeg_bin = "ffmpeg"
    recorder._ffprobe_bin = "ffprobe"
    recorder._framerate_hint = 30

    raw_path = tmp_path / "cam0.mjpeg.tmp"
    raw_path.write_bytes(b"\xff\xd8jpeg" + _JPEG_SENTINEL)
    mp4_path = tmp_path / "cam0.mp4"
    stream = _CameraStream(name="cam0", topic="/cam0")
    stream.raw_path = raw_path
    stream.mp4_path = mp4_path
    stream.frames_written = 1

    def fake_run(cmd, stdout, stderr, check=False, **kwargs):
        Path(cmd[-1]).write_bytes(b"mp4")
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(video_recorder_module.subprocess, "run", fake_run)
    monkeypatch.setattr(recorder, "_probe_frame_count", lambda path: 0)

    recorder._remux_raw_spool(stream)

    assert raw_path.exists()
    assert not mp4_path.exists()
    assert "frame count mismatch" in stream.remux_error


def test_summary_stats_json_is_written_by_default(tmp_path):
    recorder = _bare_recorder([])
    recorder._diagnostics_mode = "summary"
    stream = _CameraStream(name="cam0", topic="/cam0")
    stream.stats_path = tmp_path / "cam0_recorder_stats.json"
    stream.mp4_path = tmp_path / "cam0.mp4"
    stream.raw_path = tmp_path / "cam0.mjpeg.tmp"
    stream.sidecar_path = tmp_path / "cam0_timestamps.parquet"
    with stream.state_lock:
        stream.frames_received = 3
        stream.frames_written = 3
        stream.frames_metadata_written = 3
        stream.frames_remuxed = 3
        stream.max_raw_queue_bytes = 42
        stream.pressure_warning_count = 1

    recorder._write_stats_json(stream)

    text = stream.stats_path.read_text(encoding="utf-8")
    assert '"frames_written": 3' in text
    assert '"pressure_warning_count": 1' in text
    assert '"max_raw_queue_bytes": 42' in text


def test_diagnostics_off_skips_stats_json(tmp_path):
    recorder = _bare_recorder([])
    recorder._diagnostics_mode = "off"
    stream = _CameraStream(name="cam0", topic="/cam0")
    stream.stats_path = tmp_path / "cam0_recorder_stats.json"

    recorder._write_stats_json(stream)

    assert not stream.stats_path.exists()


def test_diagnostics_env_defaults_enabled(monkeypatch):
    monkeypatch.delenv("CYCLO_VIDEO_RECORDER_DIAGNOSTICS", raising=False)
    assert _resolve_diagnostics_mode() == "summary"
    assert _resolve_diagnostics_enabled() is False

    monkeypatch.setenv("CYCLO_VIDEO_RECORDER_DIAGNOSTICS", "0")
    assert _resolve_diagnostics_mode() == "off"
    assert _resolve_diagnostics_enabled() is False

    monkeypatch.setenv("CYCLO_VIDEO_RECORDER_DIAGNOSTICS", "false")
    assert _resolve_diagnostics_mode() == "off"
    assert _resolve_diagnostics_enabled() is False

    monkeypatch.setenv("CYCLO_VIDEO_RECORDER_DIAGNOSTICS", "1")
    assert _resolve_diagnostics_mode() == "detailed"
    assert _resolve_diagnostics_enabled() is True
