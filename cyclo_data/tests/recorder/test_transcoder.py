# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exhaustive tests for the background MJPEG → H.264 transcoder.

These tests are designed to surface the exception paths the user
specifically asked about: sidecar/MP4 mismatches, mid-flight crashes,
back-to-back submits, missing inputs, etc. Tests use ffmpeg to build
small synthetic MJPEG MP4s + parquet sidecars so they're hermetic and
runnable in the cyclo_intelligence docker image without any robot
hardware.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


# The tests import from cyclo_data — make the source tree importable
# when running outside the colcon install (e.g. ``pytest`` on host).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))

from cyclo_data.recorder.transcoder import (  # noqa: E402
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_NOT_REQUIRED,
    STATUS_PENDING,
    TranscodeWorker,
    _detect_encoder,
    _mp4_codec_name,
    _mp4_dimensions,
    _mp4_frame_count,
    _patch_status,
)
import cyclo_data.recorder.transcoder as transcoder_module  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def encoder():
    """Probe the H.264 encoder once for the whole session."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    try:
        return _detect_encoder()
    except FileNotFoundError as exc:
        pytest.skip(f"ffmpeg is not installed: {exc}")


def _make_mjpeg_mp4(path: Path, num_frames: int, *, w: int = 64, h: int = 48) -> None:
    """Build a tiny MJPEG-in-MP4 with ``num_frames`` solid-colour frames."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    if num_frames == 0:
        # ffmpeg can't make a zero-frame mp4 — emit an empty file. Callers
        # that pass 0 expect "no transcode needed".
        path.write_bytes(b"")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        try:
            from PIL import Image
        except Exception as exc:
            pytest.skip(f"PIL unavailable: {exc}")
        for i in range(num_frames):
            arr = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)
            Image.fromarray(arr).save(tdir / f"f_{i:06d}.jpg", quality=80)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "30", "-i", str(tdir / "f_%06d.jpg"),
            "-c:v", "copy", str(path),
        ]
        subprocess.run(cmd, check=True)


def _make_sidecar(path: Path, num_rows: int, *, fps: int = 30) -> None:
    """Write a parquet sidecar with monotonic timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    step_ns = 1_000_000_000 // fps
    base_ns = 1_000_000_000_000
    table = pa.table({
        "frame_index": pa.array(list(range(num_rows)), type=pa.int32()),
        "header_stamp_ns": pa.array(
            [base_ns + i * step_ns for i in range(num_rows)], type=pa.int64(),
        ),
        "recv_ns": pa.array(
            [base_ns + i * step_ns for i in range(num_rows)], type=pa.int64(),
        ),
    })
    pq.write_table(table, path)


def _make_episode(
    root: Path,
    cameras: dict[str, tuple[int, int]],
    *,
    write_info: bool = True,
    initial_status: str = STATUS_PENDING,
    rotations: dict[str, int] | None = None,
) -> Path:
    """Materialise an episode directory.

    ``cameras`` maps ``cam_name`` → ``(mp4_frames, sidecar_rows)`` so the
    caller can deliberately introduce mismatches. ``rotations`` is
    optional ``{cam_name: degrees}``.
    """
    ep = Path(root) / "Task_X" / "0"
    videos_dir = ep / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for cam, (mp4_frames, sidecar_rows) in cameras.items():
        _make_mjpeg_mp4(videos_dir / f"{cam}.mp4", mp4_frames)
        _make_sidecar(videos_dir / f"{cam}_timestamps.parquet", sidecar_rows)
    if write_info:
        info = {
            "task_instruction": "test",
            "robot_type": "test_robot",
            "episode_index": 0,
            "format_version": "robotis_v2",
            "transcoding_status": initial_status,
        }
        (ep / "episode_info.json").write_text(json.dumps(info, indent=2))
    if rotations:
        camera_info_dir = ep / "camera_info"
        camera_info_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": "robotis_camera_metadata_v1",
            "source": "robot_config",
            "cameras": {
                cam: {
                    "rotation_deg": int(deg),
                    "rotation_applied_at": "record",
                }
                for cam, deg in rotations.items()
            },
        }
        (camera_info_dir / "camera_metadata.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    return ep


def _make_nested_episode(
    root: Path,
    segments: dict[str, dict[str, tuple[int, int]]],
    *,
    write_info: bool = True,
    initial_status: str = STATUS_PENDING,
    rotations: dict[str, int] | None = None,
) -> Path:
    """Materialise a segmented archived episode.

    ``segments`` maps ``segment_name`` → ``{cam_name: (mp4_frames, sidecar_rows)}``.
    """
    ep = Path(root) / "Task_X" / "0"
    for segment, cameras in segments.items():
        videos_dir = ep / "videos" / segment
        videos_dir.mkdir(parents=True, exist_ok=True)
        for cam, (mp4_frames, sidecar_rows) in cameras.items():
            _make_mjpeg_mp4(videos_dir / f"{cam}.mp4", mp4_frames)
            _make_sidecar(videos_dir / f"{cam}_timestamps.parquet", sidecar_rows)
    if write_info:
        info = {
            "task_instruction": "test",
            "robot_type": "test_robot",
            "episode_index": 0,
            "format_version": "robotis_v2",
            "transcoding_status": initial_status,
        }
        (ep / "episode_info.json").write_text(json.dumps(info, indent=2))
    if rotations:
        camera_info_dir = ep / "camera_info"
        camera_info_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": "robotis_camera_metadata_v1",
            "source": "robot_config",
            "cameras": {
                cam: {
                    "rotation_deg": int(deg),
                    "rotation_applied_at": "record",
                }
                for cam, deg in rotations.items()
            },
        }
        (camera_info_dir / "camera_metadata.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    return ep


def _read_status(episode_dir: Path) -> dict:
    return json.loads((episode_dir / "episode_info.json").read_text())


def test_patch_status_preserves_korean_instruction_as_utf8(tmp_path):
    info_path = tmp_path / "episode_info.json"
    info_path.write_text(
        json.dumps(
            {
                "subtask_instruction": "화장품 집기",
                "transcoding_status": STATUS_PENDING,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _patch_status(info_path, STATUS_DONE)

    raw = info_path.read_text(encoding="utf-8")
    assert "화장품 집기" in raw
    assert "\\ud654" not in raw


def test_ffprobe_helpers_handle_blank_stdout_without_index_error(monkeypatch, tmp_path):
    blank = subprocess.CompletedProcess(
        args=["ffprobe"], returncode=0, stdout="\n  \n",
    )
    monkeypatch.setattr(
        transcoder_module.subprocess,
        "run",
        lambda *args, **kwargs: blank,
    )

    with pytest.raises(RuntimeError, match="could not determine codec"):
        _mp4_codec_name(tmp_path / "blank.mp4")
    with pytest.raises(RuntimeError, match="could not determine dimensions"):
        _mp4_dimensions(tmp_path / "blank.mp4")


def _ffprobe_codec(mp4: Path) -> str:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
            str(mp4),
        ],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def test_pending_raw_spool_remuxes_to_mp4_before_h264(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    raw = videos / "cam0.mjpeg.tmp"
    raw.write_bytes(b"\xff\xd8jpeg")
    (videos / "cam0_timestamps.parquet").write_bytes(b"sidecar")
    stats = videos / "cam0_recorder_stats.json"
    stats.write_text(
        json.dumps({
            "frames_written": 3,
            "first_recv_ns": 1_000_000_000,
            "last_recv_ns": 1_100_000_000,
            "remux_status": STATUS_PENDING,
        }),
        encoding="utf-8",
    )

    def fake_run(cmd, stdout, stderr, check=False, **kwargs):
        Path(cmd[-1]).write_bytes(b"mjpeg-mp4")
        return type("Result", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(transcoder_module.subprocess, "run", fake_run)
    monkeypatch.setattr(transcoder_module, "_mp4_frame_count", lambda _path: 3)
    monkeypatch.setattr(transcoder_module, "_sidecar_row_count", lambda _path: 3)
    monkeypatch.setattr(
        transcoder_module,
        "_estimate_raw_framerate",
        lambda _sidecar, _stats: 30.0,
    )

    worker = TranscodeWorker(logger=None, parallelism=1)
    try:
        failed = worker._remux_pending_raw_spools(videos)
    finally:
        worker.shutdown(wait=True)

    assert failed == {}
    assert not raw.exists()
    assert (videos / "cam0.mp4").read_bytes() == b"mjpeg-mp4"
    updated = json.loads(stats.read_text())
    assert updated["remux_status"] == STATUS_DONE
    assert updated["frames_remuxed"] == 3
    assert updated["remux_error"] is None


def test_mp4_frame_count_uses_fast_container_counts(monkeypatch, tmp_path):
    mp4 = tmp_path / "large.mp4"
    mp4.write_bytes(b"placeholder")
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        if "-show_entries" in cmd and "stream=nb_frames" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="N/A\n", stderr="")
        if "-show_entries" in cmd and "stream=nb_read_packets" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="16502\n", stderr="")
        raise AssertionError("slow count_frames fallback should not be used")

    monkeypatch.setattr(transcoder_module.subprocess, "run", fake_run)

    assert _mp4_frame_count(mp4) == 16502
    assert any("stream=nb_read_packets" in call for call in calls)
    assert not any("stream=nb_read_frames" in call for call in calls)


def test_raw_remux_probe_failure_cleans_tmp_and_marks_failed(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    raw = videos / "cam0.mjpeg.tmp"
    raw.write_bytes(b"\xff\xd8jpeg")
    tmp_mp4 = videos / "cam0.remuxing.mp4"
    stats = videos / "cam0_recorder_stats.json"
    stats.write_text(
        json.dumps({
            "frames_written": 3,
            "remux_status": STATUS_PENDING,
        }),
        encoding="utf-8",
    )
    (videos / "cam0_timestamps.parquet").write_bytes(b"sidecar")

    def fake_run(cmd, stdout, stderr, check=False, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial mp4")
        return type("Result", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(transcoder_module.subprocess, "run", fake_run)
    monkeypatch.setattr(transcoder_module, "_sidecar_row_count", lambda _path: 3)
    monkeypatch.setattr(
        transcoder_module,
        "_estimate_raw_framerate",
        lambda _sidecar, _stats: 30.0,
    )

    def probe_fails(_path):
        raise RuntimeError("ffprobe frame-count timed out")

    monkeypatch.setattr(transcoder_module, "_mp4_frame_count", probe_fails)

    worker = TranscodeWorker(logger=None, parallelism=1)
    try:
        failed = worker._remux_pending_raw_spools(videos)
    finally:
        worker.shutdown(wait=True)

    assert "cam0" in failed
    assert raw.exists()
    assert not tmp_mp4.exists()
    updated = json.loads(stats.read_text())
    assert updated["remux_status"] == STATUS_FAILED
    assert "ffprobe frame-count timed out" in updated["remux_error"]


def test_raw_remux_failure_is_failed_not_not_required(tmp_path, monkeypatch):
    ep = tmp_path / "Task_X" / "0"
    videos = ep / "videos"
    videos.mkdir(parents=True)
    raw = videos / "cam0.mjpeg.tmp"
    raw.write_bytes(b"\xff\xd8jpeg")
    (videos / "cam0_timestamps.parquet").write_bytes(b"sidecar")
    (ep / "episode_info.json").write_text(
        json.dumps({
            "transcoding_status": STATUS_PENDING,
            "video_remux_status": STATUS_PENDING,
        })
    )

    def fake_run(cmd, stdout, stderr, check=False, **kwargs):
        return type("Result", (), {"returncode": 1, "stderr": b"bad mjpeg"})()

    monkeypatch.setattr(transcoder_module.subprocess, "run", fake_run)
    monkeypatch.setattr(transcoder_module, "_sidecar_row_count", lambda _path: 1)
    monkeypatch.setattr(
        transcoder_module,
        "_estimate_raw_framerate",
        lambda _sidecar, _stats: 30.0,
    )

    worker = TranscodeWorker(logger=None, parallelism=1)
    worker._encoder = ("libx264", [])
    try:
        result = worker._run_one(ep)
    finally:
        worker.shutdown(wait=True)

    assert not result.success
    assert raw.exists()
    info = _read_status(ep)
    assert info["transcoding_status"] == STATUS_FAILED
    assert info["video_remux_status"] == STATUS_FAILED
    assert "cam0" in info["transcoding_cameras_failed"]
    assert info["transcoding_status"] != STATUS_NOT_REQUIRED


def test_remux_failure_stops_before_h264_encode(tmp_path, monkeypatch):
    ep = _make_episode(tmp_path, {"cam_ok": (3, 3)})
    raw = ep / "videos" / "cam_raw.mjpeg.tmp"
    raw.write_bytes(b"\xff\xd8jpeg")
    _make_sidecar(ep / "videos" / "cam_raw_timestamps.parquet", 1)

    worker = TranscodeWorker(logger=None, parallelism=1)
    worker._encoder = ("libx264", [])

    monkeypatch.setattr(
        worker,
        "_remux_pending_raw_spools",
        lambda _videos_dir: {"cam_raw": "RuntimeError('bad raw')"},
    )

    def should_not_encode(*args, **kwargs):
        raise AssertionError("H.264 encode should wait until remux succeeds")

    monkeypatch.setattr(worker, "_transcode_camera", should_not_encode)

    try:
        result = worker._run_one(ep)
    finally:
        worker.shutdown(wait=True)

    assert not result.success
    assert result.cameras_failed == {"cam_raw": "RuntimeError('bad raw')"}
    info = _read_status(ep)
    assert info["transcoding_status"] == STATUS_FAILED
    assert info["video_remux_status"] == STATUS_FAILED


def test_video_remux_status_done_before_h264_encode(tmp_path, monkeypatch):
    ep = _make_episode(tmp_path, {"cam0": (3, 3)})
    worker = TranscodeWorker(logger=None, parallelism=1)
    worker._encoder = ("libx264", [])

    monkeypatch.setattr(worker, "_remux_pending_raw_spools", lambda _videos_dir: {})

    def assert_status_before_encode(*args, **kwargs):
        info = _read_status(ep)
        assert info["transcoding_status"] == "running"
        assert info["video_remux_status"] == STATUS_DONE

    monkeypatch.setattr(worker, "_transcode_camera", assert_status_before_encode)

    try:
        result = worker._run_one(ep)
    finally:
        worker.shutdown(wait=True)

    assert result.success
    assert _read_status(ep)["video_remux_status"] == STATUS_DONE


@pytest.fixture
def worker(encoder):
    w = TranscodeWorker(logger=None, parallelism=2)
    yield w
    w.shutdown(wait=True)


# ----------------------------------------------------------------------
# A — Happy path
# ----------------------------------------------------------------------


class TestHappyPath:
    """A1-A3: the obvious success cases."""

    def test_a1_single_camera_normal(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (30, 30)})
        res = worker.submit(ep).result(timeout=60)
        assert res.success, res
        assert res.cameras_done == ["cam0"]
        assert res.cameras_failed == {}
        # File replaced in place, still named cam0.mp4
        mp4 = ep / "videos" / "cam0.mp4"
        assert mp4.exists()
        assert _ffprobe_codec(mp4) == "h264"
        assert _mp4_frame_count(mp4) == 30
        # Status updated
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_DONE
        assert info["transcoding_cameras_failed"] == {}
        # No orphan .tmp left behind
        assert list((ep / "videos").glob("*.h264.tmp")) == []

    def test_a2_multiple_cameras(self, tmp_path, worker):
        ep = _make_episode(
            tmp_path,
            {f"cam{i}": (20, 20) for i in range(4)},
        )
        res = worker.submit(ep).result(timeout=120)
        assert res.success
        assert sorted(res.cameras_done) == ["cam0", "cam1", "cam2", "cam3"]
        for cam in res.cameras_done:
            mp4 = ep / "videos" / f"{cam}.mp4"
            assert _ffprobe_codec(mp4) == "h264"
            assert _mp4_frame_count(mp4) == 20

    def test_a4_rotation_270_applied_to_wrist_cam(self, tmp_path, worker):
        """rotation_deg=270 should swap width/height in the H.264 output."""
        ep = _make_episode(
            tmp_path,
            {"cam_wrist": (20, 20)},
            rotations={"cam_wrist": 270},
        )
        res = worker.submit(ep).result(timeout=60)
        assert res.success, res
        # Source MP4 was 64x48 (w x h). After 270° rotation the output
        # should be 48x64 (w x h swapped).
        out = ep / "videos" / "cam_wrist.mp4"
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                str(out),
            ],
            capture_output=True, text=True,
        )
        w, h = probe.stdout.strip().split(",")
        assert int(w) == 48 and int(h) == 64, (
            f"expected 48x64 after rotation, got {w}x{h}"
        )

    def test_a4b_retry_skips_already_h264_camera(self, tmp_path, worker):
        """Retrying a partially transcoded episode must not rotate twice."""
        ep = _make_episode(
            tmp_path,
            {"cam_wrist": (20, 20)},
            rotations={"cam_wrist": 270},
        )
        first = worker.submit(ep).result(timeout=60)
        assert first.success, first
        second = worker.submit(ep).result(timeout=60)
        assert second.success, second

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                str(ep / "videos" / "cam_wrist.mp4"),
            ],
            capture_output=True, text=True,
        )
        w, h = probe.stdout.strip().split(",")
        assert int(w) == 48 and int(h) == 64, (
            f"retry should keep the first rotation result, got {w}x{h}"
        )

    def test_a5_rotation_0_no_change(self, tmp_path, worker):
        """rotation_deg=0 (or missing) must leave dimensions intact."""
        ep = _make_episode(
            tmp_path, {"cam_head": (20, 20)}, rotations={"cam_head": 0},
        )
        res = worker.submit(ep).result(timeout=60)
        assert res.success
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                str(ep / "videos" / "cam_head.mp4"),
            ],
            capture_output=True, text=True,
        )
        w, h = probe.stdout.strip().split(",")
        assert int(w) == 64 and int(h) == 48

    def test_a3_encoder_detection_works(self, encoder):
        name, opts = encoder
        # Whatever we got, it must be H.264-class and runnable.
        assert "264" in name or name == "libx264"
        # And the cached value is stable.
        again = _detect_encoder()
        assert again == encoder

    def test_a6_nested_segment_camera_transcodes_in_place(self, tmp_path, worker):
        ep = _make_nested_episode(tmp_path, {"0_0": {"cam0": (12, 12)}})
        res = worker.submit(ep).result(timeout=60)
        assert res.success, res
        assert res.cameras_done == ["0_0/cam0"]
        mp4 = ep / "videos" / "0_0" / "cam0.mp4"
        assert _ffprobe_codec(mp4) == "h264"
        assert _mp4_frame_count(mp4) == 12
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_DONE
        assert info["transcoding_cameras_failed"] == {}
        assert "transcoding_cameras_done" not in info
        assert "transcoding_cameras_skipped" not in info
        assert "transcoding_elapsed_sec" not in info
        assert "transcoding_updated" not in info

    def test_a7_nested_multiple_segments_and_cameras(self, tmp_path, worker):
        ep = _make_nested_episode(
            tmp_path,
            {
                "0_0": {"cam_left": (10, 10), "cam_right": (10, 10)},
                "0_1": {"cam_left": (8, 8), "cam_right": (8, 8)},
            },
        )
        res = worker.submit(ep).result(timeout=120)
        assert res.success, res
        expected = ["0_0/cam_left", "0_0/cam_right", "0_1/cam_left", "0_1/cam_right"]
        assert sorted(res.cameras_done) == expected
        for camera_id in expected:
            segment, cam = camera_id.split("/", 1)
            assert _ffprobe_codec(ep / "videos" / segment / f"{cam}.mp4") == "h264"


# ----------------------------------------------------------------------
# B — Edge cases
# ----------------------------------------------------------------------


class TestEdgeCases:
    """B1-B6: mismatch, empty, missing, corrupted."""

    def test_b1_sidecar_one_more_than_mp4(self, tmp_path, worker):
        """Classic EOI-missing scenario: parquet has 1 row more than MP4."""
        ep = _make_episode(tmp_path, {"cam0": (29, 30)})
        res = worker.submit(ep).result(timeout=60)
        assert res.success, res
        assert _ffprobe_codec(ep / "videos" / "cam0.mp4") == "h264"

    def test_b1b_sidecar_two_more_rejects(self, tmp_path, worker):
        """Two-frame deficit exceeds the tolerance → transcode fails."""
        ep = _make_episode(tmp_path, {"cam0": (28, 30)})
        res = worker.submit(ep).result(timeout=60)
        assert not res.success
        assert "cam0" in res.cameras_failed
        # Raw MP4 must still be intact (MJPEG).
        assert _ffprobe_codec(ep / "videos" / "cam0.mp4") == "mjpeg"
        # Status reflects the failure with diagnostic context.
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_FAILED
        assert "cam0" in info["transcoding_cameras_failed"]

    def test_b2_empty_episode(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (0, 0)})
        res = worker.submit(ep).result(timeout=60)
        assert res.success
        # Raw MP4 was empty bytes; transcoder deletes it.
        assert not (ep / "videos" / "cam0.mp4").exists()

    def test_b3_missing_sidecar(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (10, 10)})
        # Remove sidecar AFTER creation.
        (ep / "videos" / "cam0_timestamps.parquet").unlink()
        # Discovery requires the sidecar to exist, so this maps to
        # "no cameras to transcode" → not_required.
        res = worker.submit(ep).result(timeout=60)
        assert res.success
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_NOT_REQUIRED

    def test_b4_one_frame_episode(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (1, 1)})
        res = worker.submit(ep).result(timeout=60)
        assert res.success
        assert _ffprobe_codec(ep / "videos" / "cam0.mp4") == "h264"
        assert _mp4_frame_count(ep / "videos" / "cam0.mp4") == 1

    def test_b5_corrupt_mp4(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (10, 10)})
        # Truncate the MP4 to garbage so ffmpeg can't read it.
        (ep / "videos" / "cam0.mp4").write_bytes(b"not a valid mp4")
        res = worker.submit(ep).result(timeout=60)
        assert not res.success
        assert "cam0" in res.cameras_failed
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_FAILED

    def test_b6_nested_camera_failure_preserves_raw(self, tmp_path, worker):
        ep = _make_nested_episode(tmp_path, {"0_0": {"cam0": (5, 30)}})
        res = worker.submit(ep).result(timeout=60)
        assert not res.success
        assert "0_0/cam0" in res.cameras_failed
        assert _ffprobe_codec(ep / "videos" / "0_0" / "cam0.mp4") == "mjpeg"
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_FAILED
        assert "0_0/cam0" in info["transcoding_cameras_failed"]


# ----------------------------------------------------------------------
# C — Recovery
# ----------------------------------------------------------------------


class TestRecovery:
    """C1-C4: orphan files, resume scan, failed retry."""

    def test_c1_orphan_h264_tmp_cleaned_on_retry(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (15, 15)})
        # Drop an orphan .h264.tmp into videos/ as if a previous run
        # crashed mid-encode.
        orphan = ep / "videos" / "cam0.h264.tmp"
        orphan.write_bytes(b"garbage")
        res = worker.submit(ep).result(timeout=60)
        assert res.success
        # Orphan must be gone, real transcode succeeded.
        assert not orphan.exists()
        assert _ffprobe_codec(ep / "videos" / "cam0.mp4") == "h264"

    def test_c2_resume_pending_picks_up_pending(self, tmp_path):
        # Two pending episodes laid out under a fake workspace.
        for ep_idx in (0, 1):
            if ep_idx == 0:
                videos_dir = tmp_path / "Task_X" / str(ep_idx) / "videos"
            else:
                videos_dir = tmp_path / "Task_X" / str(ep_idx) / "videos" / "1_0"
            videos_dir.mkdir(parents=True, exist_ok=True)
            _make_mjpeg_mp4(videos_dir / "cam0.mp4", 10)
            _make_sidecar(videos_dir / "cam0_timestamps.parquet", 10)
            (tmp_path / "Task_X" / str(ep_idx) / "episode_info.json").write_text(
                json.dumps({
                    "transcoding_status": STATUS_PENDING,
                })
            )
        # An episode already marked done must NOT be re-queued.
        done_ep = tmp_path / "Task_X" / "2"
        (done_ep / "videos").mkdir(parents=True)
        (done_ep / "episode_info.json").write_text(
            json.dumps({"transcoding_status": STATUS_DONE})
        )

        worker = TranscodeWorker(logger=None, parallelism=2)
        try:
            futs = worker.submit_pending_recovery(tmp_path)
            assert len(futs) == 2  # 0 and 1, not 2
            for fut in futs:
                res = fut.result(timeout=60)
                assert res.success
        finally:
            worker.shutdown(wait=True)

    def test_c3_failed_status_preserves_raw(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (5, 30)})  # huge mismatch
        worker.submit(ep).result(timeout=60)
        # Raw MJPEG must survive a failed transcode.
        assert _ffprobe_codec(ep / "videos" / "cam0.mp4") == "mjpeg"
        info = _read_status(ep)
        assert info["transcoding_status"] == STATUS_FAILED
        assert info["transcoding_cameras_failed"]

    def test_c4_failed_then_resubmit_idempotent(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (10, 10)})
        first = worker.submit(ep).result(timeout=60)
        assert first.success
        # A second submit of an already-done episode should re-run cleanly
        # (the raw MJPEG is gone, but the H.264 file is now the "raw" the
        # next pass reads — it should detect codec mismatch via verify).
        # We expect it to either re-encode successfully or detect a
        # consistent state; never to corrupt files.
        second = worker.submit(ep).result(timeout=60)
        # In either case, the MP4 stays H.264 and parseable.
        mp4 = ep / "videos" / "cam0.mp4"
        assert _ffprobe_codec(mp4) == "h264"
        assert _mp4_frame_count(mp4) == 10


# ----------------------------------------------------------------------
# D — Concurrency
# ----------------------------------------------------------------------


class TestConcurrency:
    """D1-D3: idempotent submit, rapid submit, race with stop."""

    def test_d1_submit_same_episode_twice_dedupes(self, tmp_path, worker):
        ep = _make_episode(tmp_path, {"cam0": (10, 10)})
        f1 = worker.submit(ep)
        f2 = worker.submit(ep)
        # Both call sites must observe the same in-flight Future.
        assert f1 is f2
        res = f1.result(timeout=60)
        assert res.success

    def test_d2_submit_many_in_a_row(self, tmp_path):
        # Use parallelism=1 to force serial draining and verify the queue
        # never deadlocks or drops jobs.
        worker = TranscodeWorker(logger=None, parallelism=1)
        try:
            eps = []
            futures = []
            for i in range(5):
                ep = _make_episode(
                    tmp_path / f"task_{i}",
                    {"cam0": (8, 8)},
                )
                eps.append(ep)
                futures.append(worker.submit(ep))
            for fut in futures:
                assert fut.result(timeout=60).success
            for ep in eps:
                assert _ffprobe_codec(ep / "videos" / "cam0.mp4") == "h264"
        finally:
            worker.shutdown(wait=True)

    def test_d3_shutdown_drains_inflight(self, tmp_path):
        worker = TranscodeWorker(logger=None, parallelism=2)
        ep = _make_episode(tmp_path, {"cam0": (20, 20)})
        fut = worker.submit(ep)
        worker.shutdown(wait=True)
        # The shutdown waited, so the job completed.
        assert fut.done()
        assert fut.result().success

    def test_d4_shutdown_is_idempotent(self):
        worker = TranscodeWorker(logger=None, parallelism=1)
        worker.shutdown(wait=False)
        worker.shutdown(wait=False)
        with pytest.raises(RuntimeError, match="shut down"):
            worker.submit(Path("/tmp/nonexistent_episode"))
