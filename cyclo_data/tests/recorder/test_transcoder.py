# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

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
    _mp4_frame_count,
    _patch_status,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def encoder():
    """Probe the H.264 encoder once for the whole session."""
    return _detect_encoder()


def _make_mjpeg_mp4(path: Path, num_frames: int, *, w: int = 64, h: int = 48) -> None:
    """Build a tiny MJPEG-in-MP4 with ``num_frames`` solid-colour frames."""
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
            "recorder_format_version": 2,
            "video_files": {cam: f"videos/{cam}.mp4" for cam in cameras},
            "camera_rotations": dict(rotations or {}),
            "transcoding_status": initial_status,
        }
        (ep / "episode_info.json").write_text(json.dumps(info, indent=2))
    return ep


def _read_status(episode_dir: Path) -> dict:
    return json.loads((episode_dir / "episode_info.json").read_text())


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
        assert info["transcoding_encoder"] == res.encoder
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
            (tmp_path / "Task_X" / str(ep_idx) / "videos").mkdir(
                parents=True, exist_ok=True
            )
            _make_mjpeg_mp4(
                tmp_path / "Task_X" / str(ep_idx) / "videos" / "cam0.mp4", 10,
            )
            _make_sidecar(
                tmp_path / "Task_X" / str(ep_idx) / "videos"
                / "cam0_timestamps.parquet", 10,
            )
            (tmp_path / "Task_X" / str(ep_idx) / "episode_info.json").write_text(
                json.dumps({
                    "transcoding_status": STATUS_PENDING,
                    "video_files": {"cam0": "videos/cam0.mp4"},
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
