# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Background MJPEG-to-H.264 transcoder for recording format v2.

Recording writes per-camera MJPEG MP4s for near-zero live CPU/GPU cost.
After STOP, this module re-encodes each camera's MP4 to H.264 in a
worker pool so the final on-disk format is universally playable
(VSCode webview / browser <video>) and ~20x smaller.

Crash-safety story
------------------

For every camera, the worker:
  1. encodes ``<cam>.mp4`` → ``<cam>.h264.tmp`` (raw stays intact)
  2. verifies the new file's frame count against the sidecar
  3. atomically renames ``<cam>.h264.tmp`` → ``<cam>.mp4``
     (POSIX ``rename`` overwrites the raw in one step)

So at every moment exactly one of these holds:
  * only ``<cam>.mp4`` (MJPEG, transcode pending)
  * ``<cam>.mp4`` (MJPEG) + ``<cam>.h264.tmp`` (incomplete encode)
  * only ``<cam>.mp4`` (H.264, done)

The orphan ``.h264.tmp`` left by a crash mid-encode is detected on
service start and discarded before retry. ``episode_info.json``
records ``transcoding_status`` so the orchestrator UI and the
downstream LeRobot converter know whether the episode is ready.

The worker pool is global to the cyclo_data process, persists across
recordings, and survives back-to-back START_RECORD calls without
blocking the response path — STOP returns immediately, transcode
happens in the background.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Callable, Dict, Iterable, Optional

import pyarrow.parquet as pq


_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"

# Verify pass tolerates this many frames of mismatch between sidecar
# row count and transcoded MP4 nb_frames. ffmpeg's mjpeg demuxer is
# known to drop the very first packet when it lacks an EOI marker
# (some camera drivers emit a truncated initial frame) so a 95-row
# sidecar may yield 94 H.264 frames. Anything beyond that suggests
# real frame loss and the transcode is rejected.
_VERIFY_FRAME_TOLERANCE = 1

# Concurrent transcode jobs. Tuned for Jetson Orin: libx264 at
# ultrafast+720p is roughly 1 core per stream, so two parallel streams
# leaves 4+ cores free for the next recording. NVENC has its own
# session limit so when it's available we drop parallelism to 1.
_DEFAULT_PARALLELISM_CPU = 2
_DEFAULT_PARALLELISM_HW = 1

# Status field values written into episode_info.json.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_NOT_REQUIRED = "not_required"  # no videos in this episode


_encoder_cache: "tuple[str, list[str]] | None" = None


def _detect_encoder() -> tuple[str, list[str]]:
    """Pick the fastest *runtime-usable* H.264 encoder once per process.

    Mirrors ``video_sync._h264_encoder`` but kept local so the recorder
    process doesn't import the converter package (avoids dragging
    cyclo_data.converter dependencies into the live recording path).
    """
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    candidates: list[tuple[str, list[str]]] = [
        ("h264_nvenc", ["-preset", "p4", "-tune", "ll", "-rc", "vbr", "-cq", "23"]),
        ("h264_v4l2m2m", ["-b:v", "5M"]),
        ("libx264", ["-preset", "ultrafast", "-crf", "23"]),
    ]
    probe = subprocess.run(
        [_FFMPEG, "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=False,
    )
    listed = probe.stdout
    for name, opts in candidates:
        if name not in listed:
            continue
        # Smoke-encode a 1-frame stream to /dev/null to confirm the
        # encoder is actually usable (compiled-in != usable: NVENC
        # often fails with "Operation not permitted" in containers).
        smoke = subprocess.run(
            [
                _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:r=15:d=0.1",
                "-c:v", name, *opts,
                "-pix_fmt", "yuv420p",
                "-f", "null", "-",
            ],
            capture_output=True, timeout=10,
        )
        if smoke.returncode == 0:
            _encoder_cache = (name, opts)
            return _encoder_cache
    raise RuntimeError("No usable H.264 encoder found")


@dataclass
class TranscodeResult:
    episode_dir: Path
    success: bool
    elapsed_sec: float
    encoder: str
    cameras_done: list[str]
    cameras_failed: dict[str, str]   # cam -> error message
    error: Optional[str] = None


class TranscodeWorker:
    """Pool that turns per-camera MJPEG MP4s into H.264 in the background.

    One instance per cyclo_data process. ``submit(episode_dir)`` is
    idempotent — re-submitting the same dir while a previous job is
    in flight returns the same Future.
    """

    def __init__(
        self,
        logger=None,
        parallelism: Optional[int] = None,
    ) -> None:
        self._logger = logger
        # Encoder probe is deferred until the first submit so importing
        # this module is cheap and doesn't spawn ffmpeg subprocesses.
        self._encoder: Optional[tuple[str, list[str]]] = None
        self._parallelism_override = parallelism

        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._inflight: Dict[str, Future] = {}
        self._shutdown = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        episode_dir: Path,
        on_complete: Optional[Callable[[TranscodeResult], None]] = None,
    ) -> Future:
        """Queue ``episode_dir`` for transcoding. Idempotent."""
        episode_dir = Path(episode_dir).resolve()
        # Clean up orphan .h264.tmp here on the submit side instead of
        # in every _run_one — the only producers of orphan tmps are
        # crash-mid-transcode (handled by submit_pending_recovery) and
        # a same-key retry after failure. Both go through submit(), so
        # the worker thread doesn't have to glob on every job.
        self._cleanup_orphan_tmps(episode_dir)
        key = str(episode_dir)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("TranscodeWorker is shut down")
            if key in self._inflight:
                fut = self._inflight[key]
                if not fut.done():
                    return fut
                # finished — fall through to re-submit (e.g. retry after
                # a previous failed status was patched).
                del self._inflight[key]
            self._ensure_pool()
            future = self._pool.submit(self._run_one, episode_dir)  # type: ignore[union-attr]
            self._inflight[key] = future

        def _on_done(f: Future, _key=key, _cb=on_complete) -> None:
            with self._lock:
                self._inflight.pop(_key, None)
            if _cb is None:
                return
            try:
                res = f.result()
            except Exception as exc:  # pragma: no cover - defensive
                res = TranscodeResult(
                    episode_dir=Path(_key), success=False, elapsed_sec=0.0,
                    encoder="unknown", cameras_done=[], cameras_failed={},
                    error=repr(exc),
                )
            try:
                _cb(res)
            except Exception:  # pragma: no cover - callback isolation
                pass

        future.add_done_callback(_on_done)
        return future

    def submit_pending_recovery(
        self,
        workspace_root: Path,
        on_complete: Optional[Callable[[TranscodeResult], None]] = None,
    ) -> list[Future]:
        """Scan ``workspace_root`` and enqueue every pending/running episode.

        ``running`` means a previous process crashed mid-transcode — we
        treat it the same as ``pending`` because the orphan ``.h264.tmp``
        files will be cleaned up at the start of the new attempt.
        """
        episodes: list[Path] = []
        for info_path in Path(workspace_root).glob("*/[0-9]*/episode_info.json"):
            try:
                with open(info_path) as f:
                    info = json.load(f)
            except Exception:
                continue
            status = info.get("transcoding_status")
            if status in (STATUS_PENDING, STATUS_RUNNING):
                episodes.append(info_path.parent)
        futures: list[Future] = []
        for ep in episodes:
            self._log_info(f"Transcoder resume: queueing {ep}")
            futures.append(self.submit(ep, on_complete=on_complete))
        return futures

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup_orphan_tmps(self, episode_dir: Path) -> None:
        """Remove ``<cam>.h264.tmp`` left behind by a crashed prior encode.

        Called from submit() so the hot worker thread doesn't pay for a
        glob on every job — fresh recordings created via VideoRecorder
        cannot have orphans (the videos/ dir is brand new), so the cost
        only matters during recovery + retry.
        """
        videos_dir = episode_dir / "videos"
        if not videos_dir.exists():
            return
        for stale in videos_dir.glob("*.h264.tmp"):
            try:
                stale.unlink()
                self._log_info(f"transcode: cleaned orphan {stale.name}")
            except OSError:
                pass

    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warning(msg)

    def _log_error(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.error(msg)

    def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        if self._encoder is None:
            self._encoder = _detect_encoder()
        if self._parallelism_override is not None:
            parallelism = max(1, int(self._parallelism_override))
        elif self._encoder[0].endswith("nvenc"):
            parallelism = _DEFAULT_PARALLELISM_HW
        else:
            parallelism = _DEFAULT_PARALLELISM_CPU
        self._log_info(
            f"TranscodeWorker pool: encoder={self._encoder[0]} parallelism={parallelism}"
        )
        self._pool = ThreadPoolExecutor(
            max_workers=parallelism, thread_name_prefix="transcode",
        )

    def _run_one(self, episode_dir: Path) -> TranscodeResult:
        # Transcoding is background work and must never starve the live
        # recording subscribers or the robot control loop. Bump the
        # worker thread's nice value so the kernel scheduler hands CPU
        # to anything more time-sensitive when contention arises.
        # ``os.nice`` is per-thread on Linux 2.6+, so this only affects
        # this transcode (not the parent rclpy executor).
        try:
            os.nice(10)
        except OSError:
            pass
        t0 = time.time()
        info_path = episode_dir / "episode_info.json"
        videos_dir = episode_dir / "videos"
        assert self._encoder is not None
        encoder_name, encoder_opts = self._encoder

        # Pull camera_rotations from episode_info.json — the recorder
        # stored these per camera from the yaml at save_robotis_metadata
        # time. The values represent the physical sensor orientation
        # offset (e.g. ``270`` for an upside-down wrist camera) and we
        # apply ``-vf transpose=N`` to fix it during this single
        # decode+encode pass instead of pushing the cost downstream.
        rotations: Dict[str, int] = {}
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f) or {}
                rotations = {
                    cam: int(deg) for cam, deg in
                    (info.get("camera_rotations") or {}).items()
                }
            except Exception:
                rotations = {}

        # Orphan ``.h264.tmp`` cleanup happens at submit() time now
        # (see TranscodeWorker._cleanup_orphan_tmps) so this worker
        # thread can skip the per-job glob.

        # Discover cameras: any <cam>.mp4 that has a paired sidecar.
        cameras: list[str] = []
        if videos_dir.exists():
            for mp4 in sorted(videos_dir.glob("*.mp4")):
                cam = mp4.stem
                if cam.endswith("_synced"):
                    continue  # converter-side derivative, never transcode
                if (videos_dir / f"{cam}_timestamps.parquet").exists():
                    cameras.append(cam)

        if not cameras:
            self._log_info(
                f"transcode: {episode_dir.name} has no cameras; marking not_required"
            )
            _patch_status(info_path, STATUS_NOT_REQUIRED, encoder=encoder_name)
            return TranscodeResult(
                episode_dir=episode_dir, success=True, elapsed_sec=time.time() - t0,
                encoder=encoder_name, cameras_done=[], cameras_failed={},
            )

        _patch_status(info_path, STATUS_RUNNING, encoder=encoder_name)

        done: list[str] = []
        failed: dict[str, str] = {}
        for cam in cameras:
            try:
                self._transcode_camera(
                    cam_name=cam,
                    videos_dir=videos_dir,
                    encoder_name=encoder_name,
                    encoder_opts=encoder_opts,
                    rotation_deg=rotations.get(cam, 0),
                )
                done.append(cam)
            except Exception as exc:
                self._log_error(
                    f"transcode {episode_dir.name}/{cam}: {exc!r}"
                )
                failed[cam] = repr(exc)

        elapsed = time.time() - t0
        success = len(failed) == 0
        final_status = STATUS_DONE if success else STATUS_FAILED
        # Record per-camera applied rotations so downstream convert
        # passes know whether to re-rotate (cf. base_converter +
        # video_sync). Only cameras with success are listed.
        applied = {cam: int(rotations.get(cam, 0)) for cam in done}
        _patch_status(
            info_path,
            final_status,
            encoder=encoder_name,
            elapsed_sec=elapsed,
            cameras_done=done,
            cameras_failed=failed,
            rotations_applied=applied,
        )
        return TranscodeResult(
            episode_dir=episode_dir, success=success, elapsed_sec=elapsed,
            encoder=encoder_name, cameras_done=done, cameras_failed=failed,
        )

    @staticmethod
    def _transpose_filter(rotation_deg: int) -> Optional[str]:
        """Map a ``rotation_deg`` value (0/90/180/270) to ffmpeg ``-vf``.

        Convention matches the legacy rosbag2mp4 pipeline and the
        physical_ai_tools reference so existing yaml configs port over
        unchanged:

        * 0   → no filter (pass through)
        * 90  → ``transpose=1`` (clockwise)
        * 180 → ``transpose=2,transpose=2`` (= 180° flip)
        * 270 → ``transpose=2`` (counter-clockwise)

        Anything else is logged and treated as 0 (no rotation).
        """
        if not rotation_deg:
            return None
        deg = int(rotation_deg) % 360
        if deg == 0:
            return None
        if deg == 90:
            return "transpose=1"
        if deg == 180:
            return "transpose=2,transpose=2"
        if deg == 270:
            return "transpose=2"
        return None

    def _transcode_camera(
        self,
        cam_name: str,
        videos_dir: Path,
        encoder_name: str,
        encoder_opts: Iterable[str],
        rotation_deg: int = 0,
    ) -> None:
        raw_mp4 = videos_dir / f"{cam_name}.mp4"
        tmp_mp4 = videos_dir / f"{cam_name}.h264.tmp"
        sidecar = videos_dir / f"{cam_name}_timestamps.parquet"

        if not raw_mp4.exists():
            raise FileNotFoundError(f"raw MP4 missing: {raw_mp4}")
        if not sidecar.exists():
            raise FileNotFoundError(f"sidecar missing: {sidecar}")

        # Empty episode (0 rows) — refuse to encode but treat as success.
        # Resulting MP4 with zero frames isn't useful but isn't an error
        # either; drop the raw and write an empty stub so callers can
        # treat the file uniformly.
        sidecar_rows = _sidecar_row_count(sidecar)
        if sidecar_rows == 0:
            self._log_warn(
                f"transcode {cam_name}: sidecar has 0 rows; deleting raw"
            )
            raw_mp4.unlink(missing_ok=True)
            return

        cmd = [
            _FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
            "-i", str(raw_mp4),
            "-c:v", encoder_name, *list(encoder_opts),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            # CRITICAL: the recorder writes the raw MP4 with VFR PTS
            # (via ``-use_wallclock_as_timestamps 1``) but the container
            # still carries an r_frame_rate=25/1 stream tag. Without an
            # explicit fps_mode, ffmpeg defaults to CFR re-sampling at
            # that tag rate, *duplicating* frames to fill the gaps
            # between sparse PTS values — so a 608-frame 40s episode
            # becomes a 1000+ frame H.264 that fails the verify step.
            # ``passthrough`` keeps every input frame at its original
            # PTS, preserving the exact 1:1 mapping to the sidecar.
            "-fps_mode", "passthrough",
            # Output filename is .h264.tmp (chosen so a partial file
            # can't be mistaken for the final MP4); ffmpeg can't infer
            # the muxer from that suffix so we pin it to mp4.
            "-f", "mp4",
            str(tmp_mp4),
        ]
        rot_filter = self._transpose_filter(rotation_deg)
        if rot_filter is not None:
            # Insert the -vf right before the output path so it applies
            # to the encoded stream.
            cmd = cmd[:-1] + ["-vf", rot_filter, cmd[-1]]
            self._log_info(
                f"transcode {cam_name}: applying rotation_deg={rotation_deg} "
                f"({rot_filter})"
            )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            tmp_mp4.unlink(missing_ok=True)
            stderr_tail = (result.stderr or "")[-400:]
            raise RuntimeError(
                f"ffmpeg encode rc={result.returncode}: {stderr_tail}"
            )
        if not tmp_mp4.exists() or tmp_mp4.stat().st_size == 0:
            tmp_mp4.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg produced no output file")

        # Verify pass — frame count tolerated mismatch.
        encoded_frames = _mp4_frame_count(tmp_mp4)
        if abs(encoded_frames - sidecar_rows) > _VERIFY_FRAME_TOLERANCE:
            tmp_mp4.unlink(missing_ok=True)
            raise RuntimeError(
                f"frame count mismatch: encoded={encoded_frames} "
                f"sidecar={sidecar_rows} (tolerance={_VERIFY_FRAME_TOLERANCE})"
            )

        # Atomic replace — os.replace is POSIX-rename + Windows-friendly
        # and crucially overwrites the destination in one syscall.
        os.replace(tmp_mp4, raw_mp4)


# ----------------------------------------------------------------------
# Helpers (module-level so they're picklable + reusable across tests)
# ----------------------------------------------------------------------


def _sidecar_row_count(sidecar: Path) -> int:
    return int(pq.read_metadata(str(sidecar)).num_rows)


_FFPROBE_FRAME_COUNT_TIMEOUT = 30  # seconds


def _mp4_frame_count(mp4: Path) -> int:
    """Return the frame count of an MP4 via ffprobe.

    Uses ``-count_frames`` which is slow on huge files but exact, so
    the verify pass refuses to ship a transcode that doesn't match
    the sidecar. Raises ``RuntimeError`` on timeout or parse failure so
    the caller can fail this single camera without blocking the worker
    pool.
    """
    try:
        out = subprocess.run(
            [
                _FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "default=nw=1:nk=1", str(mp4),
            ],
            capture_output=True, text=True,
            timeout=_FFPROBE_FRAME_COUNT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffprobe frame-count timed out after {_FFPROBE_FRAME_COUNT_TIMEOUT}s "
            f"for {mp4.name}"
        ) from exc
    try:
        return int(out.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe could not determine frame count for {mp4.name}: "
            f"stderr={out.stderr!r}"
        ) from exc


def _patch_status(
    info_path: Path,
    status: str,
    *,
    encoder: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
    cameras_done: Optional[list[str]] = None,
    cameras_failed: Optional[dict[str, str]] = None,
    rotations_applied: Optional[dict[str, int]] = None,
) -> None:
    """Read-modify-write episode_info.json atomically.

    Uses write-to-tmp-then-rename so a crash mid-write never leaves the
    JSON half-truncated.
    """
    if not info_path.exists():
        # Stub it out — better than failing the whole transcode.
        info_path.write_text(json.dumps({}))
    try:
        with open(info_path) as f:
            info = json.load(f) or {}
    except Exception:
        info = {}
    info["transcoding_status"] = status
    if encoder is not None:
        info["transcoding_encoder"] = encoder
    if elapsed_sec is not None:
        info["transcoding_elapsed_sec"] = round(elapsed_sec, 3)
    if cameras_done is not None:
        info["transcoding_cameras_done"] = list(cameras_done)
    if cameras_failed is not None:
        info["transcoding_cameras_failed"] = dict(cameras_failed)
    if rotations_applied is not None:
        # Merge with any existing applied rotations so a partial re-run
        # doesn't lose history for cameras that already finished.
        prev = info.get("camera_rotations_applied") or {}
        prev.update({k: int(v) for k, v in rotations_applied.items()})
        info["camera_rotations_applied"] = prev
    info["transcoding_updated"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, info_path)
