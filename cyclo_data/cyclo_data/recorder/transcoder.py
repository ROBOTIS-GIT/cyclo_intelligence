# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Background MJPEG-to-H.264 transcoder for recording format v2.

Recording writes per-camera raw MJPEG spools for near-zero live CPU/GPU
cost. After STOP, this module first remuxes each ``*.mjpeg.tmp`` spool
to an MJPEG MP4, then re-encodes each camera's MP4 to H.264 in a worker
pool so the final on-disk format is universally playable (VSCode
webview / browser <video>) and ~20x smaller.

Crash-safety story
------------------

For every camera, the worker:
  1. remuxes ``<cam>.mjpeg.tmp`` → ``<cam>.remuxing.mp4``
  2. verifies the remuxed MP4 frame count against the sidecar
  3. atomically renames ``<cam>.remuxing.mp4`` → ``<cam>.mp4``
     and removes the raw spool
  4. encodes ``<cam>.mp4`` → ``<cam>.h264.tmp`` (MJPEG MP4 stays intact)
  5. verifies the new file's frame count against the sidecar
  6. atomically renames ``<cam>.h264.tmp`` → ``<cam>.mp4``
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
import yaml


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


@dataclass(frozen=True)
class _CameraJob:
    camera_id: str
    camera_name: str
    videos_dir: Path


@dataclass(frozen=True)
class _RawRemuxJob:
    camera_id: str
    camera_name: str
    videos_dir: Path
    raw_path: Path
    mp4_path: Path
    sidecar_path: Path
    stats_path: Path


class _SkipCamera(Exception):
    """Camera stream is unusable for this episode but should not fail it."""


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
        for pattern in ("*.h264.tmp", "*.remuxing.mp4"):
            for stale in videos_dir.rglob(pattern):
                try:
                    stale.unlink()
                    self._log_info(f"transcode: cleaned orphan {stale}")
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

        # Pull record-time camera rotations from camera_metadata.yaml.
        # episode_info.json stays focused on task/episode semantics while
        # camera_info/ owns camera provenance and calibration-adjacent data.
        rotations = _read_camera_metadata_rotations(episode_dir)
        if not rotations and info_path.exists():
            # Legacy fallback for recordings made before camera_metadata.yaml.
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

        _patch_status(
            info_path,
            STATUS_RUNNING,
            encoder=encoder_name,
            video_remux_status=STATUS_RUNNING,
        )

        remux_failed = self._remux_pending_raw_spools(videos_dir)
        if remux_failed:
            _patch_status(
                info_path,
                STATUS_FAILED,
                encoder=encoder_name,
                cameras_failed=remux_failed,
                video_remux_status=STATUS_FAILED,
            )
            return TranscodeResult(
                episode_dir=episode_dir,
                success=False,
                elapsed_sec=time.time() - t0,
                encoder=encoder_name,
                cameras_done=[],
                cameras_failed=remux_failed,
            )

        _patch_status(
            info_path,
            STATUS_RUNNING,
            encoder=encoder_name,
            cameras_failed={},
            video_remux_status=STATUS_DONE,
        )

        camera_jobs = self._discover_camera_jobs(videos_dir)

        if not camera_jobs:
            self._log_info(
                f"transcode: {episode_dir.name} has no cameras; marking not_required"
            )
            _patch_status(
                info_path,
                STATUS_NOT_REQUIRED,
                encoder=encoder_name,
                video_remux_status=STATUS_NOT_REQUIRED,
            )
            return TranscodeResult(
                episode_dir=episode_dir, success=True, elapsed_sec=time.time() - t0,
                encoder=encoder_name, cameras_done=[], cameras_failed={},
            )

        done: list[str] = []
        skipped: dict[str, str] = {}
        failed: dict[str, str] = {}
        for job in camera_jobs:
            try:
                self._transcode_camera(
                    cam_name=job.camera_name,
                    videos_dir=job.videos_dir,
                    encoder_name=encoder_name,
                    encoder_opts=encoder_opts,
                    rotation_deg=rotations.get(job.camera_name, 0),
                )
                done.append(job.camera_id)
            except _SkipCamera as exc:
                self._log_warn(
                    f"transcode {episode_dir.name}/{job.camera_id}: skipped: {exc}"
                )
                skipped[job.camera_id] = str(exc)
            except Exception as exc:
                self._log_error(
                    f"transcode {episode_dir.name}/{job.camera_id}: {exc!r}"
                )
                failed[job.camera_id] = repr(exc)

        elapsed = time.time() - t0
        success = len(failed) == 0
        final_status = STATUS_DONE if success else STATUS_FAILED
        _patch_status(
            info_path,
            final_status,
            encoder=encoder_name,
            elapsed_sec=elapsed,
            cameras_done=done,
            cameras_failed=failed,
            cameras_skipped=skipped,
            video_remux_status=STATUS_DONE,
        )
        return TranscodeResult(
            episode_dir=episode_dir, success=success, elapsed_sec=elapsed,
            encoder=encoder_name, cameras_done=done, cameras_failed=failed,
        )

    @staticmethod
    def _camera_id_for(videos_dir: Path, camera_dir: Path, camera_name: str) -> str:
        if camera_dir == videos_dir:
            return camera_name
        return f"{camera_dir.relative_to(videos_dir).as_posix()}/{camera_name}"

    @classmethod
    def _discover_raw_spool_jobs(cls, videos_dir: Path) -> list[_RawRemuxJob]:
        if not videos_dir.exists():
            return []

        jobs: list[_RawRemuxJob] = []
        for raw_path in sorted(videos_dir.rglob("*.mjpeg.tmp")):
            if not raw_path.is_file():
                continue
            camera_name = raw_path.name[:-len(".mjpeg.tmp")]
            camera_dir = raw_path.parent
            jobs.append(_RawRemuxJob(
                camera_id=cls._camera_id_for(videos_dir, camera_dir, camera_name),
                camera_name=camera_name,
                videos_dir=camera_dir,
                raw_path=raw_path,
                mp4_path=camera_dir / f"{camera_name}.mp4",
                sidecar_path=camera_dir / f"{camera_name}_timestamps.parquet",
                stats_path=camera_dir / f"{camera_name}_recorder_stats.json",
            ))
        return jobs

    def _remux_pending_raw_spools(self, videos_dir: Path) -> dict[str, str]:
        failed: dict[str, str] = {}
        for job in self._discover_raw_spool_jobs(videos_dir):
            if job.mp4_path.exists() and job.mp4_path.stat().st_size > 0:
                job.raw_path.unlink(missing_ok=True)
                _patch_recorder_stats(
                    job.stats_path,
                    {
                        "remux_status": STATUS_DONE,
                        "remux_error": None,
                    },
                )
                continue
            try:
                self._remux_raw_spool(job)
            except _SkipCamera as exc:
                self._log_warn(
                    f"remux {job.camera_id}: skipped: {exc}"
                )
            except Exception as exc:
                failed[job.camera_id] = repr(exc)
                self._log_error(
                    f"remux {job.camera_id}: {exc!r}"
                )
        return failed

    def _remux_raw_spool(self, job: _RawRemuxJob) -> None:
        if not job.sidecar_path.exists():
            raise FileNotFoundError(f"sidecar missing: {job.sidecar_path}")

        sidecar_rows = _sidecar_row_count(job.sidecar_path)
        if sidecar_rows <= 0:
            job.raw_path.unlink(missing_ok=True)
            job.sidecar_path.unlink(missing_ok=True)
            _patch_recorder_stats(
                job.stats_path,
                {
                    "remux_status": STATUS_NOT_REQUIRED,
                    "remux_error": None,
                },
            )
            raise _SkipCamera("sidecar has 0 rows; removed raw spool")

        tmp_mp4 = job.videos_dir / f"{job.camera_name}.remuxing.mp4"
        tmp_mp4.unlink(missing_ok=True)
        fps = _estimate_raw_framerate(job.sidecar_path, job.stats_path)
        cmd = [
            _FFMPEG,
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-framerate", f"{fps:.6f}",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-i", str(job.raw_path),
            "-c:v", "copy",
            "-an",
            "-fps_mode", "passthrough",
            "-video_track_timescale", "90000",
            str(tmp_mp4),
        ]
        start = time.monotonic()
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        elapsed_ns = int((time.monotonic() - start) * 1_000_000_000)
        if result.returncode != 0:
            tmp_mp4.unlink(missing_ok=True)
            error = result.stderr.decode("utf-8", errors="replace")
            _patch_recorder_stats(
                job.stats_path,
                {
                    "remux_status": STATUS_FAILED,
                    "remux_error": error,
                    "remux_duration_ns": elapsed_ns,
                },
            )
            raise RuntimeError(f"ffmpeg remux rc={result.returncode}: {error}")

        try:
            remuxed_frames = _mp4_frame_count(tmp_mp4)
        except Exception as exc:
            tmp_mp4.unlink(missing_ok=True)
            error = str(exc)
            _patch_recorder_stats(
                job.stats_path,
                {
                    "remux_status": STATUS_FAILED,
                    "remux_error": error,
                    "remux_duration_ns": elapsed_ns,
                },
            )
            raise RuntimeError(error) from exc
        if remuxed_frames != sidecar_rows:
            tmp_mp4.unlink(missing_ok=True)
            error = (
                f"frame count mismatch mp4={remuxed_frames} "
                f"sidecar={sidecar_rows}"
            )
            _patch_recorder_stats(
                job.stats_path,
                {
                    "remux_status": STATUS_FAILED,
                    "remux_error": error,
                    "frames_remuxed": remuxed_frames,
                    "remux_duration_ns": elapsed_ns,
                },
            )
            raise RuntimeError(error)

        os.replace(tmp_mp4, job.mp4_path)
        job.raw_path.unlink(missing_ok=True)
        _patch_recorder_stats(
            job.stats_path,
            {
                "remux_status": STATUS_DONE,
                "remux_error": None,
                "frames_remuxed": remuxed_frames,
                "remux_duration_ns": elapsed_ns,
            },
        )

    @staticmethod
    def _discover_camera_jobs(videos_dir: Path) -> list[_CameraJob]:
        """Find flat and segmented camera MP4s that have timestamp sidecars."""
        if not videos_dir.exists():
            return []

        jobs: list[_CameraJob] = []

        for mp4 in sorted(videos_dir.glob("*.mp4")):
            cam = mp4.stem
            if cam.endswith("_synced"):
                continue
            if (videos_dir / f"{cam}_timestamps.parquet").exists():
                jobs.append(_CameraJob(cam, cam, videos_dir))

        for mp4 in sorted(videos_dir.glob("*/*.mp4")):
            cam = mp4.stem
            if cam.endswith("_synced"):
                continue
            segment_dir = mp4.parent
            if (segment_dir / f"{cam}_timestamps.parquet").exists():
                jobs.append(_CameraJob(f"{segment_dir.name}/{cam}", cam, segment_dir))

        return jobs

    @staticmethod
    def _transpose_filter(rotation_deg: int) -> Optional[str]:
        """Map a ``rotation_deg`` value (0/90/180/270) to ffmpeg ``-vf``.

        Convention matches the legacy rosbag2mp4 pipeline and the
        cyclo_intelligence reference so existing yaml configs port over
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
            sidecar.unlink(missing_ok=True)
            raise _SkipCamera("sidecar has 0 rows; removed camera files")

        width, height = _mp4_dimensions(raw_mp4)
        if width < 2 or height < 2:
            raw_mp4.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            raise _SkipCamera(
                f"degenerate video dimensions {width}x{height}; "
                "removed camera files for this episode"
            )

        codec = _mp4_codec_name(raw_mp4)
        if codec == "h264":
            encoded_frames = _mp4_frame_count(raw_mp4)
            if abs(encoded_frames - sidecar_rows) <= _VERIFY_FRAME_TOLERANCE:
                self._log_info(
                    f"transcode {cam_name}: already h264; skipping encode"
                )
                return
            raise RuntimeError(
                f"h264 frame count mismatch: encoded={encoded_frames} "
                f"sidecar={sidecar_rows} (tolerance={_VERIFY_FRAME_TOLERANCE})"
            )

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
        filters = []
        rot_filter = self._transpose_filter(rotation_deg)
        if rot_filter is not None:
            filters.append(rot_filter)
        if width % 2 or height % 2:
            filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        if filters:
            # Insert the -vf right before the output path so it applies
            # to the encoded stream.
            cmd = cmd[:-1] + ["-vf", ",".join(filters), cmd[-1]]
            self._log_info(
                f"transcode {cam_name}: applying rotation_deg={rotation_deg} "
                f"({','.join(filters)})"
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
        try:
            encoded_frames = _mp4_frame_count(tmp_mp4)
        except Exception:
            tmp_mp4.unlink(missing_ok=True)
            raise
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


def _patch_recorder_stats(stats_path: Path, updates: dict) -> None:
    if not stats_path.exists():
        return
    try:
        with open(stats_path, encoding="utf-8") as f:
            payload = json.load(f) or {}
        payload.update(updates)
        tmp = stats_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, stats_path)
    except Exception:
        return


def _estimate_raw_framerate(sidecar: Path, stats_path: Path) -> float:
    default = 30.0
    if stats_path.exists():
        try:
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f) or {}
            frames = int(stats.get("frames_written") or 0)
            first_ns = stats.get("first_recv_ns")
            last_ns = stats.get("last_recv_ns")
            if (
                frames > 1
                and first_ns is not None
                and last_ns is not None
                and int(last_ns) > int(first_ns)
            ):
                fps = (frames - 1) / ((int(last_ns) - int(first_ns)) / 1e9)
                if 1.0 <= fps <= 120.0:
                    return float(fps)
        except Exception:
            pass

    try:
        table = pq.read_table(str(sidecar), columns=["header_stamp_ns"])
        values = table.column(0).to_pylist()
        if len(values) > 1 and values[-1] is not None and values[0] is not None:
            first_ns = int(values[0])
            last_ns = int(values[-1])
            if last_ns > first_ns:
                fps = (len(values) - 1) / ((last_ns - first_ns) / 1e9)
                if 1.0 <= fps <= 120.0:
                    return float(fps)
    except Exception:
        pass
    return default


def _sidecar_row_count(sidecar: Path) -> int:
    return int(pq.read_metadata(str(sidecar)).num_rows)


_FFPROBE_FRAME_COUNT_TIMEOUT = 30  # seconds


def _mp4_dimensions(mp4: Path) -> tuple[int, int]:
    try:
        out = subprocess.run(
            [
                _FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x", str(mp4),
            ],
            capture_output=True, text=True,
            timeout=_FFPROBE_FRAME_COUNT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffprobe dimensions timed out after {_FFPROBE_FRAME_COUNT_TIMEOUT}s "
            f"for {mp4.name}"
        ) from exc
    lines = [
        line.strip()
        for line in (out.stdout or "").splitlines()
        if line.strip()
    ]
    text = lines[0] if lines else ""
    try:
        width, height = text.split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise RuntimeError(
            f"ffprobe could not determine dimensions for {mp4.name}: "
            f"stdout={out.stdout!r} stderr={out.stderr!r}"
        ) from exc


def _mp4_codec_name(mp4: Path) -> str:
    try:
        out = subprocess.run(
            [
                _FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nw=1:nk=1", str(mp4),
            ],
            capture_output=True, text=True,
            timeout=_FFPROBE_FRAME_COUNT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffprobe codec probe timed out after {_FFPROBE_FRAME_COUNT_TIMEOUT}s "
            f"for {mp4.name}"
        ) from exc
    lines = [
        line.strip()
        for line in (out.stdout or "").splitlines()
        if line.strip()
    ]
    codec = lines[0] if lines else ""
    if codec:
        return codec.lower()
    raise RuntimeError(
        f"ffprobe could not determine codec for {mp4.name}: "
        f"stderr={out.stderr!r}"
    )


def _mp4_frame_count(mp4: Path) -> int:
    """Return the frame count of an MP4 via ffprobe.

    Prefer container metadata / packet counts because ``-count_frames``
    can take minutes on multi-GB MJPEG files. MP4 video tracks have one
    packet per frame in our recorder/transcoder pipeline, so packet count
    gives the same safety check without decoding every JPEG. Falls back
    to ``-count_frames`` for odd files that lack both fast counts.
    """
    attempts = [
        (
            "nb_frames",
            [
                _FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=nw=1:nk=1", str(mp4),
            ],
        ),
        (
            "nb_read_packets",
            [
                _FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "default=nw=1:nk=1", str(mp4),
            ],
        ),
        (
            "nb_read_frames",
            [
                _FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nw=1:nk=1", str(mp4),
            ],
        ),
    ]
    errors: list[str] = []
    for label, cmd in attempts:
        try:
            out = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=_FFPROBE_FRAME_COUNT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            errors.append(
                f"{label} timed out after {_FFPROBE_FRAME_COUNT_TIMEOUT}s"
            )
            continue
        text = (out.stdout or "").strip()
        if text and text != "N/A":
            try:
                return int(text.splitlines()[0])
            except ValueError:
                errors.append(f"{label} stdout={out.stdout!r}")
                continue
        if out.stderr:
            errors.append(f"{label} stderr={out.stderr!r}")
        else:
            errors.append(f"{label} unavailable")
    detail = "; ".join(errors) if errors else "no ffprobe output"
    raise RuntimeError(
        f"ffprobe could not determine frame count for {mp4.name}: {detail}"
    )


def _mp4_frame_count_slow_decode(mp4: Path) -> int:
    """Legacy exact decode count kept for manual diagnostics/tests."""
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


def _read_camera_metadata_rotations(episode_dir: Path) -> Dict[str, int]:
    metadata_path = episode_dir / "camera_info" / "camera_metadata.yaml"
    if not metadata_path.exists():
        return {}
    try:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        cameras = data.get("cameras") or {}
        if not isinstance(cameras, dict):
            return {}
        rotations: Dict[str, int] = {}
        for name, entry in cameras.items():
            if not isinstance(entry, dict):
                continue
            rotations[str(name)] = int(entry.get("rotation_deg", 0) or 0)
        return rotations
    except Exception:
        return {}


def _patch_status(
    info_path: Path,
    status: str,
    *,
    encoder: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
    cameras_done: Optional[list[str]] = None,
    cameras_failed: Optional[dict[str, str]] = None,
    cameras_skipped: Optional[dict[str, str]] = None,
    video_remux_status: Optional[str] = None,
) -> None:
    """Read-modify-write episode_info.json atomically.

    Uses write-to-tmp-then-rename so a crash mid-write never leaves the
    JSON half-truncated.
    """
    if not info_path.exists():
        # Stub it out — better than failing the whole transcode.
        info_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
    try:
        with open(info_path) as f:
            info = json.load(f) or {}
    except Exception:
        info = {}
    info["transcoding_status"] = status
    if video_remux_status is not None:
        info["video_remux_status"] = video_remux_status
    if cameras_failed is not None:
        info["transcoding_cameras_failed"] = dict(cameras_failed)
    else:
        info.setdefault("transcoding_cameras_failed", {})
    for stale_key in (
        "transcoding_encoder",
        "transcoding_elapsed_sec",
        "transcoding_cameras_done",
        "transcoding_cameras_skipped",
        "transcoding_updated",
    ):
        info.pop(stale_key, None)
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, info_path)
