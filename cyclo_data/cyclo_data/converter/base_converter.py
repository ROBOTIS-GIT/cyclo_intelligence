#!/usr/bin/env python3
#
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
#
# Author: Dongyun Kim

"""
Shared infrastructure for ROSbag → LeRobot converters (v2.1, v3.0).

Holds the rosbag extraction, causal-sync resampling, video discovery, feature
building, and per-episode statistics — i.e. everything that is identical
between LeRobot v2.1 and v3.0. Format-specific writers live in
``to_lerobot_v21.py`` and ``to_lerobot_v30.py``.
"""

import bisect
import hashlib
import json
import os
import pickle
import shutil  # noqa: F401  (re-exported transitively for legacy callers)
import struct
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa  # noqa: F401  (re-exported transitively for legacy callers)
import pyarrow.parquet as pq  # noqa: F401

from cyclo_data.reader.bag_reader import BagReader
from cyclo_data.reader.metadata_manager import MetadataManager
from cyclo_data.reader.video_metadata_extractor import VideoMetadataExtractor
from shared.robot_configs import schema as robot_schema


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_FPS = 30

# Floor applied to per-joint observation.state / action std when writing
# stats. A joint that was not actuated during a recording has true std=0,
# which makes downstream (x - mean) / std normalization explode to
# ±Inf / NaN and crashes training. Flooring also tames "noise-only"
# joints (std well below sensor noise): after normalization their
# contribution stays bounded instead of being amplified into huge inputs.
# Joints with real motion sit well above 1e-3 rad so they are unaffected.
STATS_STD_FLOOR = 1e-3

# Stream-copy concat has fixed validation/probing overhead. For tiny clips,
# ultrafast H.264 encode can beat copy+validation; for real camera segments,
# avoiding pixel work wins decisively. Gate on width * height * frames so the
# fast path is used where it is actually faster.
_SEGMENT_COPY_MIN_PIXEL_FRAMES_ENV = "CYCLO_SEGMENT_COPY_MIN_PIXEL_FRAMES"
_DEFAULT_SEGMENT_COPY_MIN_PIXEL_FRAMES = 250_000_000
_VIDEO_SYNC_CAMERA_WORKERS_ENV = "CYCLO_VIDEO_SYNC_CAMERA_WORKERS"
_VIDEO_SYNC_TOTAL_WORKERS_ENV = "CYCLO_VIDEO_SYNC_TOTAL_WORKERS"
_VIDEO_SYNC_STAGING_DIR_ENV = "CYCLO_VIDEO_SYNC_STAGING_DIR"
_CONVERSION_ACTIVE_WORKERS_ENV = "CYCLO_CONVERSION_ACTIVE_WORKERS"
_CONVERSION_WORKER_NICE_ENV = "CYCLO_CONVERSION_WORKER_NICE"
_DEFAULT_VIDEO_SYNC_TOTAL_WORKERS = 6
_VIDEO_SYNC_CLEAN_CACHE_ENV = "CYCLO_VIDEO_SYNC_CLEAN_CACHE"
_EXTRACT_CACHE_DISABLE_ENV = "CYCLO_EXTRACT_CACHE_DISABLE"
_EXTRACT_CACHE_VERSION = 2
_RAW_CDR_EXTRACT_DISABLE_ENV = "CYCLO_EXTRACT_DISABLE_RAW_CDR"
_PREPARED_EPISODE_CACHE_DISABLE_ENV = "CYCLO_PREPARED_EPISODE_CACHE_DISABLE"
_PREPARED_EPISODE_CACHE_VERSION = 3
_VIDEO_COPY_MODE_ENV = "CYCLO_VIDEO_COPY_MODE"
_VIDEO_STATS_SAMPLES_ENV = "CYCLO_VIDEO_STATS_SAMPLES"
_CONVERTER_INFO_LOGS_ENV = "CYCLO_CONVERTER_INFO_LOGS"
_DEFAULT_VIDEO_STATS_SAMPLES = 8
_FICLONE_IOCTL = 0x40049409
_REFLINK_UNSUPPORTED_DEV_PAIRS: set[Tuple[int, int]] = set()


def _fast_absolute_path(path: Path) -> str:
    """Return an absolute path string without resolving symlinks."""
    return os.path.abspath(os.fspath(path))


def _same_file_or_same_path(src: Path, dst: Path) -> bool:
    """Cheap same-file guard for copy helpers.

    ``Path.resolve()`` walks every path component, which is surprisingly costly
    on cached conversions with many videos. Absolute string equality handles the
    common case; ``samefile`` covers existing symlink/hardlink aliases.
    """
    if _fast_absolute_path(src) == _fast_absolute_path(dst):
        return True
    try:
        return Path(dst).exists() and os.path.samefile(src, dst)
    except OSError:
        return False


class _WorkerLogger:
    """Keep child process hot paths quiet while preserving warnings/errors."""

    @staticmethod
    def info(msg: str) -> None:
        return None

    @staticmethod
    def warning(msg: str) -> None:
        print(f"[WARNING] {msg}")

    @staticmethod
    def error(msg: str) -> None:
        print(f"[ERROR] {msg}")


def _convert_rosbag_worker(bag_path_str, episode_index, config_dict):
    """Top-level function for ProcessPoolExecutor (must be picklable).

    Creates a fresh base converter in each worker and parses a single
    rosbag episode. Worker only invokes extraction methods (all defined
    on the base class), so the base instance is sufficient regardless
    of which subclass triggered the parallel run.
    """
    config = ConversionConfig(**config_dict)
    converter = RosbagToLerobotConverterBase(config, logger=_WorkerLogger())
    result = converter.convert_single_rosbag(Path(bag_path_str), episode_index)
    return episode_index, result


# Hard ceiling so a many-core host does not oversubscribe disk, OpenCV decode,
# and ffmpeg subprocesses. Keep the default portable for workstation + edge
# devices such as Orin; dedicated benchmark hosts can still opt into wider
# pools with CYCLO_CONVERSION_MAX_WORKERS.
_CONVERSION_WORKER_CEILING = 4
_CONVERSION_MAX_PROFILE_WORKER_CEILING = 16


def _max_speed_profile_requested() -> bool:
    profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
    return profile in {"max", "maximum", "max_speed", "fastest"}


def _converter_info_logs_enabled() -> bool:
    raw = os.environ.get(_CONVERTER_INFO_LOGS_ENV)
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return not _max_speed_profile_requested()


def _resolve_conversion_worker_count(work_units: int) -> int:
    """Pick the worker count for conversion ProcessPoolExecutor pools.

    Defaults to **half of the visible CPUs** so the other half stays
    free for the ROS control loop, camera drivers, and any inference
    container that's running alongside. Without this cap a many-core host
    can launch enough episode workers × camera ffmpegs to starve the 100 Hz
    control loop and slow the conversion through I/O contention.

    Override with ``CYCLO_CONVERSION_MAX_WORKERS`` when an operator
    explicitly wants to dedicate the machine to conversion (e.g. CI,
    benchmark host) or further reduce parallelism on a thermally
    constrained system.
    """
    if work_units <= 1:
        return 1
    env_override = os.environ.get('CYCLO_CONVERSION_MAX_WORKERS')
    if env_override:
        try:
            override = max(1, int(env_override))
            return max(1, min(override, work_units))
        except ValueError:
            pass
    cpu_total = os.cpu_count() or 4
    half = max(1, cpu_total // 2)
    ceiling = (
        _CONVERSION_MAX_PROFILE_WORKER_CEILING
        if _max_speed_profile_requested()
        else _CONVERSION_WORKER_CEILING
    )
    return max(1, min(half, work_units, ceiling))


def _conversion_worker_init() -> None:
    """ProcessPoolExecutor initializer — yield CPU to higher-priority work.

    Niceness +10 makes the worker a
    "background" citizen so the kernel scheduler prefers the ROS
    control loop and inference container when CPUs are saturated.
    Set ``CYCLO_CONVERSION_WORKER_NICE=0`` for dedicated batch conversion
    hosts, or any value in the ``-20..+19`` range when operators need a
    different scheduling tradeoff.

    Best-effort: silently no-op on platforms / containers that block
    the syscall.
    """
    default_nice = "0" if _max_speed_profile_requested() else "10"
    raw_nice = os.environ.get(_CONVERSION_WORKER_NICE_ENV, default_nice)
    try:
        nice_value = int(raw_nice)
    except (TypeError, ValueError):
        nice_value = 10

    if nice_value == 0:
        return
    nice_value = max(-20, min(19, nice_value))

    try:
        os.nice(nice_value)
    except (OSError, AttributeError):
        pass


def _clone_or_copy_file(src: Path, dst: Path) -> str:
    """Create ``dst`` from ``src`` using CoW reflink when available.

    Reflinks keep source and destination as separate inodes, unlike
    hardlinks, but avoid copying video bytes on filesystems that support
    Linux ``FICLONE``. Falls back to an atomic ``shutil.copyfile`` everywhere
    else; dataset video copies need byte correctness, not source xattrs.

    Set ``CYCLO_VIDEO_COPY_MODE=hardlink`` to prefer hardlinks for immutable
    local datasets on filesystems without reflink support.

    Returns ``"reflink"``, ``"hardlink"``, ``"copy"``, or ``"same_path"``
    for logging/tests.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _same_file_or_same_path(src, dst):
        return "same_path"

    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    if os.environ.get(_VIDEO_COPY_MODE_ENV, "").strip().lower() == "hardlink":
        try:
            os.link(src, tmp)
            os.replace(tmp, dst)
            return "hardlink"
        except Exception:
            tmp.unlink(missing_ok=True)

    dev_pair: Optional[Tuple[int, int]] = None
    try:
        dev_pair = (src.stat().st_dev, dst.parent.stat().st_dev)
    except OSError:
        dev_pair = None

    if dev_pair not in _REFLINK_UNSUPPORTED_DEV_PAIRS:
        try:
            import fcntl

            with open(src, "rb") as src_fh, open(tmp, "wb") as dst_fh:
                fcntl.ioctl(dst_fh.fileno(), _FICLONE_IOCTL, src_fh.fileno())
            os.replace(tmp, dst)
            return "reflink"
        except Exception:
            tmp.unlink(missing_ok=True)
            if dev_pair is not None:
                _REFLINK_UNSUPPORTED_DEV_PAIRS.add(dev_pair)

    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return "copy"


@contextmanager
def _active_conversion_workers(worker_count: int):
    """Expose parent process-pool width to child conversion workers.

    Camera sync runs inside each episode worker. Without this marker every
    worker assumes it is alone and may spawn four ffmpeg encoders, turning an
    8-episode conversion into 32 concurrent video jobs. The marker lets the
    per-episode resolver divide a global sync budget across active episodes.
    """
    previous = os.environ.get(_CONVERSION_ACTIVE_WORKERS_ENV)
    os.environ[_CONVERSION_ACTIVE_WORKERS_ENV] = str(max(1, int(worker_count)))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_CONVERSION_ACTIVE_WORKERS_ENV, None)
        else:
            os.environ[_CONVERSION_ACTIVE_WORKERS_ENV] = previous


@dataclass
class StalenessMetrics:
    """Metrics for tracking data staleness during causal sync resampling."""

    topic: str
    total_samples: int = 0
    stale_warning_count: int = 0
    stale_error_count: int = 0
    max_staleness_ms: float = 0.0
    mean_staleness_ms: float = 0.0
    stale_samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def warning_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.stale_warning_count / self.total_samples

    @property
    def error_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.stale_error_count / self.total_samples

    @property
    def status(self) -> str:
        if self.stale_error_count > 0:
            return "ERROR"
        if self.stale_warning_count > 0:
            return "WARNING"
        return "GOOD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "total_samples": self.total_samples,
            "staleness": {
                "warning_count": self.stale_warning_count,
                "error_count": self.stale_error_count,
                "warning_ratio": round(self.warning_ratio * 100, 2),
                "error_ratio": round(self.error_ratio * 100, 2),
                "max_ms": round(self.max_staleness_ms, 2),
                "mean_ms": round(self.mean_staleness_ms, 2),
            },
            "status": self.status,
            "stale_samples": self.stale_samples[:20],  # Limit to first 20
        }


@dataclass
class ConversionConfig:
    """Configuration for ROSbag to LeRobot conversion."""

    repo_id: str
    output_dir: Path
    fps: int = DEFAULT_FPS
    robot_type: str = "unknown"
    use_videos: bool = True
    chunks_size: int = DEFAULT_CHUNK_SIZE

    # Robot config file path (e.g., ffw_sg2_rev1_config.yaml)
    robot_config_path: Optional[str] = None

    # Topic mappings (populated from robot config or auto-detected)
    state_topics: List[str] = field(default_factory=list)
    action_topics: List[str] = field(default_factory=list)

    # Trim settings
    apply_trim: bool = True
    apply_exclude_regions: bool = True

    # Staleness thresholds
    quality_warning_multiplier: float = 2.0
    quality_error_multiplier: float = 4.0

    # ---- Conversion selection knobs (StartConversion.srv) ----
    # Empty / None = use defaults from robot_config (legacy behaviour).
    selected_cameras: List[str] = field(default_factory=list)
    camera_rotations: Dict[str, int] = field(default_factory=dict)
    image_resize: Optional[Tuple[int, int]] = None  # (height, width)
    selected_state_topics: List[str] = field(default_factory=list)
    selected_action_topics: List[str] = field(default_factory=list)
    selected_joints: List[str] = field(default_factory=list)
    # Audit metadata for the root info.json conversion_config snapshot.
    source_rosbags: List[str] = field(default_factory=list)


@dataclass
class EpisodeData:
    """Data container for a single episode."""

    episode_index: int
    timestamps: List[float] = field(default_factory=list)
    observation_state: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    video_files: Dict[str, Path] = field(default_factory=dict)
    tasks: List[str] = field(default_factory=list)
    length: int = 0
    source_path: Optional[Path] = None
    recording_mode: str = "single"
    full_episode_index: Optional[int] = None
    subtask_index: int = 0
    subtask_total: int = 0
    subtask_instruction: str = ""
    subtask_instructions: List[str] = field(default_factory=list)
    subtask_segments: List[Dict[str, Any]] = field(default_factory=list)
    subtask_indices: List[int] = field(default_factory=list)
    task_name: str = ""
    # Absolute MCAP log_time (seconds since epoch) for each row of
    # ``timestamps``. Populated by ``_resample_to_fps`` so the video
    # sync step can map per-camera MP4 frames onto the same grid.
    grid_log_times_sec: List[float] = field(default_factory=list)
    frame_reuse_reports: List[Dict[str, Any]] = field(default_factory=list)
    observation_state_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)


class RosbagToLerobotConverterBase:
    """
    Base class for ROSbag-to-LeRobot conversion.

    Owns all logic that is independent of the LeRobot dataset format
    version: bag reading, joint extraction, causal-sync resampling,
    video discovery, feature building, and per-episode statistics.

    Format-specific subclasses (``RosbagToLerobotConverter`` for v2.1,
    ``RosbagToLerobotV30Converter`` for v3.0) supply the writers.
    """

    def __init__(self, config: ConversionConfig, logger=None):
        self.config = config
        self.logger = logger
        self._metadata_manager = MetadataManager(logger)
        self._video_extractor = VideoMetadataExtractor(logger)

        self._features: Dict[str, Dict] = {}
        self._tasks: Dict[int, str] = {}
        self._task_to_index: Dict[str, int] = {}
        self._episodes: Dict[int, Dict] = {}
        self._episodes_stats: Dict[int, Dict] = {}
        self._total_frames = 0
        self._total_episodes = 0
        self._staleness_reports: Dict[int, Dict[str, StalenessMetrics]] = {}
        # Init for v3.0's enable_quality_report path; populated by future
        # quality-report machinery, empty dict short-circuits the writer.
        self._quality_reports: Dict[int, Dict[str, Any]] = {}

        self._state_joint_names: List[str] = []
        self._action_joint_names: List[str] = []
        self._camera_mapping: Dict[str, str] = {}  # topic -> camera_name
        self._camera_rotations: Dict[str, int] = {}  # camera_name -> rotation_deg
        self._joint_order: List[str] = []  # Ordered list of joints to include
        self._joint_order_by_group: Dict[str, List[str]] = {}  # group_key -> joint names
        self._state_topic_key_map: Dict[str, str] = {}  # topic -> group key
        self._action_topic_key_map: Dict[str, str] = {}  # topic -> group key

        # Per-episode bisect-keys cache for ``_find_previous_value(_in_list)``.
        # Previously this lived in a mutable-default-arg dict on the method
        # (the classic Python footgun) — it survived across episodes within
        # a worker process and could in principle return stale keys if a
        # new ``state_messages`` list reused a GC'd address. Instance scope
        # plus an explicit ``clear()`` in ``_extract_joint_data`` keeps the
        # per-call lookup speedup with zero risk of cross-episode leakage.
        self._bisect_keys_cache: Dict[int, List[float]] = {}
        self._video_stats_sidecar_cache: Dict[Path, Dict[str, Any]] = {}
        self._video_frame_count_cache: Dict[Tuple[str, int, int], Optional[int]] = {}
        self._video_streams_probe_cache: Dict[
            Tuple[str, int, int], Optional[Dict[str, Any]]
        ] = {}
        self._frame_reuse_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._frame_reuse_lock = threading.Lock()

        # Load robot config — caller's explicit path wins, otherwise
        # auto-resolve via the schema (which searches the standard
        # ORCHESTRATOR_CONFIG_PATH / /orchestrator_config / source-tree
        # locations). The ROS service callers don't pass
        # ``robot_config_path`` on every request, so without this
        # fallback ``_joint_order_by_group`` stays empty and the
        # downstream ``_build_features`` ``names`` field collapses to
        # the placeholder ``joint_N`` list — the symptom the user
        # spotted in v2.1 ``info.json``.
        robot_config_path = config.robot_config_path
        if not robot_config_path and config.robot_type and config.robot_type != "unknown":
            try:
                resolved = robot_schema.find_robot_config_path(config.robot_type)
                robot_config_path = str(resolved)
                self._log_info(
                    f"robot_config_path auto-resolved for "
                    f"robot_type={config.robot_type!r}: {robot_config_path}"
                )
            except Exception as exc:
                self._log_warning(
                    f"robot_config_path auto-resolve failed for "
                    f"robot_type={config.robot_type!r}: {exc!r}"
                )
        if robot_config_path:
            self._load_robot_config_file(robot_config_path)

        # Apply selection knobs after robot_config has populated the
        # discovered defaults — empty selection lists mean "use all
        # discovered" so the legacy behaviour is preserved.
        self._apply_selection_knobs()

    def _log_info(self, msg: str):
        if not _converter_info_logs_enabled():
            return
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"[ERROR] {msg}")

    def _log_warning(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

    def _reset_frame_reuse_reports(self) -> None:
        with self._frame_reuse_lock:
            self._frame_reuse_reports = {}

    def _remember_frame_reuse_report(
        self,
        report: Optional[Dict[str, Any]],
    ) -> None:
        if not report:
            return
        key = (int(report["episode_index"]), str(report["camera"]))
        with self._frame_reuse_lock:
            self._frame_reuse_reports[key] = report

    def _remember_episode_frame_reuse_report(
        self,
        episode: EpisodeData,
        report: Optional[Dict[str, Any]],
    ) -> None:
        if not report:
            return
        key = (int(report["episode_index"]), str(report["camera"]))
        kept: List[Dict[str, Any]] = []
        replaced = False
        frame_reuse_reports = self._episode_frame_reuse_reports(episode)
        for existing in frame_reuse_reports:
            existing_key = (
                int(existing.get("episode_index", episode.episode_index)),
                str(existing.get("camera", "")),
            )
            if existing_key == key:
                kept.append(report)
                replaced = True
            else:
                kept.append(existing)
        if not replaced:
            kept.append(report)
        episode.frame_reuse_reports = kept

    @staticmethod
    def _episode_frame_reuse_reports(episode: EpisodeData) -> List[Dict[str, Any]]:
        reports = getattr(episode, "frame_reuse_reports", None)
        if reports is None:
            reports = []
            episode.frame_reuse_reports = reports
        return reports

    def _record_frame_reuse_report(
        self,
        *,
        episode: EpisodeData,
        camera_name: str,
        indices: np.ndarray,
        grid_ns: np.ndarray,
        frame_timestamps: Any,
    ) -> None:
        from cyclo_data.reader.frame_timestamps import build_frame_reuse_report

        report = build_frame_reuse_report(
            indices,
            grid_ns,
            frame_timestamps,
            episode_index=int(episode.episode_index),
            camera=camera_name,
            fps=int(self.config.fps),
            time_source="header",
        )
        self._remember_episode_frame_reuse_report(episode, report)
        self._remember_frame_reuse_report(report)

    @staticmethod
    def _shift_frame_reuse_report(
        report: Dict[str, Any],
        *,
        episode_index: int,
        frame_offset: int,
    ) -> Dict[str, Any]:
        shifted = dict(report)
        shifted["episode_index"] = int(episode_index)
        shifted_runs: List[Dict[str, Any]] = []
        for run in report.get("runs") or []:
            shifted_run = dict(run)
            shifted_run["target_start_frame"] = (
                int(shifted_run["target_start_frame"]) + int(frame_offset)
            )
            shifted_run["target_end_frame"] = (
                int(shifted_run["target_end_frame"]) + int(frame_offset)
            )
            shifted_runs.append(shifted_run)
        shifted["runs"] = shifted_runs
        return shifted

    @staticmethod
    def _merge_frame_reuse_reports(
        reports: List[Dict[str, Any]],
        *,
        episode_index: int,
        camera: str,
        total_target_frames: int,
    ) -> Optional[Dict[str, Any]]:
        if not reports:
            return None
        reports = sorted(
            reports,
            key=lambda item: (
                min(
                    [int(run["target_start_frame"]) for run in item.get("runs") or []]
                    or [0]
                ),
                int(item.get("episode_index", episode_index)),
            ),
        )
        runs: List[Dict[str, Any]] = []
        total_source_frames = 0
        reused_target_frames = 0
        clamped_before_first_count = 0
        time_sources = set()
        target_frames_from_parts = 0
        for report in reports:
            runs.extend([dict(run) for run in report.get("runs") or []])
            total_source_frames += int(report.get("total_source_frames", 0) or 0)
            reused_target_frames += int(report.get("reused_target_frames", 0) or 0)
            clamped_before_first_count += int(
                report.get("clamped_before_first_count", 0) or 0
            )
            target_frames_from_parts += int(report.get("total_target_frames", 0) or 0)
            time_sources.add(str(report.get("time_source", "header") or "header"))
        total_target = int(total_target_frames or target_frames_from_parts)
        reuse_ratio = (
            float(reused_target_frames) / float(total_target)
            if total_target > 0 else 0.0
        )
        return {
            "episode_index": int(episode_index),
            "camera": str(camera),
            "target_fps": int(reports[0].get("target_fps", 0) or 0),
            "time_source": "recv" if "recv" in time_sources else "header",
            "total_target_frames": total_target,
            "total_source_frames": int(total_source_frames),
            "reused_target_frames": int(reused_target_frames),
            "reuse_ratio": reuse_ratio,
            "clamped_before_first_count": int(clamped_before_first_count),
            "runs": sorted(
                runs,
                key=lambda run: (
                    int(run["target_start_frame"]),
                    int(run["target_end_frame"]),
                    int(run.get("source_frame_index", 0) or 0),
                ),
            ),
        }

    def _collect_episode_frame_reuse_reports(
        self,
        episodes_data: List[EpisodeData],
    ) -> None:
        for episode in episodes_data:
            for report in self._episode_frame_reuse_reports(episode):
                report = dict(report)
                report["episode_index"] = int(episode.episode_index)
                self._remember_frame_reuse_report(report)

    def _record_frame_reuse_for_video(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> None:
        if not episode.grid_log_times_sec:
            return
        sidecar = Path(video_path).parent / f"{camera_name}_timestamps.parquet"
        if not sidecar.exists():
            return
        try:
            from cyclo_data.reader.frame_timestamps import load_frame_timestamps

            frame_timestamps = load_frame_timestamps(sidecar, camera_name)
            grid_ns = (
                np.asarray(
                    episode.grid_log_times_sec[: int(episode.length)],
                    dtype=np.float64,
                )
                * 1_000_000_000
            ).astype(np.int64)
            indices = frame_timestamps.map_to_grid(grid_ns, time_source="header")
            self._record_frame_reuse_report(
                episode=episode,
                camera_name=camera_name,
                indices=indices,
                grid_ns=grid_ns,
                frame_timestamps=frame_timestamps,
            )
        except Exception as exc:  # noqa: BLE001 - metadata must not break conversion
            self._log_warning(
                f"{camera_name}: failed to build frame reuse report for "
                f"episode {int(episode.episode_index)} ({exc!r})"
            )

    def _write_frame_reuse_metadata(self, output_dir: Path) -> None:
        path = Path(output_dir) / "meta" / "frame_reuse.parquet"
        legacy_paths = [
            Path(output_dir) / "meta" / "frame_reuse.jsonl",
            Path(output_dir) / "meta" / "frame_reuse.json.gz",
        ]
        with self._frame_reuse_lock:
            reports = sorted(
                self._frame_reuse_reports.values(),
                key=lambda item: (int(item["episode_index"]), str(item["camera"])),
            )
        rows: List[Dict[str, Any]] = []
        for report in reports:
            episode_index = int(report["episode_index"])
            camera = str(report["camera"])
            for run in report.get("runs") or []:
                for target_frame_index in range(
                    int(run["target_start_frame"]),
                    int(run["target_end_frame"]) + 1,
                ):
                    rows.append({
                        "episode_index": int(episode_index),
                        "camera": camera,
                        "target_frame_index": int(target_frame_index),
                    })
        rows.sort(
            key=lambda item: (
                int(item["episode_index"]),
                str(item["camera"]),
                int(item["target_frame_index"]),
            )
        )

        if not rows:
            path.unlink(missing_ok=True)
            for legacy_path in legacy_paths:
                legacy_path.unlink(missing_ok=True)
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                prefix=path.stem + ".",
                suffix=".tmp",
                dir=str(path.parent),
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
            table = pa.table(
                {
                    "episode_index": [
                        int(row["episode_index"]) for row in rows
                    ],
                    "camera": [str(row["camera"]) for row in rows],
                    "target_frame_index": [
                        int(row["target_frame_index"]) for row in rows
                    ],
                },
                schema=pa.schema([
                    ("episode_index", pa.int32()),
                    ("camera", pa.string()),
                    ("target_frame_index", pa.int32()),
                ]),
            )
            compression = None
            for candidate in ("zstd", "snappy"):
                if pa.Codec.is_available(candidate):
                    compression = candidate
                    break
            pq.write_table(
                table,
                tmp_path,
                compression=compression,
                use_dictionary=["camera"],
            )
            os.replace(tmp_path, path)
            for legacy_path in legacy_paths:
                legacy_path.unlink(missing_ok=True)
            self._log_info(f"Wrote frame reuse metadata: {path}")
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise

    def _load_robot_config_file(self, config_path: str):
        """Load robot config from YAML file (e.g., ffw_sg2_rev1_config.yaml).

        Phase 4: yaml is VLA-semantic. State / action topic→key maps still
        carry the legacy ``follower_<group>`` / ``leader_<modality>`` keys
        because _resolve_filter_target_names + _merge_*_messages downstream
        key off that prefix to align state and action slices.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            self._log_error(f"Robot config not found: {config_path}")
            return

        # Recover robot_type from the yaml's top-level key first.
        # ConversionConfig.robot_type defaults to "unknown" — without this
        # step we'd pass an empty/wrong key to schema.load_robot_section
        # and bail out, leaving every topic→key map blank and forcing the
        # downstream heuristics (which can't recover yaml-driven group
        # ordering) to take over.
        try:
            import yaml as _yaml
            with open(config_path, 'r') as f:
                raw = _yaml.safe_load(f) or {}
            params = raw.get("orchestrator", {}).get("ros__parameters", {})
        except Exception as e:
            self._log_error(f"Failed to read robot config: {e}")
            return
        if not params:
            self._log_error(
                f"orchestrator.ros__parameters missing in {config_path}"
            )
            return

        if self.config.robot_type == "unknown" or self.config.robot_type not in params:
            self.config.robot_type = next(iter(params.keys()))

        try:
            section = robot_schema.load_robot_section(
                self.config.robot_type,
                explicit_path=str(config_path),
            )
        except Exception as e:
            self._log_error(f"Failed to load robot config: {e}")
            return

        state_groups = robot_schema.get_state_groups(section)
        action_groups = robot_schema.get_action_groups(section)
        image_groups = robot_schema.get_image_topics(section)

        state_topics: Dict[str, str] = {}
        for name, cfg in state_groups.items():
            key = f"follower_{name}"
            state_topics[key] = cfg["topic"]
            self._state_topic_key_map[cfg["topic"]] = key

        action_topics: Dict[str, str] = {}
        for modality, cfg in action_groups.items():
            key = f"leader_{modality}"
            action_topics[key] = cfg["topic"]
            self._action_topic_key_map[cfg["topic"]] = key

        self.config.state_topics = list(state_topics.values())
        self.config.action_topics = list(action_topics.values())
        self._log_info(
            f"Loaded topics — state: {list(state_topics.keys())}, "
            f"action: {list(action_topics.keys())}"
        )

        for cam_name, cfg in image_groups.items():
            self._camera_mapping[cfg["topic"]] = cam_name
            # ``rotation_deg`` is consumed at recording time by the
            # transcoder, not here. Kept on ``self._camera_rotations``
            # only as informational metadata.
            self._camera_rotations[cam_name] = int(
                cfg.get("rotation_deg") or 0
            )

        # _joint_order_by_group keyed by ``leader_<modality>`` — preserved
        # for _resolve_filter_target_names and the per-group merge logic.
        # Flat _joint_order is the concatenation in yaml insertion order.
        flattened: List[str] = []
        self._joint_order_by_group = {}
        for modality, cfg in action_groups.items():
            joints = list(cfg["joint_names"])
            self._joint_order_by_group[f"leader_{modality}"] = joints
            flattened.extend(joints)
        self._joint_order = flattened
        self._log_info(
            f"Loaded joint_order: {list(self._joint_order_by_group.keys())} "
            f"(total {len(self._joint_order)} joints)"
        )

    def _joint_names_from_config(self, group_prefix: str) -> List[str]:
        """Concatenate joint_order entries whose group key starts with ``prefix``.

        Mirrors the order used by _merge_state_messages /
        _merge_action_messages: groups sorted alphabetically, joints
        within each group taken verbatim from joint_order_by_group.
        Used as a fallback in _build_features when the per-episode
        accumulators (_state_joint_names / _action_joint_names) are
        empty — that happens in the parallel parsing path where worker
        children's attributes don't propagate back to the main process.
        """
        keys = sorted(
            k for k in self._joint_order_by_group if k.startswith(group_prefix)
        )
        names: List[str] = []
        for k in keys:
            names.extend(self._joint_order_by_group[k])
        return names

    def _resolve_filter_target_names(self, group_key: str) -> List[str]:
        """Resolve the ordered joint_names a given state/action group should
        be sliced down to.

        State and action are symmetric for VLA: every joint we command
        (action's leader_<X>) we also observe (state's follower_<X>).
        Predecessor configs only listed leader_* in joint_order; deriving
        the state-side filter from that single source keeps the yaml
        non-redundant and matches that expectation.

        Resolution order for a state group_key:
          1. Direct hit in joint_order_by_group (caller's explicit override).
          2. follower_<modality> → leader_<modality> joint_names.
          3. follower_upper_body (collapsed multi-arm follower) → union of
             every leader_* group except leader_mobile, in joint_order
             insertion order. Matches the per-arm 8/8/2/1 layout the
             leaders advertise.
        Returns an empty list when nothing maps — callers treat that as
        "no filter, take the message verbatim".
        """
        if group_key in self._joint_order_by_group:
            return list(self._joint_order_by_group[group_key])

        if group_key.startswith("follower_"):
            modality = group_key[len("follower_"):]
            leader_key = f"leader_{modality}"
            if leader_key in self._joint_order_by_group:
                return list(self._joint_order_by_group[leader_key])

            if group_key == "follower_upper_body":
                names: List[str] = []
                for k, joints in self._joint_order_by_group.items():
                    if not k.startswith("leader_"):
                        continue
                    if "mobile" in k.lower():
                        continue
                    names.extend(joints)
                return names

        return []

    def _apply_selection_knobs(self) -> None:
        """Apply ConversionConfig selection lists to the discovered defaults.

        Called from ``__init__`` after the robot_config has populated
        state_topics / action_topics / _joint_order / _camera_mapping.
        Empty selection lists are no-ops.
        """
        # State topic subset.
        if self.config.selected_state_topics:
            wanted = set(self.config.selected_state_topics)
            kept = [t for t in self.config.state_topics if t in wanted]
            if kept:
                self._log_info(
                    f"selected_state_topics filter: "
                    f"{len(self.config.state_topics)} → {len(kept)}"
                )
                self.config.state_topics = kept
            else:
                self._log_warning(
                    f"selected_state_topics {self.config.selected_state_topics} "
                    f"didn't match any of {self.config.state_topics}; "
                    f"keeping all"
                )

        # Action topic subset.
        if self.config.selected_action_topics:
            wanted = set(self.config.selected_action_topics)
            kept = [t for t in self.config.action_topics if t in wanted]
            if kept:
                self._log_info(
                    f"selected_action_topics filter: "
                    f"{len(self.config.action_topics)} → {len(kept)}"
                )
                self.config.action_topics = kept
            else:
                self._log_warning(
                    f"selected_action_topics {self.config.selected_action_topics} "
                    f"didn't match any of {self.config.action_topics}; "
                    f"keeping all"
                )

        # Joint subset / reorder. Preserve the order from
        # selected_joints (caller's intent) rather than _joint_order's
        # original order.
        if self.config.selected_joints:
            available = set(self._joint_order)
            kept = [j for j in self.config.selected_joints if j in available]
            if kept:
                self._log_info(
                    f"selected_joints filter: "
                    f"{len(self._joint_order)} → {len(kept)}"
                )
                self._joint_order = kept
                # Also subset each per-group list to the survivors.
                kept_set = set(kept)
                self._joint_order_by_group = {
                    g: [j for j in joints if j in kept_set]
                    for g, joints in self._joint_order_by_group.items()
                }
                # Drop empty groups.
                self._joint_order_by_group = {
                    g: joints for g, joints in self._joint_order_by_group.items()
                    if joints
                }
            else:
                self._log_warning(
                    f"selected_joints {self.config.selected_joints} "
                    f"didn't match any of {self._joint_order}; keeping all"
                )

    def _log_staleness_summary(self, staleness_metrics: Dict[str, StalenessMetrics]):
        for topic, metrics in staleness_metrics.items():
            if metrics.status == "GOOD":
                continue
            self._log_warning(
                f"Staleness {metrics.status} for {topic}: "
                f"warnings={metrics.stale_warning_count}, errors={metrics.stale_error_count}, "
                f"max={metrics.max_staleness_ms:.1f}ms, mean={metrics.mean_staleness_ms:.1f}ms"
            )

    def convert_single_rosbag(
        self,
        bag_path: Path,
        episode_index: int,
    ) -> Optional[EpisodeData]:
        bag_path = Path(bag_path)
        if not bag_path.exists():
            self._log_error(f"Bag path does not exist: {bag_path}")
            return None

        # Refuse to convert episodes whose background H.264 transcode
        # isn't finished — converting against the raw MJPEG source would
        # silently drop the yaml-driven rotation (and any other future
        # transcode-time treatment), producing a misoriented LeRobot
        # dataset. Better to fail loud and let the caller wait/retry.
        if not self._can_convert_transcode_state(bag_path):
            return None

        self._log_info(f"Converting rosbag: {bag_path} (episode {episode_index})")

        # Load per-episode robot_config.yaml if exists and no global config was loaded
        if not self.config.robot_config_path:
            robot_config = self._metadata_manager.load_robot_config(bag_path)
            if robot_config:
                self._update_config_from_robot_config(robot_config)

        trim_points = None
        exclude_regions = []
        if self.config.apply_trim:
            trim_points = self._metadata_manager.get_trim_points(bag_path)
        if self.config.apply_exclude_regions:
            exclude_regions = self._metadata_manager.get_exclude_regions(bag_path)

        episode_info = self._metadata_manager.load_episode_info(bag_path)
        prepared_cache_path = self._prepared_episode_cache_path(
            bag_path, episode_info, trim_points, exclude_regions
        )
        if prepared_cache_path is not None:
            cached_episode = self._load_prepared_episode_cache(
                prepared_cache_path,
                episode_index=episode_index,
                bag_path=bag_path,
            )
            if cached_episode is not None:
                self._log_info(
                    f"{bag_path.name}: reused prepared episode cache "
                    f"({cached_episode.length} frames)"
                )
                return cached_episode

        if self._is_archived_segment_episode(bag_path, episode_info):
            episode_data = self._convert_archived_segment_episode(
                bag_path, episode_index, episode_info
            )
        else:
            episode_data = self._extract_joint_data(
                bag_path, episode_index, trim_points, exclude_regions
            )
        if episode_data is None:
            return None
        episode_data.source_path = bag_path

        self._apply_episode_info(episode_data, episode_info)

        if (
            self.config.use_videos
            and not self._is_archived_segment_episode(bag_path, episode_info)
        ):
            video_files = self._find_video_files(bag_path)
            episode_data.video_files = video_files

            # Recording format v2 path: every camera has a sidecar parquet
            # under ``videos/<cam>_timestamps.parquet``. Build a synced MP4
            # per camera so that frame N == grid step N (LeRobot's video
            # reader assumes 1:1 with the parquet rows). For each grid
            # step we pick the most recent MP4 frame with
            # ``header_stamp_ns`` <= ``grid_log_time``. Camera sync must use
            # publisher/header time so network or rosbridge delays do not
            # shift images. The joint extraction step builds the grid from
            # observation.state header.stamp when available, while action
            # topics intentionally stay on MCAP/log-time because command
            # messages often have no publisher header.
            episode_data = self._sync_videos_to_grid(bag_path, episode_data)

            # Legacy fallback: when no sidecars exist (recordings made by
            # the v1 pipeline that called rosbag2mp4 before this rewrite)
            # the synced MP4 step is a no-op and the parquet rows still
            # need a 1:1 trim against the raw MP4 frame count.
            if video_files and not self._episode_has_sidecars(bag_path):
                video_frame_counts = {}
                for cam_name, vpath in video_files.items():
                    fc = self._get_video_frame_count(vpath)
                    if fc is not None:
                        video_frame_counts[cam_name] = fc

                if video_frame_counts:
                    target_frames = min(video_frame_counts.values())
                    if episode_data.length > target_frames:
                        excess = episode_data.length - target_frames
                        self._log_info(
                            f"Trimming parquet from {episode_data.length} to "
                            f"{target_frames} rows to match video frames "
                            f"(removing {excess} from end)"
                        )
                        episode_data.timestamps = episode_data.timestamps[:target_frames]
                        episode_data.observation_state = episode_data.observation_state[:target_frames]
                        episode_data.action = episode_data.action[:target_frames]
                        episode_data.grid_log_times_sec = (
                            episode_data.grid_log_times_sec[:target_frames]
                        )
                        episode_data.length = target_frames
        elif not self.config.use_videos:
            episode_data.video_files = {}

        self._assign_subtask_indices(episode_data)

        task_markers = self._metadata_manager.get_task_markers(bag_path)
        if task_markers:
            episode_data.tasks = list(
                set(m.get("instruction", "default_task") for m in task_markers)
            )
        else:
            # Fall back to episode_info.json which records the
            # task_instruction from the recording session.
            instruction = str(episode_info.get("task_instruction", "") or "")
            episode_data.tasks = [instruction or "default_task"]

        if prepared_cache_path is not None:
            self._store_prepared_episode_cache(prepared_cache_path, episode_data)

        return episode_data

    def _apply_episode_info(self, episode_data: EpisodeData, info: Dict[str, Any]) -> None:
        if not info:
            return
        episode_data.task_name = str(info.get("task_name", "") or "")
        episode_data.recording_mode = str(info.get("recording_mode", "single") or "single")
        try:
            episode_data.full_episode_index = int(
                info.get("full_episode_index", episode_data.episode_index)
            )
        except (TypeError, ValueError):
            episode_data.full_episode_index = episode_data.episode_index
        try:
            episode_data.subtask_index = int(info.get("subtask_index", 0) or 0)
        except (TypeError, ValueError):
            episode_data.subtask_index = 0
        try:
            episode_data.subtask_total = int(info.get("subtask_total", 0) or 0)
        except (TypeError, ValueError):
            episode_data.subtask_total = 0
        episode_data.subtask_instruction = str(
            info.get("subtask_instruction", "") or ""
        )
        raw_subtasks = info.get("subtask_instructions", []) or []
        if isinstance(raw_subtasks, list):
            episode_data.subtask_instructions = [
                str(item or "").strip()
                for item in raw_subtasks
                if str(item or "").strip()
            ]
        segments = info.get("segments")
        if not episode_data.subtask_instructions and isinstance(segments, list):
            episode_data.subtask_instructions = [
                str(segment.get("sub_task_instruction", "") or "").strip()
                for segment in segments
                if isinstance(segment, dict)
                and str(segment.get("sub_task_instruction", "") or "").strip()
            ]
        if isinstance(segments, list) and not episode_data.subtask_segments:
            normalized_segments: List[Dict[str, Any]] = []
            for idx, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    continue
                duration = segment.get("frame_duration")
                if not isinstance(duration, list) or len(duration) != 2:
                    continue
                try:
                    start = float(duration[0])
                    end = float(duration[1])
                except (TypeError, ValueError):
                    continue
                if end < start:
                    continue
                instruction = str(segment.get("sub_task_instruction", "") or "")
                normalized_segments.append({
                    "subtask_index": idx,
                    "sub_task_instruction": instruction,
                    "frame_duration": [start, end],
                })
            episode_data.subtask_segments = normalized_segments

    def _prepared_episode_cache_path(
        self,
        bag_path: Path,
        episode_info: Dict[str, Any],
        trim_points: Optional[Dict],
        exclude_regions: List[Dict],
    ) -> Optional[Path]:
        """Return cache path for a fully prepared EpisodeData object."""
        if os.environ.get(_PREPARED_EPISODE_CACHE_DISABLE_ENV):
            return None
        try:
            key_payload = self._prepared_episode_cache_key(
                Path(bag_path), episode_info, trim_points, exclude_regions
            )
        except OSError as exc:
            self._log_warning(
                f"{Path(bag_path).name}: prepared episode cache disabled "
                f"({exc!r})"
            )
            return None
        digest = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        root = Path(bag_path)
        cache_root = root if root.is_dir() else root.parent
        return (
            cache_root
            / ".cyclo_cache"
            / "prepared_episode"
            / f"{digest}.pickle"
        )

    def _prepared_episode_cache_key(
        self,
        bag_path: Path,
        episode_info: Dict[str, Any],
        trim_points: Optional[Dict],
        exclude_regions: List[Dict],
    ) -> Dict[str, Any]:
        sources = [
            self._file_signature(path)
            for path in self._prepared_episode_source_files(bag_path)
        ]
        if not any(str(item["path"]).endswith(".mcap") for item in sources):
            raise OSError(f"no MCAP files found for {bag_path}")
        return {
            "version": _PREPARED_EPISODE_CACHE_VERSION,
            "sources": sources,
            "episode_info": episode_info or {},
            "fps": int(self.config.fps),
            "use_videos": bool(self.config.use_videos),
            "state_topics": list(self.config.state_topics),
            "action_topics": list(self.config.action_topics),
            "selected_cameras": list(self.config.selected_cameras),
            "camera_rotations": dict(self.config.camera_rotations),
            "image_resize": (
                list(self.config.image_resize)
                if self.config.image_resize else None
            ),
            "selected_state_topics": list(self.config.selected_state_topics),
            "selected_action_topics": list(self.config.selected_action_topics),
            "selected_joints": list(self.config.selected_joints),
            "joint_order": list(self._joint_order),
            "joint_order_by_group": self._joint_order_by_group,
            "state_topic_key_map": self._state_topic_key_map,
            "action_topic_key_map": self._action_topic_key_map,
            "quality_warning_multiplier": float(
                self.config.quality_warning_multiplier
            ),
            "quality_error_multiplier": float(
                self.config.quality_error_multiplier
            ),
            "trim_points": trim_points or {},
            "exclude_regions": exclude_regions or [],
        }

    def _prepared_episode_source_files(self, bag_path: Path) -> List[Path]:
        files: List[Path] = []
        files.extend(self._episode_extract_source_files(bag_path))
        root = bag_path if bag_path.is_dir() else bag_path.parent
        for name in ("episode_info.json", "robot_config.yaml"):
            candidate = root / name
            if candidate.exists():
                files.append(candidate)
        videos_root = root / "videos"
        if self.config.use_videos and videos_root.exists():
            for path in sorted(videos_root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix == ".mp4" and path.stem.endswith("_synced"):
                    continue
                if path.suffix == ".mp4" or path.name.endswith("_timestamps.parquet"):
                    files.append(path)
        return sorted(set(files), key=lambda p: str(p))

    @staticmethod
    def _file_signature(path: Path) -> Dict[str, Any]:
        path = Path(path)
        stat = path.stat()
        return {
            "path": _fast_absolute_path(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @classmethod
    def _file_probe_cache_key(cls, path: Path) -> Tuple[str, int, int]:
        signature = cls._file_signature(Path(path))
        return (
            str(signature["path"]),
            int(signature["size"]),
            int(signature["mtime_ns"]),
        )

    def _load_prepared_episode_cache(
        self,
        cache_path: Path,
        *,
        episode_index: int,
        bag_path: Path,
    ) -> Optional[EpisodeData]:
        try:
            with open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{cache_path.name}: prepared cache unreadable ({exc!r})"
            )
            return None

        if (
            not isinstance(payload, dict)
            or payload.get("version") != _PREPARED_EPISODE_CACHE_VERSION
            or not isinstance(payload.get("episode"), EpisodeData)
        ):
            return None
        if not self._prepared_cache_outputs_match(payload):
            return None

        episode = payload["episode"]
        original_episode_index = int(payload.get("episode_index", episode.episode_index))
        original_full_index = episode.full_episode_index
        episode.episode_index = int(episode_index)
        if original_full_index == original_episode_index:
            episode.full_episode_index = int(episode_index)
        episode.source_path = Path(bag_path)
        try:
            episode._cyclo_prepared_cache_signature = self._file_signature(cache_path)
        except OSError:
            pass
        self._state_joint_names = list(payload.get("state_joint_names") or [])
        self._action_joint_names = list(payload.get("action_joint_names") or [])
        episode.observation_state_names = list(self._state_joint_names or [])
        episode.action_names = list(self._action_joint_names or [])
        staleness = payload.get("staleness_metrics")
        if isinstance(staleness, dict):
            self._staleness_reports[int(episode_index)] = staleness
        return episode

    def _prepared_cache_outputs_match(self, payload: Dict[str, Any]) -> bool:
        video_stats = payload.get("video_file_stats")
        if not isinstance(video_stats, dict):
            return False
        for raw in video_stats.values():
            if not isinstance(raw, dict):
                return False
            try:
                path = Path(raw["path"])
                current = self._file_signature(path)
            except (OSError, KeyError):
                return False
            if current != raw:
                return False
        return True

    def _store_prepared_episode_cache(
        self,
        cache_path: Path,
        episode: EpisodeData,
    ) -> None:
        # Avoid pinning transient raw-video fallback failures in the fast cache.
        if episode.video_files and not all(
            Path(path).stem.endswith("_synced")
            for path in episode.video_files.values()
        ):
            return
        video_file_stats: Dict[str, Dict[str, Any]] = {}
        try:
            for camera_name, video_path in episode.video_files.items():
                video_file_stats[camera_name] = self._file_signature(Path(video_path))
        except OSError:
            return

        payload = {
            "version": _PREPARED_EPISODE_CACHE_VERSION,
            "episode_index": int(episode.episode_index),
            "episode": episode,
            "state_joint_names": list(self._state_joint_names or []),
            "action_joint_names": list(self._action_joint_names or []),
            "staleness_metrics": self._staleness_reports.get(episode.episode_index, {}),
            "video_file_stats": video_file_stats,
        }
        tmp_path: Optional[Path] = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "wb",
                prefix=cache_path.stem + ".",
                suffix=".tmp",
                dir=str(cache_path.parent),
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            try:
                episode._cyclo_prepared_cache_signature = self._file_signature(
                    cache_path
                )
            except OSError:
                pass
        except Exception as exc:  # noqa: BLE001
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            self._log_warning(
                f"{Path(cache_path).name}: failed to write prepared cache "
                f"({exc!r})"
            )

    def _try_load_prepared_episode_for_bag(
        self,
        bag_path: Path,
        episode_index: int,
    ) -> Optional[EpisodeData]:
        """Load a fully prepared episode cache without entering worker setup."""
        if os.environ.get(_PREPARED_EPISODE_CACHE_DISABLE_ENV):
            return None

        bag_path = Path(bag_path)
        cache_root = bag_path if bag_path.is_dir() else bag_path.parent
        cache_dir = cache_root / ".cyclo_cache" / "prepared_episode"
        if not cache_dir.exists():
            return None

        if not self._can_convert_transcode_state(bag_path):
            return None

        if not self.config.robot_config_path:
            robot_config = self._metadata_manager.load_robot_config(bag_path)
            if robot_config:
                self._update_config_from_robot_config(robot_config)

        trim_points = None
        exclude_regions: List[Dict] = []
        if self.config.apply_trim:
            trim_points = self._metadata_manager.get_trim_points(bag_path)
        if self.config.apply_exclude_regions:
            exclude_regions = self._metadata_manager.get_exclude_regions(bag_path)

        episode_info = self._metadata_manager.load_episode_info(bag_path)
        cache_path = self._prepared_episode_cache_path(
            bag_path,
            episode_info,
            trim_points,
            exclude_regions,
        )
        if cache_path is None:
            return None

        cached_episode = self._load_prepared_episode_cache(
            cache_path,
            episode_index=episode_index,
            bag_path=bag_path,
        )
        if cached_episode is not None:
            self._log_info(
                f"{bag_path.name}: parent reused prepared episode cache "
                f"({cached_episode.length} frames)"
            )
        return cached_episode

    def _is_archived_segment_episode(
        self,
        bag_path: Path,
        episode_info: Dict[str, Any],
    ) -> bool:
        """Return True for full episodes with archived segment MCAP files."""
        segments = episode_info.get("segments")
        if not isinstance(segments, list) or len(segments) < 1:
            return False
        mcap_paths = self._segment_mcap_paths(Path(bag_path), len(segments))
        return len(mcap_paths) == len(segments)

    def _segment_mcap_paths(self, bag_path: Path, count: int) -> List[Path]:
        """Find per-subtask MCAP files in archived full-episode order."""
        bag_path = Path(bag_path)
        if not bag_path.is_dir():
            return []
        try:
            full_idx = int(bag_path.name)
        except ValueError:
            full_idx = None

        expected: List[Path] = []
        if full_idx is not None:
            expected = [bag_path / f"{full_idx}_{idx}.mcap" for idx in range(count)]
            if all(path.exists() for path in expected):
                return expected

        mcap_paths = sorted(bag_path.glob("*.mcap"))
        if len(mcap_paths) == count:
            return mcap_paths
        return []

    def _find_segment_video_files(
        self,
        bag_path: Path,
        segment_stem: str,
    ) -> Dict[str, Path]:
        """Find raw camera MP4 files for one archived subtask segment."""
        video_dir = Path(bag_path) / "videos" / segment_stem
        if not video_dir.exists():
            videos_root = Path(bag_path) / "videos"
            if videos_root.exists():
                raise FileNotFoundError(
                    f"{bag_path.name}: expected video segment directory "
                    f"{video_dir.relative_to(bag_path)} for {segment_stem}"
                )
            return {}
        video_files: Dict[str, Path] = {}
        for mp4_file in sorted(video_dir.glob("*.mp4")):
            if mp4_file.stem.endswith("_synced"):
                continue
            camera_name = self._get_camera_name_for_video(mp4_file.stem)
            video_files.setdefault(camera_name, mp4_file)
        return video_files

    def _convert_archived_segment_episode(
        self,
        bag_path: Path,
        episode_index: int,
        episode_info: Dict[str, Any],
    ) -> Optional[EpisodeData]:
        """Convert a full episode by resampling each subtask independently.

        Archived subtask episodes keep one MCAP/video folder per subtask
        under a single full-episode folder. Their MCAP log times may have
        wall-clock gaps when an operator cancelled and re-recorded a later
        subtask. Reading all MCAP files at once would create a LeRobot
        grid across those gaps and causal-sync would hold the previous
        state/image. Instead each segment is converted on its own time
        base, then rows and synced videos are stitched into a continuous
        LeRobot episode.
        """
        segments = episode_info.get("segments") or []
        mcap_paths = self._segment_mcap_paths(bag_path, len(segments))
        if not mcap_paths:
            return None

        segment_episodes: List[EpisodeData] = []
        for subtask_idx, mcap_path in enumerate(mcap_paths):
            segment_episode = self._extract_joint_data(
                mcap_path,
                episode_index,
                trim_points=None,
                exclude_regions=[],
            )
            if segment_episode is None:
                self._log_warning(
                    f"{bag_path.name}: subtask {subtask_idx} "
                    f"({mcap_path.name}) produced no rows"
                )
                continue
            if self.config.use_videos:
                segment_episode.video_files = self._find_segment_video_files(
                    bag_path,
                    mcap_path.stem,
                )
                segment_episode = self._sync_videos_to_grid(
                    mcap_path, segment_episode
                )
            else:
                segment_episode.video_files = {}
            segment = segments[subtask_idx] if subtask_idx < len(segments) else {}
            if isinstance(segment, dict):
                segment_episode.subtask_instruction = str(
                    segment.get("sub_task_instruction", "") or ""
                )
            segment_episodes.append(segment_episode)

        if not segment_episodes:
            return None

        if len(segment_episodes) == 1:
            single = segment_episodes[0]
            single.episode_index = episode_index
            single.source_path = bag_path
            single.recording_mode = "single"
            single.full_episode_index = episode_index
            single.task_name = str(episode_info.get("task_name", "") or "")
            single.subtask_index = 0
            single.subtask_total = 1
            single.subtask_instructions = [
                str(segment.get("sub_task_instruction", "") or "").strip()
                for segment in segments
                if isinstance(segment, dict)
                and str(segment.get("sub_task_instruction", "") or "").strip()
            ]
            instruction = (
                single.subtask_instruction
                or (
                    single.subtask_instructions[0]
                    if single.subtask_instructions else "Subtask 1"
                )
            )
            single.subtask_instruction = instruction
            single.subtask_indices = [0] * int(single.length)
            single.subtask_segments = [{
                "subtask_index": 0,
                "sub_task_instruction": instruction,
                "frame_duration": [
                    0.0,
                    int(single.length) / float(self.config.fps or DEFAULT_FPS),
                ],
            }]
            self._log_info(
                f"{bag_path.name}: converted 1 archived subtask segment "
                "without row/video stitching"
            )
            return single

        fps = float(self.config.fps or DEFAULT_FPS)
        stitched = EpisodeData(
            episode_index=episode_index,
            source_path=bag_path,
            recording_mode="single",
            full_episode_index=episode_index,
            task_name=str(episode_info.get("task_name", "") or ""),
            subtask_instructions=[
                str(segment.get("sub_task_instruction", "") or "").strip()
                for segment in segments
                if isinstance(segment, dict)
            ],
        )

        frame_cursor = 0
        frame_reuse_by_camera: Dict[str, List[Dict[str, Any]]] = {}
        for subtask_idx, segment_episode in enumerate(segment_episodes):
            length = int(segment_episode.length)
            if length <= 0:
                continue
            start_frame = frame_cursor
            if not stitched.observation_state_names:
                stitched.observation_state_names = list(
                    segment_episode.observation_state_names or []
                )
            if not stitched.action_names:
                stitched.action_names = list(segment_episode.action_names or [])
            stitched.observation_state.extend(segment_episode.observation_state)
            stitched.action.extend(segment_episode.action)
            stitched.timestamps.extend(
                [(frame_cursor + offset) / fps for offset in range(length)]
            )
            # ``grid_log_times_sec`` is no longer used after per-segment
            # video sync, but keep it continuous for diagnostics/stats.
            stitched.grid_log_times_sec.extend(
                [(frame_cursor + offset) / fps for offset in range(length)]
            )
            stitched.subtask_indices.extend([subtask_idx] * length)
            instruction = (
                segment_episode.subtask_instruction
                or (
                    stitched.subtask_instructions[subtask_idx]
                    if subtask_idx < len(stitched.subtask_instructions)
                    else f"Subtask {subtask_idx + 1}"
                )
            )
            end_frame = frame_cursor + length
            stitched.subtask_segments.append({
                "subtask_index": subtask_idx,
                "sub_task_instruction": instruction,
                "frame_duration": [start_frame / fps, end_frame / fps],
            })
            for report in self._episode_frame_reuse_reports(segment_episode):
                camera = str(report.get("camera", ""))
                if not camera:
                    continue
                frame_reuse_by_camera.setdefault(camera, []).append(
                    self._shift_frame_reuse_report(
                        report,
                        episode_index=episode_index,
                        frame_offset=start_frame,
                    )
                )
            frame_cursor = end_frame

        stitched.length = len(stitched.timestamps)
        for camera, reports in frame_reuse_by_camera.items():
            merged = self._merge_frame_reuse_reports(
                reports,
                episode_index=episode_index,
                camera=camera,
                total_target_frames=stitched.length,
            )
            if merged is not None:
                stitched.frame_reuse_reports.append(merged)
        if self.config.use_videos:
            stitched.video_files = self._stitch_subtask_videos(
                episode_index,
                segment_episodes,
            )
        else:
            stitched.video_files = {}
        self._log_info(
            f"{bag_path.name}: converted {len(segment_episodes)} archived "
            f"subtask segment(s) into {stitched.length} continuous frames"
        )
        return stitched

    def _assign_subtask_indices(self, episode_data: EpisodeData) -> None:
        """Map each output row timestamp to its source subtask segment."""
        if not episode_data.subtask_segments or episode_data.length <= 0:
            episode_data.subtask_indices = []
            return

        ordered = sorted(
            episode_data.subtask_segments,
            key=lambda segment: segment["frame_duration"][0],
        )
        last_idx = ordered[-1]["subtask_index"]
        indices: List[int] = []
        for ts in episode_data.timestamps[:episode_data.length]:
            value = last_idx
            for segment in ordered:
                start, end = segment["frame_duration"]
                if start <= float(ts) < end:
                    value = int(segment["subtask_index"])
                    break
            indices.append(value)
        episode_data.subtask_indices = indices

    def _collect_task_names(self, episodes_data: List[EpisodeData]) -> None:
        """Keep a lightweight task -> task_name map for writer metadata."""
        self._task_names_by_task: Dict[str, str] = {}
        for episode in episodes_data:
            task = episode.tasks[0] if episode.tasks else "default_task"
            task_name = episode.task_name or task
            self._task_names_by_task.setdefault(task, task_name)

    def _subtask_instruction_map(self, episode: EpisodeData) -> Dict[int, str]:
        """Return subtask index -> instruction for an episode."""
        mapping: Dict[int, str] = {}
        for segment in episode.subtask_segments:
            try:
                idx = int(segment.get("subtask_index", 0))
            except (TypeError, ValueError):
                continue
            instruction = str(segment.get("sub_task_instruction", "") or "").strip()
            if instruction:
                mapping[idx] = instruction

        for idx, instruction in enumerate(episode.subtask_instructions):
            clean = str(instruction or "").strip()
            if clean:
                mapping.setdefault(idx, clean)
        return mapping

    def _subtask_rows_for_dataset(
        self,
        episodes_data: List[EpisodeData],
    ) -> List[Dict[str, Any]]:
        """Build rows for meta/subtasks.parquet."""
        by_index: Dict[int, str] = {}
        for episode in episodes_data:
            for idx, instruction in self._subtask_instruction_map(episode).items():
                by_index.setdefault(idx, instruction)
        return [
            {"subtask_index": idx, "subtask": by_index[idx]}
            for idx in sorted(by_index)
        ]

    def _subtask_annotations_for_episode(
        self,
        episode: EpisodeData,
    ) -> List[Dict[str, Any]]:
        """Build frame-index based subtask annotations for one episode."""
        if not episode.subtask_segments and not episode.subtask_indices:
            return []

        instruction_by_index = self._subtask_instruction_map(episode)
        fps = float(self.config.fps or DEFAULT_FPS)
        annotations: List[Dict[str, Any]] = []

        if episode.subtask_indices and len(episode.subtask_indices) == episode.length:
            start_frame = 0
            current_idx = int(episode.subtask_indices[0])
            for frame_idx, idx_value in enumerate(episode.subtask_indices[1:], start=1):
                idx = int(idx_value)
                if idx == current_idx:
                    continue
                annotations.append({
                    "sub_task_idx": current_idx,
                    "sub_task_instruction": instruction_by_index.get(
                        current_idx,
                        f"Subtask {current_idx + 1}",
                    ),
                    "frame_duration": [int(start_frame), int(frame_idx)],
                })
                start_frame = frame_idx
                current_idx = idx
            annotations.append({
                "sub_task_idx": current_idx,
                "sub_task_instruction": instruction_by_index.get(
                    current_idx,
                    f"Subtask {current_idx + 1}",
                ),
                "frame_duration": [int(start_frame), int(episode.length)],
            })
            return annotations

        for segment in episode.subtask_segments:
            try:
                idx = int(segment.get("subtask_index", 0))
                start, end = segment.get("frame_duration", [0.0, 0.0])
                start_f = max(0, min(episode.length, int(round(float(start) * fps))))
                end_f = max(start_f, min(episode.length, int(round(float(end) * fps))))
            except (TypeError, ValueError):
                continue
            annotations.append({
                "sub_task_idx": idx,
                "sub_task_instruction": instruction_by_index.get(
                    idx,
                    f"Subtask {idx + 1}",
                ),
                "frame_duration": [start_f, end_f],
            })
        return annotations

    def _annotation_chunk_dir_name(self, chunk_idx: int) -> str:
        return f"chunk-{chunk_idx:03d}"

    def _annotation_episode_filename(self, episode_idx: int) -> str:
        return f"episode_{episode_idx:06d}.json"

    def _episode_chunk_index(self, episode_idx: int) -> int:
        return episode_idx // int(self.config.chunks_size or DEFAULT_CHUNK_SIZE)

    def _write_subtasks_parquet(
        self,
        output_dir: Path,
        episodes_data: List[EpisodeData],
    ) -> None:
        """Write optional LeRobot subtask lookup metadata."""
        rows = self._subtask_rows_for_dataset(episodes_data)
        if not rows:
            return
        path = Path(output_dir) / "meta" / "subtasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "subtask_index": pa.array(
                [int(row["subtask_index"]) for row in rows],
                type=pa.int64(),
            ),
            "subtask": pa.array(
                [str(row["subtask"]) for row in rows],
                type=pa.string(),
            ),
        })
        pq.write_table(table, path)
        self._log_info(f"Wrote subtasks metadata: {path}")

    def _write_subtask_annotations(
        self,
        output_dir: Path,
        episodes_data: List[EpisodeData],
    ) -> None:
        """Write per-episode subtask annotations without skill/primitive data."""
        output_dir = Path(output_dir)
        wrote_any = False
        for episode in episodes_data:
            annotations = self._subtask_annotations_for_episode(episode)
            if not annotations:
                continue
            ep_idx = episode.episode_index
            chunk_idx = self._episode_chunk_index(ep_idx)
            path = (
                output_dir
                / "annotations"
                / self._annotation_chunk_dir_name(chunk_idx)
                / self._annotation_episode_filename(ep_idx)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            task = episode.tasks[0] if episode.tasks else "default_task"
            payload = {
                "task_name": episode.task_name or task,
                "data_folder": "",
                "meta_data": {
                    "task_duration": int(episode.length),
                    "valid_duration": [0, int(episode.length)],
                },
                "sub_task_annotation": annotations,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            wrote_any = True
        if wrote_any:
            self._log_info(f"Wrote subtask annotations under: {output_dir / 'annotations'}")

    def _cleanup_output_temp_dirs(self) -> int:
        """Remove converter-local temporary folders from a final dataset."""
        removed = 0
        output_dir = Path(self.config.output_dir)
        for dirname in ("_subtask_video_concat", "_stitched_subtasks"):
            path = output_dir / dirname
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        if removed:
            self._log_info(
                f"Cleaned up {removed} temporary folder(s) under {output_dir}"
            )
        return removed

    def _cleanup_source_synced_cache(self, roots: List[Path]) -> int:
        """Optionally remove synced-video cache files produced during conversion."""
        if not os.environ.get(_VIDEO_SYNC_CLEAN_CACHE_ENV):
            return 0
        removed = 0
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if (
                    (path.suffix == ".mp4" and path.stem.endswith("_synced"))
                    or path.name.endswith("_synced.cache.json")
                ):
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
        if removed:
            self._log_info(f"Cleaned up {removed} synced video cache file(s)")
        return removed

    def prepare_episodes_for_writing(
        self,
        episodes_data: List[EpisodeData],
    ) -> List[EpisodeData]:
        """Collapse complete recorded subtask groups into long-horizon episodes."""
        if not any(ep.recording_mode == "subtask" for ep in episodes_data):
            if not self._validate_consistent_subtask_counts(episodes_data):
                return []
            return episodes_data

        grouped: Dict[int, List[EpisodeData]] = {}
        ordered_items: List[Tuple[int, EpisodeData | tuple[int, List[EpisodeData]]]] = []
        for ep in episodes_data:
            if ep.recording_mode != "subtask":
                ordered_items.append((ep.episode_index, ep))
                continue
            full_idx = (
                ep.full_episode_index
                if ep.full_episode_index is not None
                else ep.episode_index
            )
            grouped.setdefault(int(full_idx), []).append(ep)

        for full_idx, group in grouped.items():
            min_raw_idx = min(ep.episode_index for ep in group)
            ordered_items.append((min_raw_idx, (full_idx, group)))

        prepared: List[EpisodeData] = []
        for _, item in sorted(ordered_items, key=lambda pair: pair[0]):
            if isinstance(item, EpisodeData):
                prepared.append(item)
                continue
            full_idx, group = item
            stitched = self._stitch_subtask_group(full_idx, group)
            if stitched is not None:
                prepared.append(stitched)

        if not self._validate_consistent_subtask_counts(prepared):
            return []

        for new_idx, ep in enumerate(prepared):
            ep.episode_index = new_idx
        return prepared

    def _validate_consistent_subtask_counts(
        self,
        episodes_data: List[EpisodeData],
    ) -> bool:
        """Reject datasets that mix single-task and subtask episode schemas."""
        counts = {
            len(ep.subtask_segments or [])
            for ep in episodes_data
        }
        if len(counts) <= 1:
            return True
        if 0 in counts:
            self._log_error(
                "Cannot mix single-task and subtask episodes in one dataset: "
                f"found subtask counts={sorted(counts)}."
            )
            return False
        return True

    def _stitch_subtask_group(
        self,
        full_idx: int,
        group: List[EpisodeData],
    ) -> Optional[EpisodeData]:
        by_subtask = {ep.subtask_index: ep for ep in group}
        expected_total = max(
            [ep.subtask_total for ep in group if ep.subtask_total] or [len(group)]
        )
        missing = [idx for idx in range(expected_total) if idx not in by_subtask]
        if missing:
            self._log_warning(
                f"Skipping incomplete subtask group full_episode={full_idx}: "
                f"missing subtask(s) {missing}"
            )
            return None

        ordered = [by_subtask[idx] for idx in range(expected_total)]
        stitched = EpisodeData(
            episode_index=full_idx,
            tasks=[ordered[0].tasks[0] if ordered[0].tasks else "default_task"],
            source_path=ordered[0].source_path,
            recording_mode="stitched_subtask",
            full_episode_index=full_idx,
            subtask_index=0,
            subtask_total=expected_total,
            subtask_instructions=[
                ep.subtask_instruction or f"Subtask {ep.subtask_index + 1}"
                for ep in ordered
            ],
        )

        offset = 0.0
        step = 1.0 / float(self.config.fps or DEFAULT_FPS)
        for ep in ordered:
            if not ep.timestamps:
                continue
            base_ts = float(ep.timestamps[0])
            if not stitched.observation_state_names:
                stitched.observation_state_names = list(
                    ep.observation_state_names or []
                )
            if not stitched.action_names:
                stitched.action_names = list(ep.action_names or [])
            remapped = [float(ts) - base_ts + offset for ts in ep.timestamps]
            stitched.timestamps.extend(remapped)
            stitched.observation_state.extend(ep.observation_state)
            stitched.action.extend(ep.action)
            if ep.grid_log_times_sec:
                base_grid = float(ep.grid_log_times_sec[0])
                stitched.grid_log_times_sec.extend(
                    [float(ts) - base_grid + offset for ts in ep.grid_log_times_sec]
                )
            offset = (stitched.timestamps[-1] + step) if stitched.timestamps else offset

        stitched.length = len(stitched.timestamps)
        stitched.subtask_indices = []
        stitched.subtask_segments = []
        cursor = 0
        frame_reuse_by_camera: Dict[str, List[Dict[str, Any]]] = {}
        for idx, ep in enumerate(ordered):
            length = len(ep.timestamps)
            if length <= 0:
                continue
            start_frame = cursor
            end_frame = cursor + length
            stitched.subtask_indices.extend([idx] * length)
            instruction = (
                ep.subtask_instruction
                or (
                    ep.subtask_instructions[idx]
                    if idx < len(ep.subtask_instructions)
                    else f"Subtask {idx + 1}"
                )
            )
            start_s = start_frame / float(self.config.fps or DEFAULT_FPS)
            end_s = end_frame / float(self.config.fps or DEFAULT_FPS)
            stitched.subtask_segments.append({
                "subtask_index": idx,
                "sub_task_instruction": instruction,
                "frame_duration": [start_s, end_s],
            })
            for report in self._episode_frame_reuse_reports(ep):
                camera = str(report.get("camera", ""))
                if not camera:
                    continue
                frame_reuse_by_camera.setdefault(camera, []).append(
                    self._shift_frame_reuse_report(
                        report,
                        episode_index=full_idx,
                        frame_offset=start_frame,
                    )
                )
            cursor = end_frame
        for camera, reports in frame_reuse_by_camera.items():
            merged = self._merge_frame_reuse_reports(
                reports,
                episode_index=full_idx,
                camera=camera,
                total_target_frames=stitched.length,
            )
            if merged is not None:
                stitched.frame_reuse_reports.append(merged)
        stitched.video_files = self._stitch_subtask_videos(full_idx, ordered)
        self._log_info(
            f"Stitched full_episode={full_idx} from {len(ordered)} subtasks "
            f"({stitched.length} frames)"
        )
        return stitched

    def _stitch_subtask_videos(
        self,
        full_idx: int,
        ordered: List[EpisodeData],
    ) -> Dict[str, Path]:
        if not ordered or not all(ep.video_files for ep in ordered):
            return {}
        common_cameras = set(ordered[0].video_files)
        for ep in ordered[1:]:
            common_cameras &= set(ep.video_files)
        if not common_cameras:
            return {}
        if len(ordered) == 1:
            self._log_info(
                "Single subtask episode: reusing synced videos without "
                "stitch re-encode"
            )
            return {
                camera_name: Path(ordered[0].video_files[camera_name])
                for camera_name in sorted(common_cameras)
            }

        from cyclo_data.converter.video_sync import (
            _ffmpeg,
            _ffmpeg_threads_arg,
            _h264_encoder,
        )

        out_dir = Path(self.config.output_dir) / "_stitched_subtasks" / f"full_{full_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stitched: Dict[str, Path] = {}
        for camera_name in sorted(common_cameras):
            srcs = [Path(ep.video_files[camera_name]) for ep in ordered]
            out_path = out_dir / f"{camera_name}.mp4"
            list_path: Optional[Path] = None
            if out_path.exists():
                try:
                    out_mtime = out_path.stat().st_mtime
                    if (
                        all(src.exists() and src.stat().st_mtime <= out_mtime for src in srcs)
                        and self._video_decodes_successfully(out_path)
                    ):
                        stitched[camera_name] = out_path
                        continue
                except OSError:
                    pass

            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", suffix=".ffconcat", delete=False
                ) as list_file:
                    list_path = Path(list_file.name)
                    for src in srcs:
                        escaped = str(src.resolve()).replace("'", "'\\''")
                        list_file.write(f"file '{escaped}'\n")
                ffmpeg_bin = _ffmpeg()
                if self._try_prepare_segment_video_copy(
                    ffmpeg_bin, list_path, srcs, out_path
                ):
                    self._store_stitched_video_stats_from_sources(
                        out_path, camera_name, srcs
                    )
                    stitched[camera_name] = out_path
                    continue

                encoder_height, encoder_width = self._get_video_dimensions(
                    srcs[0]
                )
                encoder, encoder_opts = _h264_encoder(
                    ffmpeg_bin,
                    width=encoder_width,
                    height=encoder_height,
                )
                cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(list_path),
                    "-an",
                    "-vf", f"fps={int(self.config.fps or DEFAULT_FPS)}",
                    "-c:v", encoder,
                    *encoder_opts,
                    *_ffmpeg_threads_arg(),
                    "-pix_fmt", "yuv420p",
                    str(out_path),
                ]
                subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, check=True,
                )
                if not self._video_decodes_successfully(out_path):
                    out_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"stitched video failed decode validation: {out_path}"
                    )
                self._store_stitched_video_stats_from_sources(
                    out_path, camera_name, srcs
                )
                stitched[camera_name] = out_path
            except Exception as exc:  # noqa: BLE001
                self._log_warning(
                    f"Failed to stitch videos for full_episode={full_idx} "
                    f"camera={camera_name}: {exc}"
                )
            finally:
                if list_path is not None:
                    try:
                        list_path.unlink()
                    except Exception:
                        pass
        return stitched

    def _store_stitched_video_stats_from_sources(
        self,
        out_path: Path,
        camera_name: str,
        srcs: List[Path],
    ) -> None:
        """Cache stitched-video stats by merging source video stats."""
        stats_parts: List[Dict[str, Any]] = []
        for src in srcs:
            stats = self._load_precomputed_video_stats(Path(src), camera_name)
            if not stats:
                return
            stats_parts.append(stats)
        merged = self._merge_video_stats(stats_parts)
        if merged:
            self._store_video_stats_cached(out_path, camera_name, merged)

    @staticmethod
    def _merge_video_stats(stats_parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Merge RGB video stats sidecars without re-decoding video frames."""
        if not stats_parts:
            return None

        counts: List[float] = []
        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        mins: List[np.ndarray] = []
        maxs: List[np.ndarray] = []
        try:
            for stats in stats_parts:
                count = float((stats.get("count") or [0])[0])
                if count <= 0:
                    return None
                counts.append(count)

                def _channels(key: str) -> np.ndarray:
                    return np.asarray(stats[key], dtype=np.float64).reshape(3)

                means.append(_channels("mean"))
                stds.append(_channels("std"))
                mins.append(_channels("min"))
                maxs.append(_channels("max"))

            weights = np.asarray(counts, dtype=np.float64)
            total = float(weights.sum())
            mean_arr = np.average(np.vstack(means), axis=0, weights=weights)
            second_moment = np.average(
                np.vstack([std * std + mean * mean for std, mean in zip(stds, means)]),
                axis=0,
                weights=weights,
            )
            std_arr = np.sqrt(np.maximum(second_moment - mean_arr * mean_arr, 0.0))
            min_arr = np.min(np.vstack(mins), axis=0)
            max_arr = np.max(np.vstack(maxs), axis=0)

            def _wrap(values: np.ndarray) -> List[List[List[float]]]:
                return [[[float(v)]] for v in values.tolist()]

            return {
                "min": _wrap(min_arr),
                "max": _wrap(max_arr),
                "mean": _wrap(mean_arr),
                "std": _wrap(std_arr),
                "count": [int(total)],
            }
        except Exception:
            return None

    def _update_config_from_robot_config(self, robot_config: Dict):
        """Update conversion config from robot_config.yaml."""
        if "robot_type" in robot_config:
            self.config.robot_type = robot_config["robot_type"]

        if "state_topics" in robot_config:
            topics = robot_config["state_topics"]
            if isinstance(topics, dict):
                self.config.state_topics = list(topics.values())
                # Build topic -> group key mapping
                for key, topic_path in topics.items():
                    self._state_topic_key_map[topic_path] = key
            elif isinstance(topics, list):
                self.config.state_topics = topics

        if "action_topics" in robot_config:
            topics = robot_config["action_topics"]
            if isinstance(topics, dict):
                self.config.action_topics = list(topics.values())
                for key, topic_path in topics.items():
                    self._action_topic_key_map[topic_path] = key
            elif isinstance(topics, list):
                self.config.action_topics = topics

        if "fps" in robot_config:
            self.config.fps = robot_config["fps"]

        if "camera_mapping" in robot_config:
            self._camera_mapping = robot_config["camera_mapping"]
            self._log_info(f"Loaded camera mapping: {self._camera_mapping}")

        # Load joint_order (nested dict) for per-group ordering
        if "joint_order" in robot_config:
            joint_order = robot_config["joint_order"]
            if isinstance(joint_order, dict):
                self._joint_order_by_group = {}
                flattened = []
                for key, joints in joint_order.items():
                    if isinstance(joints, list):
                        self._joint_order_by_group[key] = joints
                        flattened.extend(joints)
                    else:
                        self._joint_order_by_group[key] = [joints]
                        flattened.append(joints)
                self._joint_order = flattened
                self._log_info(
                    f"Loaded joint_order by group: {list(self._joint_order_by_group.keys())} "
                    f"(total {len(self._joint_order)} joints)"
                )
            else:
                self._joint_order = joint_order
                self._log_info(
                    f"Loaded joint_order with {len(self._joint_order)} joints"
                )

        # Prefer total_joint_order (flat list) if explicitly provided
        if "total_joint_order" in robot_config:
            self._joint_order = robot_config["total_joint_order"]
            self._log_info(
                f"Overriding with total_joint_order: {len(self._joint_order)} joints"
            )

    def _extract_velocity_from_odometry(self, msg) -> Optional[np.ndarray]:
        """Extract velocity values from Odometry message."""
        if hasattr(msg, "twist") and hasattr(msg.twist, "twist"):
            twist = msg.twist.twist
            return np.array([
                twist.linear.x,
                twist.linear.y,
                twist.angular.z,
            ], dtype=np.float32)
        return None

    def _extract_velocity_from_twist(self, msg) -> Optional[np.ndarray]:
        """Extract velocity values from Twist message."""
        if hasattr(msg, "linear") and hasattr(msg, "angular"):
            return np.array([
                msg.linear.x,
                msg.linear.y,
                msg.angular.z,
            ], dtype=np.float32)
        return None

    @staticmethod
    def _cdr_align(offset: int, alignment: int, origin: int = 4) -> int:
        """Align a CDR payload offset relative to the post-encapsulation origin."""
        relative = offset - origin
        return origin + ((relative + alignment - 1) & ~(alignment - 1))

    @classmethod
    def _cdr_read_u32(cls, data: bytes, offset: int) -> Tuple[int, int]:
        offset = cls._cdr_align(offset, 4)
        return struct.unpack_from("<I", data, offset)[0], offset + 4

    @classmethod
    def _cdr_read_i32(cls, data: bytes, offset: int) -> Tuple[int, int]:
        offset = cls._cdr_align(offset, 4)
        return struct.unpack_from("<i", data, offset)[0], offset + 4

    @classmethod
    def _cdr_read_string(cls, data: bytes, offset: int) -> Tuple[str, int]:
        length, offset = cls._cdr_read_u32(data, offset)
        if length <= 0:
            return "", offset
        raw = data[offset:offset + length - 1]
        return raw.decode("utf-8", "replace"), offset + length

    @classmethod
    def _cdr_read_string_sequence(
        cls, data: bytes, offset: int
    ) -> Tuple[List[str], int]:
        count, offset = cls._cdr_read_u32(data, offset)
        values: List[str] = []
        for _ in range(count):
            value, offset = cls._cdr_read_string(data, offset)
            values.append(value)
        return values, offset

    @classmethod
    def _cdr_read_float64_sequence(
        cls, data: bytes, offset: int
    ) -> Tuple[np.ndarray, int]:
        count, offset = cls._cdr_read_u32(data, offset)
        offset = cls._cdr_align(offset, 8)
        if count <= 0:
            return np.asarray([], dtype=np.float32), offset
        byte_count = count * 8
        values = np.frombuffer(
            data,
            dtype="<f8",
            count=count,
            offset=offset,
        ).astype(np.float32)
        return values, offset + byte_count

    @classmethod
    def _cdr_read_float64_sequence_at(
        cls,
        data: bytes,
        offset: int,
        expected_count: int,
    ) -> np.ndarray:
        count, offset = cls._cdr_read_u32(data, offset)
        if count != expected_count:
            raise ValueError(
                f"CDR float64 sequence count changed: "
                f"expected {expected_count}, got {count}"
            )
        offset = cls._cdr_align(offset, 8)
        return np.frombuffer(
            data,
            dtype="<f8",
            count=count,
            offset=offset,
        ).astype(np.float32)

    @classmethod
    def _cdr_skip_header(cls, data: bytes, offset: int = 4) -> int:
        _, offset = cls._cdr_read_i32(data, offset)
        _, offset = cls._cdr_read_u32(data, offset)
        _, offset = cls._cdr_read_string(data, offset)
        return offset

    @staticmethod
    def _raw_cdr_topic_supported(topic_type: str) -> bool:
        return (
            "sensor_msgs/msg/JointState" in topic_type
            or "trajectory_msgs/msg/JointTrajectory" in topic_type
            or "nav_msgs/msg/Odometry" in topic_type
            or "geometry_msgs/msg/Twist" in topic_type
        )

    @classmethod
    def _extract_raw_cdr_positions(
        cls, topic_type: str, data: bytes
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        positions, joint_names, _, _ = cls._extract_raw_cdr_positions_with_offset(
            topic_type, data
        )
        return positions, joint_names

    @classmethod
    def _extract_raw_cdr_positions_with_offset(
        cls, topic_type: str, data: bytes
    ) -> Tuple[Optional[np.ndarray], List[str], Optional[int], Optional[int]]:
        """Extract only the converter-needed fields from common ROS2 CDR blobs."""
        if len(data) < 4:
            raise ValueError("CDR payload is too short")
        if data[:2] != b"\x00\x01":
            raise ValueError("unsupported CDR encapsulation")

        if "sensor_msgs/msg/JointState" in topic_type:
            offset = cls._cdr_skip_header(data)
            joint_names, offset = cls._cdr_read_string_sequence(data, offset)
            position_offset = offset
            positions, _ = cls._cdr_read_float64_sequence(data, offset)
            return positions, joint_names, position_offset, int(len(positions))

        if "trajectory_msgs/msg/JointTrajectory" in topic_type:
            offset = cls._cdr_skip_header(data)
            joint_names, offset = cls._cdr_read_string_sequence(data, offset)
            point_count, offset = cls._cdr_read_u32(data, offset)
            if point_count <= 0:
                return None, joint_names, None, None
            position_offset = offset
            positions, _ = cls._cdr_read_float64_sequence(data, offset)
            return positions, joint_names, position_offset, int(len(positions))

        if "nav_msgs/msg/Odometry" in topic_type:
            offset = cls._cdr_skip_header(data)
            _, offset = cls._cdr_read_string(data, offset)  # child_frame_id
            offset = cls._cdr_align(offset, 8)
            # PoseWithCovariance: pose(7 doubles) + covariance(36 doubles).
            offset += 43 * 8
            linear_x, linear_y, _linear_z, _angular_x, _angular_y, angular_z = (
                struct.unpack_from("<6d", data, offset)
            )
            return np.asarray(
                [linear_x, linear_y, angular_z], dtype=np.float32
            ), [], None, None

        if "geometry_msgs/msg/Twist" in topic_type:
            offset = cls._cdr_align(4, 8)
            linear_x, linear_y, _linear_z, _angular_x, _angular_y, angular_z = (
                struct.unpack_from("<6d", data, offset)
            )
            return np.asarray(
                [linear_x, linear_y, angular_z], dtype=np.float32
            ), [], None, None

        return None, [], None, None

    def _get_topic_group_key(self, topic: str, role: str) -> str:
        """Get the group key for a topic, using config mapping or deriving from path."""
        if role == "state" and topic in self._state_topic_key_map:
            return self._state_topic_key_map[topic]
        if role == "action" and topic in self._action_topic_key_map:
            return self._action_topic_key_map[topic]
        # Derive from topic path
        parts = topic.strip("/").split("/")
        for part in parts:
            if "follower" in part or "leader" in part:
                role_word = "follower" if "follower" in part else "leader"
                body_part = part.replace(f"_{role_word}", "").replace(f"{role_word}_", "")
                return f"{role_word}_{body_part}"
        if "odom" in topic.lower():
            return "follower_mobile"
        if "cmd_vel" in topic.lower():
            return "leader_mobile"
        return topic

    def _collect_joint_data_raw_cdr(
        self,
        reader: BagReader,
        topic_types: Dict[str, str],
        topics_to_read: Optional[List[str]],
        trim_start: float,
        trim_end: float,
        exclude_regions: List[Dict],
    ) -> Optional[
        Tuple[
            Dict[str, List[Tuple[float, np.ndarray]]],
            Dict[str, List[str]],
            Dict[str, List[Tuple[float, np.ndarray]]],
            Dict[str, List[str]],
        ]
    ]:
        """Fast state/action extraction from raw CDR for common robot messages."""
        if os.environ.get(_RAW_CDR_EXTRACT_DISABLE_ENV):
            return None

        state_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        state_joint_names_by_topic: Dict[str, List[str]] = {}
        action_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        action_joint_names_by_topic: Dict[str, List[str]] = {}
        topic_plan: Dict[str, Tuple[str, str, List[str]]] = {}

        candidate_topics = topics_to_read or list(topic_types.keys())
        for topic in candidate_topics:
            topic_type = topic_types.get(topic, "")
            role = ""
            if self._is_state_topic(topic, topic_types):
                role = "state"
            elif self._is_action_topic(topic, topic_types):
                role = "action"
            if not role:
                continue
            if not self._raw_cdr_topic_supported(topic_type):
                self._log_info(
                    f"Raw CDR extraction does not support {topic_type}; "
                    "falling back to ROS decoder"
                )
                return None

            fallback_names: List[str] = []
            if "Odometry" in topic_type or "Twist" in topic_type:
                group_key = self._get_topic_group_key(topic, role)
                fallback_names = list(
                    self._joint_order_by_group.get(
                        group_key,
                        ["linear_x", "linear_y", "angular_z"],
                    )
                )

            topic_plan[topic] = (role, topic_type, fallback_names)

        if not topic_plan:
            return None

        if any(role == "state" for role, _, _ in topic_plan.values()):
            self._log_info(
                "Raw CDR extraction skipped because observation.state uses "
                "header.stamp timestamps; using ROS decoder instead"
            )
            return None

        position_layout_cache: Dict[str, Tuple[int, int, List[str]]] = {}
        exclude_windows = [
            (
                float(region.get("start", {}).get("time", 0)),
                float(region.get("end", {}).get("time", 0)),
            )
            for region in exclude_regions
        ]

        try:
            raw_iter = reader.read_raw_messages(topic_filter=topics_to_read)
            for topic, raw_data, timestamp, topic_type in raw_iter:
                if timestamp < trim_start or timestamp > trim_end:
                    continue
                if exclude_windows and any(
                    start <= timestamp <= end
                    for start, end in exclude_windows
                ):
                    continue

                plan = topic_plan.get(topic)
                if plan is None:
                    continue
                role, topic_type, fallback_names = plan

                layout = position_layout_cache.get(topic)
                if layout is not None:
                    position_offset, position_count, cached_names = layout
                    try:
                        positions = self._cdr_read_float64_sequence_at(
                            raw_data,
                            position_offset,
                            position_count,
                        )
                        joint_names = cached_names
                    except Exception:
                        positions, joint_names, position_offset, position_count = (
                            self._extract_raw_cdr_positions_with_offset(
                                topic_type, raw_data
                            )
                        )
                        if (
                            position_offset is not None
                            and position_count is not None
                        ):
                            position_layout_cache[topic] = (
                                position_offset,
                                position_count,
                                list(joint_names),
                            )
                else:
                    positions, joint_names, position_offset, position_count = (
                        self._extract_raw_cdr_positions_with_offset(
                            topic_type, raw_data
                        )
                    )
                    if position_offset is not None and position_count is not None:
                        position_layout_cache[topic] = (
                            position_offset,
                            position_count,
                            list(joint_names),
                        )
                if positions is None:
                    continue

                if role == "state":
                    need_joint_names = topic not in state_joint_names_by_topic
                    state_messages_by_topic.setdefault(topic, []).append(
                        (timestamp, positions)
                    )
                    if need_joint_names and joint_names:
                        state_joint_names_by_topic[topic] = list(joint_names)
                    elif need_joint_names and fallback_names:
                        state_joint_names_by_topic[topic] = fallback_names
                else:
                    need_joint_names = topic not in action_joint_names_by_topic
                    action_messages_by_topic.setdefault(topic, []).append(
                        (timestamp, positions)
                    )
                    if need_joint_names and joint_names:
                        action_joint_names_by_topic[topic] = list(joint_names)
                    elif need_joint_names and fallback_names:
                        action_joint_names_by_topic[topic] = fallback_names
        except Exception as exc:  # noqa: BLE001 - safe fallback to decoder
            self._log_info(
                f"Raw CDR extraction failed ({exc!r}); falling back to ROS decoder"
            )
            return None

        if not state_messages_by_topic:
            return None

        self._log_info(
            "Used raw CDR extraction for "
            f"{len(state_messages_by_topic)} state topic(s), "
            f"{len(action_messages_by_topic)} action topic(s)"
        )
        return (
            state_messages_by_topic,
            state_joint_names_by_topic,
            action_messages_by_topic,
            action_joint_names_by_topic,
        )

    @staticmethod
    def _message_header_timestamp_sec(
        msg: Any,
        fallback_timestamp_sec: float,
    ) -> Tuple[float, bool]:
        """Return ``header.stamp`` seconds when present, else MCAP log time."""
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return fallback_timestamp_sec, False

        sec = int(getattr(stamp, "sec", 0) or 0)
        nanosec = int(getattr(stamp, "nanosec", 0) or 0)
        if sec == 0 and nanosec == 0:
            return fallback_timestamp_sec, False
        return sec + nanosec / 1e9, True

    def _timestamp_is_selected(
        self,
        timestamp_sec: float,
        trim_start: float,
        trim_end: float,
        exclude_regions: List[Dict],
    ) -> bool:
        if timestamp_sec < trim_start or timestamp_sec > trim_end:
            return False
        return not self._is_in_exclude_region(timestamp_sec, exclude_regions)

    def _extract_joint_data(
        self,
        bag_path: Path,
        episode_index: int,
        trim_points: Optional[Dict],
        exclude_regions: List[Dict],
    ) -> Optional[EpisodeData]:
        """Extract joint state and action data from ROSbag."""
        # Clear the per-instance bisect-keys cache. Each episode builds a
        # fresh state_messages / action_messages list, but Python may reuse
        # the previous list's memory address for the new one — without this
        # reset, ``_find_previous_value(_in_list)`` would silently return
        # keys belonging to the prior episode and corrupt the resample.
        self._bisect_keys_cache.clear()

        cache_path = self._episode_extract_cache_path(
            bag_path, trim_points, exclude_regions
        )
        if cache_path is not None:
            cached_episode = self._load_episode_extract_cache(
                cache_path,
                episode_index=episode_index,
            )
            if cached_episode is not None:
                self._log_info(
                    f"{Path(bag_path).name}: reused cached joint extraction "
                    f"({cached_episode.length} frames)"
                )
                return cached_episode

        reader = BagReader(bag_path, self.logger)
        if not reader.open():
            self._log_error(f"Failed to open rosbag: {bag_path}")
            return None

        episode = EpisodeData(episode_index=episode_index)

        # Determine time bounds from trim points
        trim_start = (
            trim_points.get("start", {}).get("time", 0.0) if trim_points else 0.0
        )
        trim_end = (
            trim_points.get("end", {}).get("time", float("inf"))
            if trim_points
            else float("inf")
        )

        # Group both state and action messages by topic
        state_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        state_joint_names_by_topic: Dict[str, List[str]] = {}
        action_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        action_joint_names_by_topic: Dict[str, List[str]] = {}

        topic_types = reader.get_topic_types()

        # Build topic filter to avoid decoding unnecessary messages (TF, CameraInfo, etc.)
        topics_to_read = None
        if self.config.state_topics or self.config.action_topics:
            topics_to_read = list(
                set(self.config.state_topics + self.config.action_topics)
            )
            self._log_info(f"Reading {len(topics_to_read)} topics (filtered)")

        raw_collected = self._collect_joint_data_raw_cdr(
            reader,
            topic_types,
            topics_to_read,
            trim_start,
            trim_end,
            exclude_regions,
        )
        if raw_collected is not None:
            (
                state_messages_by_topic,
                state_joint_names_by_topic,
                action_messages_by_topic,
                action_joint_names_by_topic,
            ) = raw_collected
        else:
            state_log_time_fallback_topics: set[str] = set()
            for topic, msg, timestamp in reader.read_messages(topic_filter=topics_to_read):
                topic_type = topic_types.get(topic, "")

                # Process state topics
                if self._is_state_topic(topic, topic_types):
                    positions = None
                    joint_names = []
                    need_joint_names = topic not in state_joint_names_by_topic

                    if "Odometry" in topic_type:
                        positions = self._extract_velocity_from_odometry(msg)
                        if need_joint_names:
                            group_key = self._get_topic_group_key(topic, "state")
                            if group_key in self._joint_order_by_group:
                                joint_names = self._joint_order_by_group[group_key]
                            else:
                                joint_names = ["linear_x", "linear_y", "angular_z"]
                    elif hasattr(msg, "position") and msg.position:
                        positions = np.asarray(msg.position, dtype=np.float32)
                        if need_joint_names:
                            joint_names = (
                                list(msg.name)
                                if hasattr(msg, "name") and msg.name
                                else []
                            )

                    if positions is not None:
                        sample_timestamp, used_header_stamp = (
                            self._message_header_timestamp_sec(msg, timestamp)
                        )
                        if not used_header_stamp:
                            state_log_time_fallback_topics.add(topic)
                        if not self._timestamp_is_selected(
                            sample_timestamp,
                            trim_start,
                            trim_end,
                            exclude_regions,
                        ):
                            continue
                        state_messages_by_topic.setdefault(topic, []).append(
                            (sample_timestamp, positions)
                        )
                        if topic not in state_joint_names_by_topic and joint_names:
                            state_joint_names_by_topic[topic] = joint_names

                # Process action topics
                elif self._is_action_topic(topic, topic_types):
                    positions = None
                    joint_names = []
                    need_joint_names = topic not in action_joint_names_by_topic

                    if "Twist" in topic_type:
                        positions = self._extract_velocity_from_twist(msg)
                        if need_joint_names:
                            group_key = self._get_topic_group_key(topic, "action")
                            if group_key in self._joint_order_by_group:
                                joint_names = self._joint_order_by_group[group_key]
                            else:
                                joint_names = ["linear_x", "linear_y", "angular_z"]
                    else:
                        positions = self._extract_action_positions(msg)
                        if need_joint_names:
                            joint_names = self._extract_joint_names(msg)

                    if positions is not None:
                        if not self._timestamp_is_selected(
                            timestamp,
                            trim_start,
                            trim_end,
                            exclude_regions,
                        ):
                            continue
                        action_messages_by_topic.setdefault(topic, []).append(
                            (timestamp, positions)
                        )
                        if topic not in action_joint_names_by_topic and joint_names:
                            action_joint_names_by_topic[topic] = joint_names
            if state_log_time_fallback_topics:
                topics = ", ".join(sorted(state_log_time_fallback_topics))
                self._log_warning(
                    "State topic(s) without valid header.stamp used MCAP "
                    f"log_time fallback: {topics}"
                )

        if not state_messages_by_topic:
            self._log_warning(f"No state messages found in {bag_path}")
            return None

        state_messages = self._merge_state_messages(
            state_messages_by_topic, state_joint_names_by_topic
        )
        action_messages = self._merge_action_messages(
            action_messages_by_topic, action_joint_names_by_topic
        )

        if not state_messages:
            self._log_warning(f"No valid merged state messages in {bag_path}")
            return None

        episode, staleness_metrics = self._resample_to_fps(
            episode, state_messages, action_messages, trim_start
        )
        episode.observation_state_names = list(self._state_joint_names or [])
        episode.action_names = list(self._action_joint_names or [])

        self._staleness_reports[episode_index] = staleness_metrics
        self._log_staleness_summary(staleness_metrics)

        if cache_path is not None:
            self._store_episode_extract_cache(
                cache_path,
                episode=episode,
                staleness_metrics=staleness_metrics,
            )

        return episode

    def _episode_extract_cache_path(
        self,
        bag_path: Path,
        trim_points: Optional[Dict],
        exclude_regions: List[Dict],
    ) -> Optional[Path]:
        """Return the persistent cache path for resampled state/action data."""
        if os.environ.get(_EXTRACT_CACHE_DISABLE_ENV):
            return None
        try:
            key_payload = self._episode_extract_cache_key(
                Path(bag_path), trim_points, exclude_regions
            )
        except OSError as exc:
            self._log_warning(
                f"{Path(bag_path).name}: extraction cache disabled "
                f"({exc!r})"
            )
            return None
        digest = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        root = Path(bag_path)
        cache_root = root if root.is_dir() else root.parent
        return (
            cache_root
            / ".cyclo_cache"
            / "joint_extract"
            / f"{digest}.pickle"
        )

    def _episode_extract_cache_key(
        self,
        bag_path: Path,
        trim_points: Optional[Dict],
        exclude_regions: List[Dict],
    ) -> Dict[str, Any]:
        source_files = []
        for src in self._episode_extract_source_files(bag_path):
            stat = src.stat()
            source_files.append({
                "path": str(src.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        if not source_files:
            raise OSError(f"no MCAP files found for {bag_path}")
        return {
            "version": _EXTRACT_CACHE_VERSION,
            "sources": source_files,
            "fps": int(self.config.fps),
            "state_topics": list(self.config.state_topics),
            "action_topics": list(self.config.action_topics),
            "joint_order": list(self._joint_order),
            "joint_order_by_group": self._joint_order_by_group,
            "state_topic_key_map": self._state_topic_key_map,
            "action_topic_key_map": self._action_topic_key_map,
            "quality_warning_multiplier": float(
                self.config.quality_warning_multiplier
            ),
            "quality_error_multiplier": float(
                self.config.quality_error_multiplier
            ),
            "trim_points": trim_points or {},
            "exclude_regions": exclude_regions or [],
        }

    @staticmethod
    def _episode_extract_source_files(bag_path: Path) -> List[Path]:
        if bag_path.is_file() and bag_path.suffix == ".mcap":
            return [bag_path]
        if bag_path.is_dir():
            episode_mcap = bag_path / "episode.mcap"
            if episode_mcap.exists():
                return [episode_mcap]
            return sorted(bag_path.glob("*.mcap"))
        return []

    def _load_episode_extract_cache(
        self,
        cache_path: Path,
        *,
        episode_index: int,
    ) -> Optional[EpisodeData]:
        try:
            with open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{cache_path.name}: extraction cache unreadable ({exc!r})"
            )
            return None

        if (
            not isinstance(payload, dict)
            or payload.get("version") != _EXTRACT_CACHE_VERSION
            or not isinstance(payload.get("episode"), EpisodeData)
        ):
            return None

        episode = payload["episode"]
        episode.episode_index = int(episode_index)
        episode.video_files = {}
        episode.source_path = None
        self._state_joint_names = list(payload.get("state_joint_names") or [])
        self._action_joint_names = list(payload.get("action_joint_names") or [])
        episode.observation_state_names = list(self._state_joint_names or [])
        episode.action_names = list(self._action_joint_names or [])
        staleness = payload.get("staleness_metrics")
        if isinstance(staleness, dict):
            self._staleness_reports[int(episode_index)] = staleness
        return episode

    def _store_episode_extract_cache(
        self,
        cache_path: Path,
        *,
        episode: EpisodeData,
        staleness_metrics: Dict[str, StalenessMetrics],
    ) -> None:
        payload = {
            "version": _EXTRACT_CACHE_VERSION,
            "episode": episode,
            "state_joint_names": list(self._state_joint_names or []),
            "action_joint_names": list(self._action_joint_names or []),
            "staleness_metrics": staleness_metrics,
        }
        tmp_path: Optional[Path] = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "wb",
                prefix=cache_path.stem + ".",
                suffix=".tmp",
                dir=str(cache_path.parent),
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception as exc:  # noqa: BLE001
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            self._log_warning(
                f"{Path(cache_path).name}: failed to write extraction cache "
                f"({exc!r})"
            )

    def _is_state_topic(self, topic: str, topic_types: Dict[str, str]) -> bool:
        """Check if topic is a state topic."""
        if self.config.state_topics:
            return topic in self.config.state_topics

        # Default heuristics — check action indicators first to avoid false positives
        topic_type = topic_types.get(topic, "")
        topic_lower = topic.lower()
        is_action_indicator = (
            "leader" in topic_lower
            or "action" in topic_lower
            or "command" in topic_lower
            or "cmd_vel" in topic_lower
        )
        if "JointState" in topic_type or "JointTrajectory" in topic_type:
            if not is_action_indicator and "follower" in topic_lower:
                return True
        if "Odometry" in topic_type:
            if not is_action_indicator:
                return True
        return False

    def _is_action_topic(self, topic: str, topic_types: Dict[str, str]) -> bool:
        """Check if topic is an action topic."""
        if self.config.action_topics:
            return topic in self.config.action_topics

        # Default heuristics
        topic_type = topic_types.get(topic, "")
        topic_lower = topic.lower()
        if "JointTrajectory" in topic_type or "JointState" in topic_type:
            if (
                "leader" in topic_lower
                or "action" in topic_lower
                or "command" in topic_lower
            ):
                return True
        if "Twist" in topic_type:
            if "leader" in topic_lower or "cmd_vel" in topic_lower:
                return True
        return False

    def _extract_action_positions(self, msg) -> Optional[np.ndarray]:
        """Extract position values from action message."""
        # JointTrajectory message
        if hasattr(msg, "points") and msg.points:
            point = msg.points[0]
            if hasattr(point, "positions") and point.positions:
                return np.asarray(point.positions, dtype=np.float32)

        # JointState message
        if hasattr(msg, "position") and msg.position:
            return np.asarray(msg.position, dtype=np.float32)

        return None

    def _extract_joint_names(self, msg) -> List[str]:
        """Extract joint names from message."""
        if hasattr(msg, "joint_names") and msg.joint_names:
            return list(msg.joint_names)
        if hasattr(msg, "name") and msg.name:
            return list(msg.name)
        return []

    def _filter_positions_by_joint_order(
        self,
        positions: np.ndarray,
        msg_names: List[str],
        joint_order: List[str],
    ) -> Optional[np.ndarray]:
        """
        Filter positions array to only include joints in joint_order.

        Args:
            positions: Array of joint positions from message
            msg_names: Joint names from message (same order as positions)
            joint_order: Ordered list of joints to include in output

        Returns:
            Filtered positions array with only joints in joint_order,
            or None if any joint in joint_order is missing from msg_names.
        """
        if len(positions) != len(msg_names):
            self._log_warning(
                f"Position/name length mismatch: {len(positions)} vs {len(msg_names)}"
            )
            return None

        # Build name-to-index mapping
        name_to_idx = {name: idx for idx, name in enumerate(msg_names)}

        # Extract positions in joint_order
        filtered = []
        for joint_name in joint_order:
            if joint_name not in name_to_idx:
                self._log_warning(
                    f"Joint '{joint_name}' from joint_order not found in message"
                )
                return None
            filtered.append(positions[name_to_idx[joint_name]])

        return np.array(filtered, dtype=np.float32)

    def _joint_order_index_array(
        self,
        msg_names: List[str],
        joint_order: List[str],
    ) -> Optional[np.ndarray]:
        """Build a reusable index array for a message's stable joint layout."""
        if len(msg_names) < len(joint_order):
            self._log_warning(
                f"Position/name length mismatch: {len(msg_names)} names for "
                f"{len(joint_order)} requested joints"
            )
            return None

        name_to_idx = {name: idx for idx, name in enumerate(msg_names)}
        indices: List[int] = []
        for joint_name in joint_order:
            idx = name_to_idx.get(joint_name)
            if idx is None:
                self._log_warning(
                    f"Joint '{joint_name}' from joint_order not found in message"
                )
                return None
            indices.append(idx)
        return np.asarray(indices, dtype=np.intp)

    def _is_in_exclude_region(
        self, timestamp: float, exclude_regions: List[Dict]
    ) -> bool:
        """Check if timestamp falls within any exclude region."""
        for region in exclude_regions:
            start = region.get("start", {}).get("time", 0)
            end = region.get("end", {}).get("time", 0)
            if start <= timestamp <= end:
                return True
        return False

    def _merge_action_messages(
        self,
        action_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]],
        action_joint_names_by_topic: Dict[str, List[str]],
    ) -> List[Tuple[float, np.ndarray]]:
        """Merge action messages from multiple topics into a single action vector."""
        if not action_messages_by_topic:
            return []

        # Determine topic ordering using group keys
        topic_to_group: Dict[str, str] = {}
        for topic in action_messages_by_topic.keys():
            group_key = self._get_topic_group_key(topic, "action")
            topic_to_group[topic] = group_key

        # Sort topics by their group key for consistent ordering
        sorted_topics = sorted(
            action_messages_by_topic.keys(),
            key=lambda t: topic_to_group.get(t, t)
        )

        # Build combined joint names, applying per-group joint_order if available
        combined_names = []
        for topic in sorted_topics:
            group_key = topic_to_group[topic]
            if group_key in self._joint_order_by_group:
                combined_names.extend(self._joint_order_by_group[group_key])
            else:
                names = action_joint_names_by_topic.get(topic, [])
                combined_names.extend(names)
        self._action_joint_names = combined_names

        filter_indices_by_topic: Dict[str, Optional[np.ndarray]] = {}
        filter_failed_topics = set()
        for topic in sorted_topics:
            group_key = topic_to_group[topic]
            if group_key not in self._joint_order_by_group:
                filter_indices_by_topic[topic] = None
                continue
            group_names = action_joint_names_by_topic.get(topic, [])
            if not group_names:
                filter_indices_by_topic[topic] = None
                continue
            indices = self._joint_order_index_array(
                group_names, self._joint_order_by_group[group_key]
            )
            if indices is None:
                filter_failed_topics.add(topic)
            filter_indices_by_topic[topic] = indices

        # Use timestamps from the first topic as reference
        reference_topic = sorted_topics[0]
        reference_timestamps = [t for t, _ in action_messages_by_topic[reference_topic]]
        message_times_by_topic = {
            topic: [t for t, _ in action_messages_by_topic[topic]]
            for topic in sorted_topics
        }
        cursor_by_topic = {topic: -1 for topic in sorted_topics}

        # For each reference timestamp, concatenate actions from all topics
        # Only include timestamps where ALL topics have valid previous values
        merged_messages: List[Tuple[float, np.ndarray]] = []

        for timestamp in reference_timestamps:
            combined_parts: List[np.ndarray] = []
            all_topics_have_data = True

            for topic in sorted_topics:
                if topic in filter_failed_topics:
                    all_topics_have_data = False
                    break
                msgs = action_messages_by_topic[topic]
                times = message_times_by_topic[topic]
                idx = cursor_by_topic[topic]
                while idx + 1 < len(times) and times[idx + 1] <= timestamp:
                    idx += 1
                cursor_by_topic[topic] = idx
                if idx < 0:
                    all_topics_have_data = False
                    break
                if (timestamp - times[idx]) > 0.05:
                    all_topics_have_data = False
                    break
                prev_value = msgs[idx][1]
                indices = filter_indices_by_topic.get(topic)
                if indices is not None:
                    prev_value = prev_value[indices]
                combined_parts.append(prev_value)

            if all_topics_have_data and combined_parts:
                merged_messages.append(
                    (
                        timestamp,
                        np.concatenate(combined_parts).astype(
                            np.float32, copy=False
                        ),
                    )
                )

        return merged_messages

    def _merge_state_messages(
        self,
        state_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]],
        state_joint_names_by_topic: Dict[str, List[str]],
    ) -> List[Tuple[float, np.ndarray]]:
        """Merge state messages from multiple topics into a single state vector.

        Uses joint_order_by_group to filter/reorder each topic's joints,
        then concatenates them in sorted group key order.
        """
        if not state_messages_by_topic:
            return []

        # If only one topic and no grouping needed, use simple path
        if len(state_messages_by_topic) == 1 and not self._joint_order_by_group:
            topic = list(state_messages_by_topic.keys())[0]
            names = state_joint_names_by_topic.get(topic, [])
            if names:
                self._state_joint_names = names
            return state_messages_by_topic[topic]

        # Determine topic ordering using group keys
        topic_to_group: Dict[str, str] = {}
        for topic in state_messages_by_topic.keys():
            group_key = self._get_topic_group_key(topic, "state")
            topic_to_group[topic] = group_key

        # Sort topics by canonical action-side ordering (leader_* keys in
        # joint_order, yaml insertion order). follower_<X> sorts to the
        # position of leader_<X>, follower_upper_body sorts to the first
        # non-mobile leader. Result: state and action emit dimensions in
        # the same per-modality order (mobile last for ffw_sg2, matching
        # the predecessor cyclo_intelligence layout).
        canonical_keys = [
            k for k in self._joint_order_by_group if k.startswith("leader_")
        ]

        def _state_sort_key(topic: str) -> Tuple[int, int, str]:
            gk = topic_to_group[topic]
            if gk.startswith("follower_"):
                modality = gk[len("follower_"):]
                leader_key = f"leader_{modality}"
                if leader_key in canonical_keys:
                    return (0, canonical_keys.index(leader_key), gk)
                if gk == "follower_upper_body":
                    for i, k in enumerate(canonical_keys):
                        if "mobile" not in k.lower():
                            return (0, i, gk)
            if gk in canonical_keys:
                return (0, canonical_keys.index(gk), gk)
            return (1, len(canonical_keys), gk)

        sorted_topics = sorted(
            state_messages_by_topic.keys(),
            key=_state_sort_key,
        )

        # Build combined joint names, applying joint_order if a state-side
        # filter resolves (directly or via follower_<X>→leader_<X>
        # symmetry — see _resolve_filter_target_names).
        combined_names = []
        for topic in sorted_topics:
            group_key = topic_to_group[topic]
            target = self._resolve_filter_target_names(group_key)
            if target:
                combined_names.extend(target)
            else:
                names = state_joint_names_by_topic.get(topic, [])
                combined_names.extend(names)
        self._state_joint_names = combined_names

        filter_indices_by_topic: Dict[str, Optional[np.ndarray]] = {}
        filter_failed_topics = set()
        for topic in sorted_topics:
            group_key = topic_to_group[topic]
            target_names = self._resolve_filter_target_names(group_key)
            if not target_names:
                filter_indices_by_topic[topic] = None
                continue
            group_names = state_joint_names_by_topic.get(topic, [])
            if not group_names:
                filter_indices_by_topic[topic] = None
                continue
            indices = self._joint_order_index_array(group_names, target_names)
            if indices is None:
                filter_failed_topics.add(topic)
            filter_indices_by_topic[topic] = indices

        # Use timestamps from the first topic as reference
        # (all joint topics publish at ~100Hz, so any topic works)
        # This avoids creating the full cross-product of timestamps
        reference_topic = sorted_topics[0]
        reference_timestamps = [t for t, _ in state_messages_by_topic[reference_topic]]
        message_times_by_topic = {
            topic: [t for t, _ in state_messages_by_topic[topic]]
            for topic in sorted_topics
        }
        cursor_by_topic = {topic: -1 for topic in sorted_topics}

        # For each reference timestamp, merge state from all topics using causal sync
        merged_messages: List[Tuple[float, np.ndarray]] = []

        for timestamp in reference_timestamps:
            combined_parts: List[np.ndarray] = []
            all_topics_have_data = True

            for topic in sorted_topics:
                if topic in filter_failed_topics:
                    all_topics_have_data = False
                    break
                msgs = state_messages_by_topic[topic]
                times = message_times_by_topic[topic]
                idx = cursor_by_topic[topic]
                while idx + 1 < len(times) and times[idx + 1] <= timestamp:
                    idx += 1
                cursor_by_topic[topic] = idx
                if idx < 0:
                    all_topics_have_data = False
                    break
                if (timestamp - times[idx]) > 0.05:
                    all_topics_have_data = False
                    break
                prev_value = msgs[idx][1]
                indices = filter_indices_by_topic.get(topic)
                if indices is not None:
                    prev_value = prev_value[indices]
                combined_parts.append(prev_value)

            if all_topics_have_data and combined_parts:
                merged_messages.append(
                    (
                        timestamp,
                        np.concatenate(combined_parts).astype(
                            np.float32, copy=False
                        ),
                    )
                )

        self._log_info(
            f"Merged state from {len(sorted_topics)} topics: "
            f"{len(merged_messages)} merged samples, {len(combined_names)} dimensions"
        )

        return merged_messages

    def _find_previous_value_in_list(
        self,
        messages: List[Tuple[float, np.ndarray]],
        target_time: float,
        tolerance: float = float("inf"),
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Find the most recent message value at or before target time (causal sync).

        Uses binary search (bisect) for O(log n) performance.
        Messages must be sorted by timestamp.

        Returns:
            Tuple of (value, staleness_ms) where staleness_ms is how old the value is.
            Returns (None, 0.0) if no valid previous value exists.

        Implementation note: keys cache lives on the instance (cleared by
        ``_extract_joint_data`` per episode). The previous default-argument
        dict was shared across all calls in the same worker process and
        could in principle return stale keys when a new ``state_messages``
        list reused a GC'd address — instance scope makes that impossible.
        """
        if not messages:
            return None, 0.0

        # Cache timestamp keys for repeated lookups on the same list
        list_id = id(messages)
        if (list_id not in self._bisect_keys_cache
                or len(self._bisect_keys_cache[list_id]) != len(messages)):
            self._bisect_keys_cache[list_id] = [t for t, _ in messages]
        keys = self._bisect_keys_cache[list_id]

        # Binary search: find rightmost index where time <= target_time
        idx = bisect.bisect_right(keys, target_time) - 1
        if idx < 0:
            return None, 0.0

        best_time = keys[idx]
        best_value = messages[idx][1]

        staleness_ms = (target_time - best_time) * 1000.0
        if staleness_ms > tolerance * 1000.0:
            return None, staleness_ms

        return best_value, staleness_ms

    def _resample_to_fps(
        self,
        episode: EpisodeData,
        state_messages: List[Tuple[float, np.ndarray]],
        action_messages: List[Tuple[float, np.ndarray]],
        start_time: float,
    ) -> Tuple[EpisodeData, Dict[str, StalenessMetrics]]:
        """Resample messages to target FPS using causal sync (previous value only)."""
        staleness_metrics: Dict[str, StalenessMetrics] = {
            "observation.state": StalenessMetrics(topic="observation.state"),
            "action": StalenessMetrics(topic="action"),
        }

        if not state_messages:
            return episode, staleness_metrics

        state_times = [t for t, _ in state_messages]
        min_time = min(state_times)
        max_time = max(state_times)

        # Find the first valid start time where both state AND action have data
        # This avoids zero-filled frames at the beginning
        effective_min_time = min_time
        if action_messages:
            action_times = [t for t, _ in action_messages]
            first_action_time = min(action_times)
            # Start from the later of first state or first action
            effective_min_time = max(min_time, first_action_time)
            if effective_min_time > min_time:
                self._log_info(
                    f"Adjusted start time: state_start={min_time:.3f}, "
                    f"action_start={first_action_time:.3f}, "
                    f"effective_start={effective_min_time:.3f}"
                )

        frame_duration = 1.0 / self.config.fps
        num_frames = int((max_time - effective_min_time) * self.config.fps) + 1

        state_staleness_values: List[float] = []
        action_staleness_values: List[float] = []

        action_dim = 0
        if action_messages:
            # Find first valid action message to get dimension
            for _, action_arr in action_messages:
                if len(action_arr) > 0:
                    action_dim = len(action_arr)
                    break

        warning_threshold_ms = (
            1000.0 / self.config.fps
        ) * self.config.quality_warning_multiplier
        error_threshold_ms = (
            1000.0 / self.config.fps
        ) * self.config.quality_error_multiplier

        for frame_idx in range(num_frames):
            target_time = effective_min_time + frame_idx * frame_duration
            # Relative time is from effective start
            relative_time = target_time - effective_min_time

            state, state_staleness_ms = self._find_previous_value(
                state_messages, target_time, frame_duration
            )
            if state is None:
                continue

            staleness_metrics["observation.state"].total_samples += 1
            state_staleness_values.append(state_staleness_ms)
            self._track_staleness(
                staleness_metrics["observation.state"],
                frame_idx,
                state_staleness_ms,
                warning_threshold_ms,
                error_threshold_ms,
            )

            if action_messages and action_dim > 0:
                action, action_staleness_ms = self._find_previous_value(
                    action_messages, target_time, frame_duration
                )
                # Skip this frame if no valid action data
                if action is None:
                    self._log_warning(
                        f"Frame {frame_idx}: No action data at t={target_time:.3f}"
                    )
                    continue

                staleness_metrics["action"].total_samples += 1
                action_staleness_values.append(action_staleness_ms)
                self._track_staleness(
                    staleness_metrics["action"],
                    frame_idx,
                    action_staleness_ms,
                    warning_threshold_ms,
                    error_threshold_ms,
                )
            else:
                action = np.zeros(len(state), dtype=np.float32)

            episode.timestamps.append(relative_time)
            episode.grid_log_times_sec.append(target_time)
            episode.observation_state.append(state)
            episode.action.append(action)

        episode.length = len(episode.timestamps)

        if state_staleness_values:
            staleness_metrics["observation.state"].mean_staleness_ms = float(
                np.mean(state_staleness_values)
            )
            staleness_metrics["observation.state"].max_staleness_ms = float(
                np.max(state_staleness_values)
            )

        if action_staleness_values:
            staleness_metrics["action"].mean_staleness_ms = float(
                np.mean(action_staleness_values)
            )
            staleness_metrics["action"].max_staleness_ms = float(
                np.max(action_staleness_values)
            )

        return episode, staleness_metrics

    def _track_staleness(
        self,
        metrics: StalenessMetrics,
        frame_idx: int,
        staleness_ms: float,
        warning_threshold_ms: float,
        error_threshold_ms: float,
    ):
        if staleness_ms > error_threshold_ms:
            metrics.stale_error_count += 1
            metrics.stale_samples.append(
                {
                    "frame_index": frame_idx,
                    "staleness_ms": round(staleness_ms, 2),
                    "severity": "error",
                }
            )
        elif staleness_ms > warning_threshold_ms:
            metrics.stale_warning_count += 1
            metrics.stale_samples.append(
                {
                    "frame_index": frame_idx,
                    "staleness_ms": round(staleness_ms, 2),
                    "severity": "warning",
                }
            )

    def _find_previous_value(
        self,
        messages: List[Tuple[float, np.ndarray]],
        target_time: float,
        expected_interval_sec: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Find the most recent message value at or before target time (causal sync).

        Uses binary search (bisect) for O(log n) performance.
        Messages must be sorted by timestamp.

        Args:
            messages: List of (timestamp, value) tuples
            target_time: Target time to find previous value for
            expected_interval_sec: Expected interval between messages (for staleness calc)

        Returns:
            Tuple of (value, staleness_ms). Returns (None, 0.0) if no previous value.

        See ``_find_previous_value_in_list`` docstring for the cache rationale
        — same shared-state bug, same fix (per-instance cache cleared per
        ``_extract_joint_data`` call).
        """
        if not messages:
            return None, 0.0

        # Cache timestamp keys for repeated lookups on the same list
        list_id = id(messages)
        if (list_id not in self._bisect_keys_cache
                or len(self._bisect_keys_cache[list_id]) != len(messages)):
            self._bisect_keys_cache[list_id] = [t for t, _ in messages]
        keys = self._bisect_keys_cache[list_id]

        # Binary search: find rightmost index where time <= target_time
        idx = bisect.bisect_right(keys, target_time) - 1
        if idx < 0:
            return None, 0.0

        best_time = keys[idx]
        best_value = messages[idx][1]

        staleness_ms = (target_time - best_time) * 1000.0
        return best_value, staleness_ms

    def _ensure_video_stats_cached(
        self, video_path: Path, camera_name: str
    ) -> None:
        """Precompute + persist video stats to ``video_stats.json``.

        ``_compute_video_stats`` already looks for a ``video_stats.json``
        sidecar next to the video file and skips its cv2 decode loop on
        cache hit. The recording-v2 pipeline doesn't write that file
        anywhere, so Phase 2 (``_compute_episode_stats``) would otherwise
        re-decode 100 frames per camera per episode — that's the serial
        bottleneck that pegged Phase 2 at multi-minute runs.

        By precomputing during ``_sync_videos_to_grid`` (Phase 1) the
        decode work moves into the parallel pool *and* gets cached
        forever for subsequent runs.

        Idempotent: bails immediately if the sidecar already has this
        camera's entry. Best-effort: any failure (cv2 missing,
        unreadable mp4, json write fails) is logged at warning and the
        call is a no-op — Phase 2 falls back to its own decode path.
        """
        video_path = Path(video_path)
        if not video_path.exists() or self._video_stats_sample_budget() <= 0:
            return
        stats_path = video_path.parent / "video_stats.json"
        existing: Dict[str, Any] = {}
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
            except (OSError, ValueError) as exc:
                self._log_warning(
                    f"{camera_name}: video_stats.json unreadable "
                    f"({exc!r}); regenerating"
                )
                loaded = None
            # Defensive: a stale sidecar that's not a dict (e.g. legacy
            # list shape, ``null``, hand-edited) would let the downstream
            # ``existing[camera_name] = stats`` raise TypeError and
            # propagate to ``_sync_videos_to_grid``'s outer except — which
            # then falls back to the raw MP4 and corrupts parquet/mp4
            # length parity. Coerce non-dicts back to {} here.
            if isinstance(loaded, dict):
                existing = loaded
                self._video_stats_sidecar_cache[stats_path] = existing
            else:
                if loaded is not None:
                    self._log_warning(
                        f"{camera_name}: video_stats.json was "
                        f"{type(loaded).__name__}, expected dict; "
                        "discarding and regenerating"
                    )
                existing = {}
            if camera_name in existing:
                return
        # Compute fresh. ``_compute_video_stats`` would itself try the
        # sidecar first; since we just verified the camera isn't in it,
        # the call will fall through to decoding.
        try:
            stats = self._compute_video_stats(video_path, camera_name)
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{camera_name}: video_stats precompute raised "
                f"({exc!r}); Phase 2 will compute lazily"
            )
            return
        if not stats:
            return
        existing[camera_name] = stats
        try:
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(existing, fh)
            self._video_stats_sidecar_cache[stats_path] = existing
        except OSError as exc:
            self._log_warning(
                f"{camera_name}: failed to write video_stats.json "
                f"({exc!r}); Phase 2 will recompute"
            )

    def _store_video_stats_cached(
        self, video_path: Path, camera_name: str, stats: Optional[Dict[str, Any]]
    ) -> None:
        """Persist already-computed video stats next to ``video_path``."""
        if not stats:
            return
        stats_path = Path(video_path).parent / "video_stats.json"
        existing: Dict[str, Any] = {}
        if stats_path.exists():
            try:
                loaded = json.loads(stats_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, ValueError) as exc:
                self._log_warning(
                    f"{camera_name}: failed to read existing video_stats.json "
                    f"({exc!r}); overwriting"
                )
        existing[camera_name] = stats
        try:
            stats_path.write_text(
                json.dumps(existing),
                encoding="utf-8",
            )
            self._video_stats_sidecar_cache[stats_path] = existing
        except OSError as exc:
            self._log_warning(
                f"{camera_name}: failed to write streamed video stats "
                f"({exc!r}); Phase 2 will compute lazily"
            )

    def _can_convert_transcode_state(self, bag_path: Path) -> bool:
        """Return True iff the episode is safe to feed to LeRobot conversion.

        For recording format v2 episodes the background transcoder bakes
        the yaml ``rotation_deg`` (and future transcode-time treatments)
        into the H.264 source MP4. Converting before the transcode has
        finished would silently drop those — so we refuse and log a
        clear actionable error per status.

        Pre-v2 episodes (no ``transcoding_status`` field) are accepted
        unconditionally for backward compatibility.
        """
        info = bag_path / "episode_info.json"
        if not info.exists():
            return True  # not a v2 episode; nothing to gate on
        try:
            import json as _json
            with open(info) as f:
                meta = _json.load(f) or {}
        except Exception:
            return True
        # No status field at all → legacy episode, allow.
        if "transcoding_status" not in meta:
            return True

        status = meta.get("transcoding_status")
        if status in ("done", "not_required"):
            return True
        if status in ("pending", "running"):
            self._log_error(
                f"{bag_path.name}: SKIPPED — transcode is still {status!r}. "
                "Wait for the background transcoder to finish (check "
                "episode_info.json) and re-run conversion."
            )
            return False
        if status == "failed":
            cams_failed = meta.get("transcoding_cameras_failed", {})
            self._log_error(
                f"{bag_path.name}: SKIPPED — previous transcode failed for "
                f"cameras {list(cams_failed.keys()) or 'unknown'}. Inspect "
                f"the recording, fix the cause, then re-trigger the "
                f"transcoder (or delete episode_info.json's transcoding_status "
                f"to retry from raw MJPEG)."
            )
            return False
        # Unknown status string — be conservative and refuse.
        self._log_error(
            f"{bag_path.name}: SKIPPED — unrecognised transcoding_status={status!r}"
        )
        return False

    @staticmethod
    def _synced_cache_identity_matches(
        cached_key: Dict[str, Any],
        desired_key: Dict[str, Any],
    ) -> bool:
        """Return True when cache identity fields match, ignoring metadata."""
        if not isinstance(cached_key, dict):
            return False
        return all(
            cached_key.get(key) == value
            for key, value in desired_key.items()
        )

    def _episode_has_sidecars(self, bag_path: Path) -> bool:
        """True if ``videos/<cam>_timestamps.parquet`` exists for any cam."""
        videos = bag_path / "videos"
        if not videos.exists():
            return False
        return any(videos.rglob("*_timestamps.parquet"))

    @staticmethod
    def _video_sync_output_dir(videos_dir: Path) -> Path:
        """Return where synced MP4/cache sidecars should be written."""
        staging_root = os.environ.get(_VIDEO_SYNC_STAGING_DIR_ENV, "").strip()
        if not staging_root:
            return videos_dir
        videos_dir = Path(videos_dir)
        try:
            identity = str(videos_dir.resolve())
        except OSError:
            identity = str(videos_dir.absolute())
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        out_dir = Path(staging_root) / "synced" / digest
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _resolve_video_sync_camera_workers(camera_count: int) -> int:
        """Pick per-episode camera sync parallelism."""
        if camera_count <= 1:
            return 1
        raw = os.environ.get(_VIDEO_SYNC_CAMERA_WORKERS_ENV)
        if raw:
            try:
                return max(1, min(int(raw), camera_count))
            except ValueError:
                pass

        active_workers = 1
        raw_active = os.environ.get(_CONVERSION_ACTIVE_WORKERS_ENV)
        if raw_active:
            try:
                active_workers = max(1, int(raw_active))
            except ValueError:
                active_workers = 1

        total_budget = _DEFAULT_VIDEO_SYNC_TOTAL_WORKERS
        raw_total = os.environ.get(_VIDEO_SYNC_TOTAL_WORKERS_ENV)
        if raw_total:
            try:
                total_budget = max(1, int(raw_total))
            except ValueError:
                total_budget = _DEFAULT_VIDEO_SYNC_TOTAL_WORKERS

        per_episode_budget = max(1, total_budget // active_workers)
        return max(1, min(camera_count, 4, per_episode_budget))

    def _sync_videos_to_grid(
        self, bag_path: Path, episode: EpisodeData,
    ) -> EpisodeData:
        """Re-pack each camera's source MP4 to one frame per grid step.

        Reads ``videos/<cam>_timestamps.parquet`` written by the
        recorder, maps the EpisodeData log-time grid (sub-second
        UNIX seconds) onto frame indices via causal lookup, then
        runs ``video_sync.remux_selected_frames`` to produce a
        ``videos/<cam>_synced.mp4``. The synced MP4 replaces the raw
        recording in ``episode.video_files`` so downstream copy is
        identical to the old rosbag2mp4 flow.
        """
        if not episode.video_files or not episode.grid_log_times_sec:
            return episode
        has_source_sidecars = any(
            (Path(src_path).parent / f"{cam_name}_timestamps.parquet").exists()
            for cam_name, src_path in episode.video_files.items()
        )
        if not self._episode_has_sidecars(bag_path) and not has_source_sidecars:
            return episode
        from cyclo_data.converter.video_sync import remux_selected_frames
        from cyclo_data.reader.frame_timestamps import (
            FrameTimestamps,
            load_frame_timestamps,
        )

        grid_ns = np.asarray(
            [int(t * 1_000_000_000) for t in episode.grid_log_times_sec],
            dtype=np.int64,
        )
        videos_dir = bag_path / "videos"
        if not videos_dir.exists():
            first_video = next(iter(episode.video_files.values()), None)
            videos_dir = Path(first_video).parent if first_video else videos_dir
        synced: Dict[str, Path] = {}
        synced_counts: Dict[str, int] = {}
        # UI-supplied rotation is treated as an *additional* override on
        # top of whatever the recorder/transcoder already baked into the
        # source MP4. The yaml's ``rotation_deg`` is applied at H.264
        # transcode time in ``cyclo_data/recorder/transcoder.py``, so by
        # the time we get here the source is already correctly oriented
        # and UI default 0 means "leave as-is" — exactly what we want.
        ui_rotations = self.config.camera_rotations or {}
        sync_jobs: List[Dict[str, Any]] = []
        for cam_name, src_path in episode.video_files.items():
            src_path = Path(src_path)
            sidecar = videos_dir / f"{cam_name}_timestamps.parquet"
            if not sidecar.exists():
                sidecar = src_path.parent / f"{cam_name}_timestamps.parquet"
            if not sidecar.exists():
                self._log_warning(
                    f"{cam_name}: no sidecar {sidecar.name}; leaving raw MP4"
                )
                synced[cam_name] = src_path
                continue
            try:
                ft = load_frame_timestamps(sidecar, cam_name)
            except Exception as exc:
                self._log_error(
                    f"{cam_name}: failed to read {sidecar.name}: {exc!r}"
                )
                synced[cam_name] = src_path
                continue
            if ft.num_frames == 0:
                self._log_warning(f"{cam_name}: empty sidecar; skipping sync")
                synced[cam_name] = src_path
                continue
            indices = ft.map_to_grid(grid_ns, time_source="header")
            self._record_frame_reuse_report(
                episode=episode,
                camera_name=cam_name,
                indices=indices,
                grid_ns=grid_ns,
                frame_timestamps=ft,
            )
            sync_output_dir = self._video_sync_output_dir(videos_dir)
            out_path = sync_output_dir / f"{cam_name}_synced.mp4"
            rotation_extra = int(ui_rotations.get(cam_name, 0) or 0)
            target_fps = int(self.config.fps)
            image_resize = self.config.image_resize  # (height, width) or None
            resize_key = (
                list(image_resize) if image_resize else None
            )
            src_stat = src_path.stat()
            indices_i64 = np.asarray(indices, dtype=np.int64)
            indices_hash = hashlib.sha256(indices_i64.tobytes()).hexdigest()

            # Cache reuse: when v2.1 and v3.0 run on the same dataset
            # back-to-back, v3.0's Phase 1 would otherwise redo the
            # entire sync remux that v2.1 already produced. The sidecar
            # records the params that *would* have produced this MP4 —
            # mismatch on fps / rotation / resize / frame_count
            # invalidates the cache so changing knobs via the UI always
            # regenerates.
            cache_sidecar = sync_output_dir / f"{cam_name}_synced.cache.json"
            desired_key = {
                "target_fps": target_fps,
                "rotation_deg": rotation_extra,
                "image_resize": resize_key,
                "frame_count": int(indices.size),
                "frame_indices_sha256": indices_hash,
                "source_size": int(src_stat.st_size),
                "source_mtime_ns": int(src_stat.st_mtime_ns),
            }
            if (
                out_path.exists()
                and cache_sidecar.exists()
                and out_path.stat().st_size > 0
            ):
                try:
                    cached_key = json.loads(cache_sidecar.read_text())
                    if self._synced_cache_identity_matches(
                        cached_key, desired_key
                    ):
                        cached_frames = self._get_video_frame_count(out_path)
                        if cached_frames != int(indices.size):
                            self._log_warning(
                                f"{cam_name}: cached synced MP4 frame count "
                                f"mismatch (cache={indices.size}, "
                                f"mp4={cached_frames}); regenerating"
                            )
                        else:
                            self._log_info(
                                f"{cam_name}: reusing cached synced MP4 "
                                f"({indices.size} frames, fps={target_fps}, "
                                f"rot={rotation_extra}°)"
                            )
                            # Even on cache hit, top up the video_stats.json
                            # sidecar so Phase 2 stays a cache hit. If a
                            # previous run wrote synced.mp4 + cache.json but
                            # not video_stats.json (e.g. ran before this
                            # precompute was added), this fills the gap.
                            self._ensure_video_stats_cached(out_path, cam_name)
                            synced[cam_name] = out_path
                            synced_counts[cam_name] = int(cached_frames)
                            continue
                except (OSError, ValueError) as exc:
                    self._log_warning(
                        f"{cam_name}: cache sidecar unreadable "
                        f"({exc!r}); regenerating"
                    )
                except Exception as exc:
                    self._log_warning(
                        f"{cam_name}: cached synced MP4 validation failed "
                        f"({exc!r}); regenerating"
                    )
            sync_jobs.append({
                "cam_name": cam_name,
                "src_path": src_path,
                "indices": indices,
                "out_path": out_path,
                "target_fps": target_fps,
                "rotation_extra": rotation_extra,
                "image_resize": image_resize,
                "cache_sidecar": cache_sidecar,
                "desired_key": desired_key,
            })

        def _run_sync_job(job: Dict[str, Any]):
            return job, remux_selected_frames(
                job["src_path"],
                job["indices"],
                job["out_path"],
                target_fps=job["target_fps"],
                rotation_deg=job["rotation_extra"],
                image_resize=job["image_resize"],
            )

        def _finish_sync_job(job: Dict[str, Any], sync_result) -> None:
            cam_name = job["cam_name"]
            out_path = job["out_path"]
            indices = job["indices"]
            produced_frames = sync_result.frame_count
            cache_is_valid = produced_frames == int(indices.size)
            # Best-effort sidecar write — failure here just means
            # the next run won't get a cache hit. Don't fail the
            # whole conversion over a metadata write.
            try:
                if cache_is_valid:
                    cache_payload = dict(job["desired_key"])
                    if (
                        sync_result.output_height is not None
                        and sync_result.output_width is not None
                    ):
                        height = int(sync_result.output_height)
                        width = int(sync_result.output_width)
                    else:
                        height, width = self._get_video_dimensions(out_path)
                    cache_payload["output_height"] = int(height)
                    cache_payload["output_width"] = int(width)
                    cache_payload["output_codec"] = "h264"
                    cache_payload["output_pix_fmt"] = "yuv420p"
                    cache_payload["has_audio"] = False
                    job["cache_sidecar"].write_text(json.dumps(cache_payload))
                else:
                    job["cache_sidecar"].unlink(missing_ok=True)
                    self._log_info(
                        f"{cam_name}: not caching synced MP4 because "
                        f"frame count is {produced_frames}, expected "
                        f"{indices.size}"
                    )
            except OSError as exc:
                self._log_warning(
                    f"{cam_name}: failed to write cache sidecar "
                    f"({exc!r}); cache disabled for this run"
                )
            self._log_info(
                f"{cam_name}: synced MP4 {indices.size} frames "
                f"-> {out_path.name} (UI extra rotation={job['rotation_extra']}°, "
                f"mode={sync_result.mode}, "
                f"fallback={sync_result.used_fallback})"
            )
            # Streaming sync can compute stats from the selected
            # frames while they are already in memory. If it fell back
            # to the legacy JPEG path, keep the old decode-once cache
            # behaviour.
            if sync_result.stats:
                self._store_video_stats_cached(
                    out_path, cam_name, sync_result.stats
                )
            else:
                self._ensure_video_stats_cached(out_path, cam_name)
            synced[cam_name] = out_path
            if produced_frames is not None:
                synced_counts[cam_name] = int(produced_frames)

        workers = self._resolve_video_sync_camera_workers(len(sync_jobs))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            self._log_info(
                f"Syncing {len(sync_jobs)} camera videos with {workers} workers"
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_job = {
                    executor.submit(_run_sync_job, job): job
                    for job in sync_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        finished_job, sync_result = future.result()
                        _finish_sync_job(finished_job, sync_result)
                    except Exception as exc:
                        cam_name = job["cam_name"]
                        self._log_error(
                            f"{cam_name}: remux failed ({exc!r}); "
                            "leaving raw MP4"
                        )
                        synced[cam_name] = job["src_path"]
        else:
            for job in sync_jobs:
                try:
                    finished_job, sync_result = _run_sync_job(job)
                    _finish_sync_job(finished_job, sync_result)
                except Exception as exc:
                    cam_name = job["cam_name"]
                    self._log_error(
                        f"{cam_name}: remux failed ({exc!r}); leaving raw MP4"
                    )
                    synced[cam_name] = job["src_path"]

        episode.video_files = synced

        # Safety: when remux falls back to the raw MP4 for one or more
        # cameras (synced.mp4 generation raised or returned the source
        # path), the per-camera frame counts can disagree:
        #   - successful synced.mp4 → indices.size frames (= episode.length)
        #   - fallback raw mp4 → recorder frame count (often < indices.size)
        # Downstream lerobot loaders demand parquet rows == every camera's
        # mp4 frames; an off-by-N mismatch raises ``Invalid frame index``
        # at training time. We align everything to ``min(frame_count)``:
        #   1. truncate the in-memory episode (timestamps / state / action
        #      / grid) so parquet rows == min,
        #   2. trim any mp4 that's still longer (the synced.mp4 outputs
        #      we generated at the original episode.length) down to min
        #      via a fast ``-c copy`` stream copy.
        if episode.length > 0 and synced:
            counts: Dict[str, int] = {}
            for cam_name, video_path in synced.items():
                fc = synced_counts.get(cam_name)
                if fc is None:
                    fc = self._get_video_frame_count(video_path)
                if fc is not None:
                    counts[cam_name] = fc
            if counts:
                min_frames = min(counts.values())
                if min_frames < episode.length or any(
                    c > min_frames for c in counts.values()
                ):
                    mismatched = {
                        cam: c for cam, c in counts.items()
                        if c != episode.length
                    }
                    if mismatched:
                        self._log_warning(
                            f"Episode {episode.episode_index}: "
                            f"camera frame counts disagree with grid "
                            f"length={episode.length}; aligning all "
                            f"to min={min_frames}. Disagreement: "
                            f"{mismatched}"
                        )
                    if min_frames < episode.length:
                        episode.timestamps = episode.timestamps[:min_frames]
                        if episode.observation_state:
                            episode.observation_state = (
                                episode.observation_state[:min_frames]
                            )
                        if episode.action:
                            episode.action = episode.action[:min_frames]
                        if episode.grid_log_times_sec:
                            episode.grid_log_times_sec = (
                                episode.grid_log_times_sec[:min_frames]
                            )
                        episode.length = min_frames
                    # Trim any mp4 longer than min_frames.
                    for cam_name, video_path in synced.items():
                        if counts.get(cam_name, min_frames) <= min_frames:
                            continue
                        self._trim_video_to_n_frames(video_path, min_frames)
                    # If an episode was shortened after sync, any
                    # ``*_synced.cache.json`` keyed to the pre-trim grid
                    # is no longer trustworthy. Drop it so the next
                    # conversion validates/regenerates instead of
                    # silently reusing a stale MP4.
                    for video_path in synced.values():
                        if video_path.stem.endswith("_synced"):
                            video_path.with_name(
                                video_path.stem + ".cache.json"
                            ).unlink(missing_ok=True)

        return episode

    def _trim_video_to_n_frames(self, video_path: Path, n: int) -> None:
        """Truncate ``video_path`` to its first ``n`` frames in place.

        Used as a safety step in ``_sync_videos_to_grid`` to bring
        per-camera mp4 frame counts into agreement after a fallback
        pulls episode.length down to a smaller value than the synced
        mp4 we already generated. ``-c copy`` keeps it a fast remux:
        no decode, no re-encode, output is bit-identical for the kept
        frames. Atomic rename to avoid partial files on failure.
        """
        video_path = Path(video_path)
        if not video_path.exists() or n <= 0:
            return
        # Keep the ``.mp4`` suffix on the tmp path so ffmpeg can
        # auto-detect the container. The previous ``.mp4.trim.tmp``
        # form left ``.tmp`` as the suffix and ffmpeg refused with
        # "use a standard extension or specify the format manually".
        tmp_path = video_path.with_name(video_path.stem + ".trim_tmp.mp4")
        try:
            import subprocess
            from cyclo_data.converter.video_sync import _ffmpeg

            cmd = [
                _ffmpeg(), "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(video_path),
                "-frames:v", str(n),
                "-c", "copy",
                "-movflags", "+faststart",
                "-f", "mp4",
                str(tmp_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                self._log_warning(
                    f"trim {video_path.name} to {n} frames failed: "
                    f"{result.stderr[-300:]}"
                )
                tmp_path.unlink(missing_ok=True)
                return
            tmp_path.replace(video_path)
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"trim {video_path.name} to {n} frames raised: {exc!r}"
            )
            tmp_path.unlink(missing_ok=True)

    def _find_video_files(self, bag_path: Path) -> Dict[str, Path]:
        """Find MP4 video files in the rosbag directory.

        Supports MP4 converter output (cam_*.mp4 in root dir)
        and legacy format (videos/ subdirectory).
        """
        segment_videos = self._prepare_segment_video_files(bag_path)
        if segment_videos:
            self._log_info(
                f"Prepared segment video files: {list(segment_videos.keys())}"
            )
            return segment_videos

        video_files = {}

        search_paths = [bag_path, bag_path / "videos"]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for mp4_file in sorted(search_path.glob("*.mp4")):
                # ``<cam>_synced.mp4`` is a derivative produced by
                # ``_sync_videos_to_grid`` on the previous conversion run;
                # skip it during initial discovery so we always start from
                # the recorder's raw MP4.
                if mp4_file.stem.endswith("_synced"):
                    continue
                camera_name = self._get_camera_name_for_video(mp4_file.stem)
                if camera_name not in video_files:
                    video_files[camera_name] = mp4_file

        if video_files:
            self._log_info(f"Found video files: {list(video_files.keys())}")

        return video_files

    def _prepare_segment_video_files(self, bag_path: Path) -> Dict[str, Path]:
        info = self._metadata_manager.load_episode_info(bag_path)
        video_segments = info.get("video_segments") or []

        ordered = []
        if isinstance(video_segments, list) and video_segments:
            for segment in video_segments:
                if not isinstance(segment, dict):
                    continue
                video_dir = bag_path / str(segment.get("video_dir", ""))
                if not video_dir.exists():
                    raise FileNotFoundError(
                        f"{bag_path.name}: video_segments references missing "
                        f"directory {video_dir.relative_to(bag_path)}"
                    )
                cameras = [
                    str(cam)
                    for cam in (segment.get("cameras") or [])
                    if str(cam)
                ]
                if not cameras:
                    cameras = [
                        path.stem
                        for path in sorted(video_dir.glob("*.mp4"))
                        if not path.stem.endswith("_synced")
                    ]
                ordered.append((video_dir, set(cameras)))
        else:
            videos_root = bag_path / "videos"
            if videos_root.is_dir():
                for mcap_path in sorted(bag_path.glob("*.mcap")):
                    video_dir = videos_root / mcap_path.stem
                    if not video_dir.exists():
                        segment_dirs = [
                            path for path in videos_root.iterdir()
                            if path.is_dir()
                        ]
                        if segment_dirs:
                            raise FileNotFoundError(
                                f"{bag_path.name}: expected video segment "
                                f"directory {video_dir.relative_to(bag_path)} "
                                f"for {mcap_path.name}"
                            )
                        continue
                    cameras = [
                        path.stem
                        for path in sorted(video_dir.glob("*.mp4"))
                        if not path.stem.endswith("_synced")
                    ]
                    if cameras:
                        ordered.append((video_dir, set(cameras)))

        if not ordered:
            return {}
        common_cameras = set.intersection(*(cameras for _, cameras in ordered))
        if not common_cameras:
            self._log_warning(
                f"{bag_path.name}: no camera exists in every video segment"
            )
            return {}

        from cyclo_data.converter.video_sync import _ffmpeg, _h264_encoder

        ffmpeg_bin = _ffmpeg()
        out_root = (
            Path(self.config.output_dir)
            / "_subtask_video_concat"
            / f"{bag_path.parent.name}_{bag_path.name}"
        )
        out_root.mkdir(parents=True, exist_ok=True)

        prepared: Dict[str, Path] = {}
        for camera in sorted(common_cameras):
            srcs = [video_dir / f"{camera}.mp4" for video_dir, _ in ordered]
            if not all(src.exists() and src.stat().st_size > 0 for src in srcs):
                continue

            out_path = out_root / f"{camera}.mp4"
            sidecar_out = out_root / f"{camera}_timestamps.parquet"
            source_mtime = max(src.stat().st_mtime for src in srcs)
            sidecars = [
                video_dir / f"{camera}_timestamps.parquet"
                for video_dir, _ in ordered
            ]
            sidecar_mtime = max(
                [p.stat().st_mtime for p in sidecars if p.exists()] or [0]
            )
            newest_input = max(source_mtime, sidecar_mtime)
            if (
                out_path.exists()
                and out_path.stat().st_size > 0
                and out_path.stat().st_mtime >= newest_input
                and (
                    sidecar_out.exists()
                    or not all(p.exists() for p in sidecars)
                )
            ):
                if self._video_decodes_successfully(out_path):
                    prepared[camera] = out_path
                    continue
                self._log_warning(
                    f"{bag_path.name}: cached segment video is invalid; "
                    f"rebuilding {out_path}"
                )
                out_path.unlink(missing_ok=True)
                sidecar_out.unlink(missing_ok=True)

            list_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", suffix=".ffconcat", delete=False
                ) as list_file:
                    list_path = Path(list_file.name)
                    for src in srcs:
                        escaped = str(src.resolve()).replace("'", "'\\''")
                        list_file.write(f"file '{escaped}'\n")
                if self._try_prepare_segment_video_copy(
                    ffmpeg_bin, list_path, srcs, out_path
                ):
                    if all(path.exists() for path in sidecars):
                        tables = self._concat_segment_sidecars(sidecars)
                        pq.write_table(pa.concat_tables(tables), sidecar_out)
                    prepared[camera] = out_path
                    continue

                encoder_height, encoder_width = self._get_video_dimensions(
                    srcs[0]
                )
                encoder, encoder_opts = _h264_encoder(
                    ffmpeg_bin,
                    width=encoder_width,
                    height=encoder_height,
                )
                cmd = [
                    ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(list_path),
                    "-an",
                    "-c:v", encoder,
                    *encoder_opts,
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-fps_mode", "passthrough",
                    str(out_path),
                ]
                subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, check=True,
                )
                if not self._video_decodes_successfully(out_path):
                    out_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"prepared segment video failed decode validation: {out_path}"
                    )

                if all(path.exists() for path in sidecars):
                    tables = self._concat_segment_sidecars(sidecars)
                    pq.write_table(pa.concat_tables(tables), sidecar_out)
                prepared[camera] = out_path
            except Exception as exc:  # noqa: BLE001
                self._log_warning(
                    f"{bag_path.name}: failed to prepare segment video "
                    f"{camera}: {exc!r}"
                )
                out_path.unlink(missing_ok=True)
                sidecar_out.unlink(missing_ok=True)
            finally:
                if list_path is not None:
                    list_path.unlink(missing_ok=True)

        return prepared

    def _try_prepare_segment_video_copy(
        self,
        ffmpeg_bin: str,
        list_path: Path,
        srcs: List[Path],
        out_path: Path,
    ) -> bool:
        """Try validated stream-copy concat for subtask video segments."""
        try:
            from cyclo_data.converter.video_sync import (
                _VIDEO_SYNC_DISABLE_COPY_FASTPATH_ENV,
            )

            if os.environ.get(_VIDEO_SYNC_DISABLE_COPY_FASTPATH_ENV):
                return False
            compatibility = self._segment_copy_compatibility(srcs)
            if compatibility is None:
                return False
            expected_frames, pixel_frames = compatibility
            threshold = _DEFAULT_SEGMENT_COPY_MIN_PIXEL_FRAMES
            env_threshold = os.environ.get(_SEGMENT_COPY_MIN_PIXEL_FRAMES_ENV)
            if env_threshold:
                try:
                    threshold = max(0, int(env_threshold))
                except ValueError:
                    pass
            if pixel_frames < threshold:
                return False

            cmd = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-map", "0:v:0",
                "-an",
                "-c:v", "copy",
                "-movflags", "+faststart",
                str(out_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                out_path.unlink(missing_ok=True)
                self._log_warning(
                    f"segment stream-copy concat failed for {out_path.name}: "
                    f"{result.stderr[-300:]}"
                )
                return False

            produced_frames = self._get_video_frame_count(out_path)
            if produced_frames != expected_frames:
                out_path.unlink(missing_ok=True)
                self._log_warning(
                    f"segment stream-copy frame count mismatch for "
                    f"{out_path.name}: expected {expected_frames}, "
                    f"got {produced_frames}"
                )
                return False
            if not self._video_decodes_successfully(out_path):
                out_path.unlink(missing_ok=True)
                self._log_warning(
                    f"segment stream-copy decode validation failed: "
                    f"{out_path.name}"
                )
                return False
            self._log_info(
                f"Prepared segment video with stream copy: {out_path.name} "
                f"({expected_frames} frames)"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            out_path.unlink(missing_ok=True)
            self._log_warning(
                f"segment stream-copy concat raised for {out_path.name}: "
                f"{exc!r}; re-encoding"
            )
            return False

    def _segment_video_frame_total(self, srcs: List[Path]) -> Optional[int]:
        compatibility = self._segment_copy_compatibility(srcs)
        return compatibility[0] if compatibility is not None else None

    def _segment_videos_support_copy_concat(self, srcs: List[Path]) -> bool:
        """Return True when H.264 segment streams are safe to concat-copy."""
        return self._segment_copy_compatibility(srcs) is not None

    def _segment_copy_pixel_frame_estimate(self, srcs: List[Path]) -> Optional[int]:
        """Return sum(width * height * frames) for segment-copy gating."""
        compatibility = self._segment_copy_compatibility(srcs)
        return compatibility[1] if compatibility is not None else None

    def _segment_copy_compatibility(
        self, srcs: List[Path],
    ) -> Optional[Tuple[int, int]]:
        """Return (frames, pixel_frames) when segment streams can copy-concat."""
        if not srcs:
            return None
        reference: Optional[Tuple[Any, ...]] = None
        total_frames = 0
        total_pixel_frames = 0
        for src in srcs:
            frame_count = self._get_video_frame_count(src)
            if frame_count is None or frame_count <= 0:
                return None
            info = self._probe_video_streams(src)
            if not info:
                return None
            streams = info.get("streams") or []
            if any(stream.get("codec_type") == "audio" for stream in streams):
                return None
            video_stream = next(
                (s for s in streams if s.get("codec_type") == "video"), None
            )
            if not video_stream or video_stream.get("codec_name") != "h264":
                return None
            try:
                if int(video_stream.get("has_b_frames", 0) or 0) > 0:
                    return None
            except (TypeError, ValueError):
                return None
            signature = self._video_stream_copy_signature(video_stream)
            if reference is None:
                reference = signature
            elif signature != reference:
                return None
            width = int(video_stream.get("width") or 0)
            height = int(video_stream.get("height") or 0)
            if width <= 0 or height <= 0:
                return None
            total_frames += int(frame_count)
            total_pixel_frames += width * height * int(frame_count)
        return total_frames, total_pixel_frames

    def _probe_video_streams(self, video_path: Path) -> Optional[Dict[str, Any]]:
        cache_key: Optional[Tuple[str, int, int]]
        try:
            cache_key = self._file_probe_cache_key(video_path)
        except OSError:
            cache_key = None
        if cache_key is not None and cache_key in self._video_streams_probe_cache:
            return self._video_streams_probe_cache[cache_key]

        result_value: Optional[Dict[str, Any]] = None
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-print_format",
                    "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                result_value = None
            else:
                result_value = json.loads(result.stdout or "{}")
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"Failed to probe streams for {video_path}: {exc}")
            result_value = None
        if cache_key is not None and result_value is not None:
            self._video_streams_probe_cache[cache_key] = result_value
        return result_value

    @staticmethod
    def _video_stream_copy_signature(stream: Dict[str, Any]) -> Tuple[Any, ...]:
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        return (
            stream.get("codec_name"),
            stream.get("profile"),
            stream.get("level"),
            int(stream.get("width") or 0),
            int(stream.get("height") or 0),
            stream.get("pix_fmt"),
            stream.get("sample_aspect_ratio"),
            stream.get("field_order"),
            stream.get("color_range"),
            stream.get("color_space"),
            stream.get("color_transfer"),
            stream.get("color_primaries"),
            stream.get("chroma_location"),
            stream.get("time_base"),
            rate,
        )

    @staticmethod
    def _video_decodes_successfully(video_path: Path) -> bool:
        """Return True when ffmpeg can decode at least one video frame."""
        if not video_path.exists() or video_path.stat().st_size <= 0:
            return False
        try:
            from cyclo_data.converter.video_sync import (
                _ffmpeg,
                _video_decodes_successfully,
            )

            return _video_decodes_successfully(video_path, _ffmpeg())
        except Exception:
            return False

    def _concat_segment_sidecars(self, sidecars: List[Path]) -> List[pa.Table]:
        """Read segment timestamp sidecars and make frame_index continuous."""
        tables: List[pa.Table] = []
        frame_offset = 0
        for sidecar in sidecars:
            table = pq.read_table(sidecar)
            if "frame_index" in table.column_names:
                col_idx = table.column_names.index("frame_index")
                field = table.schema.field(col_idx)
                frame_index = pa.array(
                    range(frame_offset, frame_offset + table.num_rows),
                    type=field.type,
                )
                table = table.set_column(col_idx, field, frame_index)
            frame_offset += table.num_rows
            tables.append(table)
        return tables

    def _get_camera_name_for_video(self, filename: str) -> str:
        """Get camera name from video filename.

        MP4 converter outputs files like 'cam_left_head.mp4',
        so the stem is already the camera name.
        """
        name = filename.replace("_compressed", "")

        # Direct match: MP4 converter uses cam_name as filename
        if self._camera_mapping:
            # Check if filename matches any known camera name
            for topic, camera_name in self._camera_mapping.items():
                if name == camera_name:
                    return camera_name
                # Legacy: sanitized topic match
                sanitized_topic = topic.replace("/", "_").lstrip("_")
                if sanitized_topic in name or name in sanitized_topic:
                    return camera_name

        # Filename is already the camera name (e.g., cam_left_head)
        if name.startswith("cam_"):
            return name

        return name

    def _get_video_dimensions(self, video_path: Path) -> Tuple[int, int]:
        """Get video height and width using OpenCV."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if width > 0 and height > 0:
                    return height, width
        except Exception as e:
            self._log_warning(f"Failed to get video dimensions: {e}")
        return 480, 640

    def _get_video_frame_count(self, video_path: Path) -> Optional[int]:
        """Get the number of frames in a video file using OpenCV."""
        cache_key: Optional[Tuple[str, int, int]]
        try:
            cache_key = self._file_probe_cache_key(video_path)
        except OSError:
            cache_key = None
        if cache_key is not None and cache_key in self._video_frame_count_cache:
            return self._video_frame_count_cache[cache_key]

        frame_count_value: Optional[int] = None
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count > 0:
                    frame_count_value = frame_count
        except Exception as e:
            self._log_warning(f"Failed to get video frame count: {e}")
        if cache_key is not None and frame_count_value is not None:
            self._video_frame_count_cache[cache_key] = frame_count_value
        return frame_count_value

    def _get_synced_video_info_from_cache(
        self, video_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Return known LeRobot video info for converter-produced synced MP4s."""
        video_path = Path(video_path)
        if (
            not video_path.exists()
            or video_path.stat().st_size <= 0
            or not video_path.stem.endswith("_synced")
        ):
            return None
        cache_path = video_path.with_name(video_path.stem + ".cache.json")
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(cache, dict):
            return None
        try:
            fps = float(cache.get("target_fps"))
            height = int(cache["output_height"])
            width = int(cache["output_width"])
        except (KeyError, TypeError, ValueError):
            return None
        if fps <= 0 or height <= 0 or width <= 0:
            return None
        if int(round(fps)) != int(self.config.fps):
            return None
        return {
            'video.fps': float(fps),
            'video.height': int(height),
            'video.width': int(width),
            'video.channels': 3,
            'video.codec': str(cache.get("output_codec") or "h264"),
            'video.pix_fmt': str(cache.get("output_pix_fmt") or "yuv420p"),
            'video.is_depth_map': False,
            'has_audio': bool(cache.get("has_audio", False)),
        }

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Probe a video file via ffprobe and return the LeRobot v2.1 ``info`` block.

        LeRobot v2.1 features expect every ``observation.images.*`` to carry
        an ``info`` dict with codec / pix_fmt / fps / dimensions / channels /
        is_depth_map / has_audio. Falls back to the converter's known
        encode params (h264 / yuv420p / fps from config) if ffprobe is
        unavailable or fails.
        """
        import subprocess

        cached_info = self._get_synced_video_info_from_cache(video_path)
        if cached_info is not None:
            return cached_info

        height, width = self._get_video_dimensions(video_path)
        info = {
            'video.fps': float(self.config.fps),
            'video.height': int(height),
            'video.width': int(width),
            'video.channels': 3,
            'video.codec': 'h264',
            'video.pix_fmt': 'yuv420p',
            'video.is_depth_map': False,
            'has_audio': False,
        }
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-show_streams', '-print_format', 'json',
                    str(video_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return info
            probe = json.loads(result.stdout or '{}')
            streams = probe.get('streams') or []
            video_stream = next(
                (s for s in streams if s.get('codec_type') == 'video'), None,
            )
            if video_stream:
                if video_stream.get('codec_name'):
                    info['video.codec'] = video_stream['codec_name']
                if video_stream.get('pix_fmt'):
                    info['video.pix_fmt'] = video_stream['pix_fmt']
                # avg_frame_rate is "num/den"
                fr = video_stream.get('avg_frame_rate') or video_stream.get('r_frame_rate')
                if isinstance(fr, str) and '/' in fr:
                    num, _, den = fr.partition('/')
                    try:
                        d = float(den)
                        if d > 0:
                            info['video.fps'] = float(num) / d
                    except ValueError:
                        pass
            info['has_audio'] = any(
                s.get('codec_type') == 'audio' for s in streams
            )
        except (FileNotFoundError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            self._log_warning(f'ffprobe failed for {video_path}: {e}')
        return info

    def _video_feature_key(self, camera_name: str) -> str:
        return f"observation.images.{camera_name}"

    def _build_features(self, episodes_data: List[EpisodeData]):
        """Build feature definitions from episode data."""
        # Get dimensions from first episode
        first_ep = episodes_data[0]

        state_dim = (
            len(first_ep.observation_state[0]) if first_ep.observation_state else 0
        )
        action_dim = len(first_ep.action[0]) if first_ep.action else 0

        # Default features (required by LeRobot)
        self._features = {
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            "index": {"dtype": "int64", "shape": (1,), "names": None},
            "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        }
        if any(ep.subtask_indices for ep in episodes_data):
            self._features["subtask_index"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }

        # State / action joint names. Prefer per-episode names accumulated
        # by _merge_state_messages / _merge_action_messages, then fall back
        # to robot_config's joint_order grouped by side (follower_* for
        # state, leader_* for action) — that survives the
        # ProcessPoolExecutor parsing path, where worker children
        # populate the per-side _*_joint_names attributes but never
        # propagate them to the main process.
        #
        # Symmetry fallback: when joint_order only carries leader_*
        # (predecessor schema), the follower-prefix lookup returns []
        # and state would land on placeholder ``joint_N`` names. Reuse
        # the leader-prefix list for state if its dimension matches —
        # state and action describe the same joint set in our schema,
        # just observed vs commanded, so the names are identical.
        # Last resort: generic "joint_N" so the feature still has
        # names of the right length.
        state_names_from_config = self._joint_names_from_config('follower_')
        action_names_from_config = self._joint_names_from_config('leader_')

        def _names_for_dim(
            candidates: List[Optional[List[str]]],
            dim: int,
        ) -> Optional[List[str]]:
            for names in candidates:
                if names is not None and len(names) == dim:
                    return list(names)
            return None

        def _episode_names_for_dim(attr: str, dim: int) -> List[str]:
            for episode in episodes_data:
                names = list(getattr(episode, attr, []) or [])
                if len(names) == dim:
                    return names
            return []

        if state_dim > 0:
            self._features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": (
                    _names_for_dim(
                        [
                            _episode_names_for_dim(
                                "observation_state_names",
                                state_dim,
                            ),
                            self._state_joint_names,
                            state_names_from_config,
                            action_names_from_config,
                        ],
                        state_dim,
                    )
                    or [f"joint_{i}" for i in range(state_dim)]
                ),
            }

        if action_dim > 0:
            self._features["action"] = {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": (
                    _names_for_dim(
                        [
                            _episode_names_for_dim("action_names", action_dim),
                            self._action_joint_names,
                            action_names_from_config,
                        ],
                        action_dim,
                    )
                    or [f"joint_{i}" for i in range(action_dim)]
                ),
            }

        # Add video features. LeRobot v2.1 spec: shape is CHW (3, H, W),
        # names track shape order, and an ``info`` block carries codec /
        # pix_fmt / fps / dimensions / has_audio for downstream loaders.
        for ep in episodes_data:
            for camera_name, video_path in ep.video_files.items():
                feature_key = self._video_feature_key(camera_name)
                if feature_key not in self._features:
                    info = self._get_video_info(video_path)
                    self._features[feature_key] = {
                        "dtype": "video",
                        "shape": (3, int(info['video.height']), int(info['video.width'])),
                        "names": ["channels", "height", "width"],
                        "info": info,
                    }

    def _collect_tasks(self, episodes_data: List[EpisodeData]):
        """Populate ``self._tasks`` / ``self._task_to_index`` in episode-first-appearance order.

        Sorting alphabetically would make tasks.jsonl disagree with the
        order tasks show up in episodes.jsonl when the episode ordering
        itself isn't alphabetical (e.g. after a merge that placed source
        folders in a non-alphabetical order).
        """
        seen: set = set()
        for ep in episodes_data:
            for task in ep.tasks:
                if task in seen:
                    continue
                seen.add(task)
                idx = len(self._tasks)
                self._tasks[idx] = task
                self._task_to_index[task] = idx

    def _write_root_info_json(self) -> None:
        """Write the root-level info.json (conversion config snapshot).

        This sits at the dataset root (alongside README.md / data/ /
        meta/ / videos/) and records the choices the user made when
        running the conversion: which sources, which cameras / topics /
        joints were selected, what rotations / resize were applied,
        which episodes were skipped per source. It's a write-only audit
        artifact — downstream LeRobot loaders read meta/info.json, not
        this one — but external tools and the next reader benefit from
        knowing what knob was set when the dataset was produced.
        """
        output_dir = Path(self.config.output_dir)
        # Audit snapshot — fill empty selection fields from the discovered
        # robot_config defaults so the recorded config reflects what was
        # actually used in the conversion (not just what the caller
        # explicitly set).
        cameras = list(self.config.selected_cameras) or list(
            self._camera_mapping.values()
        )
        state_topics = self._audit_topic_selection(
            list(self.config.selected_state_topics),
            list(self.config.state_topics),
            self._state_topic_key_map,
            strip_prefix='follower_',
        )
        action_topics = self._audit_topic_selection(
            list(self.config.selected_action_topics),
            list(self.config.action_topics),
            self._action_topic_key_map,
            strip_prefix='leader_',
        )
        task_name = self._root_task_name()
        # Camera rotations — include every known camera with an explicit
        # 0 for unrotated, mirroring the reference dataset's snapshot.
        # Robot-config rotations are the source of truth because rotation
        # is applied at raw recording time. Explicit conversion rotations
        # are retained as an override for legacy/manual conversions that
        # do not have robot-config provenance available.
        rotation_source = dict(getattr(self, '_camera_rotations', {}) or {})
        for cam, deg in self.config.camera_rotations.items():
            deg = int(deg)
            if cam not in rotation_source or deg:
                rotation_source[cam] = deg
        rotations: Dict[str, int] = {}
        for cam in cameras:
            deg = int(rotation_source.get(cam, 0) or 0)
            if deg:
                rotations[cam] = deg
        snapshot = {
            'source_rosbags': list(self.config.source_rosbags),
            'conversion_config': {
                'robot_type': self.config.robot_type,
                'task_name': task_name,
                'fps': int(self.config.fps),
                'camera_rotations': rotations,
                'selected_end_effector_topics': [],
                'selected_cameras': cameras,
                'output_dataset_name': output_dir.name,
                'image_resize': (
                    list(self.config.image_resize)
                    if self.config.image_resize else None
                ),
                'selected_joint_state_topics': state_topics,
                'primitive_instructions': [],
                'selected_action_topics': action_topics,
            },
        }
        info_path = output_dir / 'info.json'
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            self._log_info(f'Wrote root info.json: {info_path}')
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f'Failed to write root info.json: {exc}')

    def _root_task_name(self) -> str:
        """Return the main task instruction for root conversion_config."""
        tasks = [
            str(task or '').strip()
            for _, task in sorted(getattr(self, '_tasks', {}).items())
            if str(task or '').strip()
        ]
        if len(tasks) == 1:
            return tasks[0]
        return ''

    @staticmethod
    def _audit_topic_selection(
        selected: List[str],
        defaults: List[str],
        topic_key_map: Dict[str, str],
        *,
        strip_prefix: str = '',
    ) -> List[str]:
        """Return robot-config logical names for root info.json audit fields."""
        values = selected or defaults
        known_keys = set(topic_key_map.values())
        result: List[str] = []
        for value in values:
            key = topic_key_map.get(value, value)
            if key not in known_keys and value in known_keys:
                key = value
            if strip_prefix and key.startswith(strip_prefix):
                key = key[len(strip_prefix):]
            if key not in result:
                result.append(key)
        return result

    def _compute_episode_stats(
        self, episode: EpisodeData, global_start_index: int = 0,
    ) -> Dict[str, Dict]:
        """Compute statistics for an episode (LeRobot v2.1 format)."""
        stats = {}
        num_frames = episode.length

        if episode.observation_state:
            states = np.array(episode.observation_state)
            stats["observation.state"] = {
                "mean": np.mean(states, axis=0).tolist(),
                "std": np.maximum(np.std(states, axis=0), STATS_STD_FLOOR).tolist(),
                "min": np.min(states, axis=0).tolist(),
                "max": np.max(states, axis=0).tolist(),
                "count": [num_frames],
            }

        if episode.action:
            actions = np.array(episode.action)
            stats["action"] = {
                "mean": np.mean(actions, axis=0).tolist(),
                "std": np.maximum(np.std(actions, axis=0), STATS_STD_FLOOR).tolist(),
                "min": np.min(actions, axis=0).tolist(),
                "max": np.max(actions, axis=0).tolist(),
                "count": [num_frames],
            }

        for camera_name, video_path in episode.video_files.items():
            feature_key = self._video_feature_key(camera_name)
            video_stats = self._compute_video_stats(video_path, camera_name)
            if video_stats:
                stats[feature_key] = video_stats

        # Per-frame index / timestamp stats — LeRobot v2.1 reference
        # carries these alongside the data feature stats so downstream
        # tooling can reason about ranges without re-reading the parquet.
        if num_frames > 0:
            timestamps = np.array(episode.timestamps, dtype=np.float64) \
                if episode.timestamps else np.arange(num_frames, dtype=np.float64) / float(
                    self.config.fps if self.config.fps else 1
                )
            stats["timestamp"] = self._scalar_stats(timestamps, num_frames)

            frame_idx = np.arange(num_frames, dtype=np.int64)
            stats["frame_index"] = self._scalar_stats(frame_idx, num_frames)

            global_idx = np.arange(num_frames, dtype=np.int64) + int(global_start_index)
            stats["index"] = self._scalar_stats(global_idx, num_frames)

            ep_idx_arr = np.full(num_frames, episode.episode_index, dtype=np.int64)
            stats["episode_index"] = self._scalar_stats(ep_idx_arr, num_frames)

            task = episode.tasks[0] if episode.tasks else "default_task"
            ti = self._task_to_index.get(task, 0)
            ti_arr = np.full(num_frames, ti, dtype=np.int64)
            stats["task_index"] = self._scalar_stats(ti_arr, num_frames)

        return stats

    @staticmethod
    def _scalar_stats(arr: np.ndarray, num_frames: int) -> Dict[str, Any]:
        """min/max/mean/std/count for a 1-D scalar array, wrapped in lists."""
        return {
            "min": [arr.min().item()],
            "max": [arr.max().item()],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [num_frames],
        }

    def _load_precomputed_video_stats(
        self, video_path: Path, camera_name: str
    ) -> Optional[Dict]:
        """Try to load pre-computed video stats from video_stats.json (Stage 1)."""
        stats_path = video_path.parent / "video_stats.json"
        all_stats = self._video_stats_sidecar_cache.get(stats_path)
        if all_stats is None:
            if not stats_path.exists():
                return None
            try:
                with open(stats_path, "r") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    return None
                all_stats = loaded
                self._video_stats_sidecar_cache[stats_path] = all_stats
            except Exception as e:
                self._log_warning(
                    f"Failed to load pre-computed stats from {stats_path}: {e}"
                )
                return None
        try:
            if camera_name in all_stats:
                return all_stats[camera_name]
        except Exception as e:
            self._log_warning(
                f"Failed to load pre-computed stats from {stats_path}: {e}"
            )
        return None

    @staticmethod
    def _video_stats_sample_budget(max_samples: Optional[int] = None) -> int:
        if max_samples is not None:
            return max(0, int(max_samples))
        if _VIDEO_STATS_SAMPLES_ENV not in os.environ:
            profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
            if profile in {"max", "maximum", "max_speed", "fastest"}:
                return 0
        raw = os.environ.get(
            _VIDEO_STATS_SAMPLES_ENV,
            str(_DEFAULT_VIDEO_STATS_SAMPLES),
        )
        try:
            return max(0, int(raw))
        except ValueError:
            return _DEFAULT_VIDEO_STATS_SAMPLES

    def _compute_video_stats(
        self,
        video_path: Path,
        camera_name: str = "",
        max_samples: Optional[int] = None,
    ) -> Optional[Dict]:
        """Compute video statistics (per-channel RGB, normalized to [0,1]).

        First checks for pre-computed stats from Stage 1 (video_stats.json).
        Falls back to decoding MP4 if not available.
        """
        # Try pre-computed stats first
        precomputed = self._load_precomputed_video_stats(video_path, camera_name)
        if precomputed is not None:
            return precomputed

        sample_budget = self._video_stats_sample_budget(max_samples)
        if sample_budget <= 0:
            return None

        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None
            sample_indices = np.linspace(
                0,
                total_frames - 1,
                min(sample_budget, total_frames),
                dtype=int,
            )

            samples = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    samples.append(frame_rgb)

            cap.release()

            if not samples:
                return None

            frames = np.array(samples, dtype=np.float32) / 255.0
            r_channel = frames[:, :, :, 0]
            g_channel = frames[:, :, :, 1]
            b_channel = frames[:, :, :, 2]

            def channel_stats(channel):
                return {
                    "mean": float(np.mean(channel)),
                    "std": float(np.std(channel)),
                    "min": float(np.min(channel)),
                    "max": float(np.max(channel)),
                }

            r_stats = channel_stats(r_channel)
            g_stats = channel_stats(g_channel)
            b_stats = channel_stats(b_channel)

            return {
                "min": [[[r_stats["min"]]], [[g_stats["min"]]], [[b_stats["min"]]]],
                "max": [[[r_stats["max"]]], [[g_stats["max"]]], [[b_stats["max"]]]],
                "mean": [[[r_stats["mean"]]], [[g_stats["mean"]]], [[b_stats["mean"]]]],
                "std": [[[r_stats["std"]]], [[g_stats["std"]]], [[b_stats["std"]]]],
                "count": [len(samples)],
            }
        except Exception as e:
            self._log_warning(f"Failed to compute video stats for {video_path}: {e}")
            return None

    def _serialize_stats(self, stats: Dict) -> Dict:
        """Serialize stats dictionary for JSON."""
        serialized = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_stats(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, int, float)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    def _write_quality_reports(self, output_dir: Path) -> None:
        """Dump per-episode quality reports as JSON when ``enable_quality_report``.

        Currently nothing populates ``_quality_reports`` — kept as a
        forward-compatible stub so the v3.0 writer's
        ``if self.config.enable_quality_report and self._quality_reports:``
        branch doesn't AttributeError if a future producer fills the dict.
        """
        if not self._quality_reports:
            return
        report_path = Path(output_dir) / "meta" / "quality_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self._quality_reports, f, indent=2, ensure_ascii=False)
            self._log_info(f"Wrote quality report: {report_path}")
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"Failed to write quality report: {exc}")
