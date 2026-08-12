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
ROSbag + MP4 to LeRobot v3.0 Dataset Converter.

Converts recorded robot data (ROSbag with joint states + MP4 videos) to
LeRobot v3.0 dataset format for training with LeRobot framework.

Key differences from v2.1:
- File-based storage: Multiple episodes per Parquet/MP4 file
- Episodes metadata stored as chunked Parquet (not JSONL)
- Tasks stored as Parquet (not JSONL)
- Video files concatenated per camera

LeRobot v3.0 Dataset Structure:
    dataset_name/
    ├── data/
    │   └── chunk-{chunk:03d}/
    │       └── file-{file:03d}.parquet        # Multiple episodes per file
    ├── meta/
    │   ├── info.json
    │   ├── stats.json                         # Global statistics
    │   ├── tasks.parquet                      # Task index -> task string
    │   └── episodes/
    │       └── chunk-{chunk:03d}/
    │           └── file-{file:03d}.parquet    # Episode metadata with offsets
    └── videos/
        └── {camera_key}/
            └── chunk-{chunk:03d}/
                └── file-{file:03d}.mp4        # Multiple episodes concatenated
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Shared rosbag-extraction / stats / feature-building lives in
# base_converter.py. v3.0 inherits the base directly (sibling of v2.1)
# rather than chaining through v21 — formats are independent now.
from .base_converter import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
    STATS_STD_FLOOR,
    StalenessMetrics,
    _active_conversion_workers,
    _clone_or_copy_file,
    _conversion_worker_init,
    _convert_rosbag_worker,
    _fast_absolute_path,
    _resolve_conversion_worker_count,
    _same_file_or_same_path,
)
from .video_sync import (
    _VIDEO_SYNC_DISABLE_COPY_FASTPATH_ENV,
    _StreamingRgbStats,
    _add_frame_yuv420p_for_stats,
    _contiguous_forward_run_length,
    _drain_exact,
    _ffmpeg,
    _ffmpeg_h264_decoder_args,
    _ffmpeg_threads_arg,
    _force_h264_software_encoder,
    _h264_encoder,
    _mp4_faststart_args,
    _ffmpeg_pipe_size,
    _quick_video_dimensions,
    _read_exact,
    _read_exact_into,
    _set_pipe_size,
    _splice_exact,
    _terminate_process,
    _validated_video_count,
    _video_stats_sample_positions,
    _write_repeated_frame_bytes,
    remux_selected_frames,
)


CODEBASE_VERSION_V30 = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

# Path templates for v3.0
CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = "meta/episodes/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = "data/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
_VIDEO_AGG_CAMERA_WORKERS_ENV = "CYCLO_VIDEO_AGG_CAMERA_WORKERS"
_V30_DIRECT_AGG_DISABLE_ENV = "CYCLO_V30_DISABLE_DIRECT_AGGREGATE"
_V30_CONCAT_DECODER_DISABLE_ENV = "CYCLO_V30_DISABLE_CONCAT_DECODER"
_V30_TRUST_SIDECAR_FRAME_COUNT_ENV = "CYCLO_V30_TRUST_SIDECAR_FRAME_COUNT"
_V30_VALIDATE_DIRECT_AGGREGATE_ENV = "CYCLO_V30_VALIDATE_DIRECT_AGGREGATE"
_V30_SOURCE_AGG_CACHE_DISABLE_ENV = "CYCLO_V30_DISABLE_SOURCE_AGGREGATE_CACHE"
_V30_SOURCE_AGG_CACHE_POPULATE_ENV = "CYCLO_V30_POPULATE_SOURCE_AGGREGATE_CACHE"
_V30_WRITE_SOURCE_REUSE_OUTPUT_CACHE_ENV = (
    "CYCLO_V30_WRITE_SOURCE_REUSE_OUTPUT_CACHE"
)
_V30_DATA_AGG_CACHE_DISABLE_ENV = "CYCLO_V30_DISABLE_DATA_AGGREGATE_CACHE"
_V30_PARQUET_COMPRESSION_ENV = "CYCLO_V30_PARQUET_COMPRESSION"
_V30_PARQUET_USE_DICTIONARY_ENV = "CYCLO_V30_PARQUET_USE_DICTIONARY"
_V30_DATA_AGGREGATE_CACHE_VERSION = 2
_V30_EPISODES_PARQUET_CACHE_VERSION = 1
_V30_TASKS_PARQUET_CACHE_VERSION = 1
_V30_SUBTASKS_PARQUET_CACHE_VERSION = 1
_FICLONE_IOCTL = 0x40049409
_REFLINK_UNSUPPORTED_DEV_PAIRS: set[Tuple[int, int]] = set()
_H264_ENCODER_TUNING_ENVS = {
    "CYCLO_H264_ENCODER",
    "CYCLO_X264_PRESET",
    "CYCLO_X264_CRF",
    "CYCLO_X264_QP",
    "CYCLO_X264_TUNE",
    "CYCLO_X264_GOP",
    "CYCLO_X264_THREADS",
}
_MAX_SPEED_PROFILES = {"max", "maximum", "max_speed", "fastest"}
_VideoAggregateJob = Tuple[str, int, int, List[Tuple[int, Path, float]]]
_DirectSourceCacheMatch = Tuple[Path, Path, Dict[str, Any]]


class _DirectV30ConcatDecoderError(RuntimeError):
    """Raised when v3 direct aggregation cannot stream concat-decoded frames."""


def _concat_file_line(video_path: Path) -> str:
    path = _fast_absolute_path(Path(video_path)).replace("'", "'\\''")
    return f"file '{path}'\n"


def _v30_max_speed_profile() -> bool:
    profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
    return profile in _MAX_SPEED_PROFILES


def _v30_parquet_write_kwargs() -> Dict[str, Any]:
    """Return Parquet writer tuning for the active conversion profile."""
    kwargs: Dict[str, Any] = {}
    raw_compression = os.environ.get(_V30_PARQUET_COMPRESSION_ENV)
    if raw_compression is not None:
        compression = raw_compression.strip()
        if compression.lower() in {"", "0", "false", "none", "off", "uncompressed"}:
            kwargs["compression"] = None
        else:
            kwargs["compression"] = compression
    elif _v30_max_speed_profile():
        kwargs["compression"] = None

    raw_dictionary = os.environ.get(_V30_PARQUET_USE_DICTIONARY_ENV)
    if raw_dictionary is not None:
        kwargs["use_dictionary"] = (
            raw_dictionary.strip().lower() in {"1", "true", "yes", "on"}
        )
    elif _v30_max_speed_profile():
        kwargs["use_dictionary"] = False
    return kwargs


@dataclass
class V30ConversionConfig(ConversionConfig):
    """Extended configuration for v3.0 conversion."""

    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    enable_quality_report: bool = False


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode in v3.0 format."""

    episode_index: int
    length: int
    tasks: List[str]
    recording_mode: str = "single"
    full_episode_index: Optional[int] = None
    subtask_instructions: List[str] = field(default_factory=list)

    # Data file location
    data_chunk_index: int = 0
    data_file_index: int = 0
    dataset_from_index: int = 0  # Start frame index within file
    dataset_to_index: int = 0  # End frame index within file

    # Video file locations (per camera)
    video_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Episode statistics (flattened)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Parquet storage."""
        result = {
            "episode_index": self.episode_index,
            "length": self.length,
            "tasks": self.tasks,
            "recording_mode": self.recording_mode,
            "data/chunk_index": self.data_chunk_index,
            "data/file_index": self.data_file_index,
            "dataset_from_index": self.dataset_from_index,
            "dataset_to_index": self.dataset_to_index,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        if self.full_episode_index is not None:
            result["full_episode_index"] = self.full_episode_index
        if self.subtask_instructions:
            result["subtask_instructions"] = self.subtask_instructions

        # Add video metadata per camera
        for camera_key, video_info in self.video_metadata.items():
            result[f"videos/{camera_key}/chunk_index"] = video_info.get(
                "chunk_index", 0
            )
            result[f"videos/{camera_key}/file_index"] = video_info.get("file_index", 0)
            result[f"videos/{camera_key}/from_timestamp"] = video_info.get(
                "from_timestamp", 0.0
            )
            result[f"videos/{camera_key}/to_timestamp"] = video_info.get(
                "to_timestamp", 0.0
            )

        # Add flattened stats
        for stat_key, stat_value in self.stats.items():
            if isinstance(stat_value, np.ndarray):
                result[f"stats/{stat_key}"] = stat_value.tolist()
            else:
                result[f"stats/{stat_key}"] = stat_value

        return result

    @classmethod
    def from_cache_dict(cls, raw: Dict[str, Any]) -> "EpisodeMetadata":
        """Recreate internal metadata from the v3 data aggregate cache."""
        full_episode_index = raw.get("full_episode_index")
        return cls(
            episode_index=int(raw.get("episode_index", 0)),
            length=int(raw.get("length", 0)),
            tasks=list(raw.get("tasks") or []),
            recording_mode=str(raw.get("recording_mode", "single")),
            full_episode_index=(
                int(full_episode_index)
                if full_episode_index is not None else None
            ),
            subtask_instructions=list(raw.get("subtask_instructions") or []),
            data_chunk_index=int(raw.get("data_chunk_index", 0)),
            data_file_index=int(raw.get("data_file_index", 0)),
            dataset_from_index=int(raw.get("dataset_from_index", 0)),
            dataset_to_index=int(raw.get("dataset_to_index", 0)),
            video_metadata=dict(raw.get("video_metadata") or {}),
            stats=dict(raw.get("stats") or {}),
        )

    def to_cache_dict(self) -> Dict[str, Any]:
        """Serialize internal metadata without flattening Parquet column names."""
        return {
            "episode_index": int(self.episode_index),
            "length": int(self.length),
            "tasks": list(self.tasks),
            "recording_mode": self.recording_mode,
            "full_episode_index": self.full_episode_index,
            "subtask_instructions": list(self.subtask_instructions),
            "data_chunk_index": int(self.data_chunk_index),
            "data_file_index": int(self.data_file_index),
            "dataset_from_index": int(self.dataset_from_index),
            "dataset_to_index": int(self.dataset_to_index),
            "video_metadata": dict(self.video_metadata),
            "stats": self.stats,
        }


class RosbagToLerobotV30Converter(RosbagToLerobotConverterBase):
    """
    Converts ROSbag recordings with MP4 videos to LeRobot v3.0 dataset format.

    Sibling of the v2.1 converter — both inherit shared rosbag
    extraction / stats logic from RosbagToLerobotConverterBase. Adds
    v3.0-specific writers:
    - File-based aggregation (multiple episodes per file)
    - Chunked Parquet episodes metadata
    - Video concatenation
    - Global statistics in stats.json
    """

    def __init__(self, config: V30ConversionConfig, logger=None):
        super().__init__(config, logger)
        self.config: V30ConversionConfig = config

        # v3.0 specific tracking
        self._episode_metadata_list: List[EpisodeMetadata] = []
        self._current_data_chunk_idx = 0
        self._current_data_file_idx = 0
        self._current_data_file_size_mb = 0.0
        self._current_data_file_frames = 0
        self._episode_metadata_by_index: Dict[int, EpisodeMetadata] = {}
        self._current_data_aggregate_cache_key: Optional[Dict[str, Any]] = None

        # Per-camera video tracking
        self._video_tracking: Dict[str, Dict[str, Any]] = {}

        # Temporary storage for aggregation
        self._pending_parquet_data: List[Any] = []
        self._pending_video_files: Dict[
            str, List[Tuple[Path, float]]
        ] = {}  # camera -> [(path, duration)]
        self._direct_video_aggregation = False
        self._direct_video_sources_by_episode: Dict[int, Dict[str, Path]] = {}
        self._direct_video_stats_cache: Dict[
            Tuple[int, str], Optional[Dict[str, Any]]
        ] = {}
        self._quick_video_dimensions_cache: Dict[Path, Tuple[int, int]] = {}
        self._quick_video_dimensions_lock = threading.Lock()
        self._direct_source_cache_root_cache: Dict[Path, Path] = {}
        self._direct_source_episode_root_cache: Dict[Path, Path] = {}
        self._direct_source_dataset_sibling_cache: Dict[Path, bool] = {}
        self._direct_source_aggregate_cache_dir_cache: Dict[
            Tuple[str, ...], Optional[Path]
        ] = {}

    def _video_feature_key(self, camera_name: str) -> str:
        return f"observation.images.rgb.{camera_name}"

    def _video_dimensions_cache_key(self, video_path: Path) -> Path:
        return Path(_fast_absolute_path(Path(video_path)))

    def _remember_video_dimensions(
        self,
        video_path: Path,
        width: int,
        height: int,
    ) -> None:
        width_i = int(width)
        height_i = int(height)
        if width_i <= 0 or height_i <= 0:
            return
        key = self._video_dimensions_cache_key(video_path)
        with self._quick_video_dimensions_lock:
            self._quick_video_dimensions_cache[key] = (width_i, height_i)

    def _cached_video_dimensions(self, video_path: Path) -> Tuple[int, int]:
        key = self._video_dimensions_cache_key(video_path)
        with self._quick_video_dimensions_lock:
            return self._quick_video_dimensions_cache.get(key, (0, 0))

    def _aggregate_cache_video_dimensions(self, video_path: Path) -> Tuple[int, int]:
        try:
            cache = json.loads(
                self._aggregated_video_cache_path(Path(video_path)).read_text(
                    encoding="utf-8",
                )
            )
            width = int(cache.get("output_width") or 0)
            height = int(cache.get("output_height") or 0)
        except (OSError, ValueError, TypeError):
            return 0, 0
        self._remember_video_dimensions(video_path, width, height)
        return width, height

    def _remember_video_dimensions_from_cache_key(
        self,
        cache_key: Dict[str, Any],
        *,
        output_path: Optional[Path] = None,
    ) -> None:
        width = int(cache_key.get("output_width") or 0)
        height = int(cache_key.get("output_height") or 0)
        if output_path is not None:
            self._remember_video_dimensions(output_path, width, height)
        if width <= 0 or height <= 0:
            return
        inputs = cache_key.get("inputs")
        if not isinstance(inputs, list):
            return
        for item in inputs:
            if not isinstance(item, dict):
                continue
            video = item.get("video")
            if not isinstance(video, dict):
                continue
            source_path = video.get("path")
            if source_path:
                self._remember_video_dimensions(Path(source_path), width, height)

    def _quick_video_dimensions_cached(self, video_path: Path) -> Tuple[int, int]:
        """Return cached ``(width, height)`` metadata for direct video jobs."""
        path = self._video_dimensions_cache_key(video_path)
        with self._quick_video_dimensions_lock:
            cached = self._quick_video_dimensions_cache.get(path)
            if cached is not None:
                return cached
        dims = _quick_video_dimensions(path)
        with self._quick_video_dimensions_lock:
            self._quick_video_dimensions_cache[path] = dims
        return dims

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Return metadata for v3 aggregate outputs."""
        width, height = self._cached_video_dimensions(Path(video_path))
        if width <= 0 or height <= 0:
            width, height = self._aggregate_cache_video_dimensions(Path(video_path))
        if width <= 0 or height <= 0:
            width, height = self._quick_video_dimensions_cached(Path(video_path))
        if width <= 0 or height <= 0:
            height, width = self._get_video_dimensions(Path(video_path))
        return {
            "video.fps": float(self.config.fps),
            "video.height": int(height),
            "video.width": int(width),
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }

    def _direct_video_sources_for_bag(
        self,
        bag_path: Path,
    ) -> Optional[Dict[str, Path]]:
        """Return raw per-camera MP4s eligible for direct v3 aggregation."""
        bag_path = Path(bag_path)
        episode_info = self._metadata_manager.load_episode_info(bag_path)
        if self._is_archived_segment_episode(bag_path, episode_info):
            segments = episode_info.get("segments") or []
            if len(segments) != 1:
                return None
            mcap_paths = self._segment_mcap_paths(bag_path, len(segments))
            if len(mcap_paths) != 1:
                return None
            videos = self._find_segment_video_files(bag_path, mcap_paths[0].stem)
        else:
            videos = self._find_video_files(bag_path)

        if not videos:
            return None
        sources = {name: Path(path) for name, path in videos.items()}
        for camera_name, video_path in videos.items():
            sidecar = Path(video_path).parent / f"{camera_name}_timestamps.parquet"
            if not sidecar.exists():
                return None
        self._remember_direct_source_roots_for_bag(bag_path, sources)
        return sources

    def _remember_direct_source_roots_for_bag(
        self,
        bag_path: Path,
        sources: Dict[str, Path],
    ) -> None:
        """Seed direct source-cache root lookups during video discovery."""
        if not sources:
            return
        try:
            episode_dir = Path(_fast_absolute_path(Path(bag_path)))
            root = self._episode_source_cache_root(episode_dir)
        except Exception:
            return
        self._direct_source_episode_root_cache[episode_dir] = root
        for video_path in sources.values():
            path = Path(_fast_absolute_path(Path(video_path)))
            self._direct_source_cache_root_cache[path] = root
            videos_dir = path.parent.parent
            if videos_dir.name == "videos":
                self._direct_source_episode_root_cache[videos_dir.parent] = root

    def _can_use_direct_video_aggregation(
        self,
        bag_paths: List[Path],
    ) -> bool:
        """True when v3 can skip per-episode synced MP4 intermediates."""
        self._direct_video_sources_by_episode = {}
        if os.environ.get(_V30_DIRECT_AGG_DISABLE_ENV):
            return False
        if not self.config.use_videos:
            return False
        if self.config.image_resize is not None:
            return False
        if any(int(value or 0) for value in self.config.camera_rotations.values()):
            return False
        if not bag_paths:
            return False

        for idx, bag_path in enumerate(bag_paths):
            sources = self._direct_video_sources_for_bag(Path(bag_path))
            if not sources:
                self._direct_video_sources_by_episode = {}
                return False
            self._direct_video_sources_by_episode[int(idx)] = sources
        return True

    def _attach_direct_video_sources(
        self,
        episodes_data: List[EpisodeData],
    ) -> None:
        """Attach raw source videos after no-video episode extraction."""
        for episode in episodes_data:
            sources = self._direct_video_sources_by_episode.get(
                int(episode.episode_index)
            )
            if not sources:
                raise RuntimeError(
                    "direct v3 aggregation source videos missing for "
                    f"episode {episode.episode_index}"
                )
            episode.video_files = dict(sources)

    def _compute_episode_stats(
        self,
        episode: EpisodeData,
        global_start_index: int = 0,
    ) -> Dict[str, Dict]:
        if not self._direct_video_aggregation or not episode.video_files:
            return super()._compute_episode_stats(episode, global_start_index)

        raw_video_files = dict(episode.video_files)
        episode.video_files = {}
        try:
            return super()._compute_episode_stats(episode, global_start_index)
        finally:
            episode.video_files = raw_video_files

    def _compute_direct_video_stats(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """Sample stats from exact raw frames selected for direct v3 output."""
        cache_key = (int(episode.episode_index), camera_name)
        if cache_key in self._direct_video_stats_cache:
            return self._direct_video_stats_cache[cache_key]

        sample_budget = self._video_stats_sample_budget()
        if sample_budget <= 0 or episode.length <= 0:
            self._direct_video_stats_cache[cache_key] = None
            return None
        try:
            import cv2

            indices = self._grid_indices_for_raw_video(
                episode,
                camera_name,
                Path(video_path),
            )
            if indices.size == 0:
                self._direct_video_stats_cache[cache_key] = None
                return None
            sample_positions = np.linspace(
                0,
                int(indices.size) - 1,
                min(sample_budget, int(indices.size)),
                dtype=int,
            )
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self._direct_video_stats_cache[cache_key] = None
                return None
            samples = []
            try:
                for out_idx in sample_positions:
                    src_idx = int(indices[int(out_idx)])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                    ok, frame = cap.read()
                    if ok:
                        samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            if not samples:
                self._direct_video_stats_cache[cache_key] = None
                return None

            frames = np.asarray(samples, dtype=np.float32) / 255.0
            channels = [frames[:, :, :, idx] for idx in range(3)]
            result = {
                "min": [[[float(np.min(ch))]] for ch in channels],
                "max": [[[float(np.max(ch))]] for ch in channels],
                "mean": [[[float(np.mean(ch))]] for ch in channels],
                "std": [[[float(np.std(ch))]] for ch in channels],
                "count": [len(samples)],
            }
            self._direct_video_stats_cache[cache_key] = result
            return result
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{camera_name}: direct video stats failed "
                f"for episode {episode.episode_index}: {exc!r}"
            )
            self._direct_video_stats_cache[cache_key] = None
            return None

    def _merge_direct_video_stats_into_metadata(self) -> None:
        """Attach sampled direct-aggregate video stats to episode metadata."""
        if not self._direct_video_stats_cache:
            return
        metadata_by_index = {
            int(metadata.episode_index): metadata
            for metadata in self._episode_metadata_list
        }
        for (episode_index, camera_name), stats in self._direct_video_stats_cache.items():
            if not stats:
                continue
            metadata = metadata_by_index.get(int(episode_index))
            if metadata is None:
                continue
            feature_key = self._video_feature_key(camera_name)
            for stat_type, stat_value in stats.items():
                metadata.stats[f"{feature_key}/{stat_type}"] = stat_value

    def convert_multiple_rosbags(self, bag_paths: List[Path]) -> bool:
        """
        Convert multiple ROSbag recordings to a single LeRobot v3.0 dataset.

        Args:
            bag_paths: List of paths to ROSbag directories

        Returns:
            True if successful, False otherwise
        """
        self._log_info(f"Converting {len(bag_paths)} rosbags to LeRobot v3.0 dataset")
        self._reset_frame_reuse_reports()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure
        (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

        direct_video_aggregation = self._can_use_direct_video_aggregation(
            [Path(path) for path in bag_paths]
        )
        original_use_videos = bool(self.config.use_videos)
        if direct_video_aggregation:
            self._log_info(
                "Using v3 direct video aggregation fast path "
                "(raw MP4 + sidecar -> final aggregate)"
            )
            self.config.use_videos = False

        episodes_data: List[EpisodeData] = []
        cached_episode_indices: set[int] = set()
        try:
            if len(bag_paths) > 1:
                for idx, bag_path in enumerate(bag_paths):
                    cached_episode = self._try_load_prepared_episode_for_bag(
                        Path(bag_path),
                        idx,
                    )
                    if cached_episode is not None:
                        episodes_data.append(cached_episode)
                        cached_episode_indices.add(idx)
                if cached_episode_indices:
                    self._log_info(
                        f"Parent reused {len(cached_episode_indices)}/"
                        f"{len(bag_paths)} prepared episode cache entries"
                    )

            missing_bag_paths = [
                (idx, Path(bag_path))
                for idx, bag_path in enumerate(bag_paths)
                if idx not in cached_episode_indices
            ]

            if len(bag_paths) <= 1:
                for idx, bag_path in enumerate(bag_paths):
                    episode_data = self.convert_single_rosbag(Path(bag_path), idx)
                    if episode_data is not None:
                        episodes_data.append(episode_data)
            elif not missing_bag_paths:
                episodes_data.sort(key=lambda ep: ep.episode_index)
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                config_dict = {
                    'repo_id': self.config.repo_id,
                    'output_dir': self.config.output_dir,
                    'fps': self.config.fps,
                    'robot_type': self.config.robot_type,
                    'use_videos': self.config.use_videos,
                    'chunks_size': self.config.chunks_size,
                    'robot_config_path': self.config.robot_config_path,
                    'state_topics': self.config.state_topics,
                    'action_topics': self.config.action_topics,
                    'apply_trim': self.config.apply_trim,
                    'apply_exclude_regions': self.config.apply_exclude_regions,
                    'quality_warning_multiplier': self.config.quality_warning_multiplier,
                    'quality_error_multiplier': self.config.quality_error_multiplier,
                    'selected_cameras': list(self.config.selected_cameras),
                    'camera_rotations': dict(self.config.camera_rotations),
                    'image_resize': (
                        tuple(self.config.image_resize)
                        if self.config.image_resize else None
                    ),
                    'selected_state_topics': list(self.config.selected_state_topics),
                    'selected_action_topics': list(self.config.selected_action_topics),
                    'selected_joints': list(self.config.selected_joints),
                    'source_rosbags': list(self.config.source_rosbags),
                }
                max_workers = _resolve_conversion_worker_count(len(missing_bag_paths))
                self._log_info(
                    f"Starting parallel rosbag parsing with {max_workers} workers "
                    f"for {len(missing_bag_paths)} cache miss(es)"
                )
                with _active_conversion_workers(max_workers):
                    with ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_conversion_worker_init,
                    ) as executor:
                        futures = {}
                        for idx, bag_path in missing_bag_paths:
                            future = executor.submit(
                                _convert_rosbag_worker,
                                str(bag_path), idx, config_dict,
                            )
                            futures[future] = idx

                        for future in as_completed(futures):
                            idx = futures[future]
                            try:
                                episode_index, episode_data = future.result()
                                if episode_data is not None:
                                    episodes_data.append(episode_data)
                                    self._log_info(
                                        f"Episode {episode_index} parsed successfully"
                                    )
                                else:
                                    self._log_warning(
                                        f"Episode {idx} returned no data"
                                    )
                            except Exception as e:
                                self._log_error(
                                    f"Error parsing episode {idx}: {e}"
                                )
                episodes_data.sort(key=lambda ep: ep.episode_index)
        finally:
            self.config.use_videos = original_use_videos

        if not episodes_data:
            self._log_error("No episodes were successfully converted")
            return False

        self._direct_video_aggregation = bool(direct_video_aggregation)
        if self._direct_video_aggregation:
            self._attach_direct_video_sources(episodes_data)

        success = self.write_from_episodes(episodes_data)
        if success:
            self._cleanup_output_temp_dirs()
            self._cleanup_source_synced_cache([Path(path) for path in bag_paths])
        return success

    def write_from_episodes(self, episodes_data: List[EpisodeData]) -> bool:
        """Write a v3.0 dataset from already parsed episodes."""
        if not episodes_data:
            self._log_error("No episodes were provided for LeRobot v3.0 writing")
            return False
        self._reset_frame_reuse_reports()
        episodes_data = self.prepare_episodes_for_writing(episodes_data)
        if not episodes_data:
            self._log_error("No complete episodes remained after subtask stitching")
            return False
        self._collect_episode_frame_reuse_reports(episodes_data)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

        self._features = {}
        self._tasks = {}
        self._task_to_index = {}
        self._total_episodes = 0
        self._total_frames = 0
        self._episode_metadata_list = []
        self._current_data_chunk_idx = 0
        self._current_data_file_idx = 0
        self._current_data_file_size_mb = 0.0
        self._current_data_file_frames = 0
        self._current_data_aggregate_cache_key = None
        self._video_tracking = {}
        self._pending_parquet_data = []
        self._pending_video_files = {}
        self._direct_video_stats_cache = {}

        self._collect_tasks(episodes_data)
        self._collect_task_names(episodes_data)

        self._write_aggregated_data(episodes_data)

        # Phase 4: Write aggregated video files
        self._write_aggregated_videos(episodes_data)

        # Feature metadata needs video dimensions. Build it after video
        # aggregation so direct/cache paths can seed dimensions without
        # reopening MP4 files during cached conversions.
        self._build_features(episodes_data)

        # Phase 5: Write episodes metadata (Parquet)
        self._write_episodes_parquet(episodes_data)

        # Phase 6: Write tasks (Parquet)
        self._write_tasks_parquet(episodes_data)

        # Phase 7: Write optional subtask metadata/annotations
        self._write_subtasks_parquet(output_dir, episodes_data)
        self._write_subtask_annotations(output_dir, episodes_data)

        # Phase 8: Write global stats
        self._write_global_stats()

        # Phase 9: Write info.json
        self._write_info_json_v30()
        # Root info.json (conversion config snapshot) — same writer as v2.1.
        self._write_root_info_json()
        self._write_frame_reuse_metadata(output_dir)

        # Phase 10: Write quality reports (optional)
        if self.config.enable_quality_report and self._quality_reports:
            self._write_quality_reports(output_dir)

        self._log_info(f"Successfully converted {len(episodes_data)} episodes to v3.0")
        return True

    @staticmethod
    def _array_cache_signature(values: Sequence[Any]) -> Dict[str, Any]:
        if not values:
            return {"dtype": "", "shape": [0], "sha256": ""}
        array = np.ascontiguousarray(np.asarray(values))
        return {
            "dtype": str(array.dtype),
            "shape": [int(value) for value in array.shape],
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    def _episode_data_cache_signature(self, episode: EpisodeData) -> Dict[str, Any]:
        video_files = {}
        for camera_name, video_path in sorted(episode.video_files.items()):
            try:
                video_files[camera_name] = self._file_signature(Path(video_path))
            except OSError:
                video_files[camera_name] = {
                    "path": _fast_absolute_path(Path(video_path)),
                    "missing": True,
                }
        prepared_cache_signature = getattr(
            episode,
            "_cyclo_prepared_cache_signature",
            None,
        )
        if isinstance(prepared_cache_signature, dict):
            return {
                "episode_index": int(episode.episode_index),
                "length": int(episode.length),
                "tasks": list(episode.tasks),
                "recording_mode": episode.recording_mode,
                "full_episode_index": episode.full_episode_index,
                "subtask_instructions": list(episode.subtask_instructions),
                "episode_success": episode.episode_success,
                "prepared_cache": prepared_cache_signature,
                "video_files": video_files,
            }
        return {
            "episode_index": int(episode.episode_index),
            "length": int(episode.length),
            "tasks": list(episode.tasks),
            "recording_mode": episode.recording_mode,
            "full_episode_index": episode.full_episode_index,
            "subtask_instructions": list(episode.subtask_instructions),
            "episode_success": episode.episode_success,
            "timestamps": self._array_cache_signature(episode.timestamps),
            "observation_state": self._array_cache_signature(
                episode.observation_state
            ),
            "action": self._array_cache_signature(episode.action),
            "subtask_indices": self._array_cache_signature(episode.subtask_indices),
            "video_files": video_files,
        }

    def _data_aggregate_cache_key(
        self,
        episodes_data: List[EpisodeData],
        *,
        has_subtask_feature: bool,
        has_episode_success_feature: bool,
    ) -> Dict[str, Any]:
        return {
            "version": _V30_DATA_AGGREGATE_CACHE_VERSION,
            "codebase_version": CODEBASE_VERSION_V30,
            "fps": int(self.config.fps),
            "data_file_size_in_mb": int(self.config.data_file_size_in_mb),
            "direct_video_aggregation": bool(self._direct_video_aggregation),
            "video_stats_sample_budget": int(self._video_stats_sample_budget()),
            "parquet_write_kwargs": _v30_parquet_write_kwargs(),
            "has_subtask_feature": bool(has_subtask_feature),
            "has_episode_success_feature": bool(
                has_episode_success_feature
            ),
            "tasks": {
                str(idx): task for idx, task in sorted(self._tasks.items())
            },
            "task_to_index": dict(sorted(self._task_to_index.items())),
            "episodes": [
                self._episode_data_cache_signature(episode)
                for episode in episodes_data
            ],
        }

    def _episode_source_cache_root(self, source_path: Path) -> Path:
        path = Path(_fast_absolute_path(Path(source_path)))
        episode_dir = path if path.is_dir() else path.parent
        cached_episode_root = self._direct_source_episode_root_cache.get(episode_dir)
        if cached_episode_root is not None:
            return cached_episode_root
        if not (episode_dir / "episode_info.json").exists():
            self._direct_source_episode_root_cache[episode_dir] = episode_dir
            return episode_dir
        dataset_dir = episode_dir.parent
        has_sibling_episode = self._direct_source_dataset_sibling_cache.get(
            dataset_dir
        )
        if has_sibling_episode is None:
            try:
                has_sibling_episode = False
                for child in dataset_dir.iterdir():
                    if child == episode_dir or not child.is_dir():
                        continue
                    if (child / "episode_info.json").exists():
                        has_sibling_episode = True
                        break
            except OSError:
                has_sibling_episode = False
            self._direct_source_dataset_sibling_cache[dataset_dir] = has_sibling_episode
        root = dataset_dir if has_sibling_episode else episode_dir
        self._direct_source_episode_root_cache[episode_dir] = root
        return root

    def _data_aggregate_cache_dir(
        self,
        episodes_data: List[EpisodeData],
    ) -> Optional[Path]:
        if os.environ.get(_V30_DATA_AGG_CACHE_DISABLE_ENV):
            return None
        roots: List[str] = []
        try:
            for episode in episodes_data:
                if episode.source_path is None:
                    return None
                roots.append(str(self._episode_source_cache_root(episode.source_path)))
            if not roots:
                return None
            return (
                Path(os.path.commonpath(roots))
                / ".cyclo_cache"
                / "data_aggregate_v30"
            )
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"v3 data aggregate cache disabled ({exc!r})")
            return None

    @staticmethod
    def _data_aggregate_cache_digest(cache_key: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _data_aggregate_cache_path(
        self,
        episodes_data: List[EpisodeData],
        cache_key: Dict[str, Any],
    ) -> Optional[Path]:
        cache_dir = self._data_aggregate_cache_dir(episodes_data)
        if cache_dir is None:
            return None
        return cache_dir / self._data_aggregate_cache_digest(cache_key)

    def _restore_data_aggregate_metadata(
        self,
        manifest: Dict[str, Any],
    ) -> bool:
        metadata_items = manifest.get("episode_metadata")
        if not isinstance(metadata_items, list):
            return False
        try:
            self._episode_metadata_list = [
                EpisodeMetadata.from_cache_dict(item)
                for item in metadata_items
                if isinstance(item, dict)
            ]
            if len(self._episode_metadata_list) != len(metadata_items):
                return False
            self._total_episodes = int(manifest.get("total_episodes", 0))
            self._total_frames = int(manifest.get("total_frames", 0))
            self._episode_metadata_by_index = {
                int(ep.episode_index): ep for ep in self._episode_metadata_list
            }
            return True
        except Exception:
            return False

    def _try_reuse_data_aggregate_cache(
        self,
        output_dir: Path,
        episodes_data: List[EpisodeData],
        cache_key: Dict[str, Any],
    ) -> bool:
        cache_path = self._data_aggregate_cache_path(episodes_data, cache_key)
        if cache_path is None:
            return False
        manifest_path = cache_path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if manifest.get("cache_key") != cache_key:
            return False
        data_files = manifest.get("data_files")
        if not isinstance(data_files, list):
            return False
        try:
            for rel_path in data_files:
                if not isinstance(rel_path, str):
                    return False
                src = cache_path / rel_path
                dst = output_dir / rel_path
                if not src.exists() or src.stat().st_size <= 0:
                    return False
                self._clone_or_copy_no_hardlink(src, dst)
            if not self._restore_data_aggregate_metadata(manifest):
                return False
            self._log_info(
                f"Reused v3 data aggregate cache: {len(data_files)} file(s)"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"v3 data aggregate cache reuse failed ({exc!r}); regenerating"
            )
            return False

    def _store_data_aggregate_cache(
        self,
        output_dir: Path,
        episodes_data: List[EpisodeData],
        cache_key: Dict[str, Any],
        written_data_files: List[Path],
    ) -> None:
        cache_path = self._data_aggregate_cache_path(episodes_data, cache_key)
        if cache_path is None or not written_data_files:
            return
        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.tmp"
        )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            data_files: List[str] = []
            tmp_dir.mkdir(parents=True, exist_ok=False)
            for path in written_data_files:
                rel_path = Path(path).relative_to(output_dir).as_posix()
                dst = tmp_dir / rel_path
                self._clone_or_copy_no_hardlink(Path(path), dst)
                data_files.append(rel_path)
            manifest = {
                "version": _V30_DATA_AGGREGATE_CACHE_VERSION,
                "cache_key": cache_key,
                "data_files": data_files,
                "episode_metadata": [
                    metadata.to_cache_dict()
                    for metadata in self._episode_metadata_list
                ],
                "total_episodes": int(self._total_episodes),
                "total_frames": int(self._total_frames),
            }
            manifest_path_tmp = tmp_dir / "manifest.json"
            manifest_path_tmp.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_dir, cache_path)
            self._log_info(
                f"Stored v3 data aggregate cache: {len(data_files)} file(s)"
            )
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._log_warning(
                f"failed to store v3 data aggregate cache ({exc!r})"
            )

    def _episodes_parquet_cache_key(self) -> Optional[Dict[str, Any]]:
        data_key = self._current_data_aggregate_cache_key
        if not data_key:
            return None
        return {
            "version": _V30_EPISODES_PARQUET_CACHE_VERSION,
            "data_key_digest": self._data_aggregate_cache_digest(data_key),
            "video_file_size_in_mb": int(self.config.video_file_size_in_mb),
            "parquet_write_kwargs": _v30_parquet_write_kwargs(),
            "episode_metadata": [
                metadata.to_cache_dict()
                for metadata in self._episode_metadata_list
            ],
        }

    def _episodes_parquet_cache_path(
        self,
        episodes_data: Optional[List[EpisodeData]],
        cache_key: Dict[str, Any],
    ) -> Optional[Path]:
        if not episodes_data:
            return None
        data_cache_dir = self._data_aggregate_cache_dir(episodes_data)
        if data_cache_dir is None:
            return None
        digest = hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return data_cache_dir.parent / "episodes_parquet_v30" / digest

    def _try_reuse_episodes_parquet_cache(
        self,
        episodes_data: Optional[List[EpisodeData]],
        cache_key: Dict[str, Any],
        file_path: Path,
    ) -> bool:
        cache_path = self._episodes_parquet_cache_path(episodes_data, cache_key)
        if cache_path is None:
            return False
        manifest_path = cache_path / "manifest.json"
        parquet_path = cache_path / "episodes.parquet"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if manifest.get("cache_key") != cache_key:
            return False
        try:
            if not parquet_path.exists() or parquet_path.stat().st_size <= 0:
                return False
            self._clone_or_copy_no_hardlink(parquet_path, file_path)
            self._log_info("Reused v3 episodes metadata parquet cache")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"v3 episodes parquet cache reuse failed ({exc!r}); regenerating"
            )
            return False

    def _store_episodes_parquet_cache(
        self,
        episodes_data: Optional[List[EpisodeData]],
        cache_key: Dict[str, Any],
        file_path: Path,
    ) -> None:
        cache_path = self._episodes_parquet_cache_path(episodes_data, cache_key)
        if cache_path is None or not file_path.exists():
            return
        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.tmp"
        )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_dir.mkdir(parents=True, exist_ok=False)
            self._clone_or_copy_no_hardlink(file_path, tmp_dir / "episodes.parquet")
            (tmp_dir / "manifest.json").write_text(
                json.dumps({"cache_key": cache_key}, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_dir, cache_path)
            self._log_info("Stored v3 episodes metadata parquet cache")
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._log_warning(
                f"failed to store v3 episodes parquet cache ({exc!r})"
            )

    def _small_parquet_cache_path(
        self,
        episodes_data: Optional[List[EpisodeData]],
        cache_key: Dict[str, Any],
        cache_name: str,
    ) -> Optional[Path]:
        if not episodes_data:
            return None
        data_cache_dir = self._data_aggregate_cache_dir(episodes_data)
        if data_cache_dir is None:
            return None
        digest = hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return data_cache_dir.parent / cache_name / digest

    def _tasks_parquet_cache_key(self) -> Optional[Dict[str, Any]]:
        data_key = self._current_data_aggregate_cache_key
        if not data_key:
            return None
        task_names = getattr(self, "_task_names_by_task", {})
        return {
            "version": _V30_TASKS_PARQUET_CACHE_VERSION,
            "data_key_digest": self._data_aggregate_cache_digest(data_key),
            "parquet_write_kwargs": _v30_parquet_write_kwargs(),
            "tasks": {
                str(idx): task for idx, task in sorted(self._tasks.items())
            },
            "task_names": dict(sorted(task_names.items())),
        }

    def _subtasks_parquet_cache_key(
        self,
        rows: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        data_key = self._current_data_aggregate_cache_key
        if not data_key:
            return None
        return {
            "version": _V30_SUBTASKS_PARQUET_CACHE_VERSION,
            "data_key_digest": self._data_aggregate_cache_digest(data_key),
            "parquet_write_kwargs": _v30_parquet_write_kwargs(),
            "rows": rows,
        }

    def _try_reuse_small_parquet_cache(
        self,
        episodes_data: Optional[List[EpisodeData]],
        cache_key: Dict[str, Any],
        file_path: Path,
        *,
        cache_name: str,
        artifact_name: str,
    ) -> bool:
        cache_path = self._small_parquet_cache_path(
            episodes_data,
            cache_key,
            cache_name,
        )
        if cache_path is None:
            return False
        manifest_path = cache_path / "manifest.json"
        parquet_path = cache_path / artifact_name
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if manifest.get("cache_key") != cache_key:
            return False
        try:
            if not parquet_path.exists() or parquet_path.stat().st_size <= 0:
                return False
            self._clone_or_copy_no_hardlink(parquet_path, file_path)
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{cache_name} reuse failed ({exc!r}); regenerating"
            )
            return False

    def _store_small_parquet_cache(
        self,
        episodes_data: Optional[List[EpisodeData]],
        cache_key: Dict[str, Any],
        file_path: Path,
        *,
        cache_name: str,
        artifact_name: str,
    ) -> None:
        cache_path = self._small_parquet_cache_path(
            episodes_data,
            cache_key,
            cache_name,
        )
        if cache_path is None or not file_path.exists():
            return
        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.tmp"
        )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_dir.mkdir(parents=True, exist_ok=False)
            self._clone_or_copy_no_hardlink(file_path, tmp_dir / artifact_name)
            (tmp_dir / "manifest.json").write_text(
                json.dumps({"cache_key": cache_key}, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_dir, cache_path)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._log_warning(f"failed to store {cache_name} ({exc!r})")

    def _write_aggregated_data(self, episodes_data: List[EpisodeData]):
        """Write episode data to aggregated Parquet files."""
        self._log_info("Writing aggregated data files...")

        output_dir = Path(self.config.output_dir)
        has_subtask_feature = any(ep.subtask_indices for ep in episodes_data)
        has_episode_success_feature = self._has_episode_success_feature(
            episodes_data
        )
        cache_key = self._data_aggregate_cache_key(
            episodes_data,
            has_subtask_feature=has_subtask_feature,
            has_episode_success_feature=has_episode_success_feature,
        )
        self._current_data_aggregate_cache_key = cache_key
        if self._try_reuse_data_aggregate_cache(
            output_dir,
            episodes_data,
            cache_key,
        ):
            return

        pending_episodes: List[EpisodeData] = []
        pending_frame_count = 0
        pending_global_start_index = 0
        pending_size_mb = 0.0
        global_frame_index = 0
        written_data_files: List[Path] = []

        for episode in episodes_data:
            ep_idx = episode.episode_index
            num_frames = episode.length

            # Track where this episode starts in the current file
            dataset_from_index = pending_frame_count
            dataset_to_index = pending_frame_count + num_frames

            # Create episode metadata
            ep_stats = self._compute_episode_stats(episode)
            ep_metadata = EpisodeMetadata(
                episode_index=ep_idx,
                length=num_frames,
                tasks=episode.tasks,
                recording_mode=episode.recording_mode,
                full_episode_index=episode.full_episode_index,
                subtask_instructions=list(episode.subtask_instructions),
                data_chunk_index=self._current_data_chunk_idx,
                data_file_index=self._current_data_file_idx,
                dataset_from_index=dataset_from_index,
                dataset_to_index=dataset_to_index,
                stats=self._flatten_stats(ep_stats),
            )
            self._episode_metadata_list.append(ep_metadata)

            pending_episodes.append(episode)
            pending_frame_count += num_frames
            global_frame_index += num_frames

            # Estimate size (rough approximation)
            pending_size_mb = pending_frame_count * 0.001  # ~1KB per frame estimate

            # Check if we need to flush
            if pending_size_mb >= self.config.data_file_size_in_mb:
                data_path = self._flush_episode_data_file(
                    output_dir,
                    pending_episodes,
                    pending_global_start_index,
                    pending_frame_count,
                    has_subtask_feature,
                    has_episode_success_feature,
                )
                if data_path is not None:
                    written_data_files.append(data_path)
                pending_episodes = []
                pending_frame_count = 0
                pending_global_start_index = global_frame_index
                pending_size_mb = 0.0
                self._advance_chunk_file_index("data")

        if pending_episodes:
            data_path = self._flush_episode_data_file(
                output_dir,
                pending_episodes,
                pending_global_start_index,
                pending_frame_count,
                has_subtask_feature,
                has_episode_success_feature,
            )
            if data_path is not None:
                written_data_files.append(data_path)

        self._total_episodes = len(episodes_data)
        self._total_frames = global_frame_index
        self._episode_metadata_by_index = {
            int(ep.episode_index): ep for ep in self._episode_metadata_list
        }
        self._store_data_aggregate_cache(
            output_dir,
            episodes_data,
            cache_key,
            written_data_files,
        )

    def _flush_episode_data_file(
        self,
        output_dir: Path,
        episodes: List[EpisodeData],
        global_start_index: int,
        num_frames: int,
        has_subtask_feature: bool,
        has_episode_success_feature: bool,
    ) -> Optional[Path]:
        """Write an aggregated data Parquet file directly from episode columns."""
        if not episodes or num_frames <= 0:
            return None

        chunk_idx = self._current_data_chunk_idx
        file_idx = self._current_data_file_idx
        file_path = output_dir / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        first_state = next(
            (
                episode.observation_state[0]
                for episode in episodes
                if episode.observation_state
            ),
            None,
        )
        first_action = next(
            (episode.action[0] for episode in episodes if episode.action),
            None,
        )
        state_dim = len(first_state) if first_state is not None else 0
        action_dim = len(first_action) if first_action is not None else 0

        schema_fields = [
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
        if has_episode_success_feature:
            schema_fields.append(pa.field("episode_success", pa.bool_()))
        if has_subtask_feature:
            schema_fields.append(pa.field("subtask_index", pa.int64()))
        if state_dim > 0:
            schema_fields.append(
                pa.field("observation.state", pa.list_(pa.float32(), state_dim))
            )
        if action_dim > 0:
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), action_dim)))

        timestamps = np.empty(num_frames, dtype=np.float32)
        frame_index = np.empty(num_frames, dtype=np.int64)
        episode_index = np.empty(num_frames, dtype=np.int64)
        task_index = np.empty(num_frames, dtype=np.int64)
        episode_success = (
            np.empty(num_frames, dtype=np.bool_)
            if has_episode_success_feature else None
        )
        subtask_index = (
            np.empty(num_frames, dtype=np.int64) if has_subtask_feature else None
        )
        state_values = (
            np.empty((num_frames, state_dim), dtype=np.float32)
            if state_dim > 0 else None
        )
        action_values = (
            np.empty((num_frames, action_dim), dtype=np.float32)
            if action_dim > 0 else None
        )

        offset = 0
        for episode in episodes:
            length = int(episode.length)
            end = offset + length
            timestamps[offset:end] = np.asarray(
                episode.timestamps[:length],
                dtype=np.float32,
            )
            frame_index[offset:end] = np.arange(length, dtype=np.int64)
            episode_index[offset:end] = int(episode.episode_index)
            task = episode.tasks[0] if episode.tasks else "default_task"
            task_index[offset:end] = self._task_to_index.get(task, 0)
            if episode_success is not None:
                episode_success[offset:end] = bool(episode.episode_success)
            if subtask_index is not None:
                if len(episode.subtask_indices) == length:
                    subtask_index[offset:end] = np.asarray(
                        episode.subtask_indices,
                        dtype=np.int64,
                    )
                else:
                    subtask_index[offset:end] = 0
            if state_values is not None:
                state_values[offset:end] = np.asarray(
                    episode.observation_state[:length],
                    dtype=np.float32,
                )
            if action_values is not None:
                action_values[offset:end] = np.asarray(
                    episode.action[:length],
                    dtype=np.float32,
                )
            offset = end

        arrays = [
            pa.array(timestamps, type=pa.float32()),
            pa.array(frame_index, type=pa.int64()),
            pa.array(episode_index, type=pa.int64()),
            pa.array(
                np.arange(
                    global_start_index,
                    global_start_index + num_frames,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
            pa.array(task_index, type=pa.int64()),
        ]
        if episode_success is not None:
            arrays.append(pa.array(episode_success, type=pa.bool_()))
        if subtask_index is not None:
            arrays.append(pa.array(subtask_index, type=pa.int64()))
        if state_values is not None:
            state_flat = pa.array(state_values.reshape(-1), type=pa.float32())
            arrays.append(pa.FixedSizeListArray.from_arrays(state_flat, state_dim))
        if action_values is not None:
            action_flat = pa.array(action_values.reshape(-1), type=pa.float32())
            arrays.append(pa.FixedSizeListArray.from_arrays(action_flat, action_dim))

        hf_metadata = self._data_file_hf_metadata(
            has_subtask_feature,
            has_episode_success_feature,
            state_dim,
            action_dim,
        )
        schema = pa.schema(schema_fields).with_metadata(
            {"huggingface": hf_metadata}
        )
        table = pa.table(
            dict(zip([field.name for field in schema_fields], arrays)),
            schema=schema,
        )
        pq.write_table(table, file_path, **_v30_parquet_write_kwargs())
        self._log_info(f"Wrote data file: {file_path.name} ({num_frames} frames)")
        return file_path

    @staticmethod
    def _data_file_hf_metadata(
        has_subtask_feature: bool,
        has_episode_success_feature: bool,
        state_dim: int,
        action_dim: int,
    ) -> str:
        hf_features = {
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }
        if has_episode_success_feature:
            hf_features["episode_success"] = {
                "dtype": "bool",
                "_type": "Value",
            }
        if has_subtask_feature:
            hf_features["subtask_index"] = {"dtype": "int64", "_type": "Value"}
        if state_dim > 0:
            hf_features["observation.state"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": state_dim,
                "_type": "Sequence",
            }
        if action_dim > 0:
            hf_features["action"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": action_dim,
                "_type": "Sequence",
            }
        return json.dumps({"info": {"features": hf_features}})

    def _flush_data_file(self, output_dir: Path, frames: List[Dict[str, Any]]):
        """Write accumulated frames to a Parquet file with HuggingFace-compatible schema."""
        if not frames:
            return

        chunk_idx = self._current_data_chunk_idx
        file_idx = self._current_data_file_idx

        file_path = output_dir / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        first_state = frames[0].get("observation.state")
        first_action = frames[0].get("action")
        state_dim = len(first_state) if first_state is not None else 0
        action_dim = len(first_action) if first_action is not None else 0

        schema_fields = [
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
        has_subtask_feature = any("subtask_index" in frame for frame in frames)
        if has_subtask_feature:
            schema_fields.append(pa.field("subtask_index", pa.int64()))

        if state_dim > 0:
            schema_fields.append(
                pa.field("observation.state", pa.list_(pa.float32(), state_dim))
            )
        if action_dim > 0:
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), action_dim)))

        schema = pa.schema(schema_fields)

        num_frames = len(frames)
        arrays = [
            pa.array(
                np.fromiter(
                    (float(f["timestamp"]) for f in frames),
                    dtype=np.float32,
                    count=num_frames,
                ),
                type=pa.float32(),
            ),
            pa.array(
                np.fromiter(
                    (int(f["frame_index"]) for f in frames),
                    dtype=np.int64,
                    count=num_frames,
                ),
                type=pa.int64(),
            ),
            pa.array(
                np.fromiter(
                    (int(f["episode_index"]) for f in frames),
                    dtype=np.int64,
                    count=num_frames,
                ),
                type=pa.int64(),
            ),
            pa.array(
                np.fromiter(
                    (int(f["index"]) for f in frames),
                    dtype=np.int64,
                    count=num_frames,
                ),
                type=pa.int64(),
            ),
            pa.array(
                np.fromiter(
                    (int(f["task_index"]) for f in frames),
                    dtype=np.int64,
                    count=num_frames,
                ),
                type=pa.int64(),
            ),
        ]
        if has_subtask_feature:
            arrays.append(
                pa.array(
                    np.fromiter(
                        (int(f.get("subtask_index", 0)) for f in frames),
                        dtype=np.int64,
                        count=num_frames,
                    ),
                    type=pa.int64(),
                )
            )

        if state_dim > 0:
            state_values = np.asarray(
                [f["observation.state"] for f in frames],
                dtype=np.float32,
            )
            state_flat = pa.array(
                state_values.reshape(-1),
                type=pa.float32(),
            )
            arrays.append(
                pa.FixedSizeListArray.from_arrays(state_flat, state_dim)
            )

        if action_dim > 0:
            action_values = np.asarray(
                [f["action"] for f in frames],
                dtype=np.float32,
            )
            action_flat = pa.array(
                action_values.reshape(-1),
                type=pa.float32(),
            )
            arrays.append(
                pa.FixedSizeListArray.from_arrays(action_flat, action_dim)
            )

        hf_features = {
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }
        if has_subtask_feature:
            hf_features["subtask_index"] = {"dtype": "int64", "_type": "Value"}

        if state_dim > 0:
            hf_features["observation.state"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": state_dim,
                "_type": "Sequence",
            }
        if action_dim > 0:
            hf_features["action"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": action_dim,
                "_type": "Sequence",
            }

        hf_metadata = json.dumps({"info": {"features": hf_features}})
        schema = schema.with_metadata({"huggingface": hf_metadata})

        table = pa.table(
            dict(zip([f.name for f in schema_fields], arrays)), schema=schema
        )
        pq.write_table(table, file_path, **_v30_parquet_write_kwargs())

        self._log_info(f"Wrote data file: {file_path.name} ({num_frames} frames)")

    def _write_aggregated_videos(self, episodes_data: List[EpisodeData]):
        """Write aggregated video files by concatenating episode videos."""
        self._log_info("Writing aggregated video files...")

        output_dir = Path(self.config.output_dir)
        episode_by_index = {
            int(episode.episode_index): episode for episode in episodes_data
        }

        # Group videos by camera
        camera_videos: Dict[
            str, List[Tuple[int, Path, float]]
        ] = {}  # camera -> [(ep_idx, path, duration)]

        for episode in episodes_data:
            for camera_name, video_path in episode.video_files.items():
                self._record_frame_reuse_for_video(
                    episode,
                    camera_name,
                    Path(video_path),
                )
                if camera_name not in camera_videos:
                    camera_videos[camera_name] = []

                camera_videos[camera_name].append(
                    (episode.episode_index, video_path, 0.0)
                )

        jobs: List[_VideoAggregateJob] = []
        for camera_name, videos in camera_videos.items():
            camera_key = self._video_feature_key(camera_name)
            for chunk_idx, file_idx, pending in self._plan_video_batches(videos):
                jobs.append((camera_key, chunk_idx, file_idx, pending))

        def aggregate_job_weight(
            job: _VideoAggregateJob,
        ) -> Tuple[int, int]:
            source_bytes = 0
            for _, video_path, _ in job[3]:
                try:
                    source_bytes += Path(video_path).stat().st_size
                except OSError:
                    pass
            output_frames = sum(
                int(episode_by_index[int(ep_idx)].length)
                for ep_idx, _, _ in job[3]
            )
            return source_bytes, output_frames

        source_cache_matches = self._precompute_direct_source_cache_matches(
            jobs,
            episode_by_index,
        )
        all_source_cache_hits = (
            self._direct_video_aggregation
            and bool(jobs)
            and len(source_cache_matches) == len(jobs)
        )
        if all_source_cache_hits:
            execution_jobs = list(jobs)
        else:
            execution_jobs = sorted(jobs, key=aggregate_job_weight, reverse=True)
        if (
            self._direct_video_aggregation
            and not self._can_reuse_direct_cache_without_encoder_probe()
        ):
            self._warm_direct_video_encoder(execution_jobs)
        workers = self._resolve_video_aggregation_workers_for_jobs(
            len(camera_videos),
            execution_jobs,
            source_cache_matches,
        )
        if workers > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            self._log_info(
                f"Writing {len(camera_videos)} camera video aggregates "
                f"with {workers} workers"
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for camera_key, chunk_idx, file_idx, pending in execution_jobs:
                    if self._direct_video_aggregation:
                        futures.append(
                            executor.submit(
                                self._write_direct_aggregated_video,
                                output_dir,
                                camera_key,
                                chunk_idx,
                                file_idx,
                                pending,
                                episode_by_index,
                                source_cache_matches.get(
                                    (camera_key, chunk_idx, file_idx)
                                ),
                            )
                        )
                    else:
                        futures.append(
                            executor.submit(
                                self._concatenate_videos,
                                output_dir,
                                camera_key,
                                chunk_idx,
                                file_idx,
                                pending,
                            )
                        )
                for future in as_completed(futures):
                    future.result()
        else:
            for camera_key, chunk_idx, file_idx, pending in execution_jobs:
                if self._direct_video_aggregation:
                    self._write_direct_aggregated_video(
                        output_dir,
                        camera_key,
                        chunk_idx,
                        file_idx,
                        pending,
                        episode_by_index,
                        source_cache_matches.get(
                            (camera_key, chunk_idx, file_idx)
                        ),
                    )
                else:
                    self._concatenate_videos(
                        output_dir,
                        camera_key,
                        chunk_idx,
                        file_idx,
                        pending,
                    )

        if self._direct_video_aggregation:
            self._merge_direct_video_stats_into_metadata()

        for camera_key, chunk_idx, file_idx, pending in jobs:
            self._update_video_metadata(camera_key, chunk_idx, file_idx, pending)

        # ``<cam>_synced.mp4`` files are kept on disk as Phase 1 cache —
        # the next conversion run hits the ``<cam>_synced.cache.json``
        # gate in ``_sync_videos_to_grid`` and skips remux entirely.
        # Disk cost is modest (~2-3 MB per camera per episode);
        # operators who want to reclaim the space can wipe
        # ``<episode>/videos/*_synced.*`` after the dataset is final.

    def _precompute_direct_source_cache_matches(
        self,
        jobs: Sequence[_VideoAggregateJob],
        episode_by_index: Dict[int, EpisodeData],
    ) -> Dict[Tuple[str, int, int], _DirectSourceCacheMatch]:
        """Return source-cache hits for direct aggregate jobs."""
        if (
            not self._direct_video_aggregation
            or not jobs
            or not self._can_reuse_direct_cache_without_encoder_probe()
        ):
            return {}
        cache_dir_exists = False
        for job in jobs:
            if not job[3]:
                continue
            cache_dir = self._direct_source_aggregate_cache_dir(job[3])
            if cache_dir is not None and cache_dir.exists():
                cache_dir_exists = True
                break
        if not cache_dir_exists:
            return {}

        matches: Dict[Tuple[str, int, int], _DirectSourceCacheMatch] = {}
        file_signature_cache: Dict[Path, Dict[str, Any]] = {}
        grid_hash_cache: Dict[int, str] = {}
        for camera_key, chunk_idx, file_idx, pending in jobs:
            if not pending:
                continue
            camera_name = self._camera_name_from_feature_key(camera_key)
            expected_frames = int(
                sum(episode_by_index[int(ep_idx)].length for ep_idx, _, _ in pending)
            )
            content_key = self._direct_aggregated_video_content_cache_key(
                pending,
                expected_frames,
                episode_by_index,
                camera_name,
                file_signature_cache=file_signature_cache,
                grid_hash_cache=grid_hash_cache,
            )
            match = self._find_direct_source_aggregate_cache_by_content(
                pending,
                content_key,
            )
            if match is not None:
                matches[(camera_key, chunk_idx, file_idx)] = match
        return matches

    def _resolve_video_aggregation_workers_for_jobs(
        self,
        camera_count: int,
        jobs: Sequence[_VideoAggregateJob],
        source_cache_matches: Dict[Tuple[str, int, int], _DirectSourceCacheMatch],
    ) -> int:
        if os.environ.get(_VIDEO_AGG_CAMERA_WORKERS_ENV):
            return self._resolve_video_aggregation_workers(
                camera_count,
                len(jobs),
            )
        if (
            jobs
            and _v30_max_speed_profile()
            and self._direct_video_aggregation
            and len(source_cache_matches) == len(jobs)
        ):
            return 1
        return self._resolve_video_aggregation_workers(camera_count, len(jobs))

    def _warm_direct_video_encoder(
        self,
        jobs: Sequence[_VideoAggregateJob],
    ) -> None:
        """Populate the H.264 encoder probe cache before threaded video jobs."""
        for _, _, _, pending in jobs:
            if not pending:
                continue
            try:
                width, height = self._quick_video_dimensions_cached(
                    Path(pending[0][1])
                )
                if width > 0 and height > 0:
                    _h264_encoder(_ffmpeg(), width=width, height=height)
            except Exception:
                pass
            return

    @staticmethod
    def _direct_aggregate_encoder_opts(
        encoder: str,
        encoder_opts: Sequence[str],
    ) -> List[str]:
        """Return encoder opts for direct aggregate output.

        The max-speed libx264 profile defaults to all-intra (`-g 1`). That is
        useful for isolated clips in some fallback paths, but v3 direct
        aggregates are long, continuous files where a bounded inter GOP reduces
        encoder work and output size. Keep explicit user GOP settings intact.
        """
        if (
            str(encoder).strip().lower() != "libx264"
            or "CYCLO_X264_GOP" in os.environ
        ):
            return list(encoder_opts)
        cleaned: List[str] = []
        opts = list(encoder_opts)
        idx = 0
        while idx < len(opts):
            if opts[idx] == "-g" and idx + 1 < len(opts) and opts[idx + 1] == "1":
                idx += 2
                continue
            cleaned.append(opts[idx])
            idx += 1
        return cleaned

    def _plan_video_batches(
        self,
        videos: List[Tuple[int, Path, float]],
    ) -> List[Tuple[int, int, List[Tuple[int, Path, float]]]]:
        """Plan v3.0 aggregate video files for one camera."""
        chunk_idx = 0
        file_idx = 0
        current_size_mb = 0.0
        pending_videos: List[Tuple[int, Path, float]] = []
        batches: List[Tuple[int, int, List[Tuple[int, Path, float]]]] = []

        for ep_idx, video_path, duration in videos:
            video_size_mb = video_path.stat().st_size / (1024 * 1024)

            if (
                current_size_mb + video_size_mb >= self.config.video_file_size_in_mb
                and pending_videos
            ):
                batches.append((chunk_idx, file_idx, list(pending_videos)))

                chunk_idx, file_idx = self._update_chunk_file_indices(
                    chunk_idx, file_idx
                )
                pending_videos = []
                current_size_mb = 0.0

            pending_videos.append((ep_idx, video_path, duration))
            current_size_mb += video_size_mb

        if pending_videos:
            batches.append((chunk_idx, file_idx, list(pending_videos)))
        return batches

    @staticmethod
    def _resolve_video_aggregation_workers(
        camera_count: int,
        job_count: Optional[int] = None,
    ) -> int:
        max_jobs = max(1, int(job_count or camera_count))
        if camera_count <= 1 or max_jobs <= 1:
            return 1
        max_workers = max(1, max_jobs)
        raw = os.environ.get(_VIDEO_AGG_CAMERA_WORKERS_ENV)
        if raw:
            try:
                return max(1, min(int(raw), max_workers))
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 4
        if max_jobs > camera_count:
            if _v30_max_speed_profile():
                # Each aggregate worker owns an ffmpeg decoder+encoder pair.
                # Let chunked camera work overlap beyond one job per camera,
                # but still cap by visible CPUs below so edge systems do not
                # oversubscribe memory bandwidth.
                chunk_ceiling = camera_count * 2
            else:
                chunk_ceiling = 5
            default_workers = max(2, min(chunk_ceiling, cpu_count // 2))
        else:
            default_workers = max(1, min(4, cpu_count // 2))
        return max(1, min(default_workers, max_workers))

    @staticmethod
    def _trust_sidecar_frame_count() -> bool:
        if _V30_TRUST_SIDECAR_FRAME_COUNT_ENV in os.environ:
            raw = os.environ.get(_V30_TRUST_SIDECAR_FRAME_COUNT_ENV, "")
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
        return profile in {"max", "maximum", "max_speed", "fastest"}

    def _direct_aggregate_requires_decode_validation(self) -> bool:
        return not self._trust_sidecar_frame_count()

    def _direct_aggregate_requires_output_validation(self) -> bool:
        raw = os.environ.get(_V30_VALIDATE_DIRECT_AGGREGATE_ENV)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return not self._trust_sidecar_frame_count()

    @staticmethod
    def _camera_name_from_feature_key(camera_key: str) -> str:
        prefix = "observation.images.rgb."
        if camera_key.startswith(prefix):
            return camera_key[len(prefix):]
        return camera_key.rsplit(".", 1)[-1]

    def _grid_indices_for_raw_video(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> np.ndarray:
        indices, _ = self._grid_indices_and_source_count_for_raw_video(
            episode,
            camera_name,
            video_path,
        )
        return indices

    def _grid_indices_and_source_count_for_raw_video(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> Tuple[np.ndarray, int]:
        sidecar = Path(video_path).parent / f"{camera_name}_timestamps.parquet"
        from cyclo_data.reader.frame_timestamps import load_frame_timestamps

        ft = load_frame_timestamps(sidecar, camera_name)
        grid_ns = (
            np.asarray(
                episode.grid_log_times_sec[: int(episode.length)],
                dtype=np.float64,
            )
            * 1_000_000_000
        ).astype(np.int64)
        indices = ft.map_to_grid(grid_ns, time_source="header")
        self._record_frame_reuse_report(
            episode=episode,
            camera_name=camera_name,
            indices=indices,
            grid_ns=grid_ns,
            frame_timestamps=ft,
        )
        return indices, int(ft.num_frames)

    def _direct_aggregated_video_content_cache_key(
        self,
        videos: List[Tuple[int, Path, float]],
        expected_frames: int,
        episode_by_index: Dict[int, EpisodeData],
        camera_name: str,
        *,
        file_signature_cache: Optional[Dict[Path, Dict[str, Any]]] = None,
        grid_hash_cache: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        def file_signature(path: Path) -> Dict[str, Any]:
            path = Path(path)
            if file_signature_cache is None:
                return self._file_signature(path)
            cached = file_signature_cache.get(path)
            if cached is None:
                cached = self._file_signature(path)
                file_signature_cache[path] = cached
            return cached

        def grid_hash(episode: EpisodeData) -> str:
            ep_idx = int(episode.episode_index)
            if grid_hash_cache is not None:
                cached = grid_hash_cache.get(ep_idx)
                if cached is not None:
                    return cached
            grid = np.asarray(
                episode.grid_log_times_sec[: int(episode.length)],
                dtype=np.float64,
            )
            digest = hashlib.sha256(grid.tobytes()).hexdigest()
            if grid_hash_cache is not None:
                grid_hash_cache[ep_idx] = digest
            return digest

        inputs = []
        for ep_idx, video_path, _ in videos:
            episode = episode_by_index[int(ep_idx)]
            path = Path(video_path)
            sidecar = path.parent / f"{camera_name}_timestamps.parquet"
            self._record_frame_reuse_for_video(episode, camera_name, path)
            inputs.append({
                "episode_index": int(ep_idx),
                "video": file_signature(path),
                "sidecar": file_signature(sidecar),
                "grid_sha256": grid_hash(episode),
                "frames": int(episode.length),
            })
        cache_key = {
            "version": 1,
            "mode": "direct_v3_raw_sidecar",
            "target_fps": float(self.config.fps),
            "expected_frames": int(expected_frames),
            "inputs": inputs,
        }
        if self._trust_sidecar_frame_count():
            cache_key["source_frame_count_validation"] = "sidecar"
        return cache_key

    def _direct_aggregated_video_cache_key(
        self,
        videos: List[Tuple[int, Path, float]],
        expected_frames: int,
        episode_by_index: Dict[int, EpisodeData],
        camera_name: str,
        *,
        ffmpeg: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        encoder: Optional[str] = None,
        encoder_opts: Optional[Sequence[str]] = None,
        content_key: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if content_key is None:
            content_key = self._direct_aggregated_video_content_cache_key(
                videos,
                expected_frames,
                episode_by_index,
                camera_name,
            )
        if width is None or height is None:
            first_path = Path(videos[0][1]) if videos else Path()
            width, height = (
                self._quick_video_dimensions_cached(first_path)
                if videos else (0, 0)
            )
        if encoder is None or encoder_opts is None:
            ffmpeg = ffmpeg or _ffmpeg()
            encoder, encoder_opts = _h264_encoder(ffmpeg, width=width, height=height)
        cache_key = dict(content_key)
        cache_key["encoder"] = encoder
        cache_key["encoder_opts"] = list(encoder_opts)
        cache_key["output_width"] = int(width)
        cache_key["output_height"] = int(height)
        return cache_key

    @staticmethod
    def _can_reuse_direct_cache_without_encoder_probe() -> bool:
        profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
        if profile not in {"max", "maximum", "max_speed", "fastest"}:
            return False
        return not any(os.environ.get(name) for name in _H264_ENCODER_TUNING_ENVS)

    @staticmethod
    def _direct_cache_content_matches(
        cached_key: Dict[str, Any],
        content_key: Dict[str, Any],
    ) -> bool:
        if not isinstance(cached_key, dict):
            return False
        for key, value in content_key.items():
            if cached_key.get(key) != value:
                return False
        return True

    def _try_reuse_aggregated_video_cache_by_content(
        self,
        output_path: Path,
        cache_path: Path,
        content_key: Dict[str, Any],
        expected_frames: int,
        *,
        require_decode: bool,
        validate: bool = True,
    ) -> bool:
        if not output_path.exists() or output_path.stat().st_size <= 0:
            return False
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not self._direct_cache_content_matches(cached, content_key):
            return False
        try:
            if validate:
                self._validate_aggregated_video(
                    output_path,
                    expected_frames,
                    require_decode=require_decode,
                )
            self._remember_video_dimensions_from_cache_key(
                cached,
                output_path=output_path,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"cached direct aggregate validation failed for "
                f"{output_path.name}: {exc!r}; regenerating"
            )
            return False

    def _direct_source_aggregate_cache_dir(
        self,
        videos: List[Tuple[int, Path, float]],
    ) -> Optional[Path]:
        if os.environ.get(_V30_SOURCE_AGG_CACHE_DISABLE_ENV):
            return None
        if not videos:
            return None
        cache_key = tuple(
            _fast_absolute_path(Path(video_path))
            for _, video_path, _ in videos
        )
        if cache_key in self._direct_source_aggregate_cache_dir_cache:
            return self._direct_source_aggregate_cache_dir_cache[cache_key]
        try:
            roots = []
            for video_path in cache_key:
                path = Path(video_path)
                roots.append(str(self._direct_source_video_cache_root(path)))
            cache_dir = (
                Path(os.path.commonpath(roots))
                / ".cyclo_cache"
                / "direct_aggregate_v30"
            )
            self._direct_source_aggregate_cache_dir_cache[cache_key] = cache_dir
            return cache_dir
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"direct aggregate source cache disabled ({exc!r})")
            self._direct_source_aggregate_cache_dir_cache[cache_key] = None
            return None

    def _direct_source_video_cache_root(self, video_path: Path) -> Path:
        """Return the stable input root used for source aggregate cache files."""
        path = Path(_fast_absolute_path(Path(video_path)))
        cached = self._direct_source_cache_root_cache.get(path)
        if cached is not None:
            return cached
        videos_dir = path.parent.parent
        if videos_dir.name != "videos":
            root = path.parent
            self._direct_source_cache_root_cache[path] = root
            return root
        episode_dir = videos_dir.parent
        cached_episode_root = self._direct_source_episode_root_cache.get(episode_dir)
        if cached_episode_root is not None:
            self._direct_source_cache_root_cache[path] = cached_episode_root
            return cached_episode_root
        if not (episode_dir / "episode_info.json").exists():
            self._direct_source_cache_root_cache[path] = episode_dir
            self._direct_source_episode_root_cache[episode_dir] = episode_dir
            return episode_dir
        dataset_dir = episode_dir.parent
        has_sibling_episode = self._direct_source_dataset_sibling_cache.get(
            dataset_dir
        )
        if has_sibling_episode is not None:
            root = dataset_dir if has_sibling_episode else episode_dir
            self._direct_source_cache_root_cache[path] = root
            self._direct_source_episode_root_cache[episode_dir] = root
            return root
        try:
            has_sibling_episode = False
            for child in dataset_dir.iterdir():
                if child == episode_dir or not child.is_dir():
                    continue
                if (child / "episode_info.json").exists():
                    has_sibling_episode = True
                    break
        except OSError:
            has_sibling_episode = False
        self._direct_source_dataset_sibling_cache[dataset_dir] = has_sibling_episode
        root = dataset_dir if has_sibling_episode else episode_dir
        self._direct_source_cache_root_cache[path] = root
        self._direct_source_episode_root_cache[episode_dir] = root
        return root

    def _direct_source_aggregate_cache_paths(
        self,
        videos: List[Tuple[int, Path, float]],
        cache_key: Dict[str, Any],
    ) -> Optional[Tuple[Path, Path]]:
        cache_dir = self._direct_source_aggregate_cache_dir(videos)
        if cache_dir is None:
            return None
        try:
            digest = hashlib.sha256(
                json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            return (
                cache_dir / f"{digest}.mp4",
                cache_dir / f"{digest}.cache.json",
            )
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"direct aggregate source cache disabled ({exc!r})")
            return None

    def _direct_source_aggregate_content_index_path(
        self,
        videos: List[Tuple[int, Path, float]],
        content_key: Dict[str, Any],
        *,
        cache_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        if cache_dir is None:
            cache_dir = self._direct_source_aggregate_cache_dir(videos)
        if cache_dir is None:
            return None
        try:
            digest = hashlib.sha256(
                json.dumps(content_key, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            return cache_dir / f"content-{digest}.json"
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"direct aggregate source cache index disabled ({exc!r})")
            return None

    def _find_direct_source_aggregate_cache_by_content(
        self,
        videos: List[Tuple[int, Path, float]],
        content_key: Dict[str, Any],
    ) -> Optional[Tuple[Path, Path, Dict[str, Any]]]:
        cache_dir = self._direct_source_aggregate_cache_dir(videos)
        if cache_dir is None or not cache_dir.exists():
            return None

        index_path = self._direct_source_aggregate_content_index_path(
            videos,
            content_key,
            cache_dir=cache_dir,
        )
        if index_path is not None and index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if (
                    isinstance(index, dict)
                    and index.get("version") == 1
                    and index.get("content_key") == content_key
                ):
                    video_name = str(index.get("video") or "")
                    meta_name = str(index.get("meta") or "")
                    if (
                        video_name
                        and meta_name
                        and Path(video_name).name == video_name
                        and Path(meta_name).name == meta_name
                    ):
                        video_path = cache_dir / video_name
                        meta_path = cache_dir / meta_name
                        cached_key = json.loads(
                            meta_path.read_text(encoding="utf-8")
                        )
                        if (
                            video_path.exists()
                            and video_path.stat().st_size > 0
                            and self._direct_cache_content_matches(
                                cached_key,
                                content_key,
                            )
                        ):
                            return video_path, meta_path, cached_key
            except (OSError, ValueError):
                pass

        for meta_path in sorted(cache_dir.glob("*.cache.json")):
            try:
                cached_key = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not self._direct_cache_content_matches(cached_key, content_key):
                continue
            video_name = meta_path.name[: -len(".cache.json")] + ".mp4"
            video_path = meta_path.with_name(video_name)
            try:
                if video_path.exists() and video_path.stat().st_size > 0:
                    return video_path, meta_path, cached_key
            except OSError:
                continue
        return None

    def _store_direct_source_aggregate_content_index(
        self,
        videos: List[Tuple[int, Path, float]],
        content_key: Dict[str, Any],
        source_video_cache: Path,
        source_meta_cache: Path,
    ) -> None:
        index_path = self._direct_source_aggregate_content_index_path(
            videos,
            content_key,
            cache_dir=Path(source_meta_cache).parent,
        )
        if index_path is None:
            return
        payload = {
            "version": 1,
            "content_key": content_key,
            "video": Path(source_video_cache).name,
            "meta": Path(source_meta_cache).name,
        }
        tmp_path: Optional[Path] = None
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix=index_path.stem + ".",
                suffix=".tmp",
                dir=str(index_path.parent),
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                json.dump(payload, fh)
            os.replace(tmp_path, index_path)
        except Exception as exc:  # noqa: BLE001
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            self._log_warning(
                f"failed to write source aggregate content index "
                f"{index_path.name}: {exc!r}"
            )

    @staticmethod
    def _populate_direct_source_aggregate_cache() -> bool:
        raw = os.environ.get(_V30_SOURCE_AGG_CACHE_POPULATE_ENV, "")
        return raw.strip().lower() in {"1", "true", "yes", "on", "write"}

    @staticmethod
    def _write_output_cache_for_source_reuse() -> bool:
        raw = os.environ.get(_V30_WRITE_SOURCE_REUSE_OUTPUT_CACHE_ENV)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on", "write"}
        profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
        return profile not in {"max", "maximum", "max_speed", "fastest"}

    @staticmethod
    def _allow_direct_source_aggregate_cache_copy(encoder: str) -> bool:
        raw = os.environ.get(_V30_SOURCE_AGG_CACHE_POPULATE_ENV)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on", "write"}
        del encoder
        return False

    @staticmethod
    def _clone_or_copy_no_hardlink(
        src: Path,
        dst: Path,
        *,
        allow_copy: bool = True,
    ) -> Optional[str]:
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _same_file_or_same_path(src, dst):
            return "same_path"
        dev_pair: Optional[Tuple[int, int]] = None
        try:
            dev_pair = (src.stat().st_dev, dst.parent.stat().st_dev)
        except OSError:
            dev_pair = None
        if not allow_copy and dev_pair in _REFLINK_UNSUPPORTED_DEV_PAIRS:
            return None

        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        tmp.unlink(missing_ok=True)
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
            if not allow_copy:
                return None
            try:
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
            return "copy"

    def _try_reuse_direct_source_aggregate_cache(
        self,
        source_video_cache: Path,
        source_meta_cache: Path,
        output_path: Path,
        output_cache_path: Path,
        cache_key: Dict[str, Any],
        expected_frames: int,
        *,
        require_decode: bool,
        validate: bool = True,
    ) -> bool:
        if not source_video_cache.exists() or source_video_cache.stat().st_size <= 0:
            return False
        try:
            cached = json.loads(source_meta_cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if cached != cache_key:
            return False
        try:
            if validate:
                self._validate_aggregated_video(
                    source_video_cache,
                    expected_frames,
                    require_decode=require_decode,
                )
            copy_mode = self._clone_or_copy_no_hardlink(
                source_video_cache,
                output_path,
            )
            if self._write_output_cache_for_source_reuse():
                self._write_aggregated_video_cache(output_cache_path, cache_key)
            self._remember_video_dimensions_from_cache_key(
                cache_key,
                output_path=output_path,
            )
            self._log_info(
                f"Reused source cached direct aggregate: "
                f"{output_path.name} ({copy_mode})"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            output_path.unlink(missing_ok=True)
            self._log_warning(
                f"source aggregate cache validation/reuse failed for "
                f"{source_video_cache.name}: {exc!r}; regenerating"
            )
            return False

    def _store_direct_source_aggregate_cache(
        self,
        output_path: Path,
        source_video_cache: Path,
        source_meta_cache: Path,
        cache_key: Dict[str, Any],
        *,
        videos: Optional[List[Tuple[int, Path, float]]] = None,
        content_key: Optional[Dict[str, Any]] = None,
        allow_copy: bool = True,
    ) -> None:
        if source_video_cache.exists() and source_meta_cache.exists():
            if videos is not None and content_key is not None:
                self._store_direct_source_aggregate_content_index(
                    videos,
                    content_key,
                    source_video_cache,
                    source_meta_cache,
                )
            return
        tmp_meta: Optional[Path] = None
        try:
            copy_mode = self._clone_or_copy_no_hardlink(
                output_path,
                source_video_cache,
                allow_copy=allow_copy,
            )
            if copy_mode is None:
                return
            source_meta_cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix=source_meta_cache.stem + ".",
                suffix=".tmp",
                dir=str(source_meta_cache.parent),
                delete=False,
            ) as fh:
                tmp_meta = Path(fh.name)
                json.dump(cache_key, fh)
            os.replace(tmp_meta, source_meta_cache)
            if videos is not None and content_key is not None:
                self._store_direct_source_aggregate_content_index(
                    videos,
                    content_key,
                    source_video_cache,
                    source_meta_cache,
                )
            self._log_info(
                f"Stored source direct aggregate cache: "
                f"{source_video_cache.name} ({copy_mode})"
            )
        except Exception as exc:  # noqa: BLE001
            if tmp_meta is not None:
                tmp_meta.unlink(missing_ok=True)
            self._log_warning(
                f"failed to store source direct aggregate cache "
                f"{source_video_cache.name}: {exc!r}"
            )

    def _write_direct_aggregated_video(
        self,
        output_dir: Path,
        camera_key: str,
        chunk_idx: int,
        file_idx: int,
        videos: List[Tuple[int, Path, float]],
        episode_by_index: Dict[int, EpisodeData],
        precomputed_source_cache_match: Optional[_DirectSourceCacheMatch] = None,
    ) -> None:
        """Write one v3 aggregate directly from raw MP4s and sidecar timing."""
        output_path = output_dir / DEFAULT_VIDEO_PATH.format(
            video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        camera_name = self._camera_name_from_feature_key(camera_key)
        expected_frames = int(
            sum(episode_by_index[int(ep_idx)].length for ep_idx, _, _ in videos)
        )
        require_decode_validation = self._direct_aggregate_requires_decode_validation()
        require_output_validation = self._direct_aggregate_requires_output_validation()
        cache_path = self._aggregated_video_cache_path(output_path)
        content_key: Optional[Dict[str, Any]] = None
        if precomputed_source_cache_match is not None:
            source_video_cache, source_meta_cache, cached_key = (
                precomputed_source_cache_match
            )
            if self._try_reuse_direct_source_aggregate_cache(
                source_video_cache,
                source_meta_cache,
                output_path,
                cache_path,
                cached_key,
                expected_frames,
                require_decode=require_decode_validation,
                validate=require_output_validation,
            ):
                return
        if self._can_reuse_direct_cache_without_encoder_probe():
            content_key = self._direct_aggregated_video_content_cache_key(
                videos,
                expected_frames,
                episode_by_index,
                camera_name,
            )
            if self._try_reuse_aggregated_video_cache_by_content(
                output_path,
                cache_path,
                content_key,
                expected_frames,
                require_decode=require_decode_validation,
                validate=require_output_validation,
            ):
                self._log_info(
                    f"Reused cached direct aggregate video: {output_path.name} "
                    f"({expected_frames} frames, encoder probe skipped)"
                )
                return
            source_cache_match = self._find_direct_source_aggregate_cache_by_content(
                videos,
                content_key,
            )
            if source_cache_match is not None:
                source_video_cache, source_meta_cache, cached_key = source_cache_match
                if self._try_reuse_direct_source_aggregate_cache(
                    source_video_cache,
                    source_meta_cache,
                    output_path,
                    cache_path,
                    cached_key,
                    expected_frames,
                    require_decode=require_decode_validation,
                    validate=require_output_validation,
                ):
                    return
        ffmpeg = _ffmpeg()
        width, height = self._quick_video_dimensions_cached(Path(videos[0][1]))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid direct aggregate dimensions for {camera_key}")
        encoder, encoder_opts = _h264_encoder(ffmpeg, width=width, height=height)
        encoder_opts = self._direct_aggregate_encoder_opts(encoder, encoder_opts)
        cache_key = self._direct_aggregated_video_cache_key(
            videos,
            expected_frames,
            episode_by_index,
            camera_name,
            ffmpeg=ffmpeg,
            width=width,
            height=height,
            encoder=encoder,
            encoder_opts=encoder_opts,
            content_key=content_key,
        )
        if self._try_reuse_aggregated_video_cache(
            output_path,
            cache_path,
            cache_key,
            expected_frames,
            require_decode=require_decode_validation,
            validate=require_output_validation,
        ):
            self._log_info(
                f"Reused cached direct aggregate video: {output_path.name} "
                f"({expected_frames} frames)"
            )
            return
        source_cache_paths = self._direct_source_aggregate_cache_paths(
            videos,
            cache_key,
        )
        if source_cache_paths is not None:
            source_video_cache, source_meta_cache = source_cache_paths
            if self._try_reuse_direct_source_aggregate_cache(
                source_video_cache,
                source_meta_cache,
                output_path,
                cache_path,
                cache_key,
                expected_frames,
                require_decode=require_decode_validation,
                validate=require_output_validation,
            ):
                return

        frame_size = width * height * 3 // 2
        pipe_size = _ffmpeg_pipe_size(frame_size)
        fps_str = f"{float(self.config.fps):g}"
        encoder_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            *_ffmpeg_threads_arg(),
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}",
            "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", encoder,
            *encoder_opts,
            "-pix_fmt", "yuv420p",
            "-r", fps_str,
            "-an",
            "-video_track_timescale", "90000",
            *_mp4_faststart_args(),
            str(output_path),
        ]

        use_concat_decoder_attempt = (
            len(videos) > 1
            and not os.environ.get(_V30_CONCAT_DECODER_DISABLE_ENV)
        )
        concat_retry_used = False
        while True:
            encoder_process: Optional[subprocess.Popen] = None
            try:
                encoder_process = subprocess.Popen(
                    encoder_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                assert encoder_process.stdin is not None
                _set_pipe_size(encoder_process.stdin, pipe_size)
                total_written = 0
                frame_requests: List[
                    Tuple[int, Path, np.ndarray, int, Optional[_StreamingRgbStats]]
                ] = []
                concat_decoder_ok = use_concat_decoder_attempt
                trust_sidecar_frame_count = not require_decode_validation
                for ep_idx, video_path, _ in videos:
                    episode = episode_by_index[int(ep_idx)]
                    path = Path(video_path)
                    indices, source_frame_count = (
                        self._grid_indices_and_source_count_for_raw_video(
                            episode,
                            camera_name,
                            path,
                        )
                    )
                    if trust_sidecar_frame_count:
                        if int(source_frame_count) <= 0:
                            concat_decoder_ok = False
                    else:
                        actual_frame_count = self._get_video_frame_count(path)
                        if (
                            actual_frame_count is None
                            or int(actual_frame_count) != int(source_frame_count)
                        ):
                            concat_decoder_ok = False
                    stats = (
                        _StreamingRgbStats()
                        if self._video_stats_sample_budget() > 0
                        else None
                    )
                    frame_requests.append(
                        (int(ep_idx), path, indices, int(source_frame_count), stats)
                    )

                if concat_decoder_ok:
                    try:
                        total_written = (
                            self._pipe_selected_yuv420_frames_concat_decoder(
                                ffmpeg,
                                frame_requests,
                                frame_size,
                                encoder_process.stdin,
                                width=width,
                                height=height,
                            )
                        )
                    except Exception as exc:
                        raise _DirectV30ConcatDecoderError(str(exc)) from exc
                else:
                    for ep_idx, video_path, indices, _, stats in frame_requests:
                        total_written += self._pipe_selected_yuv420_frames(
                            ffmpeg,
                            video_path,
                            indices,
                            frame_size,
                            encoder_process.stdin,
                            width=width,
                            height=height,
                            stats=stats,
                        )
                for ep_idx, _, _, _, stats in frame_requests:
                    if stats is not None and stats.frame_count > 0:
                        self._direct_video_stats_cache[
                            (int(ep_idx), camera_name)
                        ] = stats.to_stats()

                encoder_process.stdin.close()
                stderr = (
                    encoder_process.stderr.read().decode(errors="replace")
                    if encoder_process.stderr is not None else ""
                )
                rc = encoder_process.wait(timeout=300)
                if rc != 0:
                    raise RuntimeError(
                        f"ffmpeg direct aggregate rc={rc}: {stderr[-500:]}"
                    )
                if total_written != expected_frames:
                    raise RuntimeError(
                        f"direct aggregate wrote {total_written} frames; "
                        f"expected {expected_frames}"
                    )
                if require_output_validation:
                    _validated_video_count(
                        output_mp4=output_path,
                        expected_frames=expected_frames,
                        target_fps=int(self.config.fps),
                        ffmpeg=ffmpeg,
                        label="direct-v3",
                        require_decode=require_decode_validation,
                    )
                self._write_aggregated_video_cache(cache_path, cache_key)
                self._remember_video_dimensions_from_cache_key(
                    cache_key,
                    output_path=output_path,
                )
                if source_cache_paths is not None:
                    allow_source_cache_copy = (
                        self._allow_direct_source_aggregate_cache_copy(encoder)
                    )
                    self._store_direct_source_aggregate_cache(
                        output_path,
                        source_video_cache,
                        source_meta_cache,
                        cache_key,
                        videos=videos,
                        content_key=content_key,
                        allow_copy=allow_source_cache_copy,
                    )
                self._log_info(
                    f"Direct-aggregated {len(videos)} raw videos: "
                    f"{output_path.name} ({expected_frames} frames @ {fps_str} fps)"
                )
                return
            except _DirectV30ConcatDecoderError as exc:
                output_path.unlink(missing_ok=True)
                if use_concat_decoder_attempt and not concat_retry_used:
                    concat_retry_used = True
                    use_concat_decoder_attempt = False
                    self._log_warning(
                        f"direct aggregate concat decoder failed for "
                        f"{output_path.name}: {exc!r}; retrying with "
                        "per-file decoders"
                    )
                    continue
                self._log_warning(
                    f"direct aggregate concat decoder failed for "
                    f"{output_path.name}: {exc!r}; falling back to synced clips"
                )
                self._write_direct_aggregate_synced_fallback(
                    output_dir,
                    camera_key,
                    chunk_idx,
                    file_idx,
                    videos,
                    episode_by_index,
                )
                return
            except Exception as exc:  # noqa: BLE001
                output_path.unlink(missing_ok=True)
                self._log_warning(
                    f"direct aggregate failed for {output_path.name}: "
                    f"{exc!r}; falling back to synced clips"
                )
                self._write_direct_aggregate_synced_fallback(
                    output_dir,
                    camera_key,
                    chunk_idx,
                    file_idx,
                    videos,
                    episode_by_index,
                )
                return
            finally:
                _terminate_process(encoder_process, close_stdin=True)

    def _pipe_selected_yuv420_frames(
        self,
        ffmpeg: str,
        video_path: Path,
        indices: Sequence[int],
        frame_size: int,
        output,
        *,
        width: int,
        height: int,
        stats: Optional[_StreamingRgbStats] = None,
    ) -> int:
        decoder_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            *_ffmpeg_h264_decoder_args(ffmpeg),
            "-i", str(video_path),
            "-map", "0:v:0",
            "-an",
            "-fps_mode", "passthrough",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "pipe:1",
        ]
        decoder: Optional[subprocess.Popen] = None
        discard_fd: Optional[int] = None
        try:
            decoder = subprocess.Popen(
                decoder_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            assert decoder.stdout is not None
            _set_pipe_size(decoder.stdout, _ffmpeg_pipe_size(frame_size))
            current_idx = -1
            last_frame: Optional[bytearray] = None
            read_buffer = bytearray(frame_size)
            drain_buffer = bytearray(min(frame_size, 1024 * 1024))
            if hasattr(os, "splice"):
                try:
                    discard_fd = os.open(os.devnull, os.O_WRONLY)
                except OSError:
                    discard_fd = None
            written = 0
            sample_positions = (
                _video_stats_sample_positions(len(indices))
                if stats is not None else set()
            )
            out_idx = 0
            indices_len = len(indices)
            while out_idx < indices_len:
                requested_idx = int(indices[out_idx])
                run_end = out_idx + 1
                while (
                    run_end < indices_len
                    and int(indices[run_end]) == requested_idx
                ):
                    run_end += 1
                run_length = run_end - out_idx
                needs_stats_sample = (
                    stats is not None
                    and any(
                        out_idx <= sample_idx < run_end
                        for sample_idx in sample_positions
                    )
                )
                while current_idx < requested_idx:
                    skipped_frames = requested_idx - current_idx - 1
                    if skipped_frames > 0:
                        skipped_bytes = skipped_frames * frame_size
                        n = _drain_exact(
                            decoder.stdout,
                            skipped_bytes,
                            discard_fd=discard_fd,
                            buffer=drain_buffer,
                        )
                        if n != skipped_bytes:
                            raise RuntimeError(
                                f"short drain from {video_path.name}: "
                                f"{n}/{skipped_bytes}"
                            )
                        current_idx += skipped_frames
                        last_frame = None
                        continue
                    if (
                        not needs_stats_sample
                        and run_length == 1
                        and current_idx + 1 == requested_idx
                    ):
                        forward_run = (
                            _contiguous_forward_run_length(indices, out_idx)
                            if stats is None else 1
                        )
                        splice_end = out_idx + forward_run
                        splice_bytes = frame_size * forward_run
                        n = _splice_exact(decoder.stdout, output, splice_bytes)
                        if n == splice_bytes:
                            current_idx += forward_run
                            last_frame = None
                            written += forward_run
                            out_idx = splice_end
                            run_end = splice_end
                            break
                        if n:
                            raise RuntimeError(
                                f"short splice from {video_path.name}: "
                                f"{n}/{splice_bytes}"
                            )
                    n = _read_exact_into(decoder.stdout, read_buffer, frame_size)
                    if n != frame_size:
                        if last_frame is None:
                            raise RuntimeError(
                                f"no frames decoded from {video_path.name}"
                            )
                        break
                    current_idx += 1
                    if last_frame is None:
                        last_frame = read_buffer
                        read_buffer = bytearray(frame_size)
                    else:
                        last_frame, read_buffer = read_buffer, last_frame
                if out_idx == run_end:
                    continue
                if last_frame is None:
                    raise RuntimeError(f"no frame available for {video_path.name}")
                _write_repeated_frame_bytes(output, last_frame, run_length)
                if stats is not None:
                    for sample_idx in sample_positions:
                        if out_idx <= sample_idx < run_end:
                            _add_frame_yuv420p_for_stats(
                                stats,
                                last_frame,
                                width=width,
                                height=height,
                            )
                written += run_length
                out_idx = run_end
            return written
        finally:
            try:
                if discard_fd is not None:
                    os.close(discard_fd)
            except OSError:
                pass
            _terminate_process(decoder)

    def _pipe_selected_yuv420_frames_concat_decoder(
        self,
        ffmpeg: str,
        frame_requests: Sequence[
            Tuple[int, Path, np.ndarray, int, Optional[_StreamingRgbStats]]
        ],
        frame_size: int,
        output,
        *,
        width: int,
        height: int,
    ) -> int:
        """Decode a batch of source videos through one ffmpeg concat pipe."""
        if not frame_requests:
            return 0
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as concat_file:
            for _, video_path, _, _, _ in frame_requests:
                concat_file.write(_concat_file_line(Path(video_path)))
            concat_list_path = concat_file.name

        decoder_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "fatal",
            *_ffmpeg_h264_decoder_args(ffmpeg),
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-map", "0:v:0",
            "-an",
            "-fps_mode", "passthrough",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "pipe:1",
        ]
        decoder: Optional[subprocess.Popen] = None
        discard_fd: Optional[int] = None
        try:
            decoder = subprocess.Popen(
                decoder_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            assert decoder.stdout is not None
            _set_pipe_size(decoder.stdout, _ffmpeg_pipe_size(frame_size))
            current_idx = -1
            source_offset = 0
            last_frame: Optional[bytearray] = None
            read_buffer = bytearray(frame_size)
            drain_buffer = bytearray(min(frame_size, 1024 * 1024))
            if hasattr(os, "splice"):
                try:
                    discard_fd = os.open(os.devnull, os.O_WRONLY)
                except OSError:
                    discard_fd = None
            written = 0
            for _, video_path, indices, source_count, stats in frame_requests:
                sample_positions = (
                    _video_stats_sample_positions(len(indices))
                    if stats is not None else set()
                )
                out_idx = 0
                indices_len = len(indices)
                while out_idx < indices_len:
                    requested_idx_raw = int(indices[out_idx])
                    run_end = out_idx + 1
                    while (
                        run_end < indices_len
                        and int(indices[run_end]) == requested_idx_raw
                    ):
                        run_end += 1
                    run_length = run_end - out_idx
                    requested_idx = source_offset + requested_idx_raw
                    needs_stats_sample = (
                        stats is not None
                        and any(
                            out_idx <= sample_idx < run_end
                            for sample_idx in sample_positions
                        )
                    )
                    while current_idx < requested_idx:
                        skipped_frames = requested_idx - current_idx - 1
                        if skipped_frames > 0:
                            skipped_bytes = skipped_frames * frame_size
                            n = _drain_exact(
                                decoder.stdout,
                                skipped_bytes,
                                discard_fd=discard_fd,
                                buffer=drain_buffer,
                            )
                            if n != skipped_bytes:
                                raise RuntimeError(
                                    "short drain from concat video batch: "
                                    f"{n}/{skipped_bytes}"
                                )
                            current_idx += skipped_frames
                            last_frame = None
                            continue
                        if (
                            not needs_stats_sample
                            and run_length == 1
                            and current_idx + 1 == requested_idx
                        ):
                            forward_run = (
                                _contiguous_forward_run_length(indices, out_idx)
                                if stats is None else 1
                            )
                            splice_end = out_idx + forward_run
                            splice_bytes = frame_size * forward_run
                            n = _splice_exact(decoder.stdout, output, splice_bytes)
                            if n == splice_bytes:
                                current_idx += forward_run
                                last_frame = None
                                written += forward_run
                                out_idx = splice_end
                                run_end = splice_end
                                break
                            if n:
                                raise RuntimeError(
                                    "short splice from concat video batch: "
                                    f"{n}/{splice_bytes}"
                                )
                        n = _read_exact_into(decoder.stdout, read_buffer, frame_size)
                        if n != frame_size:
                            if last_frame is None:
                                raise RuntimeError(
                                    "no frames decoded from concat video batch"
                                )
                            break
                        current_idx += 1
                        if last_frame is None:
                            last_frame = read_buffer
                            read_buffer = bytearray(frame_size)
                        else:
                            last_frame, read_buffer = read_buffer, last_frame
                    if out_idx == run_end:
                        continue
                    if last_frame is None:
                        raise RuntimeError(
                            f"no frame available for {Path(video_path).name}"
                        )
                    _write_repeated_frame_bytes(output, last_frame, run_length)
                    if stats is not None:
                        for sample_idx in sample_positions:
                            if out_idx <= sample_idx < run_end:
                                _add_frame_yuv420p_for_stats(
                                    stats,
                                    last_frame,
                                    width=width,
                                    height=height,
                                )
                    written += run_length
                    out_idx = run_end
                source_offset += int(source_count)
            return written
        finally:
            try:
                if discard_fd is not None:
                    os.close(discard_fd)
            except OSError:
                pass
            _terminate_process(decoder)
            Path(concat_list_path).unlink(missing_ok=True)

    def _write_direct_aggregate_synced_fallback(
        self,
        output_dir: Path,
        camera_key: str,
        chunk_idx: int,
        file_idx: int,
        videos: List[Tuple[int, Path, float]],
        episode_by_index: Dict[int, EpisodeData],
    ) -> None:
        camera_name = self._camera_name_from_feature_key(camera_key)
        fallback_dir = (
            output_dir
            / "_direct_aggregate_fallback"
            / camera_key
            / f"chunk-{chunk_idx:03d}-file-{file_idx:03d}"
        )
        fallback_dir.mkdir(parents=True, exist_ok=True)
        synced: List[Tuple[int, Path, float]] = []
        with _force_h264_software_encoder():
            for ep_idx, video_path, duration in videos:
                episode = episode_by_index[int(ep_idx)]
                indices = self._grid_indices_for_raw_video(
                    episode,
                    camera_name,
                    Path(video_path),
                )
                out_path = fallback_dir / f"episode_{int(ep_idx):08d}.mp4"
                remux_selected_frames(
                    Path(video_path),
                    indices,
                    out_path,
                    target_fps=int(self.config.fps),
                )
                synced.append((int(ep_idx), out_path, duration))
            self._concatenate_videos(
                output_dir,
                camera_key,
                chunk_idx,
                file_idx,
                synced,
            )

    def _concatenate_videos(
        self,
        output_dir: Path,
        camera_key: str,
        chunk_idx: int,
        file_idx: int,
        videos: List[Tuple[int, Path, float]],
    ):
        """Concatenate episode videos into a validated H.264 aggregate."""
        output_path = output_dir / DEFAULT_VIDEO_PATH.format(
            video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        expected_frames = self._expected_aggregated_frame_count(videos)
        cache_path = self._aggregated_video_cache_path(output_path)
        cache_key = self._aggregated_video_cache_key(videos, expected_frames)
        if self._try_reuse_aggregated_video_cache(
            output_path,
            cache_path,
            cache_key,
            expected_frames,
        ):
            self._log_info(
                f"Reused cached aggregate video: {output_path.name} "
                f"({expected_frames} frames)"
            )
            return

        ffmpeg = _ffmpeg()
        fps = float(self.config.fps)
        fps_str = f"{fps:g}"
        if (
            len(videos) == 1
            and not os.environ.get(_VIDEO_SYNC_DISABLE_COPY_FASTPATH_ENV)
            and self._videos_support_copy_concat(videos)
        ):
            if self._copy_single_compatible_video(
                Path(videos[0][1]),
                output_path,
                expected_frames=expected_frames,
            ):
                self._log_info(
                    f"Copied single aggregate video: {output_path.name} "
                    f"({expected_frames} frames @ {fps_str} fps)"
                )
                self._write_aggregated_video_cache(cache_path, cache_key)
                return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for _, video_path, _ in videos:
                f.write(_concat_file_line(Path(video_path)))
            concat_list_path = f.name

        try:
            if self._try_concatenate_videos_copy(
                ffmpeg,
                concat_list_path,
                output_path,
                videos,
                expected_frames,
            ):
                self._log_info(
                    f"Concatenated {len(videos)} videos with stream copy: "
                    f"{output_path.name} ({expected_frames} frames @ {fps_str} fps)"
                )
                self._write_aggregated_video_cache(cache_path, cache_key)
                return

            encoder_height, encoder_width = self._get_video_dimensions(
                Path(videos[0][1])
            )
            encoder, encoder_opts = _h264_encoder(
                ffmpeg,
                width=encoder_width,
                height=encoder_height,
            )
            input_args: List[str] = []
            filter_parts: List[str] = []
            concat_inputs: List[str] = []
            for idx, (_, video_path, _) in enumerate(videos):
                input_args.extend(["-i", str(video_path)])
                label = f"v{idx}"
                filter_parts.append(
                    f"[{idx}:v]fps={fps_str},setpts=PTS-STARTPTS[{label}]"
                )
                concat_inputs.append(f"[{label}]")
            filter_parts.append(
                "".join(concat_inputs)
                + f"concat=n={len(videos)}:v=1:a=0,"
                + f"fps={fps_str},setpts=N/({fps_str}*TB)[outv]"
            )
            cmd = [
                ffmpeg,
                "-y",
                *_ffmpeg_threads_arg(),
                *input_args,
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[outv]",
                "-r",
                fps_str,
                "-an",
                "-c:v",
                encoder,
                *encoder_opts,
                "-pix_fmt",
                "yuv420p",
                *_mp4_faststart_args(),
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self._log_error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to concatenate videos: {result.stderr}")

            self._validate_aggregated_video(
                output_path,
                expected_frames,
                require_decode=True,
            )
            self._log_info(
                f"Concatenated {len(videos)} videos with CFR re-encode: "
                f"{output_path.name} ({expected_frames} frames @ {fps_str} fps)"
            )
            self._write_aggregated_video_cache(cache_path, cache_key)
        finally:
            Path(concat_list_path).unlink(missing_ok=True)

    @staticmethod
    def _aggregated_video_cache_path(output_path: Path) -> Path:
        return output_path.with_name(output_path.name + ".cache.json")

    def _aggregated_video_cache_key(
        self,
        videos: List[Tuple[int, Path, float]],
        expected_frames: int,
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "mode": "v3_concat",
            "target_fps": float(self.config.fps),
            "expected_frames": int(expected_frames),
            "inputs": [
                {
                    "episode_index": int(ep_idx),
                    "video": self._file_signature(Path(video_path)),
                }
                for ep_idx, video_path, _ in videos
            ],
        }

    def _try_reuse_aggregated_video_cache(
        self,
        output_path: Path,
        cache_path: Path,
        cache_key: Dict[str, Any],
        expected_frames: int,
        *,
        require_decode: bool = True,
        validate: bool = True,
    ) -> bool:
        if not output_path.exists() or output_path.stat().st_size <= 0:
            return False
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if cached != cache_key:
            return False
        try:
            if validate:
                self._validate_aggregated_video(
                    output_path,
                    expected_frames,
                    require_decode=require_decode,
                )
            self._remember_video_dimensions_from_cache_key(
                cache_key,
                output_path=output_path,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"cached aggregate validation failed for "
                f"{output_path.name}: {exc!r}; regenerating"
            )
            return False

    def _write_aggregated_video_cache(
        self,
        cache_path: Path,
        cache_key: Dict[str, Any],
    ) -> None:
        tmp_path: Optional[Path] = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix=cache_path.stem + ".",
                suffix=".tmp",
                dir=str(cache_path.parent),
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                json.dump(cache_key, fh)
            os.replace(tmp_path, cache_path)
        except Exception as exc:  # noqa: BLE001
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            self._log_warning(
                f"failed to write aggregate video cache {cache_path.name}: "
                f"{exc!r}"
            )

    def _copy_single_compatible_video(
        self,
        source_path: Path,
        output_path: Path,
        *,
        expected_frames: int,
    ) -> bool:
        """Clone/copy one already-compatible video and validate the result."""
        source_path = Path(source_path)
        output_path = Path(output_path)
        copied = False
        same_path = _same_file_or_same_path(source_path, output_path)
        try:
            if not same_path:
                _clone_or_copy_file(source_path, output_path)
                copied = True
            self._validate_aggregated_video(
                output_path,
                expected_frames,
                require_decode=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            if copied:
                output_path.unlink(missing_ok=True)
            self._log_warning(
                f"single-video aggregate copy validation failed for "
                f"{output_path.name}: {exc!r}; re-encoding"
            )
            return False

    def _try_concatenate_videos_copy(
        self,
        ffmpeg: str,
        concat_list_path: str,
        output_path: Path,
        videos: List[Tuple[int, Path, float]],
        expected_frames: int,
    ) -> bool:
        if os.environ.get(_VIDEO_SYNC_DISABLE_COPY_FASTPATH_ENV):
            return False
        if not videos:
            return False
        if not self._videos_support_copy_concat(videos):
            return False
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c:v",
            "copy",
            "-an",
            "-video_track_timescale",
            "90000",
            *_mp4_faststart_args(),
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            output_path.unlink(missing_ok=True)
            return False
        try:
            self._validate_aggregated_video(
                output_path,
                expected_frames,
                require_decode=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            output_path.unlink(missing_ok=True)
            self._log_warning(
                f"stream-copy aggregate validation failed for "
                f"{output_path.name}: {exc!r}; re-encoding"
            )
            return False

    def _synced_copy_concat_info(
        self,
        videos: List[Tuple[int, Path, float]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Return cache-backed info for converter-produced synced MP4s."""
        infos: List[Dict[str, Any]] = []
        for _, video_path, _ in videos:
            path = Path(video_path)
            if not path.exists() or path.stat().st_size <= 0:
                return None
            if not path.stem.endswith("_synced"):
                return None
            cache_path = path.with_name(path.stem + ".cache.json")
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if not isinstance(cache, dict):
                return None
            try:
                fps = float(cache.get("target_fps"))
                frame_count = int(cache.get("frame_count"))
            except (TypeError, ValueError):
                return None
            if int(round(fps)) != int(self.config.fps) or frame_count <= 0:
                return None

            height = cache.get("output_height")
            width = cache.get("output_width")
            try:
                height_i = int(height)
                width_i = int(width)
            except (TypeError, ValueError):
                height_i, width_i = self._get_video_dimensions(path)
                if height_i <= 0 or width_i <= 0:
                    return None
                cache["output_height"] = int(height_i)
                cache["output_width"] = int(width_i)
                cache.setdefault("output_codec", "h264")
                cache.setdefault("output_pix_fmt", "yuv420p")
                cache.setdefault("has_audio", False)
                try:
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
                except OSError:
                    pass
            infos.append({
                "frame_count": int(frame_count),
                "fps": float(fps),
                "height": int(height_i),
                "width": int(width_i),
                "codec_name": str(cache.get("output_codec") or "h264"),
                "pix_fmt": str(cache.get("output_pix_fmt") or "yuv420p"),
                "has_audio": bool(cache.get("has_audio", False)),
            })
        if not infos:
            return None
        first = infos[0]
        for info in infos[1:]:
            if (
                info["height"] != first["height"]
                or info["width"] != first["width"]
                or info["codec_name"] != first["codec_name"]
                or info["pix_fmt"] != first["pix_fmt"]
                or info["has_audio"]
            ):
                return None
        if first["has_audio"] or first["codec_name"] != "h264":
            return None
        return infos

    def _videos_support_copy_concat(
        self,
        videos: List[Tuple[int, Path, float]],
    ) -> bool:
        """Return True when videos are safe for MP4 concat stream copy."""
        if not videos:
            return False
        if self._synced_copy_concat_info(videos) is not None:
            return True

        reference: Optional[Tuple[Any, ...]] = None
        for _, video_path, _ in videos:
            path = Path(video_path)
            frame_count = self._get_video_frame_count(path)
            if frame_count is None or frame_count <= 0:
                return False
            info = self._probe_video_streams(path)
            if not info:
                return False
            streams = info.get("streams") or []
            if any(stream.get("codec_type") == "audio" for stream in streams):
                return False
            video_stream = next(
                (s for s in streams if s.get("codec_type") == "video"), None
            )
            if not video_stream or video_stream.get("codec_name") != "h264":
                return False
            stream_fps = self._stream_fps(video_stream)
            if stream_fps is None or abs(stream_fps - float(self.config.fps)) > 0.01:
                return False
            try:
                if int(video_stream.get("has_b_frames", 0) or 0) > 0:
                    return False
            except (TypeError, ValueError):
                return False
            signature = self._video_stream_copy_signature(video_stream)
            if reference is None:
                reference = signature
            elif signature != reference:
                return False
        return True

    def _compatible_for_concat_copy(
        self,
        videos: List[Tuple[int, Path, float]],
        expected_frames: int,
    ) -> bool:
        if not videos:
            return False
        cached_infos = self._synced_copy_concat_info(videos)
        if cached_infos is not None:
            return (
                sum(int(info["frame_count"]) for info in cached_infos)
                == int(expected_frames)
            )
        total = 0
        baseline: Optional[Tuple[Any, ...]] = None
        for _, video_path, _ in videos:
            path = Path(video_path)
            stream = self._probe_video_stream(path)
            if not stream:
                return False
            info = self._probe_video_streams(path)
            if not info:
                return False
            streams = info.get("streams") or []
            if any(item.get("codec_type") == "audio" for item in streams):
                return False
            signature = self._video_stream_copy_signature(stream)
            if baseline is None:
                baseline = signature
            elif signature != baseline:
                return False
            if stream.get("codec_name") != "h264":
                return False
            stream_fps = self._stream_fps(stream)
            if stream_fps is None or abs(stream_fps - float(self.config.fps)) > 0.01:
                return False
            try:
                if int(stream.get("has_b_frames", 0) or 0) > 0:
                    return False
            except (TypeError, ValueError):
                return False
            frame_count = self._get_video_frame_count(path)
            if frame_count is None:
                return False
            total += int(frame_count)
        return int(total) == int(expected_frames)

    def _probe_video_stream(self, video_path: Path) -> Optional[Dict[str, Any]]:
        info = self._probe_video_streams(Path(video_path))
        if not info:
            return None
        streams = info.get("streams") or []
        return next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )

    @staticmethod
    def _stream_fps(stream: Dict[str, Any]) -> Optional[float]:
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        if not rate:
            return None
        try:
            if "/" in str(rate):
                num, _, den = str(rate).partition("/")
                den_f = float(den)
                if den_f == 0:
                    return None
                return float(num) / den_f
            return float(rate)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _stream_has_audio(video_path: Path) -> bool:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return True

    def _expected_aggregated_frame_count(
        self, videos: List[Tuple[int, Path, float]]
    ) -> int:
        """Return the row-count-backed frame total for an aggregate video."""
        length_by_episode = {
            ep.episode_index: ep.length for ep in self._episode_metadata_list
        }
        missing = [
            ep_idx for ep_idx, _, _ in videos
            if ep_idx not in length_by_episode
        ]
        if missing:
            raise RuntimeError(
                "Cannot validate aggregated video frame count; missing "
                f"episode metadata for episodes {missing}"
            )
        return int(sum(length_by_episode[ep_idx] for ep_idx, _, _ in videos))

    def _validate_aggregated_video(
        self,
        video_path: Path,
        expected_frames: int,
        *,
        require_decode: bool = True,
    ) -> None:
        """Validate frame count and nominal FPS for a v3.0 aggregate MP4."""
        frame_count, fps, decode_ok = self._probe_video_count_fps_decode(
            video_path,
            require_decode=require_decode,
        )
        if frame_count is None:
            message = f"Could not determine frame count for {video_path}"
            self._log_error(message)
            raise RuntimeError(message)
        if frame_count != expected_frames:
            message = (
                f"Aggregated video frame count mismatch for {video_path}: "
                f"expected {expected_frames}, got {frame_count}"
            )
            self._log_error(message)
            raise RuntimeError(message)

        if fps is None:
            self._log_warning(f"Could not determine FPS for {video_path}")
            return
        expected_fps = float(self.config.fps)
        if abs(fps - expected_fps) > 0.01:
            message = (
                f"Aggregated video FPS mismatch for {video_path}: "
                f"expected {expected_fps:g}, got {fps:g}"
            )
            self._log_error(message)
            raise RuntimeError(message)
        if require_decode and not decode_ok:
            message = f"Aggregated video failed decode validation for {video_path}"
            self._log_error(message)
            raise RuntimeError(message)

    def _probe_video_count_fps_decode(
        self,
        video_path: Path,
        *,
        require_decode: bool = True,
    ) -> Tuple[Optional[int], Optional[float], bool]:
        frame_count = self._get_video_frame_count(video_path)
        fps = self._probe_video_fps(video_path)
        decode_ok = True
        if require_decode:
            decode_ok = self._video_decodes_successfully(video_path)
        return frame_count, fps, decode_ok

    def _probe_video_fps(self, video_path: Path) -> Optional[float]:
        """Probe avg_frame_rate with ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            rate = result.stdout.strip()
            if "/" in rate:
                num, _, den = rate.partition("/")
                den_f = float(den)
                if den_f == 0:
                    return None
                return float(num) / den_f
            return float(rate)
        except Exception as e:
            self._log_warning(f"Failed to probe FPS for {video_path}: {e}")
            return None

    def _update_video_metadata(
        self,
        camera_key: str,
        chunk_idx: int,
        file_idx: int,
        videos: List[Tuple[int, Path, float]],
    ):
        """Update episode metadata with video file locations and timestamps.

        Uses data-based timestamps (episode length / fps) instead of actual video
        duration to ensure all cameras have identical timestamps, matching the
        LeRobot reference format (e.g., lerobot/aloha_static_towel).
        """
        current_timestamp = 0.0

        for ep_idx, _, _ in videos:
            ep_metadata = self._episode_metadata_by_index.get(int(ep_idx))
            if ep_metadata is None:
                continue
            data_based_duration = ep_metadata.length / self.config.fps
            ep_metadata.video_metadata[camera_key] = {
                "chunk_index": chunk_idx,
                "file_index": file_idx,
                "from_timestamp": current_timestamp,
                "to_timestamp": current_timestamp + data_based_duration,
            }
            current_timestamp += data_based_duration

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            self._log_warning(f"Failed to get video duration: {e}")

        # Fallback: estimate from frame count
        return self.config.fps * 10.0  # Assume 10 seconds

    def _write_episodes_parquet(
        self,
        episodes_data_for_cache: Optional[List[EpisodeData]] = None,
    ):
        """Write episodes metadata to Parquet file."""
        self._log_info("Writing episodes metadata...")

        output_dir = Path(self.config.output_dir)

        # Convert metadata to list of dicts
        episodes_data = [ep.to_dict() for ep in self._episode_metadata_list]

        if not episodes_data:
            return

        # For simplicity, write all episodes to a single file
        # (could be chunked for very large datasets)
        file_path = output_dir / DEFAULT_EPISODES_PATH.format(
            chunk_index=0, file_index=0
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        cache_key = self._episodes_parquet_cache_key()
        if cache_key is not None and self._try_reuse_episodes_parquet_cache(
            episodes_data_for_cache,
            cache_key,
            file_path,
        ):
            return

        table = pa.Table.from_pylist(episodes_data)
        pq.write_table(table, file_path, **_v30_parquet_write_kwargs())
        if cache_key is not None:
            self._store_episodes_parquet_cache(
                episodes_data_for_cache,
                cache_key,
                file_path,
            )

        self._log_info(f"Wrote episodes metadata: {file_path}")

    def _write_tasks_parquet(
        self,
        episodes_data_for_cache: Optional[List[EpisodeData]] = None,
    ):
        """Write tasks to Parquet file."""
        self._log_info("Writing tasks...")

        output_dir = Path(self.config.output_dir)
        file_path = output_dir / DEFAULT_TASKS_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)

        task_names = getattr(self, "_task_names_by_task", {})
        tasks_data = [
            {
                "task_index": idx,
                "task": task,
                "task_name": task_names.get(task, task),
            }
            for idx, task in self._tasks.items()
        ]

        if not tasks_data:
            tasks_data = [
                {
                    "task_index": 0,
                    "task": "default_task",
                    "task_name": "default_task",
                }
            ]

        cache_key = self._tasks_parquet_cache_key()
        if cache_key is not None and self._try_reuse_small_parquet_cache(
            episodes_data_for_cache,
            cache_key,
            file_path,
            cache_name="tasks_parquet_v30",
            artifact_name="tasks.parquet",
        ):
            self._log_info("Reused v3 tasks parquet cache")
            return

        table = pa.Table.from_pylist(tasks_data)
        pq.write_table(table, file_path, **_v30_parquet_write_kwargs())
        if cache_key is not None:
            self._store_small_parquet_cache(
                episodes_data_for_cache,
                cache_key,
                file_path,
                cache_name="tasks_parquet_v30",
                artifact_name="tasks.parquet",
            )

        self._log_info(f"Wrote tasks: {file_path}")

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

        cache_key = self._subtasks_parquet_cache_key(rows)
        if cache_key is not None and self._try_reuse_small_parquet_cache(
            episodes_data,
            cache_key,
            path,
            cache_name="subtasks_parquet_v30",
            artifact_name="subtasks.parquet",
        ):
            self._log_info("Reused v3 subtasks parquet cache")
            return

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
        pq.write_table(table, path, **_v30_parquet_write_kwargs())
        if cache_key is not None:
            self._store_small_parquet_cache(
                episodes_data,
                cache_key,
                path,
                cache_name="subtasks_parquet_v30",
                artifact_name="subtasks.parquet",
            )
        self._log_info(f"Wrote subtasks metadata: {path}")

    def _write_global_stats(self):
        """Write global statistics to stats.json.

        Combines per-episode stats with the length-weighted pooled
        variance formula:

            global_mean = sum(w_i * mean_i)
            global_var = sum(w_i * (var_i + (mean_i - global_mean)^2))

        This preserves between-episode variance for joints whose mean
        shifts between takes; averaging episode stds alone under-reports
        the true dataset std and can destabilize downstream normalization.
        """
        self._log_info("Computing global statistics...")

        output_dir = Path(self.config.output_dir)

        per_feature: Dict[str, Dict[str, List[Any]]] = {}

        for ep_metadata in self._episode_metadata_list:
            ep_buckets: Dict[str, Dict[str, Any]] = {}
            for stat_key, stat_value in ep_metadata.stats.items():
                feature_key, stat_type = stat_key.rsplit("/", 1)
                ep_buckets.setdefault(feature_key, {})[stat_type] = stat_value

            for feature_key, ep_stats in ep_buckets.items():
                if not all(k in ep_stats for k in ("mean", "std", "count")):
                    continue
                slot = per_feature.setdefault(
                    feature_key,
                    {"mean": [], "std": [], "min": [], "max": [], "count": []},
                )
                slot["mean"].append(np.asarray(ep_stats["mean"], dtype=np.float64))
                slot["std"].append(np.asarray(ep_stats["std"], dtype=np.float64))
                slot["min"].append(np.asarray(ep_stats["min"], dtype=np.float64))
                slot["max"].append(np.asarray(ep_stats["max"], dtype=np.float64))
                count = ep_stats["count"]
                slot["count"].append(
                    int(count[0] if isinstance(count, (list, tuple)) else count)
                )

        # Compute aggregated stats
        aggregated_stats: Dict[str, Dict[str, List[float]]] = {}
        for feature_key, stats in per_feature.items():
            if not stats["mean"]:
                continue

            mean_arrays = np.stack(stats["mean"])
            std_arrays = np.stack(stats["std"])
            min_arrays = np.stack(stats["min"])
            max_arrays = np.stack(stats["max"])
            counts = np.asarray(stats["count"], dtype=np.float64)

            total = counts.sum()
            if total <= 0:
                continue

            weights = (counts / total).reshape(-1, 1)
            global_mean = (weights * mean_arrays).sum(axis=0)
            pooled_var = (
                weights
                * (std_arrays**2 + (mean_arrays - global_mean) ** 2)
            ).sum(axis=0)
            pooled_std = np.maximum(np.sqrt(pooled_var), STATS_STD_FLOOR)

            aggregated_stats[feature_key] = {
                "mean": global_mean.tolist(),
                "std": pooled_std.tolist(),
                "min": np.min(min_arrays, axis=0).tolist(),
                "max": np.max(max_arrays, axis=0).tolist(),
            }

        stats_path = output_dir / "meta" / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_stats, f, indent=2, ensure_ascii=False)

        self._log_info(f"Wrote global stats: {stats_path}")

    def _write_info_json_v30(self):
        """Write info.json in v3.0 format."""
        output_dir = Path(self.config.output_dir)

        num_video_keys = sum(
            1 for k in self._features if k.startswith("observation.images.")
        )

        # Add fps to each feature
        features_with_fps = {}
        for key, feature in self._features.items():
            features_with_fps[key] = feature.copy()
            if feature.get("dtype") != "video":
                features_with_fps[key]["fps"] = self.config.fps

        info = {
            "codebase_version": CODEBASE_VERSION_V30,
            "robot_type": self.config.robot_type,
            "total_episodes": self._total_episodes,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": self.config.chunks_size,
            "data_files_size_in_mb": self.config.data_file_size_in_mb,
            "video_files_size_in_mb": self.config.video_file_size_in_mb,
            "fps": self.config.fps,
            "splits": {"train": f"0:{self._total_episodes}"},
            "data_path": DEFAULT_DATA_PATH,
            "video_path": DEFAULT_VIDEO_PATH if self.config.use_videos else None,
            "features": features_with_fps,
        }
        if "subtask_index" in self._features:
            info["annotation_path"] = (
                "annotations/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.json"
            )

        info_path = output_dir / "meta" / "info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        self._log_info(f"Wrote info.json (v3.0): {info_path}")

    def _flatten_stats(self, stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Flatten nested stats dict for Parquet storage."""
        flattened = {}
        for feature_key, feature_stats in stats.items():
            for stat_type, stat_value in feature_stats.items():
                flat_key = f"{feature_key}/{stat_type}"
                if isinstance(stat_value, np.ndarray):
                    flattened[flat_key] = stat_value.tolist()
                elif isinstance(stat_value, list):
                    flattened[flat_key] = stat_value
                else:
                    flattened[flat_key] = [stat_value]
        return flattened

    def _advance_chunk_file_index(self, file_type: str):
        """Advance chunk/file indices."""
        if file_type == "data":
            self._current_data_chunk_idx, self._current_data_file_idx = (
                self._update_chunk_file_indices(
                    self._current_data_chunk_idx, self._current_data_file_idx
                )
            )

    def _update_chunk_file_indices(
        self, chunk_idx: int, file_idx: int
    ) -> Tuple[int, int]:
        """Update chunk and file indices."""
        if file_idx >= self.config.chunks_size - 1:
            return chunk_idx + 1, 0
        return chunk_idx, file_idx + 1


def convert_rosbags_to_lerobot_v30(
    bag_paths: List[str],
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "unknown",
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    logger=None,
) -> bool:
    """
    Convenience function to convert multiple ROSbags to LeRobot v3.0 dataset.

    Args:
        bag_paths: List of paths to ROSbag directories
        output_dir: Output directory for the dataset
        repo_id: Repository ID for the dataset
        fps: Target frames per second
        robot_type: Robot type identifier
        data_file_size_in_mb: Target size for data files (default 100MB)
        video_file_size_in_mb: Target size for video files (default 200MB)
        logger: Optional logger instance

    Returns:
        True if successful, False otherwise
    """
    config = V30ConversionConfig(
        repo_id=repo_id,
        output_dir=Path(output_dir),
        fps=fps,
        robot_type=robot_type,
        data_file_size_in_mb=data_file_size_in_mb,
        video_file_size_in_mb=video_file_size_in_mb,
    )

    converter = RosbagToLerobotV30Converter(config, logger)
    return converter.convert_multiple_rosbags([Path(p) for p in bag_paths])
