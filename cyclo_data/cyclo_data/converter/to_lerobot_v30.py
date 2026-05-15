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

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
    _conversion_worker_init,
    _convert_rosbag_worker,
    _resolve_conversion_worker_count,
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
            "data/chunk_index": self.data_chunk_index,
            "data/file_index": self.data_file_index,
            "dataset_from_index": self.dataset_from_index,
            "dataset_to_index": self.dataset_to_index,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

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

        # Per-camera video tracking
        self._video_tracking: Dict[str, Dict[str, Any]] = {}

        # Temporary storage for aggregation
        self._pending_parquet_data: List[pd.DataFrame] = []
        self._pending_video_files: Dict[
            str, List[Tuple[Path, float]]
        ] = {}  # camera -> [(path, duration)]

    def convert_multiple_rosbags(self, bag_paths: List[Path]) -> bool:
        """
        Convert multiple ROSbag recordings to a single LeRobot v3.0 dataset.

        Args:
            bag_paths: List of paths to ROSbag directories

        Returns:
            True if successful, False otherwise
        """
        self._log_info(f"Converting {len(bag_paths)} rosbags to LeRobot v3.0 dataset")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure
        (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

        episodes_data: List[EpisodeData] = []

        # Phase 1: Extract data from each rosbag. Matches v2.1's
        # parallelism (ProcessPoolExecutor up to half the host CPUs) so
        # v3.0-only runs aren't 8× slower than v2.1-only runs. When v2.1
        # ran first against the same dataset, _sync_videos_to_grid hits
        # the synced-MP4 cache sidecar and skips remux entirely.
        if len(bag_paths) <= 1:
            # Single episode: skip pool overhead.
            for idx, bag_path in enumerate(bag_paths):
                episode_data = self.convert_single_rosbag(Path(bag_path), idx)
                if episode_data is not None:
                    episodes_data.append(episode_data)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # Must mirror v2.1's config_dict — every selection knob the
            # converter reads (camera_rotations / image_resize / selected_*)
            # has to ride the pickle to the worker process, otherwise
            # the worker defaults the field and silently drops the UI
            # value.
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
            max_workers = _resolve_conversion_worker_count(len(bag_paths))
            self._log_info(
                f"Starting parallel rosbag parsing with {max_workers} workers"
            )
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_conversion_worker_init,
            ) as executor:
                futures = {}
                for idx, bag_path in enumerate(bag_paths):
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

            # Sort by episode_index — as_completed delivers out of order.
            episodes_data.sort(key=lambda ep: ep.episode_index)

        if not episodes_data:
            self._log_error("No episodes were successfully converted")
            return False

        self._build_features(episodes_data)
        self._collect_tasks(episodes_data)

        self._write_aggregated_data(episodes_data)

        # Phase 4: Write aggregated video files
        self._write_aggregated_videos(episodes_data)

        # Phase 5: Write episodes metadata (Parquet)
        self._write_episodes_parquet()

        # Phase 6: Write tasks (Parquet)
        self._write_tasks_parquet()

        # Phase 7: Write global stats
        self._write_global_stats()

        # Phase 8: Write info.json
        self._write_info_json_v30()
        # Root info.json (conversion config snapshot) — same writer as v2.1.
        self._write_root_info_json()

        # Phase 9: Write quality reports (optional)
        if self.config.enable_quality_report and self._quality_reports:
            self._write_quality_reports(output_dir)

        self._log_info(f"Successfully converted {len(episodes_data)} episodes to v3.0")
        return True

    def _write_aggregated_data(self, episodes_data: List[EpisodeData]):
        """Write episode data to aggregated Parquet files."""
        self._log_info("Writing aggregated data files...")

        output_dir = Path(self.config.output_dir)
        pending_frames: List[Dict[str, Any]] = []
        pending_size_mb = 0.0
        global_frame_index = 0

        for episode in episodes_data:
            ep_idx = episode.episode_index
            num_frames = episode.length

            # Track where this episode starts in the current file
            dataset_from_index = len(pending_frames)

            # Add frames for this episode
            for frame_idx in range(num_frames):
                frame_data = {
                    "timestamp": episode.timestamps[frame_idx],
                    "frame_index": frame_idx,
                    "episode_index": ep_idx,
                    "index": global_frame_index,
                    "task_index": self._task_to_index.get(
                        episode.tasks[0] if episode.tasks else "default_task", 0
                    ),
                }

                if episode.observation_state:
                    frame_data["observation.state"] = episode.observation_state[
                        frame_idx
                    ].tolist()

                if episode.action:
                    frame_data["action"] = episode.action[frame_idx].tolist()

                pending_frames.append(frame_data)
                global_frame_index += 1

            dataset_to_index = len(pending_frames)

            # Create episode metadata
            ep_stats = self._compute_episode_stats(episode)
            ep_metadata = EpisodeMetadata(
                episode_index=ep_idx,
                length=num_frames,
                tasks=episode.tasks,
                data_chunk_index=self._current_data_chunk_idx,
                data_file_index=self._current_data_file_idx,
                dataset_from_index=dataset_from_index,
                dataset_to_index=dataset_to_index,
                stats=self._flatten_stats(ep_stats),
            )
            self._episode_metadata_list.append(ep_metadata)

            # Estimate size (rough approximation)
            pending_size_mb = len(pending_frames) * 0.001  # ~1KB per frame estimate

            # Check if we need to flush
            if pending_size_mb >= self.config.data_file_size_in_mb:
                self._flush_data_file(output_dir, pending_frames)
                pending_frames = []
                pending_size_mb = 0.0
                self._advance_chunk_file_index("data")

        if pending_frames:
            self._flush_data_file(output_dir, pending_frames)

        self._total_episodes = len(episodes_data)
        self._total_frames = global_frame_index

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

        state_dim = (
            len(frames[0].get("observation.state", []))
            if frames[0].get("observation.state")
            else 0
        )
        action_dim = len(frames[0].get("action", [])) if frames[0].get("action") else 0

        schema_fields = [
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]

        if state_dim > 0:
            schema_fields.append(
                pa.field("observation.state", pa.list_(pa.float32(), state_dim))
            )
        if action_dim > 0:
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), action_dim)))

        schema = pa.schema(schema_fields)

        num_frames = len(frames)
        arrays = [
            pa.array([float(f["timestamp"]) for f in frames], type=pa.float32()),
            pa.array([f["frame_index"] for f in frames], type=pa.int64()),
            pa.array([f["episode_index"] for f in frames], type=pa.int64()),
            pa.array([f["index"] for f in frames], type=pa.int64()),
            pa.array([f["task_index"] for f in frames], type=pa.int64()),
        ]

        if state_dim > 0:
            state_values = [[float(v) for v in f["observation.state"]] for f in frames]
            arrays.append(
                pa.array(state_values, type=pa.list_(pa.float32(), state_dim))
            )

        if action_dim > 0:
            action_values = [[float(v) for v in f["action"]] for f in frames]
            arrays.append(
                pa.array(action_values, type=pa.list_(pa.float32(), action_dim))
            )

        hf_features = {
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }

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
        pq.write_table(table, file_path)

        self._log_info(f"Wrote data file: {file_path.name} ({num_frames} frames)")

    def _write_aggregated_videos(self, episodes_data: List[EpisodeData]):
        """Write aggregated video files by concatenating episode videos."""
        self._log_info("Writing aggregated video files...")

        output_dir = Path(self.config.output_dir)

        # Group videos by camera
        camera_videos: Dict[
            str, List[Tuple[int, Path, float]]
        ] = {}  # camera -> [(ep_idx, path, duration)]

        for episode in episodes_data:
            for camera_name, video_path in episode.video_files.items():
                if camera_name not in camera_videos:
                    camera_videos[camera_name] = []

                duration = self._get_video_duration(video_path)
                camera_videos[camera_name].append(
                    (episode.episode_index, video_path, duration)
                )

        # Process each camera
        for camera_name, videos in camera_videos.items():
            self._write_aggregated_videos_for_camera(output_dir, camera_name, videos)

        # ``<cam>_synced.mp4`` files are kept on disk as Phase 1 cache —
        # the next conversion run hits the ``<cam>_synced.cache.json``
        # gate in ``_sync_videos_to_grid`` and skips remux entirely.
        # Disk cost is modest (~2-3 MB per camera per episode);
        # operators who want to reclaim the space can wipe
        # ``<episode>/videos/*_synced.*`` after the dataset is final.

    def _write_aggregated_videos_for_camera(
        self,
        output_dir: Path,
        camera_name: str,
        videos: List[Tuple[int, Path, float]],
    ):
        """Concatenate and write videos for a single camera."""
        camera_key = f"observation.images.{camera_name}"

        chunk_idx = 0
        file_idx = 0
        current_size_mb = 0.0
        current_duration = 0.0
        pending_videos: List[Tuple[int, Path, float]] = []

        for ep_idx, video_path, duration in videos:
            video_size_mb = video_path.stat().st_size / (1024 * 1024)

            # Check if adding this video would exceed limit
            if (
                current_size_mb + video_size_mb >= self.config.video_file_size_in_mb
                and pending_videos
            ):
                # Write current batch
                self._concatenate_videos(
                    output_dir,
                    camera_key,
                    chunk_idx,
                    file_idx,
                    pending_videos,
                )

                # Update episode metadata for written videos
                self._update_video_metadata(
                    camera_key, chunk_idx, file_idx, pending_videos
                )

                # Advance to next file
                chunk_idx, file_idx = self._update_chunk_file_indices(
                    chunk_idx, file_idx
                )
                pending_videos = []
                current_size_mb = 0.0
                current_duration = 0.0

            # Track video for this episode
            pending_videos.append((ep_idx, video_path, duration))
            current_size_mb += video_size_mb
            current_duration += duration

        # Write remaining videos
        if pending_videos:
            self._concatenate_videos(
                output_dir, camera_key, chunk_idx, file_idx, pending_videos
            )
            self._update_video_metadata(camera_key, chunk_idx, file_idx, pending_videos)

    def _concatenate_videos(
        self,
        output_dir: Path,
        camera_key: str,
        chunk_idx: int,
        file_idx: int,
        videos: List[Tuple[int, Path, float]],
    ):
        """Concatenate episode videos into a CFR H.264 aggregate.

        The v3.0 layout stores multiple episodes in one MP4 per camera. A
        stream-copy concat (``-c copy``) can preserve per-file H.264 PTS /
        B-frame quirks and produce an aggregate whose frame count looks
        correct but decodes or seeks to wrong frames near episode boundaries.
        Re-encoding with an explicit CFR filter regenerates timestamps so
        LeRobot's timestamp-based frame lookup stays aligned.
        """
        output_path = output_dir / DEFAULT_VIDEO_PATH.format(
            video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        expected_frames = self._expected_aggregated_frame_count(videos)

        # Create concat list file for ffmpeg
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for _, video_path, _ in videos:
                f.write(f"file '{video_path.resolve()}'\n")
            concat_list_path = f.name

        try:
            fps = float(self.config.fps)
            fps_str = f"{fps:g}"
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list_path,
                "-vf",
                f"fps={fps_str},setpts=N/({fps_str}*TB)",
                "-r",
                fps_str,
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self._log_error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to concatenate videos: {result.stderr}")

            self._validate_aggregated_video(output_path, expected_frames)
            self._log_info(
                f"Concatenated {len(videos)} videos with CFR re-encode: "
                f"{output_path.name} ({expected_frames} frames @ {fps_str} fps)"
            )
        finally:
            Path(concat_list_path).unlink(missing_ok=True)

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
        self, video_path: Path, expected_frames: int
    ) -> None:
        """Validate frame count and nominal FPS for a v3.0 aggregate MP4."""
        frame_count = self._get_video_frame_count(video_path)
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

        fps = self._probe_video_fps(video_path)
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
            for ep_metadata in self._episode_metadata_list:
                if ep_metadata.episode_index == ep_idx:
                    data_based_duration = ep_metadata.length / self.config.fps
                    ep_metadata.video_metadata[camera_key] = {
                        "chunk_index": chunk_idx,
                        "file_index": file_idx,
                        "from_timestamp": current_timestamp,
                        "to_timestamp": current_timestamp + data_based_duration,
                    }
                    current_timestamp += data_based_duration
                    break

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

    def _write_episodes_parquet(self):
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

        df = pd.DataFrame(episodes_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)

        self._log_info(f"Wrote episodes metadata: {file_path}")

    def _write_tasks_parquet(self):
        """Write tasks to Parquet file."""
        self._log_info("Writing tasks...")

        output_dir = Path(self.config.output_dir)
        file_path = output_dir / DEFAULT_TASKS_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)

        tasks_data = [
            {"task_index": idx, "task": task} for idx, task in self._tasks.items()
        ]

        if not tasks_data:
            tasks_data = [{"task_index": 0, "task": "default_task"}]

        df = pd.DataFrame(tasks_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)

        self._log_info(f"Wrote tasks: {file_path}")

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
