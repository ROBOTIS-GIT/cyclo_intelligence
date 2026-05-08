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
ROSbag + MP4 to LeRobot v2.1 Dataset Converter.

Converts recorded robot data (ROSbag with joint states + MP4 videos) to
LeRobot v2.1 dataset format. Shared rosbag extraction and stats logic
lives in ``base_converter.py``; this module owns the v2.1-format
writers (per-episode parquet + JSONL meta).

LeRobot v2.1 Dataset Structure:
    dataset_name/
    ├── data/
    │   └── chunk-{chunk:03d}/
    │       └── episode_{episode:06d}.parquet
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── episodes_stats.jsonl
    │   └── tasks.jsonl
    └── videos/
        └── chunk-{chunk:03d}/
            └── observation.images.{camera}/
                └── episode_{episode:06d}.mp4
"""

import json
import shutil
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

# Re-export the shared dataclasses / worker so existing callers
# (pipeline_worker, scripts, tests) keep `from .to_lerobot_v21 import …`
# working unchanged.
from .base_converter import (  # noqa: F401
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FPS,
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
    StalenessMetrics,
    _convert_rosbag_worker,
)


CODEBASE_VERSION = "v2.1"


class RosbagToLerobotConverter(RosbagToLerobotConverterBase):
    """LeRobot v2.1 converter: per-episode parquet + JSONL meta."""

    def convert_multiple_rosbags(
        self,
        bag_paths: List[Path],
    ) -> bool:
        """
        Convert multiple ROSbag recordings to a single LeRobot dataset.

        Uses ProcessPoolExecutor for parallel episode parsing when multiple
        bag_paths are provided. Each worker creates its own converter instance.

        Args:
            bag_paths: List of paths to ROSbag directories

        Returns:
            True if successful, False otherwise
        """
        self._log_info(f"Converting {len(bag_paths)} rosbags to LeRobot dataset")

        # Initialize output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes_data: List[EpisodeData] = []

        # Build a picklable config dict for worker processes
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
        }

        if len(bag_paths) <= 1:
            # Single episode: no parallelization overhead
            for idx, bag_path in enumerate(bag_paths):
                episode_data = self.convert_single_rosbag(Path(bag_path), idx)
                if episode_data is not None:
                    episodes_data.append(episode_data)
        else:
            # Parallel episode parsing using ProcessPoolExecutor
            import os
            from concurrent.futures import ProcessPoolExecutor, as_completed

            max_workers = min(os.cpu_count() or 4, len(bag_paths), 8)
            self._log_info(
                f"Starting parallel rosbag parsing with {max_workers} workers"
            )

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

            # Sort by episode_index to maintain deterministic order
            episodes_data.sort(key=lambda ep: ep.episode_index)

        if not episodes_data:
            self._log_error("No episodes were successfully converted")
            return False

        self._build_features(episodes_data)
        self._write_dataset(episodes_data)

        self._log_info(f"Successfully converted {len(episodes_data)} episodes")
        return True

    def _write_dataset(self, episodes_data: List[EpisodeData]):
        """Write all dataset files to output directory."""
        output_dir = Path(self.config.output_dir)

        # Create directory structure
        (output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

        # Collect tasks in episode-first-appearance order (shared with v3.0).
        self._collect_tasks(episodes_data)

        # Write episodes
        for episode_data in episodes_data:
            self._write_episode(episode_data)

        # Write metadata files
        self._write_info_json()
        self._write_tasks_jsonl()
        self._write_root_info_json()

    def _write_episode(self, episode: EpisodeData):
        """Write a single episode's data files."""
        output_dir = Path(self.config.output_dir)
        ep_idx = episode.episode_index
        chunk_idx = ep_idx // self.config.chunks_size

        # Create chunk directories
        data_chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)

        video_chunk_dir = output_dir / "videos" / f"chunk-{chunk_idx:03d}"
        video_chunk_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet file
        parquet_path = data_chunk_dir / f"episode_{ep_idx:06d}.parquet"
        self._write_parquet(episode, parquet_path)

        # Copy video files
        for camera_name, src_video in episode.video_files.items():
            video_dir = video_chunk_dir / f"observation.images.{camera_name}"
            video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = video_dir / f"episode_{ep_idx:06d}.mp4"
            shutil.copy2(src_video, dst_video)
            self._log_info(f"Copied video: {src_video.name} -> {dst_video}")

        # Write episode metadata
        episode_dict = {
            "episode_index": ep_idx,
            "tasks": episode.tasks,
            "length": episode.length,
        }
        self._episodes[ep_idx] = episode_dict
        self._append_jsonl(episode_dict, output_dir / "meta" / "episodes.jsonl")

        # Compute and write episode stats. We pass the global starting
        # index (== total frames across previously written episodes) so
        # the stats for the synthetic ``index`` column are accurate.
        ep_stats = self._compute_episode_stats(
            episode, global_start_index=self._total_frames,
        )
        self._episodes_stats[ep_idx] = ep_stats
        stats_entry = {
            "episode_index": ep_idx,
            "stats": self._serialize_stats(ep_stats),
        }
        self._append_jsonl(stats_entry, output_dir / "meta" / "episodes_stats.jsonl")

        # Update totals
        self._total_frames += episode.length
        self._total_episodes += 1

    def _write_parquet(self, episode: EpisodeData, parquet_path: Path):
        """Write episode data to parquet file with HuggingFace-compatible schema."""
        num_frames = episode.length

        # Determine dimensions
        state_dim = (
            len(episode.observation_state[0]) if episode.observation_state else 0
        )
        action_dim = len(episode.action[0]) if episode.action else 0

        # Build schema with fixed_size_list for HuggingFace compatibility
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

        # Build data arrays with explicit types
        arrays = [
            pa.array(
                [float(episode.timestamps[i]) for i in range(num_frames)],
                type=pa.float32(),
            ),
            pa.array(list(range(num_frames)), type=pa.int64()),
            pa.array([episode.episode_index] * num_frames, type=pa.int64()),
            pa.array(
                list(range(self._total_frames, self._total_frames + num_frames)),
                type=pa.int64(),
            ),
        ]

        # Task index
        default_task = episode.tasks[0] if episode.tasks else "default_task"
        task_idx = self._task_to_index.get(default_task, 0)
        arrays.append(pa.array([task_idx] * num_frames, type=pa.int64()))

        # Add observation.state as fixed_size_list
        if episode.observation_state:
            state_values = [
                [float(v) for v in state] for state in episode.observation_state
            ]
            arrays.append(
                pa.array(state_values, type=pa.list_(pa.float32(), state_dim))
            )

        # Add action as fixed_size_list
        if episode.action:
            action_values = [[float(v) for v in action] for action in episode.action]
            arrays.append(
                pa.array(action_values, type=pa.list_(pa.float32(), action_dim))
            )

        # Build HuggingFace metadata
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

        # Add metadata to schema
        schema = schema.with_metadata({"huggingface": hf_metadata})

        # Create table with schema
        table = pa.table(
            dict(zip([f.name for f in schema_fields], arrays)), schema=schema
        )
        pq.write_table(table, parquet_path)
        self._log_info(f"Wrote parquet: {parquet_path}")

    def _write_info_json(self):
        """Write info.json metadata file."""
        output_dir = Path(self.config.output_dir)

        num_video_keys = sum(
            1 for k in self._features if k.startswith("observation.images.")
        )

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.config.robot_type,
            "total_episodes": self._total_episodes,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": self._total_episodes * num_video_keys,
            "total_chunks": (self._total_episodes // self.config.chunks_size) + 1,
            "chunks_size": self.config.chunks_size,
            "fps": self.config.fps,
            "splits": {"train": f"0:{self._total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if self.config.use_videos
            else None,
            "features": self._features,
        }

        info_path = output_dir / "meta" / "info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        self._log_info(f"Wrote info.json: {info_path}")

    def _write_tasks_jsonl(self):
        """Write tasks.jsonl metadata file."""
        output_dir = Path(self.config.output_dir)
        tasks_path = output_dir / "meta" / "tasks.jsonl"

        with open(tasks_path, "w", encoding="utf-8") as f:
            for task_idx, task in self._tasks.items():
                entry = {"task_index": task_idx, "task": task}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._log_info(f"Wrote tasks.jsonl: {tasks_path}")

    def _append_jsonl(self, data: dict, filepath: Path):
        """Append a single entry to a JSONL file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def convert_rosbags_to_lerobot(
    bag_paths: List[str],
    output_dir: str,
    repo_id: str,
    fps: int = DEFAULT_FPS,
    robot_type: str = "unknown",
    logger=None,
) -> bool:
    """
    Convenience function to convert multiple ROSbags to LeRobot dataset.

    Args:
        bag_paths: List of paths to ROSbag directories
        output_dir: Output directory for the dataset
        repo_id: Repository ID for the dataset (e.g., "user/dataset_name")
        fps: Target frames per second
        robot_type: Robot type identifier
        logger: Optional logger instance

    Returns:
        True if successful, False otherwise

    Example:
        >>> convert_rosbags_to_lerobot(
        ...     bag_paths=["/data/rosbag_001", "/data/rosbag_002"],
        ...     output_dir="/datasets/my_robot_dataset",
        ...     repo_id="robotis/ai_worker_pick_place",
        ...     fps=30,
        ...     robot_type="ai_worker",
        ... )
    """
    config = ConversionConfig(
        repo_id=repo_id,
        output_dir=Path(output_dir),
        fps=fps,
        robot_type=robot_type,
    )

    converter = RosbagToLerobotConverter(config, logger)
    return converter.convert_multiple_rosbags([Path(p) for p in bag_paths])
