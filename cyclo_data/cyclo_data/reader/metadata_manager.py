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

"""Metadata manager for ROSbag robot_config.yaml files."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class MetadataManager:
    """
    Manager for ROSbag metadata files.

    Handles reading and writing of robot_config.yaml and metadata.yaml files.
    """

    def __init__(self, logger=None):
        """
        Initialize MetadataManager.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)

    def load_robot_config(self, bag_path: Path) -> Optional[Dict]:
        """
        Load robot_config.yaml from rosbag directory.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            Config dictionary or None if not found
        """
        config_path = Path(bag_path) / "robot_config.yaml"
        if not config_path.exists():
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self._log_info(f"Loaded robot config from: {config_path}")
            return config
        except Exception as e:
            self._log_error(f"Failed to load robot config: {e}")
            return None

    def save_robot_config(self, bag_path: Path, config: Dict) -> bool:
        """
        Save robot_config.yaml to rosbag directory.

        Args:
            bag_path: Path to the ROSbag directory
            config: Configuration dictionary to save

        Returns:
            True if successful, False otherwise
        """
        config_path = Path(bag_path) / "robot_config.yaml"

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            self._log_info(f"Saved robot config to: {config_path}")
            return True
        except Exception as e:
            self._log_error(f"Failed to save robot config: {e}")
            return False

    def get_recording_date(self, bag_path: Path) -> Optional[str]:
        """
        Get recording date from metadata.yaml.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            ISO format datetime string or None
        """
        metadata_path = Path(bag_path) / "metadata.yaml"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)

            if metadata and "rosbag2_bagfile_information" in metadata:
                bag_info = metadata["rosbag2_bagfile_information"]
                starting_time = bag_info.get("starting_time", {})
                ns_since_epoch = starting_time.get("nanoseconds_since_epoch", 0)
                if ns_since_epoch > 0:
                    dt = datetime.fromtimestamp(ns_since_epoch / 1e9, tz=timezone.utc)
                    return dt.isoformat().replace("+00:00", "Z")
        except Exception as e:
            self._log_error(f"Failed to get recording date: {e}")

        return None

    def get_directory_size(self, bag_path: Path) -> int:
        """
        Get total size of bag directory in bytes.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            bag_path = Path(bag_path)
            for item in bag_path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            self._log_error(f"Failed to get directory size: {e}")
        return total_size

    def get_task_markers(self, bag_path: Path) -> List[Dict]:
        """
        Get task markers from robot_config.yaml.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            List of task marker dictionaries
        """
        config = self.load_robot_config(bag_path)
        if config is None:
            return []
        return config.get("task_markers", [])

    def get_trim_points(self, bag_path: Path) -> Optional[Dict]:
        """
        Get trim points from robot_config.yaml.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            Trim points dictionary or None
        """
        config = self.load_robot_config(bag_path)
        if config is None:
            return None
        return config.get("trim_points", None)

    def get_exclude_regions(self, bag_path: Path) -> List[Dict]:
        """
        Get exclude regions from robot_config.yaml.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            List of exclude region dictionaries
        """
        config = self.load_robot_config(bag_path)
        if config is None:
            return []
        return config.get("exclude_regions", [])

    def load_episode_info(self, bag_path: Path) -> Dict:
        """Load episode_info.json from a rosbag episode directory."""
        info_path = Path(bag_path) / "episode_info.json"
        if not info_path.exists():
            return {}

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self._log_error(f"Failed to load episode_info.json: {e}")
            return {}

    def get_episode_segments(
        self,
        bag_path: Path,
        duration: Optional[float] = None,
    ) -> List[Dict]:
        """Return episode segments with frame_duration in relative seconds."""
        info = self.load_episode_info(bag_path)
        raw_segments = info.get("segments", [])
        if not isinstance(raw_segments, list):
            return []

        unit = str(info.get("segment_time_unit", "") or "").lower()
        fps = self._coerce_positive_float(info.get("fps"))
        convert_frames = unit in {"frame", "frames"}

        if unit not in {"second", "seconds", "sec", "secs"} and not convert_frames:
            convert_frames = self._segments_look_frame_based(
                raw_segments, fps=fps, duration=duration
            )

        segments = []
        for idx, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                continue
            duration_pair = segment.get("frame_duration")
            if (
                not isinstance(duration_pair, list)
                or len(duration_pair) != 2
            ):
                continue

            start = self._coerce_float(duration_pair[0])
            end = self._coerce_float(duration_pair[1])
            if start is None or end is None:
                continue
            if convert_frames and fps:
                start /= fps
                end /= fps
            if end < start:
                continue

            normalized = {
                "sub_task_instruction": str(
                    segment.get("sub_task_instruction", "") or ""
                ),
                "frame_duration": [float(start), float(end)],
            }
            segments.append(normalized)

        segments.sort(key=lambda s: s["frame_duration"][0])
        return segments

    def update_episode_segments(self, bag_path: Path, segments: List[Dict]) -> bool:
        """Write timestamp-based segments into episode_info.json."""
        bag_path = Path(bag_path)
        info_path = bag_path / "episode_info.json"
        info = self.load_episode_info(bag_path)

        normalized = []
        for idx, segment in enumerate(segments or []):
            if not isinstance(segment, dict):
                continue
            duration_pair = segment.get("frame_duration")
            if (
                not isinstance(duration_pair, list)
                or len(duration_pair) != 2
            ):
                continue
            start = self._coerce_float(duration_pair[0])
            end = self._coerce_float(duration_pair[1])
            if start is None or end is None or end < start:
                continue
            normalized.append({
                "sub_task_instruction": str(
                    segment.get("sub_task_instruction", "") or ""
                ),
                "frame_duration": [float(start), float(end)],
            })

        normalized.sort(key=lambda s: s["frame_duration"][0])

        info["segments"] = normalized
        info["segment_time_unit"] = "seconds"
        info["segment_time_reference"] = "episode_start"

        try:
            info_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = info_path.with_suffix(info_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
                f.write("\n")
            tmp_path.replace(info_path)
            self._log_info(f"Saved episode segments to: {info_path}")
            return True
        except Exception as e:
            self._log_error(f"Failed to save episode segments: {e}")
            return False

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _coerce_positive_float(cls, value: Any) -> Optional[float]:
        coerced = cls._coerce_float(value)
        if coerced is None or coerced <= 0:
            return None
        return coerced

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _segments_look_frame_based(
        cls,
        segments: List[Dict],
        fps: Optional[float],
        duration: Optional[float],
    ) -> bool:
        if not fps:
            return False

        values = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            duration_pair = segment.get("frame_duration")
            if not isinstance(duration_pair, list) or len(duration_pair) != 2:
                continue
            for value in duration_pair:
                coerced = cls._coerce_float(value)
                if coerced is not None:
                    values.append(coerced)

        if not values:
            return False
        if any(abs(value - round(value)) > 1e-6 for value in values):
            return False

        max_value = max(values)
        if duration is not None and duration > 0:
            return max_value > duration + 1.0
        return max_value > 300

    def update_task_markers(
        self,
        bag_path: Path,
        task_markers: List[Dict],
        trim_points: Optional[Dict] = None,
        exclude_regions: Optional[List[Dict]] = None,
        segments: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Update task markers, trim points, and exclude regions.

        Args:
            bag_path: Path to the ROSbag directory
            task_markers: List of task marker dictionaries
            trim_points: Optional dict with 'start' and 'end' trim point info
            exclude_regions: Optional list of exclude region dicts
            segments: Optional episode_info.json segments in relative seconds

        Returns:
            Result dictionary with success status and message
        """
        result = {"success": False, "message": ""}
        bag_path = Path(bag_path)
        config_path = bag_path / "robot_config.yaml"

        saved_config = False
        if config_path.exists():
            config = self.load_robot_config(bag_path) or {}

            # Sort markers by frame and update legacy replay annotations.
            sorted_markers = sorted(task_markers, key=lambda m: m.get("frame", 0))
            config["task_markers"] = sorted_markers

            # Update trim points
            if trim_points is not None:
                if trim_points:
                    config["trim_points"] = trim_points
                elif "trim_points" in config:
                    del config["trim_points"]

            # Update exclude regions
            if exclude_regions is not None:
                if exclude_regions:
                    sorted_regions = sorted(
                        exclude_regions,
                        key=lambda r: r.get("start", {}).get("time", 0),
                    )
                    config["exclude_regions"] = sorted_regions
                elif "exclude_regions" in config:
                    del config["exclude_regions"]

            if not self.save_robot_config(bag_path, config):
                result["message"] = "Failed to save config"
                return result
            saved_config = True

        if segments is not None and not self.update_episode_segments(
            bag_path, segments
        ):
            result["message"] = "Failed to save episode segments"
            return result

        result["success"] = True
        result["message"] = (
            f"Saved {len(task_markers)} task markers"
            if saved_config
            else "Skipped robot_config.yaml"
        )
        if trim_points:
            result["message"] += ", trim points"
        if exclude_regions:
            result["message"] += f", {len(exclude_regions)} exclude regions"
        if segments is not None:
            result["message"] += f", {len(segments)} segments"

        return result

    def get_camera_name_map(self, bag_path: Path) -> Dict[str, str]:
        """
        Get camera topic to name mapping from robot_config.yaml.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            Dictionary mapping topic paths to camera names
        """
        config = self.load_robot_config(bag_path)
        if config is None or "camera_topics" not in config:
            return {}

        camera_name_map = {}
        for cam_name, topic_path in config["camera_topics"].items():
            camera_name_map[topic_path] = cam_name

        return camera_name_map

    def get_action_topic_order(self, bag_path: Path) -> List[str]:
        """
        Get ordered list of action topics from robot_config.yaml.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            List of action topic paths in order
        """
        config = self.load_robot_config(bag_path)

        if config and "action_topics" in config:
            action_topics = config["action_topics"]
            return [action_topics[k] for k in sorted(action_topics.keys())]

        # Default order if no config available
        return [
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            "/leader/joystick_controller_left/joint_trajectory",
            "/leader/joystick_controller_right/joint_trajectory",
        ]

    def is_action_topic(self, topic: str, bag_path: Path) -> bool:
        """
        Check if a topic is an action topic.

        Args:
            topic: Topic name to check
            bag_path: Path to the ROSbag directory

        Returns:
            True if topic is an action topic
        """
        config = self.load_robot_config(bag_path)

        if config and "action_topics" in config:
            action_topic_paths = list(config["action_topics"].values())
            return topic in action_topic_paths

        # Fallback to name-based detection
        return "action" in topic.lower() or topic.startswith("/leader")
