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

"""Replay data handler for ROSbag visualization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

# reader/ owns metadata + video metadata under Step 3's 7-way layout —
# recorder/ pulls them in cross-module rather than local relative.
from cyclo_data.reader.metadata_manager import MetadataManager
from cyclo_data.reader.video_metadata_extractor import VideoMetadataExtractor

# NOTE: rosbag_recorder.msg.ImageMetadata + the ``has_raw_images`` /
# MCAP-direct-streaming path it powered are recording format v1
# artefacts. Recording format v2 keeps images out of MCAP entirely
# (they live next to it as per-camera MP4 + Parquet sidecar), so the
# importer is gone.


class ReplayDataHandler:
    """Handler for extracting replay data from ROSbag files."""

    def __init__(self, logger=None):
        self.logger = logger
        self._metadata_manager = MetadataManager(logger)
        self._video_extractor = VideoMetadataExtractor(logger)

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)

    def _load_metadata(self, bag_path: Path) -> Optional[Dict]:
        return self._metadata_manager.load_robot_config(bag_path)

    def _get_recording_date(self, bag_path: Path) -> Optional[str]:
        return self._metadata_manager.get_recording_date(bag_path)

    def _get_directory_size(self, bag_path: Path) -> int:
        return self._metadata_manager.get_directory_size(bag_path)

    def _get_task_markers(self, bag_path: Path) -> List[Dict]:
        return self._metadata_manager.get_task_markers(bag_path)

    def _get_trim_points(self, bag_path: Path) -> Optional[Dict]:
        return self._metadata_manager.get_trim_points(bag_path)

    def _get_exclude_regions(self, bag_path: Path) -> List[Dict]:
        return self._metadata_manager.get_exclude_regions(bag_path)

    def _extract_camera_name_from_topic(self, topic: str) -> str:
        return self._video_extractor.extract_camera_name_from_topic(topic)

    def _get_action_topic_order(self, metadata: Optional[Dict]) -> List[str]:
        if metadata and "action_topics" in metadata:
            action_topics = metadata["action_topics"]
            return [action_topics[k] for k in sorted(action_topics.keys())]

        return [
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            "/leader/joystick_controller_left/joint_trajectory",
            "/leader/joystick_controller_right/joint_trajectory",
        ]

    def _is_action_topic(self, topic: str, metadata: Optional[Dict]) -> bool:
        if metadata and "action_topics" in metadata:
            action_topic_paths = list(metadata["action_topics"].values())
            return topic in action_topic_paths
        return "action" in topic.lower() or "leader" in topic.lower()

    def update_task_markers(
        self,
        bag_path: str,
        task_markers: List[Dict],
        trim_points: Optional[Dict] = None,
        exclude_regions: Optional[List[Dict]] = None,
    ) -> Dict:
        return self._metadata_manager.update_task_markers(
            Path(bag_path), task_markers, trim_points, exclude_regions
        )

    def get_replay_data(self, bag_path: str) -> Dict:
        """
        Extract replay data from a ROSbag directory.

        Args:
            bag_path: Path to the ROSbag directory

        Returns:
            Dictionary containing video info, frame timestamps, and joint data
        """
        result = {
            "success": False,
            "message": "",
            "video_files": [],
            "video_topics": [],
            "video_names": [],  # Human-readable camera names from config
            "video_fps": [],
            "frame_indices": [],
            "frame_timestamps": [],
            "joint_timestamps": [],
            "joint_names": [],
            "joint_positions": [],
            "action_timestamps": [],
            "action_names": [],
            "action_values": [],
            "start_time": 0.0,
            "end_time": 0.0,
            "duration": 0.0,
            "robot_type": "",
            "metadata": None,
            # Extended metadata fields
            "recording_date": None,
            "file_size_bytes": 0,
            "task_markers": [],
            "trim_points": None,
            "exclude_regions": [],
            "frame_counts": {},
            # Recording format v2 transcode state — UI gates playback on
            # this so a half-transcoded MJPEG MP4 never reaches an
            # HTML5 <video> tag (Chromium can't decode MJPEG).
            "transcoding_status": "done",  # legacy episodes default to ready
            "transcoding_cameras_failed": {},
        }

        # Validate bag path
        bag_path_obj = Path(bag_path)
        if not bag_path_obj.exists():
            result["message"] = f"Bag path does not exist: {bag_path}"
            return result

        # Load metadata if available
        metadata = self._load_metadata(bag_path_obj)
        if metadata:
            result["metadata"] = metadata
            result["robot_type"] = metadata.get("robot_type", "")

        # Get extended metadata
        result["recording_date"] = self._get_recording_date(bag_path_obj)
        result["file_size_bytes"] = self._get_directory_size(bag_path_obj)
        result["task_markers"] = self._get_task_markers(bag_path_obj)
        result["trim_points"] = self._get_trim_points(bag_path_obj)
        result["exclude_regions"] = self._get_exclude_regions(bag_path_obj)

        # Find MCAP file
        mcap_files = list(bag_path_obj.glob("*.mcap"))
        if not mcap_files:
            # Try db3 format
            mcap_files = list(bag_path_obj.glob("*.db3"))

        if not mcap_files:
            result["message"] = f"No bag file found in: {bag_path}"
            return result

        # Surface recording format v2 transcode state so the UI can
        # gate playback (raw MJPEG MP4 doesn't play in Chromium).
        info_path = bag_path_obj / "episode_info.json"
        if info_path.exists():
            try:
                import json as _json
                with open(info_path) as f:
                    info = _json.load(f) or {}
                # If the episode predates the transcoder ("no field"),
                # treat as ready — its source MP4 (if any) is whatever
                # the legacy pipeline produced.
                result["transcoding_status"] = str(
                    info.get("transcoding_status", "done")
                )
                result["transcoding_cameras_failed"] = dict(
                    info.get("transcoding_cameras_failed") or {}
                )
            except Exception as exc:
                self._log_error(f"replay: failed to read {info_path.name}: {exc!r}")

        try:
            # Read bag file
            reader = SequentialReader()
            storage_options = StorageOptions(
                uri=str(bag_path),
                storage_id="mcap" if mcap_files[0].suffix == ".mcap" else "sqlite3",
            )
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader.open(storage_options, converter_options)

            # Get topic information
            topic_types = reader.get_all_topics_and_types()
            topic_type_map = {t.name: t.type for t in topic_types}
            self._log_info(f"MCAP topics: {topic_type_map}")

            # Recording format v2 MCAP carries state + action + /tf
            # only (no CompressedImage, no CameraInfo). The legacy v1
            # bag may still have those, so the storage filter below is
            # defensive — it just keeps the topics we care about and
            # ignores everything else.
            skip_types = {"CompressedImage", "CameraInfo"}
            read_topics = [
                topic
                for topic, ttype in topic_type_map.items()
                if not any(s in ttype for s in skip_types)
            ]
            if read_topics:
                storage_filter = StorageFilter(topics=read_topics)
                reader.set_filter(storage_filter)

            # Collect data
            image_metadata_by_topic: Dict[str, List[Tuple[int, float]]] = {}
            # Store state data per topic for proper merging
            # (multiple JointState topics have different joint counts)
            state_data_by_topic: Dict[
                str, List[Tuple[float, List[str], List[float]]]
            ] = {}
            # Store action data per topic for proper merging
            action_data_by_topic: Dict[
                str, List[Tuple[float, List[str], List[float]]]
            ] = {}
            min_time = float("inf")
            max_time = float("-inf")

            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                timestamp_sec = timestamp / 1e9

                min_time = min(min_time, timestamp_sec)
                max_time = max(max_time, timestamp_sec)

                topic_type = topic_type_map.get(topic, "")

                # Handle JointState messages - check if it's action or state
                if topic_type == "sensor_msgs/msg/JointState":
                    try:
                        msg = deserialize_message(data, JointState)
                        if self._is_action_topic(topic, metadata):
                            if topic not in action_data_by_topic:
                                action_data_by_topic[topic] = []
                            action_data_by_topic[topic].append(
                                (timestamp_sec, list(msg.name), list(msg.position))
                            )
                        else:
                            if topic not in state_data_by_topic:
                                state_data_by_topic[topic] = []
                            state_data_by_topic[topic].append(
                                (timestamp_sec, list(msg.name), list(msg.position))
                            )
                    except Exception as e:
                        self._log_error(f"Failed to deserialize JointState: {e}")

                # Handle JointTrajectory messages (leader topics are action)
                elif topic_type == "trajectory_msgs/msg/JointTrajectory":
                    try:
                        msg = deserialize_message(data, JointTrajectory)
                        # Leader topics are action data - store per topic
                        if msg.points and len(msg.points) > 0:
                            if topic not in action_data_by_topic:
                                action_data_by_topic[topic] = []
                            action_data_by_topic[topic].append(
                                (
                                    timestamp_sec,
                                    list(msg.joint_names),
                                    list(msg.points[0].positions),
                                )
                            )
                    except Exception as e:
                        self._log_error(f"Failed to deserialize JointTrajectory: {e}")

            # Build camera name mapping from metadata
            camera_name_map = {}  # topic -> camera_name
            if metadata and "camera_topics" in metadata:
                for cam_name, topic_path in metadata["camera_topics"].items():
                    camera_name_map[topic_path] = cam_name

            # Recording format v2: every recorded camera has a Parquet
            # sidecar with per-frame ``header_stamp_ns`` / ``recv_ns`` so
            # we can populate ``image_metadata_by_topic`` directly without
            # touching the (now-absent) MCAP ImageMetadata channel.
            videos_dir = bag_path_obj / "videos"
            try:
                from cyclo_data.reader.frame_timestamps import (
                    load_frame_timestamps,
                )
            except Exception:
                load_frame_timestamps = None  # type: ignore[assignment]
            sidecar_by_camera: Dict[str, Any] = {}
            if videos_dir.exists() and load_frame_timestamps is not None:
                for sidecar in sorted(
                    videos_dir.glob("*_timestamps.parquet")
                ):
                    cam_name = sidecar.stem[: -len("_timestamps")]
                    try:
                        sidecar_by_camera[cam_name] = load_frame_timestamps(
                            sidecar, cam_name
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        self._log_error(
                            f"replay: failed to read {sidecar.name}: {exc!r}"
                        )

            # Process video files
            if videos_dir.exists():
                for video_file in sorted(videos_dir.glob("*.mp4")):
                    if video_file.stem.endswith("_synced"):
                        # Skip converter-side derivatives that happen to
                        # live in the same dir.
                        continue
                    result["video_files"].append(f"videos/{video_file.name}")

                    # Find matching topic
                    video_name = video_file.stem
                    matching_topic = None
                    if video_name in sidecar_by_camera:
                        # v2 path: per-camera sidecar is the source of
                        # truth. Fold its rows into image_metadata_by_topic
                        # keyed by the video filename (no source topic in
                        # the v2 MCAP) so the rest of this method works
                        # unchanged.
                        ft = sidecar_by_camera[video_name]
                        # Use header.stamp seconds (publisher clock) so
                        # the timeline lines up with anything that
                        # publishes header.stamp in the MCAP.
                        per_frame = [
                            (int(idx), stamp / 1e9)
                            for idx, stamp in zip(
                                ft.frame_index, ft.header_stamp_ns
                            )
                        ]
                        if per_frame:
                            image_metadata_by_topic[video_name] = per_frame
                            matching_topic = video_name
                            if per_frame[0][1] > 0:
                                min_time = min(min_time, per_frame[0][1])
                            if per_frame[-1][1] > 0:
                                max_time = max(max_time, per_frame[-1][1])
                    if matching_topic is None:
                        for topic in image_metadata_by_topic.keys():
                            sanitized = topic.replace("/", "_").lstrip("_")
                            if sanitized == video_name:
                                matching_topic = topic
                                break

                    result["video_topics"].append(matching_topic or video_name)

                    # Find camera name from metadata or extract from topic/filename
                    camera_name = None
                    if matching_topic:
                        camera_name = camera_name_map.get(matching_topic)
                        if not camera_name:
                            # Fallback: extract meaningful name from topic path
                            camera_name = self._extract_camera_name_from_topic(
                                matching_topic
                            )
                    else:
                        # No matching topic, try to extract from video filename
                        camera_name = self._extract_camera_name_from_topic(video_name)
                    result["video_names"].append(camera_name or video_name)

                    # Calculate FPS from metadata
                    if matching_topic and matching_topic in image_metadata_by_topic:
                        metadata = image_metadata_by_topic[matching_topic]
                        if len(metadata) > 1:
                            # Sort by frame index
                            metadata.sort(key=lambda x: x[0])
                            time_diffs = [
                                metadata[i + 1][1] - metadata[i][1]
                                for i in range(len(metadata) - 1)
                            ]
                            avg_diff = sum(time_diffs) / len(time_diffs)
                            fps = 1.0 / avg_diff if avg_diff > 0 else 30.0
                            result["video_fps"].append(fps)
                        else:
                            result["video_fps"].append(30.0)
                    else:
                        result["video_fps"].append(30.0)

            # Process frame timestamps (use first video topic)
            if image_metadata_by_topic:
                first_topic = list(image_metadata_by_topic.keys())[0]
                metadata_list = sorted(
                    image_metadata_by_topic[first_topic], key=lambda x: x[0]
                )
                for frame_idx, timestamp in metadata_list:
                    result["frame_indices"].append(frame_idx)
                    result["frame_timestamps"].append(timestamp - min_time)

            # Build frame counts per camera
            for i, video_name in enumerate(result["video_names"]):
                topic = (
                    result["video_topics"][i]
                    if i < len(result["video_topics"])
                    else None
                )
                if topic and topic in image_metadata_by_topic:
                    result["frame_counts"][video_name] = len(
                        image_metadata_by_topic[topic]
                    )
                else:
                    result["frame_counts"][video_name] = 0

            # Process joint (state) data — per-topic merge
            # Multiple JointState topics have different joint counts,
            # so we must merge them by timestamp (same logic as action data).
            if state_data_by_topic:
                # Auto-detect topic order (sorted for consistency)
                state_topic_order = sorted(state_data_by_topic.keys())

                # Collect all joint names in topic order
                all_joint_names: List[str] = []
                state_topic_joint_names: Dict[str, List[str]] = {}
                for topic in state_topic_order:
                    if state_data_by_topic[topic]:
                        names = state_data_by_topic[topic][0][1]
                        state_topic_joint_names[topic] = names
                        all_joint_names.extend(names)

                result["joint_names"] = all_joint_names

                # Group data by approximate timestamp (within 10ms)
                state_timestamp_data: Dict[float, Dict] = {}
                for topic in state_topic_order:
                    for ts, names, values in state_data_by_topic[topic]:
                        ts_key = round(ts * 100) / 100
                        if ts_key not in state_timestamp_data:
                            state_timestamp_data[ts_key] = {}
                        state_timestamp_data[ts_key][topic] = (names, values)

                # Build flat array with forward fill for missing topics
                last_state_values: Dict[str, List[float]] = {}
                for ts_key in sorted(state_timestamp_data.keys()):
                    result["joint_timestamps"].append(ts_key - min_time)
                    for topic in state_topic_order:
                        if topic in state_timestamp_data[ts_key]:
                            _, values = state_timestamp_data[ts_key][topic]
                            result["joint_positions"].extend(values)
                            last_state_values[topic] = values
                        elif topic in state_topic_joint_names:
                            if topic in last_state_values:
                                result["joint_positions"].extend(
                                    last_state_values[topic]
                                )
                            else:
                                result["joint_positions"].extend(
                                    [0.0]
                                    * len(state_topic_joint_names[topic])
                                )

            # Process action data from all topics
            if action_data_by_topic:
                # Get topic order from metadata or use default
                topic_order = self._get_action_topic_order(metadata)

                # Also include any topics not in the predefined order
                for topic in action_data_by_topic.keys():
                    if topic not in topic_order:
                        topic_order.append(topic)

                # Collect all action names in order
                all_action_names = []
                topic_joint_names = {}
                for topic in topic_order:
                    if topic in action_data_by_topic and action_data_by_topic[topic]:
                        names = action_data_by_topic[topic][0][1]
                        topic_joint_names[topic] = names
                        all_action_names.extend(names)

                result["action_names"] = all_action_names

                # Merge action data by timestamp
                # Group data by approximate timestamp (within 10ms)
                timestamp_data = {}
                for topic in topic_order:
                    if topic not in action_data_by_topic:
                        continue
                    for ts, names, values in action_data_by_topic[topic]:
                        # Round to 10ms for grouping
                        ts_key = round(ts * 100) / 100
                        if ts_key not in timestamp_data:
                            timestamp_data[ts_key] = {}
                        timestamp_data[ts_key][topic] = (names, values)

                # Build action values in correct order (with forward fill for missing data)
                last_values_by_topic = {}  # Store last known values for each topic
                for ts_key in sorted(timestamp_data.keys()):
                    result["action_timestamps"].append(ts_key - min_time)
                    for topic in topic_order:
                        if topic in timestamp_data[ts_key]:
                            _, values = timestamp_data[ts_key][topic]
                            result["action_values"].extend(values)
                            last_values_by_topic[topic] = values  # Remember last values
                        elif topic in topic_joint_names:
                            # Forward fill: use last known values instead of zeros
                            if topic in last_values_by_topic:
                                result["action_values"].extend(
                                    last_values_by_topic[topic]
                                )
                            else:
                                # No previous data yet, use zeros as fallback
                                result["action_values"].extend(
                                    [0.0] * len(topic_joint_names[topic])
                                )

            # Set duration info
            if min_time != float("inf") and max_time != float("-inf"):
                result["start_time"] = 0.0
                result["end_time"] = max_time - min_time
                result["duration"] = max_time - min_time

            result["success"] = True
            result["message"] = "Replay data loaded successfully"
            self._log_info(
                f"Loaded replay data: {len(result['video_files'])} videos, "
                f"{len(result['frame_timestamps'])} frames, "
                f"{len(result['joint_timestamps'])} joint samples, "
                f"{len(result['action_timestamps'])} action samples"
            )

        except Exception as e:
            result["message"] = f"Failed to read bag file: {str(e)}"
            self._log_error(result["message"])

        return result

    def get_video_file_path(self, bag_path: str, video_file: str) -> Optional[str]:
        """
        Get the full path to a video file.

        Args:
            bag_path: Path to the ROSbag directory
            video_file: Relative path to the video file

        Returns:
            Full path to the video file or None if not found
        """
        full_path = Path(bag_path) / video_file
        if full_path.exists():
            return str(full_path)
        return None

    def get_rosbag_list(self, folder_path: str) -> Dict:
        """
        Get list of ROSbag directories in a folder.

        A directory is considered a ROSbag if it contains:
        - metadata.yaml file (rosbag2 metadata)
        - .mcap or .db3 file

        Args:
            folder_path: Path to the parent folder

        Returns:
            Dictionary with rosbag list info
        """
        result = {
            "success": False,
            "message": "",
            "rosbags": [],
            "parent_path": "",
        }

        folder = Path(folder_path)
        if not folder.exists():
            result["message"] = f"Folder not found: {folder_path}"
            return result

        if not folder.is_dir():
            result["message"] = f"Path is not a directory: {folder_path}"
            return result

        result["parent_path"] = str(folder)
        rosbags = []

        # Check all subdirectories
        for item in sorted(folder.iterdir()):
            if not item.is_dir():
                continue

            # Check if it's a valid ROSbag directory
            has_metadata = (item / "metadata.yaml").exists()
            has_mcap = any(item.glob("*.mcap"))
            has_db3 = any(item.glob("*.db3"))
            has_videos = (item / "videos").exists()

            if has_metadata and (has_mcap or has_db3):
                bag_info = {
                    "name": item.name,
                    "path": str(item),
                    "has_videos": has_videos,
                }

                # Try to get duration from metadata
                try:
                    metadata_path = item / "metadata.yaml"
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = yaml.safe_load(f)
                        if metadata and "rosbag2_bagfile_information" in metadata:
                            bag_info["duration_ns"] = (
                                metadata["rosbag2_bagfile_information"]
                                .get("duration", {})
                                .get("nanoseconds", 0)
                            )
                except Exception:
                    pass

                rosbags.append(bag_info)

        result["rosbags"] = rosbags
        result["success"] = True
        result["message"] = f"Found {len(rosbags)} ROSbag(s)"
        self._log_info(result["message"])

        return result
