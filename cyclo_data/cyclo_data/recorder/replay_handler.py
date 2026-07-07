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

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
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

    def _load_robot_semantic_layout(self, robot_type: str) -> Dict[str, Any]:
        if not robot_type:
            return {}

        candidates: List[Path] = []
        for env_var in ("ORCHESTRATOR_CONFIG_PATH", "ROBOT_CLIENT_CONFIG_DIR"):
            env_dir = os.environ.get(env_var)
            if env_dir:
                candidates.append(Path(env_dir) / f"{robot_type}_config.yaml")

        candidates.extend(
            [
                Path("/orchestrator_config") / f"{robot_type}_config.yaml",
                Path("/root/ros2_ws/src/cyclo_intelligence/shared/shared/robot_configs")
                / f"{robot_type}_config.yaml",
                Path("/root/ros2_ws/install/shared/share/shared/robot_configs")
                / f"{robot_type}_config.yaml",
            ]
        )

        for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
            if prefix:
                candidates.append(
                    Path(prefix)
                    / "share"
                    / "shared"
                    / "robot_configs"
                    / f"{robot_type}_config.yaml"
                )

        here = Path(__file__).resolve()
        for parent in here.parents:
            candidates.extend(
                [
                    parent
                    / "shared"
                    / "shared"
                    / "robot_configs"
                    / f"{robot_type}_config.yaml",
                    parent / "shared" / "robot_configs" / f"{robot_type}_config.yaml",
                ]
            )

        config_path = next((path for path in candidates if path.exists()), None)
        if config_path is None:
            self._log_error(
                f"replay: robot config for {robot_type} not found; searched "
                f"{[str(path) for path in candidates]}"
            )
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            section = raw["orchestrator"]["ros__parameters"][robot_type]
            observation = section.get("observation") or {}
            return {
                "state": observation.get("state") or {},
                "action": section.get("action") or {},
            }
        except Exception as exc:
            self._log_error(
                f"replay: failed to load robot config {config_path}: {exc!r}"
            )
            return {}

    def _topic_name_layout(
        self, groups: Optional[Dict[str, Dict[str, Any]]]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        topic_order: List[str] = []
        names_by_topic: Dict[str, List[str]] = {}

        for cfg in (groups or {}).values():
            topic = cfg.get("topic")
            names = list(cfg.get("joint_names") or [])
            if not topic or not names:
                continue
            if topic not in names_by_topic:
                topic_order.append(topic)
                names_by_topic[topic] = []
            existing = set(names_by_topic[topic])
            for name in names:
                if name not in existing:
                    names_by_topic[topic].append(name)
                    existing.add(name)

        return topic_order, names_by_topic

    def _topic_order(
        self,
        data_by_topic: Dict[str, List[Tuple[float, List[str], List[float]]]],
        configured_order: List[str],
    ) -> List[str]:
        topic_order = [topic for topic in configured_order if topic in data_by_topic]
        for topic in sorted(data_by_topic.keys()):
            if topic not in topic_order:
                topic_order.append(topic)
        return topic_order

    def _values_for_names(
        self,
        source_names: List[str],
        source_values: List[float],
        target_names: List[str],
    ) -> List[float]:
        if not target_names:
            return [float(value) for value in source_values]

        index_by_name = {name: idx for idx, name in enumerate(source_names)}
        values: List[float] = []
        for name in target_names:
            idx = index_by_name.get(name)
            if idx is None or idx >= len(source_values):
                values.append(0.0)
                continue
            try:
                values.append(float(source_values[idx]))
            except (TypeError, ValueError):
                values.append(0.0)
        return values

    def _twist_values_by_name(self, msg: Twist) -> Dict[str, float]:
        return {
            "linear_x": float(msg.linear.x),
            "linear_y": float(msg.linear.y),
            "linear_z": float(msg.linear.z),
            "angular_x": float(msg.angular.x),
            "angular_y": float(msg.angular.y),
            "angular_z": float(msg.angular.z),
        }

    def _odom_values_by_name(self, msg: Odometry) -> Dict[str, float]:
        twist = msg.twist.twist
        return {
            "linear_x": float(twist.linear.x),
            "linear_y": float(twist.linear.y),
            "linear_z": float(twist.linear.z),
            "angular_x": float(twist.angular.x),
            "angular_y": float(twist.angular.y),
            "angular_z": float(twist.angular.z),
        }

    def _synthetic_values_for_names(
        self, values_by_name: Dict[str, float], target_names: List[str]
    ) -> List[float]:
        return [float(values_by_name.get(name, 0.0)) for name in target_names]

    def update_task_markers(
        self,
        bag_path: str,
        task_markers: List[Dict],
        trim_points: Optional[Dict] = None,
        exclude_regions: Optional[List[Dict]] = None,
        segments: Optional[List[Dict]] = None,
    ) -> Dict:
        return self._metadata_manager.update_task_markers(
            Path(bag_path), task_markers, trim_points, exclude_regions, segments
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
            "segments": [],
            # Segment-aware video playback for recording format v2. Built
            # from existing episode_info.json + metadata.yaml + videos/<mcap-stem>/.
            "video_segments": [],
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

        episode_info = self._metadata_manager.load_episode_info(bag_path_obj)
        if not result["robot_type"] and episode_info.get("robot_type"):
            result["robot_type"] = str(episode_info.get("robot_type") or "")

        semantic_layout = self._load_robot_semantic_layout(result["robot_type"])
        state_topic_order_config, state_names_by_topic_config = (
            self._topic_name_layout(semantic_layout.get("state"))
        )
        action_topic_order_config, action_names_by_topic_config = (
            self._topic_name_layout(semantic_layout.get("action"))
        )

        # Get extended metadata
        result["recording_date"] = self._get_recording_date(bag_path_obj)
        result["file_size_bytes"] = self._get_directory_size(bag_path_obj)
        result["task_markers"] = self._get_task_markers(bag_path_obj)
        result["trim_points"] = self._get_trim_points(bag_path_obj)
        result["exclude_regions"] = self._get_exclude_regions(bag_path_obj)
        result["segments"] = self._metadata_manager.get_episode_segments(bag_path_obj)
        segment_time_map = self._build_segment_time_map(
            bag_path_obj, result["segments"]
        )

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
        if episode_info:
            try:
                # If the episode predates the transcoder ("no field"),
                # treat as ready — its source MP4 (if any) is whatever
                # the legacy pipeline produced.
                result["transcoding_status"] = str(
                    episode_info.get("transcoding_status", "done")
                )
                result["transcoding_cameras_failed"] = dict(
                    episode_info.get("transcoding_cameras_failed") or {}
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
            min_time = 0.0 if segment_time_map else float("inf")
            max_time = float("-inf")

            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                timestamp_sec = self._timestamp_to_replay_seconds(
                    int(timestamp), segment_time_map
                )
                if timestamp_sec is None:
                    continue

                min_time = min(min_time, timestamp_sec)
                max_time = max(max_time, timestamp_sec)

                topic_type = topic_type_map.get(topic, "")

                # Handle JointState messages - check if it's action or state
                if topic_type == "sensor_msgs/msg/JointState":
                    try:
                        msg = deserialize_message(data, JointState)
                        is_action = (
                            topic in action_names_by_topic_config
                            or (
                                topic not in state_names_by_topic_config
                                and self._is_action_topic(topic, metadata)
                            )
                        )
                        if is_action:
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

                # Handle Odometry state values from robot config.
                elif topic_type == "nav_msgs/msg/Odometry":
                    target_names = state_names_by_topic_config.get(topic)
                    if not target_names:
                        continue
                    try:
                        msg = deserialize_message(data, Odometry)
                        values = self._synthetic_values_for_names(
                            self._odom_values_by_name(msg),
                            target_names,
                        )
                        if topic not in state_data_by_topic:
                            state_data_by_topic[topic] = []
                        state_data_by_topic[topic].append(
                            (timestamp_sec, list(target_names), values)
                        )
                    except Exception as e:
                        self._log_error(f"Failed to deserialize Odometry: {e}")

                # Handle Twist action values from robot config.
                elif topic_type == "geometry_msgs/msg/Twist":
                    target_names = action_names_by_topic_config.get(topic)
                    if not target_names:
                        continue
                    try:
                        msg = deserialize_message(data, Twist)
                        values = self._synthetic_values_for_names(
                            self._twist_values_by_name(msg),
                            target_names,
                        )
                        if topic not in action_data_by_topic:
                            action_data_by_topic[topic] = []
                        action_data_by_topic[topic].append(
                            (timestamp_sec, list(target_names), values)
                        )
                    except Exception as e:
                        self._log_error(f"Failed to deserialize Twist: {e}")

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
            sidecar_by_video_path: Dict[Path, Any] = {}
            if videos_dir.exists() and load_frame_timestamps is not None:
                for sidecar in sorted(
                    videos_dir.rglob("*_timestamps.parquet")
                ):
                    cam_name = sidecar.stem[: -len("_timestamps")]
                    video_path = sidecar.with_name(f"{cam_name}.mp4")
                    if video_path in sidecar_by_video_path:
                        continue
                    try:
                        sidecar_by_video_path[video_path] = load_frame_timestamps(
                            sidecar, cam_name
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        self._log_error(
                            f"replay: failed to read {sidecar.name}: {exc!r}"
                        )

            # Process video files
            if videos_dir.exists():
                for video_file in sorted(videos_dir.rglob("*.mp4")):
                    if video_file.stem.endswith("_synced"):
                        # Skip converter-side derivatives that happen to
                        # live in the same dir.
                        continue
                    result["video_files"].append(
                        video_file.relative_to(bag_path_obj).as_posix()
                    )

                    # Find matching topic
                    video_name = video_file.stem
                    matching_topic = None
                    if video_file in sidecar_by_video_path:
                        # v2 path: per-camera sidecar is the source of
                        # truth. Fold its rows into image_metadata_by_topic
                        # keyed by the relative video path (no source topic
                        # in the v2 MCAP). Segmented datasets can repeat the
                        # same camera filename under videos/<segment>/, so a
                        # plain camera-name key would drop later segments.
                        ft = sidecar_by_video_path[video_file]
                        sidecar_topic = (
                            video_file.relative_to(bag_path_obj)
                            .with_suffix("")
                            .as_posix()
                        )
                        # Use header.stamp seconds (publisher clock) so
                        # the timeline lines up with anything that
                        # publishes header.stamp in the MCAP.
                        per_frame = []
                        for idx, stamp in zip(ft.frame_index, ft.header_stamp_ns):
                            mapped_stamp = self._timestamp_to_replay_seconds(
                                int(stamp), segment_time_map
                            )
                            if mapped_stamp is None:
                                continue
                            per_frame.append((int(idx), mapped_stamp))
                        if per_frame:
                            image_metadata_by_topic[sidecar_topic] = per_frame
                            matching_topic = sidecar_topic
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
                                video_name
                            )
                    else:
                        # No matching topic, try to extract from video filename
                        camera_name = self._extract_camera_name_from_topic(video_name)
                    result["video_names"].append(camera_name or video_name)

                    # Calculate FPS from metadata
                    if matching_topic and matching_topic in image_metadata_by_topic:
                        topic_frame_metadata = image_metadata_by_topic[matching_topic]
                        if len(topic_frame_metadata) > 1:
                            # Sort by frame index
                            topic_frame_metadata.sort(key=lambda x: x[0])
                            time_diffs = [
                                topic_frame_metadata[i + 1][1]
                                - topic_frame_metadata[i][1]
                                for i in range(len(topic_frame_metadata) - 1)
                            ]
                            avg_diff = sum(time_diffs) / len(time_diffs)
                            fps = 1.0 / avg_diff if avg_diff > 0 else 30.0
                            result["video_fps"].append(fps)
                        else:
                            result["video_fps"].append(30.0)
                    else:
                        result["video_fps"].append(30.0)

            result["video_segments"] = self._build_video_segments(
                bag_path_obj,
                result["segments"],
                image_metadata_by_topic,
            )

            # Process frame timestamps (use first video topic)
            if image_metadata_by_topic:
                first_topic = list(image_metadata_by_topic.keys())[0]
                metadata_list = sorted(
                    image_metadata_by_topic[first_topic], key=lambda x: x[0]
                )
                for frame_idx, timestamp in metadata_list:
                    result["frame_indices"].append(frame_idx)
                    result["frame_timestamps"].append(timestamp - min_time)

            result["frame_counts"] = self._build_frame_counts(
                result["video_names"],
                result["video_topics"],
                image_metadata_by_topic,
                result["video_segments"],
            )
            replay_sample_bucket_s = self._replay_sample_bucket_seconds(
                result["video_fps"],
                result["video_segments"],
            )
            replay_sample_origin_s = (
                min_time if math.isfinite(min_time) else 0.0
            )

            # Process joint (state) data — per-topic merge
            # Multiple JointState topics have different joint counts,
            # so we must merge them by timestamp (same logic as action data).
            if state_data_by_topic:
                state_topic_order = self._topic_order(
                    state_data_by_topic,
                    state_topic_order_config,
                )

                # Collect all joint names in topic order
                all_joint_names: List[str] = []
                state_topic_joint_names: Dict[str, List[str]] = {}
                for topic in state_topic_order:
                    if state_data_by_topic[topic]:
                        names = state_names_by_topic_config.get(
                            topic, state_data_by_topic[topic][0][1]
                        )
                        state_topic_joint_names[topic] = names
                        all_joint_names.extend(names)

                result["joint_names"] = all_joint_names

                # Group data at replay display resolution. The bucket is
                # derived from camera FPS so dense joint streams do not make
                # replay payloads much heavier than the video they accompany.
                state_timestamp_data: Dict[float, Dict] = {}
                for topic in state_topic_order:
                    for ts, names, values in state_data_by_topic[topic]:
                        ts_key = self._timestamp_bucket_key(
                            ts,
                            replay_sample_bucket_s,
                            replay_sample_origin_s,
                        )
                        if ts_key not in state_timestamp_data:
                            state_timestamp_data[ts_key] = {}
                        state_timestamp_data[ts_key][topic] = (
                            names,
                            self._values_for_names(
                                names,
                                values,
                                state_topic_joint_names.get(topic, names),
                            ),
                        )

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
                topic_order = list(action_topic_order_config)
                if not topic_order:
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
                        names = action_names_by_topic_config.get(
                            topic, action_data_by_topic[topic][0][1]
                        )
                        topic_joint_names[topic] = names
                        all_action_names.extend(names)

                result["action_names"] = all_action_names

                # Merge action data by timestamp using the same replay
                # display-resolution bucket as state data.
                timestamp_data = {}
                for topic in topic_order:
                    if topic not in action_data_by_topic:
                        continue
                    for ts, names, values in action_data_by_topic[topic]:
                        ts_key = self._timestamp_bucket_key(
                            ts,
                            replay_sample_bucket_s,
                            replay_sample_origin_s,
                        )
                        if ts_key not in timestamp_data:
                            timestamp_data[ts_key] = {}
                        timestamp_data[ts_key][topic] = (
                            names,
                            self._values_for_names(
                                names,
                                values,
                                topic_joint_names.get(topic, names),
                            ),
                        )

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
            if segment_time_map:
                segment_end = max(
                    float(item["replay_end_s"]) for item in segment_time_map
                )
                result["start_time"] = 0.0
                result["end_time"] = segment_end
                result["duration"] = segment_end

            if not result["segments"]:
                result["segments"] = self._metadata_manager.get_episode_segments(
                    bag_path_obj, duration=result["duration"]
                )
            if not result["video_segments"]:
                result["video_segments"] = self._build_video_segments(
                    bag_path_obj,
                    result["segments"],
                    image_metadata_by_topic,
                )

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

    def _load_rosbag_metadata(self, bag_path: Path) -> Dict:
        metadata_path = Path(bag_path) / "metadata.yaml"
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f) or {}
            bag_info = metadata.get("rosbag2_bagfile_information", {})
            return bag_info if isinstance(bag_info, dict) else {}
        except Exception as exc:
            self._log_error(f"replay: failed to read metadata.yaml: {exc!r}")
            return {}

    def _metadata_file_entries(self, bag_path: Path) -> List[Dict]:
        bag_info = self._load_rosbag_metadata(bag_path)
        files = bag_info.get("files") or []
        if isinstance(files, list) and files:
            entries = [item for item in files if isinstance(item, dict)]
            if entries:
                return entries

        paths = bag_info.get("relative_file_paths") or []
        if isinstance(paths, list) and paths:
            return [{"path": path} for path in paths if isinstance(path, str)]

        return [{"path": path.name} for path in sorted(Path(bag_path).glob("*.mcap"))]

    def _build_segment_time_map(
        self, bag_path: Path, segments: List[Dict]
    ) -> List[Dict]:
        entries = self._metadata_file_entries(bag_path)
        if not entries or not segments:
            return []

        time_map: List[Dict] = []
        for index, entry in enumerate(entries):
            if index >= len(segments):
                break
            segment = segments[index]
            frame_duration = segment.get("frame_duration")
            starting_time = entry.get("starting_time", {}) or {}
            duration = entry.get("duration", {}) or {}
            wall_start_ns = starting_time.get("nanoseconds_since_epoch")
            wall_duration_ns = duration.get("nanoseconds")
            if (
                not isinstance(frame_duration, list)
                or len(frame_duration) != 2
                or wall_start_ns is None
                or wall_duration_ns is None
            ):
                continue
            try:
                replay_start_s = float(frame_duration[0])
                replay_end_s = float(frame_duration[1])
                wall_start_ns = int(wall_start_ns)
                wall_duration_ns = int(wall_duration_ns)
            except (TypeError, ValueError):
                continue
            if replay_end_s < replay_start_s or wall_duration_ns <= 0:
                continue
            time_map.append({
                "replay_start_s": replay_start_s,
                "replay_end_s": replay_end_s,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_start_ns + wall_duration_ns,
            })
        return time_map

    def _timestamp_to_replay_seconds(
        self, timestamp_ns: int, segment_time_map: List[Dict]
    ) -> Optional[float]:
        if not segment_time_map:
            return float(timestamp_ns) / 1e9

        # Small tolerance avoids dropping boundary samples with minor clock
        # jitter between MCAP file metadata and message/header stamps.
        tolerance_ns = 50_000_000
        for item in segment_time_map:
            wall_start_ns = int(item["wall_start_ns"])
            wall_end_ns = int(item["wall_end_ns"])
            if wall_start_ns - tolerance_ns <= timestamp_ns <= wall_end_ns + tolerance_ns:
                offset_s = max(0.0, float(timestamp_ns - wall_start_ns) / 1e9)
                replay_start_s = float(item["replay_start_s"])
                replay_end_s = float(item["replay_end_s"])
                return min(replay_end_s, replay_start_s + offset_s)
        return None

    def _camera_name_for_video(self, video_file: Path) -> str:
        return self._extract_camera_name_from_topic(video_file.stem)

    def _target_replay_sample_hz(
        self,
        video_fps: List[float],
        video_segments: Optional[List[Dict]] = None,
    ) -> Optional[int]:
        fps_values: List[float] = []

        def add_fps(value: Any) -> None:
            try:
                fps = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(fps) and fps > 0:
                fps_values.append(fps)

        for fps in video_fps or []:
            add_fps(fps)

        for segment in video_segments or []:
            segment_fps = segment.get("video_fps")
            if not isinstance(segment_fps, list):
                continue
            for fps in segment_fps:
                add_fps(fps)

        if not fps_values:
            return None

        return int(math.ceil(max(fps_values) / 5.0) * 5)

    def _replay_sample_bucket_seconds(
        self,
        video_fps: List[float],
        video_segments: Optional[List[Dict]] = None,
    ) -> Optional[float]:
        target_hz = self._target_replay_sample_hz(video_fps, video_segments)
        if not target_hz:
            return None
        return 1.0 / float(target_hz)

    def _timestamp_bucket_key(
        self,
        timestamp_s: float,
        bucket_s: Optional[float],
        origin_s: float = 0.0,
    ) -> float:
        if not bucket_s or not math.isfinite(bucket_s) or bucket_s <= 0:
            return round(timestamp_s * 100) / 100

        origin = origin_s if math.isfinite(origin_s) else 0.0
        bucket_index = math.floor(((timestamp_s - origin) / bucket_s) + 1e-9)
        return round(origin + bucket_index * bucket_s, 9)

    def _build_frame_counts(
        self,
        video_names: List[str],
        video_topics: List[str],
        image_metadata_by_topic: Dict[str, List[Tuple[int, float]]],
        video_segments: Optional[List[Dict]] = None,
    ) -> Dict[str, int]:
        frame_counts: Dict[str, int] = {}
        has_segment_counts = False

        for segment in video_segments or []:
            segment_counts = segment.get("frame_counts")
            if not isinstance(segment_counts, dict):
                continue

            for camera_name, count in segment_counts.items():
                name = str(camera_name or "")
                if not name:
                    continue
                try:
                    count_value = max(0, int(count))
                except (TypeError, ValueError):
                    count_value = 0
                frame_counts[name] = frame_counts.get(name, 0) + count_value
                has_segment_counts = True

        if has_segment_counts:
            return frame_counts

        for i, video_name in enumerate(video_names):
            name = str(video_name or f"video_{i}")
            topic = video_topics[i] if i < len(video_topics) else None
            count = len(image_metadata_by_topic.get(topic, [])) if topic else 0
            frame_counts[name] = frame_counts.get(name, 0) + count

        return frame_counts

    def _build_video_segments(
        self,
        bag_path: Path,
        segments: List[Dict],
        image_metadata_by_topic: Dict[str, List[Tuple[int, float]]],
    ) -> List[Dict]:
        """Build a light segment/video map from existing episode files.

        The UI can use this to show only the cameras for the active subtask
        segment instead of loading every segment's MP4s at once.
        """
        bag_path = Path(bag_path)
        videos_root = bag_path / "videos"
        file_entries = self._metadata_file_entries(bag_path)
        if not file_entries:
            return []

        video_segments: List[Dict] = []
        for index, file_entry in enumerate(file_entries):
            mcap_name = str(file_entry.get("path") or "")
            if not mcap_name:
                continue
            segment_name = Path(mcap_name).stem
            video_dir = videos_root / segment_name
            video_files = [
                path
                for path in sorted(video_dir.glob("*.mp4"))
                if not path.stem.endswith("_synced")
            ]
            if not video_files:
                continue

            segment = segments[index] if index < len(segments) else {}
            frame_duration = segment.get("frame_duration")
            if (
                not isinstance(frame_duration, list)
                or len(frame_duration) != 2
            ):
                start_s = 0.0
                duration_ns = (
                    file_entry.get("duration", {}) or {}
                ).get("nanoseconds", 0)
                end_s = float(duration_ns) / 1e9 if duration_ns else 0.0
            else:
                start_s = float(frame_duration[0])
                end_s = float(frame_duration[1])

            names: List[str] = []
            fps_values: List[float] = []
            frame_counts: Dict[str, int] = {}
            rel_video_files: List[str] = []
            timestamp_sidecars: Dict[str, str] = {}

            for video_file in video_files:
                rel_path = video_file.relative_to(bag_path).as_posix()
                rel_video_files.append(rel_path)
                camera_name = self._camera_name_for_video(video_file)
                names.append(camera_name)

                sidecar = video_file.with_name(f"{video_file.stem}_timestamps.parquet")
                if sidecar.exists():
                    timestamp_sidecars[camera_name] = sidecar.relative_to(
                        bag_path
                    ).as_posix()

                topic_key = video_file.relative_to(bag_path).with_suffix("").as_posix()
                per_frame = image_metadata_by_topic.get(topic_key, [])
                frame_counts[camera_name] = len(per_frame)
                if len(per_frame) > 1:
                    sorted_frames = sorted(per_frame, key=lambda x: x[0])
                    diffs = [
                        sorted_frames[i + 1][1] - sorted_frames[i][1]
                        for i in range(len(sorted_frames) - 1)
                    ]
                    avg_diff = sum(diffs) / len(diffs)
                    fps_values.append(1.0 / avg_diff if avg_diff > 0 else 30.0)
                else:
                    fps_values.append(30.0)

            starting_time = file_entry.get("starting_time", {}) or {}
            duration_info = file_entry.get("duration", {}) or {}
            video_segments.append({
                "index": index,
                "name": segment_name,
                "mcap": mcap_name,
                "video_dir": video_dir.relative_to(bag_path).as_posix(),
                "video_files": rel_video_files,
                "video_names": names,
                "video_fps": fps_values,
                "frame_counts": frame_counts,
                "timestamp_sidecars": timestamp_sidecars,
                "sub_task_instruction": str(
                    segment.get("sub_task_instruction", "") or ""
                ),
                "frame_duration": [float(start_s), float(end_s)],
                "replay_start_s": float(start_s),
                "replay_end_s": float(end_s),
                "wall_start_ns": starting_time.get("nanoseconds_since_epoch"),
                "wall_duration_ns": duration_info.get("nanoseconds"),
            })

        return video_segments

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
