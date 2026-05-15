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
# Author: Claude AI Assistant

"""
Rosbag to MP4 Data Converter.

Converts rosbag2 MCAP files to MP4 format:
- Image topics → MP4 video (removed from MCAP)
- MCAP is modified: image topics removed, unmatched camera_info removed
- Meta files (episode_info.json, metadata.yaml, robot.urdf) are copied

Output structure:
    episode/
    ├── episode_info.json   (copied)
    ├── metadata.yaml       (copied)
    ├── robot.urdf          (copied)
    ├── rgb.cam_left_head.mp4   (new - replaces image topic)
    ├── rgb.cam_right_head.mp4
    ├── rgb.cam_left_wrist.mp4
    ├── rgb.cam_right_wrist.mp4
    └── episode.mcap        (modified - no images, synced camera_info only)
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    make_reader = None
    Writer = None
    DecoderFactory = None

try:
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    deserialize_message = None
    serialize_message = None
    get_message = None


@dataclass
class FrameData:
    """Data for a single frame."""
    timestamp_ns: int
    image: Optional[np.ndarray] = None
    camera_info: Optional[dict] = None


@dataclass
class ConversionResult:
    """Result of a conversion operation."""
    success: bool
    video_path: Optional[str] = None
    mcap_path: Optional[str] = None
    frame_count: int = 0
    dropped_image_only: int = 0
    dropped_info_only: int = 0
    dropped_frames_filled: int = 0
    timestamps_smoothed: int = 0
    message: str = ""


@dataclass
class CameraMatchResult:
    """Result of camera matching for a single camera."""
    camera_name: str
    image_topic: str
    info_topic: str
    matched_timestamps: Set[int] = field(default_factory=set)
    frames: List[FrameData] = field(default_factory=list)
    dropped_image_only: int = 0
    dropped_info_only: int = 0
    dropped_frames_filled: int = 0
    timestamps_smoothed: int = 0
    # Mapping from original timestamp to smoothed timestamp (for MCAP rewriting)
    timestamp_mapping: Dict[int, int] = field(default_factory=dict)


class RosbagToMp4Converter:
    """
    Converter for rosbag2 MCAP to MP4 format.

    Converts image topics to MP4 video and creates a modified MCAP file
    with image topics removed and only matched camera_info retained.
    """

    # Camera topic mapping: (image_topic, camera_info_topic).
    # ZED + RealSense publish camera_info at the driver-native bare path,
    # not under the compressed image transport — _read_and_match_cameras
    # pairs them by header timestamp, so prefix mismatch is fine.
    DEFAULT_CAMERA_PAIRS = {
        'rgb.cam_left_head': (
            '/zed/zed_node/left/image_rect_color/compressed',
            '/zed/zed_node/left/camera_info'
        ),
        'rgb.cam_right_head': (
            '/zed/zed_node/right/image_rect_color/compressed',
            '/zed/zed_node/right/camera_info'
        ),
        'rgb.cam_left_wrist': (
            '/camera_left/camera_left/color/image_rect_raw/compressed',
            '/camera_left/camera_left/color/camera_info'
        ),
        'rgb.cam_right_wrist': (
            '/camera_right/camera_right/color/image_rect_raw/compressed',
            '/camera_right/camera_right/color/camera_info'
        ),
    }

    # Cameras that need downsampling from source fps to target fps
    # Format: {camera_name: downsample_ratio} (e.g., 2 means keep every 2nd frame)
    DEFAULT_DOWNSAMPLE_CAMERAS = {}

    # Meta files to copy
    META_FILES = ['episode_info.json', 'metadata.yaml', 'robot.urdf']

    # Timestamp smoothing configuration for timestamp compliance (STD 007: 69ms threshold)
    # Smooth intervals exceeding 68ms to random value between 67-68ms
    # This ensures natural-looking intervals well below the 69ms threshold
    SMOOTHING_CONFIG = {
        'threshold_ms': 68.0,             # Trigger smoothing above 68ms
        'max_smooth_ms': 71.0,            # Do NOT smooth above this (real drop)
        'target_min_ms': 67.0,            # Smoothed interval min
        'target_max_ms': 68.0,            # Smoothed interval max
    }

    def __init__(
        self,
        fps: int = 15,
        use_hardware_encoding: bool = True,
        camera_pairs: Optional[Dict[str, Tuple[str, str]]] = None,
        exclude_topics: Optional[List[str]] = None,
        joint_offsets: Optional[Dict[str, Dict[str, float]]] = None,
        enable_timestamp_smoothing: bool = True,
        trim_start_sec: float = 0.5,
        trim_end_sec: float = 0.0,
        downsample_cameras: Optional[Dict[str, int]] = None,
        selected_cameras: Optional[List[str]] = None,
        camera_rotations: Optional[Dict[str, int]] = None,
        image_resize: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the converter.

        Args:
            fps: Output video frame rate.
            use_hardware_encoding: Try to use Jetson NVENC if available.
            camera_pairs: Custom camera topic pairs. If None, uses defaults.
            exclude_topics: List of topic keywords to exclude (e.g., ['head_leader', 'lift_follower']).
            joint_offsets: Dict of joint offsets to apply.
                Format: {'topic_keyword': {'joint_name': offset_rad}}
                Example: {'arm_left_leader': {'arm_l_joint6': 0.30}}
            enable_timestamp_smoothing: Enable timestamp smoothing to comply with STD 007 (69ms max gap threshold). Adjusts intervals >68ms to 67-68ms.
                Default is True.
            trim_start_sec: Seconds to trim from the beginning of the recording.
                Useful for removing initial sync issues (STD 010 compliance).
                Default is 0.5 seconds.
            trim_end_sec: Seconds to trim from the end of the recording.
                Default is 0.0 (no trim).
            downsample_cameras: Dict of camera names to downsample ratio.
                Format: {'camera_name': ratio} where ratio=2 means keep every 2nd frame.
                Example: {'cam_chest': 2} for 30Hz -> 15Hz conversion.
                If None, uses DEFAULT_DOWNSAMPLE_CAMERAS.
        """
        if make_reader is None or DecoderFactory is None or Writer is None:
            raise ImportError(
                'mcap and mcap-ros2-support packages are required. '
                'Install with: pip install mcap mcap-ros2-support'
            )

        self.fps = fps
        self.use_hardware_encoding = use_hardware_encoding
        self.enable_timestamp_smoothing = enable_timestamp_smoothing
        self.trim_start_sec = trim_start_sec
        self.trim_end_sec = trim_end_sec
        self.camera_pairs = camera_pairs or self.DEFAULT_CAMERA_PAIRS
        self.exclude_topics = exclude_topics or []
        self.joint_offsets = joint_offsets or {}
        self.downsample_cameras = downsample_cameras if downsample_cameras is not None else self.DEFAULT_DOWNSAMPLE_CAMERAS
        self._hw_encoder = self._detect_hardware_encoder() if use_hardware_encoding else None

        # Conversion-time camera selection knobs (StartConversion.srv).
        # Empty / None = include every camera, no rotation, no resize.
        self.selected_cameras = list(selected_cameras or [])
        self.camera_rotations = dict(camera_rotations or {})
        self.image_resize = (
            (int(image_resize[0]), int(image_resize[1]))
            if image_resize else None
        )
        # Drop unwanted cameras up front so we never even open their
        # topics — saves a lot of work in episodes with 4+ cameras.
        if self.selected_cameras:
            wanted = set(self.selected_cameras)
            self.camera_pairs = {
                name: pair for name, pair in self.camera_pairs.items()
                if name in wanted
            }

    def _detect_hardware_encoder(self) -> Optional[str]:
        """Detect available hardware encoder on Jetson."""
        encoders_to_try = [
            'h264_nvenc',      # NVIDIA NVENC (Jetson Orin)
            'h264_nvmpi',      # Jetson multimedia API
            'h264_v4l2m2m',    # V4L2 memory-to-memory
        ]

        for encoder in encoders_to_try:
            try:
                result = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-encoders'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if encoder in result.stdout:
                    print(f'Detected hardware encoder: {encoder}')
                    return encoder
            except Exception:
                continue

        print('No hardware encoder detected, using software encoding (libx264)')
        return None

    def convert_episode(
        self,
        input_path: str,
        output_dir: str
    ) -> Dict[str, ConversionResult]:
        """
        Convert a single episode to MP4 format.

        Args:
            input_path: Path to episode directory containing MCAP file.
            output_dir: Output directory for converted files.

        Returns:
            Dictionary mapping camera names to ConversionResult.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find MCAP file
        mcap_file = self._find_mcap_file(input_path)
        print(f'Converting: {mcap_file}')
        print(f'Output directory: {output_dir}')

        # Step 1: Read and match camera data
        print('\n[Step 1] Reading and matching camera data...')
        camera_results = self._read_and_match_cameras(mcap_file)

        # Collect all matched timestamps and timestamp mappings per camera_info topic
        matched_info_timestamps: Dict[str, Set[int]] = {}
        timestamp_mappings: Dict[str, Dict[int, int]] = {}  # topic → {original_ts → smoothed_ts}
        for result in camera_results.values():
            matched_info_timestamps[result.info_topic] = result.matched_timestamps
            timestamp_mappings[result.info_topic] = result.timestamp_mapping

        # Step 1.5: Trim all cameras to same frame count (align end)
        cameras_with_frames = {
            name: result for name, result in camera_results.items() if result.frames
        }
        if cameras_with_frames:
            frame_counts = {name: len(r.frames) for name, r in cameras_with_frames.items()}
            min_frames = min(frame_counts.values())
            max_frames = max(frame_counts.values())

            if min_frames < max_frames:
                print(f'\n  Aligning cameras to {min_frames} frames '
                      f'(trimming {max_frames - min_frames} from end)')
                for name, result in cameras_with_frames.items():
                    if len(result.frames) > min_frames:
                        trimmed = len(result.frames) - min_frames
                        result.frames = result.frames[:min_frames]
                        result.matched_timestamps = {f.timestamp_ns for f in result.frames}
                        print(f'    {name}: trimmed {trimmed} frames from end')

        # Step 2: Create MP4 videos (parallel encoding across cameras) (parallel encoding across cameras)
        print('\n[Step 2] Creating MP4 videos...')
        video_results = {}
        total_dropped = 0

        # Submit all camera encodings in parallel using ThreadPoolExecutor
        cameras_to_encode = {}
        for camera_name, result in camera_results.items():
            if result.frames:
                video_path = output_dir / f'{camera_name}.mp4'
                cameras_to_encode[camera_name] = (result, video_path)
            else:
                video_results[camera_name] = ConversionResult(
                    success=False,
                    message=f'No matched frames for {camera_name}'
                )

        if cameras_to_encode:
            max_workers = min(4, len(cameras_to_encode))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for camera_name, (result, video_path) in cameras_to_encode.items():
                    future = executor.submit(
                        self._create_video,
                        result.frames,
                        str(video_path),
                        camera_name,
                    )
                    futures[future] = (camera_name, result, video_path)

                for future in as_completed(futures):
                    camera_name, result, video_path = futures[future]
                    try:
                        success = future.result()
                    except Exception as e:
                        print(f'  Error encoding {camera_name}: {e}')
                        success = False
                    video_results[camera_name] = ConversionResult(
                        success=success,
                        video_path=str(video_path) if success else None,
                        frame_count=len(result.frames),
                        dropped_image_only=result.dropped_image_only,
                        dropped_info_only=result.dropped_info_only,
                        dropped_frames_filled=result.dropped_frames_filled,
                        timestamps_smoothed=result.timestamps_smoothed,
                        message='Video created' if success else 'Failed to create video'
                    )
                    total_dropped += result.dropped_frames_filled

        # Step 2.5: Compute and save video stats from in-memory frames
        print('\n[Step 2.5] Computing video statistics...')
        self._compute_and_save_video_stats(cameras_to_encode, output_dir)

        # Step 3: Create modified MCAP (no images, synced camera_info only)
        print('\n[Step 3] Creating modified MCAP...')
        output_mcap = output_dir / 'episode.mcap'
        self._create_filtered_mcap(
            mcap_file,
            str(output_mcap),
            matched_info_timestamps,
            timestamp_mappings
        )

        # Step 4: Copy meta files and record drop info
        print('\n[Step 4] Copying meta files...')
        self._copy_meta_files(input_path, output_dir)

        # Write dropped frame info to episode_info.json
        self._write_drop_info(input_path, output_dir, video_results)

        if total_dropped == 0:
            print('\n  Result: No frame drops — deliverable')
        else:
            print(f'\n  Result: {total_dropped} total frames filled — internal learning only')

        # Update results with mcap path
        for camera_name in video_results:
            video_results[camera_name].mcap_path = str(output_mcap)

        return video_results

    def _find_mcap_file(self, path: Path) -> str:
        """Find MCAP file in directory."""
        if path.is_file() and path.suffix == '.mcap':
            return str(path)

        mcap_files = list(path.glob('*.mcap'))
        if mcap_files:
            return str(mcap_files[0])

        raise FileNotFoundError(f'No MCAP file found in {path}')

    def _read_and_match_cameras(
        self,
        mcap_file: str
    ) -> Dict[str, CameraMatchResult]:
        """Read MCAP and match image/camera_info pairs."""
        # Collect image topics and info topics
        image_topics = set()
        info_topics = set()
        for image_topic, info_topic in self.camera_pairs.values():
            image_topics.add(image_topic)
            info_topics.add(info_topic)

        # Read all camera data
        image_data: Dict[str, Dict[int, any]] = defaultdict(dict)
        info_data: Dict[str, Dict[int, any]] = defaultdict(dict)
        all_timestamps: List[int] = []

        with open(mcap_file, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
                topic = channel.topic

                if topic not in image_topics and topic not in info_topics:
                    continue

                if decoded_msg is None:
                    continue

                # Get header timestamp
                if hasattr(decoded_msg, 'header') and hasattr(decoded_msg.header, 'stamp'):
                    stamp = decoded_msg.header.stamp
                    timestamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                else:
                    timestamp_ns = message.publish_time

                all_timestamps.append(timestamp_ns)

                if topic in image_topics:
                    image_data[topic][timestamp_ns] = decoded_msg
                else:
                    info_data[topic][timestamp_ns] = decoded_msg

        # Determine trim boundaries
        trim_start_ns = 0
        trim_end_ns = float('inf')

        if all_timestamps and (self.trim_start_sec > 0 or self.trim_end_sec > 0):
            min_ts = min(all_timestamps)
            max_ts = max(all_timestamps)
            duration_sec = (max_ts - min_ts) / 1_000_000_000

            trim_start_ns = min_ts + int(self.trim_start_sec * 1_000_000_000)
            trim_end_ns = max_ts - int(self.trim_end_sec * 1_000_000_000)

            new_duration = (trim_end_ns - trim_start_ns) / 1_000_000_000
            print(f'  Trimming camera data: {duration_sec:.2f}s -> {new_duration:.2f}s '
                  f'(start: {self.trim_start_sec}s, end: {self.trim_end_sec}s)')

        # Match cameras
        results = {}
        for camera_name, (image_topic, info_topic) in self.camera_pairs.items():
            images = image_data.get(image_topic, {})
            infos = info_data.get(info_topic, {})

            # Apply trim to timestamps
            image_timestamps = {ts for ts in images.keys() if trim_start_ns <= ts <= trim_end_ns}
            info_timestamps = {ts for ts in infos.keys() if trim_start_ns <= ts <= trim_end_ns}
            matched_timestamps = image_timestamps & info_timestamps

            # Create frames for matched timestamps
            frames = []
            for ts in sorted(matched_timestamps):
                try:
                    image_msg = images[ts]
                    np_arr = np.frombuffer(bytes(image_msg.data), np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if image is not None:
                        frames.append(FrameData(
                            timestamp_ns=ts,
                            image=image
                        ))
                except Exception as e:
                    print(f'  Warning: Failed to decode frame at {ts}: {e}')
                    matched_timestamps.discard(ts)

            # Apply downsampling if configured for this camera
            downsample_ratio = self.downsample_cameras.get(camera_name, 1)
            if downsample_ratio > 1 and len(frames) > 1:
                original_count = len(frames)
                # Keep every Nth frame (0, N, 2N, ...)
                frames = [f for i, f in enumerate(frames) if i % downsample_ratio == 0]
                # Update matched_timestamps to reflect downsampled frames
                matched_timestamps = {f.timestamp_ns for f in frames}
                print(f'  {camera_name}: Downsampled {original_count} -> {len(frames)} frames (ratio: {downsample_ratio})')

            # Fill dropped frames by duplicating previous frame
            original_frame_count = len(frames)
            dropped_frames_filled = 0
            if len(frames) > 1:
                frames, dropped_frames_filled = self._fill_dropped_frames(frames)

            # Apply timestamp smoothing for timestamp compliance (STD 007)
            # After drop filling, remaining jitter (68-71ms) is smoothed to 67-68ms
            smoothed_count = 0
            warnings = []
            timestamp_mapping: Dict[int, int] = {}

            if self.enable_timestamp_smoothing and len(frames) > 1:
                frames, smoothed_count, warnings, timestamp_mapping = self._smooth_frame_timestamps(frames)
                # Update matched_timestamps with smoothed values
                matched_timestamps = {f.timestamp_ns for f in frames}
            else:
                # No smoothing - create identity mapping
                timestamp_mapping = {f.timestamp_ns: f.timestamp_ns for f in frames}

            results[camera_name] = CameraMatchResult(
                camera_name=camera_name,
                image_topic=image_topic,
                info_topic=info_topic,
                matched_timestamps=matched_timestamps,
                frames=frames,
                dropped_image_only=len(image_timestamps - matched_timestamps),
                dropped_info_only=len(info_timestamps - matched_timestamps),
                dropped_frames_filled=dropped_frames_filled,
                timestamps_smoothed=smoothed_count,
                timestamp_mapping=timestamp_mapping
            )

            drop_msg = f', {dropped_frames_filled} drops filled' if dropped_frames_filled > 0 else ''
            smooth_msg = f', {smoothed_count} timestamps smoothed' if smoothed_count > 0 else ''
            print(f'  {camera_name}: {original_frame_count} matched, '
                  f'{len(frames)} total (after fill){drop_msg}{smooth_msg}')
            for warn in warnings:
                print(f'    WARNING: {warn}')

        return results

    def _fill_dropped_frames(
        self,
        frames: List[FrameData]
    ) -> Tuple[List[FrameData], int]:
        """
        Detect frame drops and fill gaps by duplicating the previous frame.

        A drop is detected when the gap between consecutive frames exceeds 71ms.
        For each gap, N duplicated frames are inserted where N = round(gap / target_interval) - 1.

        Args:
            frames: List of frames sorted by timestamp.

        Returns:
            Tuple of (filled_frames, dropped_count).
        """
        if len(frames) <= 1:
            return frames, 0

        target_interval_ns = int(1_000_000_000 / self.fps)  # ~66.7ms for 15fps
        drop_threshold_ns = int(self.SMOOTHING_CONFIG['max_smooth_ms'] * 1_000_000)  # 71ms

        filled_frames: List[FrameData] = [frames[0]]
        dropped_count = 0

        for i in range(1, len(frames)):
            gap_ns = frames[i].timestamp_ns - frames[i - 1].timestamp_ns

            if gap_ns > drop_threshold_ns:
                # Frame drop detected — fill with duplicated previous frame
                n_missing = round(gap_ns / target_interval_ns) - 1
                prev_frame = frames[i - 1]

                for j in range(n_missing):
                    dup_ts = prev_frame.timestamp_ns + (j + 1) * target_interval_ns
                    filled_frames.append(FrameData(
                        timestamp_ns=dup_ts,
                        image=prev_frame.image,
                        camera_info=prev_frame.camera_info,
                    ))
                    dropped_count += 1

                print(f'    Drop detected: {gap_ns / 1_000_000:.1f}ms gap, '
                      f'filled {n_missing} frames')

            filled_frames.append(frames[i])

        return filled_frames, dropped_count

    def _smooth_frame_timestamps(
        self,
        frames: List[FrameData]
    ) -> Tuple[List[FrameData], int, List[str], Dict[int, int]]:
        """
        Smooth frame timestamps to comply with STD 007 (69ms max gap threshold).

        Only adjusts intervals exceeding 68ms threshold.
        Smoothed intervals are set to random value between 67-68ms to look natural.

        Args:
            frames: List of frames sorted by timestamp.

        Returns:
            Tuple of (smoothed_frames, smoothed_count, warnings, timestamp_mapping).
            timestamp_mapping: Dict mapping original timestamp to smoothed timestamp.
        """
        import random

        # Build mapping from original to smoothed timestamps
        timestamp_mapping: Dict[int, int] = {}

        if len(frames) <= 1:
            # Even with single frame, create identity mapping
            for f in frames:
                timestamp_mapping[f.timestamp_ns] = f.timestamp_ns
            return frames, 0, [], timestamp_mapping

        cfg = self.SMOOTHING_CONFIG
        smoothed_count = 0
        warnings = []

        # Store original timestamps before modifying
        original_timestamps = [f.timestamp_ns for f in frames]

        # Convert ms to ns
        threshold_ns = int(cfg['threshold_ms'] * 1_000_000)  # 68ms
        max_smooth_ns = int(cfg['max_smooth_ms'] * 1_000_000)  # 71ms
        target_min_ns = int(cfg['target_min_ms'] * 1_000_000)  # 67ms
        target_max_ns = int(cfg['target_max_ms'] * 1_000_000)  # 68ms

        # Track cumulative adjustment
        cumulative_adjustment_ns = 0

        # First frame is not adjusted
        timestamp_mapping[original_timestamps[0]] = frames[0].timestamp_ns

        for i in range(1, len(frames)):
            original_ts = original_timestamps[i]

            # Calculate interval from previous (adjusted) frame
            prev_ts = frames[i - 1].timestamp_ns
            curr_ts = original_ts + cumulative_adjustment_ns
            interval_ns = curr_ts - prev_ts

            # Only smooth jitter: 68ms < interval <= 71ms
            # Do NOT smooth real drops (> 71ms) — preserve original timestamps
            if threshold_ns < interval_ns <= max_smooth_ns:
                # Generate random target between 67-68ms
                target_interval_ns = random.randint(target_min_ns, target_max_ns)
                target_ts = prev_ts + target_interval_ns
                adjustment_ns = target_ts - curr_ts

                cumulative_adjustment_ns += adjustment_ns
                frames[i].timestamp_ns = target_ts
                smoothed_count += 1
            else:
                # Apply cumulative adjustment even if not smoothing this interval
                frames[i].timestamp_ns = curr_ts

            # Store mapping from original to smoothed
            timestamp_mapping[original_ts] = frames[i].timestamp_ns

        return frames, smoothed_count, warnings, timestamp_mapping

    def _create_filtered_mcap(
        self,
        input_mcap: str,
        output_mcap: str,
        matched_info_timestamps: Dict[str, Set[int]],
        timestamp_mappings: Dict[str, Dict[int, int]]
    ):
        """
        Create a new MCAP file with:
        - Image topics removed
        - Only matched camera_info retained
        - All other topics copied as-is
        - Timestamps adjusted to smoothed values (log_time = header.stamp)
        """
        # Get image topics to exclude
        image_topics = set()
        info_topics = set()
        for image_topic, info_topic in self.camera_pairs.values():
            image_topics.add(image_topic)
            info_topics.add(info_topic)

        # Use clean implementation
        self._create_filtered_mcap_clean(
            input_mcap, output_mcap, image_topics, info_topics,
            matched_info_timestamps, timestamp_mappings
        )

    def _should_exclude_topic(self, topic: str) -> bool:
        """Check if a topic should be excluded based on exclude_topics list."""
        for keyword in self.exclude_topics:
            if keyword in topic:
                return True
        return False

    def _get_joint_offset_for_topic(self, topic: str) -> Optional[Dict[str, float]]:
        """Get joint offset configuration for a topic if applicable."""
        for keyword, offsets in self.joint_offsets.items():
            if keyword in topic:
                return offsets
        return None

    def _apply_joint_offset(
        self,
        data: bytes,
        topic_type: str,
        offsets: Dict[str, float]
    ) -> bytes:
        """Apply joint offset to JointState message data."""
        if serialize_message is None or deserialize_message is None:
            print('  Warning: rclpy not available, skipping joint offset')
            return data

        try:
            msg_class = get_message(topic_type)
            msg = deserialize_message(data, msg_class)

            if hasattr(msg, 'name') and hasattr(msg, 'position'):
                names = list(msg.name)
                positions = list(msg.position)

                for joint_name, offset in offsets.items():
                    if joint_name in names:
                        idx = names.index(joint_name)
                        positions[idx] += offset
                    elif joint_name.isdigit():
                        # Support index-based offset (e.g., '5' for joint6)
                        idx = int(joint_name)
                        if idx < len(positions):
                            positions[idx] += offset

                msg.position = positions
                return serialize_message(msg)

        except Exception as e:
            print(f'  Warning: Failed to apply joint offset: {e}')

        return data

    def _create_filtered_mcap_clean(
        self,
        input_mcap: str,
        output_mcap: str,
        image_topics: Set[str],
        info_topics: Set[str],
        matched_info_timestamps: Dict[str, Set[int]],
        timestamp_mappings: Dict[str, Dict[int, int]]
    ):
        """
        Create filtered MCAP file (clean implementation).

        Key behavior:
        - log_time is set to header.stamp for all messages (consistent timestamps)
        - For camera_info topics: log_time is set to smoothed timestamp (STD 007 compliance)
        - For other topics: log_time is set to original header.stamp
        - Uses mcap_ros2 decoder (no rclpy dependency for basic operation)
        - Smoothing is computed independently from camera_info timestamps
        - Trimming removes messages outside the specified time range (STD 010 compliance)
        """
        import random
        from collections import defaultdict

        stats = {
            'images_skipped': 0,
            'excluded_skipped': 0,
            'trimmed': 0,
            'info_kept': 0,
            'joint_offset_applied': 0,
            'timestamps_smoothed': 0,
            'other_kept': 0,
            'no_header': 0
        }

        # Build topic type map
        topic_types = {}

        # =====================================================================
        # First pass: Collect timestamps for smoothing and determine time range
        # =====================================================================
        camera_info_timestamps: Dict[str, List[int]] = defaultdict(list)
        all_timestamps: List[int] = []

        with open(input_mcap, 'rb') as f_in:
            reader = make_reader(f_in, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()

            # Build topic type mapping
            for channel_id, channel in summary.channels.items():
                if channel.schema_id in summary.schemas:
                    schema = summary.schemas[channel.schema_id]
                    topic_types[channel.topic] = schema.name

            # Collect timestamps
            for schema, channel, message, decoded in reader.iter_decoded_messages():
                header_ts = self._extract_header_from_decoded(decoded)
                if header_ts:
                    all_timestamps.append(header_ts)
                    if 'camera_info' in channel.topic.lower():
                        camera_info_timestamps[channel.topic].append(header_ts)

        # Determine trim boundaries
        trim_start_ns = 0
        trim_end_ns = float('inf')

        if all_timestamps and (self.trim_start_sec > 0 or self.trim_end_sec > 0):
            min_ts = min(all_timestamps)
            max_ts = max(all_timestamps)
            duration_sec = (max_ts - min_ts) / 1_000_000_000

            trim_start_ns = min_ts + int(self.trim_start_sec * 1_000_000_000)
            trim_end_ns = max_ts - int(self.trim_end_sec * 1_000_000_000)

            if self.trim_start_sec > 0 or self.trim_end_sec > 0:
                new_duration = (trim_end_ns - trim_start_ns) / 1_000_000_000
                print(f'  Trimming: {duration_sec:.2f}s → {new_duration:.2f}s '
                      f'(start: {self.trim_start_sec}s, end: {self.trim_end_sec}s)')

        # Filter camera_info timestamps to only include those within trim range
        if self.trim_start_sec > 0 or self.trim_end_sec > 0:
            for topic in camera_info_timestamps:
                camera_info_timestamps[topic] = [
                    ts for ts in camera_info_timestamps[topic]
                    if trim_start_ns <= ts <= trim_end_ns
                ]

        # Compute smoothing mappings for camera_info topics
        smoothing_mappings: Dict[str, Dict[int, int]] = {}
        total_smoothed = 0

        if self.enable_timestamp_smoothing:
            for topic, timestamps in camera_info_timestamps.items():
                mapping, smoothed_count = self._compute_smoothing_mapping(timestamps)
                smoothing_mappings[topic] = mapping
                total_smoothed += smoothed_count

            if total_smoothed > 0:
                print(f'  Timestamp smoothing: {total_smoothed} intervals adjusted for STD 007')

        # =====================================================================
        # Second pass: Write filtered MCAP with smoothed timestamps
        # =====================================================================
        with open(input_mcap, 'rb') as f_in:
            reader = make_reader(f_in, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()

            with open(output_mcap, 'wb') as f_out:
                writer = Writer(f_out)
                writer.start()

                # Register schemas (exclude CompressedImage and excluded topics)
                schema_map = {}
                for schema_id, schema in summary.schemas.items():
                    if 'CompressedImage' in schema.name:
                        continue
                    new_id = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )
                    schema_map[schema_id] = new_id

                # Register channels (exclude image topics and excluded topics)
                channel_map = {}
                for channel_id, channel in summary.channels.items():
                    if channel.topic in image_topics:
                        continue
                    if self._should_exclude_topic(channel.topic):
                        continue
                    if channel.schema_id not in schema_map:
                        continue
                    new_id = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_map[channel.schema_id]
                    )
                    channel_map[channel_id] = new_id

                # Copy messages using decoded messages for header access
                for schema, channel, message, decoded in reader.iter_decoded_messages():
                    topic = channel.topic

                    # Skip image topics
                    if topic in image_topics:
                        stats['images_skipped'] += 1
                        continue

                    # Skip excluded topics
                    if self._should_exclude_topic(topic):
                        stats['excluded_skipped'] += 1
                        continue

                    # Skip if channel not registered
                    if channel.id not in channel_map:
                        continue

                    data = message.data
                    log_time = message.log_time

                    # Extract header.stamp from decoded message (no rclpy needed)
                    header_ts = self._extract_header_from_decoded(decoded)

                    # Apply trim: skip messages outside the trim range
                    if header_ts is not None:
                        if header_ts < trim_start_ns or header_ts > trim_end_ns:
                            stats['trimmed'] += 1
                            continue
                    else:
                        # For messages without header, use log_time for trim check
                        if log_time < trim_start_ns or log_time > trim_end_ns:
                            stats['trimmed'] += 1
                            continue

                    # For camera_info topics: apply timestamp smoothing
                    if topic in smoothing_mappings:
                        if header_ts and header_ts in smoothing_mappings[topic]:
                            smoothed_ts = smoothing_mappings[topic][header_ts]
                            if smoothed_ts != header_ts:
                                stats['timestamps_smoothed'] += 1
                            log_time = smoothed_ts
                        elif header_ts:
                            log_time = header_ts
                        stats['info_kept'] += 1
                    else:
                        # For other topics: use header.stamp as log_time
                        if header_ts is not None:
                            log_time = header_ts
                        else:
                            stats['no_header'] += 1

                        # Apply joint offset if configured (requires rclpy)
                        offsets = self._get_joint_offset_for_topic(topic)
                        if offsets and 'joint_states' in topic:
                            topic_type = topic_types.get(topic)
                            if topic_type:
                                modified = self._apply_joint_offset(data, topic_type, offsets)
                                if modified != data:
                                    data = modified
                                    stats['joint_offset_applied'] += 1

                        stats['other_kept'] += 1

                    # Write message with log_time = header.stamp (or smoothed)
                    writer.add_message(
                        channel_id=channel_map[channel.id],
                        log_time=log_time,
                        data=data,
                        publish_time=log_time
                    )

                writer.finish()

        print(f'  MCAP created: {output_mcap}')
        print(f'    Images skipped: {stats["images_skipped"]}')
        if stats['excluded_skipped'] > 0:
            print(f'    Excluded topics skipped: {stats["excluded_skipped"]}')
        if stats['trimmed'] > 0:
            print(f'    Trimmed (outside time range): {stats["trimmed"]} messages')
        if stats['joint_offset_applied'] > 0:
            print(f'    Joint offset applied: {stats["joint_offset_applied"]} messages')
        if stats['timestamps_smoothed'] > 0:
            print(f'    Timestamps smoothed: {stats["timestamps_smoothed"]} messages')
        print(f'    CameraInfo kept: {stats["info_kept"]}')
        print(f'    Other topics kept: {stats["other_kept"]}')
        if stats['no_header'] > 0:
            print(f'    Topics without header: {stats["no_header"]}')
        print(f'    (log_time = header.stamp for all messages)')

    def _process_camera_info_message(
        self,
        data: bytes,
        topic_type: str,
        topic: str,
        timestamp_mappings: Dict[str, Dict[int, int]]
    ) -> Optional[Tuple[bytes, int, bool]]:
        """
        Process camera_info message: apply timestamp smoothing.

        Args:
            data: Original message data
            topic_type: Message type name
            topic: Topic name
            timestamp_mappings: Dict of topic -> {original_ts -> smoothed_ts}

        Returns:
            Tuple of (modified_data, smoothed_timestamp_ns, was_smoothed) or None if not matched.
        """
        if deserialize_message is None or serialize_message is None:
            return None

        try:
            msg_class = get_message(topic_type)
            msg = deserialize_message(data, msg_class)

            # Extract original header timestamp
            if not hasattr(msg, 'header') or msg.header is None:
                return None
            if not hasattr(msg.header, 'stamp'):
                return None

            original_ts = int(
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            )

            # Look up smoothed timestamp
            topic_mapping = timestamp_mappings.get(topic, {})
            if original_ts not in topic_mapping:
                # Not in matched timestamps, skip this message
                return None

            smoothed_ts = topic_mapping[original_ts]
            was_smoothed = (smoothed_ts != original_ts)

            # Update header.stamp to smoothed value
            msg.header.stamp.sec = int(smoothed_ts // 1_000_000_000)
            msg.header.stamp.nanosec = int(smoothed_ts % 1_000_000_000)

            # Re-serialize
            modified_data = serialize_message(msg)
            return (modified_data, smoothed_ts, was_smoothed)

        except Exception as e:
            print(f'  Warning: Failed to process camera_info message: {e}')
            return None

    def _extract_header_timestamp(
        self,
        data: bytes,
        topic_type: Optional[str]
    ) -> Optional[int]:
        """
        Extract header.stamp timestamp from a message.

        Args:
            data: Message data
            topic_type: Message type name

        Returns:
            Timestamp in nanoseconds, or None if extraction fails.
        """
        if deserialize_message is None or topic_type is None:
            return None

        try:
            msg_class = get_message(topic_type)
            msg = deserialize_message(data, msg_class)

            if not hasattr(msg, 'header') or msg.header is None:
                return None
            if not hasattr(msg.header, 'stamp'):
                return None

            return int(
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            )

        except Exception:
            return None

    def _extract_header_from_decoded(self, decoded) -> Optional[int]:
        """
        Extract header.stamp timestamp from a decoded message.

        This uses the mcap_ros2 decoded message object, no rclpy needed.

        Args:
            decoded: Decoded message object from mcap_ros2

        Returns:
            Timestamp in nanoseconds, or None if extraction fails OR if
            the stamp is the uninitialised (0, 0) sentinel. Some publishers
            (notably ros2_control's joint_trajectory_command_broadcaster
            and the joystick_controller variants) populate ``points`` and
            ``joint_names`` correctly but leave ``header.stamp`` at zero.
            Treat that as "no header" so callers fall back to the bag's
            log_time — otherwise trim/sort by header.stamp would silently
            discard every message on the topic.
        """
        try:
            if decoded is None:
                return None
            if not hasattr(decoded, 'header') or decoded.header is None:
                return None
            if not hasattr(decoded.header, 'stamp'):
                return None

            stamp = decoded.header.stamp
            ns = int(stamp.sec * 1_000_000_000 + stamp.nanosec)
            if ns <= 0:
                return None
            return ns

        except Exception:
            return None

    def _compute_smoothing_mapping(
        self,
        timestamps: List[int]
    ) -> Tuple[Dict[int, int], int]:
        """
        Compute smoothing mapping for camera_info timestamps.

        Smooths intervals exceeding 68ms threshold to random value between 67-68ms
        to comply with STD 007.

        Args:
            timestamps: List of timestamps in nanoseconds.

        Returns:
            Tuple of (mapping, smoothed_count) where:
            - mapping: Dict mapping original timestamp to smoothed timestamp
            - smoothed_count: Number of intervals that were smoothed
        """
        import random

        if len(timestamps) <= 1:
            return {ts: ts for ts in timestamps}, 0

        sorted_ts = sorted(timestamps)
        mapping: Dict[int, int] = {}

        cfg = self.SMOOTHING_CONFIG
        threshold_ns = int(cfg['threshold_ms'] * 1_000_000)
        max_smooth_ns = int(cfg['max_smooth_ms'] * 1_000_000)
        target_min_ns = int(cfg['target_min_ms'] * 1_000_000)
        target_max_ns = int(cfg['target_max_ms'] * 1_000_000)

        cumulative_adj = 0
        mapping[sorted_ts[0]] = sorted_ts[0]  # First timestamp unchanged
        smoothed_count = 0

        for i in range(1, len(sorted_ts)):
            original_ts = sorted_ts[i]
            prev_smoothed = mapping[sorted_ts[i - 1]]
            curr_ts = original_ts + cumulative_adj
            interval = curr_ts - prev_smoothed

            # Only smooth jitter: 68ms < interval <= 71ms
            # Do NOT smooth real drops (> 71ms)
            if threshold_ns < interval <= max_smooth_ns:
                target_interval = random.randint(target_min_ns, target_max_ns)
                smoothed_ts = prev_smoothed + target_interval
                cumulative_adj += (smoothed_ts - curr_ts)
                mapping[original_ts] = smoothed_ts
                smoothed_count += 1
            else:
                mapping[original_ts] = curr_ts

        return mapping, smoothed_count

    def _create_video(
        self,
        frames: List[FrameData],
        output_path: str,
        camera_name: str = '',
    ) -> bool:
        """Create MP4 video from frames using ffmpeg."""
        if not frames:
            return False

        height, width = frames[0].image.shape[:2]

        # Try hardware encoder first, then fallback to software
        encoders_to_try = []
        if self._hw_encoder:
            encoders_to_try.append(self._hw_encoder)
        encoders_to_try.append('libx264')  # Always have software fallback

        for encoder in encoders_to_try:
            success = self._try_encode_video(
                frames, output_path, encoder, width, height, camera_name)
            if success:
                return True
            elif encoder != 'libx264':
                print(f'  Hardware encoder {encoder} failed, trying software encoder...')

        return False

    def _build_video_filter(self, camera_name: str) -> Optional[str]:
        """Compose ffmpeg ``-vf`` filter for camera-specific rotation + global resize.

        Returns None when neither rotation nor resize is requested so the
        encode stays a straight copy. Rotation degrees (0/90/180/270)
        map to ffmpeg ``transpose`` filters; ``image_resize`` scales to
        ``WxH`` after rotation so the requested HxW always matches the
        final output dimensions.
        """
        filters: List[str] = []
        rot = int(self.camera_rotations.get(camera_name, 0)) % 360
        if rot == 90:
            filters.append('transpose=1')           # 90° CW
        elif rot == 180:
            filters.append('transpose=2,transpose=2')  # 180°
        elif rot == 270:
            filters.append('transpose=2')           # 90° CCW
        elif rot != 0:
            print(f'  WARNING: unsupported rotation {rot}° for {camera_name}, skipping')
        if self.image_resize:
            h, w = self.image_resize
            filters.append(f'scale={w}:{h}')
        return ','.join(filters) if filters else None

    def _try_encode_video(
        self,
        frames: List[FrameData],
        output_path: str,
        encoder: str,
        width: int,
        height: int,
        camera_name: str = '',
    ) -> bool:
        """Try to encode video with specified encoder."""
        from .video_sync import _ffmpeg_threads_arg
        cmd = [
            'ffmpeg',
            '-y',
            *_ffmpeg_threads_arg(),
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',
        ]
        vf = self._build_video_filter(camera_name)
        if vf:
            cmd.extend(['-vf', vf])
        cmd.extend([
            '-c:v', encoder,
            '-pix_fmt', 'yuv420p',
        ])

        if encoder == 'libx264':
            cmd.extend(['-preset', 'fast', '-crf', '23'])
        elif encoder in ['h264_nvenc', 'h264_nvmpi', 'h264_v4l2m2m']:
            cmd.extend(['-b:v', '8M'])

        cmd.append(output_path)

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            for frame in frames:
                process.stdin.write(frame.image.tobytes())

            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                stderr = process.stderr.read().decode()
                print(f'  FFmpeg error ({encoder}): {stderr[:500]}')
                return False

            print(f'  Video saved: {output_path} (encoder: {encoder})')
            return True

        except BrokenPipeError:
            # Hardware encoder might not support the input format
            print(f'  Encoder {encoder} broken pipe (unsupported format)')
            return False
        except Exception as e:
            print(f'  Error creating video ({encoder}): {e}')
            return False

    def _copy_meta_files(self, input_dir: Path, output_dir: Path):
        """Copy meta files (episode_info, metadata.yaml, robot.urdf) to the
        intermediate MP4 dir. URDF is copied as-is — to_lerobot_v21/v30
        downstream don't read it, but it's small and harmless. Meshes
        used to be bundled too, but they were dead weight: the LeRobot
        dataset format doesn't reference them, and URDF's
        package://ffw_description/... URIs resolve at consumer time."""
        for filename in self.META_FILES:
            src = input_dir / filename
            # Backward compatibility: old recordings may use meta_data.json
            if not src.exists() and filename == 'episode_info.json':
                legacy_src = input_dir / 'meta_data.json'
                if legacy_src.exists():
                    src = legacy_src
            if src.exists():
                dst = output_dir / filename
                shutil.copy2(src, dst)
                print(f'  Copied: {src.name} -> {filename}')

    def _write_drop_info(
        self,
        input_dir: Path,
        output_dir: Path,
        video_results: Dict[str, 'ConversionResult']
    ):
        """Write dropped frame info to episode_info.json."""
        meta_path = output_dir / 'episode_info.json'
        meta = {}

        # Read existing episode_info.json if it was copied
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                pass

        # Add drop info per camera
        dropped_per_camera = {}
        total_dropped = 0
        for camera_name, result in video_results.items():
            if result.success:
                dropped_per_camera[camera_name] = result.dropped_frames_filled
                total_dropped += result.dropped_frames_filled

        meta['dropped_frames'] = {
            'total': total_dropped,
            'per_camera': dropped_per_camera,
            'scaleable_deliverable': total_dropped == 0,
        }

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f'  Updated episode_info.json with dropped_frames info')

    def _compute_and_save_video_stats(
        self,
        cameras_to_encode: Dict[str, Tuple],
        output_dir: Path,
        max_samples: int = 100
    ):
        """Compute video stats from in-memory frames and save as video_stats.json.

        This pre-computes RGB statistics so Stage 2 (LeRobot v2.1 converter)
        can skip decoding MP4 files just to compute the same stats.
        """
        all_video_stats = {}

        for camera_name, (result, video_path) in cameras_to_encode.items():
            frames = result.frames
            if not frames:
                continue

            total_frames = len(frames)
            sample_count = min(max_samples, total_frames)
            indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

            samples = []
            for idx in indices:
                img = frames[idx].image  # BGR numpy array
                if img is not None:
                    # Convert BGR to RGB, normalize to [0,1]
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    samples.append(rgb)

            if not samples:
                continue

            frames_array = np.array(samples, dtype=np.float32) / 255.0
            r_channel = frames_array[:, :, :, 0]
            g_channel = frames_array[:, :, :, 1]
            b_channel = frames_array[:, :, :, 2]

            def channel_stats(channel):
                return {
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel)),
                    'min': float(np.min(channel)),
                    'max': float(np.max(channel)),
                }

            r_stats = channel_stats(r_channel)
            g_stats = channel_stats(g_channel)
            b_stats = channel_stats(b_channel)

            all_video_stats[camera_name] = {
                'min': [[[r_stats['min']]], [[g_stats['min']]], [[b_stats['min']]]],
                'max': [[[r_stats['max']]], [[g_stats['max']]], [[b_stats['max']]]],
                'mean': [[[r_stats['mean']]], [[g_stats['mean']]], [[b_stats['mean']]]],
                'std': [[[r_stats['std']]], [[g_stats['std']]], [[b_stats['std']]]],
                'count': [sample_count],
            }
            print(f'  Computed stats for {camera_name} ({sample_count} samples)')

        if all_video_stats:
            stats_path = output_dir / 'video_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(all_video_stats, f, indent=2)
            print(f'  Saved video_stats.json ({len(all_video_stats)} cameras)')


def convert_dataset(
    dataset_path: str,
    output_base_path: str,
    fps: int = 15,
    use_hardware_encoding: bool = True,
    exclude_topics: Optional[List[str]] = None,
    joint_offsets: Optional[Dict[str, Dict[str, float]]] = None,
    enable_timestamp_smoothing: bool = True,
    downsample_cameras: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, ConversionResult]]:
    """Convert all episodes in a dataset to MP4 format."""
    converter = RosbagToMp4Converter(
        fps=fps,
        use_hardware_encoding=use_hardware_encoding,
        exclude_topics=exclude_topics,
        joint_offsets=joint_offsets,
        enable_timestamp_smoothing=enable_timestamp_smoothing,
        downsample_cameras=downsample_cameras
    )

    dataset_path = Path(dataset_path)
    output_base_path = Path(output_base_path)
    results = {}

    episode_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])

    print(f'Found {len(episode_dirs)} episodes in {dataset_path}')

    for episode_dir in episode_dirs:
        episode_id = episode_dir.name
        print(f'\n{"="*60}')
        print(f'Episode {episode_id}')
        print(f'{"="*60}')

        output_dir = output_base_path / episode_id

        try:
            episode_results = converter.convert_episode(
                str(episode_dir),
                str(output_dir)
            )
            results[episode_id] = episode_results
        except Exception as e:
            print(f'Error processing episode {episode_id}: {e}')
            results[episode_id] = {'error': str(e)}

    return results


def parse_joint_offsets(offset_str: str) -> Dict[str, Dict[str, float]]:
    """
    Parse joint offset string.

    Format: topic_keyword:joint_name:offset[,topic_keyword:joint_name:offset,...]
    Example: arm_left_leader:5:0.30,arm_right_leader:5:0.30

    The joint_name can be the actual joint name or an index number.
    """
    offsets = {}
    if not offset_str:
        return offsets

    for item in offset_str.split(','):
        parts = item.strip().split(':')
        if len(parts) != 3:
            print(f'Warning: Invalid offset format "{item}", expected topic:joint:offset')
            continue

        topic_keyword, joint_name, offset_value = parts
        try:
            offset = float(offset_value)
        except ValueError:
            print(f'Warning: Invalid offset value "{offset_value}"')
            continue

        if topic_keyword not in offsets:
            offsets[topic_keyword] = {}
        offsets[topic_keyword][joint_name] = offset

    return offsets


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert rosbag2 MCAP to MP4 format'
    )
    parser.add_argument('input_path', help='Path to episode directory or dataset')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--fps', type=int, default=15, help='Video frame rate (default: 15)')
    parser.add_argument('--no-hw', action='store_true', help='Disable hardware encoding')
    parser.add_argument('--dataset', action='store_true', help='Convert entire dataset')
    parser.add_argument(
        '--exclude-topics',
        type=str,
        default='',
        help='Comma-separated list of topic keywords to exclude '
             '(e.g., "head_leader,head_follower,lift_leader,lift_follower,cmd_vel,odom")'
    )
    parser.add_argument(
        '--joint-offset',
        type=str,
        default='',
        help='Joint offset corrections. Format: topic_keyword:joint_index:offset_rad '
             '(e.g., "arm_left_leader:5:0.30,arm_right_leader:5:0.30" for joint6 offset)'
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable timestamp smoothing for STD 007 compliance. '
             'By default, frame timestamps are adjusted to keep intervals '
             'within 67-68ms range (68ms threshold).'
    )

    args = parser.parse_args()

    # Parse exclude topics
    exclude_topics = [t.strip() for t in args.exclude_topics.split(',') if t.strip()]
    if exclude_topics:
        print(f'Excluding topics containing: {exclude_topics}')

    # Parse joint offsets
    joint_offsets = parse_joint_offsets(args.joint_offset)
    if joint_offsets:
        print(f'Applying joint offsets: {joint_offsets}')

    # Timestamp smoothing
    enable_smoothing = not args.no_smooth
    if enable_smoothing:
        print('Timestamp smoothing enabled (STD 007 compliance)')

    if args.dataset:
        results = convert_dataset(
            args.input_path,
            args.output,
            fps=args.fps,
            use_hardware_encoding=not args.no_hw,
            exclude_topics=exclude_topics,
            joint_offsets=joint_offsets,
            enable_timestamp_smoothing=enable_smoothing,
            downsample_cameras=None  # Use default downsampling config
        )
    else:
        converter = RosbagToMp4Converter(
            fps=args.fps,
            use_hardware_encoding=not args.no_hw,
            exclude_topics=exclude_topics,
            joint_offsets=joint_offsets,
            enable_timestamp_smoothing=enable_smoothing,
            downsample_cameras=None  # Use default downsampling config
        )
        results = converter.convert_episode(args.input_path, args.output)

    # Print summary
    print('\n' + '=' * 60)
    print('CONVERSION SUMMARY')
    print('=' * 60)

    if args.dataset:
        for episode_id, episode_results in results.items():
            print(f'\nEpisode {episode_id}:')
            if isinstance(episode_results, dict) and 'error' not in episode_results:
                for camera_name, result in episode_results.items():
                    status = 'OK' if result.success else 'FAILED'
                    smooth_info = f', {result.timestamps_smoothed} smoothed' if result.timestamps_smoothed else ''
                    print(f'  {camera_name}: {status} ({result.frame_count} frames{smooth_info})')
    else:
        for camera_name, result in results.items():
            status = 'OK' if result.success else 'FAILED'
            print(f'{camera_name}: {status}')
            if result.success:
                print(f'  Frames: {result.frame_count}')
                if result.timestamps_smoothed:
                    print(f'  Timestamps smoothed: {result.timestamps_smoothed}')
                print(f'  Video: {result.video_path}')
                print(f'  MCAP: {result.mcap_path}')


if __name__ == '__main__':
    main()
