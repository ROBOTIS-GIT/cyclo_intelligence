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

"""Unit tests for RosbagToLerobotConverter."""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Mock ROS2 modules that are not available outside Docker
for mod_name in [
    "rosbag2_py", "rclpy", "rclpy.serialization",
    "sensor_msgs", "sensor_msgs.msg",
    "trajectory_msgs", "trajectory_msgs.msg",
    "nav_msgs", "nav_msgs.msg",
    "geometry_msgs", "geometry_msgs.msg",
    "rosbag_recorder", "rosbag_recorder.msg",
    "mcap", "mcap.reader", "mcap_ros2", "mcap_ros2.decoder",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Add mock classes to sensor_msgs.msg etc.
sys.modules["sensor_msgs.msg"].JointState = MagicMock
sys.modules["trajectory_msgs.msg"].JointTrajectory = MagicMock
sys.modules["nav_msgs.msg"].Odometry = MagicMock
sys.modules["geometry_msgs.msg"].Twist = MagicMock
sys.modules["rosbag2_py"].SequentialReader = MagicMock
sys.modules["rosbag2_py"].StorageOptions = MagicMock
sys.modules["rosbag2_py"].ConverterOptions = MagicMock
sys.modules["rclpy.serialization"].deserialize_message = MagicMock
sys.modules["mcap.reader"].make_reader = MagicMock
sys.modules["mcap_ros2.decoder"].DecoderFactory = MagicMock

from cyclo_data.converter.to_lerobot_v21 import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverter,
)

# The host test environment may carry a pandas/numpy ABI mismatch. The v3.0
# converter only needs pandas for parquet aggregation, not for the video concat
# tests below, so provide a tiny import stub before loading it.
if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = MagicMock
    sys.modules["pandas"] = pandas_stub

from cyclo_data.converter.to_lerobot_v30 import (
    EpisodeMetadata,
    RosbagToLerobotV30Converter,
    V30ConversionConfig,
)
from cyclo_data.converter import video_sync


class TestConversionConfig(unittest.TestCase):
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self):
        config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path("/tmp/test"),
        )
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.robot_type, "unknown")
        self.assertTrue(config.use_videos)
        self.assertEqual(config.chunks_size, 1000)
        self.assertTrue(config.apply_trim)
        self.assertTrue(config.apply_exclude_regions)

    def test_custom_values(self):
        config = ConversionConfig(
            repo_id="user/my_dataset",
            output_dir=Path("/datasets/output"),
            fps=60,
            robot_type="ai_worker",
            chunks_size=500,
            apply_trim=False,
        )
        self.assertEqual(config.fps, 60)
        self.assertEqual(config.robot_type, "ai_worker")
        self.assertEqual(config.chunks_size, 500)
        self.assertFalse(config.apply_trim)


class TestEpisodeData(unittest.TestCase):
    """Tests for EpisodeData dataclass."""

    def test_default_initialization(self):
        episode = EpisodeData(episode_index=0)
        self.assertEqual(episode.episode_index, 0)
        self.assertEqual(episode.timestamps, [])
        self.assertEqual(episode.observation_state, [])
        self.assertEqual(episode.action, [])
        self.assertEqual(episode.video_files, {})
        self.assertEqual(episode.tasks, [])
        self.assertEqual(episode.length, 0)

    def test_with_data(self):
        episode = EpisodeData(
            episode_index=5,
            timestamps=[0.0, 0.033, 0.066],
            observation_state=[
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1]),
                np.array([1.2, 2.2]),
            ],
            action=[
                np.array([0.1, 0.2]),
                np.array([0.11, 0.21]),
                np.array([0.12, 0.22]),
            ],
            tasks=["pick object"],
            length=3,
        )
        self.assertEqual(episode.episode_index, 5)
        self.assertEqual(len(episode.timestamps), 3)
        self.assertEqual(episode.length, 3)


class TestRosbagToLerobotConverter(unittest.TestCase):
    """Tests for RosbagToLerobotConverter class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path(self.temp_dir),
        )
        self.converter = RosbagToLerobotConverter(self.config)

    def test_initialization(self):
        self.assertEqual(self.converter.config.repo_id, "test/dataset")
        self.assertEqual(self.converter._total_episodes, 0)
        self.assertEqual(self.converter._total_frames, 0)

    def test_get_topic_group_key_basic(self):
        result = self.converter._get_topic_group_key(
            "/robot/arm_left_follower/joint_states", "state")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("follower", result)

    def test_is_in_exclude_region(self):
        exclude_regions = [
            {"start": {"time": 1.0}, "end": {"time": 2.0}},
            {"start": {"time": 5.0}, "end": {"time": 6.0}},
        ]

        self.assertFalse(self.converter._is_in_exclude_region(0.5, exclude_regions))
        self.assertTrue(self.converter._is_in_exclude_region(1.5, exclude_regions))
        self.assertFalse(self.converter._is_in_exclude_region(3.0, exclude_regions))
        self.assertTrue(self.converter._is_in_exclude_region(5.5, exclude_regions))
        self.assertFalse(self.converter._is_in_exclude_region(7.0, exclude_regions))

    def test_find_previous_value_in_list(self):
        messages = [
            (0.0, np.array([1.0])),
            (1.0, np.array([2.0])),
            (2.0, np.array([3.0])),
        ]

        result, staleness = self.converter._find_previous_value_in_list(messages, 0.4)
        np.testing.assert_array_equal(result, np.array([1.0]))

        result, staleness = self.converter._find_previous_value_in_list(messages, 1.5)
        np.testing.assert_array_equal(result, np.array([2.0]))

        result, staleness = self.converter._find_previous_value_in_list(messages, 2.0)
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_find_previous_value_in_list_empty(self):
        result, staleness = self.converter._find_previous_value_in_list([], 1.0)
        self.assertIsNone(result)

    def test_build_features(self):
        episodes = [
            EpisodeData(
                episode_index=0,
                observation_state=[np.array([1.0, 2.0, 3.0])],
                action=[np.array([0.1, 0.2, 0.3])],
                length=1,
            )
        ]
        self.converter._state_joint_names = ["j1", "j2", "j3"]
        self.converter._action_joint_names = ["a1", "a2", "a3"]

        self.converter._build_features(episodes)

        self.assertIn("observation.state", self.converter._features)
        self.assertIn("action", self.converter._features)
        self.assertEqual(self.converter._features["observation.state"]["shape"], (3,))
        self.assertEqual(self.converter._features["action"]["shape"], (3,))

    def test_compute_episode_stats(self):
        episode = EpisodeData(
            episode_index=0,
            observation_state=[
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                np.array([5.0, 6.0]),
            ],
            action=[
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
                np.array([0.5, 0.6]),
            ],
            length=3,
        )

        stats = self.converter._compute_episode_stats(episode)

        self.assertIn("observation.state", stats)
        self.assertIn("action", stats)
        self.assertIn("mean", stats["observation.state"])
        self.assertIn("std", stats["observation.state"])
        self.assertIn("min", stats["observation.state"])
        self.assertIn("max", stats["observation.state"])

        np.testing.assert_array_almost_equal(
            stats["observation.state"]["mean"], [3.0, 4.0]
        )

    def test_serialize_stats(self):
        stats = {
            "observation.state": {
                "mean": np.array([1.0, 2.0]),
                "std": np.array([0.1, 0.2]),
            }
        }

        serialized = self.converter._serialize_stats(stats)

        self.assertIsInstance(serialized["observation.state"]["mean"], list)
        self.assertEqual(serialized["observation.state"]["mean"], [1.0, 2.0])

    def test_resample_to_fps(self):
        episode = EpisodeData(episode_index=0)
        state_messages = [
            (0.0, np.array([1.0])),
            (0.5, np.array([2.0])),
            (1.0, np.array([3.0])),
        ]
        action_messages = [
            (0.0, np.array([0.1])),
            (0.5, np.array([0.2])),
            (1.0, np.array([0.3])),
        ]

        self.converter.config.fps = 10
        result, staleness = self.converter._resample_to_fps(
            episode, state_messages, action_messages, 0.0
        )

        self.assertGreater(result.length, 0)
        self.assertEqual(len(result.timestamps), result.length)
        self.assertEqual(len(result.observation_state), result.length)
        self.assertEqual(len(result.action), result.length)

    def test_is_state_topic_new_naming(self):
        topic_types = {
            "/robot/arm_left_follower/joint_states": "sensor_msgs/msg/JointState",
            "/robot/arm_right_follower/joint_states": "sensor_msgs/msg/JointState",
            "/robot/head_follower/joint_states": "sensor_msgs/msg/JointState",
            "/odom": "nav_msgs/msg/Odometry",
            "/robot/arm_left_leader/joint_states": "sensor_msgs/msg/JointState",
            "/cmd_vel": "geometry_msgs/msg/Twist",
        }
        self.assertTrue(self.converter._is_state_topic(
            "/robot/arm_left_follower/joint_states", topic_types))
        self.assertTrue(self.converter._is_state_topic("/odom", topic_types))
        self.assertFalse(self.converter._is_state_topic(
            "/robot/arm_left_leader/joint_states", topic_types))
        self.assertFalse(self.converter._is_state_topic("/cmd_vel", topic_types))

    def test_is_action_topic_new_naming(self):
        topic_types = {
            "/robot/arm_left_follower/joint_states": "sensor_msgs/msg/JointState",
            "/robot/arm_left_leader/joint_states": "sensor_msgs/msg/JointState",
            "/cmd_vel": "geometry_msgs/msg/Twist",
            "/odom": "nav_msgs/msg/Odometry",
        }
        self.assertTrue(self.converter._is_action_topic(
            "/robot/arm_left_leader/joint_states", topic_types))
        self.assertTrue(self.converter._is_action_topic("/cmd_vel", topic_types))
        self.assertFalse(self.converter._is_action_topic(
            "/robot/arm_left_follower/joint_states", topic_types))
        self.assertFalse(self.converter._is_action_topic("/odom", topic_types))

    def test_merge_state_messages_multi_topic(self):
        state_msgs = {
            "/robot/arm_left_follower/joint_states": [
                (0.0, np.array([1.0, 2.0], dtype=np.float32)),
                (0.01, np.array([1.1, 2.1], dtype=np.float32)),
            ],
            "/robot/arm_right_follower/joint_states": [
                (0.0, np.array([3.0, 4.0], dtype=np.float32)),
                (0.01, np.array([3.1, 4.1], dtype=np.float32)),
            ],
        }
        state_names = {
            "/robot/arm_left_follower/joint_states": ["j1", "j2"],
            "/robot/arm_right_follower/joint_states": ["j3", "j4"],
        }

        merged = self.converter._merge_state_messages(state_msgs, state_names)

        self.assertGreater(len(merged), 0)
        # Merged vector should be 4 dimensions (2 + 2)
        self.assertEqual(len(merged[0][1]), 4)
        self.assertEqual(self.converter._state_joint_names, ["j1", "j2", "j3", "j4"])

    def test_merge_state_messages_with_joint_order(self):
        self.converter._joint_order_by_group = {
            "follower_arm_left": ["j1", "j2"],
            "follower_arm_right": ["j3", "j4"],
        }
        self.converter._state_topic_key_map = {
            "/robot/arm_left_follower/joint_states": "follower_arm_left",
            "/robot/arm_right_follower/joint_states": "follower_arm_right",
        }

        state_msgs = {
            "/robot/arm_left_follower/joint_states": [
                (0.0, np.array([1.0, 2.0], dtype=np.float32)),
            ],
            "/robot/arm_right_follower/joint_states": [
                (0.0, np.array([3.0, 4.0], dtype=np.float32)),
            ],
        }
        state_names = {
            "/robot/arm_left_follower/joint_states": ["j1", "j2"],
            "/robot/arm_right_follower/joint_states": ["j3", "j4"],
        }

        merged = self.converter._merge_state_messages(state_msgs, state_names)

        self.assertEqual(len(merged), 1)
        np.testing.assert_array_almost_equal(
            merged[0][1], [1.0, 2.0, 3.0, 4.0]
        )

    def test_get_topic_group_key(self):
        self.assertEqual(
            self.converter._get_topic_group_key(
                "/robot/arm_left_follower/joint_states", "state"),
            "follower_arm_left"
        )
        self.assertEqual(
            self.converter._get_topic_group_key(
                "/robot/head_leader/joint_states", "action"),
            "leader_head"
        )
        self.assertEqual(
            self.converter._get_topic_group_key("/odom", "state"),
            "follower_mobile"
        )
        self.assertEqual(
            self.converter._get_topic_group_key("/cmd_vel", "action"),
            "leader_mobile"
        )

    def test_update_config_from_robot_config_with_grouped_joint_order(self):
        robot_config = {
            "robot_type": "ffw_sg2_rev1",
            "state_topics": {
                "follower_arm_left": "/robot/arm_left_follower/joint_states",
                "follower_arm_right": "/robot/arm_right_follower/joint_states",
            },
            "action_topics": {
                "leader_arm_left": "/robot/arm_left_leader/joint_states",
                "leader_arm_right": "/robot/arm_right_leader/joint_states",
            },
            "joint_order": {
                "follower_arm_left": ["j1", "j2"],
                "follower_arm_right": ["j3", "j4"],
                "leader_arm_left": ["j1", "j2"],
                "leader_arm_right": ["j3", "j4"],
            },
        }

        self.converter._update_config_from_robot_config(robot_config)

        self.assertEqual(self.converter.config.robot_type, "ffw_sg2_rev1")
        self.assertIn("follower_arm_left", self.converter._joint_order_by_group)
        self.assertEqual(
            self.converter._joint_order_by_group["follower_arm_left"], ["j1", "j2"])
        self.assertEqual(
            self.converter._state_topic_key_map["/robot/arm_left_follower/joint_states"],
            "follower_arm_left"
        )
        self.assertEqual(self.converter._joint_order, ["j1", "j2", "j3", "j4", "j1", "j2", "j3", "j4"])


class TestInfoJsonGeneration(unittest.TestCase):
    """Tests for info.json generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path(self.temp_dir),
            fps=30,
            robot_type="test_robot",
        )
        self.converter = RosbagToLerobotConverter(self.config)

    def test_write_info_json(self):
        self.converter._features = {
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
            "action": {"dtype": "float32", "shape": (6,), "names": None},
        }
        self.converter._tasks = {0: "test_task"}
        self.converter._total_episodes = 5
        self.converter._total_frames = 500

        (Path(self.temp_dir) / "meta").mkdir(parents=True, exist_ok=True)

        self.converter._write_info_json()

        info_path = Path(self.temp_dir) / "meta" / "info.json"
        self.assertTrue(info_path.exists())

        with open(info_path) as f:
            info = json.load(f)

        self.assertEqual(info["codebase_version"], "v2.1")
        self.assertEqual(info["robot_type"], "test_robot")
        self.assertEqual(info["fps"], 30)
        self.assertEqual(info["total_episodes"], 5)
        self.assertEqual(info["total_frames"], 500)
        self.assertIn("features", info)


class TestRosbagToLerobotV30VideoConcat(unittest.TestCase):
    """Tests for v3.0 aggregate video writing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = V30ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path(self.temp_dir),
            fps=15,
        )
        self.converter = RosbagToLerobotV30Converter(self.config)
        self.converter._episode_metadata_list = [
            EpisodeMetadata(episode_index=0, length=2, tasks=["task"]),
            EpisodeMetadata(episode_index=1, length=3, tasks=["task"]),
        ]
        self.input_a = Path(self.temp_dir) / "episode_000000.mp4"
        self.input_b = Path(self.temp_dir) / "episode_000001.mp4"
        self.input_a.touch()
        self.input_b.touch()

    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_concatenate_videos_uses_cfr_reencode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self.converter._get_video_frame_count = MagicMock(return_value=5)
        self.converter._probe_video_fps = MagicMock(return_value=15.0)

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.cam_wrist_right",
            0,
            0,
            [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
        )

        cmd = mock_run.call_args.args[0]
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        self.assertIn("-an", cmd)
        self.assertIn("-r", cmd)
        self.assertIn("fps=15,setpts=N/(15*TB)", cmd)
        self.assertNotIn("copy", cmd)

    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_single_video_also_uses_reencode_path(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self.converter._get_video_frame_count = MagicMock(return_value=2)
        self.converter._probe_video_fps = MagicMock(return_value=15.0)

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.cam_head_left",
            0,
            0,
            [(0, self.input_a, 0.0)],
        )

        cmd = mock_run.call_args.args[0]
        self.assertIn("libx264", cmd)
        self.assertNotIn("copy", cmd)

    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_ffmpeg_failure_raises_with_stderr(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="bad concat")

        with self.assertRaisesRegex(RuntimeError, "bad concat"):
            self.converter._concatenate_videos(
                Path(self.temp_dir),
                "observation.images.cam_head_left",
                0,
                0,
                [(0, self.input_a, 0.0)],
            )

    def test_validate_aggregated_video_rejects_frame_mismatch(self):
        self.converter._get_video_frame_count = MagicMock(return_value=4)

        with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
            self.converter._validate_aggregated_video(self.input_a, 5)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe are required for integration video concat test",
    )
    def test_synthetic_video_concat_frame_count_and_fps(self):
        try:
            import cv2
        except Exception:
            self.skipTest("cv2 is required for integration video concat test")

        input_dir = Path(self.temp_dir) / "inputs"
        input_dir.mkdir()
        video_a = input_dir / "a.mp4"
        video_b = input_dir / "b.mp4"
        self._write_synthetic_video(cv2, video_a, 2, (0, 0, 255))
        self._write_synthetic_video(cv2, video_b, 3, (0, 255, 0))

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.cam_head_left",
            0,
            0,
            [(0, video_a, 2 / 15), (1, video_b, 3 / 15)],
        )

        output = (
            Path(self.temp_dir)
            / "videos/observation.images.cam_head_left/chunk-000/file-000.mp4"
        )
        self.assertEqual(self.converter._get_video_frame_count(output), 5)
        self.assertAlmostEqual(self.converter._probe_video_fps(output), 15.0, places=2)

    def _write_synthetic_video(self, cv2, path: Path, frames: int, color):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.config.fps),
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for idx in range(frames):
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            cv2.putText(
                frame,
                str(idx),
                (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
        writer.release()


class TestVideoSyncTempResources(unittest.TestCase):
    """Tests for video_sync temporary-directory resource controls."""

    def test_default_temp_parent_is_output_adjacent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "videos" / "cam_synced.mp4"
            parent, cleanup = video_sync._resolve_tmp_parent(out)
            try:
                self.assertEqual(parent, out.parent / ".video_sync_tmp")
                self.assertTrue(parent.exists())
                self.assertTrue(cleanup)
            finally:
                video_sync._cleanup_tmp_parent(parent, cleanup)
            self.assertFalse(parent.exists())

    def test_temp_parent_env_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "custom_tmp"
            out = Path(tmpdir) / "videos" / "cam_synced.mp4"
            with patch.dict(
                os.environ, {"CYCLO_VIDEO_SYNC_TMPDIR": str(override)}
            ):
                parent, cleanup = video_sync._resolve_tmp_parent(out)
            self.assertEqual(parent, override)
            self.assertTrue(parent.exists())
            self.assertFalse(cleanup)

    def test_min_free_space_gate_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            fake_usage = types.SimpleNamespace(free=10 * 1024 * 1024)
            with patch.dict(
                os.environ, {"CYCLO_VIDEO_SYNC_MIN_FREE_MB": "20"}
            ), patch.object(
                video_sync.shutil, "disk_usage", return_value=fake_usage
            ):
                with self.assertRaisesRegex(RuntimeError, "requires at least 20"):
                    video_sync._check_tmp_free_space(parent)

    def test_remux_uses_resolved_temp_parent_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "videos" / "cam_synced.mp4"
            input_mp4 = Path(tmpdir) / "input.mp4"
            input_mp4.write_bytes(b"fake")
            seen = {}

            class FakeTemporaryDirectory:
                def __init__(self, prefix=None, dir=None):
                    seen["prefix"] = prefix
                    seen["dir"] = Path(dir)
                    self.path = Path(dir) / f"{prefix}fake"

                def __enter__(self):
                    self.path.mkdir(parents=True, exist_ok=True)
                    return str(self.path)

                def __exit__(self, exc_type, exc, tb):
                    shutil.rmtree(self.path, ignore_errors=True)

            def fake_inner(**kwargs):
                raise RuntimeError("stop before ffmpeg")

            with patch.object(
                video_sync, "_ffmpeg", return_value="ffmpeg"
            ), patch.object(
                video_sync.tempfile, "TemporaryDirectory", FakeTemporaryDirectory
            ), patch.object(
                video_sync, "_remux_selected_frames_in_tmp", side_effect=fake_inner
            ):
                with self.assertRaisesRegex(RuntimeError, "stop before ffmpeg"):
                    video_sync.remux_selected_frames(
                        input_mp4, [0], out, target_fps=15
                    )

            self.assertEqual(seen["prefix"], "video_sync_")
            self.assertEqual(seen["dir"], out.parent / ".video_sync_tmp")
            self.assertFalse((out.parent / ".video_sync_tmp").exists())

    def test_h264_extract_regenerates_monotonic_pts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_mp4 = tmp / "input.mp4"
            output_mp4 = tmp / "out.mp4"
            frames_dir = tmp / "frames"
            seq_dir = tmp / "seq"
            input_mp4.write_bytes(b"fake")
            frames_dir.mkdir()
            seq_dir.mkdir()
            commands = []

            def fake_run(cmd, *args, **kwargs):
                commands.append(list(cmd))
                if "ffprobe" in cmd[0]:
                    return types.SimpleNamespace(stdout="h264\n", returncode=0)
                if str(frames_dir / "f_%08d.jpg") in cmd:
                    for idx in range(2):
                        (frames_dir / f"f_{idx:08d}.jpg").write_bytes(
                            b"jpeg"
                        )
                    return types.SimpleNamespace(stdout="", returncode=0)
                if str(output_mp4) in cmd:
                    output_mp4.write_bytes(b"mp4")
                    return types.SimpleNamespace(stdout="", returncode=0)
                return types.SimpleNamespace(stdout="", returncode=0)

            with patch.object(
                video_sync.subprocess, "run", side_effect=fake_run
            ), patch.object(
                video_sync, "_h264_encoder", return_value=("libx264", [])
            ):
                video_sync._remux_selected_frames_in_tmp(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1],
                    output_mp4=output_mp4,
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                    frames_dir=frames_dir,
                    seq_dir=seq_dir,
                )

            extract_cmd = next(
                cmd for cmd in commands
                if str(frames_dir / "f_%08d.jpg") in cmd
            )
            self.assertIn("-vf", extract_cmd)
            self.assertIn("setpts=N/TB", extract_cmd)
            self.assertIn("-fps_mode", extract_cmd)
            self.assertIn("passthrough", extract_cmd)


if __name__ == "__main__":
    unittest.main()
