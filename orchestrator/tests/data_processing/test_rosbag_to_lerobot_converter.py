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
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow.parquet as pq

# Mock ROS2 modules that are not available outside Docker
for mod_name in [
    "rosbag2_py", "rclpy", "rclpy.serialization",
    "sensor_msgs", "sensor_msgs.msg",
    "trajectory_msgs", "trajectory_msgs.msg",
    "nav_msgs", "nav_msgs.msg",
    "geometry_msgs", "geometry_msgs.msg",
    "rosbag_recorder", "rosbag_recorder.msg",
    "mcap", "mcap.reader",
    "mcap_ros2", "mcap_ros2.decoder",
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
from cyclo_data.converter.to_lerobot_v30 import (
    RosbagToLerobotV30Converter,
    V30ConversionConfig,
)


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

    def test_merge_action_messages_with_joint_order(self):
        self.converter._joint_order_by_group = {
            "leader_arm_left": ["j2", "j1"],
            "leader_arm_right": ["j4", "j3"],
        }
        self.converter._action_topic_key_map = {
            "/robot/arm_left_leader/joint_states": "leader_arm_left",
            "/robot/arm_right_leader/joint_states": "leader_arm_right",
        }

        action_msgs = {
            "/robot/arm_left_leader/joint_states": [
                (0.0, np.array([1.0, 2.0], dtype=np.float32)),
                (0.01, np.array([1.1, 2.1], dtype=np.float32)),
            ],
            "/robot/arm_right_leader/joint_states": [
                (0.0, np.array([3.0, 4.0], dtype=np.float32)),
                (0.01, np.array([3.1, 4.1], dtype=np.float32)),
            ],
        }
        action_names = {
            "/robot/arm_left_leader/joint_states": ["j1", "j2"],
            "/robot/arm_right_leader/joint_states": ["j3", "j4"],
        }

        merged = self.converter._merge_action_messages(action_msgs, action_names)

        self.assertEqual(len(merged), 2)
        self.assertEqual(self.converter._action_joint_names, ["j2", "j1", "j4", "j3"])
        np.testing.assert_array_almost_equal(
            merged[0][1], [2.0, 1.0, 4.0, 3.0]
        )

    def test_merge_state_missing_joint_skips_samples_with_single_warning(self):
        self.converter._joint_order_by_group = {
            "follower_arm_left": ["j1", "missing_joint"],
        }
        self.converter._state_topic_key_map = {
            "/robot/arm_left_follower/joint_states": "follower_arm_left",
        }

        state_msgs = {
            "/robot/arm_left_follower/joint_states": [
                (0.0, np.array([1.0, 2.0], dtype=np.float32)),
                (0.01, np.array([1.1, 2.1], dtype=np.float32)),
            ],
        }
        state_names = {
            "/robot/arm_left_follower/joint_states": ["j1", "j2"],
        }

        with patch.object(self.converter, "_log_warning") as warn:
            merged = self.converter._merge_state_messages(state_msgs, state_names)

        self.assertEqual(merged, [])
        warn.assert_called_once()
        self.assertIn("missing_joint", warn.call_args[0][0])

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


class TestRosbagToLerobotV30Converter(unittest.TestCase):
    """Tests for v3.0 conversion writer helpers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = V30ConversionConfig(
            repo_id="test/dataset_v30",
            output_dir=Path(self.temp_dir),
            fps=10,
        )
        self.converter = RosbagToLerobotV30Converter(self.config)

    def test_write_aggregated_data_uses_equivalent_columns_and_metadata(self):
        self.converter._task_to_index = {"task_a": 4, "task_b": 7}

        episodes = [
            EpisodeData(
                episode_index=0,
                timestamps=[0.0, 0.1],
                observation_state=[
                    np.array([1.0, 2.0], dtype=np.float32),
                    np.array([1.1, 2.1], dtype=np.float32),
                ],
                action=[
                    np.array([0.1, 0.2], dtype=np.float32),
                    np.array([0.11, 0.21], dtype=np.float32),
                ],
                tasks=["task_a"],
                length=2,
            ),
            EpisodeData(
                episode_index=1,
                timestamps=[0.0, 0.1, 0.2],
                observation_state=[
                    np.array([3.0, 4.0], dtype=np.float32),
                    np.array([3.1, 4.1], dtype=np.float32),
                    np.array([3.2, 4.2], dtype=np.float32),
                ],
                action=[
                    np.array([0.3, 0.4], dtype=np.float32),
                    np.array([0.31, 0.41], dtype=np.float32),
                    np.array([0.32, 0.42], dtype=np.float32),
                ],
                tasks=["task_b"],
                length=3,
            ),
        ]

        self.converter._write_aggregated_data(episodes)

        parquet_path = (
            Path(self.temp_dir) / "data" / "chunk-000" / "file-000.parquet"
        )
        self.assertTrue(parquet_path.exists())

        table = pq.read_table(parquet_path)
        self.assertEqual(table.num_rows, 5)
        self.assertEqual(
            table.column_names,
            [
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
                "observation.state",
                "action",
            ],
        )
        self.assertEqual(table["frame_index"].to_pylist(), [0, 1, 0, 1, 2])
        self.assertEqual(table["episode_index"].to_pylist(), [0, 0, 1, 1, 1])
        self.assertEqual(table["index"].to_pylist(), [0, 1, 2, 3, 4])
        self.assertEqual(table["task_index"].to_pylist(), [4, 4, 7, 7, 7])
        self.assertEqual(table["observation.state"].to_pylist()[0], [1.0, 2.0])
        np.testing.assert_allclose(table["action"].to_pylist()[4], [0.32, 0.42])

        self.assertEqual(len(self.converter._episode_metadata_list), 2)
        first, second = self.converter._episode_metadata_list
        self.assertEqual(first.dataset_from_index, 0)
        self.assertEqual(first.dataset_to_index, 2)
        self.assertEqual(second.dataset_from_index, 2)
        self.assertEqual(second.dataset_to_index, 5)
        self.assertEqual(self.converter._total_frames, 5)
        self.assertEqual(self.converter._total_episodes, 2)


if __name__ == "__main__":
    unittest.main()
