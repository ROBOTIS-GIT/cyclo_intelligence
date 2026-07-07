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

import json
import os
import tempfile
from pathlib import Path
import pytest
import yaml

# D5/D6 moved this module from orchestrator/data_processing/ to
# cyclo_data/reader/, then D17 nested it under cyclo_data/cyclo_data/.
# The previous importlib-from-file shim is obsolete — both
# cyclo_data.reader and its __init__.py are pure Python with no ROS2
# imports, so a plain import works after `source install/setup.bash`.
from cyclo_data.reader.metadata_manager import MetadataManager


class TestMetadataManager:
    @pytest.fixture
    def temp_bag_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self):
        return MetadataManager()

    def test_load_robot_config_not_found(self, manager, temp_bag_dir):
        result = manager.load_robot_config(temp_bag_dir)
        assert result is None

    def test_load_robot_config_success(self, manager, temp_bag_dir):
        config_data = {
            "robot_type": "ai_worker",
            "camera_topics": {"cam_head": "/zed/image"},
        }
        config_path = temp_bag_dir / "robot_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = manager.load_robot_config(temp_bag_dir)
        assert result is not None
        assert result["robot_type"] == "ai_worker"
        assert "camera_topics" in result

    def test_save_robot_config(self, manager, temp_bag_dir):
        config_data = {"robot_type": "test_robot", "task_markers": []}

        success = manager.save_robot_config(temp_bag_dir, config_data)
        assert success is True

        config_path = temp_bag_dir / "robot_config.yaml"
        assert config_path.exists()

        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["robot_type"] == "test_robot"

    def test_get_directory_size(self, manager, temp_bag_dir):
        test_file = temp_bag_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("x" * 100)

        size = manager.get_directory_size(temp_bag_dir)
        assert size >= 100

    def test_get_recording_date_returns_utc_iso_string(self, manager, temp_bag_dir):
        metadata = {
            "rosbag2_bagfile_information": {
                "starting_time": {
                    "nanoseconds_since_epoch": 1736692200000000000,
                },
            },
        }
        (temp_bag_dir / "metadata.yaml").write_text(
            yaml.safe_dump(metadata), encoding="utf-8"
        )

        assert manager.get_recording_date(temp_bag_dir) == "2025-01-12T14:30:00Z"

    def test_get_task_markers_empty(self, manager, temp_bag_dir):
        markers = manager.get_task_markers(temp_bag_dir)
        assert markers == []

    def test_get_task_markers_success(self, manager, temp_bag_dir):
        config_data = {
            "task_markers": [
                {"frame": 0, "time": 0.0, "instruction": "Start"},
                {"frame": 100, "time": 3.33, "instruction": "Pick up"},
            ]
        }
        config_path = temp_bag_dir / "robot_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        markers = manager.get_task_markers(temp_bag_dir)
        assert len(markers) == 2
        assert markers[0]["instruction"] == "Start"

    def test_update_task_markers(self, manager, temp_bag_dir):
        manager.save_robot_config(temp_bag_dir, {})
        markers = [
            {"frame": 50, "time": 1.5, "instruction": "Move"},
            {"frame": 10, "time": 0.3, "instruction": "Init"},
        ]

        result = manager.update_task_markers(temp_bag_dir, markers)
        assert result["success"] is True

        loaded_markers = manager.get_task_markers(temp_bag_dir)
        assert len(loaded_markers) == 2
        assert loaded_markers[0]["frame"] == 10
        assert loaded_markers[1]["frame"] == 50

    def test_update_task_markers_with_trim_points(self, manager, temp_bag_dir):
        manager.save_robot_config(temp_bag_dir, {})
        markers = [{"frame": 0, "time": 0.0, "instruction": "Start"}]
        trim_points = {
            "start": {"time": 1.0, "frame": 30},
            "end": {"time": 10.0, "frame": 300},
        }

        result = manager.update_task_markers(
            temp_bag_dir, markers, trim_points=trim_points
        )
        assert result["success"] is True

        loaded_trim = manager.get_trim_points(temp_bag_dir)
        assert loaded_trim is not None
        assert loaded_trim["start"]["time"] == 1.0

    def test_get_episode_segments_keeps_timestamp_segments(
        self, manager, temp_bag_dir
    ):
        info = {
            "segment_time_unit": "seconds",
            "segment_time_reference": "episode_start",
            "fps": 15,
            "segments": [
                {
                    "sub_task_index": 0,
                    "sub_task_description": "",
                    "sub_task_instruction": "Move",
                    "frame_duration": [0.5, 2.25],
                }
            ],
        }
        (temp_bag_dir / "episode_info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )

        segments = manager.get_episode_segments(temp_bag_dir, duration=3.0)

        assert len(segments) == 1
        assert segments[0]["frame_duration"] == [0.5, 2.25]
        assert segments[0]["sub_task_instruction"] == "Move"

    def test_get_episode_segments_converts_legacy_frame_segments(
        self, manager, temp_bag_dir
    ):
        info = {
            "fps": 15,
            "segments": [
                {
                    "sub_task_index": -1,
                    "sub_task_description": "",
                    "sub_task_instruction": "Pick",
                    "frame_duration": [0, 552],
                },
                {
                    "sub_task_index": -1,
                    "sub_task_description": "",
                    "sub_task_instruction": "Move",
                    "frame_duration": [552, 799],
                },
            ],
        }
        (temp_bag_dir / "episode_info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )

        segments = manager.get_episode_segments(temp_bag_dir, duration=83.5)

        assert len(segments) == 2
        assert segments[0]["frame_duration"] == [0.0, 36.8]
        assert segments[1]["frame_duration"][0] == 36.8
        assert round(segments[1]["frame_duration"][1], 6) == round(799 / 15, 6)

    def test_get_episode_segments_skips_malformed_segments(
        self, manager, temp_bag_dir
    ):
        info = {
            "segment_time_unit": "seconds",
            "segments": [
                {"frame_duration": [0.0]},
                {"frame_duration": ["bad", 1.0]},
                {"sub_task_instruction": "Good", "frame_duration": [1.0, 2.0]},
            ],
        }
        (temp_bag_dir / "episode_info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )

        segments = manager.get_episode_segments(temp_bag_dir)

        assert len(segments) == 1
        assert segments[0]["sub_task_instruction"] == "Good"

    def test_update_task_markers_writes_timestamp_segments(
        self, manager, temp_bag_dir
    ):
        (temp_bag_dir / "episode_info.json").write_text(
            json.dumps({"task_instruction": "task"}), encoding="utf-8"
        )
        segments = [
            {
                "sub_task_index": 0,
                "sub_task_description": "",
                "sub_task_instruction": "Pick",
                "frame_duration": [0.0, 1.5],
            }
        ]

        result = manager.update_task_markers(
            temp_bag_dir,
            [],
            segments=segments,
        )

        assert result["success"] is True
        saved = json.loads((temp_bag_dir / "episode_info.json").read_text())
        assert saved["segment_time_unit"] == "seconds"
        assert saved["segment_time_reference"] == "episode_start"
        assert saved["segments"][0]["frame_duration"] == [0.0, 1.5]
        assert not (temp_bag_dir / "robot_config.yaml").exists()

    def test_update_task_markers_keeps_existing_robot_config(
        self, manager, temp_bag_dir
    ):
        manager.save_robot_config(temp_bag_dir, {"robot_type": "test_robot"})

        result = manager.update_task_markers(
            temp_bag_dir,
            [{"frame": 1, "time": 0.1, "instruction": "Move"}],
        )

        assert result["success"] is True
        saved = yaml.safe_load((temp_bag_dir / "robot_config.yaml").read_text())
        assert saved["robot_type"] == "test_robot"
        assert saved["task_markers"][0]["instruction"] == "Move"

    def test_get_camera_name_map(self, manager, temp_bag_dir):
        config_data = {
            "camera_topics": {
                "cam_head": "/zed/image",
                "cam_left_wrist": "/camera_left/image",
            }
        }
        config_path = temp_bag_dir / "robot_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        camera_map = manager.get_camera_name_map(temp_bag_dir)
        assert "/zed/image" in camera_map
        assert camera_map["/zed/image"] == "cam_head"

    def test_is_action_topic_from_config(self, manager, temp_bag_dir):
        config_data = {
            "action_topics": {
                "left_arm": "/leader/left/trajectory",
                "right_arm": "/leader/right/trajectory",
            }
        }
        config_path = temp_bag_dir / "robot_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        assert manager.is_action_topic("/leader/left/trajectory", temp_bag_dir) is True
        assert manager.is_action_topic("/follower/left/state", temp_bag_dir) is False

    def test_is_action_topic_fallback(self, manager, temp_bag_dir):
        assert manager.is_action_topic("/leader/left/trajectory", temp_bag_dir) is True
        assert manager.is_action_topic("/some_action_topic", temp_bag_dir) is True
        assert manager.is_action_topic("/some_state_topic", temp_bag_dir) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
