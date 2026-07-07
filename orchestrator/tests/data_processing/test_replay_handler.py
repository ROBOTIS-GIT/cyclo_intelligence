import sys
import types
from unittest.mock import MagicMock


for mod_name in [
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "rclpy",
    "rclpy.serialization",
    "rosbag2_py",
    "sensor_msgs",
    "sensor_msgs.msg",
    "trajectory_msgs",
    "trajectory_msgs.msg",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

sys.modules["geometry_msgs.msg"].Twist = MagicMock
sys.modules["nav_msgs.msg"].Odometry = MagicMock
sys.modules["rclpy.serialization"].deserialize_message = MagicMock
sys.modules["rosbag2_py"].ConverterOptions = MagicMock
sys.modules["rosbag2_py"].SequentialReader = MagicMock
sys.modules["rosbag2_py"].StorageFilter = MagicMock
sys.modules["rosbag2_py"].StorageOptions = MagicMock
sys.modules["sensor_msgs.msg"].JointState = MagicMock
sys.modules["trajectory_msgs.msg"].JointTrajectory = MagicMock

from cyclo_data.recorder.replay_handler import ReplayDataHandler


def test_build_frame_counts_aggregates_segment_camera_names():
    handler = ReplayDataHandler()

    frame_counts = handler._build_frame_counts(
        ["cam_left_head", "cam_right_head", "cam_left_head", "cam_right_head"],
        [],
        {},
        [
            {
                "frame_counts": {
                    "cam_left_head": 126,
                    "cam_right_head": 124,
                },
            },
            {
                "frame_counts": {
                    "cam_left_head": 118,
                    "cam_right_head": 117,
                },
            },
        ],
    )

    assert frame_counts == {
        "cam_left_head": 244,
        "cam_right_head": 241,
    }


def test_build_frame_counts_accumulates_duplicate_video_names_without_segments():
    handler = ReplayDataHandler()

    frame_counts = handler._build_frame_counts(
        ["cam_left_head", "cam_left_head"],
        ["videos/0_0/cam_left_head", "videos/0_1/cam_left_head"],
        {
            "videos/0_0/cam_left_head": [(0, 0.0), (1, 0.1)],
            "videos/0_1/cam_left_head": [(0, 1.0), (1, 1.1), (2, 1.2)],
        },
    )

    assert frame_counts == {"cam_left_head": 5}


def test_target_replay_sample_hz_ceilings_to_next_5hz():
    handler = ReplayDataHandler()

    assert handler._target_replay_sample_hz([14.1]) == 15
    assert handler._target_replay_sample_hz([15.0]) == 15
    assert handler._target_replay_sample_hz([15.1]) == 20
    assert handler._target_replay_sample_hz(
        [0, None],
        [{"video_fps": [29.8, "bad"]}],
    ) == 30


def test_timestamp_bucket_key_uses_camera_interval():
    handler = ReplayDataHandler()
    bucket_s = 1.0 / 15.0

    assert handler._timestamp_bucket_key(0.010, bucket_s, 0.0) == 0.0
    assert handler._timestamp_bucket_key(0.066, bucket_s, 0.0) == 0.0
    assert handler._timestamp_bucket_key(0.067, bucket_s, 0.0) == round(bucket_s, 9)
    assert handler._timestamp_bucket_key(0.134, bucket_s, 0.0) == round(
        bucket_s * 2,
        9,
    )


def test_timestamp_bucket_key_keeps_legacy_10ms_without_camera_fps():
    handler = ReplayDataHandler()

    assert handler._timestamp_bucket_key(1.234, None, 0.0) == 1.23
