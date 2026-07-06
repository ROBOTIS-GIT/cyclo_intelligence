import sys
import types
from unittest.mock import MagicMock

import pytest

for mod_name in [
    "mcap",
    "mcap.reader",
    "mcap_ros2",
    "mcap_ros2.decoder",
    "rosbag2_py",
    "rclpy",
    "rclpy.serialization",
    "sensor_msgs",
    "sensor_msgs.msg",
    "trajectory_msgs",
    "trajectory_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "geometry_msgs",
    "geometry_msgs.msg",
    "rosbag_recorder",
    "rosbag_recorder.msg",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

sys.modules["mcap.reader"].make_reader = MagicMock
sys.modules["mcap_ros2.decoder"].DecoderFactory = MagicMock
sys.modules["rosbag2_py"].SequentialReader = MagicMock
sys.modules["rosbag2_py"].StorageOptions = MagicMock
sys.modules["rosbag2_py"].ConverterOptions = MagicMock
sys.modules["rclpy.serialization"].deserialize_message = MagicMock
sys.modules["sensor_msgs.msg"].JointState = MagicMock
sys.modules["trajectory_msgs.msg"].JointTrajectory = MagicMock
sys.modules["nav_msgs.msg"].Odometry = MagicMock
sys.modules["geometry_msgs.msg"].Twist = MagicMock

from cyclo_data.converter.base_converter import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
)


def test_segment_video_discovery_rejects_legacy_renumbered_dir(tmp_path):
    bag_dir = tmp_path / "53"
    video_dir = bag_dir / "videos" / "153_0"
    video_dir.mkdir(parents=True)
    (bag_dir / "53_0.mcap").write_bytes(b"mcap")
    (video_dir / "cam_left_head.mp4").write_bytes(b"mp4")

    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/repo", output_dir=tmp_path / "out")
    )

    with pytest.raises(FileNotFoundError, match="expected video segment directory"):
        converter._find_segment_video_files(bag_dir, "53_0")


def test_segment_video_discovery_allows_flat_videos_root(tmp_path):
    bag_dir = tmp_path / "53"
    video_dir = bag_dir / "videos"
    video_dir.mkdir(parents=True)
    (bag_dir / "53_0.mcap").write_bytes(b"mcap")
    mp4_path = video_dir / "cam_left_head.mp4"
    mp4_path.write_bytes(b"mp4")
    (video_dir / "cam_left_head_synced.mp4").write_bytes(b"synced")

    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/repo", output_dir=tmp_path / "out")
    )

    video_files = converter._find_segment_video_files(bag_dir, "53_0")

    assert video_files == {"cam_left_head": mp4_path}


def test_segment_sync_cache_base_dir_namespaces_flat_video_roots(tmp_path):
    bag_dir = tmp_path / "53"
    videos_root = bag_dir / "videos"
    segment_dir = videos_root / "53_0"
    videos_root.mkdir(parents=True)
    segment_dir.mkdir()
    segment_mcap = bag_dir / "53_0.mcap"
    segment_mcap.write_bytes(b"mcap")

    assert RosbagToLerobotConverterBase._video_sync_cache_base_dir(
        segment_mcap,
        videos_root,
    ) == videos_root / ".cyclo_synced" / "53_0"
    assert RosbagToLerobotConverterBase._video_sync_cache_base_dir(
        segment_mcap,
        segment_dir,
    ) == segment_dir
    assert RosbagToLerobotConverterBase._video_sync_cache_base_dir(
        bag_dir,
        videos_root,
    ) == videos_root


def test_stitch_subtask_videos_rejects_duplicate_segment_sources(tmp_path):
    duplicate_video = tmp_path / "videos" / "cam_left_head_synced.mp4"
    duplicate_video.parent.mkdir()
    duplicate_video.write_bytes(b"mp4")
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/repo", output_dir=tmp_path / "out")
    )
    episodes = [
        EpisodeData(
            episode_index=0,
            length=3,
            video_files={"cam_left_head": duplicate_video},
        ),
        EpisodeData(
            episode_index=0,
            length=4,
            video_files={"cam_left_head": duplicate_video},
        ),
    ]

    assert converter._stitch_subtask_videos(0, episodes) == {}


def test_merge_segment_frame_reuse_reports_offsets_runs(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/repo", output_dir=tmp_path / "out", fps=15)
    )
    first = EpisodeData(
        episode_index=0,
        length=3,
        frame_reuse_reports=[{
            "episode_index": 0,
            "camera": "cam_left_head",
            "target_fps": 15,
            "time_source": "header",
            "total_target_frames": 3,
            "total_source_frames": 2,
            "reused_target_frames": 1,
            "clamped_before_first_count": 0,
            "runs": [{
                "target_start_frame": 1,
                "target_end_frame": 1,
                "count": 1,
                "source_frame_index": 0,
            }],
        }],
    )
    second = EpisodeData(
        episode_index=0,
        length=4,
        frame_reuse_reports=[{
            "episode_index": 0,
            "camera": "cam_left_head",
            "target_fps": 15,
            "time_source": "header",
            "total_target_frames": 4,
            "total_source_frames": 3,
            "reused_target_frames": 2,
            "clamped_before_first_count": 0,
            "runs": [{
                "target_start_frame": 0,
                "target_end_frame": 1,
                "count": 2,
                "source_frame_index": 0,
            }],
        }],
    )

    reports = converter._merge_segment_frame_reuse_reports(9, [first, second])

    assert len(reports) == 1
    assert reports[0]["episode_index"] == 9
    assert reports[0]["total_target_frames"] == 7
    assert reports[0]["reused_target_frames"] == 3
    assert reports[0]["runs"][0]["target_start_frame"] == 1
    assert reports[0]["runs"][0]["target_end_frame"] == 1
    assert reports[0]["runs"][1]["target_start_frame"] == 3
    assert reports[0]["runs"][1]["target_end_frame"] == 4


def test_prepare_episodes_allows_mixed_positive_subtask_counts(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/repo", output_dir=tmp_path / "out")
    )
    episodes = [
        EpisodeData(
            episode_index=0,
            length=3,
            subtask_segments=[{"subtask_index": 0}, {"subtask_index": 1}],
        ),
        EpisodeData(
            episode_index=1,
            length=3,
            subtask_segments=[
                {"subtask_index": 0},
                {"subtask_index": 1},
                {"subtask_index": 2},
            ],
        ),
    ]

    prepared = converter.prepare_episodes_for_writing(episodes)

    assert prepared == episodes


def test_prepare_episodes_rejects_mixed_single_and_subtask(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/repo", output_dir=tmp_path / "out")
    )
    episodes = [
        EpisodeData(episode_index=0, length=3, subtask_segments=[]),
        EpisodeData(
            episode_index=1,
            length=3,
            subtask_segments=[{"subtask_index": 0}, {"subtask_index": 1}],
        ),
    ]

    assert converter.prepare_episodes_for_writing(episodes) == []
