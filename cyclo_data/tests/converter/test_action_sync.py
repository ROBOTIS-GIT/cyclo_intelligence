import sys
import types

import numpy as np
import pytest


class _StubDependency:
    def __init__(self, *args, **kwargs):
        pass


bag_reader_module = types.ModuleType("cyclo_data.reader.bag_reader")
bag_reader_module.BagReader = _StubDependency
sys.modules.setdefault("cyclo_data.reader.bag_reader", bag_reader_module)

metadata_manager_module = types.ModuleType("cyclo_data.reader.metadata_manager")
metadata_manager_module.MetadataManager = _StubDependency
sys.modules.setdefault("cyclo_data.reader.metadata_manager", metadata_manager_module)

video_metadata_module = types.ModuleType(
    "cyclo_data.reader.video_metadata_extractor"
)
video_metadata_module.VideoMetadataExtractor = _StubDependency
sys.modules.setdefault(
    "cyclo_data.reader.video_metadata_extractor",
    video_metadata_module,
)

from cyclo_data.converter.base_converter import (  # noqa: E402
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
)


def _converter(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path, fps=15)
    )
    topics = [f"/action/{index}" for index in range(5)]
    group_keys = [
        "leader_arm_left",
        "leader_arm_right",
        "leader_head",
        "leader_lift",
        "leader_mobile",
    ]
    converter._action_topic_key_map = dict(zip(topics, group_keys))
    converter._joint_order_by_group = {
        group_key: [f"joint_{index}"]
        for index, group_key in enumerate(group_keys)
    }
    return converter, topics


def test_action_topics_are_committed_atomically_and_mobile_idle_is_preserved(
    tmp_path,
):
    converter, topics = _converter(tmp_path)
    period = 1.0 / 15.0
    publish_offsets = [0.0, 0.000042, 0.000058, 0.000071, 0.000089]
    action_messages = {
        topic: [
            (publish_offsets[index], np.array([float(index)], dtype=np.float32)),
            (
                period + publish_offsets[index],
                np.array([float(10 + index)], dtype=np.float32),
            ),
        ]
        for index, topic in enumerate(topics)
    }
    action_names = {
        topic: [f"joint_{index}"] for index, topic in enumerate(topics)
    }
    action_messages[topics[-1]].append(
        (
            2 * period + publish_offsets[-1],
            np.array([0.0], dtype=np.float32),
        )
    )

    merged = converter._merge_action_messages(
        action_messages,
        action_names,
    )

    assert len(merged) == 3
    assert merged[0][0] == pytest.approx(publish_offsets[-1])
    assert merged[1][0] == pytest.approx(period + publish_offsets[-1])
    np.testing.assert_array_equal(
        merged[0][1], np.arange(5, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        merged[1][1], np.arange(10, 15, dtype=np.float32)
    )
    assert merged[2][0] == pytest.approx(2 * period + publish_offsets[-1])
    np.testing.assert_array_equal(
        merged[2][1], np.array([10, 11, 12, 13, 0], dtype=np.float32)
    )

    during_second_publish, _ = converter._find_previous_value(
        merged,
        period + publish_offsets[2],
        expected_interval_sec=2 * period,
    )
    after_second_publish, _ = converter._find_previous_value(
        merged,
        period + 0.001,
        expected_interval_sec=2 * period,
    )
    after_mobile_idle, _ = converter._find_previous_value(
        merged,
        2 * period + 0.001,
        expected_interval_sec=2 * period,
    )
    np.testing.assert_array_equal(
        during_second_publish, np.arange(5, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        after_second_publish, np.arange(10, 15, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        after_mobile_idle,
        np.array([10, 11, 12, 13, 0], dtype=np.float32),
    )


def test_incomplete_action_publish_is_not_mixed_with_previous_tick(tmp_path):
    converter, topics = _converter(tmp_path)
    action_messages = {
        topic: [(0.001 * index, np.array([float(index)], dtype=np.float32))]
        for index, topic in enumerate(topics)
    }
    action_messages[topics[0]].append(
        (1.0, np.array([10.0], dtype=np.float32))
    )
    action_messages[topics[-1]].append(
        (1.001, np.array([14.0], dtype=np.float32))
    )
    action_names = {
        topic: [f"joint_{index}"] for index, topic in enumerate(topics)
    }

    with pytest.raises(ValueError, match="incomplete action publish"):
        converter._merge_action_messages(action_messages, action_names)


def test_equal_cardinality_tightly_paired_actions_use_publish_ordinal(tmp_path):
    converter, topics = _converter(tmp_path)
    # The second left publish is recorded before the first right publish.
    # These are still two paired controller ticks, not an incomplete action.
    converter.config.action_topics = topics[:2]
    action_messages = {
        topics[0]: [
            (0.010, np.array([1.0], dtype=np.float32)),
            (0.011, np.array([2.0], dtype=np.float32)),
        ],
        topics[1]: [
            (0.0115, np.array([3.0], dtype=np.float32)),
            (0.012, np.array([4.0], dtype=np.float32)),
        ],
    }
    action_names = {
        topics[0]: ["joint_0"],
        topics[1]: ["joint_1"],
    }

    merged = converter._merge_action_messages(action_messages, action_names)

    assert [timestamp for timestamp, _ in merged] == [0.0115, 0.012]
    np.testing.assert_array_equal(merged[0][1], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(merged[1][1], np.array([2.0, 4.0]))


def test_two_arm_topics_drop_one_trailing_unpaired_sample(tmp_path):
    converter, topics = _converter(tmp_path)
    converter.config.action_topics = topics[:2]
    action_messages = {
        topics[0]: [
            (0.010, np.array([1.0], dtype=np.float32)),
            (0.020, np.array([2.0], dtype=np.float32)),
        ],
        topics[1]: [(0.011, np.array([3.0], dtype=np.float32))],
    }
    action_names = {
        topics[0]: ["joint_0"],
        topics[1]: ["joint_1"],
    }

    merged = converter._merge_action_messages(action_messages, action_names)

    assert len(merged) == 1
    assert merged[0][0] == 0.011
    np.testing.assert_array_equal(merged[0][1], np.array([1.0, 3.0]))


def test_configured_action_topics_never_fall_back_to_zero_actions(tmp_path):
    converter, topics = _converter(tmp_path)
    converter.config.action_topics = topics
    episode = EpisodeData(episode_index=0)
    state_messages = [(0.0, np.ones(5, dtype=np.float32))]

    with pytest.raises(ValueError, match="configured action topics produced no data"):
        converter._resample_to_fps(
            episode,
            state_messages,
            action_messages=[],
            start_time=0.0,
        )


def test_intervention_signal_maps_to_causal_action_sources():
    labels = RosbagToLerobotConverterBase._action_sources_from_intervention_signal(
        [0.90, 1.00, 1.20, 1.40],
        [(1.00, False), (1.30, True)],
    )

    assert labels == ["vla", "vla", "vla", "human"]


def test_intervention_signal_rejects_unbounded_leading_gap():
    labels = RosbagToLerobotConverterBase._action_sources_from_intervention_signal(
        [0.50, 1.00],
        [(1.00, False)],
    )

    assert labels == []


def test_stationary_edge_bounds_keep_motion_context_and_internal_timeline():
    actions = [
        np.array([value], dtype=np.float32)
        for value in [0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3]
    ]

    start, stop = RosbagToLerobotConverterBase._stationary_edge_keep_bounds(
        actions,
        fps=2,
        delta_threshold=5e-4,
        context_seconds=0.5,
    )

    # Keep one context frame before/after the first-to-last motion interval.
    # The stationary pause between 2 and 3 remains part of the episode.
    assert (start, stop) == (3, 10)


def test_stationary_edge_trim_filters_in_memory_and_rebases_timestamps(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(
            repo_id="test",
            output_dir=tmp_path / "output",
            fps=2,
            trim_stationary_edges=True,
            stationary_context_seconds=0.5,
        )
    )
    source = tmp_path / "episode.mcap"
    source.write_bytes(b"source-must-remain-unchanged")
    episode = EpisodeData(
        episode_index=4,
        timestamps=[index * 0.5 for index in range(7)],
        grid_log_times_sec=[100.0 + index * 0.5 for index in range(7)],
        observation_state=[
            np.array([float(index)], dtype=np.float32) for index in range(7)
        ],
        action=[
            np.array([value], dtype=np.float32)
            for value in [0, 0, 1, 2, 2, 2, 2]
        ],
        length=7,
        source_path=source,
    )

    converter._trim_stationary_episode_edges(episode)

    assert episode.length == 4
    assert episode.timestamps == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert episode.grid_log_times_sec == pytest.approx(
        [100.5, 101.0, 101.5, 102.0]
    )
    assert [float(value[0]) for value in episode.action] == [0.0, 1.0, 2.0, 2.0]
    assert episode.stationary_trim_report == {
        "enabled": True,
        "original_frames": 7,
        "kept_frames": 4,
        "removed_prefix_frames": 1,
        "removed_suffix_frames": 2,
        "action_delta_threshold": 5e-4,
        "context_seconds": 0.5,
    }
    assert source.read_bytes() == b"source-must-remain-unchanged"


def test_fully_stationary_episode_is_kept_instead_of_becoming_empty(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(
            repo_id="test",
            output_dir=tmp_path,
            fps=15,
            trim_stationary_edges=True,
        )
    )
    episode = EpisodeData(
        episode_index=0,
        timestamps=[0.0, 1 / 15, 2 / 15],
        grid_log_times_sec=[10.0, 10.0 + 1 / 15, 10.0 + 2 / 15],
        observation_state=[np.zeros(1, dtype=np.float32) for _ in range(3)],
        action=[np.zeros(1, dtype=np.float32) for _ in range(3)],
        length=3,
    )

    converter._trim_stationary_episode_edges(episode)

    assert episode.length == 3
    assert episode.stationary_trim_report["kept_frames"] == 3


def test_archived_segments_trim_only_global_episode_edges(tmp_path, monkeypatch):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(
            repo_id="test",
            output_dir=tmp_path / "output",
            fps=2,
            use_videos=False,
            trim_stationary_edges=True,
            stationary_context_seconds=0.5,
        )
    )
    episode_dir = tmp_path / "0"
    episode_dir.mkdir()
    for segment_index in range(2):
        (episode_dir / f"0_{segment_index}.mcap").touch()

    values_by_segment = {
        "0_0": [0, 0, 1, 1],
        "0_1": [1, 1, 2, 2, 2, 2, 2],
    }
    extraction_windows = []

    def extract_segment(path, episode_index, trim_points, exclude_regions):
        extraction_windows.append((trim_points, exclude_regions))
        values = values_by_segment[path.stem]
        return EpisodeData(
            episode_index=episode_index,
            timestamps=[index * 0.5 for index in range(len(values))],
            grid_log_times_sec=[
                100.0 + index * 0.5 for index in range(len(values))
            ],
            observation_state=[
                np.array([float(index)], dtype=np.float32)
                for index in range(len(values))
            ],
            action=[
                np.array([float(value)], dtype=np.float32) for value in values
            ],
            action_sources=["human"] * len(values),
            length=len(values),
        )

    monkeypatch.setattr(converter, "_extract_joint_data", extract_segment)
    monkeypatch.setattr(
        converter,
        "_populate_action_sources",
        lambda path, episode: None,
    )
    episode = converter._convert_archived_segment_episode(
        episode_dir,
        0,
        {
            "task_name": "test",
            "segments": [
                {"sub_task_instruction": "first"},
                {"sub_task_instruction": "second"},
            ],
        },
        trim_points={"start": {"time": 123.0}},
        exclude_regions=[{"start": {"time": 124.0}, "end": {"time": 125.0}}],
    )

    assert episode is not None
    assert episode.length == 7
    assert [float(value[0]) for value in episode.action] == [
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
    ]
    assert episode.subtask_indices == [0, 0, 0, 1, 1, 1, 1]
    assert episode.action_sources == ["human"] * 7
    assert extraction_windows == [
        (
            {"start": {"time": 123.0}},
            [{"start": {"time": 124.0}, "end": {"time": 125.0}}],
        ),
        (
            {"start": {"time": 123.0}},
            [{"start": {"time": 124.0}, "end": {"time": 125.0}}],
        ),
    ]
    assert episode.stationary_trim_report == {
        "enabled": True,
        "original_frames": 11,
        "kept_frames": 7,
        "removed_prefix_frames": 1,
        "removed_suffix_frames": 3,
        "action_delta_threshold": 5e-4,
        "context_seconds": 0.5,
    }


def test_missing_configured_action_topics_are_omitted_when_others_exist(tmp_path):
    converter, topics = _converter(tmp_path)
    converter.config.action_topics = topics
    action_messages = {
        topics[0]: [
            (0.010, np.array([1.0], dtype=np.float32)),
            (0.020, np.array([2.0], dtype=np.float32)),
        ],
        topics[1]: [
            (0.011, np.array([3.0], dtype=np.float32)),
            (0.021, np.array([4.0], dtype=np.float32)),
        ],
    }
    action_names = {
        topics[0]: ["joint_0"],
        topics[1]: ["joint_1"],
    }

    merged = converter._merge_action_messages(action_messages, action_names)

    assert converter._action_joint_names == ["joint_0", "joint_1"]
    assert [timestamp for timestamp, _ in merged] == [0.011, 0.021]
    np.testing.assert_array_equal(merged[0][1], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(merged[1][1], np.array([2.0, 4.0]))
