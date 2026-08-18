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


def test_missing_configured_action_topic_fails_before_dimension_can_shrink(
    tmp_path,
):
    converter, topics = _converter(tmp_path)
    converter.config.action_topics = topics
    action_messages = {
        topic: [(0.0, np.array([float(index)], dtype=np.float32))]
        for index, topic in enumerate(topics[:-1])
    }
    action_names = {
        topic: [f"joint_{index}"]
        for index, topic in enumerate(topics[:-1])
    }

    with pytest.raises(ValueError, match="configured action topic.*no data"):
        converter._merge_action_messages(action_messages, action_names)
