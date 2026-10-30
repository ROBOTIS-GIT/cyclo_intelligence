"""Action streams must hold independently when a hand stops publishing."""

import pickle

import numpy as np
import pytest

from cyclo_data.converter import base_converter
from cyclo_data.converter.base_converter import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
)


def make_converter(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test/dropout", output_dir=tmp_path, fps=30)
    )
    converter._action_topic_key_map = {
        "/left": "leader_hand_left",
        "/right": "leader_hand_right",
    }
    return converter


def stream(stop=2.0, offset=0):
    return [
        (i / 100, np.array([offset + i], dtype=np.float32))
        for i in range(round(stop * 100) + 1)
    ]


def merge(converter, topics):
    return converter._merge_action_messages(
        topics, {"/left": ["left"], "/right": ["right"]}
    )


def assert_causal_values(converter, topics):
    actions = merge(converter, topics)
    episode, _ = converter._resample_to_fps(
        EpisodeData(episode_index=0), stream(), actions, 0.0
    )
    for t, actual in zip(episode.grid_log_times_sec, episode.action):
        expected = np.concatenate([
            [value for timestamp, value in topics[side] if timestamp <= t][-1]
            for side in ("/left", "/right")
        ])
        np.testing.assert_array_equal(actual, expected)
    assert episode.length > 0
    return episode


@pytest.mark.parametrize("stopped", ["/left", "/right"])
def test_stopped_hand_holds_while_live_hand_updates(tmp_path, stopped):
    converter = make_converter(tmp_path)
    topics = {"/left": stream(), "/right": stream(offset=1000)}
    topics[stopped] = topics[stopped][:51]
    episode = assert_causal_values(converter, topics)
    assert episode.length == 61
    np.testing.assert_array_equal(
        episode.action[-1], [50, 1200] if stopped == "/left" else [200, 1050]
    )


@pytest.mark.parametrize("stopped", ["/left", "/right"])
def test_reconnected_hand_resumes_without_freezing_other_hand(tmp_path, stopped):
    converter = make_converter(tmp_path)
    topics = {"/left": stream(), "/right": stream(offset=1000)}
    topics[stopped] = [(t, v) for t, v in topics[stopped] if not 0.5 < t < 1.5]
    episode = assert_causal_values(converter, topics)
    assert episode.length == 61
    np.testing.assert_array_equal(episode.action[-1], [200, 1200])


def test_asynchronous_topics_use_latest_value_without_looking_ahead(tmp_path):
    converter = make_converter(tmp_path)
    topics = {"/left": stream(), "/right": [(t + 0.003, v) for t, v in stream(offset=1000)]}
    assert_causal_values(converter, topics)


def test_no_backfill_before_each_topic_has_first_command(tmp_path):
    converter = make_converter(tmp_path)
    topics = {"/left": stream(), "/right": stream(offset=1000)[100:]}
    actions = merge(converter, topics)
    assert actions[0][0] == 1.0
    np.testing.assert_array_equal(actions[0][1], [100, 1100])


def test_empty_topic_does_not_invent_commands(tmp_path):
    converter = make_converter(tmp_path)
    assert merge(converter, {"/left": stream(), "/right": []}) == []


def test_both_hands_stop_and_hold_their_own_last_values(tmp_path):
    converter = make_converter(tmp_path)
    topics = {"/left": stream(stop=0.5), "/right": stream(stop=1.0, offset=1000)}
    episode = assert_causal_values(converter, topics)
    np.testing.assert_array_equal(episode.action[-1], [50, 1100])


def test_joint_order_is_preserved_during_dropout(tmp_path):
    converter = make_converter(tmp_path)
    converter._joint_order_by_group = {
        "leader_hand_left": ["left_b", "left_a"],
        "leader_hand_right": ["right"],
    }
    actions = converter._merge_action_messages(
        {"/left": [(0.0, np.array([10, 20]))], "/right": stream(offset=1000)},
        {"/left": ["left_a", "left_b"], "/right": ["right"]},
    )
    assert converter._action_joint_names == ["left_b", "left_a", "right"]
    np.testing.assert_array_equal(actions[-1][1], [20, 10, 1200])


def test_healthy_synchronous_streams_are_unchanged(tmp_path):
    converter = make_converter(tmp_path)
    episode = assert_causal_values(
        converter, {"/left": stream(), "/right": stream(offset=1000)}
    )
    assert episode.length == 61


def test_duplicate_timestamp_uses_last_command_without_duplicate_frames(tmp_path):
    converter = make_converter(tmp_path)
    topics = {
        "/left": [(0.0, np.array([1])), (0.0, np.array([2]))],
        "/right": [(0.0, np.array([10])), (1.0, np.array([11]))],
    }
    actions = merge(converter, topics)
    assert [timestamp for timestamp, _ in actions] == [0.0, 1.0]
    np.testing.assert_array_equal(actions[0][1], [2, 10])
    np.testing.assert_array_equal(actions[1][1], [2, 11])


def test_invalid_joint_mapping_does_not_emit_partial_action(tmp_path):
    converter = make_converter(tmp_path)
    converter._joint_order_by_group = {"leader_hand_left": ["missing_joint"]}
    assert merge(converter, {"/left": stream(), "/right": stream()}) == []


def test_resampling_starts_when_all_topics_first_have_commands(tmp_path):
    converter = make_converter(tmp_path)
    converter.config.fps = 10
    actions = merge(converter, {
        "/left": [(0.0, np.array([1])), (0.1, np.array([2])), (0.2, np.array([3]))],
        "/right": [(0.03, np.array([10])), (0.13, np.array([20]))],
    })
    state = [(0.0, np.array([100])), (0.1, np.array([200])), (0.2, np.array([300]))]
    episode, _ = converter._resample_to_fps(EpisodeData(0), state, actions, 0.0)
    assert episode.grid_log_times_sec == pytest.approx([0.03, 0.13])
    np.testing.assert_array_equal(episode.observation_state, [[100], [200]])
    np.testing.assert_array_equal(episode.action, [[1, 10], [2, 20]])


@pytest.mark.parametrize("seed", range(10))
def test_mixed_rate_groups_match_independent_causal_lookup(tmp_path, seed):
    converter = make_converter(tmp_path)
    rng = np.random.default_rng(seed)
    layouts = [("arm", 7), ("hand", 20), ("head", 2), ("mobile", 3)]
    converter._action_topic_key_map = {
        f"/{name}": f"leader_{name}" for name, _ in layouts
    }
    topics, names = {}, {}
    for name, width in reversed(layouts):
        times = np.unique(rng.integers(0, 2000, size=120)) / 1000
        if name == "hand":
            times = times[(times < 0.5) | (times > 1.5)]
        elif name == "mobile":
            times = times[times < 0.5]
        topics[f"/{name}"] = [
            (float(t), rng.normal(size=width).astype(np.float32)) for t in times
        ]
        names[f"/{name}"] = [f"{name}_{i}" for i in range(width)]

    actions = converter._merge_action_messages(topics, names)
    first_valid = max(messages[0][0] for messages in topics.values())
    expected_times = sorted({
        t for messages in topics.values() for t, _ in messages if t >= first_valid
    })
    assert [t for t, _ in actions] == expected_times
    for timestamp, actual in actions:
        expected = np.concatenate([
            [value for t, value in topics[f"/{name}"] if t <= timestamp][-1]
            for name, _ in layouts
        ])
        np.testing.assert_array_equal(actual, expected)
    assert converter._action_joint_names == [
        joint for name, _ in layouts for joint in names[f"/{name}"]
    ]


@pytest.mark.parametrize("kind", ["extract", "prepared"])
def test_old_cache_is_rejected_and_current_cache_is_readable(tmp_path, kind):
    converter = make_converter(tmp_path)
    cache = tmp_path / "episode.pkl"
    current_version = (
        base_converter._EXTRACT_CACHE_VERSION if kind == "extract"
        else base_converter._PREPARED_EPISODE_CACHE_VERSION
    )
    old_version = 2 if kind == "extract" else 3
    assert current_version > old_version
    for version, should_load in ((old_version, False), (current_version, True)):
        with cache.open("wb") as handle:
            pickle.dump({"version": version, "episode": EpisodeData(0),
                         "video_file_stats": {}}, handle)
        if kind == "extract":
            result = converter._load_episode_extract_cache(cache, episode_index=0)
        else:
            result = converter._load_prepared_episode_cache(
                cache, episode_index=0, bag_path=tmp_path
            )
        assert (result is not None) == should_load
