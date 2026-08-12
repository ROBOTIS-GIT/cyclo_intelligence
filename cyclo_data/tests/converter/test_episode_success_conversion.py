import json
import sys
import types

import numpy as np
import pyarrow.parquet as pq
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

video_metadata_module = types.ModuleType("cyclo_data.reader.video_metadata_extractor")
video_metadata_module.VideoMetadataExtractor = _StubDependency
sys.modules.setdefault(
    "cyclo_data.reader.video_metadata_extractor",
    video_metadata_module,
)

from cyclo_data.converter.base_converter import (  # noqa: E402
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
    _PREPARED_EPISODE_CACHE_VERSION,
)
from cyclo_data.converter.to_lerobot_v21 import (  # noqa: E402
    RosbagToLerobotConverter,
    _V21_EPISODE_PARQUET_CACHE_VERSION,
)
from cyclo_data.converter.to_lerobot_v30 import (  # noqa: E402
    RosbagToLerobotV30Converter,
    V30ConversionConfig,
    _V30_DATA_AGGREGATE_CACHE_VERSION,
)


def _episode(index, success, length=2):
    return EpisodeData(
        episode_index=index,
        timestamps=[frame / 10.0 for frame in range(length)],
        observation_state=[np.array([index, frame], dtype=np.float32)
                           for frame in range(length)],
        action=[np.array([frame], dtype=np.float32)
                for frame in range(length)],
        tasks=["pick"],
        length=length,
        episode_success=success,
    )


def test_episode_info_reads_only_boolean_success_labels(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    episode = _episode(0, None)

    converter._apply_episode_info(episode, {"episode_success": True})
    assert episode.episode_success is True

    with pytest.raises(ValueError, match="must be a boolean"):
        converter._apply_episode_info(episode, {"episode_success": 1})


def test_mixed_success_labels_are_rejected(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )

    with pytest.raises(ValueError, match="Mixed episode_success"):
        converter._build_features([_episode(0, True), _episode(1, None)])


def test_v21_repeats_episode_success_on_every_row(tmp_path):
    converter = RosbagToLerobotConverter(
        ConversionConfig(repo_id="test", output_dir=tmp_path, use_videos=False)
    )
    episodes = [_episode(0, True), _episode(1, False)]
    converter._build_features(episodes)
    converter._task_to_index = {"pick": 0}
    converter._total_frames = 0

    success_path = tmp_path / "success.parquet"
    failure_path = tmp_path / "failure.parquet"
    converter._write_parquet(episodes[0], success_path)
    converter._write_parquet(episodes[1], failure_path)

    assert pq.read_table(success_path)["episode_success"].to_pylist() == [True, True]
    assert pq.read_table(failure_path)["episode_success"].to_pylist() == [False, False]
    assert converter._v21_features_for_info()["episode_success"]["dtype"] == "bool"


def test_unlabeled_v21_keeps_legacy_schema(tmp_path):
    converter = RosbagToLerobotConverter(
        ConversionConfig(repo_id="test", output_dir=tmp_path, use_videos=False)
    )
    episode = _episode(0, None)
    converter._build_features([episode])
    converter._task_to_index = {"pick": 0}
    converter._total_frames = 0
    path = tmp_path / "unlabeled.parquet"

    converter._write_parquet(episode, path)

    assert "episode_success" not in pq.read_schema(path).names
    assert "episode_success" not in converter._v21_features_for_info()


def test_direct_v30_writes_success_feature_and_stats(tmp_path):
    converter = RosbagToLerobotV30Converter(
        V30ConversionConfig(
            repo_id="test",
            output_dir=tmp_path,
            use_videos=False,
        )
    )

    assert converter.write_from_episodes([
        _episode(0, True),
        _episode(1, False),
    ]) is True

    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path)
    assert table["episode_success"].to_pylist() == [True, True, False, False]

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["episode_success"]["dtype"] == "bool"
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert "episode_success" in stats
    assert stats["episode_success"]["mean"] == [0.5]
    episodes_path = (
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    episode_stats = pq.read_table(episodes_path)
    assert episode_stats["stats/episode_success/mean"].to_pylist() == [
        [1.0],
        [0.0],
    ]
    root_info = json.loads((tmp_path / "info.json").read_text())
    assert "episode_success" not in root_info


def test_success_label_invalidates_all_parquet_caches(tmp_path):
    v21 = RosbagToLerobotConverter(
        ConversionConfig(repo_id="test", output_dir=tmp_path / "v21")
    )
    v30 = RosbagToLerobotV30Converter(
        V30ConversionConfig(repo_id="test", output_dir=tmp_path / "v30")
    )
    success = _episode(0, True)
    failure = _episode(0, False)

    assert v21._v21_episode_data_cache_signature(success) != (
        v21._v21_episode_data_cache_signature(failure)
    )
    assert v30._episode_data_cache_signature(success) != (
        v30._episode_data_cache_signature(failure)
    )
    assert _PREPARED_EPISODE_CACHE_VERSION >= 4
    assert _V21_EPISODE_PARQUET_CACHE_VERSION >= 2
    assert _V30_DATA_AGGREGATE_CACHE_VERSION >= 2
