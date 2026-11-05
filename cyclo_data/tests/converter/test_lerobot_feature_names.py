import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sys
import types


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

from cyclo_data.converter.base_converter import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
)
from cyclo_data.converter.to_lerobot_v30 import (
    RosbagToLerobotV30Converter,
    V30ConversionConfig,
)


def test_feature_names_allow_state_action_dimension_mismatch(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    arm_names = [f"joint_{i}" for i in range(57)]
    mobile_names = ["linear_x", "linear_y", "angular_z"]

    converter._joint_order_by_group = {
        "leader_upper_body": arm_names,
        "leader_mobile": mobile_names,
    }
    converter._action_joint_names = []
    converter._state_joint_names = []

    episode = EpisodeData(
        episode_index=0,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
        observation_state_names=arm_names + mobile_names,
        action_names=arm_names,
    )

    converter._build_features([episode])

    assert converter._features["observation.state"]["shape"] == (60,)
    assert converter._features["observation.state"]["names"] == arm_names + mobile_names
    assert converter._features["action"]["shape"] == (57,)
    assert converter._features["action"]["names"] == arm_names


def test_feature_names_ignore_mismatched_config_fallback(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    parsed_action_names = [f"arm_joint_{i}" for i in range(57)]
    converter._joint_order_by_group = {
        "leader_upper_body": parsed_action_names,
        "leader_mobile": ["linear_x", "linear_y", "angular_z"],
    }

    episode = EpisodeData(
        episode_index=0,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
        action_names=parsed_action_names,
    )

    converter._build_features([episode])

    assert converter._features["action"]["names"] == parsed_action_names
    assert converter._features["action"]["names"] != [f"joint_{i}" for i in range(57)]


def test_feature_names_can_use_later_episode_names(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    action_names = [f"arm_joint_{i}" for i in range(57)]

    episodes = [
        EpisodeData(
            episode_index=0,
            observation_state=[np.zeros(60, dtype=np.float32)],
            action=[np.zeros(57, dtype=np.float32)],
        ),
        EpisodeData(
            episode_index=1,
            observation_state=[np.zeros(60, dtype=np.float32)],
            action=[np.zeros(57, dtype=np.float32)],
            action_names=action_names,
        ),
    ]

    converter._build_features(episodes)

    assert converter._features["action"]["names"] == action_names


def test_feature_names_ignore_none_name_sources(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    action_names = [f"arm_joint_{i}" for i in range(57)]
    converter._state_joint_names = None
    converter._action_joint_names = None

    first = EpisodeData(
        episode_index=0,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
    )
    first.observation_state_names = None
    first.action_names = None
    second = EpisodeData(
        episode_index=1,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
        action_names=action_names,
    )

    converter._build_features([first, second])

    assert converter._features["action"]["names"] == action_names


@pytest.mark.parametrize("instructions", [
    [],
    ["Pick up the object."],
    ["Place the object.", "Pick up the object.", "물체를 잡으세요."],
])
def test_v30_tasks_restore_instruction_as_pandas_index(tmp_path, instructions):
    converter = RosbagToLerobotV30Converter(
        V30ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    converter._tasks = dict(enumerate(instructions))
    converter._task_names_by_task = {
        task: f"Task_{i:06d}" for i, task in enumerate(instructions)
    }

    converter._write_tasks_parquet()

    path = tmp_path / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(path)
    expected = instructions or ["default_task"]
    assert tasks.index.name == "task"
    assert tasks.index.tolist() == expected
    assert tasks["task_index"].tolist() == list(range(len(expected)))
    for i, instruction in enumerate(expected):
        # This is how LeRobotDataset.__getitem__ resolves the prompt.
        assert tasks.iloc[i].name == instruction
        assert tasks.iloc[i]["task_name"] == (
            f"Task_{i:06d}" if instructions else "default_task"
        )
    table = pq.read_table(path)
    assert table["task"].to_pylist() == expected
    assert table.schema.field("task").type == pa.string()
    assert table.schema.field("task_name").type == pa.string()


def test_v30_tasks_bypass_legacy_cache_and_reuse_corrected_cache(tmp_path, monkeypatch):
    converter = RosbagToLerobotV30Converter(
        V30ConversionConfig(repo_id="test", output_dir=tmp_path / "output")
    )
    source = tmp_path / "source"
    source.mkdir()
    episodes = [EpisodeData(episode_index=0, source_path=source)]
    converter._tasks = {0: "Pick up the object."}
    converter._task_names_by_task = {"Pick up the object.": "Task_000650"}
    converter._current_data_aggregate_cache_key = {"test": "task-index"}
    key = converter._tasks_parquet_cache_key()
    assert key["version"] == 2
    legacy_key = {**key, "version": 1}
    legacy = tmp_path / "legacy.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "task_index": 0, "task": "Pick up the object.", "task_name": "Task_000650",
    }]), legacy)
    converter._store_small_parquet_cache(
        episodes, legacy_key, legacy,
        cache_name="tasks_parquet_v30", artifact_name="tasks.parquet",
    )
    assert pd.read_parquet(legacy).iloc[0].name == 0

    converter._write_tasks_parquet(episodes)
    output = tmp_path / "output" / "meta" / "tasks.parquet"
    assert pd.read_parquet(output).iloc[0].name == "Pick up the object."
    cache = converter._small_parquet_cache_path(episodes, key, "tasks_parquet_v30")
    assert (cache / "tasks.parquet").exists()

    def unexpected_write(*args, **kwargs):
        raise AssertionError("The corrected tasks cache should be reused")

    monkeypatch.setattr(pq, "write_table", unexpected_write)
    converter._write_tasks_parquet(episodes)
    assert pd.read_parquet(output).iloc[0].name == "Pick up the object."
