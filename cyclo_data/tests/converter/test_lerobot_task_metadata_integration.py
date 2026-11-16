"""Run in a LeRobot environment to exercise the real dataset reader."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader


class _UnusedReader:
    def __init__(self, *args, **kwargs):
        pass


# Parsed episodes do not require ROS/bag decoding for this integration test.
for module_name, class_name in (
    ("cyclo_data.reader.bag_reader", "BagReader"),
    ("cyclo_data.reader.metadata_manager", "MetadataManager"),
    ("cyclo_data.reader.video_metadata_extractor", "VideoMetadataExtractor"),
):
    module = types.ModuleType(module_name)
    setattr(module, class_name, _UnusedReader)
    sys.modules.setdefault(module_name, module)

from cyclo_data.converter.to_lerobot_v30 import (
    EpisodeData,
    RosbagToLerobotV30Converter,
    V30ConversionConfig,
)


@pytest.mark.parametrize("instructions", [
    ["Pick up the bottle."],
    ["Pick up the bottle.", "Place it in the basket.", "Pick up the bottle."],
    ["\ubcd1\uc744 \uc9d1\uc73c\uc138\uc694."],
])
def test_converted_v30_task_reaches_real_training_batch(tmp_path, instructions):
    root = tmp_path / "dataset"
    converter = RosbagToLerobotV30Converter(V30ConversionConfig(
        repo_id="test/task-metadata", output_dir=root, fps=30, use_videos=False,
    ))
    episodes = [EpisodeData(
        episode_index=i, length=2, timestamps=[0.0, 1.0 / 30],
        observation_state=[np.array([0.1, 0.2, 0.3], dtype=np.float32)] * 2,
        action=[np.array([0.3, 0.4], dtype=np.float32)] * 2,
        tasks=[instruction] * 2,
    ) for i, instruction in enumerate(instructions)]
    assert converter.write_from_episodes(episodes)
    dataset = LeRobotDataset("test/task-metadata", root=root)
    expected = [instruction for instruction in instructions for _ in range(2)]
    assert [dataset[i]["task"] for i in range(len(dataset))] == expected
    batch = next(iter(DataLoader(dataset, batch_size=len(expected))))
    assert batch["task"] == expected
    assert all(isinstance(task, str) for task in batch["task"])
