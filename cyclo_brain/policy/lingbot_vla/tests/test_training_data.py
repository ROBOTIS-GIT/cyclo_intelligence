import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from test_training_inputs import input_profile
from lingbot_vla_training.data import PreparedEpisodes, TrainingDataset


@pytest.fixture
def prepared(tmp_path, input_profile):
    profile, info = input_profile
    info.update(codebase_version="v2.1", fps=10, total_episodes=2, chunks_size=100,
                splits={"train": "0:1"}, data_path="data/{episode_index}.parquet",
                video_path="videos/{video_key}/{episode_index}.mp4")
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    (tmp_path / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test"}) + "\n")
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "length": 2}) for i in range(2)))
    for i in range(2):
        pq.write_table(pa.table({"observation.state": [[i * 100, 2], [i * 100 + 2, 4]],
                                 "action": [[i * 100 + 4, 6], [i * 100 + 6, 8]],
                                 "episode_index": [i, i], "task_index": [0, 0], "timestamp": [0.0, 0.1]}),
                       tmp_path / f"data/{i}.parquet")
        for key in profile["cameras"].values():
            path = tmp_path / f"videos/{key}/{i}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return tmp_path, profile


def test_train_only_population_statistics_and_episode_location(prepared):
    root, profile = prepared
    data = PreparedEpisodes(root, profile)
    assert len(data) == 2
    stats = data.statistics()
    np.testing.assert_allclose(stats["state"]["mean"], [1, 3])
    np.testing.assert_allclose(stats["action"]["mean"], [5, 7])
    np.testing.assert_allclose(stats["state"]["std"], [1, 1])
    assert data.locate(1)[1] == 1
    for index in (-1, 2):
        with pytest.raises(IndexError):
            data.locate(index)


def test_batch_decoder_receives_python_timestamps_and_boundary_mask(prepared, monkeypatch):
    root, profile = prepared
    episodes = PreparedEpisodes(root, profile)
    decoded = []

    def decode(path, timestamps, tolerance, backend):
        assert all(type(value) is float for value in timestamps)
        assert backend == "pyav"
        decoded.append(timestamps)
        return torch.zeros(2, 3, 8, 8, dtype=torch.uint8)

    def images(processor, observation, **kwargs):
        pixels = torch.stack(list(observation["image"].values()))
        result = (pixels.float(), torch.ones(len(pixels), dtype=torch.bool), pixels,
                  torch.ones(len(pixels), 3, dtype=torch.long))
        return (*result, {}) if kwargs.get("return_augment_params") else result

    monkeypatch.setitem(sys.modules, "lingbotvla.data.vla_data.video_utils", SimpleNamespace(decode_video_frames=decode))
    monkeypatch.setitem(sys.modules, "lingbotvla.data.vla_data.transform", SimpleNamespace(
        prepare_images=images, prepare_language=lambda *a: (torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.bool))))
    dataset = TrainingDataset(episodes, episodes.statistics(), SimpleNamespace(image_processor=None, tokenizer=None),
                              None, chunk_size=3, image_size=8, warmup=True)
    batch = dataset[1]
    assert len(decoded) == len(profile["cameras"])
    assert not batch["future_valid"]
    assert batch["action_is_pad"].tolist() == [False, True, True]
    assert not batch["joint_mask"][1:].any()
    assert batch["pil_images"].dtype == torch.uint8


@pytest.mark.parametrize("change", ["v3", "split", "fps", "escape", "missing_video", "bad_episode", "bad_timing", "nan", "unknown_task"])
def test_malformed_data_rejected(prepared, tmp_path, change):
    root, profile = prepared
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if change == "v3":
        info["codebase_version"] = "v3.0"
    elif change == "split":
        info["splits"]["train"] = "1:0"
    elif change == "fps":
        info["fps"] = 0
    elif change == "escape":
        info["data_path"] = "../../outside.parquet"
    elif change == "missing_video":
        next((root / "videos").rglob("*.mp4")).unlink()
    else:
        path = root / "data/0.parquet"
        data = pq.read_table(path).to_pydict()
        if change == "bad_episode":
            data["episode_index"][1] = 1
        elif change == "bad_timing":
            data["timestamp"][1] = 0.9
        elif change == "unknown_task":
            data["task_index"][1] = 99
        else:
            data["action"][1][0] = float("nan")
        pq.write_table(pa.table(data), path)
    info_path.write_text(json.dumps(info))
    with pytest.raises((ValueError, FileNotFoundError)):
        PreparedEpisodes(root, profile)
