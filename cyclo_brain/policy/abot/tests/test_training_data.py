import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_training.dataset import PreparedDataset
from abot_training.profile import canonical_hash, load_profile, read_json, validate_profile


@pytest.fixture
def prepared(tmp_path):
    av = pytest.importorskip("av")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "robot_type": "test_robot", "total_episodes": 2,
        "total_frames": 7, "fps": 10, "chunks_size": 1, "splits": {"train": "0:2"},
        "data_path": "data/{episode_chunk}/episode_{episode_index}.parquet",
        "video_path": "videos/{episode_chunk}/{video_key}/{episode_index}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["s1", "s2"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["a1", "a2"]},
            "observation.images.head": {"dtype": "video"},
        },
    }
    profile = {
        "version": 2, "profile_id": "test-abot", "model_id": "abot-m0", "policy_id": "abot:m0",
        "robot_type": "test_robot", "state_names": ["s1", "s2"], "action_names": ["a1", "a2"],
        "cameras": {"head": "observation.images.head"}, "camera_order": ["head"],
        "assets": {"base": {"provider": "modelscope", "repo_id": "test/base", "revision": "a" * 40},
                   "vlm": {"provider": "huggingface", "repo_id": "test/vlm", "revision": "b" * 40}},
        "asset_roles": {"base": "base", "processor": "vlm"},
        "action_representation": "absolute", "normalization": "min_max",
        "padding": {"state_dim": 3, "action_dim": 4}, "padding_mask": "exclude_padded_channels",
        "modalities": {"state": {"channels": ["s1", "s2"]}, "action": {"channels": ["a1", "a2"]},
                       "video": {"head": "observation.images.head"}},
        "use_vggt": False, "head_initialization": "reinitialize_mismatched",
        "vocabulary_initialization": "restore_checkpoint",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes = [{"episode_index": 0, "length": 3}, {"episode_index": 1, "length": 4}]
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test instruction"}))
    for ep in episodes:
        i, n = ep["episode_index"], ep["length"]
        data = root / f"data/{i}/episode_{i}.parquet"
        data.parent.mkdir(parents=True)
        values = [[float(10 * i + t), 2.0] for t in range(n)]
        pq.write_table(pa.table({"observation.state": values, "action": values,
                                "timestamp": np.arange(n) / 10, "episode_index": [i] * n,
                                "task_index": [0] * n}), data)
        video = root / f"videos/{i}/observation.images.head/{i}.mp4"
        video.parent.mkdir(parents=True)
        with av.open(str(video), mode="w") as out:
            stream = out.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv420p"
            for t in range(n):
                rgb = np.full((16, 32, 3), 30 + i * 100 + t * 10, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
    return root, profile, info


def dataset(prepared, **kwargs):
    root, profile, _ = prepared
    return PreparedDataset(root, profile, horizon=2, image_size=[12, 8], **kwargs)


def test_real_batch_order_boundary_padding_and_reload_stats(prepared):
    data = dataset(prepared)
    assert len(data) == 5
    sample = data[1]
    assert sample["state"].shape == (1, 3)
    assert sample["action"].shape == (2, 4)
    np.testing.assert_array_equal(sample["action_mask"], [True, True, False, False])
    np.testing.assert_array_equal(sample["action"][:, 2:], 0)
    np.testing.assert_array_equal(sample["state"][:, 2:], 0)
    np.testing.assert_allclose(sample["action"][:, 0], np.array([1, 2]) / 13 * 2 - 1)
    assert data[2]["action"][0, 0] == pytest.approx(10 / 13 * 2 - 1)
    assert data[len(data) - 1]["action"][-1, 0] == pytest.approx(1)
    assert sample["image"][0].size == (12, 8)
    assert sample["image"][0].mode == "RGB"
    assert sample["lang"] == "test instruction"
    restored = dataset(prepared, statistics=json.loads(json.dumps(data.statistics)))
    np.testing.assert_array_equal(restored[1]["action"], sample["action"])


@pytest.mark.parametrize("mode", ["min_max", "mean_std", "quantile"])
def test_statistics_from_actual_vectors_not_averaged_episode_stats(prepared, mode):
    prepared[1]["normalization"] = mode
    data = dataset(prepared)
    expected = np.array([0, 1, 2, 10, 11, 12, 13], dtype=np.float64)
    stats = data.statistics["action"]
    assert stats["count"] == 7
    assert stats["mean"][0] == pytest.approx(expected.mean())
    assert stats["std"][0] == pytest.approx(expected.std())
    assert stats["std"][1] == 0
    if mode == "quantile":
        assert stats["q01"][0] == pytest.approx(np.quantile(expected, .01))


@pytest.mark.parametrize("field,value,match", [
    ("state_names", ["s2", "s1"], "names/order"),
    ("action_names", ["a1"], "names/order"),
    ("padding", {"state_dim": 1, "action_dim": 4}, "truncate"),
    ("padding_mask", "none", "excluded"),
    ("action_representation", "relative", "absolute"),
    ("use_vggt", True, "use_vggt"),
    ("camera_order", [], "camera_order"),
    ("head_initialization", "ignore_errors", "head_initialization"),
    ("vocabulary_initialization", "trim", "vocabulary_initialization"),
])
def test_profile_rejects_unsafe_changes(prepared, field, value, match):
    _, profile, info = prepared
    profile[field] = value
    with pytest.raises(ValueError, match=match):
        validate_profile(profile, info)


def test_profile_hash_and_version_and_traversal(prepared, tmp_path):
    _, profile, info = prepared
    path = tmp_path / "test-abot.json"
    path.write_text(json.dumps(profile))
    assert load_profile(tmp_path, "test-abot", canonical_hash(profile), info) == profile
    with pytest.raises(ValueError, match="hash mismatch"):
        load_profile(tmp_path, "test-abot", "0" * 64, info)
    with pytest.raises(ValueError, match="profile_id"):
        load_profile(tmp_path, "../test-abot", canonical_hash(profile), info)
    info["codebase_version"] = "v3.0"
    with pytest.raises(ValueError, match="conversion belongs to Bench"):
        validate_profile(profile, info)
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


@pytest.mark.parametrize("failure", ["timestamp", "state", "episode", "task", "missing_video", "escape"])
def test_invalid_dataset_fails_before_training(prepared, failure, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, _, info = prepared
    path = root / "data/0/episode_0.parquet"
    table = pq.read_table(path).to_pydict()
    if failure == "timestamp":
        table["timestamp"][1] = -1
    elif failure == "state":
        table["observation.state"][1][0] = float("nan")
    elif failure == "episode":
        table["episode_index"][1] = 100
    elif failure == "task":
        table["task_index"][1] = 20
    elif failure == "missing_video":
        (root / "videos/0/observation.images.head/0.mp4").unlink()
    elif failure == "escape":
        outside = tmp_path / "outside.parquet"
        path.rename(outside)
        path.symlink_to(outside)
    if failure not in ("escape", "missing_video"):
        pq.write_table(pa.table(table), path)
    with pytest.raises((ValueError, FileNotFoundError)):
        dataset(prepared)


def test_saved_statistics_do_not_skip_sample_validation(prepared):
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = dataset(prepared)
    path = prepared[0] / "data/0/episode_0.parquet"
    table = pq.read_table(path).to_pydict()
    table["timestamp"][1] = -1
    pq.write_table(pa.table(table), path)
    restored = dataset(prepared, statistics=data.statistics)
    with pytest.raises(ValueError, match="timestamps"):
        restored[0]


def test_task_000759_profile_keeps_approved_cameras_and_explicit_asset_providers():
    path = Path(__file__).resolve().parents[1] / "training_profiles/ffw-sg2-task-000759-3cam-v1.json"
    profile = read_json(path)
    assert profile["camera_order"] == ["head", "left_wrist", "right_wrist"]
    assert list(profile["cameras"].values()) == [
        "observation.images.rgb.cam_left_head", "observation.images.rgb.cam_left_wrist",
        "observation.images.rgb.cam_right_wrist",
    ]
    assert len(profile["state_names"]) == len(profile["action_names"]) == 22
    info = {"codebase_version": "v2.1", "robot_type": profile["robot_type"], "features": {
        "observation.state": {"names": profile["state_names"], "shape": [22]},
        "action": {"names": profile["action_names"], "shape": [22]},
        **{key: {"dtype": "video"} for key in profile["cameras"].values()},
    }}
    assert validate_profile(profile, info) == profile
    assert profile["assets"][profile["asset_roles"]["base"]]["provider"] == "modelscope"
    assert profile["assets"][profile["asset_roles"]["processor"]]["provider"] == "huggingface"
    del profile["assets"]["abot_pretrained"]["provider"]
    with pytest.raises(ValueError, match="provider"):
        validate_profile(profile, info)
