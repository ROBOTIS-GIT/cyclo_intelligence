import importlib.util
from pathlib import Path

import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "rldx_prepare", Path(__file__).resolve().parents[1] / "scripts/prepare_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_task_column_and_official_task_index_layout():
    assert module.task_map(pd.DataFrame({"task_index": [0], "task": ["pick"]})) == {0: "pick"}
    assert module.task_map(pd.DataFrame({"task_index": [0]}, index=["pick"])) == {0: "pick"}


def test_numeric_index_is_never_used_as_instruction():
    with pytest.raises(ValueError, match="textual task"):
        module.task_map(pd.DataFrame({"task_index": [0]}))


def test_channel_order_is_not_guessed():
    with pytest.raises(ValueError, match="ordered, unique"):
        module.feature_names({"features": {"action": {"shape": [2], "names": ["a", "a"]}}}, "action")


def test_v21_preparation_preserves_source_and_channel_order(tmp_path):
    import json
    import subprocess

    source = tmp_path / "source"
    destination = tmp_path / "run" / "prepared"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=red:s=32x32:r=10", "-frames:v", "2", "-c:v", "libx264",
        "-y", str(video),
    ], check=True)
    info = {
        "codebase_version": "v2.1", "robot_type": "fixture", "fps": 10,
        "chunks_size": 1000, "total_episodes": 1, "total_frames": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"shape": [2], "names": ["elbow", "shoulder"]},
            "action": {"shape": [2], "names": ["shoulder", "elbow"]},
            "observation.images.front": {"dtype": "video", "shape": [3, 32, 32],
                                         "info": {"video.height": 32, "video.width": 32}},
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick"}) + "\n")
    (source / "meta/episodes.jsonl").write_text(json.dumps({
        "episode_index": 0, "length": 2, "tasks": ["pick"],
    }) + "\n")
    pd.DataFrame({
        "episode_index": [0, 0], "index": [550, 551], "task_index": [0, 0],
        "timestamp": [0.0, 0.1],
        "observation.state": [[1.0, 2.0], [3.0, 4.0]],
        "action": [[5.0, 6.0], [7.0, 8.0]],
    }).to_parquet(source / "data/chunk-000/episode_000000.parquet", index=False)
    module.prepare(source, destination)
    metadata = json.loads((destination / "cyclo_input_metadata.json").read_text())
    assert metadata["state_names"] == ["elbow", "shoulder"]
    assert metadata["action_names"] == ["shoulder", "elbow"]
    assert json.loads((destination / "meta/stats.json").read_text())["action"]["mean"] == [6.0, 7.0]
    assert (destination / "data").is_symlink()
    assert (destination / "videos").is_symlink()
    assert not (source / "cyclo_input_metadata.json").exists()


def test_v21_rejects_missing_text_and_bad_video(tmp_path):
    import json

    source = tmp_path / "dataset"
    (source / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "fps": 10, "chunks_size": 1000,
        "total_episodes": 1, "total_frames": 1,
        "features": {
            "observation.state": {"shape": [1], "names": ["joint"]},
            "action": {"shape": [1], "names": ["joint"]},
            "observation.images.front": {"dtype": "video"},
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": ""}\n')
    (source / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 1, "tasks": [""]}\n')
    with pytest.raises(ValueError, match="instructions"):
        module.prepare(source, tmp_path / "output")


def test_v3_conversion_preserves_names_text_and_video_boundaries(tmp_path):
    import json
    import subprocess

    source = tmp_path / "source-v3"
    destination = tmp_path / "prepared-v21"
    (source / "meta/episodes").mkdir(parents=True)
    (source / "data").mkdir()
    video = source / "videos/front.mp4"
    video.parent.mkdir()
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=32x32:r=10", "-frames:v", "2", "-c:v", "libx264",
        "-y", str(video),
    ], check=True)
    info = {
        "codebase_version": "v3.0", "robot_type": "fixture", "fps": 10,
        "total_episodes": 1, "total_frames": 2, "total_tasks": 1,
        "data_path": "data/chunk_{chunk_index:03d}_file_{file_index:03d}.parquet",
        "video_path": "videos/front.mp4",
        "features": {
            "observation.state": {"shape": [2], "names": ["elbow", "shoulder"], "dtype": "float32"},
            "action": {"shape": [2], "names": ["shoulder", "elbow"], "dtype": "float32"},
            "observation.images.front": {"dtype": "video", "shape": [3, 32, 32],
                                         "info": {"video.height": 32, "video.width": 32}},
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0], "task": ["pick plate"]}).to_parquet(
        source / "meta/tasks.parquet", index=False,
    )
    pd.DataFrame({
        "episode_index": [0], "length": [2], "data/chunk_index": [0],
        "data/file_index": [0],
        "videos/observation.images.front/from_timestamp": [0.0],
        "videos/observation.images.front/to_timestamp": [0.2],
        "videos/observation.images.front/chunk_index": [0],
        "videos/observation.images.front/file_index": [0],
    }).to_parquet(source / "meta/episodes/chunk_000.parquet", index=False)
    pd.DataFrame({
        "episode_index": [0, 0], "frame_index": [0, 1], "task_index": [0, 0],
        "timestamp": [0.0, 0.1],
        "observation.state": [[1.0, 2.0], [3.0, 4.0]],
        "action": [[5.0, 6.0], [7.0, 8.0]],
    }).to_parquet(source / "data/chunk_000_file_000.parquet", index=False)
    module.prepare(source, destination)
    metadata = json.loads((destination / "cyclo_input_metadata.json").read_text())
    assert metadata["state_names"] == ["elbow", "shoulder"]
    assert metadata["action_names"] == ["shoulder", "elbow"]
    assert metadata["cameras"] == {"observation.images.front": "observation.images.front"}
    assert json.loads((destination / "meta/stats.json").read_text())["action"]["mean"] == [6.0, 7.0]
    assert not (destination / "INCOMPLETE").exists()
    assert (destination / "videos/chunk-000/observation.images.front/episode_000000.mp4").is_file()
