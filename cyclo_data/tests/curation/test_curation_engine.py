import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from cyclo_data.curation.engine import (
    CurationEngine,
    _last_motion_frame,
    _sha256_file,
    _slice_bounds,
    _video_frame_count,
)


def test_motion_and_camera_bounds_are_frame_exact():
    values = np.asarray([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.001],
        [0.0, 0.001],
    ])
    assert _last_motion_frame(values, 5e-4) == 2

    stamps = np.asarray([100, 200, 300, 400, 500], dtype=np.int64)
    assert _slice_bounds(stamps, 201, 400) == (2, 4)


def test_source_and_output_trees_cannot_overlap(tmp_path):
    rosbag_root = tmp_path / "rosbag2"
    source = rosbag_root / "Task_fixture_MCAP"
    source.mkdir(parents=True)
    engine = CurationEngine(rosbag_root=rosbag_root, work_root=tmp_path / "jobs")

    with pytest.raises(ValueError, match="separate sibling trees"):
        engine._source_specs({
            "schema_version": 1,
            "sources": [{
                "task_dir": str(source),
                "output_task_dir": str(source / "curated"),
                "episodes": [0],
            }],
        })


def _create_video(path: Path, frames: int = 10, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=64x48:r={fps}",
            "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_apply_creates_source_preserving_frame_aligned_copy(tmp_path, monkeypatch):
    rosbag_root = tmp_path / "rosbag2"
    source = rosbag_root / "Task_fixture_MCAP"
    episode = source / "7"
    video_root = episode / "videos" / "7_0"
    video_root.mkdir(parents=True)
    (episode / "7_0.mcap").write_bytes(b"source-mcap-must-not-change")
    nested_mcap = episode / "split" / "7_1.mcap"
    nested_mcap.parent.mkdir()
    nested_mcap.write_bytes(b"second-source-mcap")
    (episode / "episode_info.json").write_text(json.dumps({
        "episode_index": 7,
        "episode_success": True,
        "robot_type": "f2",
    }))
    (episode / "metadata.yaml").write_text(yaml.safe_dump({
        "rosbag2_bagfile_information": {
            "starting_time": {"nanoseconds_since_epoch": 1_000_000_000},
        },
    }))
    (episode / "robot_config.yaml").write_text("robot_type: f2\n")
    video = video_root / "cam_head.mp4"
    sidecar = video_root / "cam_head_timestamps.parquet"
    _create_video(video)
    stamps = np.arange(10, dtype=np.int64) * 100_000_000 + 1_000_000_000
    pq.write_table(pa.table({
        "frame_index": pa.array(range(10), type=pa.int32()),
        "header_stamp_ns": pa.array(stamps, type=pa.int64()),
        "recv_ns": pa.array(stamps, type=pa.int64()),
    }), sidecar)
    original_mcap_hash = _sha256_file(episode / "7_0.mcap")
    original_nested_mcap_hash = _sha256_file(nested_mcap)

    output = rosbag_root / "Task_fixture_MCAP_intervention_motion_trimmed"
    analysis_row = {
        "source_task_dir": str(source),
        "source_episode_index": 7,
        "episode_success": True,
        "status": "needs_review",
        "fps": 10,
        "grid_origin_time_ns": int(stamps[0]),
        "start_frame": 2,
        "end_frame": 6,
    }
    analysis = {
        "schema_version": 1,
        "kind": "analysis",
        "job_id": "fixture-job",
        "settings": {"fps": 10},
        "sources": [{
            "task_dir": str(source),
            "output_task_dir": str(output),
            "episode_signatures": {},
            "episodes": [analysis_row],
        }],
    }
    request = {
        "sources": [{
            "task_dir": str(source),
            "output_task_dir": str(output),
        }],
        "overrides": [{
            "source_task_dir": str(source),
            "source_episode_index": 7,
            "include": True,
            "start_time_ns": int(stamps[2]),
            "end_time_ns": int(stamps[6]),
            "episode_success": False,
        }],
    }

    engine = CurationEngine(
        rosbag_root=rosbag_root,
        work_root=tmp_path / "jobs",
    )
    monkeypatch.setattr(
        engine,
        "_validate_reviewed_window",
        lambda *args, **kwargs: ([], {
            "status": "passed",
            "checks": {"all": True},
            "metrics": {},
        }),
    )
    result = engine.apply(analysis, request, job_id="fixture-job")

    curated_episode = output / "7"
    assert result["status"] == "passed"
    assert _sha256_file(curated_episode / "7_0.mcap") == original_mcap_hash
    assert _sha256_file(episode / "7_0.mcap") == original_mcap_hash
    assert _sha256_file(curated_episode / "split/7_1.mcap") == original_nested_mcap_hash
    assert _sha256_file(nested_mcap) == original_nested_mcap_hash
    assert _video_frame_count(curated_episode / "videos/7_0/cam_head.mp4") == 5
    table = pq.read_table(
        curated_episode / "videos/7_0/cam_head_timestamps.parquet"
    )
    assert table.num_rows == 5
    assert table["frame_index"].to_pylist() == [0, 1, 2, 3, 4]
    assert json.loads(
        (curated_episode / "episode_info.json").read_text()
    )["episode_success"] is False
    robot_config = yaml.safe_load(
        (curated_episode / "robot_config.yaml").read_text()
    )
    assert robot_config["trim_points"]["start"]["frame"] == 2
    assert robot_config["trim_points"]["end"]["frame"] == 6
    assert (output / "curation_manifest.json").is_file()
