from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import threading

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))
sys.path.insert(0, str(_REPO_ROOT / "orchestrator"))
import cyclo_data  # noqa: E402
import cyclo_data.converter  # noqa: E402
import cyclo_data.hub  # noqa: E402


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        sys.modules.setdefault(parent, ModuleType(parent))
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


_stub_module("huggingface_hub", HfApi=_Dummy)
_stub_module("interfaces.msg", RecordingStatus=_Dummy)
_stub_module("cyclo_data.converter.orchestrator", DataConverter=_Dummy)
_stub_module(
    "cyclo_data.hub.progress_tracker",
    HuggingFaceLogCapture=_Dummy,
    HuggingFaceProgressTqdm=_Dummy,
)
_stub_module("psutil", cpu_percent=lambda interval=None: 0.0)

from cyclo_data.recorder.session_manager import DataManager  # noqa: E402


class _VanishingPath:
    def is_file(self):
        return True

    def stat(self):
        raise FileNotFoundError("removed during scan")


def _make_manager(root: Path, *, subtask_total: int = 2) -> DataManager:
    manager = DataManager.__new__(DataManager)
    manager._save_rosbag_path = str(root)
    manager._segmented_storage_mode = True
    manager._physical_segment_total = subtask_total
    manager._subtask_mode = subtask_total > 1
    manager._main_task_instruction = "main instruction"
    manager._subtask_instructions = [
        f"subtask {idx}" for idx in range(subtask_total)
    ]
    manager._task_info = SimpleNamespace(task_num="1234", task_name="archive test")
    manager._robot_type = "test_robot"
    manager._state_lock = threading.Lock()
    manager._saved_subtasks_cache = {}
    return manager


def test_inference_save_repo_name_uses_timestamp_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "cyclo_data.recorder.session_manager.time.strftime",
        lambda fmt, tm: "20260622_031455",
    )
    task_info = SimpleNamespace(task_num="", task_name="", task_type="inference")

    repo_name = DataManager._make_save_repo_name(tmp_path, task_info)

    assert repo_name == "Task_20260622_031455_inference_MCAP"
    assert task_info.task_num == "20260622_031455"
    assert task_info.task_name == "inference"


def test_inference_save_repo_name_avoids_existing_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "cyclo_data.recorder.session_manager.time.strftime",
        lambda fmt, tm: "20260622_031455",
    )
    (tmp_path / "Task_20260622_031455_inference_MCAP").mkdir()
    task_info = SimpleNamespace(task_num="", task_name="", task_type="inference")

    repo_name = DataManager._make_save_repo_name(tmp_path, task_info)

    assert repo_name == "Task_20260622_031455_01_inference_MCAP"
    assert task_info.task_num == "20260622_031455_01"
    assert task_info.task_name == "inference"


def _write_segment(
    root: Path,
    *,
    full_idx: int,
    subtask_idx: int,
    subtask_total: int,
    with_video: bool = True,
    subtask_instruction: str | None = None,
) -> Path:
    segment = root / str(full_idx) / "segments" / str(subtask_idx)
    segment.mkdir(parents=True, exist_ok=True)
    (segment / f"segment_{subtask_idx}.mcap").write_bytes(
        f"mcap-{subtask_idx}".encode()
    )
    start_ns = 1_000_000_000 + subtask_idx * 100_000_000
    duration_ns = 50_000_000
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 9,
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": duration_ns},
            "starting_time": {"nanoseconds_since_epoch": start_ns},
            "message_count": 3,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/joint_states",
                        "type": "sensor_msgs/msg/JointState",
                        "serialization_format": "cdr",
                        "offered_qos_profiles": "",
                    },
                    "message_count": 3,
                }
            ],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [f"segment_{subtask_idx}.mcap"],
            "files": [
                {
                    "path": f"segment_{subtask_idx}.mcap",
                    "starting_time": {"nanoseconds_since_epoch": start_ns},
                    "duration": {"nanoseconds": duration_ns},
                    "message_count": 3,
                }
            ],
            "custom_data": None,
            "ros_distro": "jazzy",
        }
    }
    (segment / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    info = {
        "recording_mode": "subtask" if subtask_total > 1 else "single_segment",
        "full_episode_index": full_idx,
        "subtask_index": subtask_idx,
        "subtask_total": subtask_total,
        "episode_index": subtask_idx,
        "subtask_instruction": subtask_instruction or f"subtask {subtask_idx}",
    }
    if with_video:
        videos = segment / "videos"
        videos.mkdir()
        (videos / "cam0.mp4").write_bytes(b"raw-mjpeg")
        (videos / "cam0_timestamps.parquet").write_bytes(b"timestamps")
        info["video_stats"] = {"cam0": {"frames_written": 1}}
    (segment / "episode_info.json").write_text(json.dumps(info, indent=2))
    return segment


def test_file_size_if_present_ignores_concurrent_removal():
    assert DataManager._file_size_if_present(_VanishingPath()) == 0


def test_archive_moves_segmented_files_and_marks_pending(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    first = _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    second = _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)

    out = manager._archive_full_episode(0)

    assert out == root / "0"
    assert (out / "0_0.mcap").read_bytes() == b"mcap-0"
    assert (out / "0_1.mcap").read_bytes() == b"mcap-1"
    assert not (first / "segment_0.mcap").exists()
    assert not (second / "segment_1.mcap").exists()
    assert not (out / "segments").exists()
    assert (out / "videos" / "0_0" / "cam0.mp4").read_bytes() == b"raw-mjpeg"
    assert (out / "videos" / "0_0" / "cam0_timestamps.parquet").read_bytes() == (
        b"timestamps"
    )

    info = json.loads((out / "episode_info.json").read_text())
    assert info["transcoding_status"] == "pending"


def test_archive_marks_episode_without_videos_not_required(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=1)
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
    )

    out = manager._archive_full_episode(0)

    assert out == root / "0"
    assert not (out / "segments").exists()
    info = json.loads((out / "episode_info.json").read_text())
    assert info["transcoding_status"] == "not_required"


def test_archive_preserves_pending_raw_spool(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=1)
    segment = _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
    )
    videos = segment / "videos"
    videos.mkdir()
    (videos / "cam0.mjpeg.tmp").write_bytes(b"raw-spool")
    (videos / "cam0_timestamps.parquet").write_bytes(b"timestamps")
    (videos / "cam0_recorder_stats.json").write_text(
        json.dumps({"frames_written": 1, "remux_status": "pending"}),
        encoding="utf-8",
    )
    info_path = segment / "episode_info.json"
    info = json.loads(info_path.read_text())
    info["video_stats"] = {"cam0": {"frames_written": 1, "remux_status": "pending"}}
    info["transcoding_status"] = "pending"
    info["video_remux_status"] = "pending"
    info_path.write_text(json.dumps(info, indent=2))

    out = manager._archive_full_episode(0)

    assert not (out / "segments").exists()
    archived_video_dir = out / "videos" / "0_0"
    assert (archived_video_dir / "cam0.mjpeg.tmp").read_bytes() == b"raw-spool"
    assert (archived_video_dir / "cam0_timestamps.parquet").read_bytes() == (
        b"timestamps"
    )
    assert (archived_video_dir / "cam0_recorder_stats.json").exists()
    summary = json.loads((out / "episode_info.json").read_text())
    assert summary["transcoding_status"] == "pending"
    assert summary["video_remux_status"] == "pending"


def test_discard_current_full_episode_removes_all_saved_subtasks(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._current_full_episode_index = 0
    manager._current_subtask_index = 1
    manager._current_scenario_number = 1
    manager._record_episode_count = 2
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)

    deleted = manager.discard_current_full_episode()

    assert deleted == 2
    assert not (root / "0").exists()
    assert manager._current_subtask_index == 0
    assert manager._current_scenario_number == 0


def test_discard_full_episode_deletes_requested_episode_without_cursor_drift(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._current_full_episode_index = 1
    manager._current_subtask_index = 1
    manager._current_scenario_number = 1
    manager._record_episode_count = 3
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)
    _write_segment(root, full_idx=1, subtask_idx=0, subtask_total=2)

    deleted = manager.discard_full_episode(0)

    assert deleted == 2
    assert not (root / "0").exists()
    assert (root / "1").exists()
    assert manager._current_full_episode_index == 1
    assert manager._current_subtask_index == 1
    assert manager._current_scenario_number == 1


def test_discard_recording_can_reset_active_episode_subtask_cursor(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._segmented_storage_mode = True
    manager._status = "recording"
    manager._start_time_s = 123.0
    manager._record_episode_count = 2
    manager._current_subtask_index = 2
    manager._current_scenario_number = 2

    manager.discard_recording(reset_subtask_index=True)

    assert manager._status == "idle"
    assert manager._record_episode_count == 2
    assert manager._current_subtask_index == 0
    assert manager._current_scenario_number == 0


def test_missing_subtasks_reports_gap_in_saved_segments(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._current_full_episode_index = 0
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=3)
    _write_segment(root, full_idx=0, subtask_idx=2, subtask_total=3)

    assert manager.saved_subtask_indices_for_full_episode() == {0, 2}
    assert manager.missing_subtasks_for_full_episode() == [1]


def test_active_segment_directory_without_episode_info_is_not_saved(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._current_full_episode_index = 0
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=3)
    active_segment = root / "0" / "segments" / "1"
    active_segment.mkdir(parents=True)
    (active_segment / "segment_1.mcap").write_bytes(b"recording")

    assert manager.saved_subtask_indices_for_full_episode() == {0}
    assert manager.missing_subtasks_for_full_episode() == [1, 2]


def test_saved_subtask_indices_uses_cache_when_segments_dir_missing(monkeypatch, tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._current_full_episode_index = 4
    manager._saved_subtasks_cache = {4: {0, 2}}
    monkeypatch.setattr(
        manager,
        "_episode_dirs_for_full_subtask",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("status lookup must not scan the whole task root")
        ),
    )

    assert manager.saved_subtask_indices_for_full_episode() == {0, 2}
    assert manager.missing_subtasks_for_full_episode() == [1]


def test_episode_dirs_for_full_subtask_uses_nested_segments_without_root_scan(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    first = _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    second = _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)
    monkeypatch.setattr(
        manager,
        "_iter_subtask_episode_dirs",
        lambda: (_ for _ in ()).throw(
            AssertionError("nested layout should not fall back to root scan")
        ),
    )

    assert manager._episode_dirs_for_full_subtask(0) == [first, second]
    assert manager._episode_dirs_for_full_subtask(0, 1) == [second]
    assert manager._saved_subtasks_cache[0] == {0, 1}


def test_update_task_info_license_toggle_does_not_rescan(monkeypatch, tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._status = "idle"
    manager._current_full_episode_index = 3
    manager._current_subtask_index = 1
    manager._current_scenario_number = 1
    manager.current_instruction = "old instruction"
    manager._include_robotis_license = False
    monkeypatch.setattr(
        manager,
        "_validate_existing_segment_count",
        lambda: (_ for _ in ()).throw(
            AssertionError("same layout refresh must not rescan validation")
        ),
    )
    monkeypatch.setattr(
        manager,
        "_find_next_subtask_position",
        lambda: (_ for _ in ()).throw(
            AssertionError("same layout refresh must not recompute cursor")
        ),
    )
    task_info = SimpleNamespace(
        task_instruction=["updated instruction"],
        subtask_instruction=["subtask 0", "subtask 1"],
        include_robotis_license=True,
    )

    manager.update_task_info(task_info)

    assert manager._include_robotis_license is True
    assert manager._current_full_episode_index == 3
    assert manager._current_subtask_index == 1
    assert manager.current_instruction == "updated instruction"


def test_full_episode_archive_errors_report_corrupt_saved_segment(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=1)
    manager._current_full_episode_index = 0
    segment = _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=1)
    (segment / "metadata.yaml").unlink()
    for mcap in segment.glob("*.mcap"):
        mcap.unlink()

    assert manager.full_episode_archive_errors() == [
        "subtask 0: missing metadata.yaml",
        "subtask 0: missing .mcap file",
    ]


def test_archive_writes_korean_subtask_instruction_as_utf8(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=2,
        with_video=False,
        subtask_instruction="화장품 집기",
    )
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=1,
        subtask_total=2,
        with_video=False,
        subtask_instruction="정리하기",
    )

    out = manager._archive_full_episode(0)

    raw = (out / "episode_info.json").read_text(encoding="utf-8")
    assert "화장품 집기" in raw
    assert "\\ud654" not in raw
