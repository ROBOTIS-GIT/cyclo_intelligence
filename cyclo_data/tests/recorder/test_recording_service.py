from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event, RLock, Thread
from unittest.mock import Mock
from types import ModuleType, SimpleNamespace
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))
sys.path.insert(0, str(_REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(_REPO_ROOT / "shared"))

import cyclo_data  # noqa: E402
import cyclo_data.recorder  # noqa: E402
import cyclo_data.services  # noqa: E402


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        module = sys.modules[name]
        for key, value in attrs.items():
            setattr(module, key, value)
        return
    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        sys.modules.setdefault(parent, ModuleType(parent))
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _RecordingStatus:
    READY = 0
    RECORDING = 1
    SAVING = 2


class _DataOperationStatus:
    OP_EDIT = 4
    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4


class _RecordingCommand:
    class Request:
        START = 0
        STOP = 1
        PAUSE = 2
        RESUME = 3
        FINISH = 4
        MOVE_TO_NEXT = 5
        RERECORD = 6
        SKIP_TASK = 7
        CANCEL = 8
        REFRESH_TOPICS = 9
        START_SEGMENT = 10
        STOP_SEGMENT = 11
        DISCARD_SEGMENT = 12
        FINISH_EPISODE = 13
        DISCARD_EPISODE = 14
        SET_TASK_INFO = 15
        CANCEL_SEGMENT = 16


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


class _EditDataset:
    class Request:
        MERGE = 0
        DELETE = 1


_stub_module(
    "interfaces.msg",
    DataOperationStatus=_DataOperationStatus,
    RecordingStatus=_RecordingStatus,
)
_stub_module("interfaces.srv", RecordingCommand=_RecordingCommand, EditDataset=_EditDataset)
_stub_module("cyclo_data.recorder.camera_info_snapshot", CameraInfoSnapshot=_Dummy)
_stub_module("cyclo_data.recorder.rosbag_control", RosbagControl=_Dummy)
_stub_module("cyclo_data.recorder.transcoder", TranscodeWorker=_Dummy)
_stub_module("cyclo_data.recorder.video_recorder", VideoRecorder=_Dummy)
_stub_module("huggingface_hub", HfApi=_Dummy)
_stub_module("cyclo_data.converter.orchestrator", DataConverter=_Dummy)
_stub_module(
    "cyclo_data.hub.progress_tracker",
    HuggingFaceLogCapture=_Dummy,
    HuggingFaceProgressTqdm=_Dummy,
)
_stub_module("psutil", cpu_percent=lambda interval=None: 0.0)

import cyclo_data.services.recording_service as recording_service_module  # noqa: E402
from cyclo_data.services.recording_service import RecordingService  # noqa: E402
from cyclo_data.services.edit_service import EditService  # noqa: E402

for _module_name in (
    "cyclo_data.recorder.camera_info_snapshot",
    "cyclo_data.recorder.rosbag_control",
    "cyclo_data.recorder.transcoder",
    "cyclo_data.recorder.video_recorder",
):
    sys.modules.pop(_module_name, None)
    _parent_name, _attr_name = _module_name.rsplit(".", 1)
    _parent = sys.modules.get(_parent_name)
    if _parent is not None and hasattr(_parent, _attr_name):
        delattr(_parent, _attr_name)


def _request(segment_index=0, tags=None, **attrs):
    return SimpleNamespace(
        segment_index=segment_index,
        task_info=SimpleNamespace(tags=tags or []),
        **attrs,
    )


def test_discard_episode_segment_index_zero_keeps_legacy_cursor_behavior():
    assert RecordingService._extract_full_episode_index(_request(0)) is None


def test_discard_episode_segment_index_encodes_full_episode_index_plus_one():
    assert RecordingService._extract_full_episode_index(_request(1)) == 0
    assert RecordingService._extract_full_episode_index(_request(8)) == 7


def test_discard_episode_accepts_transitional_explicit_target_fields():
    req = _request(0, has_full_episode_index=True, full_episode_index=7)
    assert RecordingService._extract_full_episode_index(req) == 7


def test_discard_episode_accepts_transitional_target_tag():
    req = _request(0, tags=["recording_full_episode_index:7"])
    assert RecordingService._extract_full_episode_index(req) == 7


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warn(self, message):
        self.warnings.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        pass

    def error(self, message):
        self.errors.append(message)


def _service_with_logger():
    logger = _Logger()
    service = RecordingService.__new__(RecordingService)
    service._command_lock = RLock()
    service._shutting_down = False
    service._node = SimpleNamespace(get_logger=lambda: logger)
    return service, logger


def test_shutdown_stops_native_writer_before_closing_local_writers_without_saving():
    service, _ = _service_with_logger()
    events = []
    manager = SimpleNamespace(is_recording=lambda: True, stop_recording=Mock())
    service._status_timer = None
    service._session_lock = RLock()
    service._data_manager = manager
    service._rosbag = SimpleNamespace(shutdown=lambda: events.append('bag stopped'))
    service._video_recorder = SimpleNamespace(close=lambda: events.append('video closed'))
    service._camera_info = SimpleNamespace(close=lambda: events.append('camera closed'))
    service._transcoder = None

    service.shutdown()

    assert events == ['bag stopped', 'video closed', 'camera closed']
    manager.stop_recording.assert_not_called()
    assert service._data_manager is None


def test_shutdown_rejects_late_commands_without_starting_another_episode():
    service, _ = _service_with_logger()
    service._shutting_down = True
    service._do_start = Mock()
    request = _request(command=_RecordingCommand.Request.START, robot_type='omy_f3m')
    request.task_info.task_num = 'late'
    response = SimpleNamespace(success=True, message='')

    service._callback(request, response)

    assert response.success is False
    assert 'shutdown' in response.message.lower()
    service._do_start.assert_not_called()


def test_shutdown_waits_for_inflight_command_before_stopping_writers():
    service, _ = _service_with_logger()
    entered, release, stopped = Event(), Event(), Event()
    service._status_timer = None
    service._session_lock = RLock()
    service._data_manager = None
    service._video_recorder = service._camera_info = service._transcoder = None
    service._rosbag = SimpleNamespace(shutdown=stopped.set)

    def start(request, response):
        entered.set()
        assert release.wait(2)
        return response

    service._do_start = start
    request = _request(command=_RecordingCommand.Request.START, robot_type='omy_f3m')
    request.task_info.task_num = 'inflight'
    callback = Thread(target=service._callback, args=(request, SimpleNamespace()))
    shutdown_thread = Thread(target=service.shutdown)
    callback.start()
    assert entered.wait(1)
    shutdown_thread.start()
    try:
        assert not stopped.wait(0.1), 'must not tear down an in-flight START'
    finally:
        release.set()
        callback.join(2)
        shutdown_thread.join(2)
    assert stopped.is_set()
    assert not callback.is_alive() and not shutdown_thread.is_alive()


def test_inference_save_failure_retains_ownership_and_retry_does_not_rewrite_metadata(tmp_path):
    service, _ = _service_with_logger()
    service._rosbag = SimpleNamespace(stop_rosbag=Mock(), publish_action_event=Mock())
    service._stop_episode_writers = Mock()
    service._publish_umbrella_status = Mock()
    service._last_video_stats = {}
    service._last_camera_info_files = {}
    service._last_camera_rotations = {}
    service._last_image_topics = {}
    service._last_camera_info_topics = {}
    dm = SimpleNamespace(
        save_robotis_metadata=Mock(),
        finish_full_episode=Mock(side_effect=[OSError('disk full'), tmp_path]),
        stop_recording=Mock(),
    )
    service._data_manager = dm
    request = SimpleNamespace(urdf_path='')
    response = SimpleNamespace(success=False, message='')
    with pytest.raises(OSError, match='disk full'):
        service._save_inference_episode(request, response, 'save')
    assert not response.success
    dm.stop_recording.assert_not_called()
    assert service._inference_archive_pending
    service._save_inference_episode(request, response, 'save')
    assert response.success
    dm.save_robotis_metadata.assert_called_once()
    service._rosbag.stop_rosbag.assert_called_once()
    dm.stop_recording.assert_called_once_with(finish_full_episode=False)
    assert not service._inference_archive_pending


def test_inference_discard_filesystem_failure_keeps_active_then_retries(tmp_path, monkeypatch):
    import shutil
    service, _ = _service_with_logger()
    episode = tmp_path / '4'
    segment = episode / 'segments/0'
    segment.mkdir(parents=True)
    (segment / 'partial.mcap').write_bytes(b'data')
    service._data_manager = SimpleNamespace(
        _task_info=SimpleNamespace(task_type='inference'),
        get_save_rosbag_path=lambda: str(segment),
        get_current_full_episode_index=lambda: 4,
        _full_episode_dir=lambda index: tmp_path / str(index),
        discard_recording=Mock(),
    )
    service._rosbag = SimpleNamespace(stop_and_delete_rosbag=Mock(), publish_action_event=Mock())
    service._stop_episode_writers = Mock()
    service._publish_umbrella_status = Mock()
    original = shutil.rmtree
    monkeypatch.setattr(shutil, 'rmtree', Mock(side_effect=PermissionError('denied')))
    response = SimpleNamespace(success=False, message='')
    with pytest.raises(PermissionError):
        service._do_discard(SimpleNamespace(), response, reset_subtask_index=False, event='cancel')
    service._data_manager.discard_recording.assert_not_called()
    monkeypatch.setattr(shutil, 'rmtree', original)
    service._do_discard(SimpleNamespace(), response, reset_subtask_index=False, event='cancel')
    assert response.success
    assert not episode.exists()
    service._rosbag.stop_and_delete_rosbag.assert_called_once()
    service._stop_episode_writers.assert_called_once()


def test_inference_metadata_failure_keeps_original_video_statistics(tmp_path):
    service, _ = _service_with_logger()
    service._rosbag = SimpleNamespace(stop_rosbag=Mock(), publish_action_event=Mock())
    service._stop_episode_writers = Mock()
    service._publish_umbrella_status = Mock()
    service._last_video_stats = {'cam': {'frames_written': 20}}
    service._last_camera_info_files = {}
    service._last_camera_rotations = {}
    service._last_image_topics = {}
    service._last_camera_info_topics = {}
    service._data_manager = SimpleNamespace(
        save_robotis_metadata=Mock(side_effect=[OSError('disk full'), None]),
        finish_full_episode=Mock(return_value=tmp_path), stop_recording=Mock())
    response = SimpleNamespace(success=False, message='')
    with pytest.raises(OSError):
        service._save_inference_episode(SimpleNamespace(), response, 'save')
    service._save_inference_episode(SimpleNamespace(), response, 'save')
    service._stop_episode_writers.assert_called_once()
    service._rosbag.stop_rosbag.assert_called_once()
    assert response.success
    assert service._data_manager.save_robotis_metadata.call_args.kwargs['video_stats']['cam']['frames_written'] == 20


def test_ensure_data_manager_reuses_same_task_without_candidate_scan(
    monkeypatch,
    tmp_path,
):
    service, _ = _service_with_logger()
    service._session_lock = RLock()
    service.DEFAULT_SAVE_ROOT_PATH = tmp_path
    updates = []
    existing = SimpleNamespace(
        _save_repo_name="Task_42_pick_MCAP",
        is_recording=lambda: False,
        update_task_info=lambda task_info: updates.append(task_info),
    )
    service._data_manager = existing

    class ExplodingDataManager:
        @classmethod
        def _make_save_repo_name(cls, save_root_path, task_info):
            assert save_root_path == tmp_path
            return "Task_42_pick_MCAP"

        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "same task must not construct a scan-heavy manager"
            )

    monkeypatch.setattr(
        recording_service_module,
        "DataManager",
        ExplodingDataManager,
    )
    task_info = SimpleNamespace(task_num="42", task_name="pick")

    assert service._ensure_data_manager(task_info, "ffw_sg2_rev1") is existing
    assert updates == [task_info]


def test_validate_active_segment_rejects_stale_segment_request():
    service, logger = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        get_current_subtask_index=lambda: 1,
    )
    response = SimpleNamespace(success=True, message="")

    ok = service._validate_active_segment(
        _request(segment_index=2),
        response,
        "STOP_SEGMENT",
    )

    assert ok is False
    assert response.success is False
    assert response.message == "STOP_SEGMENT: active subtask is 1, but request targeted 2"
    assert logger.warnings == [response.message]


def test_validate_active_segment_accepts_current_segment_request():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        get_current_subtask_index=lambda: 1,
    )
    response = SimpleNamespace(success=True, message="")

    ok = service._validate_active_segment(
        _request(segment_index=1),
        response,
        "STOP_SEGMENT",
    )

    assert ok is True
    assert response.success is True


def test_start_segment_rejects_when_recording_is_already_active():
    service, logger = _service_with_logger()
    service._finish_episode_in_progress = lambda: False
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    data_manager = SimpleNamespace(
        is_recording=lambda: True,
        set_current_subtask_index=lambda index: (_ for _ in ()).throw(
            AssertionError("must not change subtask while recording")
        ),
    )
    service._ensure_data_manager = lambda task_info, robot_type: data_manager
    response = SimpleNamespace(success=True, message="")
    request = _request(
        segment_index=2,
        command=_RecordingCommand.Request.START_SEGMENT,
        robot_type="ffw_sg2_rev1",
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert response.message == "START blocked: recording already active"
    assert logger.warnings == [response.message]


def test_start_segment_rejects_request_that_skips_next_missing_subtask():
    service, logger = _service_with_logger()
    service._finish_episode_in_progress = lambda: False
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [1, 2],
        set_current_subtask_index=lambda index: (_ for _ in ()).throw(
            AssertionError("must not jump over a missing subtask")
        ),
    )
    service._ensure_data_manager = lambda task_info, robot_type: data_manager
    response = SimpleNamespace(success=True, message="")
    request = _request(
        segment_index=2,
        command=_RecordingCommand.Request.START_SEGMENT,
        robot_type="ffw_sg2_rev1",
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert response.message == (
        "START_SEGMENT: next available subtask is 1, but request targeted 2"
    )
    assert logger.warnings == [response.message]


def test_start_segment_rejects_when_current_episode_is_already_complete():
    service, logger = _service_with_logger()
    service._finish_episode_in_progress = lambda: False
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [],
        set_current_subtask_index=lambda index: (_ for _ in ()).throw(
            AssertionError("must not restart a complete episode")
        ),
    )
    service._ensure_data_manager = lambda task_info, robot_type: data_manager
    response = SimpleNamespace(success=True, message="")
    request = _request(
        segment_index=1,
        command=_RecordingCommand.Request.START_SEGMENT,
        robot_type="ffw_sg2_rev1",
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert response.message == (
        "START_SEGMENT: current episode already has all subtasks; "
        "finish or discard episode before starting again"
    )
    assert logger.warnings == [response.message]


def test_finish_episode_rejects_missing_subtasks_before_archive_thread():
    service, logger = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [1],
    )
    service._start_finish_episode_thread = lambda data_manager: (_ for _ in ()).throw(
        AssertionError("archive thread must not start with missing subtasks")
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(command=_RecordingCommand.Request.FINISH_EPISODE)

    result = service._do_finish_episode(request, response)

    assert result is response
    assert response.success is False
    assert response.message == "FINISH_EPISODE: missing subtask(s) [1]"
    assert logger.warnings == [response.message]


def test_stop_segment_rejects_when_no_active_recording():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(is_recording=lambda: False)
    response = SimpleNamespace(success=True, message="")
    request = _request(command=_RecordingCommand.Request.STOP_SEGMENT)

    result = service._do_stop_and_save(
        request,
        response,
        "STOP_SEGMENT",
        event="finish",
    )

    assert result is response
    assert response.success is False
    assert response.message == "STOP_SEGMENT: no active recording"


def test_stop_segment_saves_metadata_even_without_urdf_path(tmp_path):
    service, _ = _service_with_logger()
    episode_dir = tmp_path / "0" / "segments" / "0"
    metadata_calls = []
    stopped = []
    events = []

    data_manager = SimpleNamespace(
        _record_episode_count=0,
        _segmented_storage_mode=True,
        is_recording=lambda: True,
        get_current_subtask_index=lambda: 0,
        get_status=lambda: "recording",
        get_save_rosbag_path=lambda: str(episode_dir),
        save_robotis_metadata=lambda **kwargs: metadata_calls.append(kwargs),
        stop_recording=lambda **kwargs: stopped.append(kwargs),
    )
    service._data_manager = data_manager
    service._rosbag = SimpleNamespace(
        stop_rosbag=lambda: None,
        publish_action_event=lambda event: events.append(event),
    )
    service._video_recorder = None
    service._camera_info = None
    service._last_camera_rotations = {"cam0": 0}
    service._last_image_topics = {"cam0": "/image"}
    service._last_camera_info_topics = {"cam0": "/camera_info"}
    service._publish_umbrella_status = lambda *args, **kwargs: None
    response = SimpleNamespace(success=True, message="")
    request = _request(
        command=_RecordingCommand.Request.STOP_SEGMENT,
        segment_index=0,
        urdf_path="",
    )

    result = service._do_stop_and_save(
        request,
        response,
        "STOP_SEGMENT",
        event="finish",
    )

    assert result is response
    assert response.success is True
    assert response.message == "Subtask saved"
    assert len(metadata_calls) == 1
    assert metadata_calls[0]["urdf_path"] == ""
    assert metadata_calls[0]["camera_rotations"] == {"cam0": 0}
    assert stopped == [{"finish_full_episode": False}]
    assert events == ["finish"]


def test_start_segment_rolls_back_rosbag_when_writer_start_fails(tmp_path):
    service, logger = _service_with_logger()
    episode_dir = tmp_path / "0" / "segments" / "0"
    rosbag_calls = []
    stopped_writers = []
    started = []

    class FailingVideoRecorder:
        def start_episode(self, _episode_dir):
            raise RuntimeError("camera writer boom")

        def stop_episode(self):
            stopped_writers.append(True)
            return {}

    data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [0],
        set_current_subtask_index=lambda index: None,
        get_save_rosbag_path=lambda allow_idle=False: str(episode_dir),
        start_recording=lambda: started.append(True),
    )
    service._finish_episode_in_progress = lambda: False
    service._ensure_data_manager = lambda task_info, robot_type: data_manager
    service._ensure_video_pipeline = lambda robot_type: None
    service._last_prepared_topics = ()
    service._video_recorder = FailingVideoRecorder()
    service._camera_info = None
    service._publish_umbrella_status = lambda *args, **kwargs: None

    def start_rosbag(rosbag_uri):
        rosbag_calls.append(("start", rosbag_uri))
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "partial.mcap").write_bytes(b"partial")

    service._rosbag = SimpleNamespace(
        is_available=lambda: True,
        start_rosbag=start_rosbag,
        stop_and_delete_rosbag=lambda: rosbag_calls.append(("stop_delete", None)),
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(
        command=_RecordingCommand.Request.START_SEGMENT,
        segment_index=0,
        robot_type="ffw_sg2_rev1",
        topics=[],
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert "camera writer boom" in response.message
    assert rosbag_calls == [
        ("start", str(episode_dir)),
        ("stop_delete", None),
    ]
    assert stopped_writers == [True]
    assert started == []
    assert not episode_dir.exists()
    assert logger.errors == [response.message]


def test_refresh_topics_caches_robot_type_for_idle_status():
    service, _ = _service_with_logger()
    prepared_topics = []
    video_robot_types = []
    published = []

    service._session_lock = RLock()
    service._robot_type = ''
    service._data_manager = None
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    service._prepare_rosbag_topics = lambda topics: prepared_topics.extend(topics)
    service._ensure_video_pipeline = lambda robot_type: video_robot_types.append(robot_type)
    service._recording_status_pub = SimpleNamespace(
        publish=lambda status: published.append(status)
    )
    service._video_recorder = None
    service._cpu_checker = SimpleNamespace(get_cpu_usage=lambda: 0.0)
    response = SimpleNamespace(success=False, message='')
    request = _request(
        robot_type='ffw_sg2_rev2',
        topics=['/joint_states'],
    )

    result = service._do_refresh_topics(request, response)
    service._publish_recording_status()

    assert result is response
    assert response.success is True
    assert service._robot_type == 'ffw_sg2_rev2'
    assert prepared_topics == ['/joint_states']
    assert video_robot_types == ['ffw_sg2_rev2']
    assert published[-1].robot_type == 'ffw_sg2_rev2'


def test_recording_status_prefers_current_service_robot_type_over_old_manager():
    service, _ = _service_with_logger()
    published = []

    service._session_lock = RLock()
    service._robot_type = 'ffw_sg2_rev2'
    service._data_manager = SimpleNamespace(
        get_current_record_status=lambda: SimpleNamespace(
            robot_type='ffw_sg2_rev1',
            record_phase=_RecordingStatus.READY,
        )
    )
    service._video_recorder = None
    service._recording_status_pub = SimpleNamespace(
        publish=lambda status: published.append(status)
    )

    service._publish_recording_status()

    assert published[-1].robot_type == 'ffw_sg2_rev2'


def test_cancel_segment_rejects_when_no_active_recording():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(is_recording=lambda: False)
    service._publish_umbrella_status = lambda *args, **kwargs: None
    response = SimpleNamespace(success=True, message="")
    request = _request(command=_RecordingCommand.Request.CANCEL_SEGMENT)

    result = service._do_cancel(request, response)

    assert result is response
    assert response.success is False
    assert response.message == "CANCEL_SEGMENT: no active recording"


def _episode_edit_services(tmp_path, monkeypatch, *, segments=1):
    """Exercise the real delete callback, filesystem scan, and status payload."""
    from cyclo_data.recorder.session_manager import DataManager
    from cyclo_data.recorder import session_manager

    root = tmp_path / "task"
    root.mkdir()
    for index in range(6):
        episode = root / str(index)
        episode.mkdir()
        (episode / "episode_info.json").write_text(json.dumps({
            "episode_index": index,
            **({"segments": [{"segment_index": i} for i in range(segments)]}
               if segments else {}),
        }))
        (episode / f"{index}_0.mcap").write_bytes(b"test")
        (episode / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")

    dm = DataManager.__new__(DataManager)
    dm._save_rosbag_path = str(root)
    dm._state_lock = RLock()
    dm._status = "idle"
    dm._start_time_s = 0
    dm._record_episode_count = 6
    dm._current_full_episode_index = 6
    dm._current_subtask_index = 0
    dm._current_scenario_number = 0
    dm._segmented_storage_mode = True
    dm._single_task = segments <= 1
    dm._subtask_mode = segments > 1
    dm._physical_segment_total = max(1, segments)
    dm._subtask_total = segments if segments > 1 else 0
    dm._subtask_instructions = [f"segment {i}" for i in range(dm._subtask_total)]
    dm._saved_subtasks_cache = {i: set(range(segments)) for i in range(6)}
    dm._robot_type = "test"
    dm._task_info = SimpleNamespace()
    dm.current_instruction = "test"
    dm._main_task_instruction = "test"
    dm._cpu_checker = SimpleNamespace(get_cpu_usage=lambda: 0)
    monkeypatch.setattr(session_manager.RAMChecker, "get_ram_gb", lambda: (1, 0))
    monkeypatch.setattr(session_manager.StorageChecker, "get_storage_gb", lambda _: (1, 0))

    recording, logger = _service_with_logger()
    recording._session_lock = RLock()
    recording._command_lock = RLock()
    recording._transcoder = None
    recording._data_manager = dm
    recording._finish_episode_in_progress = lambda: False
    published = []
    recording._publish_recording_status = lambda: published.append(dm.get_current_record_status())
    group = object()
    registrations = []
    node = SimpleNamespace(
        io_callback_group=group,
        get_logger=lambda: logger,
        create_service=lambda *args, **kwargs: registrations.append(kwargs),
    )
    edit = EditService(node, SimpleNamespace(publish=lambda _: None), recording)
    assert registrations[0]["callback_group"] is group
    request = SimpleNamespace(
        mode=_EditDataset.Request.DELETE,
        delete_task_dir=str(root),
        delete_episode_num=[1, 2, 3, 4, 5],
        delete_compact=True,
    )
    return edit, recording, dm, request, published


@pytest.mark.parametrize("segments", [0, 1, 2])
@pytest.mark.parametrize("indices,compact,expected", [
    ([1, 2, 3, 4, 5], True, 1),
    ([0, 1, 2, 3, 4, 5], True, 0),
    ([1, 3], True, 4),
    ([1, 3], False, 6),
    ([5], False, 5),
])
def test_delete_updates_record_status_and_next_recording(
    tmp_path, monkeypatch, segments, indices, compact, expected,
):
    edit, _, dm, request, published = _episode_edit_services(
        tmp_path, monkeypatch, segments=segments,
    )
    request.delete_episode_num = indices
    request.delete_compact = compact

    response = edit._callback(request, SimpleNamespace())

    assert response.success, response.message
    assert response.affected_count == len(indices)
    assert published[-1].current_episode_number == expected
    assert published[-1].current_subtask_index == 0
    assert dm.get_current_full_episode_index() == expected
    assert dm._record_episode_count == expected
    assert expected not in dm._saved_subtasks_cache
    dm.start_recording()
    assert dm.get_current_record_status().current_episode_number == expected
    assert not (Path(dm._save_rosbag_path) / str(expected)).exists()


@pytest.mark.parametrize("busy", ["recording", "save", "archive"])
def test_delete_rejects_busy_selected_task_before_changing_files(tmp_path, monkeypatch, busy):
    edit, recording, dm, request, published = _episode_edit_services(tmp_path, monkeypatch)
    if busy == "archive":
        recording._finish_episode_in_progress = lambda: True
    else:
        dm._status = busy

    response = edit._callback(request, SimpleNamespace())

    assert not response.success
    assert "wait for saving" in response.message
    assert len(list(Path(dm._save_rosbag_path).iterdir())) == 6
    assert dm._current_full_episode_index == 6
    assert published == []


def test_delete_other_task_leaves_recording_session_untouched(tmp_path, monkeypatch):
    edit, _, dm, request, published = _episode_edit_services(tmp_path, monkeypatch)
    other = tmp_path / "other"
    (other / "0").mkdir(parents=True)
    (other / "0" / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    (other / "0" / "0_0.mcap").write_bytes(b"test")
    request.delete_task_dir = str(other)
    request.delete_episode_num = [0]
    response = edit._callback(request, SimpleNamespace())

    assert response.success
    assert list(other.iterdir()) == []
    assert dm._current_full_episode_index == 6
    assert published == []


def test_delete_refreshes_cursor_after_partial_filesystem_failure(tmp_path, monkeypatch):
    edit, _, dm, request, published = _episode_edit_services(tmp_path, monkeypatch)
    delete = edit._editor.delete_rosbag_episodes

    def partial_delete(*args, **kwargs):
        delete(*args, **kwargs)
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(edit._editor, "delete_rosbag_episodes", partial_delete)
    response = edit._callback(request, SimpleNamespace())

    assert not response.success
    assert "simulated filesystem failure" in response.message
    assert published[-1].current_episode_number == 1
    assert dm._current_full_episode_index == 1


def test_delete_without_recording_session(tmp_path, monkeypatch):
    edit, recording, _, request, published = _episode_edit_services(tmp_path, monkeypatch)
    recording._data_manager = None

    response = edit._callback(request, SimpleNamespace())

    assert response.success
    assert published == []


def test_delete_does_not_resume_interrupted_inference_episode(tmp_path, monkeypatch):
    edit, _, dm, request, _ = _episode_edit_services(tmp_path, monkeypatch, segments=2)
    dm._task_info = SimpleNamespace(task_type="inference")
    dm._current_full_episode_index = dm._record_episode_count = 7
    segment = Path(dm._save_rosbag_path) / "6" / "segments" / "0"
    segment.mkdir(parents=True)
    info = segment / "episode_info.json"
    original = json.dumps({"episode_index": 6, "recording_mode": "subtask",
                           "full_episode_index": 6, "subtask_index": 0, "subtask_total": 2})
    info.write_text(original)
    request.delete_episode_num = [5]
    request.delete_compact = False
    result = edit._callback(request, SimpleNamespace())
    assert result.success, result.message
    assert dm.get_current_full_episode_index() == 7
    assert dm._current_subtask_index == 0
    assert info.read_text() == original


@pytest.mark.parametrize("busy", ["recording", "archive", "shutdown"])
def test_delete_other_task_rejected_without_blocking_stop(tmp_path, monkeypatch, busy):
    edit, recording, dm, request, _ = _episode_edit_services(tmp_path, monkeypatch)
    other = tmp_path / "other"
    (other / "0").mkdir(parents=True)
    (other / "0" / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    (other / "0" / "0_0.mcap").write_bytes(b"test")
    request.delete_task_dir = str(other)
    request.delete_episode_num = [0]
    if busy == "archive":
        recording._finish_episode_in_progress = lambda: True
    elif busy == "shutdown":
        recording._shutting_down = True
    else:
        dm._status = "recording"
    result = edit._callback(request, SimpleNamespace())
    assert not result.success
    assert (other / "0" / "0_0.mcap").read_bytes() == b"test"
    # A rejected edit must not retain the lock needed by STOP or shutdown.
    with ThreadPoolExecutor(max_workers=1) as pool:
        def acquire_and_release():
            acquired = recording._command_lock.acquire(blocking=False)
            if acquired:
                recording._command_lock.release()
            return acquired
        assert pool.submit(acquire_and_release).result(timeout=2)


@pytest.mark.parametrize("selected", [True, False])
def test_delete_waits_for_background_transcode(tmp_path, monkeypatch, selected):
    from concurrent.futures import Future
    from cyclo_data.recorder.transcoder import TranscodeWorker

    edit, recording, dm, request, published = _episode_edit_services(tmp_path, monkeypatch)
    root = Path(dm._save_rosbag_path)
    worker = recording._transcoder = TranscodeWorker()
    pending = Future()
    worker._inflight[str(root / "5")] = pending
    if not selected:
        recording._data_manager = None
    request.delete_episode_num = [5]
    request.delete_compact = False
    result = edit._callback(request, SimpleNamespace())
    assert not result.success
    assert "transcod" in result.message.lower()
    assert (root / "5" / "episode_info.json").is_file()
    assert dm.get_current_full_episode_index() == 6
    assert published == []
    pending.set_result(None)
    assert edit._callback(request, SimpleNamespace()).success
    if selected:
        assert dm.get_current_full_episode_index() == 5


def test_recording_command_waits_for_delete_and_cursor_refresh(tmp_path, monkeypatch):
    edit, recording, dm, request, _ = _episode_edit_services(tmp_path, monkeypatch)
    deleting = Event()
    release = Event()
    command_called = Event()
    dispatched = Event()
    delete = edit._editor.delete_rosbag_episodes

    def slow_delete(*args, **kwargs):
        deleting.set()
        assert release.wait(5)
        return delete(*args, **kwargs)

    def dispatch(*args):
        dispatched.set()
        return dm.get_current_full_episode_index()

    def command():
        command_called.set()
        return recording._callback(None, None)

    monkeypatch.setattr(edit._editor, "delete_rosbag_episodes", slow_delete)
    monkeypatch.setattr(recording, "_dispatch", dispatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(edit._callback, request, SimpleNamespace())
        try:
            assert deleting.wait(5)
            pending_command = pool.submit(command)
            assert command_called.wait(5)
            assert not dispatched.wait(0.05)
        finally:
            release.set()
        assert deletion.result(timeout=5).success
        assert pending_command.result(timeout=5) == 1


def test_delete_resolves_alias_of_selected_task(tmp_path, monkeypatch):
    edit, _, dm, request, published = _episode_edit_services(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    alias.symlink_to(dm._save_rosbag_path, target_is_directory=True)
    request.delete_task_dir = str(alias)

    assert edit._callback(request, SimpleNamespace()).success
    assert published[-1].current_episode_number == 1


def test_delete_does_not_starve_recording_command_response(tmp_path, monkeypatch):
    edit, recording, dm, request, _ = _episode_edit_services(tmp_path, monkeypatch)
    command_started = Event()
    delete_started = Event()
    response_ready = Event()

    def dispatch(*args):
        command_started.set()
        return response_ready.wait(2)

    def delete():
        delete_started.set()
        return edit._callback(request, SimpleNamespace())

    monkeypatch.setattr(recording, "_dispatch", dispatch)
    # Model a two-thread ROS executor: command, edit, then rosbag response.
    with ThreadPoolExecutor(max_workers=2) as pool:
        command = pool.submit(recording._callback, None, None)
        assert command_started.wait(5)
        deletion = pool.submit(delete)
        assert delete_started.wait(5)
        response = pool.submit(response_ready.set)
        assert command.result(timeout=5), "DELETE blocked the rosbag response"
        response.result(timeout=5)
        result = deletion.result(timeout=5)

    assert not result.success
    assert "retry" in result.message.lower()
    assert len(list(Path(dm._save_rosbag_path).iterdir())) == 6


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("total,saved", [(1, 1), (2, 1), (2, 2)])
def test_delete_preserves_segments_pending_finish(
    tmp_path, monkeypatch, compact, total, saved,
):
    edit, _, dm, request, published = _episode_edit_services(
        tmp_path, monkeypatch, segments=total,
    )
    root = Path(dm._save_rosbag_path)
    for index in range(saved):
        segment = root / "6" / "segments" / str(index)
        segment.mkdir(parents=True)
        (segment / "episode_info.json").write_text(json.dumps({
            "episode_index": 6 + index,
            "recording_mode": "subtask" if total > 1 else "single_segment",
            "full_episode_index": 6,
            "subtask_index": index,
            "subtask_total": total,
        }))
    request.delete_episode_num = [0]
    request.delete_compact = compact

    result = edit._callback(request, SimpleNamespace())

    assert not result.success
    assert "finish or discard" in result.message.lower()
    assert (root / "0").is_dir()
    assert (root / "6" / "segments" / "0" / "episode_info.json").is_file()
    assert dm.get_current_full_episode_index() == 6
    assert dm.saved_subtask_indices_for_full_episode() == set(range(saved))
    assert published == []

    # After DISCARD_EPISODE removes the saved segments, editing is allowed.
    dm.discard_current_full_episode()
    assert edit._callback(request, SimpleNamespace()).success
