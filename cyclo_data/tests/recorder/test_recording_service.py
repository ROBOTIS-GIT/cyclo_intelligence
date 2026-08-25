from __future__ import annotations

from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
import sys


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
        UNSPECIFIED = 0
        SUCCESS = 1
        FAILURE = 2


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


_stub_module(
    "interfaces.msg",
    DataOperationStatus=_DataOperationStatus,
    RecordingStatus=_RecordingStatus,
)
_stub_module("interfaces.srv", RecordingCommand=_RecordingCommand)
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
    episode_outcome = attrs.pop("episode_outcome", 0)
    return SimpleNamespace(
        segment_index=segment_index,
        episode_outcome=episode_outcome,
        task_info=SimpleNamespace(tags=tags or [], task_type=""),
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
    service._node = SimpleNamespace(get_logger=lambda: logger)
    return service, logger


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


def test_cancel_active_inference_rollout_deletes_files_without_incrementing_count(
    tmp_path,
):
    service, _ = _service_with_logger()
    episode_dir = tmp_path / "5" / "segments" / "0"
    (episode_dir / "videos").mkdir(parents=True)
    (episode_dir / "episode.mcap").write_bytes(b"active")
    (episode_dir / "videos" / "cam.mjpeg.tmp").write_bytes(b"active")
    manager = SimpleNamespace(
        _record_episode_count=5,
        is_recording=lambda: True,
        get_save_rosbag_path=lambda: str(episode_dir),
        discard_recording=lambda **_kwargs: None,
    )
    events = []
    service._data_manager = manager
    service._stop_episode_writers = lambda: None
    service._rosbag = SimpleNamespace(
        stop_and_delete_rosbag=lambda: None,
        publish_action_event=lambda event: events.append(event),
    )
    service._publish_umbrella_status = lambda *_args, **_kwargs: None
    response = SimpleNamespace(success=False, message="")
    request = _request(command=_RecordingCommand.Request.CANCEL)
    request.task_info.task_type = "inference"

    result = service._do_cancel(request, response)

    assert result.success is True
    assert not episode_dir.exists()
    assert manager._record_episode_count == 5
    assert events == ["cancel"]


def _inference_stop_fixture(tmp_path):
    service, _ = _service_with_logger()
    metadata_calls = []
    rosbag_stops = []
    manager = SimpleNamespace(
        _task_info=SimpleNamespace(task_type="inference"),
        _record_episode_count=0,
        _segmented_storage_mode=True,
        is_recording=lambda: True,
        get_status=lambda: "recording",
        get_current_subtask_index=lambda: 0,
        get_save_rosbag_path=lambda: str(tmp_path / "0" / "segments" / "0"),
        save_robotis_metadata=lambda **kwargs: metadata_calls.append(kwargs),
        stop_recording=lambda **_kwargs: None,
    )
    service._data_manager = manager
    service._rosbag = SimpleNamespace(
        stop_rosbag=lambda: rosbag_stops.append(True),
        publish_action_event=lambda _event: None,
    )
    service._video_recorder = None
    service._camera_info = None
    service._last_camera_rotations = {}
    service._last_image_topics = {}
    service._last_camera_info_topics = {}
    service._publish_umbrella_status = lambda *_args, **_kwargs: None
    service._start_finish_episode_thread = lambda _manager: True
    return service, metadata_calls, rosbag_stops


def test_inference_stop_segment_allows_unspecified_outcome(tmp_path):
    service, metadata_calls, rosbag_stops = _inference_stop_fixture(tmp_path)
    request = _request(
        command=_RecordingCommand.Request.STOP_SEGMENT,
        episode_outcome=_RecordingCommand.Request.UNSPECIFIED,
        segment_index=0,
        urdf_path="",
    )
    request.task_info.task_type = "inference"
    response = SimpleNamespace(success=False, message="")

    result = service._do_stop_and_save(
        request,
        response,
        "STOP_SEGMENT",
        "finish",
    )

    assert result.success is True
    assert result.message == "Subtask saved"
    assert rosbag_stops == [True]
    assert metadata_calls[0]["episode_success"] is None


def test_inference_stop_requires_outcome_before_rosbag_stop(tmp_path):
    service, metadata_calls, rosbag_stops = _inference_stop_fixture(tmp_path)
    request = _request(
        command=_RecordingCommand.Request.STOP,
        episode_outcome=_RecordingCommand.Request.UNSPECIFIED,
        urdf_path="",
    )
    request.task_info.task_type = "inference"
    response = SimpleNamespace(success=True, message="")

    result = service._do_stop_and_save(request, response, "STOP", "finish")

    assert result.success is False
    assert "SUCCESS or FAILURE" in result.message
    assert rosbag_stops == []
    assert metadata_calls == []


def test_inference_stop_writes_episode_success_metadata(tmp_path):
    service, metadata_calls, rosbag_stops = _inference_stop_fixture(tmp_path)
    request = _request(
        command=_RecordingCommand.Request.STOP,
        episode_outcome=_RecordingCommand.Request.SUCCESS,
        urdf_path="",
    )
    request.task_info.task_type = "inference"
    response = SimpleNamespace(success=False, message="")

    result = service._do_stop_and_save(request, response, "STOP", "finish")

    assert result.success is True
    assert rosbag_stops == [True]
    assert metadata_calls[0]["episode_success"] is True


def test_inference_stop_reports_metadata_write_failure(tmp_path):
    service, _, rosbag_stops = _inference_stop_fixture(tmp_path)
    service._data_manager.save_robotis_metadata = lambda **_kwargs: False
    service._start_finish_episode_thread = lambda _manager: pytest.fail(
        "metadata-less episode must not be archived"
    )
    published = []
    service._publish_umbrella_status = lambda *args: published.append(args)
    request = _request(
        command=_RecordingCommand.Request.STOP,
        episode_outcome=_RecordingCommand.Request.FAILURE,
        urdf_path="",
    )
    request.task_info.task_type = "inference"
    response = SimpleNamespace(success=True, message="")

    result = service._do_stop_and_save(request, response, "STOP", "finish")

    assert result.success is False
    assert "metadata could not be saved" in result.message
    assert rosbag_stops == [True]
    assert published[-1][0] == _DataOperationStatus.FAILED


def test_inference_set_task_info_does_not_create_data_manager():
    service, _ = _service_with_logger()
    service._ensure_data_manager = lambda *_args: (_ for _ in ()).throw(
        AssertionError("DataManager must be lazy for inference")
    )
    request = _request(robot_type="ffw_sg2_rev1")
    request.task_info.task_type = "inference"
    response = SimpleNamespace(success=False, message="")

    result = service._do_set_task_info(request, response)

    assert result.success is True
    assert "starts on Record" in result.message
