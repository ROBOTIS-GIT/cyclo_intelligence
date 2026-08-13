import json
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import Mock


cv_bridge_module = types.ModuleType('cv_bridge')
cv_bridge_module.CvBridge = Mock
sys.modules.setdefault('cv_bridge', cv_bridge_module)

from interfaces.msg import TaskInfo
from interfaces.srv import RecordingCommand

from orchestrator.orchestrator_node import OrchestratorNode


def _task_info(service_type='lerobot'):
    task_info = TaskInfo()
    task_info.task_type = 'inference'
    task_info.task_name = 'inference'
    task_info.record_inference_mode = True
    task_info.service_type = service_type
    task_info.inference_mode = 'robot'
    return task_info


def _node():
    node = object.__new__(OrchestratorNode)
    node._state_lock = threading.Lock()
    node._recording_command_lock = threading.Lock()
    node.on_inference = True
    node.on_recording = False
    node._loaded_inference_publish_to_robot = True
    node._inference_record_session_id = '20260811_120000'
    node._inference_record_robot_type = 'ffw_sg2_rev1'
    node._prepared_inference_task_info = _task_info()
    node._last_ui_task_info = None
    node.robot_type = 'ffw_sg2_rev1'
    node.params = {}
    node.communicator = None
    node.get_logger = Mock(return_value=Mock())
    return node


def test_inference_session_id_uses_local_timestamp_and_collision_suffix(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        'orchestrator.orchestrator_node.time.strftime',
        lambda fmt, tm: '20260812_173000',
    )

    assert OrchestratorNode._new_inference_record_session_id(tmp_path) == (
        '20260812_173000'
    )

    (tmp_path / 'Task_20260812_173000_inference_MCAP').mkdir()
    assert OrchestratorNode._new_inference_record_session_id(tmp_path) == (
        '20260812_173000_01'
    )


def test_selected_inference_folder_is_reused(monkeypatch, tmp_path):
    folder = tmp_path / 'Task_existing_session_inference_MCAP'
    episode = folder / '0'
    episode.mkdir(parents=True)
    (episode / 'episode_info.json').write_text(json.dumps({
        'robot_type': 'ffw_sg2_rev1',
        'episode_success': True,
    }))
    node = _node()
    monkeypatch.setattr(node, 'INFERENCE_RECORD_ROOT', tmp_path)
    task_info = _task_info()
    task_info.task_num = 'existing_session'

    assert node._inference_record_folder_error(task_info) is None
    node._begin_inference_record_session(task_info)

    assert node._inference_record_session_id == 'existing_session'
    assert node._get_inference_record_task_info().task_num == 'existing_session'


def test_selected_inference_folder_rejects_missing_and_wrong_robot(
    monkeypatch,
    tmp_path,
):
    node = _node()
    monkeypatch.setattr(node, 'INFERENCE_RECORD_ROOT', tmp_path)
    task_info = _task_info()
    task_info.task_num = 'missing'

    assert 'does not exist' in node._inference_record_folder_error(task_info)

    folder = tmp_path / 'Task_wrong_robot_inference_MCAP' / '0'
    folder.mkdir(parents=True)
    (folder / 'episode_info.json').write_text(json.dumps({
        'robot_type': 'omy_f3m',
    }))
    task_info.task_num = 'wrong_robot'

    assert 'current robot_type' in node._inference_record_folder_error(task_info)

    unknown_folder = tmp_path / 'Task_unknown_robot_inference_MCAP' / '0'
    unknown_folder.mkdir(parents=True)
    (unknown_folder / 'episode_info.json').write_text('{}')
    task_info.task_num = 'unknown_robot'

    assert 'do not identify robot_type' in (
        node._inference_record_folder_error(task_info)
    )


def test_simulation_inference_recording_is_rejected():
    node = _node()
    node._loaded_inference_publish_to_robot = False

    assert node._inference_record_start_error(_task_info()) == (
        'RL Recording is only available for Real Robot deploy'
    )


def test_invalid_outcome_and_active_clear_are_rejected():
    node = _node()

    assert node._inference_record_outcome_error(0) is not None
    assert node._inference_record_outcome_error(3) is not None
    assert node._inference_record_outcome_error(1) is None
    assert node._inference_record_outcome_error(2) is None

    node.on_recording = True
    assert node._inference_clear_error() is not None


def test_act_and_groot_use_the_same_recording_forwarder():
    node = _node()
    calls = []
    node.communicator = SimpleNamespace(
        get_mcap_topics=lambda: ['/joint_states', '/leader/joint_states']
    )
    node._cyclo_data = SimpleNamespace(
        send_recording_command=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                success=True,
                response=SimpleNamespace(success=True),
                message='',
            )
        )
    )

    node._forward_recording(
        RecordingCommand.Request.START,
        task_info=_task_info('lerobot'),
        include_topics=True,
    )
    node._forward_recording(
        RecordingCommand.Request.START,
        task_info=_task_info('groot'),
        include_topics=True,
    )

    assert [call['command'] for call in calls] == [
        RecordingCommand.Request.START,
        RecordingCommand.Request.START,
    ]
    assert calls[0]['topics'] == calls[1]['topics']
    assert calls[0]['task_info'].service_type == 'lerobot'
    assert calls[1]['task_info'].service_type == 'groot'


def test_inference_joystick_does_not_control_recording():
    node = _node()
    node.communicator = object()
    node._forward_recording = Mock()

    node.handle_joystick_trigger('right')
    node.handle_joystick_trigger('left')

    node._forward_recording.assert_not_called()
