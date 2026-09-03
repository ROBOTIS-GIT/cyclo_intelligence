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
from interfaces.srv import RecordingCommand, SendCommand

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


def test_copy_task_info_preserves_rlt_runtime_fields():
    task_info = _task_info('groot')
    task_info.rlt_enabled = True
    task_info.rlt_bundle_path = '/workspace/checkpoint/rlt/bundle'
    task_info.action_policy_mode = 'rlt'
    task_info.rlt_robot_override = True

    copied = OrchestratorNode._copy_task_info(task_info)

    assert copied.rlt_enabled is True
    assert copied.rlt_bundle_path == '/workspace/checkpoint/rlt/bundle'
    assert copied.action_policy_mode == 'rlt'
    assert copied.rlt_robot_override is True


def test_non_groot_backend_forces_pytorch_acceleration():
    task_info = _task_info('lerobot')
    task_info.acceleration_mode = 'tensorrt_dit'
    task_info.acceleration_engine_path = '/workspace/model/groot/engine.trt'

    assert OrchestratorNode._acceleration_for_service(
        task_info,
        '/lerobot',
    ) == ('pytorch', '')


def test_groot_backend_preserves_supported_tensorrt_acceleration():
    task_info = _task_info('groot')
    task_info.acceleration_mode = 'tensorrt'
    task_info.acceleration_engine_path = (
        '/workspace/model/groot/showroom_groot/../showroom_groot/engine.trt'
    )

    assert OrchestratorNode._acceleration_for_service(
        task_info,
        '/groot',
    ) == (
        'tensorrt_dit',
        '/workspace/model/groot/showroom_groot/engine.trt',
    )


def test_orchestrator_preserves_tt_rtc_action_request_mode():
    assert OrchestratorNode._normalize_action_request_mode('tt_rtc') == 'tt_rtc'
    assert OrchestratorNode._normalize_action_request_mode('sync') == 'sync'


def test_orchestrator_rejects_tt_rtc_for_non_groot_backend():
    assert OrchestratorNode._action_request_mode_error(
        'tt_rtc',
        '/lerobot',
    ) == 'TT-RTC action requests are supported only by GR00T N1.7'
    assert OrchestratorNode._action_request_mode_error('tt_rtc', '/groot') == ''
    assert OrchestratorNode._action_request_mode_error('async', '/lerobot') == ''


def test_orchestrator_rejects_tensorrt_for_tt_rtc():
    assert OrchestratorNode._action_request_mode_error(
        'tt_rtc',
        '/groot',
        'tensorrt_dit',
    ) == (
        'TT-RTC currently requires PyTorch; disable TensorRT before starting '
        'inference'
    )


def test_action_policy_switch_is_forwarded_for_active_rlt_session():
    node = _node()
    client = SimpleNamespace(
        inference_command=Mock(
            return_value=SimpleNamespace(
                success=True,
                message='policy switched',
            )
        )
    )
    node.container_service_client = client
    node._loaded_inference_rlt_enabled = True
    node._loaded_inference_action_policy_mode = 'base'
    task_info = _task_info('groot')
    task_info.action_policy_mode = 'rlt'
    task_info.rlt_robot_override = True
    request = SimpleNamespace(
        command=SendCommand.Request.SET_ACTION_POLICY,
        task_info=task_info,
    )
    response = SimpleNamespace(success=False, message='')

    result = node.user_interaction_callback(request, response)

    assert result.success is True
    assert result.message == 'policy switched'
    client.inference_command.assert_called_once_with(
        7,
        action_policy_mode='rlt',
        rlt_robot_override=True,
    )
    assert node._loaded_inference_action_policy_mode == 'rlt'


def test_action_policy_switch_rejects_rlt_without_preloaded_bundle():
    node = _node()
    client = SimpleNamespace(inference_command=Mock())
    node.container_service_client = client
    node._loaded_inference_rlt_enabled = False
    task_info = _task_info('groot')
    task_info.action_policy_mode = 'rlt'
    request = SimpleNamespace(
        command=SendCommand.Request.SET_ACTION_POLICY,
        task_info=task_info,
    )
    response = SimpleNamespace(success=False, message='')

    result = node.user_interaction_callback(request, response)

    assert result.success is False
    assert 'no RLT bundle was preloaded' in result.message
    client.inference_command.assert_not_called()
