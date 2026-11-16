#!/usr/bin/env python3

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from interfaces.msg import InferenceStatus
from interfaces.srv import SendCommand
from orchestrator import orchestrator_node as orchestrator_node_module
from orchestrator.orchestrator_node import OrchestratorNode


class _Client:
    CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = range(6)

    def __init__(self, node, service_prefix, callback_group=None, command_observer=None):
        self.node = node
        self._service_prefix = service_prefix
        self._cancelled = threading.Event()
        self.disconnect = Mock()
        self.resume_reply = SimpleNamespace(success=False, message='Service call timed out: /policy/inference_command')
        self.before_resume_reply = lambda: None

    def connect(self):
        return True

    def inference_command(self, command, **kwargs):
        node = self.node
        node.commands.append(command)
        node._observe_inference_command(True)
        try:
            if command == self.CMD_RESUME:
                # The Runtime accepted RESUME, but its reply may be lost.
                node.runtime['running'] = True
                self.before_resume_reply()
                return self.resume_reply
            if command == self.CMD_LOAD:
                if node.runtime['loaded']:
                    return SimpleNamespace(success=False, message='policy already loaded - UNLOAD first')
                node.runtime['loaded'] = True
            elif command in (self.CMD_PAUSE, self.CMD_STOP):
                node.runtime['running'] = False
            elif command == self.CMD_UNLOAD:
                assert not node.runtime['running']
                node.runtime['loaded'] = False
            elif command == self.CMD_START:
                node.runtime['running'] = True
            return SimpleNamespace(success=True, message='ok', data={})
        finally:
            node._observe_inference_command(False)


@pytest.fixture
def session(monkeypatch):
    node = object.__new__(OrchestratorNode)
    node._state_lock = threading.RLock()
    node._recording_command_lock = threading.RLock()
    node._inference_lifecycle_lock = threading.Lock()
    node._client_cb_group = object()
    node._init_inference_status_monitor()
    node._initial_pose_sync_status_active = True
    node._initial_pose_sync_hold_pending = False
    node._inference_status_snapshot.update(phase=InferenceStatus.SYNCING, runtime_state='syncing')
    node._loaded_inference_policy_path = '/models/policy'
    node._loaded_inference_policy_id = ''
    node._loaded_inference_policy_parameters_json = '{}'
    node._loaded_inference_instruction = 'pick'
    node._loaded_inference_publish_to_robot = False
    node._loaded_inference_acceleration_mode = 'pytorch'
    node._loaded_inference_acceleration_engine_path = ''
    node._loaded_inference_action_request_mode = 'async'
    node._loaded_inference_control_hz = 100
    node._loaded_inference_inference_hz = 15
    node._loaded_inference_chunk_align_window_s = 0.3
    node._loaded_inference_initial_pose_sync = False
    node._loaded_inference_initial_pose_sync_duration_s = 5.0
    node.on_recording = True
    node.on_inference = True
    node._inference_recording = SimpleNamespace(session_id='active-recording')
    node.timer_manager = None
    node.robot_type = 'ffw_sg2'
    node.commands = []
    node.runtime = dict(loaded=True, running=False)
    node.get_logger = Mock(return_value=Mock())
    node._cache_ui_task_info = Mock()
    node.init_robot_control_parameters_from_user_task = Mock()
    node._inference_status_publisher = Mock()
    node._publish_inference_phase = Mock(wraps=node._publish_inference_phase)
    client = node.container_service_client = _Client(node, '/groot')
    pending = []
    monkeypatch.setattr(orchestrator_node_module, 'ContainerServiceClient', _Client)
    monkeypatch.setattr(orchestrator_node_module.threading, 'Thread',
                        lambda target, **kwargs: SimpleNamespace(start=lambda: pending.append(target)))
    return node, client, pending


def _resume(node, command=SendCommand.Request.START_INFERENCE):
    task_info = SimpleNamespace(
        task_instruction=[], policy_path='/models/policy', service_type='groot',
        inference_mode='simulation', tags=[], acceleration_mode='',
        acceleration_engine_path='', action_request_mode='',
        action_steps=0 if command == SendCommand.Request.START_INFERENCE else -1,
    )
    return node.user_interaction_callback(
        SimpleNamespace(command=command, task_info=task_info),
        SimpleNamespace(success=False, message=''),
    )


@pytest.mark.parametrize('message', [
    'not running', 'LOAD first', 'Service not available after 10.0s: /groot/inference_command',
    'Service call timed out: /groot/inference_command',
    'Service call returned None: /lerobot/inference_command',
    'Not connected to container services', 'Service client not initialized: /groot/inference_command',
    'Service call failed: failed to send request', 'No response from service (timeout or error)',
    'Service call cancelled: /groot/inference_command', 'Failed to resume inference: GPU out of memory',
])
@pytest.mark.parametrize('command', [SendCommand.Request.START_INFERENCE, SendCommand.Request.RESUME_INFERENCE])
def test_resume_error_preserves_unknown_live_session_without_cleanup(session, message, command):
    node, client, pending = session
    client.resume_reply = SimpleNamespace(success=False, message=message)
    snapshot = dict(node._inference_status_snapshot)

    result = _resume(node, command)
    for target in pending:
        target()

    assert not result.success
    assert message in result.message
    assert node.commands == [_Client.CMD_RESUME]
    assert node.container_service_client is client
    assert node._loaded_inference_policy_path == '/models/policy'
    assert node._loaded_inference_instruction == 'pick'
    assert node._initial_pose_sync_status_active
    assert node._inference_status_snapshot == snapshot
    assert node.runtime == dict(loaded=True, running=True)
    assert node.on_recording
    assert node._inference_recording.session_id == 'active-recording'
    assert not client._cancelled.is_set()
    client.disconnect.assert_not_called()
    node._publish_inference_phase.assert_not_called()
    node.init_robot_control_parameters_from_user_task.assert_not_called()
    assert not pending


@pytest.mark.parametrize('command', [SendCommand.Request.START_INFERENCE, SendCommand.Request.RESUME_INFERENCE])
@pytest.mark.parametrize('stop_command', [_Client.CMD_PAUSE, _Client.CMD_STOP])
@pytest.mark.parametrize('late_success', [False, True])
@pytest.mark.parametrize('sync_active', [False, True])
def test_late_resume_cannot_override_successful_same_client_stop(session, command, stop_command, late_success, sync_active):
    node, client, pending = session
    node._initial_pose_sync_status_active = sync_active
    client.resume_reply = SimpleNamespace(
        success=late_success,
        message='syncing' if late_success else 'Service call timed out: /policy/inference_command',
    )

    def stop_before_reply():
        if stop_command == _Client.CMD_STOP:
            assert node._stop_initial_pose_sync_for_teardown(client, force=True)
        else:
            assert node._pause_inference_client(client).success
        assert not node.runtime['running']
        node._publish_inference_phase(InferenceStatus.PAUSED)

    client.before_resume_reply = stop_before_reply
    result = _resume(node, command)
    for target in pending:
        target()

    assert not result.success
    assert node.commands == [_Client.CMD_RESUME, stop_command]
    assert node.runtime == dict(loaded=True, running=False)
    assert node.container_service_client is client
    assert not node._initial_pose_sync_status_active
    assert node._inference_status_snapshot['phase'] == InferenceStatus.PAUSED
    assert node._inference_status_snapshot['runtime_state'] == 'paused'
    assert node.on_recording
    assert node._inference_recording.session_id == 'active-recording'
    node._publish_inference_phase.assert_called_once_with(InferenceStatus.PAUSED)
    client.disconnect.assert_not_called()
    assert not pending


def test_operator_can_stop_then_explicitly_clear_after_lost_resume(session):
    node, client, pending = session
    assert not _resume(node).success
    assert node.runtime['running']

    def command(value):
        return node.user_interaction_callback(
            SimpleNamespace(command=value, task_info=SimpleNamespace(task_type='inference')),
            SimpleNamespace(success=False, message=''),
        )

    stopped = command(SendCommand.Request.STOP_INFERENCE)
    assert stopped.success, stopped.message
    assert not node.runtime['running']
    assert node.container_service_client is client
    assert node._inference_status_snapshot['phase'] == InferenceStatus.PAUSED
    assert node.on_recording

    # Clear still refuses to discard an active recording.
    blocked = command(SendCommand.Request.FINISH)
    assert not blocked.success
    assert 'Save or Discard recording' in blocked.message
    assert node.commands == [_Client.CMD_RESUME, _Client.CMD_PAUSE]

    node.on_recording = False
    node._forward_recording = Mock(return_value=SimpleNamespace(success=True))
    cleared = command(SendCommand.Request.FINISH)
    assert cleared.success, cleared.message
    assert len(pending) == 1
    pending[0]()

    assert node.commands == [_Client.CMD_RESUME, _Client.CMD_PAUSE, _Client.CMD_STOP, _Client.CMD_UNLOAD]
    assert node.runtime == dict(loaded=False, running=False)
    assert node.container_service_client is None
    assert node._inference_status_snapshot['phase'] == InferenceStatus.READY
    client.disconnect.assert_called_once()
