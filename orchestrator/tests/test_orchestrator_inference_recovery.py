#!/usr/bin/env python3

import threading
from types import SimpleNamespace

import pytest

from interfaces.srv import SendCommand
from orchestrator import orchestrator_node as orchestrator_node_module
from orchestrator.orchestrator_node import OrchestratorNode


@pytest.mark.parametrize(
    'message',
    [
        'not running',
        '  LOAD FIRST  ',
        (
            'Service not available after 10.0s: '
            '/groot/inference_command'
        ),
        'SERVICE CALL TIMED OUT: /groot/inference_command',
        'Service call returned None: /lerobot/inference_command',
        'Not connected to container services',
        'Service client not initialized: /groot/inference_command',
        'Service call failed: failed to send request',
        'No response from service (timeout or error)',
    ],
)
def test_cached_resume_transport_failure_requires_fresh_load(message):
    assert OrchestratorNode._is_stale_inference_client_failure(message)


@pytest.mark.parametrize(
    'message',
    [
        '',
        'Failed to resume inference: GPU out of memory',
        'RESUME rejected while policy is loading',
        'Invalid inference command',
        'Connection refused by policy runtime',
        # A concurrent STOP tears down and cancels the cached client. Do not
        # turn that intentional cancellation into a new LOAD/START session.
        'Service call cancelled: /groot/inference_command',
    ],
)
def test_cached_resume_backend_rejection_does_not_trigger_reload(message):
    assert not OrchestratorNode._is_stale_inference_client_failure(message)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args, **kwargs):
        self.messages.append(('info', str(message)))

    def warning(self, message, *args, **kwargs):
        self.messages.append(('warning', str(message)))

    def error(self, message, *args, **kwargs):
        self.messages.append(('error', str(message)))


class _CachedClient:
    def __init__(self, on_resume=None):
        self._service_prefix = '/groot'
        self._cancelled = threading.Event()
        self.on_resume = on_resume
        self.commands = []
        self.disconnect_calls = 0

    def inference_command(self, command, **kwargs):
        self.commands.append(command)
        if self.on_resume is not None:
            self.on_resume()
        return SimpleNamespace(
            success=False,
            message=(
                'Service not available after 10.0s: '
                '/groot/inference_command'
            ),
        )

    def disconnect(self):
        self.disconnect_calls += 1


class _FreshClient:
    CMD_LOAD = 0
    CMD_START = 1
    CMD_RESUME = 3
    CMD_STOP = 4
    CMD_UNLOAD = 5
    instances = []

    def __init__(self, node, service_prefix, callback_group):
        self.node = node
        self._service_prefix = service_prefix
        self.callback_group = callback_group
        self._cancelled = threading.Event()
        self.connected = False
        self.__class__.instances.append(self)

    def connect(self):
        self.connected = True
        return True


class _DeferredThread:
    instances = []

    def __init__(self, target, **kwargs):
        self.target = target
        self.kwargs = kwargs
        self.started = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True


def _bare_node(client):
    node = object.__new__(OrchestratorNode)
    node._state_lock = threading.Lock()
    node._inference_lifecycle_lock = threading.Lock()
    node._client_cb_group = object()
    node.container_service_client = client
    node._loaded_inference_policy_path = '/models/policy'
    node._loaded_inference_publish_to_robot = False
    node._loaded_inference_acceleration_mode = 'pytorch'
    node._loaded_inference_acceleration_engine_path = ''
    node._loaded_inference_action_request_mode = 'async'
    node._loaded_inference_control_hz = 100
    node._loaded_inference_inference_hz = 15
    node._loaded_inference_chunk_align_window_s = 0.3
    node._loaded_inference_initial_pose_sync = False
    node._loaded_inference_initial_pose_sync_duration_s = 5.0
    node._initial_pose_sync_status_timer = None
    node._initial_pose_sync_status_generation = 0
    node._initial_pose_sync_hold_pending = False
    node.robot_type = 'ffw_sg2'
    node.logger = _Logger()
    node.published_phases = []
    node._cache_ui_task_info = lambda *args, **kwargs: None
    node.init_robot_control_parameters_from_user_task = (
        lambda *args, **kwargs: None
    )
    node._publish_inference_phase = (
        lambda phase, **kwargs: node.published_phases.append(phase)
    )
    node.get_logger = lambda: node.logger
    return node


def _start_request():
    task_info = SimpleNamespace(
        task_instruction=[],
        policy_path='/models/policy',
        service_type='groot',
        inference_mode='simulation',
        tags=[],
        acceleration_mode='',
        acceleration_engine_path='',
        action_request_mode='',
    )
    return SimpleNamespace(
        command=SendCommand.Request.START_INFERENCE,
        task_info=task_info,
    )


def test_stale_resume_discards_old_client_and_enters_fresh_load(monkeypatch):
    _FreshClient.instances = []
    _DeferredThread.instances = []
    old_client = _CachedClient()
    node = _bare_node(old_client)
    monkeypatch.setattr(
        orchestrator_node_module, 'ContainerServiceClient', _FreshClient
    )
    monkeypatch.setattr(
        orchestrator_node_module.threading, 'Thread', _DeferredThread
    )
    response = SimpleNamespace(success=False, message='')

    result = node.user_interaction_callback(_start_request(), response)

    assert result.success
    assert 'inference loading' in result.message.lower()
    assert old_client.commands == [_FreshClient.CMD_RESUME]
    assert old_client._cancelled.is_set()
    assert old_client.disconnect_calls == 1
    assert len(_FreshClient.instances) == 1
    assert node.container_service_client is _FreshClient.instances[0]
    assert _FreshClient.instances[0].connected
    assert len(_DeferredThread.instances) == 1
    assert _DeferredThread.instances[0].started


def test_delayed_stale_resume_does_not_detach_replacement_client(monkeypatch):
    _FreshClient.instances = []
    replacement_client = _CachedClient()
    node = None

    def replace_client():
        with node._state_lock:
            node.container_service_client = replacement_client

    old_client = _CachedClient(on_resume=replace_client)
    node = _bare_node(old_client)
    monkeypatch.setattr(
        orchestrator_node_module, 'ContainerServiceClient', _FreshClient
    )
    response = SimpleNamespace(success=True, message='')

    result = node.user_interaction_callback(_start_request(), response)

    assert not result.success
    assert result.message == (
        'Inference session changed while RESUME was pending'
    )
    assert node.container_service_client is replacement_client
    assert old_client.disconnect_calls == 0
    assert replacement_client.disconnect_calls == 0
    assert not replacement_client._cancelled.is_set()
    assert _FreshClient.instances == []
