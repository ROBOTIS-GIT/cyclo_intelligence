#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Seongwoo Kim

import sys
import threading
import time
import types

import pytest


class _QoSProfile:
    def __init__(self, *args, **kwargs):
        pass


class _ReliabilityPolicy:
    RELIABLE = object()


def _install_ros_stubs():
    rclpy_mod = types.ModuleType('rclpy')
    qos_mod = types.ModuleType('rclpy.qos')
    qos_mod.QoSProfile = _QoSProfile
    qos_mod.ReliabilityPolicy = _ReliabilityPolicy

    sensor_msgs_mod = types.ModuleType('sensor_msgs')
    sensor_msgs_msg_mod = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg_mod.JointState = object

    trajectory_msgs_mod = types.ModuleType('trajectory_msgs')
    trajectory_msgs_msg_mod = types.ModuleType('trajectory_msgs.msg')
    trajectory_msgs_msg_mod.JointTrajectory = object
    trajectory_msgs_msg_mod.JointTrajectoryPoint = object

    geometry_msgs_mod = types.ModuleType('geometry_msgs')
    geometry_msgs_msg_mod = types.ModuleType('geometry_msgs.msg')
    geometry_msgs_msg_mod.Twist = object

    nav_msgs_mod = types.ModuleType('nav_msgs')
    nav_msgs_msg_mod = types.ModuleType('nav_msgs.msg')
    nav_msgs_msg_mod.Odometry = object

    interfaces_mod = types.ModuleType('interfaces')
    interfaces_msg_mod = types.ModuleType('interfaces.msg')
    interfaces_srv_mod = types.ModuleType('interfaces.srv')

    class _InferenceStatus:
        READY = 0
        INFERENCING = 2
        PAUSED = 3

    class _TaskInfo:
        inference_mode = ''
        action_request_mode = ''
        acceleration_mode = ''
        acceleration_engine_path = ''

    class _SendCommandRequest:
        START_INFERENCE = 1
        STOP_INFERENCE = 2
        RESUME_INFERENCE = 3
        FINISH = 4

    class _SendCommand:
        Request = _SendCommandRequest

    class _InferenceCommand:
        pass

    interfaces_msg_mod.InferenceStatus = _InferenceStatus
    interfaces_msg_mod.TaskInfo = _TaskInfo
    interfaces_srv_mod.InferenceCommand = _InferenceCommand
    interfaces_srv_mod.SendCommand = _SendCommand

    sys.modules.setdefault('rclpy', rclpy_mod)
    sys.modules.setdefault('rclpy.qos', qos_mod)
    sys.modules.setdefault('sensor_msgs', sensor_msgs_mod)
    sys.modules.setdefault('sensor_msgs.msg', sensor_msgs_msg_mod)
    sys.modules.setdefault('trajectory_msgs', trajectory_msgs_mod)
    sys.modules.setdefault('trajectory_msgs.msg', trajectory_msgs_msg_mod)
    sys.modules.setdefault('geometry_msgs', geometry_msgs_mod)
    sys.modules.setdefault('geometry_msgs.msg', geometry_msgs_msg_mod)
    sys.modules.setdefault('nav_msgs', nav_msgs_mod)
    sys.modules.setdefault('nav_msgs.msg', nav_msgs_msg_mod)
    sys.modules.setdefault('interfaces', interfaces_mod)
    sys.modules.setdefault('interfaces.msg', interfaces_msg_mod)
    sys.modules.setdefault('interfaces.srv', interfaces_srv_mod)


def _real_ros_modules_available():
    """Load real ROS modules when the test process has a sourced workspace."""
    try:
        from rclpy.qos import QoSProfile  # noqa: F401
        from rclpy.qos import ReliabilityPolicy  # noqa: F401
        from interfaces.msg import InferenceStatus  # noqa: F401
        from interfaces.msg import TaskInfo  # noqa: F401
        from interfaces.srv import InferenceCommand  # noqa: F401
        from interfaces.srv import SendCommand  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


if not _real_ros_modules_available():
    _install_ros_stubs()

from orchestrator.bt.actions import send_command as send_command_module  # noqa: E402, E501
from orchestrator.bt.actions.arm_state_gate import ArmStateGate  # noqa: E402
from orchestrator.bt.actions.joint_control import (  # noqa: E402
    _coerce_positions,
)
from orchestrator.bt.actions.send_command import SendCommand  # noqa: E402
from orchestrator.bt.bt_core import NodeStatus  # noqa: E402
from orchestrator.bt.node_registry import (  # noqa: E402
    _annotation_to_port_type,
)
from orchestrator.bt.node_registry import build_registry  # noqa: E402
from orchestrator.bt.node_registry import catalog_payload  # noqa: E402


class _DummyServiceClient:
    def service_is_ready(self):
        return True


class _DummyNode:
    def create_client(self, *args, **kwargs):
        return _DummyServiceClient()

    def create_subscription(self, *args, **kwargs):
        return object()

    def get_logger(self):
        class _Logger:
            def info(self, *args, **kwargs):
                pass

            def warn(self, *args, **kwargs):
                pass

            def error(self, *args, **kwargs):
                pass

        return _Logger()


class _ControlledReadinessClient:
    def __init__(self, ready_event):
        self.ready_event = ready_event
        self.checked = threading.Event()

    def service_is_ready(self):
        self.checked.set()
        return self.ready_event.is_set()


class _ControlledReadinessNode(_DummyNode):
    def __init__(self, ready_event):
        self.ready_event = ready_event
        self.backend_client = None
        self.backend_service_name = ''

    def create_client(self, service_type, service_name, *args, **kwargs):
        if service_name.endswith('/inference_command'):
            self.backend_service_name = service_name
            self.backend_client = _ControlledReadinessClient(
                self.ready_event
            )
            return self.backend_client
        return super().create_client(
            service_type,
            service_name,
            *args,
            **kwargs,
        )


def _make_wait_action(**overrides):
    kwargs = {
        'node': _DummyNode(),
        'left_target_joints': 'arm_l_joint1, arm_l_joint2',
        'left_target_positions': '0.1, -0.2',
        'right_target_joints': 'arm_r_joint1, arm_r_joint2',
        'right_target_positions': '0.3, -0.4',
    }
    kwargs.update(overrides)
    return ArmStateGate(**kwargs)


def _joint_state(
    left=(0.1, -0.2),
    right=(0.3, -0.4),
    gripper_l=0.0,
    gripper_r=0.0,
):
    return types.SimpleNamespace(
        name=[
            'arm_l_joint1',
            'arm_l_joint2',
            'arm_r_joint1',
            'arm_r_joint2',
            'gripper_l_joint1',
            'gripper_r_joint1',
        ],
        position=[
            left[0],
            left[1],
            right[0],
            right[1],
            gripper_l,
            gripper_r,
        ],
    )


def _tick_until_terminal(action, timeout=2.0):
    deadline = time.monotonic() + timeout
    status = NodeStatus.RUNNING
    while time.monotonic() < deadline:
        status = action.tick()
        if status != NodeStatus.RUNNING:
            return status
        time.sleep(0.002)
    raise AssertionError(
        f'action did not finish (state={action._state}, status={status})'
    )


def _ready_backend_status(backend='groot'):
    return {
        'name': backend,
        'image_pulled': True,
        'container_state': 'running',
        'raw_state': 'running',
        'services': [
            {'name': 'main-runtime', 'state': 'up'},
            {'name': 'engine-process', 'state': 'up'},
        ],
    }


def test_coerce_positions_accepts_comma_separated_strings():
    assert _coerce_positions('0.0, 1.5, -2') == [0.0, 1.5, -2.0]
    assert _coerce_positions('') == []


def test_stringified_annotations_map_to_port_types():
    assert _annotation_to_port_type('bool', None) == 'bool'
    assert _annotation_to_port_type('list[float]', None) == 'number'
    assert _annotation_to_port_type('Optional[int]', None) == 'number'
    assert _annotation_to_port_type('str', None) == 'string'


def test_resume_send_command_legacy_simulation_ignores_mode():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'ResumeInference',
        {'command': 'RESUME', 'inference_mode': 'simulation'},
    )

    assert action.inference_mode == ''


def test_load_send_command_sets_acceleration_mode():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'LoadInference',
        {
            'command': 'LOAD',
            'model': 'groot:n17',
            'acceleration_mode': 'tensorrt',
            'acceleration_engine_path': 'custom.trt',
        },
    )
    task_info = action._build_task_info()

    assert action.acceleration_mode == 'tensorrt_dit'
    assert task_info.acceleration_mode == 'tensorrt_dit'
    assert task_info.acceleration_engine_path == 'custom.trt'


def test_load_send_command_sets_action_request_mode():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'LoadInference',
        {
            'command': 'LOAD',
            'action_request_mode': 'sync',
        },
    )
    task_info = action._build_task_info()

    assert action.action_request_mode == 'sync'
    assert task_info.action_request_mode == 'sync'


def test_send_command_legacy_xml_defaults_to_inference_target():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'LegacyStopInference',
        {'command': 'STOP'},
    )

    assert action.target_str == 'INFERENCE'
    assert action.command_str == 'STOP'


def test_send_command_catalog_exposes_default_target_port():
    registry = build_registry()
    catalog = {entry['tag']: entry for entry in catalog_payload(registry)}
    ports = {
        port['name']: port
        for port in catalog['SendCommand']['ports']
    }

    assert ports['target'] == {
        'name': 'target',
        'type': 'string',
        'default': 'INFERENCE',
    }


def test_docker_start_waits_for_all_declared_services(monkeypatch):
    calls = []
    status_calls = 0

    def fake_http(method, url, timeout):
        nonlocal status_calls
        calls.append((method, url, timeout))
        if method == 'POST':
            return {'ok': True, 'message': 'start accepted'}
        status_calls += 1
        status = _ready_backend_status()
        if status_calls == 1:
            status['services'] = [
                {'name': 'main-runtime', 'state': 'up'},
                {'name': 'engine-process', 'state': 'down'},
            ]
        return status

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    monkeypatch.setattr(send_command_module, '_BACKEND_STATUS_POLL_SEC', 0.001)
    action = SendCommand(
        _DummyNode(),
        target='DOCKER',
        command='START',
        model='groot:n17',
    )

    assert _tick_until_terminal(action) == NodeStatus.SUCCESS
    assert status_calls == 2
    assert calls[0][0:2] == (
        'POST',
        'http://127.0.0.1:7100/backends/groot/start?auto_provision=true',
    )
    assert calls[-1][0:2] == (
        'GET',
        'http://127.0.0.1:7100/backends/groot/status',
    )


def test_docker_start_waits_for_ros_inference_service(monkeypatch):
    service_ready = threading.Event()
    node = _ControlledReadinessNode(service_ready)

    def fake_http(method, url, timeout):
        if method == 'POST':
            return {'ok': True, 'message': 'start accepted'}
        return _ready_backend_status()

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    monkeypatch.setattr(
        send_command_module,
        '_BACKEND_SERVICE_POLL_SEC',
        0.001,
    )
    action = SendCommand(
        node,
        target='DOCKER',
        command='START',
        model='groot:n17',
    )

    assert action.tick() == NodeStatus.RUNNING
    assert node.backend_client.checked.wait(2.0)
    assert action._backend_job.snapshot()[0] is None
    assert action.tick() == NodeStatus.RUNNING
    assert node.backend_service_name == '/groot/inference_command'

    service_ready.set()
    assert _tick_until_terminal(action) == NodeStatus.SUCCESS


def test_docker_stop_waits_until_container_is_stopped(monkeypatch):
    calls = []

    def fake_http(method, url, timeout):
        calls.append((method, url, timeout))
        if method == 'POST':
            return {'ok': True, 'message': 'stop accepted'}
        return {
            'name': 'lerobot',
            'container_state': 'not_created',
            'services': [],
        }

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    action = SendCommand(
        _DummyNode(),
        target='DOCKER',
        command='STOP',
        model='lerobot:act',
    )

    assert _tick_until_terminal(action) == NodeStatus.SUCCESS
    assert calls[0][0:2] == (
        'POST',
        'http://127.0.0.1:7100/backends/lerobot/stop',
    )


def test_inference_load_prepares_backend_before_ros_stages(monkeypatch):
    calls = []

    def fake_http(method, url, timeout):
        calls.append((method, url, timeout))
        if method == 'POST':
            return {'ok': True, 'message': 'ready'}
        return _ready_backend_status('lerobot')

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    action = SendCommand(
        _DummyNode(),
        command='LOAD',
        model='lerobot:act',
    )

    assert action.tick() == NodeStatus.RUNNING
    deadline = time.monotonic() + 2.0
    while action._state == action._STATE_WAITING_BACKEND:
        assert time.monotonic() < deadline
        action.tick()
        time.sleep(0.002)

    assert action.target_str == 'INFERENCE'
    assert action._state == action._STATE_BEGIN_STAGE
    assert calls[0][1].endswith(
        '/backends/lerobot/start?auto_provision=true'
    )
    action.reset()


def test_inference_load_waits_for_ros_service_before_ros_stages(monkeypatch):
    service_ready = threading.Event()
    node = _ControlledReadinessNode(service_ready)

    def fake_http(method, url, timeout):
        if method == 'POST':
            return {'ok': True, 'message': 'ready'}
        return _ready_backend_status('lerobot')

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    monkeypatch.setattr(
        send_command_module,
        '_BACKEND_SERVICE_POLL_SEC',
        0.001,
    )
    action = SendCommand(
        node,
        command='LOAD',
        model='lerobot:act',
    )

    assert action.tick() == NodeStatus.RUNNING
    assert node.backend_client.checked.wait(2.0)
    assert action._state == action._STATE_WAITING_BACKEND

    service_ready.set()
    deadline = time.monotonic() + 2.0
    while action._state == action._STATE_WAITING_BACKEND:
        assert time.monotonic() < deadline
        action.tick()
        time.sleep(0.002)

    assert action._state == action._STATE_BEGIN_STAGE
    action.reset()


def test_backend_service_readiness_timeout_fails_job(monkeypatch):
    service_ready = threading.Event()
    node = _ControlledReadinessNode(service_ready)

    def fake_http(method, url, timeout):
        if method == 'POST':
            return {'ok': True, 'message': 'start accepted'}
        return _ready_backend_status()

    original_bounded = send_command_module._bounded_env_float

    def bounded_timeout(name, default, minimum, maximum):
        if name == 'CYCLO_BT_BACKEND_READY_TIMEOUT_SEC':
            return 0.01
        return original_bounded(name, default, minimum, maximum)

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    monkeypatch.setattr(
        send_command_module,
        '_bounded_env_float',
        bounded_timeout,
    )
    monkeypatch.setattr(
        send_command_module,
        '_BACKEND_SERVICE_POLL_SEC',
        0.001,
    )
    action = SendCommand(
        node,
        target='DOCKER',
        command='START',
        model='groot:n17',
    )

    assert action.tick() == NodeStatus.RUNNING
    job = action._backend_job
    job.thread.join(timeout=2.0)

    assert not job.thread.is_alive()
    assert job.snapshot()[0] is False
    assert '/groot/inference_command' in job.snapshot()[1]
    assert action.tick() == NodeStatus.FAILURE


@pytest.mark.parametrize(
    ('target', 'command', 'model'),
    [
        ('DOCKER', 'LOAD', 'groot:n17'),
        ('DOCKER', 'START', 'unknown:model'),
        ('UNKNOWN', 'START', 'groot:n17'),
        ('INFERENCE', 'LOAD', 'unknown:model'),
    ],
)
def test_send_command_rejects_unapproved_backend_actions(
    monkeypatch,
    target,
    command,
    model,
):
    def http(*args, **kwargs):
        pytest.fail('HTTP must not be called')

    monkeypatch.setattr(send_command_module, '_http_json_request', http)
    action = SendCommand(
        _DummyNode(),
        target=target,
        command=command,
        model=model,
    )

    assert action.tick() == NodeStatus.FAILURE


def test_docker_http_error_fails_action(monkeypatch):
    def fail_http(method, url, timeout):
        raise RuntimeError('registry authentication failed')

    monkeypatch.setattr(send_command_module, '_http_json_request', fail_http)
    action = SendCommand(
        _DummyNode(),
        target='DOCKER',
        command='RESTART',
        model='groot',
    )

    assert _tick_until_terminal(action) == NodeStatus.FAILURE


def test_backend_http_timeout_environment_is_bounded(monkeypatch):
    monkeypatch.setenv('CYCLO_BT_BACKEND_HTTP_TIMEOUT_SEC', '999999')
    monkeypatch.setenv('CYCLO_BT_BACKEND_READY_TIMEOUT_SEC', '-12')

    assert send_command_module._bounded_env_float(
        'CYCLO_BT_BACKEND_HTTP_TIMEOUT_SEC',
        send_command_module._DEFAULT_BACKEND_HTTP_TIMEOUT_SEC,
        send_command_module._MIN_BACKEND_HTTP_TIMEOUT_SEC,
        send_command_module._MAX_BACKEND_HTTP_TIMEOUT_SEC,
    ) == send_command_module._MAX_BACKEND_HTTP_TIMEOUT_SEC
    assert send_command_module._bounded_env_float(
        'CYCLO_BT_BACKEND_READY_TIMEOUT_SEC',
        send_command_module._DEFAULT_BACKEND_READY_TIMEOUT_SEC,
        send_command_module._MIN_BACKEND_READY_TIMEOUT_SEC,
        send_command_module._MAX_BACKEND_READY_TIMEOUT_SEC,
    ) == send_command_module._MIN_BACKEND_READY_TIMEOUT_SEC


def test_reset_during_start_issues_compensating_stop(monkeypatch):
    start_entered = threading.Event()
    release_start = threading.Event()
    compensating_stop = threading.Event()
    calls = []

    def fake_http(method, url, timeout):
        calls.append((method, url, timeout))
        if '/start?' in url:
            start_entered.set()
            assert release_start.wait(2.0)
            return {'ok': True, 'message': 'started after reset'}
        if url.endswith('/stop'):
            compensating_stop.set()
            return {'ok': True, 'message': 'stopped'}
        return _ready_backend_status()

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    action = SendCommand(
        _DummyNode(),
        target='DOCKER',
        command='START',
        model='groot:n17',
    )

    assert action.tick() == NodeStatus.RUNNING
    assert start_entered.wait(2.0)
    job = action._backend_job
    action.reset()
    release_start.set()

    assert compensating_stop.wait(2.0)
    job.thread.join(timeout=2.0)
    assert not job.thread.is_alive()
    assert job.snapshot()[0] is False
    assert any(url.endswith('/backends/groot/stop') for _, url, _ in calls)


def test_reset_while_waiting_for_ros_service_compensates_stop(monkeypatch):
    service_ready = threading.Event()
    node = _ControlledReadinessNode(service_ready)
    compensating_stop = threading.Event()

    def fake_http(method, url, timeout):
        if url.endswith('/stop'):
            compensating_stop.set()
            return {'ok': True, 'message': 'stopped'}
        if method == 'POST':
            return {'ok': True, 'message': 'started'}
        return _ready_backend_status()

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    monkeypatch.setattr(
        send_command_module,
        '_BACKEND_SERVICE_POLL_SEC',
        0.001,
    )
    action = SendCommand(
        node,
        target='DOCKER',
        command='START',
        model='groot:n17',
    )

    assert action.tick() == NodeStatus.RUNNING
    assert node.backend_client.checked.wait(2.0)
    job = action._backend_job
    action.reset()

    assert compensating_stop.wait(2.0)
    job.thread.join(timeout=2.0)
    assert not job.thread.is_alive()
    assert job.snapshot()[0] is False


def test_old_reset_job_does_not_stop_newer_backend_generation(monkeypatch):
    old_start_entered = threading.Event()
    release_old_start = threading.Event()
    call_lock = threading.Lock()
    start_count = 0
    stop_count = 0

    def fake_http(method, url, timeout):
        nonlocal start_count, stop_count
        if '/start?' in url:
            with call_lock:
                start_count += 1
                current_start = start_count
            if current_start == 1:
                old_start_entered.set()
                assert release_old_start.wait(2.0)
            return {'ok': True, 'message': 'started'}
        if url.endswith('/stop'):
            with call_lock:
                stop_count += 1
            return {'ok': True, 'message': 'stopped'}
        return _ready_backend_status()

    monkeypatch.setattr(send_command_module, '_http_json_request', fake_http)
    old_action = SendCommand(
        _DummyNode(),
        target='DOCKER',
        command='START',
        model='groot:n17',
    )
    new_action = SendCommand(
        _DummyNode(),
        target='DOCKER',
        command='START',
        model='groot:n17',
    )

    assert old_action.tick() == NodeStatus.RUNNING
    assert old_start_entered.wait(2.0)
    old_job = old_action._backend_job
    old_action.reset()

    assert _tick_until_terminal(new_action) == NodeStatus.SUCCESS
    release_old_start.set()
    old_job.thread.join(timeout=2.0)

    assert not old_job.thread.is_alive()
    assert old_job.snapshot()[0] is False
    assert 'newer backend generation' in old_job.snapshot()[1]
    assert stop_count == 0


def test_arm_state_gate_succeeds_without_gripper_detection():
    action = _make_wait_action()
    action._joint_state_callback(_joint_state())

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_left_gripper_requires_closed_then_opened():
    action = _make_wait_action(detect_left_gripper=True)
    action._joint_state_callback(_joint_state(gripper_l=0.0))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=1.0))
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.0))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_right_gripper_detection():
    action = _make_wait_action(detect_right_gripper=True)
    action._joint_state_callback(_joint_state(gripper_r=1.0))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_r=0.0))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_both_grippers_must_open():
    action = _make_wait_action(
        detect_left_gripper=True,
        detect_right_gripper=True,
    )
    action._joint_state_callback(_joint_state(gripper_l=1.0, gripper_r=1.0))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.0, gripper_r=1.0))
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.0, gripper_r=0.0))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_uses_custom_gripper_thresholds():
    action = _make_wait_action(
        detect_left_gripper=True,
        gripper_closed_value=0.8,
        gripper_open_value=0.2,
        gripper_threshold=0.02,
    )
    action._joint_state_callback(_joint_state(gripper_l=0.79))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.23))
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.21))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_rejects_mismatched_targets():
    with pytest.raises(ValueError):
        _make_wait_action(
            left_target_joints='arm_l_joint1, arm_l_joint2',
            left_target_positions='0.1',
        )


def test_arm_state_gate_timeout_fails_when_unmet():
    action = _make_wait_action(timeout_sec=0.001)

    assert action.tick() == NodeStatus.RUNNING
    action._start_time -= 1.0
    assert action.tick() == NodeStatus.FAILURE


def test_arm_state_gate_catalog_entry_has_expected_ports():
    registry = build_registry()
    catalog = {entry['tag']: entry for entry in catalog_payload(registry)}
    ports = {
        port['name']: port
        for port in catalog['ArmStateGate']['ports']
    }

    assert catalog['ArmStateGate']['category'] == 'action'
    assert ports['left_target_joints']['type'] == 'string'
    assert ports['left_target_positions']['type'] == 'string'
    assert ports['joint_threshold']['type'] == 'number'
    assert ports['detect_left_gripper']['type'] == 'bool'
    assert ports['detect_right_gripper']['type'] == 'bool'
    assert ports['gripper_threshold']['type'] == 'number'
    assert ports['timeout_sec']['type'] == 'number'


def test_joint_control_catalog_exposes_per_joint_selection():
    registry = build_registry()
    catalog = {entry['tag']: entry for entry in catalog_payload(registry)}
    port_names = [port['name'] for port in catalog['JointControl']['ports']]

    # Selector renders above its positions field, so the catalog (ctor
    # order) must list each joint_names port before its positions port.
    assert port_names.index('left_joint_names') < port_names.index(
        'left_positions'
    )
    assert port_names.index('right_joint_names') < port_names.index(
        'right_positions'
    )

    ports = {port['name']: port for port in catalog['JointControl']['ports']}
    assert ports['left_joint_names']['type'] == 'string'
    assert ports['right_joint_names']['type'] == 'string'


def test_joint_control_from_xml_params_forwards_selected_joints():
    from orchestrator.bt.actions.joint_control import JointControl

    class _PublishingNode(_DummyNode):
        def create_publisher(self, *args, **kwargs):
            return types.SimpleNamespace(publish=lambda msg: None)

    context = types.SimpleNamespace(
        node=_PublishingNode(),
        topic_config={
            'topic_map': {
                'leader_arm_left': '/leader/arm_left',
                'leader_arm_right': '/leader/arm_right',
            },
            'topic_type_map': {
                'leader_arm_left': 'trajectory_msgs/msg/JointTrajectory',
                'leader_arm_right': 'trajectory_msgs/msg/JointTrajectory',
            },
            'joint_order': {
                'leader_arm_left': [
                    f'arm_l_joint{i}' for i in range(1, 9)
                ],
                'leader_arm_right': [
                    f'arm_r_joint{i}' for i in range(1, 9)
                ],
            },
        },
    )

    action = JointControl.from_xml_params(context, 'JC', {
        'enable_head': False,
        'enable_arms': True,
        'left_joint_names': 'arm_l_joint2, arm_l_joint4',
        'left_positions': '0.5, -0.5',
        'right_joint_names': 'arm_r_joint1',
        'right_positions': '1.0',
    })

    by_group = {ch['group']: ch for ch in action._channels}
    assert by_group['arm_left']['joint_names'] == [
        'arm_l_joint2', 'arm_l_joint4',
    ]
    assert by_group['arm_left']['positions'] == [0.5, -0.5]
    assert by_group['arm_right']['joint_names'] == ['arm_r_joint1']
    assert by_group['arm_right']['positions'] == [1.0]

    # Without joint_names the full robot-yaml joint order still applies.
    legacy = JointControl.from_xml_params(context, 'JC', {
        'enable_head': False,
        'enable_arms': True,
        'left_positions': '0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8',
        'right_positions': '0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8',
    })
    legacy_left = {
        ch['group']: ch for ch in legacy._channels
    }['arm_left']
    assert len(legacy_left['joint_names']) == 8


def test_load_send_command_sets_action_processing_timing():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'LoadInference',
        {
            'command': 'LOAD',
            'control_hz': '80',
            'inference_hz': '20',
            'chunk_align_window_s': '0.25',
        },
    )
    task_info = action._build_task_info()

    assert task_info.control_hz == 80
    assert task_info.inference_hz == 20
    assert task_info.chunk_align_window_s == 0.25
