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

"""BT action for inference and policy-Docker lifecycle commands.

The four BT commands ride entirely on top of the SendCommand enums the
UI already uses (START_INFERENCE / STOP_INFERENCE / RESUME_INFERENCE /
FINISH). The only BT-specific bit is the LOAD command, which runs a
two-step sequence inside this node — START_INFERENCE (to leverage
orchestrator's "fresh load or skip if already loaded" logic) followed
by STOP_INFERENCE — so the policy ends up paused-in-memory and the BT
graph's next Resume node can kick it into INFERENCING. RESUME, STOP,
and CLEAR each run as a single-stage call.

For each stage the node polls /task/inference_status and only advances
once the phase the orchestrator publishes matches the expected target,
so a downstream BT node never starts running against a half-loaded or
mid-transition policy.

``target=DOCKER`` reuses the same node for START / STOP / RESTART of the
selected model backend. Docker operations go through the local supervisor API
on a worker thread, so even a first-use image pull or compose build never
blocks the BT executor tick. ``target=INFERENCE`` remains the default for
compatibility with trees saved before the target port existed. Its LOAD command
now ensures the selected backend is ready before the existing ROS stages.
"""

import json
import os
import socket
import threading
import time
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from interfaces.msg import InferenceStatus, TaskInfo
from interfaces.srv import InferenceCommand as InferenceCommandSrv
from interfaces.srv import SendCommand as SendCommandSrv

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

if TYPE_CHECKING:
    from rclpy.node import Node


# Per BT command, the ordered list of (SendCommandSrv enum, target phase,
# stage timeout, whether to attach task_info) the node executes. LOAD
# is the only multi-stage command — it runs START_INFERENCE first so
# the orchestrator's existing already-loaded-vs-fresh-load logic does
# the right thing, then immediately pauses the policy.
COMMAND_STAGES = {
    'LOAD': [
        {
            'command': SendCommandSrv.Request.START_INFERENCE,
            'target_phase': InferenceStatus.INFERENCING,
            'timeout': 600.0,
            'with_task_info': True,
        },
        {
            'command': SendCommandSrv.Request.STOP_INFERENCE,
            'target_phase': InferenceStatus.PAUSED,
            'timeout': 5.0,
            'with_task_info': False,
        },
    ],
    'RESUME': [
        {
            'command': SendCommandSrv.Request.RESUME_INFERENCE,
            'target_phase': InferenceStatus.INFERENCING,
            'timeout': 10.0,
            'with_task_info': True,
        },
    ],
    'STOP': [
        {
            'command': SendCommandSrv.Request.STOP_INFERENCE,
            'target_phase': InferenceStatus.PAUSED,
            'timeout': 5.0,
            'with_task_info': False,
        },
    ],
    'CLEAR': [
        {
            'command': SendCommandSrv.Request.FINISH,
            'target_phase': InferenceStatus.READY,
            'timeout': 10.0,
            'with_task_info': True,
        },
    ],
}

SERVICE_CALL_TIMEOUT_SEC = 30.0

ALLOWED_TARGETS = frozenset({'INFERENCE', 'DOCKER'})
DOCKER_COMMANDS = frozenset({'START', 'STOP', 'RESTART'})
ALLOWED_BACKENDS = frozenset({'groot', 'lerobot'})
BACKEND_SERVICES = {
    'groot': frozenset({'main-runtime', 'engine-process'}),
    'lerobot': frozenset({'main-runtime', 'engine-process'}),
}

_DEFAULT_SUPERVISOR_PORT = 7100
_DEFAULT_BACKEND_HTTP_TIMEOUT_SEC = 7260.0
_MIN_BACKEND_HTTP_TIMEOUT_SEC = 1.0
_MAX_BACKEND_HTTP_TIMEOUT_SEC = 21660.0
_DEFAULT_BACKEND_READY_TIMEOUT_SEC = 300.0
_MIN_BACKEND_READY_TIMEOUT_SEC = 1.0
_MAX_BACKEND_READY_TIMEOUT_SEC = 1800.0
_BACKEND_STATUS_POLL_SEC = 0.5
_BACKEND_SERVICE_POLL_SEC = 0.1
_BACKEND_STATUS_HTTP_TIMEOUT_SEC = 10.0
_COMPENSATING_STOP_TIMEOUT_SEC = 30.0

_BACKEND_GENERATION_LOCK = threading.Lock()
_BACKEND_GENERATIONS = {backend: 0 for backend in ALLOWED_BACKENDS}

MODEL_SERVICE_TYPES = {
    'groot': 'groot',
    'groot:n17': 'groot',
    'n17': 'groot',
    'n1.7': 'groot',
    'lerobot': 'lerobot',
    'lerobot:act': 'lerobot',
    'lerobot:diffusion': 'lerobot',
    'lerobot:smolvla': 'lerobot',
    'lerobot:xvla': 'lerobot',
    'lerobot:pi0': 'lerobot',
    'lerobot:pi05': 'lerobot',
    'lerobot:molmoact2': 'lerobot',
    'lerobot:vla_jepa': 'lerobot',
    'lerobot:fastwam': 'lerobot',
    'act': 'lerobot',
    'diffusion': 'lerobot',
    'smolvla': 'lerobot',
    'xvla': 'lerobot',
    'pi0': 'lerobot',
    'pi05': 'lerobot',
    'molmoact2': 'lerobot',
    'vla_jepa': 'lerobot',
    'fastwam': 'lerobot',
}


def _service_type_from_model(model: str) -> str:
    """Map UI model selections onto TaskInfo.service_type backends."""
    value = (model or '').strip().lower()
    if not value:
        return ''
    if value in MODEL_SERVICE_TYPES:
        return MODEL_SERVICE_TYPES[value]
    if ':' in value:
        return value.split(':', 1)[0].strip()
    return value


def _normalize_action_request_mode(value: str) -> str:
    mode = str(value or '').strip().lower()
    if mode == 'sync':
        return 'sync'
    return 'async'


def _bounded_env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite timeout from the environment and clamp it safely."""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if value != value or value in (float('inf'), float('-inf')):
        value = default
    return max(minimum, min(maximum, value))


def _supervisor_base_url() -> str:
    """Return the fixed-localhost supervisor URL with a validated port."""
    raw_port = os.environ.get(
        'CYCLO_SUPERVISOR_API_PORT',
        str(_DEFAULT_SUPERVISOR_PORT),
    )
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid CYCLO_SUPERVISOR_API_PORT: {raw_port!r}'
        ) from exc
    if port < 1 or port > 65535:
        raise ValueError(
            f'CYCLO_SUPERVISOR_API_PORT out of range: {port}'
        )
    return f'http://127.0.0.1:{port}'


def _http_error_message(body: str, fallback: str) -> str:
    """Extract FastAPI's detail/message field without trusting its shape."""
    try:
        payload = json.loads(body) if body else {}
    except (TypeError, ValueError):
        payload = {}
    if isinstance(payload, dict):
        return str(payload.get('detail') or payload.get('message') or fallback)
    return fallback


def _http_json_request(method: str, url: str, timeout: float) -> dict:
    """Perform one bounded supervisor request and return a JSON object."""
    request = Request(
        url,
        data=b'' if method.upper() == 'POST' else None,
        headers={'Accept': 'application/json'},
        method=method.upper(),
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode('utf-8', errors='replace')
            status = getattr(response, 'status', 200)
    except HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        fallback = f'HTTP {exc.code} from supervisor'
        raise RuntimeError(_http_error_message(body, fallback)) from exc
    except (URLError, socket.timeout, TimeoutError, OSError) as exc:
        reason = getattr(exc, 'reason', exc)
        raise RuntimeError(f'Supervisor request failed: {reason}') from exc

    if status < 200 or status >= 300:
        fallback = f'HTTP {status} from supervisor'
        raise RuntimeError(_http_error_message(body, fallback))
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise RuntimeError('Supervisor returned invalid JSON') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('Supervisor returned a non-object JSON response')
    return payload


def _claim_backend_generation(backend: str) -> int:
    """Claim ownership of the next lifecycle generation for a backend."""
    with _BACKEND_GENERATION_LOCK:
        generation = _BACKEND_GENERATIONS.get(backend, 0) + 1
        _BACKEND_GENERATIONS[backend] = generation
        return generation


def _is_latest_backend_generation(backend: str, generation: int) -> bool:
    """Return whether a job still owns backend compensation rights."""
    with _BACKEND_GENERATION_LOCK:
        return _BACKEND_GENERATIONS.get(backend, 0) == generation


class _BackendJob:
    """Per-execution worker state, isolated from later tree generations."""

    def __init__(self, operation: str, generation: int):
        self.operation = operation
        self.generation = generation
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.result = None
        self.message = ''
        self.thread = None

    def finish(self, result: bool, message: str):
        with self.lock:
            self.result = bool(result)
            self.message = str(message or '')

    def snapshot(self):
        with self.lock:
            return self.result, self.message


class _BackendJobCancelled(Exception):
    """Internal control flow for a tree reset while a worker is active."""

    pass


class SendCommand(BaseAction):
    """Drive inference or its policy Docker through lifecycle commands.

    INFERENCE keeps the legacy LOAD / RESUME / STOP / CLEAR behavior. DOCKER
    accepts START / STOP / RESTART and waits for the requested runtime state.
    """

    _STATE_INIT = 'init'
    _STATE_WAITING_BACKEND = 'waiting_backend'
    _STATE_BEGIN_STAGE = 'begin_stage'
    _STATE_WAITING_SERVICE = 'waiting_service'
    _STATE_CALLING = 'calling'
    _STATE_WAITING_PHASE = 'waiting_phase'
    _STATE_DONE = 'done'

    @classmethod
    def from_xml_params(cls, context, name: str, params: dict):
        """Build an action from XML ports and loader-owned dependencies."""
        task_instruction = params.get('task_instruction', '')
        if isinstance(task_instruction, list):
            task_instruction = ', '.join(task_instruction)
        target = params.get('target', 'INFERENCE')
        command = params.get('command', 'LOAD')
        command_str = str(command or '').strip().upper()
        inference_mode = (
            params.get('inference_mode', 'simulation')
            if command_str == 'LOAD'
            else ''
        )
        acceleration_mode = (
            params.get('acceleration_mode', 'pytorch')
            if command_str == 'LOAD'
            else ''
        )
        action_request_mode = (
            params.get('action_request_mode', 'async')
            if command_str == 'LOAD'
            else ''
        )
        action = cls(
            node=context.node,
            command=command,
            target=target,
            model=params.get('model', 'lerobot:act'),
            policy_path=params.get('policy_path', ''),
            task_instruction=task_instruction,
            inference_hz=params.get('inference_hz', 15),
            control_hz=params.get('control_hz', 100),
            chunk_align_window_s=params.get('chunk_align_window_s', 0.3),
            inference_mode=inference_mode,
            action_request_mode=action_request_mode,
            acceleration_mode=acceleration_mode,
            acceleration_engine_path=params.get(
                'acceleration_engine_path',
                '',
            ),
        )
        action.name = name
        return action

    def __init__(
        self,
        node: 'Node',
        command: str = 'LOAD',
        target: str = 'INFERENCE',
        # BT-facing name "model" matches the Inference UI's labeling.
        # Values may be legacy backend names ("groot" / "lerobot") or
        # Inference UI composite choices ("lerobot:act", "groot:n17").
        # Internally this is normalized to TaskInfo.service_type, which
        # orchestrator reads to pick the backend container.
        model: str = 'lerobot:act',
        policy_path: str = '',
        task_instruction: str = '',
        inference_hz: int = 15,
        control_hz: int = 100,
        chunk_align_window_s: float = 0.3,
        inference_mode: str = 'simulation',
        action_request_mode: str = 'async',
        acceleration_mode: str = 'pytorch',
        acceleration_engine_path: str = '',
        service_name: str = '/task/command',
    ):
        """Initialize an inference or Docker lifecycle action."""
        super().__init__(node, name='SendCommand')
        self.command_str = (command or '').strip().upper()
        self.target_str = (target or 'INFERENCE').strip().upper()
        self.model = model
        self.backend = _service_type_from_model(model)
        self.policy_path = policy_path
        self.task_instruction = task_instruction
        self.inference_hz = int(inference_hz) if inference_hz else 0
        self.control_hz = int(control_hz) if control_hz else 0
        self.chunk_align_window_s = (
            float(chunk_align_window_s) if chunk_align_window_s else 0.0
        )
        self.inference_mode = (
            (inference_mode or 'simulation').strip().lower()
            if self.command_str == 'LOAD'
            else ''
        )
        self.action_request_mode = (
            _normalize_action_request_mode(action_request_mode)
            if self.command_str == 'LOAD'
            else ''
        )
        self.acceleration_mode = (
            self._normalize_acceleration_mode(acceleration_mode)
            if self.command_str == 'LOAD'
            else ''
        )
        self.acceleration_engine_path = (
            str(acceleration_engine_path or '').strip()
            if self.command_str == 'LOAD'
            else ''
        )

        self._client = None
        self._backend_service_client = None
        self._backend_service_name = ''
        self._status_sub = None
        if self.target_str == 'INFERENCE':
            self._client = self.node.create_client(
                SendCommandSrv,
                service_name,
            )

        # supervisor readiness only proves that the container and its s6
        # processes are up. The main runtime advertises this service later,
        # after its engine handshake and Zenoh initialization complete. Keep
        # the Docker worker pending until ROS discovery sees the real command
        # endpoint, otherwise orchestrator's immediate RESUME/LOAD can race it.
        backend_operation = self._backend_operation()
        if (
            self.backend in ALLOWED_BACKENDS
            and backend_operation in {'START', 'RESTART'}
        ):
            self._backend_service_name = (
                f'/{self.backend}/inference_command'
            )
            self._backend_service_client = self.node.create_client(
                InferenceCommandSrv,
                self._backend_service_name,
            )

        self._latest_phase = None
        self._latest_error = ''
        self._phase_lock = threading.Lock()
        # Subscribe up front so phase transitions that land between the
        # srv response and the phase-wait state aren't missed.
        if self.target_str == 'INFERENCE':
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            self._status_sub = self.node.create_subscription(
                InferenceStatus,
                '/task/inference_status',
                self._status_callback,
                qos,
            )

        self._state = self._STATE_INIT
        self._stages = (
            COMMAND_STAGES.get(self.command_str, [])
            if self.target_str == 'INFERENCE'
            else []
        )
        self._stage_idx = 0
        self._future = None
        self._result = None
        self._service_wait_started = None
        self._phase_deadline = None
        self._backend_job = None

    def _status_callback(self, msg: InferenceStatus):
        with self._phase_lock:
            self._latest_phase = msg.inference_phase
            self._latest_error = getattr(msg, 'error', '')

    def _reset_phase_cache(self):
        with self._phase_lock:
            self._latest_phase = None
            self._latest_error = ''

    @property
    def _stage(self):
        return self._stages[self._stage_idx]

    def _configuration_error(self) -> str:
        if self.target_str not in ALLOWED_TARGETS:
            return (
                f'Unknown target: {self.target_str}. '
                f'Expected one of {sorted(ALLOWED_TARGETS)}'
            )
        if self.target_str == 'DOCKER':
            if self.command_str not in DOCKER_COMMANDS:
                return (
                    f'Unknown Docker command: {self.command_str}. '
                    f'Expected one of {sorted(DOCKER_COMMANDS)}'
                )
            if self.backend not in ALLOWED_BACKENDS:
                return (
                    f'Unsupported Docker backend from model {self.model!r}: '
                    f'{self.backend or "<empty>"}'
                )
            return ''
        if self.command_str not in COMMAND_STAGES:
            return f'Unknown inference command: {self.command_str}'
        if (
            self.command_str == 'LOAD'
            and self.backend not in ALLOWED_BACKENDS
        ):
            return (
                f'Unsupported inference backend from model {self.model!r}: '
                f'{self.backend or "<empty>"}'
            )
        return ''

    def _backend_operation(self):
        if self.target_str == 'DOCKER':
            return self.command_str
        if self.target_str == 'INFERENCE' and self.command_str == 'LOAD':
            return 'START'
        return None

    @staticmethod
    def _backend_reached_target(
        backend: str,
        operation: str,
        status: dict,
    ) -> bool:
        state = str(status.get('container_state') or '').strip().lower()
        if operation == 'STOP':
            return state in {'exited', 'not_created'}
        if state != 'running':
            return False

        required = BACKEND_SERVICES[backend]
        service_states = {
            str(service.get('name') or ''): str(
                service.get('state') or ''
            ).lower()
            for service in status.get('services', [])
            if isinstance(service, dict)
        }
        return all(service_states.get(name) == 'up' for name in required)

    @staticmethod
    def _supervisor_action_url(
        base_url: str,
        backend: str,
        operation: str,
    ) -> str:
        url = f'{base_url}/backends/{backend}/{operation.lower()}'
        if operation in {'START', 'RESTART'}:
            url += '?auto_provision=true'
        return url

    def _compensating_stop(self, base_url: str, job: _BackendJob) -> str:
        """Best-effort STOP after reset races with START/RESTART completion."""
        stop_url = f'{base_url}/backends/{self.backend}/stop'
        try:
            response = _http_json_request(
                'POST',
                stop_url,
                _COMPENSATING_STOP_TIMEOUT_SEC,
            )
            if response.get('ok') is False:
                detail = response.get('message') or 'supervisor rejected STOP'
                raise RuntimeError(str(detail))
            self.log_info(
                f'Compensating Docker STOP completed for {self.backend}'
            )
            return 'cancelled; compensating STOP completed'
        # A worker must always publish a terminal result, including unexpected
        # transport and parsing failures.
        except Exception as exc:
            self.log_error(
                f'Compensating Docker STOP failed for {self.backend}: {exc}'
            )
            return f'cancelled; compensating STOP failed: {exc}'

    def _run_backend_job(self, job: _BackendJob):
        operation = job.operation
        lifecycle_may_have_started = False
        result = False
        message = ''
        base_url = ''

        try:
            if job.cancel_event.is_set():
                raise _BackendJobCancelled()

            base_url = _supervisor_base_url()
            http_timeout = _bounded_env_float(
                'CYCLO_BT_BACKEND_HTTP_TIMEOUT_SEC',
                _DEFAULT_BACKEND_HTTP_TIMEOUT_SEC,
                _MIN_BACKEND_HTTP_TIMEOUT_SEC,
                _MAX_BACKEND_HTTP_TIMEOUT_SEC,
            )
            ready_timeout = _bounded_env_float(
                'CYCLO_BT_BACKEND_READY_TIMEOUT_SEC',
                _DEFAULT_BACKEND_READY_TIMEOUT_SEC,
                _MIN_BACKEND_READY_TIMEOUT_SEC,
                _MAX_BACKEND_READY_TIMEOUT_SEC,
            )

            action_url = self._supervisor_action_url(
                base_url,
                self.backend,
                operation,
            )
            lifecycle_may_have_started = operation in {'START', 'RESTART'}
            response = _http_json_request('POST', action_url, http_timeout)
            if response.get('ok') is False:
                raise RuntimeError(
                    str(
                        response.get('message')
                        or 'supervisor rejected action'
                    )
                )
            if job.cancel_event.is_set():
                raise _BackendJobCancelled()

            deadline = time.monotonic() + ready_timeout
            status_url = f'{base_url}/backends/{self.backend}/status'
            latest_state = 'not reported'
            supervisor_state = latest_state
            supervisor_ready = False
            while True:
                if job.cancel_event.is_set():
                    raise _BackendJobCancelled()
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f'Docker {operation} readiness timed out '
                        f'(last state: {latest_state})'
                    )

                if not supervisor_ready:
                    status = _http_json_request(
                        'GET',
                        status_url,
                        min(_BACKEND_STATUS_HTTP_TIMEOUT_SEC, ready_timeout),
                    )
                    latest_state = str(
                        status.get('raw_state')
                        or status.get('container_state')
                        or 'unknown'
                    )
                    supervisor_state = latest_state
                    supervisor_ready = self._backend_reached_target(
                        self.backend,
                        operation,
                        status,
                    )

                if supervisor_ready and operation in {'START', 'RESTART'}:
                    if self._backend_service_client is None:
                        raise RuntimeError(
                            'ROS inference service client was not initialized '
                            f'for {self.backend}'
                        )
                    try:
                        service_ready = bool(
                            self._backend_service_client.service_is_ready()
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            'ROS inference service readiness check failed for '
                            f'{self._backend_service_name}: {exc}'
                        ) from exc
                    if not service_ready:
                        latest_state = (
                            f'{supervisor_state}; waiting for ROS service '
                            f'{self._backend_service_name}'
                        )
                        job.cancel_event.wait(_BACKEND_SERVICE_POLL_SEC)
                        continue

                if supervisor_ready:
                    result = True
                    message = (
                        response.get('message')
                        or f'{self.backend} Docker {operation} completed'
                    )
                    break
                job.cancel_event.wait(_BACKEND_STATUS_POLL_SEC)
        except _BackendJobCancelled:
            message = 'Docker operation cancelled by tree reset'
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
        finally:
            should_compensate = (
                job.cancel_event.is_set()
                and lifecycle_may_have_started
                and base_url
                and _is_latest_backend_generation(
                    self.backend,
                    job.generation,
                )
            )
            if should_compensate:
                result = False
                message = self._compensating_stop(base_url, job)
            elif job.cancel_event.is_set() and lifecycle_may_have_started:
                # A newer lifecycle action owns this backend. Stopping here
                # would tear down the new generation that replaced this job.
                result = False
                message = (
                    'cancelled; compensating STOP skipped because a newer '
                    'backend generation is active'
                )
            job.finish(result, message)

    def _start_backend_job(self, operation: str):
        job = _BackendJob(
            operation,
            _claim_backend_generation(self.backend),
        )
        thread = threading.Thread(
            target=self._run_backend_job,
            args=(job,),
            daemon=True,
            name=f'bt-{self.backend}-{operation.lower()}',
        )
        job.thread = thread
        self._backend_job = job
        thread.start()

    def tick(self) -> NodeStatus:
        """Advance the non-blocking backend or inference state machine."""
        if self._state == self._STATE_INIT:
            configuration_error = self._configuration_error()
            if configuration_error:
                self.log_error(configuration_error)
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            if self.target_str == 'DOCKER':
                self.log_info(
                    'SendCommand started '
                    f'(target=DOCKER, command={self.command_str}, '
                    f'backend={self.backend})'
                )
            elif self.command_str == 'LOAD':
                self.log_info(
                    'SendCommand started '
                    f'(target=INFERENCE, command={self.command_str}, '
                    f'model={self.model}, '
                    f'inference_mode={self.inference_mode}, '
                    f'acceleration_mode={self.acceleration_mode}, '
                    f'publish_to_robot={self.inference_mode == "robot"})'
                )
                if self.inference_mode == 'simulation':
                    self.log_warn(
                        'simulation mode selected; inference previews may '
                        'update, '
                        'but robot command topics will not be published'
                    )
            else:
                self.log_info(
                    'SendCommand started '
                    f'(target=INFERENCE, command={self.command_str}, '
                    f'model={self.model}, '
                    'publish_to_robot=loaded)'
                )

            backend_operation = self._backend_operation()
            if backend_operation:
                self._start_backend_job(backend_operation)
                self._state = self._STATE_WAITING_BACKEND
                return NodeStatus.RUNNING

            self._state = self._STATE_BEGIN_STAGE
            return NodeStatus.RUNNING

        if self._state == self._STATE_WAITING_BACKEND:
            result, message = self._backend_job.snapshot()
            if result is None:
                return NodeStatus.RUNNING

            self._backend_job = None
            if not result:
                self.log_error(
                    f'{self.backend} Docker {self._backend_operation()} '
                    f'failed: {message}'
                )
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            self.log_info(message)
            if self.target_str == 'DOCKER':
                self._state = self._STATE_DONE
                self._result = True
                return NodeStatus.SUCCESS

            # INFERENCE LOAD only: Docker is ready, so enter the unchanged ROS
            # service + inference phase sequence.
            self._state = self._STATE_BEGIN_STAGE
            return NodeStatus.RUNNING

        if self._state == self._STATE_BEGIN_STAGE:
            if self._stage_idx >= len(self._stages):
                self._state = self._STATE_DONE
                self._result = True
                return NodeStatus.SUCCESS
            # Clear latched phase so we don't match a transition from a
            # previous stage (LOAD's stage 1 entered while phase is still
            # INFERENCING from stage 0).
            self._reset_phase_cache()
            self._service_wait_started = time.monotonic()
            self._state = self._STATE_WAITING_SERVICE
            return NodeStatus.RUNNING

        if self._state == self._STATE_WAITING_SERVICE:
            if not self._client.service_is_ready():
                if (time.monotonic() - self._service_wait_started
                        > SERVICE_CALL_TIMEOUT_SEC):
                    self.log_error('SendCommand service not available')
                    self._state = self._STATE_DONE
                    self._result = False
                    return NodeStatus.FAILURE
                return NodeStatus.RUNNING

            req = SendCommandSrv.Request()
            req.command = self._stage['command']
            if self._stage['with_task_info']:
                req.task_info = self._build_task_info()
            self._future = self._client.call_async(req)
            self._service_wait_started = time.monotonic()
            self._state = self._STATE_CALLING
            return NodeStatus.RUNNING

        if self._state == self._STATE_CALLING:
            if not self._future.done():
                if (time.monotonic() - self._service_wait_started
                        > SERVICE_CALL_TIMEOUT_SEC):
                    self.log_error('Service call timed out')
                    self._future.cancel()
                    self._state = self._STATE_DONE
                    self._result = False
                    return NodeStatus.FAILURE
                return NodeStatus.RUNNING

            response = self._future.result()
            if response is None or not response.success:
                msg = response.message if response else 'No response'
                self.log_error(
                    f'SendCommand stage {self._stage_idx} failed: {msg}'
                )
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            self.log_info(
                f'SendCommand {self.command_str} stage '
                f'{self._stage_idx} ok: {response.message}'
            )
            self._phase_deadline = (
                time.monotonic() + self._stage['timeout']
            )
            self._state = self._STATE_WAITING_PHASE
            return NodeStatus.RUNNING

        if self._state == self._STATE_WAITING_PHASE:
            if time.monotonic() > self._phase_deadline:
                self.log_error(
                    f'{self.command_str} stage {self._stage_idx} phase '
                    f'wait timed out (target={self._stage["target_phase"]})'
                )
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            with self._phase_lock:
                phase = self._latest_phase
                error = self._latest_error

            if phase is None:
                return NodeStatus.RUNNING

            if phase == self._stage['target_phase']:
                self.log_info(
                    f'{self.command_str} stage {self._stage_idx} '
                    f'reached phase {phase}'
                )
                self._stage_idx += 1
                self._state = self._STATE_BEGIN_STAGE
                return NodeStatus.RUNNING

            # Orchestrator publishes READY + error string when an async
            # LOAD/START thread fails — surface that as the BT failure.
            if phase == InferenceStatus.READY and error:
                self.log_error(
                    f'{self.command_str} stage {self._stage_idx} '
                    f'failed during phase wait: {error}'
                )
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            return NodeStatus.RUNNING

        # _STATE_DONE
        return NodeStatus.SUCCESS if self._result else NodeStatus.FAILURE

    @staticmethod
    def _normalize_acceleration_mode(value: str) -> str:
        mode = str(value or '').strip().lower()
        if mode in {'', 'none', 'off', 'false', 'pytorch', 'eager'}:
            return 'pytorch'
        if mode in {'trt', 'tensorrt', 'tensorrt_dit', 'dit', 'dit_only'}:
            return 'tensorrt_dit'
        if mode in {
            'trt_full_pipeline',
            'tensorrt_full_pipeline',
            'full_pipeline',
        }:
            return 'tensorrt_full_pipeline'
        return mode

    def _build_task_info(self) -> TaskInfo:
        ti = TaskInfo()
        ti.task_type = 'inference'
        ti.policy_path = self.policy_path
        ti.service_type = _service_type_from_model(self.model)
        if self.command_str == 'LOAD' and hasattr(ti, 'inference_mode'):
            ti.inference_mode = self.inference_mode
        if self.command_str == 'LOAD' and hasattr(ti, 'action_request_mode'):
            ti.action_request_mode = self.action_request_mode
        ti.tags = (
            [f'inference_mode:{self.inference_mode}']
            if self.command_str == 'LOAD'
            else []
        )
        if self.command_str == 'LOAD' and hasattr(ti, 'acceleration_mode'):
            ti.acceleration_mode = self.acceleration_mode
        if (
            self.command_str == 'LOAD'
            and hasattr(ti, 'acceleration_engine_path')
        ):
            ti.acceleration_engine_path = self.acceleration_engine_path
        if self.control_hz:
            ti.control_hz = self.control_hz
        if self.inference_hz:
            ti.inference_hz = self.inference_hz
        if self.chunk_align_window_s:
            ti.chunk_align_window_s = self.chunk_align_window_s
        if self.task_instruction:
            if isinstance(self.task_instruction, list):
                ti.task_instruction = self.task_instruction
            else:
                ti.task_instruction = [self.task_instruction]
        return ti

    def reset(self):
        """Cancel pending work without letting a late START escape cleanup."""
        super().reset()
        backend_job = self._backend_job
        if backend_job is not None:
            backend_result, _ = backend_job.snapshot()
            if backend_result is None:
                # Do not join here: a first-use pull/build can legitimately run
                # for minutes. The isolated job notices cancellation and, when
                # START/RESTART may already have succeeded, issues a
                # compensating STOP before publishing its terminal result.
                backend_job.cancel_event.set()
        self._backend_job = None
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._future = None
        self._state = self._STATE_INIT
        self._stage_idx = 0
        self._result = None
        self._service_wait_started = None
        self._phase_deadline = None
        self._reset_phase_cache()


SendCommandAction = SendCommand
