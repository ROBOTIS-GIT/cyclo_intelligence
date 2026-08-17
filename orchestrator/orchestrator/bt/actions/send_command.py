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

"""BT action: Load / Resume / Stop / Clear inference via SendCommand srv.

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
"""

import threading
import time
from typing import TYPE_CHECKING

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from interfaces.msg import InferenceStatus, TaskInfo
from interfaces.srv import SendCommand as SendCommandSrv
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


class SendCommand(BaseAction):
    """Drive the orchestrator inference pipeline through lifecycle commands.

    LOAD starts and pauses a policy, RESUME starts ticking a loaded policy,
    STOP pauses without unloading, and CLEAR finishes and unloads.
    """

    _STATE_INIT = 'init'
    _STATE_BEGIN_STAGE = 'begin_stage'
    _STATE_WAITING_SERVICE = 'waiting_service'
    _STATE_CALLING = 'calling'
    _STATE_WAITING_PHASE = 'waiting_phase'
    _STATE_DONE = 'done'

    @classmethod
    def from_xml_params(cls, context, name: str, params: dict):
        task_instruction = params.get('task_instruction', '')
        if isinstance(task_instruction, list):
            task_instruction = ', '.join(task_instruction)
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
            model=params.get('model', 'lerobot:act'),
            policy_path=params.get('policy_path', ''),
            task_instruction=task_instruction,
            inference_hz=params.get('inference_hz', 15),
            control_hz=params.get('control_hz', 100),
            chunk_align_window_s=params.get('chunk_align_window_s', 0.3),
            inference_mode=inference_mode,
            action_request_mode=action_request_mode,
            acceleration_mode=acceleration_mode,
            acceleration_engine_path=params.get('acceleration_engine_path', ''),
        )
        action.name = name
        return action

    def __init__(
        self,
        node: 'Node',
        command: str = 'LOAD',
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
        super().__init__(node, name='SendCommand')
        self.command_str = (command or '').strip().upper()
        self.model = model
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

        self._client = self.node.create_client(SendCommandSrv, service_name)

        self._latest_phase = None
        self._latest_error = ''
        self._phase_lock = threading.Lock()
        # Subscribe up front so phase transitions that land between the
        # srv response and the phase-wait state aren't missed.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._status_sub = self.node.create_subscription(
            InferenceStatus,
            '/task/inference_status',
            self._status_callback,
            qos,
        )

        self._state = self._STATE_INIT
        self._stages = COMMAND_STAGES.get(self.command_str, [])
        self._stage_idx = 0
        self._future = None
        self._result = None
        self._service_wait_started = None
        self._phase_deadline = None

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

    def tick(self) -> NodeStatus:
        if self._state == self._STATE_INIT:
            if not self._stages:
                self.log_error(f'Unknown command: {self.command_str}')
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE
            if self.command_str == 'LOAD':
                self.log_info(
                    'SendCommand started '
                    f'(command={self.command_str}, model={self.model}, '
                    f'inference_mode={self.inference_mode}, '
                    f'acceleration_mode={self.acceleration_mode}, '
                    f'publish_to_robot={self.inference_mode == "robot"})'
                )
                if self.inference_mode == 'simulation':
                    self.log_warn(
                        'simulation mode selected; inference previews may update, '
                        'but robot command topics will not be published'
                    )
            else:
                self.log_info(
                    'SendCommand started '
                    f'(command={self.command_str}, model={self.model}, '
                    'publish_to_robot=loaded)'
                )
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
        if self.command_str == 'LOAD' and hasattr(ti, 'acceleration_engine_path'):
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
        super().reset()
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
