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
from interfaces.srv import SendCommand
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

if TYPE_CHECKING:
    from rclpy.node import Node


# Per BT command, the ordered list of (SendCommand enum, target phase,
# stage timeout, whether to attach task_info) the node executes. LOAD
# is the only multi-stage command — it runs START_INFERENCE first so
# the orchestrator's existing already-loaded-vs-fresh-load logic does
# the right thing, then immediately pauses the policy.
COMMAND_STAGES = {
    'LOAD': [
        {
            'command': SendCommand.Request.START_INFERENCE,
            'target_phase': InferenceStatus.INFERENCING,
            'timeout': 600.0,
            'with_task_info': True,
        },
        {
            'command': SendCommand.Request.STOP_INFERENCE,
            'target_phase': InferenceStatus.PAUSED,
            'timeout': 5.0,
            'with_task_info': False,
        },
    ],
    'RESUME': [
        {
            'command': SendCommand.Request.RESUME_INFERENCE,
            'target_phase': InferenceStatus.INFERENCING,
            'timeout': 10.0,
            'with_task_info': True,
        },
    ],
    'STOP': [
        {
            'command': SendCommand.Request.STOP_INFERENCE,
            'target_phase': InferenceStatus.PAUSED,
            'timeout': 5.0,
            'with_task_info': False,
        },
    ],
    'CLEAR': [
        {
            'command': SendCommand.Request.FINISH,
            'target_phase': InferenceStatus.READY,
            'timeout': 10.0,
            'with_task_info': False,
        },
    ],
}

SERVICE_CALL_TIMEOUT_SEC = 30.0


class SendCommandAction(BaseAction):
    """Run one BT inference command (LOAD / RESUME / STOP / CLEAR)."""

    _STATE_INIT = 'init'
    _STATE_BEGIN_STAGE = 'begin_stage'
    _STATE_WAITING_SERVICE = 'waiting_service'
    _STATE_CALLING = 'calling'
    _STATE_WAITING_PHASE = 'waiting_phase'
    _STATE_DONE = 'done'

    def __init__(
        self,
        node: 'Node',
        command: str = 'LOAD',
        # BT-facing name "model" matches the Inference UI's labeling.
        # Internally this rides on TaskInfo.service_type, which the
        # orchestrator's _determine_service_prefix reads to pick the
        # backend container ("groot" / "lerobot" / ...).
        model: str = '',
        policy_path: str = '',
        task_instruction: str = '',
        inference_hz: int = 15,
        control_hz: int = 100,
        chunk_align_window_s: float = 0.3,
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

        self._client = self.node.create_client(SendCommand, service_name)

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
            self.log_info(f'SendCommand started (command={self.command_str})')
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

            req = SendCommand.Request()
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

    def _build_task_info(self) -> TaskInfo:
        ti = TaskInfo()
        ti.policy_path = self.policy_path
        ti.service_type = self.model
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
