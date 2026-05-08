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

"""Action node for calling SendCommand service on orchestrator."""

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

COMMAND_MAP = {
    'START_INFERENCE': SendCommand.Request.START_INFERENCE,
    'STOP_INFERENCE': SendCommand.Request.STOP_INFERENCE,
    'RESUME_INFERENCE': SendCommand.Request.RESUME_INFERENCE,
    'START_RECORD': SendCommand.Request.START_RECORD,
    'STOP': SendCommand.Request.STOP,
    'START_INFERENCE_RECORD': SendCommand.Request.START_INFERENCE_RECORD,
    'STOP_INFERENCE_RECORD': SendCommand.Request.STOP_INFERENCE_RECORD,
}

SERVICE_CALL_TIMEOUT_SEC = 30.0
LOADING_TIMEOUT_SEC = 600.0
LOADING_POLL_INTERVAL_SEC = 0.5


class SendCommandAction(BaseAction):
    """Action to call SendCommand service on orchestrator."""

    # States for the tick state machine
    _STATE_INIT = 'init'
    _STATE_WAITING_SERVICE = 'waiting_service'
    _STATE_CALLING = 'calling'
    _STATE_WAITING_LOAD = 'waiting_load'
    _STATE_DONE = 'done'

    def __init__(
        self,
        node: 'Node',
        command: str = 'STOP_INFERENCE',
        policy_path: str = '',
        task_instruction: str = '',
        task_name: str = '',
        control_hz: int = 0,
        wait_until_ready: bool = False,
        service_name: str = '/task/command',
    ):
        """Initialize the SendCommand action."""
        super().__init__(node, name='SendCommand')
        self.command_str = command
        self.policy_path = policy_path
        self.task_instruction = task_instruction
        self.task_name = task_name
        self.control_hz = control_hz
        self.wait_until_ready = wait_until_ready

        self._client = self.node.create_client(SendCommand, service_name)

        self._latest_phase = None
        self._latest_error = ''
        self._phase_lock = threading.Lock()
        if self.wait_until_ready:
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._status_sub = self.node.create_subscription(
                InferenceStatus,
                '/task/inference_status',
                self._status_callback,
                qos,
            )
        else:
            self._status_sub = None

        self._state = self._STATE_INIT
        self._future = None
        self._result = None
        self._start_time = None

    def _status_callback(self, msg: InferenceStatus):
        """Receive InferenceStatus updates for wait_until_ready polling."""
        with self._phase_lock:
            self._latest_phase = msg.inference_phase
            self._latest_error = getattr(msg, 'error', '')

    def tick(self) -> NodeStatus:
        """Execute the action using a tick-based state machine."""
        if self._state == self._STATE_INIT:
            command_val = COMMAND_MAP.get(self.command_str)
            if command_val is None:
                self.log_error(f'Unknown command: {self.command_str}')
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            self.log_info(
                f'SendCommand started (command={self.command_str})'
            )
            self._state = self._STATE_WAITING_SERVICE
            self._start_time = time.monotonic()
            return NodeStatus.RUNNING

        if self._state == self._STATE_WAITING_SERVICE:
            if not self._client.service_is_ready():
                if time.monotonic() - self._start_time > SERVICE_CALL_TIMEOUT_SEC:
                    self.log_error('SendCommand service not available')
                    self._state = self._STATE_DONE
                    self._result = False
                    return NodeStatus.FAILURE
                return NodeStatus.RUNNING

            # Service is ready, send request
            req = SendCommand.Request()
            req.command = COMMAND_MAP[self.command_str]

            if self.command_str in ('START_INFERENCE', 'RESUME_INFERENCE'):
                req.task_info = TaskInfo()
                req.task_info.policy_path = self.policy_path
                req.task_info.task_name = self.task_name
                if self.control_hz:
                    req.task_info.control_hz = int(self.control_hz)
                if self.task_instruction:
                    if isinstance(self.task_instruction, list):
                        req.task_info.task_instruction = self.task_instruction
                    else:
                        req.task_info.task_instruction = [self.task_instruction]

            self._future = self._client.call_async(req)
            self._start_time = time.monotonic()
            self._state = self._STATE_CALLING
            return NodeStatus.RUNNING

        if self._state == self._STATE_CALLING:
            if not self._future.done():
                if time.monotonic() - self._start_time > SERVICE_CALL_TIMEOUT_SEC:
                    self.log_error('Service call timed out')
                    self._future.cancel()
                    self._state = self._STATE_DONE
                    self._result = False
                    return NodeStatus.FAILURE
                return NodeStatus.RUNNING

            response = self._future.result()
            if response is None or not response.success:
                msg = response.message if response else 'No response'
                self.log_error(f'SendCommand failed: {msg}')
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            self.log_info(
                f'SendCommand {self.command_str}: {response.message}'
            )

            if self.wait_until_ready and self.command_str == 'START_INFERENCE':
                self._start_time = time.monotonic()
                self._state = self._STATE_WAITING_LOAD
                self.log_info('Waiting for model loading to complete...')
                return NodeStatus.RUNNING

            self._state = self._STATE_DONE
            self._result = True
            return NodeStatus.SUCCESS

        if self._state == self._STATE_WAITING_LOAD:
            elapsed = time.monotonic() - self._start_time
            if elapsed > LOADING_TIMEOUT_SEC:
                self.log_error('Model loading timed out')
                self._state = self._STATE_DONE
                self._result = False
                return NodeStatus.FAILURE

            with self._phase_lock:
                phase = self._latest_phase
                error = self._latest_error

            if phase is not None and phase != InferenceStatus.LOADING:
                if phase in (InferenceStatus.INFERENCING, InferenceStatus.PAUSED):
                    self.log_info(f'Model loaded (phase={phase})')
                    self._state = self._STATE_DONE
                    self._result = True
                    return NodeStatus.SUCCESS
                elif phase == InferenceStatus.READY and error:
                    self.log_error(
                        f'Model loading failed: {error}'
                    )
                    self._state = self._STATE_DONE
                    self._result = False
                    return NodeStatus.FAILURE
                elif phase not in (InferenceStatus.READY,):
                    self.log_warn(
                        f'Unexpected phase {phase} during loading'
                    )

            return NodeStatus.RUNNING

        # _STATE_DONE
        return NodeStatus.SUCCESS if self._result else NodeStatus.FAILURE

    def reset(self):
        """Reset the action to its initial state."""
        super().reset()
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._future = None
        self._state = self._STATE_INIT
        self._result = None
        self._start_time = None
