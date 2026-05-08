#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Dongyun Kim

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from rclpy.node import Node

from interfaces.msg import TrainingInfo, TrainingStatus
from orchestrator.internal.communication.container_service_client import (
    ServiceResponse,
    ContainerServiceClient,
)

logger = logging.getLogger(__name__)


class ZenohTrainingManager:
    """
    Training manager that delegates to Docker container via Zenoh.

    Works with any container (LeRobot, GR00T, etc.) that implements
    the standard training service interface, parameterized by service_prefix.

    Architecture:
    - orchestrator uses ROS2 + rmw_zenoh (standard ROS2 API)
    - container uses zenoh_ros2_sdk (no ROS2 installed)
    - rmw_zenoh converts ROS2 messages to Zenoh protocol = COMPATIBLE
    """

    SUPPORTED_POLICIES = [
        'tdmpc', 'diffusion', 'act', 'vqbet', 'pi0', 'pi0_fast', 'pi05',
        'smolvla', 'groot', 'xvla', 'sac'
    ]

    SUPPORTED_DEVICES = ['cuda', 'cpu']

    _cached_policies: Optional[list] = None
    _cached_policy_details: Optional[list] = None

    def __init__(self, node: Node = None, service_prefix: str = "/lerobot",
                 client_cb_group=None):
        # ROS2 node required for creating service clients
        self._node = node
        self._service_prefix = service_prefix
        self.training_info = TrainingInfo()
        self.client = ContainerServiceClient(
            node=node, service_prefix=service_prefix,
            callback_group=client_cb_group,
        )
        self._connected = False
        self._status_callback: Optional[Callable] = None
        self._current_status = 'idle'
        self._current_step = 0
        self._total_steps = 0
        self._current_loss = float('nan')
        self._training_completed = False

        self.resume = False
        self.resume_model_path = None
        self.stop_event = threading.Event()

    def connect(self, node: Node = None) -> bool:
        if self._connected:
            return True
        if node is not None:
            self._node = node
        self._connected = self.client.connect()
        if self._connected:
            self.client.subscribe_progress(self._on_status_update)
        return self._connected

    def disconnect(self):
        if self._connected:
            self.client.disconnect()
            self._connected = False

    def _on_status_update(self, status_data: dict):
        new_status = status_data.get('status', 'unknown')

        if 'step' in status_data:
            self._current_step = status_data.get('step', 0)
        if 'total_steps' in status_data:
            self._total_steps = status_data.get('total_steps', 0)
        if 'loss' in status_data:
            loss_value = status_data.get('loss')
            if loss_value is not None and loss_value != 0:
                self._current_loss = float(loss_value)

        if self._current_status == 'training' and new_status == 'idle':
            self._training_completed = True
            logger.info('Training completed (state changed to idle)')
        elif new_status == 'error':
            self._training_completed = True
            logger.error('Training failed with error')

        self._current_status = new_status

        if self._status_callback:
            self._status_callback(status_data)

    def set_status_callback(self, callback: Callable):
        self._status_callback = callback

    @staticmethod
    def get_available_list() -> tuple[list[str], list[str]]:
        """Get available policies and devices."""
        if ZenohTrainingManager._cached_policies is None:
            ZenohTrainingManager._fetch_policies_from_container()

        policy_list = (
            ZenohTrainingManager._cached_policies
            if ZenohTrainingManager._cached_policies
            else ZenohTrainingManager.SUPPORTED_POLICIES
        )

        return (policy_list, ZenohTrainingManager.SUPPORTED_DEVICES)

    @staticmethod
    def get_policy_details() -> list:
        """Get detailed policy information from LeRobot container."""
        if ZenohTrainingManager._cached_policy_details is None:
            ZenohTrainingManager._fetch_policies_from_container()

        return ZenohTrainingManager._cached_policy_details or []

    @staticmethod
    def _fetch_policies_from_container():
        """Fetch policies from container.

        Note: Static method cannot access ROS2 node, so this currently
        just uses the default policy list. For dynamic policy fetching,
        use an instance method with a connected client.
        """
        # Cannot fetch from container without a node for service clients
        # Just use the default policies
        ZenohTrainingManager._cached_policies = ZenohTrainingManager.SUPPORTED_POLICIES

    @staticmethod
    def get_weight_save_root_path() -> Path:
        """
        Get the root path for saving training weights.

        Returns
        -------
        Path
            Path to training outputs directory
        """
        return Path.home() / '.cache' / 'lerobot' / 'outputs' / 'train'

    def get_current_training_status(self) -> TrainingStatus:
        status = TrainingStatus()
        status.training_info = self.training_info
        status.current_step = self._current_step
        status.current_loss = self._current_loss
        return status

    def train(self) -> ServiceResponse:
        """Start training and wait for completion."""
        if not self._connected:
            if not self.connect():
                return ServiceResponse(
                    success=False,
                    message='Failed to connect to training server',
                    data={},
                    request_id=''
                )

        self._training_completed = False
        self._current_status = 'idle'

        if self.resume and self.resume_model_path:
            response = self.client.resume_training(self.resume_model_path)
        else:
            steps = self.training_info.steps if self.training_info.steps > 0 else 0
            batch = self.training_info.batch_size if self.training_info.batch_size > 0 else 0
            response = self.client.start_training(
                policy_type=self.training_info.policy_type,
                dataset_path=self.training_info.dataset,
                output_dir=self.training_info.output_folder_name or '',
                num_epochs=steps,
                batch_size=batch,
                eval_freq=self.training_info.eval_freq if self.training_info.eval_freq > 0 else 0,
                log_freq=self.training_info.log_freq if self.training_info.log_freq > 0 else 0,
                save_freq=self.training_info.save_freq if self.training_info.save_freq > 0 else 0
            )

        if not response.success:
            return response

        timeout = 3600
        start_time = time.time()

        while not self._training_completed:
            time.sleep(1)

            if time.time() - start_time > timeout:
                logger.warning(f'Training timeout after {timeout}s')
                break

        final_response = ServiceResponse(
            success=self._current_status == 'idle',
            message=f'Training {self._current_status}',
            data={
                'status': self._current_status,
                'step': self._current_step,
                'loss': self._current_loss
            },
            request_id=''
        )

        return final_response

    def stop(self) -> ServiceResponse:
        self.stop_event.set()
        if not self._connected:
            return ServiceResponse(
                success=False,
                message='Not connected to training server',
                data={},
                request_id=''
            )
        return self.client.stop_training()

    def get_status(self) -> ServiceResponse:
        if not self._connected:
            return ServiceResponse(
                success=False,
                message='Not connected to training server',
                data={},
                request_id=''
            )
        return self.client.get_training_status()
