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

"""
ContainerServiceClient - Unified ROS2 Service Client for container communication.

Generic client that works with any container (GR00T, LeRobot, etc.)
by parameterizing the service prefix (e.g., "/groot", "/lerobot").

Supports both inference and training services:
  - /{prefix}/inference_command (InferenceCommand, Step 4 §10.5)
  - /{prefix}/train             (TrainModel)
  - /{prefix}/stop              (StopTraining — training only)
  - /{prefix}/status            (TrainingStatus)
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from interfaces.msg import TrainingProgress
from interfaces.srv import (
    InferenceCommand,
    StopTraining,
    TrainingStatus,
    TrainModel,
)
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

logger = logging.getLogger(__name__)


@dataclass
class ServiceResponse:
    success: bool
    message: str
    data: Dict[str, Any]
    request_id: str

    @classmethod
    def from_service_response(
        cls, response: Any, request_id: str = ''
    ) -> 'ServiceResponse':
        if response is None:
            return cls(
                success=False,
                message='No response from service (timeout or error)',
                data={},
                request_id=request_id
            )
        return cls(
            success=getattr(response, 'success', False),
            message=getattr(response, 'message', ''),
            data=cls._extract_data(response),
            request_id=request_id
        )

    @staticmethod
    def _extract_data(response: Any) -> Dict[str, Any]:
        data = {}
        for attr in [
            'job_id', 'state', 'step', 'total_steps', 'loss',
            'learning_rate', 'chunk_size', 'action_dim',
        ]:
            if hasattr(response, attr):
                data[attr] = getattr(response, attr)
        for attr in [
            'policies', 'checkpoints', 'models',
            'action_chunk', 'action_keys',
        ]:
            if hasattr(response, attr):
                data[attr] = list(getattr(response, attr))
        return data



class ContainerServiceClient:
    """Unified ROS2 service client for inference/training containers.

    Works with any container (GR00T, LeRobot, etc.) that implements
    the standard service interface, parameterized by service_prefix.
    """

    # InferenceCommand enum — mirrors interfaces/srv/InferenceCommand.srv.
    # The .srv file is canonical; these mirror constants are for ergonomic
    # call sites (orchestrator dispatchers).
    CMD_LOAD = 0
    CMD_START = 1
    CMD_PAUSE = 2
    CMD_RESUME = 3
    CMD_STOP = 4
    CMD_UNLOAD = 5
    CMD_UPDATE_INSTRUCTION = 6

    def __init__(
        self,
        node: Node,
        service_prefix: str = "/groot",
        timeout_sec: float = 180.0,
        callback_group: Optional[CallbackGroup] = None,
    ):
        self._node = node
        self._service_prefix = service_prefix
        self.timeout_sec = timeout_sec
        self._connected = False
        self._cancelled = threading.Event()
        self._callback_group = callback_group

        # Service clients
        self._inference_command_client = None
        self._stop_client = None
        self._train_client = None
        self._status_client = None

        # Subscribers
        self._progress_sub = None

    # --- Service name properties ---

    @property
    def service_inference_command(self) -> str:
        return f"{self._service_prefix}/inference_command"

    @property
    def service_stop(self) -> str:
        return f"{self._service_prefix}/stop"

    @property
    def service_train(self) -> str:
        return f"{self._service_prefix}/train"

    @property
    def service_status(self) -> str:
        return f"{self._service_prefix}/status"

    @property
    def topic_progress(self) -> str:
        return f"{self._service_prefix}/progress"

    # --- Connection management ---

    def connect(self) -> bool:
        """Create ROS2 service clients for container communication."""
        if self._connected:
            return True

        if self._node is None:
            logger.error("No ROS2 node provided")
            return False

        try:
            if self._callback_group is None:
                self._callback_group = ReentrantCallbackGroup()

            self._inference_command_client = self._node.create_client(
                InferenceCommand,
                self.service_inference_command,
                callback_group=self._callback_group,
            )
            self._stop_client = self._node.create_client(
                StopTraining,
                self.service_stop,
                callback_group=self._callback_group,
            )
            self._train_client = self._node.create_client(
                TrainModel,
                self.service_train,
                callback_group=self._callback_group,
            )
            self._status_client = self._node.create_client(
                TrainingStatus,
                self.service_status,
                callback_group=self._callback_group,
            )

            self._connected = True
            logger.info(
                f"Connected to container services "
                f"(prefix={self._service_prefix})"
            )
            return True
        except Exception as e:
            logger.error(f"Service client connection failed: {e}")
            return False

    def disconnect(self):
        """Destroy all service clients and subscribers."""
        if self._progress_sub is not None:
            try:
                self._node.destroy_subscription(self._progress_sub)
            except Exception:
                pass
            self._progress_sub = None

        clients = [
            ("_inference_command_client", self._inference_command_client),
            ("_stop_client", self._stop_client),
            ("_train_client", self._train_client),
            ("_status_client", self._status_client),
        ]
        for name, client in clients:
            if client is not None:
                try:
                    self._node.destroy_client(client)
                except Exception as e:
                    logger.debug(f"Error destroying client {name}: {e}")

        self._inference_command_client = None
        self._stop_client = None
        self._train_client = None
        self._status_client = None
        self._connected = False

    # --- Core service call ---

    def _call_service(
        self, client, request, service_name: str, timeout_sec: float = None
    ) -> ServiceResponse:
        """Call a ROS2 service and return ServiceResponse."""
        if not self._connected:
            return ServiceResponse(
                success=False,
                message="Not connected to container services",
                data={},
                request_id="",
            )

        if client is None:
            return ServiceResponse(
                success=False,
                message=f"Service client not initialized: {service_name}",
                data={},
                request_id="",
            )

        timeout = timeout_sec or self.timeout_sec

        try:
            if not client.wait_for_service(timeout_sec=5.0):
                return ServiceResponse(
                    success=False,
                    message=f"Service not available: {service_name}",
                    data={},
                    request_id="",
                )

            logger.debug(f"Calling service {service_name}")
            future = client.call_async(request)

            # Wait for the executor to resolve the future.
            # MultiThreadedExecutor + dedicated client callback group ensures
            # the response is processed on a separate thread, so this never
            # deadlocks even when called from a service callback.
            done_event = threading.Event()
            future.add_done_callback(lambda _: done_event.set())

            # Wait in short intervals so we can also check _cancelled.
            deadline = time.perf_counter() + timeout
            while not done_event.is_set():
                if self._cancelled.is_set():
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                done_event.wait(timeout=min(remaining, 1.0))

            if future.done():
                response = future.result()
                if response is not None:
                    return ServiceResponse.from_service_response(response)
                else:
                    return ServiceResponse(
                        success=False,
                        message=f"Service call returned None: {service_name}",
                        data={},
                        request_id="",
                    )
            else:
                future.cancel()
                reason = "cancelled" if self._cancelled.is_set() else "timed out"
                return ServiceResponse(
                    success=False,
                    message=f"Service call {reason}: {service_name}",
                    data={},
                    request_id="",
                )
        except Exception as e:
            logger.error(f"Service call to {service_name} failed: {e}")
            return ServiceResponse(
                success=False,
                message=f"Service call failed: {str(e)}",
                data={},
                request_id="",
            )

    # --- Inference lifecycle (Step 4 §10.5) ---

    def inference_command(
        self,
        command: int,
        model_path: str = "",
        embodiment_tag: str = "",
        robot_type: str = "",
        task_instruction: str = "",
        timeout_sec: Optional[float] = None,
    ) -> ServiceResponse:
        """Call /{prefix}/inference_command (InferenceCommand.srv).

        Use the ``CMD_*`` class constants for ``command``. Only LOAD needs
        the model/embodiment/robot_type fields; pass empty strings for the
        other commands. ``task_instruction`` is used by LOAD (training-time
        conditioning) and RESUME (online re-conditioning).

        Timeout defaults to 600 s for LOAD (CUDA init + weight load) and
        10 s for everything else; override via ``timeout_sec``.
        """
        request = InferenceCommand.Request()
        request.command = int(command)
        request.model_path = model_path
        request.embodiment_tag = embodiment_tag
        request.robot_type = robot_type
        request.task_instruction = task_instruction

        if timeout_sec is None:
            timeout_sec = 600.0 if command == self.CMD_LOAD else 10.0

        return self._call_service(
            self._inference_command_client,
            request,
            self.service_inference_command,
            timeout_sec=timeout_sec,
        )

    # --- Training services ---

    def start_training(
        self,
        policy_type: str,
        dataset_path: str,
        output_dir: str = '',
        num_epochs: int = 0,
        batch_size: int = 0,
        learning_rate: float = 0.0,
        eval_freq: int = 0,
        log_freq: int = 0,
        save_freq: int = 0,
        wandb_project: str = '',
    ) -> ServiceResponse:
        """Call /{prefix}/train to start training."""
        request = TrainModel.Request()
        request.policy_type = policy_type
        request.dataset_path = dataset_path
        request.output_dir = output_dir
        request.steps = num_epochs
        request.batch_size = batch_size
        request.learning_rate = learning_rate
        request.eval_freq = eval_freq
        request.log_freq = log_freq
        request.save_freq = save_freq
        request.wandb_project = wandb_project
        request.push_to_hub = False

        return self._call_service(
            self._train_client, request, self.service_train
        )

    def resume_training(self, checkpoint_path: str) -> ServiceResponse:
        """Call /{prefix}/train to resume training from checkpoint."""
        request = TrainModel.Request()
        request.policy_type = ''
        request.dataset_path = ''
        request.output_dir = checkpoint_path
        request.steps = 0
        request.batch_size = 0
        request.learning_rate = 0.0
        request.eval_freq = 0
        request.log_freq = 0
        request.save_freq = 0
        request.wandb_project = ''
        request.push_to_hub = False

        return self._call_service(
            self._train_client, request, self.service_train
        )

    def stop_training(self) -> ServiceResponse:
        """Call /{prefix}/stop to stop training."""
        request = StopTraining.Request()
        return self._call_service(
            self._stop_client, request, self.service_stop
        )

    def get_training_status(self) -> ServiceResponse:
        """Call /{prefix}/status to get training status."""
        request = TrainingStatus.Request()
        return self._call_service(
            self._status_client, request, self.service_status
        )

    # --- Subscriptions ---

    def subscribe_progress(self, callback: Callable[[Dict], None]) -> bool:
        """Subscribe to /{prefix}/progress topic for training updates."""
        try:
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE
            )

            def on_progress(msg: TrainingProgress):
                data = {
                    'status': msg.state,
                    'step': msg.step,
                    'total_steps': msg.total_steps,
                    'epoch': msg.epoch,
                    'loss': msg.loss,
                    'learning_rate': msg.learning_rate,
                    'gradient_norm': msg.gradient_norm,
                    'samples_per_second': msg.samples_per_second,
                    'elapsed_seconds': msg.elapsed_seconds,
                    'eta_seconds': msg.eta_seconds,
                }
                callback(data)

            self._progress_sub = self._node.create_subscription(
                TrainingProgress,
                self.topic_progress,
                on_progress,
                qos
            )
            return True
        except Exception as e:
            logger.error(f'Failed to subscribe to progress: {e}')
            return False
