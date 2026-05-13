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
# Author: Dongyun Kim, Seongwoo Kim

from datetime import datetime
import glob
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from interfaces.msg import (
    BrowserItem,
    DataOperationStatus,
    InferenceStatus,
    TaskInfo,
    TrainingStatus,
)
from interfaces.srv import (
    BrowseFile,
    GetDatasetList,
    GetHFUser,
    GetModelWeightList,
    GetPolicyList,
    GetReplayData,
    GetRobotInfo,
    GetRobotTypeList,
    GetTrainingInfo,
    GetUserList,
    HFEndpointList,
    RecordingCommand,
    SelectHFEndpoint,
    SendCommand,
    SendTrainingCommand,
    SetHFUser,
    SetRobotType,
)

from orchestrator.internal.communication.communicator import Communicator
from orchestrator.internal.communication.cyclo_data_client import CycloDataClient
# DataManager is imported only for its whoami_huggingface @staticmethod
# used by set_hf_user / get_hf_user callbacks. Session-state ownership
# lives in cyclo_data.RecordingService (Step 3 Part C2d).
# TODO(post-C2d): relocate whoami_huggingface to cyclo_data.hub and drop
# this import.
from cyclo_data.recorder.session_manager import DataManager
from cyclo_data.hub.endpoint_store import HFEndpointStore
from cyclo_data.recorder.replay_handler import ReplayDataHandler
from orchestrator.internal.communication.container_service_client import (
    ContainerServiceClient,
)
from orchestrator.timer.timer_manager import TimerManager
from orchestrator.training.zenoh_training_manager import ZenohTrainingManager
from orchestrator.internal.file_browser.file_browse_utils import FileBrowseUtils
from shared.robot_configs import schema as robot_schema
from cyclo_data.visualization.video_file_server import VideoFileServer

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class OrchestratorNode(Node):
    # Define operation modes (constants taken from Communicator)

    DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'
    DEFAULT_TOPIC_TIMEOUT = 5.0  # seconds
    PUB_QOS_SIZE = 10
    TRAINING_STATUS_TIMER_FREQUENCY = 0.5  # seconds
    VIDEO_SERVER_PORT = 8082  # Port for video file server

    class RosbagNotReadyException(Exception):
        """Exception raised when rosbag recording cannot start yet."""

        pass

    def __init__(self):
        # Phase 4: yaml is now nested (observation.images.<cam>.topic, …) —
        # let rclpy auto-declare every override that comes in via the
        # launch's parameters file so init_ros_params can read groups
        # back via get_parameters_by_prefix instead of declaring each
        # name by hand.
        super().__init__(
            'orchestrator',
            automatically_declare_parameters_from_overrides=True,
        )
        self.get_logger().info('Start Physical AI Server')

        # Callback groups for MultiThreadedExecutor.
        # Separating service servers from service clients prevents deadlock
        # when a service callback needs to call another service (e.g., FINISH
        # calling /groot/stop).
        self._service_cb_group = MutuallyExclusiveCallbackGroup()
        self._client_cb_group = ReentrantCallbackGroup()

        # _state_lock protects the session-state group below:
        #   on_recording / on_inference / start_recording_time
        #   is_training / training_thread / training_manager
        #   container_service_client
        # These are written by the inference daemon thread
        # (_load_and_start in user_interaction_callback) and the training
        # daemon thread, while service callbacks (MutuallyExclusive on
        # _service_cb_group) and subscription / timer callbacks (Reentrant
        # on _client_cb_group) read them. Without serialisation
        # _teardown_inference_client can null container_service_client
        # mid-call, and the joystick / user_interaction TOCTOU on
        # (on_recording, on_inference) can branch on stale state.
        #
        # IMPORTANT: never hold this lock across a ROS service call /
        # .call() / .call_async() — it would deadlock with any callback
        # that needs to acquire it. The lock only brackets pointer
        # reads/writes and the snapshot helper.
        self._state_lock = threading.Lock()

        self.params = None
        self.robot_section = None
        self.on_recording = False
        self.on_inference = False

        self.robot_type_list = self.get_robot_type_list()
        self.start_recording_time: float = 0.0

        self.training_thread = None
        self.is_training = False
        self.training_status_timer = None

        # Session scratch — used by recording-command forwarder to skip
        # init_robot_control_parameters_from_user_task when the task
        # identity hasn't changed. Tracks the orchestrator's view of
        # "what task are we configured for?" (cyclo_data owns the live
        # DataManager separately).
        self._current_task_name: Optional[str] = None

        self._init_core_components()

        self._init_ros_publisher()
        self._init_cyclo_data_bridge()
        self._init_ros_service()

        self._setup_timer_callbacks()

        self.goal_repo_id = None

    def _init_core_components(self):
        self.communicator: Optional[Communicator] = None
        # DataManager ownership moved to cyclo_data.RecordingService
        # (Step 3 Part C2d-2/-5). Orchestrator forwards via CycloDataClient.
        self.timer_manager: Optional[TimerManager] = None
        self.heartbeat_timer: Optional[TimerManager] = None
        self.training_timer: Optional[TimerManager] = None
        # Zenoh managers for training (Docker container communication)
        self.training_manager: Optional[ZenohTrainingManager] = None
        # InferenceCommand service client — lazily (re)created per service
        # prefix (/lerobot, /groot, …). The container owns the policy life-
        # cycle + 100 Hz control loop (§5.5); orchestrator only dispatches
        # LOAD / START / PAUSE / RESUME / STOP / UNLOAD from UI commands.
        self.container_service_client: Optional[ContainerServiceClient] = None

        # HF endpoint registry — orchestrator-owned because the
        # set/get/list/select_hf_endpoint services also read and mutate
        # it. The HfApiWorker itself and the /huggingface/status
        # publisher moved to cyclo_data.HubService (Step 3 Part C2c).
        self.hf_endpoint_store = HFEndpointStore()

        # MP4 Conversion Worker lifecycle moved to
        # cyclo_data.ConversionService (Step 3 Part C2e). orchestrator
        # forwards CONVERT_MP4 commands via CycloDataClient.start_conversion;
        # cyclo_data publishes record_phase=CONVERTING on
        # /data/recording/status directly for the UI (D18).

        # Initialize ReplayDataHandler for replay viewer
        self.replay_data_handler = ReplayDataHandler(logger=self.get_logger())

        # Initialize FileBrowseUtils for file browsing
        self.file_browse_utils = FileBrowseUtils(
            max_workers=8,
            logger=self.get_logger()
        )

        # Initialize Video File Server for replay viewer
        self._init_video_server()

    def _init_cyclo_data_bridge(self):
        """Wire the orchestrator ↔ cyclo_data bridge (Step 3 Part C2a+).

        C2a:  client wrapper + /data/status (umbrella) subscriber.
        D18:  /data/recording/status is now the UI-facing record-side
              topic — UI subscribes directly, no relay through
              orchestrator. The C2d-4 relay was retired when the phase
              field split into orthogonal record_phase / inference_phase
              (PLAN §10.3 D18, supersedes REVIEW §9.4).
        """
        self._cyclo_data = CycloDataClient(self, self._client_cb_group)
        self._data_operation_status_sub = self.create_subscription(
            DataOperationStatus,
            '/data/status',
            self._data_operation_status_callback,
            self.PUB_QOS_SIZE,
            callback_group=self._client_cb_group,
        )
        self.get_logger().info(
            'cyclo_data bridge initialised (CycloDataClient + '
            '/data/status subscriber active).'
        )

    def _snapshot_session_state(self):
        """Atomic read of (on_recording, on_inference) under _state_lock.

        Service / joystick callbacks that branch on both flags must read
        them together — otherwise the inference daemon thread can flip
        on_inference between the two bare reads and the wrong branch
        fires (e.g. "not recording" route while inference is starting).
        """
        with self._state_lock:
            return self.on_recording, self.on_inference

    def _set_session_active(self, *, on_recording=None, on_inference=None,
                             start_time=None):
        """Atomic write of any subset of (on_recording, on_inference,
        start_recording_time) under _state_lock. ``None`` means leave alone.
        """
        with self._state_lock:
            if on_recording is not None:
                self.on_recording = on_recording
            if on_inference is not None:
                self.on_inference = on_inference
            if start_time is not None:
                self.start_recording_time = start_time

    def _apply_cyclo_data_response(self, cd_result, response) -> None:
        """Mirror a CycloDataClient CallResult onto a SendCommand.Response.

        Small helper used by every Part C2d-4 forwarder branch so
        "cyclo_data said X" rides back on the SendCommand reply consistently.
        Preserves message-level detail when cyclo_data itself returned a
        structured response; falls back to the transport-level message
        (timeout / unreachable) when no response came back.
        """
        if cd_result.response is not None:
            response.success = bool(cd_result.response.success)
            response.message = (
                cd_result.response.message
                or cd_result.message
                or ''
            )
        else:
            response.success = False
            response.message = cd_result.message or 'cyclo_data call failed'

    def _forward_recording(self, command: int, task_info=None,
                           include_topics: bool = False):
        """DRY helper for every recording forwarder site.

        Populates topics + urdf_path from orchestrator-owned state
        (Communicator's topic inventory, `self.params['urdf_path']`) so
        call sites stay one-liners.
        """
        topics = (
            self.communicator.get_mcap_topics()
            if include_topics and self.communicator is not None
            else []
        )
        urdf_path = self.params.get('urdf_path', '') if self.params else ''
        return self._cyclo_data.send_recording_command(
            command=command,
            task_info=task_info if task_info is not None else self._last_ui_task_info,
            robot_type=self.robot_type,
            topics=topics,
            urdf_path=urdf_path,
        )

    def _data_operation_status_callback(self, msg: DataOperationStatus):
        """Debug-log DataOperationStatus arrivals.

        Conversion progress is now published as record_phase=CONVERTING
        on /data/recording/status by cyclo_data.ConversionService (D18,
        plan record-zippy-sunrise). HubService publishes HFOperationStatus
        directly. Nothing for orchestrator to relay here.
        """
        type_name = {
            DataOperationStatus.OP_RECORDING: 'RECORDING',
            DataOperationStatus.OP_CONVERSION: 'CONVERSION',
            DataOperationStatus.OP_HF: 'HF',
            DataOperationStatus.OP_EDIT: 'EDIT',
        }.get(msg.operation_type, f'OP_{msg.operation_type}')
        status_name = {
            DataOperationStatus.IDLE: 'IDLE',
            DataOperationStatus.RUNNING: 'RUNNING',
            DataOperationStatus.COMPLETED: 'COMPLETED',
            DataOperationStatus.FAILED: 'FAILED',
            DataOperationStatus.CANCELLED: 'CANCELLED',
        }.get(msg.status, f'STATUS_{msg.status}')
        self.get_logger().debug(
            f'[cyclo_data/status] type={type_name} status={status_name} '
            f'job_id={msg.job_id or "-"} stage={msg.stage or "-"} '
            f'msg={msg.message or "-"}'
        )

    def _init_video_server(self):
        """Initialize the video file server for replay viewer."""
        try:
            # Allow serving files from home directory and workspace
            allowed_paths = [
                str(Path.home()),
                '/workspace',  # Docker workspace directory
            ]
            self.video_server = VideoFileServer(
                port=self.VIDEO_SERVER_PORT,
                allowed_paths=allowed_paths,
                replay_data_handler=self.replay_data_handler
            )
            self.video_server.start()
            self.get_logger().info(
                f'Video file server started on port {self.VIDEO_SERVER_PORT}'
            )
            self.get_logger().info(
                f'Replay data API available at http://0.0.0.0:{self.VIDEO_SERVER_PORT}/replay-data/'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to start video server: {e}')
            self.video_server = None

    def _init_ros_publisher(self):
        self.get_logger().info('Initializing ROS publishers...')
        pub_qos_size = 100
        self.training_status_publisher = self.create_publisher(
            TrainingStatus,
            '/training/status',
            pub_qos_size
        )

    def _init_ros_service(self):
        self.get_logger().info('Initializing ROS services...')
        service_definitions = [
            ('/task/command', SendCommand, self.user_interaction_callback),
            ('/get_robot_types', GetRobotTypeList, self.get_robot_types_callback),
            ('/get_robot_info', GetRobotInfo, self.get_robot_info_callback),
            ('/set_robot_type', SetRobotType, self.set_robot_type_callback),
            ('/register_hf_user', SetHFUser, self.set_hf_user_callback),
            ('/get_registered_hf_user', GetHFUser, self.get_hf_user_callback),
            ('/huggingface/list_endpoints', HFEndpointList, self.list_hf_endpoints_callback),
            ('/huggingface/select_endpoint', SelectHFEndpoint, self.select_hf_endpoint_callback),
            ('/training/command', SendTrainingCommand, self.user_training_interaction_callback),
            ('/training/get_available_policy', GetPolicyList, self.get_available_list_callback),
            ('/training/get_user_list', GetUserList, self.get_user_list_callback),
            ('/training/get_dataset_list', GetDatasetList, self.get_dataset_list_callback),
            (
                '/training/get_model_weight_list',
                GetModelWeightList,
                self.get_model_weight_list_callback
            ),
            ('/training/get_training_info', GetTrainingInfo, self.get_training_info_callback),
            ('/replay/get_data', GetReplayData, self.get_replay_data_callback),
            ('/browse_file', BrowseFile, self.browse_file_callback),
        ]

        for service_name, service_type, callback in service_definitions:
            self.create_service(
                service_type, service_name, callback,
                callback_group=self._service_cb_group,
            )

        self.get_logger().info('ROS services initialized successfully')

    def _setup_timer_callbacks(self):
        # Step 4 §5.5: inference no longer needs an orchestrator-side 100 Hz
        # timer — the policy container (runtime/control_publisher.py) owns
        # the control loop. orchestrator publishes InferenceStatus on
        # command transitions (LOAD / START / PAUSE / RESUME / STOP)
        # instead of from a polling timer.
        self.timer_callback_dict = {
            'collection': self._data_collection_timer_callback,
        }

    def init_ros_params(self, robot_type):
        self.get_logger().info(f'Initializing ROS parameters for robot type: {robot_type}')
        # Phase 4: yaml is auto-declared as ROS2 params via the
        # automatically_declare_parameters_from_overrides flag on
        # super().__init__ (so `ros2 param get` introspection still works
        # on every nested key). Internally we read the yaml directly via
        # the schema helper — get_parameters_by_prefix would round-trip
        # through ROS2's flat string encoding for no benefit.
        self.robot_section = robot_schema.load_robot_section(robot_type)
        self.params = {
            'urdf_path': robot_schema.get_urdf_path(self.robot_section),
            'robot_name': robot_schema.get_robot_name(self.robot_section),
        }

        self.communicator = Communicator(
            node=self,
            robot_section=self.robot_section,
        )

        if self.heartbeat_timer is None:
            self.heartbeat_timer = TimerManager(node=self)
            self.heartbeat_timer.set_timer(
                timer_name='heartbeat',
                timer_frequency=1.0,
                callback_function=self.communicator.heartbeat_timer_callback
            )
            self.heartbeat_timer.start(timer_name='heartbeat')

        # Register joystick handler for immediate processing
        self.communicator.register_joystick_handler(self.handle_joystick_trigger)

        self.get_logger().info(
            f'ROS parameters initialized successfully for robot type: {robot_type}')

    def get_training_status(self):
        msg = TrainingStatus()
        # Snapshot both manager and flag together so the
        # _start_training_thread / _cleanup_training_on_completion
        # race (which can null training_manager and flip is_training)
        # can't tear the read across the dereference below.
        with self._state_lock:
            training_manager = self.training_manager
            is_training = self.is_training
        if training_manager is None:
            return
        try:
            current_status = training_manager.get_current_training_status()
            training_info = current_status.training_info
            current_step = current_status.current_step
            current_loss = current_status.current_loss
            msg.training_info = training_info
            msg.current_step = current_step
            msg.current_loss = current_loss
            msg.is_training = is_training
            msg.error = ''
        except Exception as e:
            msg.current_step = 0
            msg.current_loss = float('nan')
            msg.error = str(e)
            self.get_logger().error(f'Error publishing training status: {msg.error}')
            return msg
        return msg

    def init_robot_control_parameters_from_user_task(
            self,
            task_info):
        self.get_logger().info(
            'Initializing robot control parameters from user task...')
        # DataManager ownership moved to cyclo_data.RecordingService
        # in Step 3 Part C2d-2/-5 — orchestrator no longer instantiates one.
        # The rest of this function still drives inference-side config
        # (control_hz, joint_order, params) which stays on this node.

        control_hz = getattr(task_info, 'control_hz', 0) or 100
        inference_hz = getattr(task_info, 'inference_hz', 0) or 15
        chunk_align_window_s = getattr(task_info, 'chunk_align_window_s', 0.0)
        if chunk_align_window_s <= 0.0:
            chunk_align_window_s = 0.3
        self._control_hz = control_hz
        self._inference_hz = inference_hz
        self._chunk_align_window_s = chunk_align_window_s

        # Stop previous timer before creating a new one
        if hasattr(self, 'timer_manager') and self.timer_manager is not None:
            self.timer_manager.stop_all()

        self.timer_manager = TimerManager(node=self)

        # Inference mode runs the 100 Hz control loop inside the policy
        # container since Step 4 §5.5 — orchestrator only registers the
        # 'collection' joystick-pump timer here. The timer is started
        # explicitly by START_RECORD / joystick handlers.
        for timer_name, callback in self.timer_callback_dict.items():
            self.timer_manager.set_timer(
                timer_name=timer_name,
                timer_frequency=control_hz,
                callback_function=callback,
            )
        self.get_logger().info(
            f'Robot control parameters initialized (control_hz={control_hz})')

    def clear_parameters(self):
        if self.communicator is not None:
            self.communicator.cleanup()
            self.communicator = None

        if self.timer_manager is not None:
            self.timer_manager = None

        if self.heartbeat_timer is not None:
            self.heartbeat_timer.stop(timer_name='heartbeat')
            self.heartbeat_timer = None

        if self.training_timer is not None:
            self.training_timer.stop(timer_name='training_status')
            self.training_timer = None

        self.params = None
        self.robot_section = None

    def set_hf_user_callback(self, request, response):
        """Validate ``token`` against ``endpoint`` and persist on success.

        The previous flow validated against a single global token; this one
        keeps a per-endpoint store so the user can have one token for the
        official hub and another for the internal hub at the same time.

        Trailing '/' is stripped: HfApi appends '/api/whoami-v2' verbatim,
        so 'http://host:1000/' becomes 'http://host:1000//api/whoami-v2'
        — nginx 404s the double-slash path and (in some configs) emits a
        Location header that flips the scheme to https, which then trips
        an SSL handshake against the same HTTP port and surfaces as
        'SSL: WRONG_VERSION_NUMBER'.
        """
        endpoint = (request.endpoint or '').strip().rstrip('/')
        label = (request.label or '').strip()
        token = (request.token or '').strip()

        if not endpoint:
            response.user_id_list = []
            response.success = False
            response.message = 'endpoint is required'
            return response
        if not token:
            response.user_id_list = []
            response.success = False
            response.message = 'token is required'
            return response

        self.get_logger().info(
            f'register_hf_user: endpoint={endpoint!r} token=<{len(token)} chars>'
        )

        try:
            user_ids = DataManager.whoami_huggingface(endpoint, token)
            if not user_ids:
                response.user_id_list = []
                response.success = False
                response.message = (
                    f'Token validation timed out for endpoint {endpoint}'
                )
                return response

            primary_user = user_ids[0] if user_ids else ''
            self.hf_endpoint_store.set(
                endpoint=endpoint,
                label=label,
                token=token,
                user_id=primary_user,
            )
            self.get_logger().info(
                f'Registered HF token for {endpoint} ({primary_user})'
            )
            response.user_id_list = user_ids
            response.success = True
            response.message = (
                f'Token validated and stored for {endpoint} ({primary_user})'
            )
        except Exception as e:
            self.get_logger().error(f'Error in set_hf_user_callback: {str(e)}')
            response.user_id_list = []
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_hf_user_callback(self, request, response):
        """Return the user list for ``request.endpoint`` (empty = active)."""
        endpoint = (request.endpoint or '').strip().rstrip('/')
        try:
            entry = self.hf_endpoint_store.resolve(endpoint)
            if entry is None:
                response.user_id_list = []
                response.success = False
                response.message = (
                    'No HuggingFace endpoint registered yet — register a token '
                    'from the UI first.'
                )
                return response

            user_ids = DataManager.whoami_huggingface(entry.endpoint, entry.token)
            if not user_ids:
                response.user_id_list = []
                response.success = False
                response.message = (
                    f'Token validation timed out for {entry.endpoint}'
                )
                return response

            response.user_id_list = user_ids
            response.success = True
            response.message = (
                f'Resolved {len(user_ids)} user id(s) for {entry.endpoint}'
            )
        except Exception as e:
            self.get_logger().error(f'Error in get_hf_user_callback: {str(e)}')
            response.user_id_list = []
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def list_hf_endpoints_callback(self, request, response):
        """Return every registered endpoint plus the currently active one."""
        try:
            entries = self.hf_endpoint_store.list()
            active = self.hf_endpoint_store.get_active()
            response.endpoints = [e.endpoint for e in entries]
            response.labels = [e.label for e in entries]
            response.user_ids = [e.user_id for e in entries]
            response.active = active.endpoint if active else ''
            response.success = True
            response.message = f'{len(entries)} endpoint(s) registered'
        except Exception as e:
            self.get_logger().error(f'Error in list_hf_endpoints_callback: {str(e)}')
            response.endpoints = []
            response.labels = []
            response.user_ids = []
            response.active = ''
            response.success = False
            response.message = f'Error: {str(e)}'
        return response

    def select_hf_endpoint_callback(self, request, response):
        """Set the active endpoint. Empty string clears the selection."""
        endpoint = (request.endpoint or '').strip().rstrip('/')
        try:
            ok = self.hf_endpoint_store.set_active(endpoint)
            if not ok:
                response.success = False
                response.message = (
                    f'Endpoint not registered: {endpoint}. Register a token '
                    f'for it first.'
                )
                return response
            response.success = True
            response.message = (
                f'Active endpoint set to {endpoint or "<none>"}'
            )
        except Exception as e:
            self.get_logger().error(f'Error in select_hf_endpoint_callback: {str(e)}')
            response.success = False
            response.message = f'Error: {str(e)}'
        return response

    def get_robot_type_list(self):
        # Robot config YAMLs live in the shared package
        # (shared/robot_configs/) alongside urdf/ + ffw_description/.
        # Match the launch file's discovery so the list of available
        # robots stays in lock-step with what was loaded as ROS params.
        shared_dir = get_package_share_directory('shared')
        config_dir = os.path.join(shared_dir, 'robot_configs')
        config_files = glob.glob(os.path.join(config_dir, '*_config.yaml'))
        config_files.sort()

        robot_type_list = []
        for config_file in config_files:
            robot_type = os.path.splitext(os.path.basename(config_file))[0]
            if robot_type.endswith('_config'):
                robot_type = robot_type[:-7]
            robot_type_list.append(robot_type)

        self.get_logger().info(f'Available robot types: {robot_type_list}')
        return robot_type_list

    # handle_rosbag_recording / stop_recording_and_save / cancel_current_recording
    # all moved to cyclo_data.services.recording_service.RecordingService in
    # Step 3 Part C2d-3/-5. Orchestrator dispatches through
    # self._forward_recording(...) — the rosbag state machine is no longer
    # driven from this node.

    def _data_collection_timer_callback(self):
        """Pump joystick triggers at control rate.

        Recording status / rosbag lifecycle both live in cyclo_data now
        (Part C2d-4/-5). RecordingStatus reaches the UI via
        /data/recording/status (cyclo_data → UI direct, D18). This
        callback is reduced to a joystick pump and may collapse further
        once Step 4 Brain Migrator finishes inference restructuring.
        """
        updated, mode = self.communicator.consume_joystick_update()
        if updated:
            self.handle_joystick_trigger(joystick_mode=mode)

    def _publish_inference_phase(self, phase: int, error: str = '') -> None:
        """Publish a one-shot InferenceStatus on /task/inference_status.

        The container owns the 100 Hz control loop (§5.5); orchestrator
        only signals LOADING / INFERENCING / PAUSED / READY on commands.
        Record-side phase lives on /data/recording/status (D18).
        """
        if self.communicator is None:
            return
        robot_type = getattr(self, 'robot_type', '') or ''
        self.communicator.publish_inference_status(
            phase=phase,
            robot_type=robot_type,
            error=error,
        )

    def user_training_interaction_callback(self, request, response):
        """
        Handle training command requests (START/FINISH).

        Supports both new training and resume functionality with proper validation.
        """
        try:
            if request.command == SendTrainingCommand.Request.START:
                # Reject second START before clobbering training_manager —
                # otherwise the prior manager / thread is orphaned and its
                # status publisher keeps running attached to a dead train().
                with self._state_lock:
                    prior_thread = self.training_thread
                if prior_thread is not None and prior_thread.is_alive():
                    response.success = False
                    response.message = 'Training is already in progress'
                    return response

                # Initialize training components (ROS2 services with rmw_zenoh)
                new_manager = ZenohTrainingManager(
                    node=self, client_cb_group=self._client_cb_group,
                )
                with self._state_lock:
                    self.training_manager = new_manager
                self.training_timer = TimerManager(node=self)
                self._setup_training_status_timer()

                # Extract resume parameters
                resume = getattr(request, 'resume', False)
                resume_model_path = getattr(request, 'resume_model_path', '').strip()

                # Log training request details
                output_folder_name = request.training_info.output_folder_name
                weight_save_root_path = ZenohTrainingManager.get_weight_save_root_path()
                self.get_logger().info(
                    f'Training request - Output: {output_folder_name}, '
                    f'Resume: {resume}, Model path: {resume_model_path}'
                )

                # Validate training configuration
                validation_result = self._validate_training_request(
                    resume, resume_model_path, output_folder_name, weight_save_root_path
                )
                if not validation_result['success']:
                    response.success = False
                    response.message = validation_result['message']
                    self._cleanup_training_on_error()
                    return response

                # Configure and start training
                self._configure_training_manager(request, resume, resume_model_path)
                self._start_training_thread()

                response.success = True
                response.message = 'Training started successfully'

            else:
                # Handle FINISH command
                if request.command == SendTrainingCommand.Request.FINISH:
                    self._stop_training()
                    response.success = True
                    response.message = 'Training stopped successfully'
                else:
                    response.success = False
                    response.message = f'Unknown command: {request.command}'

        except Exception as e:
            self.get_logger().error(f'Error in training callback: {str(e)}')
            response.success = False
            response.message = f'Training error: {str(e)}'
            self._cleanup_training_on_error()

        return response

    def _setup_training_status_timer(self):
        """Set up timer for publishing training status updates."""
        self.training_timer.set_timer(
            timer_name='training_status',
            timer_frequency=self.TRAINING_STATUS_TIMER_FREQUENCY,
            callback_function=lambda: self.training_status_publisher.publish(
                self.get_training_status()
            )
        )
        self.training_timer.start(timer_name='training_status')

    def _validate_training_request(
            self,
            resume,
            resume_model_path,
            output_folder_name,
            weight_save_root_path
    ):
        """
        Validate training request parameters.

        Returns
        -------
        dict
            {'success': bool, 'message': str}

        """
        # Check output folder conflicts for new training
        if not resume:
            output_path = weight_save_root_path / output_folder_name
            if output_path.exists():
                return {
                    'success': False,
                    'message': f'Output folder already exists: {output_path}'
                }

        # Validate resume configuration
        if resume:
            if not resume_model_path:
                return {
                    'success': False,
                    'message': 'Resume model path is required when resume=True'
                }

            # Check if resume config file exists
            full_config_path = weight_save_root_path / resume_model_path
            if not full_config_path.exists():
                return {
                    'success': False,
                    'message': f'Resume config file not found: {full_config_path}'
                }

        return {'success': True, 'message': 'Validation passed'}

    def _configure_training_manager(self, request, resume, resume_model_path):
        """Configure training manager with request parameters."""
        self.training_manager.training_info = request.training_info
        self.training_manager.resume = resume
        self.training_manager.resume_model_path = resume_model_path

    def _start_training_thread(self):
        """Start training in a separate thread."""
        def run_training():
            with self._state_lock:
                manager = self.training_manager
            try:
                if manager is not None:
                    manager.train()
            except Exception as e:
                self.get_logger().error(f'Training error: {str(e)}')
            finally:
                self._cleanup_training_on_completion()

        thread = threading.Thread(target=run_training, daemon=True)
        with self._state_lock:
            self.training_thread = thread
            self.is_training = True
        thread.start()

    def _stop_training(self):
        """Stop training gracefully."""
        with self._state_lock:
            self.is_training = False
            manager = self.training_manager
            thread = self.training_thread
        if manager is not None:
            manager.stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.DEFAULT_TOPIC_TIMEOUT)
        self._cleanup_training_on_completion()

    def _cleanup_training_on_completion(self):
        """Cleanup training resources on normal completion."""
        with self._state_lock:
            self.is_training = False
            manager = self.training_manager
        self.get_logger().info('Training completed.')
        training_status = self.get_training_status()
        self.training_status_publisher.publish(training_status)
        if manager is not None:
            manager.stop_event.set()
        if hasattr(self, 'training_timer'):
            self.training_timer.stop('training_status')

    def _cleanup_training_on_error(self):
        """Cleanup training resources on error."""
        with self._state_lock:
            self.is_training = False
            manager = self.training_manager
        training_status = self.get_training_status()
        self.training_status_publisher.publish(training_status)
        if manager is not None:
            manager.stop_event.set()
        if hasattr(self, 'training_timer'):
            self.training_timer.stop('training_status')

    def user_interaction_callback(self, request, response):
        """
        Handle user commands for recording control (simplified mode).

        Commands:
        - START_RECORD: Initialize and start recording
        - STOP / FINISH: Stop and save recording
        - RERECORD: Cancel current recording (discard)
        """
        try:
            if request.command == SendCommand.Request.REFRESH_TOPICS:
                # Forward to cyclo_data /data/recording with the topic
                # inventory from our Communicator (Part C2d-4).
                if self.communicator is None:
                    response.success = False
                    response.message = 'Communicator not initialized'
                    return response
                rosbag_topics = self.communicator.get_mcap_topics()
                cd_result = self._cyclo_data.send_recording_command(
                    command=RecordingCommand.Request.REFRESH_TOPICS,
                    robot_type=self.robot_type,
                    topics=rosbag_topics,
                )
                self._apply_cyclo_data_response(cd_result, response)
                return response

            elif request.command == SendCommand.Request.START_RECORD:
                # Forwarder: orchestrator owns on_recording / timer_manager /
                # params / task_info cache; cyclo_data owns DataManager +
                # rosbag + action_event. We still need
                # init_robot_control_parameters_from_user_task for the
                # inference-side state (joint_order, control_hz, params)
                # when the task identity changes.
                task_info = request.task_info
                self._last_ui_task_info = task_info
                task_name = f'{self.robot_type}_{task_info.task_name}'

                need_new_config = (
                    getattr(self, '_current_task_name', None) != task_name
                )
                if need_new_config:
                    self.get_logger().info('Initializing new recording session')
                    self.init_robot_control_parameters_from_user_task(task_info)
                    self._current_task_name = task_name
                else:
                    self.get_logger().info(
                        f'Continuing recording session — task={task_name}')
                if self.timer_manager:
                    self.timer_manager.start(timer_name='collection')

                if self.communicator is None:
                    response.success = False
                    response.message = 'Communicator not initialized'
                    return response
                rosbag_topics = self.communicator.get_mcap_topics()
                urdf_path = self.params.get('urdf_path', '') if self.params else ''

                cd_result = self._cyclo_data.send_recording_command(
                    command=RecordingCommand.Request.START,
                    task_info=task_info,
                    robot_type=self.robot_type,
                    topics=rosbag_topics,
                    urdf_path=urdf_path,
                )
                if (cd_result.success
                        and cd_result.response is not None
                        and cd_result.response.success):
                    self._set_session_active(
                        on_recording=True,
                        start_time=time.perf_counter(),
                    )
                    response.success = True
                    response.message = cd_result.response.message or 'Recording started'
                else:
                    self._apply_cyclo_data_response(cd_result, response)

            elif request.command == SendCommand.Request.START_INFERENCE:
                task_info = request.task_info

                task_instruction = (
                    task_info.task_instruction[0]
                    if task_info.task_instruction
                    else ''
                )
                service_prefix = self._determine_service_prefix(task_info)

                # If a policy is already loaded on this container, treat
                # START_INFERENCE as RESUME (the container's Process A is
                # either paused or idle — RESUME covers both by just
                # clearing the pause flag and re-conditioning).
                # Snapshot the client so a concurrent _teardown_inference_client
                # cannot null it between the prefix check and the call.
                with self._state_lock:
                    existing_client = self.container_service_client
                if (
                    existing_client is not None
                    and existing_client._service_prefix == service_prefix
                ):
                    resume_result = existing_client.inference_command(
                        ContainerServiceClient.CMD_RESUME,
                        task_instruction=task_instruction,
                    )
                    if resume_result.success:
                        self._set_session_active(
                            on_inference=True,
                            start_time=time.perf_counter(),
                        )
                        self._publish_inference_phase(InferenceStatus.INFERENCING)
                        response.success = True
                        response.message = 'Inference resumed (model already loaded)'
                    else:
                        response.success = False
                        response.message = resume_result.message
                    self.get_logger().info(
                        f'RESUME inference result: {response.message}'
                    )
                else:
                    # Fresh start. LOAD on the policy container can take 10+
                    # minutes the first time (model load to GPU + on-disk
                    # TRT engine build), which would block the SendCommand
                    # srv well past any reasonable UI timeout. So we set up
                    # state synchronously (cheap), publish phase=LOADING,
                    # kick off LOAD → START on a daemon thread, and return
                    # success=True immediately. UI tracks progress via the
                    # /task/inference_status topic (LOADING → INFERENCING,
                    # or READY+error on failure). Mirrors the
                    # InferenceManager.start() pattern from physical_ai_tools.
                    with self._state_lock:
                        existing_client = self.container_service_client
                    if existing_client is not None:
                        self._teardown_inference_client()

                    self.init_robot_control_parameters_from_user_task(task_info)

                    new_client = ContainerServiceClient(
                        node=self,
                        service_prefix=service_prefix,
                        callback_group=self._client_cb_group,
                    )
                    new_client.connect()
                    with self._state_lock:
                        self.container_service_client = new_client

                    self._publish_inference_phase(InferenceStatus.LOADING)

                    # Capture state for the thread closure. self.* fields
                    # may change before the thread completes (e.g. another
                    # FINISH wipes container_service_client), so the thread
                    # operates on the client we just created.
                    client = new_client
                    record_inference_mode = task_info.record_inference_mode
                    model_path = task_info.policy_path
                    robot_type = self.robot_type

                    def _load_and_start():
                        try:
                            load_result = client.inference_command(
                                ContainerServiceClient.CMD_LOAD,
                                model_path=model_path,
                                embodiment_tag='new_embodiment',
                                robot_type=robot_type,
                                task_instruction=task_instruction,
                            )
                            if not load_result.success:
                                self.get_logger().error(
                                    f'Async LOAD failed: {load_result.message}'
                                )
                                self._teardown_inference_client()
                                self._publish_inference_phase(
                                    InferenceStatus.READY, error=load_result.message
                                )
                                return

                            action_keys = load_result.data.get('action_keys', [])
                            self.get_logger().info(
                                f'LOAD ok action_keys={action_keys}'
                            )

                            start_result = client.inference_command(
                                ContainerServiceClient.CMD_START,
                            )
                            if not start_result.success:
                                self.get_logger().error(
                                    f'Async START failed: {start_result.message}'
                                )
                                self._teardown_inference_client()
                                self._publish_inference_phase(
                                    InferenceStatus.READY, error=start_result.message
                                )
                                return

                            self._set_session_active(
                                on_recording=True if record_inference_mode else None,
                                on_inference=True,
                                start_time=time.perf_counter(),
                            )
                            self._publish_inference_phase(InferenceStatus.INFERENCING)
                        except Exception as e:
                            self.get_logger().error(
                                f'Async LOAD/START error: {e}', exc_info=True
                            )
                            try:
                                self._teardown_inference_client()
                                self._publish_inference_phase(
                                    InferenceStatus.READY, error=str(e)
                                )
                            except Exception:
                                pass

                    threading.Thread(
                        target=_load_and_start,
                        daemon=True,
                        name=f'inference-load-{service_prefix.strip("/")}',
                    ).start()

                    response.success = True
                    response.message = (
                        f'{service_prefix.strip("/").upper()} inference loading'
                    )

            elif request.command == SendCommand.Request.CONVERT_MP4:
                # CONVERT_MP4 path resolution stays orchestrator-side
                # (knows /workspace/rosbag2, robot_type, UI-supplied
                # source_folders). The worker itself lives in
                # cyclo_data.ConversionService — Part C2e migration.
                task_info = request.task_info
                task_name = task_info.task_name
                source_folders = [s for s in task_info.task_instruction if s.strip()]

                base_path = Path('/workspace/rosbag2')
                dataset_path = base_path / task_name
                robot_config_path = os.path.join(
                    get_package_share_directory('shared'),
                    'robot_configs', f'{self.robot_type}_config.yaml'
                )

                if len(source_folders) >= 2:
                    # Merge-then-convert mode — resolve each source.
                    source_paths = []
                    for folder_name in source_folders:
                        src = Path(folder_name) if folder_name.startswith('/') \
                            else base_path / folder_name
                        if not src.exists():
                            response.success = False
                            response.message = f'Source folder not found: {src}'
                            return response
                        source_paths.append(str(src))

                    if dataset_path.exists():
                        response.success = False
                        response.message = \
                            f'Output folder already exists: {dataset_path}'
                        return response
                else:
                    # Single-convert mode — dataset_path must exist.
                    source_paths = []
                    if not dataset_path.exists():
                        response.success = False
                        response.message = \
                            f'Dataset path does not exist: {dataset_path}'
                        return response

                # Read conversion-only knobs off SendCommand.srv (defaulting
                # for old UIs that don't yet send these fields). When neither
                # format flag is set we run both — preserves the pre-D17
                # behaviour where every conversion produced both v2.1 + v3.0.
                conversion_fps = int(getattr(request, 'conversion_fps', 0) or 0)
                convert_v21 = bool(getattr(request, 'convert_v21', False))
                convert_v30 = bool(getattr(request, 'convert_v30', False))
                if not convert_v21 and not convert_v30:
                    convert_v21 = True
                    convert_v30 = True

                # Unpack selection-knob parallel arrays into dicts /
                # tuple before forwarding. cyclo_data_client repacks
                # them into the StartConversion srv's parallel arrays —
                # we keep the wire shape for the cross-node hop only.
                rot_keys = list(getattr(request, 'camera_rotation_keys', []) or [])
                rot_values = list(getattr(request, 'camera_rotation_values', []) or [])
                ui_rotations = {
                    rot_keys[i]: int(rot_values[i])
                    for i in range(min(len(rot_keys), len(rot_values)))
                    if rot_keys[i]
                }
                # Phase 4: yaml's observation.images.<cam>.rotation_deg is
                # the source of truth for default rotations. UI overrides
                # win per-key — when the user opens the convert panel and
                # explicitly picks a value (including 0) it ships in
                # ui_rotations; cameras the user didn't touch fall back to
                # the yaml default.
                yaml_rotations: dict = {}
                if self.robot_section is not None:
                    for cam, cfg in robot_schema.get_image_topics(
                        self.robot_section
                    ).items():
                        deg = cfg.get('rotation_deg', 0)
                        if deg:
                            yaml_rotations[cam] = int(deg)
                camera_rotations = {**yaml_rotations, **ui_rotations}
                resize_h = int(getattr(request, 'image_resize_height', 0) or 0)
                resize_w = int(getattr(request, 'image_resize_width', 0) or 0)
                image_resize = (
                    (resize_h, resize_w)
                    if resize_h > 0 and resize_w > 0 else None
                )

                result = self._cyclo_data.start_conversion(
                    dataset_path=str(dataset_path),
                    robot_type=self.robot_type,
                    robot_config_path=robot_config_path,
                    source_folders=source_paths,
                    fps=conversion_fps,
                    convert_v21=convert_v21,
                    convert_v30=convert_v30,
                    selected_cameras=list(
                        getattr(request, 'selected_cameras', []) or []
                    ),
                    camera_rotations=camera_rotations,
                    image_resize=image_resize,
                    selected_state_topics=list(
                        getattr(request, 'selected_state_topics', []) or []
                    ),
                    selected_action_topics=list(
                        getattr(request, 'selected_action_topics', []) or []
                    ),
                    selected_joints=list(
                        getattr(request, 'selected_joints', []) or []
                    ),
                )
                if not result.success or result.response is None:
                    response.success = False
                    response.message = result.message or \
                        'cyclo_data /data/convert failed.'
                else:
                    cd_response = result.response
                    response.success = bool(cd_response.success)
                    response.message = (
                        getattr(cd_response, 'message', '') or result.message
                    )
                    if cd_response.success:
                        self.get_logger().info(
                            f'MP4 conversion started for: {dataset_path} '
                            f'(job_id={getattr(cd_response, "job_id", "")})'
                        )

            else:
                # Snapshot both flags together — the inference daemon
                # thread flips them in pairs, so reading them
                # independently can branch on a half-transition.
                snapshot_on_recording, snapshot_on_inference = (
                    self._snapshot_session_state()
                )
                if not snapshot_on_recording and not snapshot_on_inference:
                    # Not recording — CANCEL/RERECORD have nothing to
                    # do at idle. Forward anyway so cyclo_data's
                    # handler can publish the umbrella status response
                    # consistently with the recording paths below.
                    if request.command in (
                        SendCommand.Request.CANCEL,
                        SendCommand.Request.RERECORD,
                    ):
                        cd_result = self._cyclo_data.send_recording_command(
                            command=RecordingCommand.Request.CANCEL,
                            task_info=request.task_info,
                            robot_type=self.robot_type,
                        )
                        self._apply_cyclo_data_response(cd_result, response)
                    else:
                        response.success = False
                        response.message = 'Not currently recording'
                else:
                    if request.command == SendCommand.Request.STOP:
                        self.get_logger().info('Stopping and saving recording (forwarder)')
                        cd_result = self._cyclo_data.send_recording_command(
                            command=RecordingCommand.Request.STOP,
                            task_info=request.task_info,
                            robot_type=self.robot_type,
                            urdf_path=(
                                self.params.get('urdf_path', '')
                                if self.params else ''
                            ),
                        )
                        self._apply_cyclo_data_response(cd_result, response)

                    elif request.command == SendCommand.Request.MOVE_TO_NEXT:
                        self.get_logger().info('Saving current episode (forwarder)')
                        cd_result = self._cyclo_data.send_recording_command(
                            command=RecordingCommand.Request.MOVE_TO_NEXT,
                            task_info=request.task_info,
                            robot_type=self.robot_type,
                            urdf_path=(
                                self.params.get('urdf_path', '')
                                if self.params else ''
                            ),
                        )
                        self._apply_cyclo_data_response(cd_result, response)

                    elif request.command == SendCommand.Request.RERECORD:
                        # Stop and save the current episode (no review
                        # flag — that field was removed); orchestrator
                        # still owns inference teardown + timer_manager.
                        self.get_logger().info('Cancelling current recording (forwarder)')
                        cd_result = self._cyclo_data.send_recording_command(
                            command=RecordingCommand.Request.RERECORD,
                            task_info=request.task_info,
                            robot_type=self.robot_type,
                            urdf_path=(
                                self.params.get('urdf_path', '')
                                if self.params else ''
                            ),
                        )
                        if (cd_result.success
                                and cd_result.response is not None
                                and cd_result.response.success):
                            # Inference teardown stays orchestrator-side.
                            self._teardown_inference_client()
                            self._set_session_active(
                                on_recording=False, on_inference=False,
                            )
                            if self.timer_manager:
                                self.timer_manager.stop(timer_name='collection')
                            response.success = True
                            response.message = (
                                cd_result.response.message or 'Recording cancelled'
                            )
                        else:
                            self._apply_cyclo_data_response(cd_result, response)

                    elif request.command == SendCommand.Request.STOP_INFERENCE:
                        with self._state_lock:
                            client = self.container_service_client
                        if client is not None:
                            result = client.inference_command(
                                ContainerServiceClient.CMD_PAUSE,
                            )
                            if result.success:
                                self._publish_inference_phase(InferenceStatus.PAUSED)
                            response.success = result.success
                            response.message = result.message or 'Inference paused'
                        else:
                            response.success = False
                            response.message = 'No inference session active'

                    elif request.command == SendCommand.Request.RESUME_INFERENCE:
                        with self._state_lock:
                            client = self.container_service_client
                        if client is not None:
                            task_instruction = (
                                request.task_info.task_instruction[0]
                                if request.task_info.task_instruction
                                else ''
                            )
                            result = client.inference_command(
                                ContainerServiceClient.CMD_RESUME,
                                task_instruction=task_instruction,
                            )
                            if result.success:
                                self._publish_inference_phase(InferenceStatus.INFERENCING)
                            response.success = result.success
                            response.message = result.message or 'Inference resumed'
                        else:
                            response.success = False
                            response.message = 'No inference session active'

                    elif request.command == SendCommand.Request.UPDATE_INSTRUCTION:
                        # Mid-run language re-conditioning. Lifecycle stays
                        # at INFERENCING — no inference_phase publish.
                        with self._state_lock:
                            client = self.container_service_client
                        if client is not None:
                            task_instruction = (
                                request.task_info.task_instruction[0]
                                if request.task_info.task_instruction
                                else ''
                            )
                            result = client.inference_command(
                                ContainerServiceClient.CMD_UPDATE_INSTRUCTION,
                                task_instruction=task_instruction,
                            )
                            response.success = result.success
                            response.message = (
                                result.message or 'Instruction updated'
                            )
                        else:
                            response.success = False
                            response.message = 'No inference session active'

                    elif request.command == SendCommand.Request.START_INFERENCE_RECORD:
                        self.get_logger().info(
                            'Starting recording during inference (forwarder)')
                        cd_result = self._forward_recording(
                            RecordingCommand.Request.START,
                            task_info=request.task_info,
                            include_topics=True,
                        )
                        if (cd_result.success
                                and cd_result.response is not None
                                and cd_result.response.success):
                            self._set_session_active(
                                on_recording=True,
                                start_time=time.perf_counter(),
                            )
                            response.success = True
                            response.message = (
                                cd_result.response.message
                                or 'Recording started during inference'
                            )
                        else:
                            self._apply_cyclo_data_response(cd_result, response)

                    elif request.command == SendCommand.Request.STOP_INFERENCE_RECORD:
                        self.get_logger().info('Stopping recording during inference (forwarder)')
                        cd_result = self._forward_recording(
                            RecordingCommand.Request.STOP,
                            task_info=request.task_info,
                        )
                        if (cd_result.success
                                and cd_result.response is not None
                                and cd_result.response.success):
                            self._set_session_active(on_recording=False)
                            response.success = True
                            response.message = (
                                cd_result.response.message or 'Recording saved'
                            )
                        else:
                            self._apply_cyclo_data_response(cd_result, response)

                    elif request.command == SendCommand.Request.CANCEL_INFERENCE_RECORD:
                        # Inference page's Record-Discard — drop the
                        # episode entirely (no save). Same semantics as
                        # the record page's Discard, just forwarded
                        # under a different SendCommand so the orchestrator
                        # leaves the inference session alive.
                        self.get_logger().info('Discarding recording during inference (forwarder)')
                        cd_result = self._forward_recording(
                            RecordingCommand.Request.CANCEL,
                            task_info=request.task_info,
                        )
                        if (cd_result.success
                                and cd_result.response is not None
                                and cd_result.response.success):
                            self._set_session_active(on_recording=False)
                            response.success = True
                            response.message = (
                                cd_result.response.message or 'Recording discarded'
                            )
                        else:
                            self._apply_cyclo_data_response(cd_result, response)

                    elif request.command == SendCommand.Request.FINISH:
                        # Two UI buttons land here, with different intent:
                        #
                        #   * Record page "Save"  → task_type='record'.
                        #     Stop recording only, leave any active
                        #     inference session alone. (Old behaviour
                        #     tore down inference too, which closed
                        #     RobotClient mid-session when recording was
                        #     started from the record page during
                        #     inference.)
                        #
                        #   * Inference page "Clear" → task_type='inference'.
                        #     Tooltip is "Stop inference and unload model";
                        #     this is the inference end-of-session path
                        #     and must STOP + UNLOAD + disconnect the
                        #     container client.
                        is_inference_clear = (
                            request.task_info.task_type == 'inference'
                        )
                        self.get_logger().info(
                            'Finishing '
                            f'{"inference session" if is_inference_clear else "recording"} '
                            '(forwarder)'
                        )
                        cd_result = self._forward_recording(
                            RecordingCommand.Request.FINISH,
                            task_info=request.task_info,
                        )
                        if is_inference_clear:
                            self._teardown_inference_client()
                            self._set_session_active(
                                on_recording=False, on_inference=False,
                            )
                            if self.timer_manager:
                                self.timer_manager.stop(timer_name='collection')
                            # Flip UI out of INFERENCING/PAUSED immediately.
                            self._publish_inference_phase(InferenceStatus.READY)
                            response.success = True
                            response.message = (
                                cd_result.response.message
                                if (cd_result.response is not None)
                                else (cd_result.message or 'Inference cleared')
                            )
                        else:
                            if (cd_result.success
                                    and cd_result.response is not None
                                    and cd_result.response.success):
                                self._set_session_active(on_recording=False)
                                response.success = True
                                response.message = (
                                    cd_result.response.message
                                    or 'Recording saved'
                                )
                            else:
                                self._apply_cyclo_data_response(
                                    cd_result, response)

                    elif request.command == SendCommand.Request.SKIP_TASK:
                        # Simplified-mode semantics: skip == stop-and-save
                        # plus tear down inference state.
                        self.get_logger().info('Skipping current recording (forwarder)')
                        cd_result = self._forward_recording(
                            RecordingCommand.Request.RERECORD,
                            task_info=request.task_info,
                        )
                        self._teardown_inference_client()
                        self._set_session_active(
                            on_recording=False, on_inference=False,
                        )
                        if self.timer_manager:
                            self.timer_manager.stop(timer_name='collection')
                        response.success = True
                        response.message = (
                            cd_result.response.message
                            if (cd_result.response is not None)
                            else (cd_result.message or 'Recording cancelled')
                        )

                    elif request.command == SendCommand.Request.CANCEL:
                        # Record-page Discard — drop the active episode
                        # entirely (no needs_review save). Same task_type
                        # split as FINISH so an inference session running
                        # alongside the recording isn't torn down.
                        is_inference_cancel = (
                            request.task_info.task_type == 'inference'
                        )
                        self.get_logger().info(
                            'Discarding current recording '
                            f'({"inference session" if is_inference_cancel else "record-only"})'
                        )
                        cd_result = self._forward_recording(
                            RecordingCommand.Request.CANCEL,
                            task_info=request.task_info,
                        )
                        if is_inference_cancel:
                            self._teardown_inference_client()
                            self._set_session_active(
                                on_recording=False, on_inference=False,
                            )
                            if self.timer_manager:
                                self.timer_manager.stop(timer_name='collection')
                        else:
                            self._set_session_active(on_recording=False)
                        response.success = True
                        response.message = (
                            cd_result.response.message
                            if (cd_result.response is not None)
                            else (cd_result.message or 'Recording cancelled')
                        )

        except Exception as e:
            self.get_logger().error(f'Error in user interaction: {str(e)}')
            response.success = False
            response.message = f'Error in user interaction: {str(e)}'
            return response
        return response

    def get_robot_types_callback(self, request, response):
        if self.robot_type_list is None:
            self.get_logger().error('Robot type list is not set')
            response.robot_types = []
            response.success = False
            response.message = 'Robot type list is not set'
            return response

        self.get_logger().info(f'Available robot types: {self.robot_type_list}')
        response.robot_types = self.robot_type_list
        response.success = True
        response.message = 'Robot type list retrieved successfully'
        return response

    def get_robot_info_callback(self, request, response):
        if self.robot_section is None:
            response.robot_type = ''
            response.robot_name = ''
            response.urdf_path = ''
            response.success = False
            response.message = 'Robot type has not been set yet'
            return response

        response.robot_type = getattr(self, 'robot_type', '') or ''
        response.robot_name = robot_schema.get_robot_name(self.robot_section)
        response.urdf_path = robot_schema.get_urdf_path(self.robot_section)
        response.success = True
        response.message = 'Robot info retrieved successfully'
        return response

    def get_available_list_callback(self, request, response):
        response.success = True
        response.message = 'Policy and device lists retrieved successfully'
        response.policy_list, response.device_list = ZenohTrainingManager.get_available_list()
        return response

    def get_user_list_callback(self, request, response):
        try:
            if not self.DEFAULT_SAVE_ROOT_PATH.exists():
                response.user_list = []
                response.success = False
                response.message = f'Path {self.DEFAULT_SAVE_ROOT_PATH} does not exist.'
                return response

            folder_names = [
                name for name in os.listdir(self.DEFAULT_SAVE_ROOT_PATH)
                if (self.DEFAULT_SAVE_ROOT_PATH / name).is_dir()
            ]

            response.user_list = folder_names
            response.success = True
            response.message = f'Found {len(folder_names)} user(s).'

        except Exception as e:
            response.user_list = []
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_dataset_list_callback(self, request, response):
        user_id = request.user_id
        user_path = self.DEFAULT_SAVE_ROOT_PATH / user_id

        try:
            if not user_path.exists() or not user_path.is_dir():
                response.dataset_list = []
                response.success = False
                response.message = f"User ID '{user_id}' does not exist at path: {user_path}"
                return response

            dataset_names = [
                name for name in os.listdir(user_path)
                if (user_path / name).is_dir()
            ]

            response.dataset_list = dataset_names
            response.success = True
            response.message = f"Found {len(dataset_names)} dataset(s) for user '{user_id}'."

        except Exception as e:
            response.dataset_list = []
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_model_weight_list_callback(self, request, response):
        save_root_path = ZenohTrainingManager.get_weight_save_root_path()
        try:
            if not save_root_path.exists():
                response.success = False
                response.message = f'Path does not exist: {save_root_path}'
                response.model_weight_list = []
                return response

            model_folders = [
                f.name for f in save_root_path.iterdir()
                if f.is_dir()
            ]

            response.success = True
            response.message = f'Found {len(model_folders)} model weights'
            response.model_weight_list = model_folders

        except Exception as e:
            response.success = False
            response.message = f'Error: {str(e)}'
            response.model_weight_list = []

        return response

    def get_training_info_callback(self, request, response):
        """
        Retrieve training configuration from a saved model.

        Loads configuration from train_config.json and populates TrainingInfo message.
        """
        try:
            # Validate request
            if not request.train_config_path:
                response.success = False
                response.message = 'train_config_path is required'
                return response

            # Clean up path (remove leading/trailing whitespace)
            train_config_path = request.train_config_path.strip()
            weight_save_root_path = ZenohTrainingManager.get_weight_save_root_path()
            config_path = weight_save_root_path / train_config_path

            # Check if config file exists
            if not config_path.exists():
                response.success = False
                response.message = f'Model config file not found: {config_path}'
                return response

            # Load and parse configuration
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                self.get_logger().info(f'Successfully loaded config from: {config_path}')

                # Populate TrainingInfo message from config
                training_info = response.training_info

                # Dataset configuration
                dataset_config = config_data.get('dataset', {})
                training_info.dataset = dataset_config.get('repo_id', '')

                # Policy configuration
                policy_config = config_data.get('policy', {})
                training_info.policy_type = policy_config.get('type', '')
                training_info.policy_device = policy_config.get('device', 'cuda')

                # Output directory (extract folder name)
                output_dir = config_data.get('output_dir', '')
                if output_dir:
                    training_info.output_folder_name = Path(output_dir).name
                else:
                    training_info.output_folder_name = ''

                # Training parameters with defaults
                training_info.seed = config_data.get('seed', 1000)
                training_info.num_workers = config_data.get('num_workers', 4)
                training_info.batch_size = config_data.get('batch_size', 8)
                training_info.steps = config_data.get('steps', 100000)
                training_info.eval_freq = config_data.get('eval_freq', 20000)
                training_info.log_freq = config_data.get('log_freq', 200)
                training_info.save_freq = config_data.get('save_freq', 1000)

                response.success = True
                response.message = \
                    f'Training configuration loaded successfully from {train_config_path}'

            except json.JSONDecodeError as e:
                response.success = False
                response.message = f'Invalid JSON in config file: {str(e)}'
                return response
            except KeyError as e:
                response.success = False
                response.message = f'Missing required field in config: {str(e)}'
                return response

        except Exception as e:
            self.get_logger().error(f'Error in get_training_info_callback: {str(e)}')
            response.success = False
            response.message = f'Failed to retrieve training info: {str(e)}'

        return response

    def set_robot_type_callback(self, request, response):
        try:
            self.get_logger().info(f'Setting robot type to: {request.robot_type}')
            self.robot_type = request.robot_type
            self.clear_parameters()
            self.init_ros_params(self.robot_type)

            # Prepare rosbag subscriptions on the cyclo_data side so the
            # first recording command starts without warmup delay. Part
            # C2d-5 — prior versions called self.communicator.prepare_rosbag
            # directly; that path now lives behind RecordingCommand.REFRESH_TOPICS.
            if self.communicator is not None:
                rosbag_topics = self.communicator.get_mcap_topics()
                cd_result = self._cyclo_data.send_recording_command(
                    command=RecordingCommand.Request.REFRESH_TOPICS,
                    robot_type=self.robot_type,
                    topics=rosbag_topics,
                )
                if cd_result.success and cd_result.response is not None \
                        and cd_result.response.success:
                    self.get_logger().info(
                        f'Rosbag prepared via cyclo_data '
                        f'({len(rosbag_topics)} topics) — ready for recording')
                else:
                    self.get_logger().warn(
                        'cyclo_data REFRESH_TOPICS forward failed on set_robot_type — '
                        f'{cd_result.message or "no response"}')
            else:
                self.get_logger().warn('Communicator not initialized — prepare skipped')

            response.success = True
            response.message = f'Robot type set to {self.robot_type}'
            return response

        except Exception as e:
            self.get_logger().error(f'Failed to set robot type: {str(e)}')
            response.success = False
            response.message = f'Failed to set robot type: {str(e)}'
            return response

    def get_replay_data_callback(self, request, response):
        """Handle replay data request for viewing recorded ROSbag data."""
        try:
            bag_path = request.bag_path
            self.get_logger().info(f'Getting replay data for: {bag_path}')

            result = self.replay_data_handler.get_replay_data(bag_path)

            response.success = result.get('success', False)
            response.message = result.get('message', '')

            if response.success:
                response.video_files = result.get('video_files', [])
                response.video_topics = result.get('video_topics', [])
                response.video_fps = result.get('video_fps', [])
                response.frame_indices = result.get('frame_indices', [])
                response.frame_timestamps = result.get('frame_timestamps', [])
                response.joint_timestamps = result.get('joint_timestamps', [])
                response.joint_names = result.get('joint_names', [])
                response.joint_positions = result.get('joint_positions', [])
                response.action_timestamps = result.get('action_timestamps', [])
                response.action_names = result.get('action_names', [])
                response.action_values = result.get('action_values', [])
                response.start_time = result.get('start_time', 0.0)
                response.end_time = result.get('end_time', 0.0)
                response.duration = result.get('duration', 0.0)
                response.video_server_port = self.VIDEO_SERVER_PORT
                response.bag_path = bag_path

            return response

        except Exception as e:
            self.get_logger().error(f'Error in get_replay_data_callback: {str(e)}')
            response.success = False
            response.message = f'Error getting replay data: {str(e)}'
            return response

    def browse_file_callback(self, request, response):
        """Handle file browsing requests."""
        try:
            if request.action == 'get_path':
                result = self.file_browse_utils.handle_get_path_action(
                    request.current_path)
            elif request.action == 'go_parent':
                target_files = None
                target_folders = None

                if hasattr(request, 'target_files') and request.target_files:
                    target_files = set(request.target_files)
                if hasattr(request, 'target_folders') and request.target_folders:
                    target_folders = set(request.target_folders)

                if target_files or target_folders:
                    result = self.file_browse_utils.handle_go_parent_with_target_check(
                        request.current_path,
                        target_files,
                        target_folders)
                else:
                    result = self.file_browse_utils.handle_go_parent_action(
                        request.current_path)
            elif request.action == 'browse':
                target_files = None
                target_folders = None

                if hasattr(request, 'target_files') and request.target_files:
                    target_files = set(request.target_files)
                if hasattr(request, 'target_folders') and request.target_folders:
                    target_folders = set(request.target_folders)

                if target_files or target_folders:
                    result = self.file_browse_utils.handle_browse_with_target_check(
                        request.current_path,
                        request.target_name,
                        target_files,
                        target_folders)
                else:
                    result = self.file_browse_utils.handle_browse_action(
                        request.current_path, request.target_name)
            else:
                result = {
                    'success': False,
                    'message': f'Unknown action: {request.action}',
                    'current_path': '',
                    'parent_path': '',
                    'selected_path': '',
                    'items': []
                }

            response.success = result['success']
            response.message = result['message']
            response.current_path = result['current_path']
            response.parent_path = result['parent_path']
            response.selected_path = result['selected_path']

            response.items = []
            for item_dict in result['items']:
                item = BrowserItem()
                item.name = item_dict['name']
                item.full_path = item_dict['full_path']
                item.is_directory = item_dict['is_directory']
                item.size = item_dict['size']
                item.modified_time = item_dict['modified_time']
                item.has_target_file = item_dict.get('has_target_file', False)
                response.items.append(item)

            return response

        except Exception as e:
            self.get_logger().error(f'Error in browse_file_callback: {str(e)}')
            response.success = False
            response.message = f'Error: {str(e)}'
            response.current_path = ''
            response.parent_path = ''
            response.selected_path = ''
            response.items = []
            return response

    # LeRobot policy types (used for service_prefix detection)
    LEROBOT_POLICIES = {
        'tdmpc', 'diffusion', 'act', 'vqbet', 'pi0', 'pi0_fast', 'pi05',
        'smolvla', 'xvla', 'sac',
    }

    def _determine_service_prefix(self, task_info) -> str:
        """Determine inference service prefix from task_info or policy config.

        1. If task_info has service_type field, use it directly.
        2. Otherwise, read policy_path/config.json to detect policy type.
        3. LeRobot policy types -> "/lerobot", default -> "/groot".
        """
        # Check for explicit service_type in task_info
        service_type = getattr(task_info, 'service_type', None)
        if service_type:
            prefix = f'/{service_type.strip("/")}'
            self.get_logger().info(f'Service prefix from task_info: {prefix}')
            return prefix

        # Detect from policy config. LeRobot training output nests the
        # checkpoint under <root>/pretrained_model/ — try that path too
        # so users who paste the training root still get the right routing.
        policy_path = getattr(task_info, 'policy_path', '')
        if policy_path:
            root = Path(policy_path)
            config_path = root / 'config.json'
            if not config_path.exists() and (root / 'pretrained_model' / 'config.json').exists():
                config_path = root / 'pretrained_model' / 'config.json'
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    policy_type = config.get('type', '')
                    if policy_type in self.LEROBOT_POLICIES:
                        self.get_logger().info(
                            f'Detected LeRobot policy type: {policy_type}'
                        )
                        return '/lerobot'
                except Exception as e:
                    self.get_logger().warning(
                        f'Failed to read policy config: {e}'
                    )

        # Default to groot for backward compatibility
        return '/groot'

    def _teardown_inference_client(self):
        """Tear down the container service client (STOP + UNLOAD + disconnect).

        Called on inference session end (FINISH), on LOAD/START failure so
        the container is left in a clean state, and before starting a new
        session against a different service prefix. Non-blocking — the
        UNLOAD call happens on a background thread so UI keeps responding
        while CUDA memory releases.
        """
        # Atomic swap: detach the client under the lock so concurrent
        # callers (RESUME path, joystick handler, daemon thread) can't
        # both grab the same client and double-disconnect.
        with self._state_lock:
            client = self.container_service_client
            self.container_service_client = None
        if client is None:
            return

        def _cleanup():
            try:
                client.inference_command(ContainerServiceClient.CMD_STOP)
                client.inference_command(ContainerServiceClient.CMD_UNLOAD)
            except Exception as e:
                self.get_logger().error(f'Error tearing down inference: {e}')
            finally:
                try:
                    client._cancelled.set()
                    client.disconnect()
                except Exception:
                    pass

        threading.Thread(target=_cleanup, daemon=True).start()

    def handle_joystick_trigger(self, joystick_mode: str):
        """
        Handle joystick trigger for simplified recording control.

        - Right button: Toggle Start/Finish
          - If no session: Auto-create session and start recording
          - If idle: Start recording
          - If recording: Finish and save
        - Left button: Cancel (only during recording)
          - Discards current recording
        """
        self.get_logger().info(f'Joystick trigger: {joystick_mode}')

        # Joystick operations forward to cyclo_data (Part C2d-5). orchestrator
        # only tracks on_recording / timer_manager session state; DataManager
        # + rosbag + action_event live in cyclo_data.
        snapshot_on_recording, _ = self._snapshot_session_state()
        if joystick_mode == 'right':
            if not snapshot_on_recording:
                if self._last_ui_task_info is None:
                    # No task_info yet — auto-create path needs a cached
                    # task_info to derive folder naming. Match the prior
                    # behaviour's hard error.
                    self.get_logger().error(
                        'Joystick right ignored: start the first episode from '
                        'the UI so task info (Task Num / Task Name) is set.')
                    return
                self.get_logger().info('Right button: Starting recording (forwarder)')
                # Re-init params (joint_order etc.) for the cached task. Cheap
                # if the task hasn't changed; sets up the 'collection' timer
                # slot every time so .start() below succeeds.
                self.init_robot_control_parameters_from_user_task(
                    self._last_ui_task_info)
                if self.timer_manager:
                    self.timer_manager.start(timer_name='collection')
                cd_result = self._forward_recording(
                    RecordingCommand.Request.START,
                    task_info=self._last_ui_task_info,
                    include_topics=True,
                )
                if (cd_result.success
                        and cd_result.response is not None
                        and cd_result.response.success):
                    self._set_session_active(
                        on_recording=True,
                        start_time=time.perf_counter(),
                    )
                else:
                    self.get_logger().error(
                        f'Joystick START forward failed: '
                        f'{cd_result.response.message if cd_result.response else cd_result.message}'
                    )
            else:
                self.get_logger().info('Right button: Finishing recording (forwarder)')
                self._forward_recording(
                    RecordingCommand.Request.STOP,
                    task_info=self._last_ui_task_info,
                )
                # Match the orchestrator's prior behaviour — STOP keeps
                # on_recording True so subsequent right-clicks still toggle
                # correctly. The UI can send FINISH when it wants to end
                # the session.

        elif joystick_mode == 'left':
            # cyclo_data's CANCEL handler picks the right mode: cancel-with-
            # review if actively recording, toggle-review otherwise. Also
            # publishes the right action_event (cancel / review_on / review_off).
            self.get_logger().info('Left button: forwarding CANCEL')
            self._forward_recording(
                RecordingCommand.Request.CANCEL,
                task_info=self._last_ui_task_info,
            )

        elif joystick_mode == 'right_long_time':
            self.get_logger().info('Right long press - reserved for future use')

        elif joystick_mode == 'left_long_time':
            self.get_logger().info('Left long press - reserved for future use')

        else:
            self.get_logger().info(f'Unknown joystick trigger: {joystick_mode}')

    # _auto_create_recording_session removed in Step 3 Part C2d-5 — the
    # joystick handler now reuses self._last_ui_task_info + forwards START
    # to cyclo_data directly via _forward_recording().

    # HF API Worker lifecycle + /huggingface/status publisher moved to
    # cyclo_data.HubService (Step 3 Part C2c). The UI-facing
    # /huggingface/control forwarder was retired in Part C2c-ui — the
    # UI now calls /data/hub (HfOperation) directly. HFEndpointStore
    # stays here (set/get/list/select_hf_endpoint services mutate it);
    # cyclo_data reads the same on-disk store for its own token
    # resolution on direct /data/hub calls.

    # MP4 Conversion Worker and the 2 Hz status poll moved to
    # cyclo_data.ConversionService (Step 3 Part C2e). Conversion progress
    # reaches the UI as record_phase=CONVERTING on /data/recording/status
    # (cyclo_data → UI direct, D18).


def main(args=None):
    rclpy.init(args=args)

    node = OrchestratorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Both HF (Part C2c) and MP4 conversion (Part C2e) workers live
        # in cyclo_data now — nothing orchestrator-side to tear down here.
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
