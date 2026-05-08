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
# Author: Dongyun Kim, Seongwoo Kim, Kiwoong Park

from typing import Any, Callable, Dict, List, Optional

from interfaces.msg import (
    BrowserItem,
    DatasetInfo,
    InferenceStatus
)
from interfaces.srv import (
    BrowseFile,
    GetDatasetInfo,
    GetImageTopicList,
)
from cyclo_data.editor.episode_editor import DataEditor
from orchestrator.internal.file_browser.file_browse_utils import FileBrowseUtils
from shared.robot_configs import schema as robot_schema
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy
)
from std_msgs.msg import Empty, String


class Communicator:
    """
    Communicator class for rosbag2-only data acquisition.

    This class manages:
    - Rosbag2 recording (start/stop/save)
    - Status publishing
    - Joystick trigger for control
    - Services for file browsing and dataset editing

    Note: Camera/Joint data subscription removed - rosbag2 records directly from topics.
    """

    PUB_QOS_SIZE = 100

    def __init__(
        self,
        node: Node,
        robot_section: Dict[str, Any],
    ):
        self.node = node
        self.robot_section = robot_section
        self.file_browse_utils = FileBrowseUtils(
            max_workers=8,
            logger=self.node.get_logger())

        # Build the rosbag topic inventory from the new VLA-semantic schema.
        # Image / state / action topics are recorded directly; extras carry
        # /tf + camera_info (data-conversion concern, not training input).
        image_groups = robot_schema.get_image_topics(robot_section)
        state_groups = robot_schema.get_state_groups(robot_section)
        action_groups = robot_schema.get_action_groups(robot_section)

        self.camera_topics: Dict[str, str] = {
            name: cfg['topic'] for name, cfg in image_groups.items()
        }
        self.state_topics: Dict[str, str] = {
            name: cfg['topic'] for name, cfg in state_groups.items()
        }
        self.action_topics: Dict[str, str] = {
            modality: cfg['topic'] for modality, cfg in action_groups.items()
        }
        self.rosbag_extra_topics = robot_schema.get_recording_extra_topics(
            robot_section
        )
        self._all_topics = robot_schema.get_recording_topics(robot_section)

        # Initialize DataEditor for dataset editing
        self.data_editor = DataEditor()

        # Log topic information
        node.get_logger().info(f'Camera topics: {self.camera_topics}')
        node.get_logger().info(f'State topics: {self.state_topics}')
        node.get_logger().info(f'Action topics: {self.action_topics}')
        node.get_logger().info(f'Rosbag extra topics: {self.rosbag_extra_topics}')

        self.heartbeat_qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.init_subscribers()
        self.init_publishers()
        self.init_services()

        self.joystick_state = {
            'updated': False,
            'mode': None
        }

        # Joystick handler callback for immediate processing
        self._joystick_handler: Optional[Callable[[str], None]] = None

    def get_all_topics(self):
        """Full rosbag topic inventory: images + state + action + extras."""
        return list(self._all_topics)

    def init_subscribers(self):
        """Initialize only joystick trigger subscriber."""
        # Joystick trigger for control (keep this)
        self.joystick_trigger_subscriber = self.node.create_subscription(
            String,
            '/leader/joystick_controller/tact_trigger',
            self.joystick_trigger_callback,
            10
        )
        self.node.get_logger().info('Joystick trigger subscriber initialized')

    def init_publishers(self):
        """Initialize publishers."""
        self.node.get_logger().info('Initializing publishers...')

        # Inference status publisher — orchestrator owns the inference phase
        # half of the split (record half lives on /data/recording/status,
        # published by cyclo_data). See ~/.claude/plans/record-zippy-sunrise.md
        # and PLAN §10.3 D18.
        self.inference_status_publisher = self.node.create_publisher(
            InferenceStatus,
            '/task/inference_status',
            self.PUB_QOS_SIZE
        )

        # /task/action_event publisher moved to cyclo_data.recorder.rosbag_control
        # in Step 3 Part C2d-1/-5. cyclo_data owns the recording lifecycle.

        # /inference/trajectory_preview publisher moved to groot Process B
        # (control_publisher.py). Post-D16 split, action chunks flow
        # Process A → Process B over Zenoh; orchestrator no longer sees
        # them, so it can't republish.

        # Heartbeat publisher
        self.heartbeat_publisher = self.node.create_publisher(
            Empty,
            'heartbeat',
            self.heartbeat_qos_profile)

        self.node.get_logger().info('Publishers initialized')

    def init_services(self):
        """Initialize services."""
        self.image_topic_list_service = self.node.create_service(
            GetImageTopicList,
            '/image/get_available_list',
            self.get_image_topic_list_callback
        )

        self.file_browser_service = self.node.create_service(
            BrowseFile,
            '/browse_file',
            self.browse_file_callback
        )

        self.get_dataset_info_service = self.node.create_service(
            GetDatasetInfo,
            '/dataset/get_info',
            self.get_dataset_info_callback
        )

    # Rosbag command client + prepare/start/stop/finish/stop_and_delete_rosbag
    # and _send_rosbag_command moved to cyclo_data.recorder.rosbag_control in
    # Step 3 Part C2d-1/-5. Orchestrator forwards via CycloDataClient; the
    # rosbag service topic is now only contacted by the cyclo_data node.

    # ========== Publishers ==========

    def publish_inference_status(
        self,
        phase: int,
        robot_type: str = '',
        error: str = '',
    ) -> None:
        """Publish an InferenceStatus snapshot on /task/inference_status."""
        msg = InferenceStatus()
        msg.inference_phase = phase
        msg.robot_type = robot_type
        msg.error = error
        self.inference_status_publisher.publish(msg)

    # publish_action_event moved to cyclo_data.recorder.rosbag_control.RosbagControl
    # (Step 3 Part C2d-1/-5). Orchestrator no longer owns /task/action_event.

    # ========== Joystick Handler ==========

    def register_joystick_handler(self, handler: Callable[[str], None]):
        """
        Register a handler for joystick triggers.

        This allows immediate processing of joystick events without
        waiting for the timer callback.

        Args:
            handler: Callback function that takes joystick mode string
        """
        self._joystick_handler = handler
        self.node.get_logger().info('Joystick handler registered')

    # ========== Callbacks ==========

    def joystick_trigger_callback(self, msg: String):
        """Handle joystick trigger for recording control."""
        self.node.get_logger().info(f'Received joystick trigger: {msg.data}')
        self.joystick_state['updated'] = True
        self.joystick_state['mode'] = msg.data

        # Call registered handler immediately if available
        if self._joystick_handler is not None:
            self._joystick_handler(msg.data)
            # Mark as processed to prevent duplicate handling in timer callback
            self.joystick_state['updated'] = False

    def heartbeat_timer_callback(self):
        """Publish heartbeat."""
        heartbeat_msg = Empty()
        self.heartbeat_publisher.publish(heartbeat_msg)

    # ========== Service Callbacks ==========

    def get_image_topic_list_callback(self, request, response):
        # Walk the yaml's observation.images in insertion order so the
        # topic list and the rotation_deg list stay parallel-indexed.
        # camera_topics keeps the same order (Python 3.7+ dict
        # preserves insertion); going back to image_groups gives us
        # rotation_deg without an extra lookup.
        image_groups = robot_schema.get_image_topics(self.robot_section)
        camera_topic_list: List[str] = []
        rotation_deg_list: List[int] = []
        for cam_name, cfg in image_groups.items():
            # Skip cams that aren't part of the recording inventory
            # (camera_topics is filtered by recording role).
            if cam_name not in self.camera_topics:
                continue
            camera_topic_list.append(cfg['topic'])
            rotation_deg_list.append(int(cfg.get('rotation_deg', 0) or 0))

        if len(camera_topic_list) == 0:
            self.node.get_logger().error('No image topics found')
            response.image_topic_list = []
            response.rotation_deg_list = []
            response.success = False
            response.message = 'Please check image topics in your robot configuration.'
            return response

        response.image_topic_list = camera_topic_list
        response.rotation_deg_list = rotation_deg_list
        response.success = True
        response.message = 'Image topic list retrieved successfully'
        return response

    def browse_file_callback(self, request, response):
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

        except Exception as e:
            self.node.get_logger().error(f'Error in browse file handler: {str(e)}')
            response.success = False
            response.message = f'Error: {str(e)}'
            response.current_path = ''
            response.parent_path = ''
            response.selected_path = ''
            response.items = []

        return response

    def get_dataset_info_callback(self, request, response):
        from pathlib import Path
        try:
            task_dir = Path(request.dataset_path)
            task_info = self.data_editor.get_rosbag_task_info(task_dir)

            info = DatasetInfo()
            info.robot_type = task_info.robot_type
            info.task_instruction = task_info.task_instruction
            info.episode_count = int(task_info.episode_count)
            info.total_duration_s = float(task_info.total_duration_s)
            info.fps = int(task_info.fps)

            response.dataset_info = info
            response.success = True
            response.message = 'Task info retrieved successfully'
            return response

        except Exception as e:
            self.node.get_logger().error(f'Error in get_dataset_info_callback: {str(e)}')
            response.success = False
            response.message = f'Error: {str(e)}'
            response.dataset_info = DatasetInfo()
            return response

    # ========== Cleanup ==========

    def _destroy_service_if_exists(self, service_attr_name: str):
        if hasattr(self, service_attr_name):
            service = getattr(self, service_attr_name)
            if service is not None:
                self.node.destroy_service(service)
                setattr(self, service_attr_name, None)

    def _destroy_client_if_exists(self, client_attr_name: str):
        if hasattr(self, client_attr_name):
            client = getattr(self, client_attr_name)
            if client is not None:
                self.node.destroy_client(client)
                setattr(self, client_attr_name, None)

    def _destroy_publisher_if_exists(self, publisher_attr_name: str):
        if hasattr(self, publisher_attr_name):
            publisher = getattr(self, publisher_attr_name)
            if publisher is not None:
                self.node.destroy_publisher(publisher)
                setattr(self, publisher_attr_name, None)

    def cleanup(self):
        self.node.get_logger().info('Cleaning up Communicator resources...')

        self._cleanup_publishers()
        self._cleanup_subscribers()
        self._cleanup_services()

        self.node.get_logger().info('Communicator cleanup completed')

    def _cleanup_publishers(self):
        publisher_names = [
            'inference_status_publisher',
            'heartbeat_publisher',
        ]
        for publisher_name in publisher_names:
            self._destroy_publisher_if_exists(publisher_name)

    def _cleanup_subscribers(self):
        if hasattr(self, 'joystick_trigger_subscriber') and \
           self.joystick_trigger_subscriber is not None:
            self.node.destroy_subscription(self.joystick_trigger_subscriber)
            self.joystick_trigger_subscriber = None

    def _cleanup_services(self):
        service_names = [
            'image_topic_list_service',
            'file_browser_service',
            'get_dataset_info_service'
        ]
        for service_name in service_names:
            self._destroy_service_if_exists(service_name)

    def _cleanup_clients(self):
        # rosbag_recorder client migrated to cyclo_data.recorder.rosbag_control
        # in Step 3 Part C2d-1/-5.
        client_names: list = []
        for client_name in client_names:
            self._destroy_client_if_exists(client_name)
