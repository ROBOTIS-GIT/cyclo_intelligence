# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CycloDataClient — sync + async wrappers for the cyclo_data node services.

The cyclo_data node (see cyclo_data/cyclo_data_node.py) advertises the
data-plane services under /data/*. This client is the orchestrator-side
counterpart: one object that owns the five clients and offers typed
per-operation methods.

Wiring status (Step 3 Part C is split into three sub-commits):
  * Part C1 (this commit): client class exists; no caller is using it
    yet. Instantiation and first call sites land in Part C2.
  * Part C2: orchestrator_node instantiates CycloDataClient, routes
    UI commands through it, and a /data/status listener relays status
    to the UI. Orchestrator-side duplicate handlers are removed.
  * Part C3: obsolete enum values / legacy srv types are retired.

Design notes:
  * All calls are call_async + future.result(timeout). The service
    bodies in cyclo_data are cheap (validation + enqueue), so a short
    timeout is enough; actual heavy work is observed via
    DataOperationStatus topic.
  * Clients use a ReentrantCallbackGroup by default so the caller may
    live inside another ROS 2 service callback without deadlocking.
  * Missing cyclo_data node is surfaced as (False, "service unavailable")
    rather than raising — caller decides what to show on the UI.
"""

from dataclasses import dataclass
import logging
import threading
from typing import Optional

from interfaces.msg import TaskInfo
from interfaces.srv import (
    EditDataset,
    GetConversionStatus,
    HfOperation,
    RecordingCommand,
    StartConversion,
)

from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node


logger = logging.getLogger(__name__)


SERVICE_NAMES = {
    'recording': '/data/recording',
    'convert': '/data/convert',
    'convert_status': '/data/convert/status',
    'hub': '/data/hub',
    'edit': '/data/edit',
}

DEFAULT_TIMEOUT_SEC = 5.0


@dataclass
class CallResult:
    """Normalised return type — callers never touch ROS 2 Future objects."""

    success: bool
    message: str
    # One of: the raw response object (on success) or None (on failure).
    # Callers inspect it via `result.response.<field>` when they need
    # job_id / status / affected_count / etc.
    response: object = None


class CycloDataClient:

    def __init__(self, node: Node, callback_group: Optional[CallbackGroup] = None):
        self._node = node
        self._cb_group = callback_group or ReentrantCallbackGroup()

        self._recording = node.create_client(
            RecordingCommand, SERVICE_NAMES['recording'],
            callback_group=self._cb_group)
        self._convert = node.create_client(
            StartConversion, SERVICE_NAMES['convert'],
            callback_group=self._cb_group)
        self._convert_status = node.create_client(
            GetConversionStatus, SERVICE_NAMES['convert_status'],
            callback_group=self._cb_group)
        self._hub = node.create_client(
            HfOperation, SERVICE_NAMES['hub'],
            callback_group=self._cb_group)
        self._edit = node.create_client(
            EditDataset, SERVICE_NAMES['edit'],
            callback_group=self._cb_group)

    # ------------------------------------------------------------------
    # Public API — one method per cyclo_data operation
    # ------------------------------------------------------------------

    def send_recording_command(
        self,
        command: int,
        task_info: Optional[TaskInfo] = None,
        robot_type: str = '',
        topics=None,
        urdf_path: str = '',
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> CallResult:
        req = RecordingCommand.Request()
        req.command = command
        if task_info is not None:
            req.task_info = task_info
        req.robot_type = robot_type
        req.topics = list(topics or [])
        req.urdf_path = urdf_path
        return self._call(self._recording, req, timeout_sec, 'recording')

    def start_conversion(
        self,
        dataset_path: str,
        robot_type: str = '',
        robot_config_path: str = '',
        source_folders=None,
        fps: int = 0,
        convert_v21: bool = True,
        convert_v30: bool = True,
        selected_cameras=None,
        camera_rotations=None,
        image_resize=None,
        selected_state_topics=None,
        selected_action_topics=None,
        selected_joints=None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> CallResult:
        req = StartConversion.Request()
        req.dataset_path = dataset_path
        req.robot_type = robot_type
        req.robot_config_path = robot_config_path
        req.source_folders = list(source_folders or [])
        # 0 means 'use the cyclo_data-side default' (currently 15) so
        # callers that haven't started threading fps through yet still
        # get a working conversion.
        req.fps = int(fps)
        # Default both flags True at this layer so older callers keep
        # producing v2.1 + v3.0. When both are False the cyclo_data
        # side also interprets it as 'run both' (legacy UI safety).
        req.convert_v21 = bool(convert_v21)
        req.convert_v30 = bool(convert_v30)

        # Selection knobs — flatten dicts into the parallel-array shape
        # ROS2 srv requires.
        req.selected_cameras = list(selected_cameras or [])
        rot_keys, rot_values = [], []
        for cam, deg in (camera_rotations or {}).items():
            rot_keys.append(str(cam))
            rot_values.append(int(deg))
        req.camera_rotation_keys = rot_keys
        req.camera_rotation_values = rot_values
        if image_resize:
            req.image_resize_height = int(image_resize[0])
            req.image_resize_width = int(image_resize[1])
        else:
            req.image_resize_height = 0
            req.image_resize_width = 0
        req.selected_state_topics = list(selected_state_topics or [])
        req.selected_action_topics = list(selected_action_topics or [])
        req.selected_joints = list(selected_joints or [])

        return self._call(self._convert, req, timeout_sec, 'convert')

    def get_conversion_status(
        self,
        job_id: str,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> CallResult:
        req = GetConversionStatus.Request()
        req.job_id = job_id
        return self._call(self._convert_status, req, timeout_sec, 'convert_status')

    def hf_operation(
        self,
        operation: int,
        repo_type: int,
        repo_id: str = '',
        local_dir: str = '',
        author: str = '',
        endpoint: str = '',
        token: str = '',
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> CallResult:
        req = HfOperation.Request()
        req.operation = operation
        req.repo_type = repo_type
        req.repo_id = repo_id
        req.local_dir = local_dir
        req.author = author
        req.endpoint = endpoint
        req.token = token
        return self._call(self._hub, req, timeout_sec, 'hub')

    def edit_dataset(
        self,
        request: EditDataset.Request,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> CallResult:
        # EditDataset.srv has many mode-specific fields; easier for the
        # caller to build the request and pass it through unchanged.
        return self._call(self._edit, request, timeout_sec, 'edit')

    # ------------------------------------------------------------------
    # Readiness probe — non-blocking
    # ------------------------------------------------------------------

    def is_ready(self, timeout_sec: float = 0.0) -> bool:
        """True iff every /data/* service is discovered (optionally waiting)."""
        return all(
            client.wait_for_service(timeout_sec=timeout_sec)
            for client in (
                self._recording,
                self._convert,
                self._convert_status,
                self._hub,
                self._edit,
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call(self, client, request, timeout_sec: float, label: str) -> CallResult:
        if not client.wait_for_service(timeout_sec=0.0):
            msg = f'cyclo_data service unavailable: {SERVICE_NAMES[label]}'
            logger.warning(msg)
            return CallResult(success=False, message=msg, response=None)

        future = client.call_async(request)

        # rclpy.Future.result() has no timeout parameter. Mirror the
        # container_service_client pattern: wake on add_done_callback +
        # wait on a threading.Event. The outer MultiThreadedExecutor
        # drives the future's completion on its own threads, so this
        # blocks the caller but never deadlocks as long as this client's
        # callback group is distinct from the caller's.
        done_event = threading.Event()
        future.add_done_callback(lambda _fut: done_event.set())

        if not done_event.wait(timeout=timeout_sec):
            future.cancel()
            msg = f'cyclo_data call timed out after {timeout_sec:.1f}s: {label}'
            logger.warning(msg)
            return CallResult(success=False, message=msg, response=None)

        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 — want to surface any RPC error
            msg = f'cyclo_data call raised for {label}: {exc}'
            logger.exception(msg)
            return CallResult(success=False, message=msg, response=None)

        if response is None:
            msg = f'cyclo_data call returned None: {label}'
            logger.warning(msg)
            return CallResult(success=False, message=msg, response=None)

        success = getattr(response, 'success', False)
        message = getattr(response, 'message', '')
        return CallResult(success=success, message=message, response=response)
