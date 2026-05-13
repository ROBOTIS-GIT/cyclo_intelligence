# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Rosbag recording control — proxy over the rosbag_recorder package.

Migrated from orchestrator.Communicator in Step 3 Part C2d-1 per
REVIEW §9.2. Owns the /rosbag_recorder/send_command client and the
/task/action_event publisher on whatever node instantiates it
(cyclo_data_node). The API surface mirrors Communicator's rosbag
methods one-for-one so Part C2d-3 can swap call sites with minimal
friction.

C2d-1 scope: class is added and instantiated by RecordingService at
startup but nothing calls into it yet — orchestrator.Communicator's
parallel methods still serve the live recording path. C2d-2/-3/-4
flip the wiring and then retire the orchestrator-side copies.
"""

import threading
from typing import List, Optional

from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rosbag_recorder.srv import SendCommand
from std_msgs.msg import String


_DEFAULT_REQUEST_TIMEOUT_SEC = 5.0
_SERVICE_WAIT_TIMEOUT_SEC = 3.0
_PUB_QOS_SIZE = 10


class RosbagControl:

    ROSBAG_SERVICE_NAME = '/rosbag_recorder/send_command'
    ACTION_EVENT_TOPIC = '/task/action_event'

    def __init__(self, node: Node, callback_group: Optional[CallbackGroup] = None):
        self._node = node
        self._cb_group = callback_group or ReentrantCallbackGroup()

        self._send_command_client = node.create_client(
            SendCommand,
            self.ROSBAG_SERVICE_NAME,
            callback_group=self._cb_group,
        )
        self._action_event_publisher = node.create_publisher(
            String,
            self.ACTION_EVENT_TOPIC,
            _PUB_QOS_SIZE,
        )

        self._service_available = self._send_command_client.wait_for_service(
            timeout_sec=_SERVICE_WAIT_TIMEOUT_SEC,
        )
        if self._service_available:
            node.get_logger().info(
                f'Rosbag service available: {self.ROSBAG_SERVICE_NAME}')
        else:
            node.get_logger().warning(
                f'Rosbag service not available within '
                f'{_SERVICE_WAIT_TIMEOUT_SEC}s: {self.ROSBAG_SERVICE_NAME}'
            )

    def is_available(self) -> bool:
        return self._service_available

    # ------------------------------------------------------------------
    # High-level commands (mirror orchestrator.Communicator's API)
    # ------------------------------------------------------------------

    def prepare_rosbag(self, topics: List[str]) -> None:
        self._send_rosbag_command(
            command=SendCommand.Request.PREPARE,
            topics=topics,
        )

    def start_rosbag(
        self,
        rosbag_uri: str,
        timeout_sec: float = _DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> None:
        self._send_rosbag_command(
            command=SendCommand.Request.START,
            uri=rosbag_uri,
            wait_for_response=True,
            timeout_sec=timeout_sec,
        )

    def stop_rosbag(self) -> None:
        self._send_rosbag_command(command=SendCommand.Request.STOP)

    def stop_and_delete_rosbag(self) -> None:
        # Wait synchronously so the caller knows the bag directory has
        # actually been removed before it does its own follow-up cleanup
        # (mp4/yaml left behind by VideoRecorder, defensive rmtree).
        self._send_rosbag_command(
            command=SendCommand.Request.STOP_AND_DELETE,
            wait_for_response=True,
        )

    def finish_rosbag(self) -> None:
        self._send_rosbag_command(command=SendCommand.Request.FINISH)

    def publish_action_event(self, event: str) -> None:
        msg = String()
        msg.data = event
        self._action_event_publisher.publish(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Best-effort stop of any active recording before node shutdown."""
        if not self._service_available:
            return
        try:
            self.stop_rosbag()
        except Exception as exc:  # noqa: BLE001 — shutdown is best-effort
            self._node.get_logger().warning(
                f'Best-effort rosbag stop on shutdown failed: {exc}')

    # ------------------------------------------------------------------
    # Internals — identical control flow to Communicator._send_rosbag_command
    # so callers can be swapped without behaviour change.
    # ------------------------------------------------------------------

    def _send_rosbag_command(
        self,
        command: int,
        topics: Optional[List[str]] = None,
        uri: Optional[str] = None,
        wait_for_response: bool = False,
        timeout_sec: float = _DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> None:
        if not self._service_available:
            self._node.get_logger().error(
                f'Rosbag service unavailable — dropping command {command}')
            raise RuntimeError(f'{self.ROSBAG_SERVICE_NAME} not available')

        req = SendCommand.Request()
        req.command = command
        req.topics = topics if topics is not None else []
        req.uri = uri if uri is not None else ''

        future = self._send_command_client.call_async(req)

        if not wait_for_response:
            future.add_done_callback(self._log_fire_and_forget_result(command))
            return

        done_event = threading.Event()
        future.add_done_callback(lambda _fut: done_event.set())
        if not done_event.wait(timeout=timeout_sec):
            raise TimeoutError(
                f'Rosbag command timeout: command={command}, timeout={timeout_sec}s')

        result = future.result()
        if result is None:
            raise RuntimeError(f'Rosbag command returned no response: command={command}')
        if not result.success:
            raise RuntimeError(
                f'Rosbag command failed: command={command}, message={result.message}')
        self._node.get_logger().info(
            f'Rosbag command completed: command={command}, message={result.message}')

    def _log_fire_and_forget_result(self, command: int):
        def callback(future) -> None:
            if future.done() and future.result() is not None and future.result().success:
                self._node.get_logger().info(
                    f'Sent rosbag record command {command}: '
                    f'{future.result().message}')
            else:
                msg = (
                    future.result().message
                    if future.done() and future.result() is not None
                    else 'timeout'
                )
                self._node.get_logger().error(
                    f'Failed to send rosbag command {command}: {msg}')
        return callback
