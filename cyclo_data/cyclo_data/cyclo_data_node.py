#!/usr/bin/env python3
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""cyclo_data ROS 2 node entrypoint.

Owns the data plane (recording, conversion, editing, HuggingFace hub)
independently of the control plane (orchestrator). Services live under
the /data/ prefix. DataOperationStatus flows on /data/status.
"""

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from interfaces.msg import DataOperationStatus

from cyclo_data.services.conversion_service import ConversionService
from cyclo_data.services.edit_service import EditService
from cyclo_data.services.hub_service import HubService
from cyclo_data.services.recording_service import RecordingService


STATUS_TOPIC = '/data/status'


class CycloDataNode(Node):

    def __init__(self):
        super().__init__('cyclo_data')

        # Callback groups:
        #   state  — recording command dispatch (mutually exclusive; owns
        #            the session state machine that Part C migrates).
        #   io     — conversion / HF / edit (reentrant; long-running I/O
        #            jobs that must not block each other).
        self.state_callback_group = MutuallyExclusiveCallbackGroup()
        self.io_callback_group = ReentrantCallbackGroup()

        self._status_publisher = self.create_publisher(
            DataOperationStatus, STATUS_TOPIC, 10)

        # Services — each advertises under /data/<name> and logs on startup.
        self._recording = RecordingService(self, self._status_publisher)
        self._conversion = ConversionService(self, self._status_publisher)
        self._hub = HubService(self, self._status_publisher)
        self._edit = EditService(self, self._status_publisher)

        self.get_logger().info('cyclo_data node ready.')


def main(args=None):
    rclpy.init(args=args)
    node = CycloDataNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Services that own background workers / external resources clean
        # them up explicitly before rclpy shutdown.
        node._hub.shutdown()
        node._conversion.shutdown()
        node._recording.shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
