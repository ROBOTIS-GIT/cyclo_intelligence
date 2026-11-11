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

from pathlib import Path
import signal
import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

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
        self._edit = EditService(self, self._status_publisher, self._recording)

        # Resume any background transcodes that were interrupted by the
        # previous service exit. Done after services are advertised so a
        # transcode failure can't block service availability.
        try:
            workspace_root = Path('/workspace/rosbag2')
            if workspace_root.exists():
                self._recording.resume_pending_transcodes(workspace_root)
        except Exception as exc:
            self.get_logger().error(
                f'Transcode resume scan failed at startup: {exc!r}'
            )

        self.get_logger().info('cyclo_data node ready.')


def main(args=None):
    shutdown_requested = threading.Event()
    stop_spinning = threading.Event()
    previous_handlers = {}
    node = executor = spin_thread = None
    spin_thread_started = False
    errors = []

    def spin():
        try:
            while not stop_spinning.is_set() and rclpy.ok():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except (KeyboardInterrupt, ExternalShutdownException):
                    shutdown_requested.set()
                except Exception as exc:
                    if not errors:
                        errors.append(exc)
                    shutdown_requested.set()
                    # Keep delivering STOP responses after a callback failure,
                    # without busy-looping if the executor itself is broken.
                    stop_spinning.wait(0.01)
        finally:
            shutdown_requested.set()

    try:
        # The main thread owns teardown; signals must not close ROS first.
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(
                signum, lambda *_: shutdown_requested.set())
        node = CycloDataNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        spin_thread = threading.Thread(target=spin, name='cyclo_data_executor')
        spin_thread.start()
        spin_thread_started = True
        shutdown_requested.wait()
    finally:
        try:
            if node is not None:
                # RecordingService serializes against in-flight commands. Its
                # bounded STOP can still receive callbacks on the live executor.
                for service in (node._recording, node._hub, node._conversion):
                    try:
                        service.shutdown()
                    except Exception as exc:
                        errors.append(exc)
            stop_spinning.set()
            if spin_thread_started:
                spin_thread.join()
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
        finally:
            try:
                rclpy.try_shutdown()
            except RuntimeError as exc:
                if rclpy.ok() or 'rcl_shutdown already called' not in str(exc):
                    raise
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
    if errors:
        raise errors[0]


if __name__ == '__main__':
    main()
