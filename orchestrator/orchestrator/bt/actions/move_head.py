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

"""Action node for moving the robot head to specified positions."""

import threading
import time
from typing import List
from typing import TYPE_CHECKING

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.constants import *  # noqa: F403
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

if TYPE_CHECKING:
    from rclpy.node import Node


class MoveHead(BaseAction):
    """Action to move the robot head to target joint positions."""

    DEFAULT_HEAD_JOINTS = ['head_joint1', 'head_joint2']

    def __init__(
            self,
            node: 'Node',
            head_positions: List[float] = None,
            head_joint_names: List[str] = None,
            position_threshold: float = POSITION_THRESHOLD_RAD,  # noqa: F405
            duration: float = DEFAULT_MOVE_HEAD_DURATION_SEC,  # noqa: F405
    ):
        """Initialize the MoveHead action."""
        super().__init__(node, name='MoveHead')
        self.head_joint_names = head_joint_names or self.DEFAULT_HEAD_JOINTS
        self.head_positions = (
            head_positions if head_positions else [0.0, 0.0]
        )
        self.position_threshold = position_threshold
        self.duration = duration

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,  # noqa: F405
            reliability=ReliabilityPolicy.RELIABLE
        )
        topic_head = (
            '/leader/joystick_controller_left/joint_trajectory'
        )
        self.head_pub = self.node.create_publisher(
            JointTrajectory,
            topic_head,
            qos_profile
        )

        self.joint_state = None
        from sensor_msgs.msg import JointState
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._result = None  # None=running, True=success, False=failure
        self._control_rate = CONTROL_RATE_HZ  # noqa: F405

    def _joint_state_callback(self, msg):
        """Receive joint state updates."""
        self.joint_state = msg

    def _control_loop(self):
        """Control loop that publishes trajectories and monitors progress."""
        rate_sleep = RATE_SLEEP_SEC  # noqa: F405

        pos_str = str(self.head_positions)
        self.log_info(f'Publishing head trajectory: {pos_str}')

        head_traj = JointTrajectory()
        head_traj.joint_names = self.head_joint_names
        head_point = JointTrajectoryPoint()
        head_point.positions = self.head_positions
        head_point.time_from_start.sec = int(self.duration)
        head_traj.points.append(head_point)
        self.head_pub.publish(head_traj)

        self.log_info('Head trajectory published')

        timeout_count = 0
        max_timeout = MOVE_HEAD_TIMEOUT_TICKS  # noqa: F405
        while not self._stop_event.is_set() and timeout_count < max_timeout:
            if self.joint_state is None:
                time.sleep(rate_sleep)
                timeout_count += 1
                continue

            name_to_idx = {
                n: i for i, n in enumerate(self.joint_state.name)
            }
            all_reached = True

            for jname, target in zip(
                self.head_joint_names, self.head_positions
            ):
                idx = name_to_idx.get(jname)
                if idx is not None:
                    pos = self.joint_state.position[idx]
                    if abs(pos - target) > self.position_threshold:
                        all_reached = False
                        break
                else:
                    self.log_warn(f"Joint '{jname}' not found in /joint_states")
                    all_reached = False
                    break

            if all_reached:
                self.log_info('Head reached target positions')
                with self._lock:
                    self._result = True
                return

            time.sleep(rate_sleep)
            timeout_count += 1

        with self._lock:
            self._result = False
        self.log_error('Head timeout waiting for target positions')

    def tick(self) -> NodeStatus:
        """Execute the action and return its status."""
        if self._thread is None:
            self.joint_state = None
            self._stop_event.clear()
            with self._lock:
                self._result = None

            self._thread = threading.Thread(
                target=self._control_loop, daemon=True
            )
            self._thread.start()
            pos_str = str(self.head_positions)
            self.log_info(f'MoveHead started with positions: {pos_str}')
            return NodeStatus.RUNNING

        with self._lock:
            result = self._result

        if result is None:
            return NodeStatus.RUNNING
        return NodeStatus.SUCCESS if result else NodeStatus.FAILURE

    def reset(self):
        """Reset the action to its initial state."""
        super().reset()
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)  # noqa: F405
        self._thread = None
        with self._lock:
            self._result = None
        self.joint_state = None
