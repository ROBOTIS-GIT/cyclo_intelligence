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

"""Action node for moving the robot lift to a specified position."""

import threading
import time
from typing import TYPE_CHECKING

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.constants import *  # noqa: F403
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

if TYPE_CHECKING:
    from rclpy.node import Node


class MoveLift(BaseAction):
    """Action to move the robot lift to a target position."""

    DEFAULT_LIFT_JOINT = 'lift_joint'

    def __init__(
        self,
        node: 'Node',
        lift_position: float = ZERO_VELOCITY,  # noqa: F405
        lift_joint_name: str = None,
        position_threshold: float = POSITION_THRESHOLD_RAD,  # noqa: F405
        duration: float = DEFAULT_MOVE_LIFT_DURATION_SEC,  # noqa: F405
    ):
        """Initialize the MoveLift action."""
        super().__init__(node, name='MoveLift')
        self.lift_joint_name = lift_joint_name or self.DEFAULT_LIFT_JOINT
        self.target_position = lift_position
        self.position_threshold = position_threshold
        self.duration = duration

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,  # noqa: F405
            reliability=ReliabilityPolicy.RELIABLE
        )

        topic_lift = (
            '/leader/joystick_controller_right/joint_trajectory'
        )
        self.lift_pub = self.node.create_publisher(
            JointTrajectory,
            topic_lift,
            qos_profile
        )

        self.joint_state = None
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

    def _get_joint_position(self, joint_name: str):
        """Get the current position of a joint by name."""
        if self.joint_state is None:
            return None

        try:
            idx = self.joint_state.name.index(joint_name)
            return self.joint_state.position[idx]
        except (ValueError, IndexError):
            return None

    def _control_loop(self):
        """Control loop that publishes trajectories and monitors progress."""
        rate_sleep = RATE_SLEEP_SEC  # noqa: F405

        lift_traj = JointTrajectory()
        lift_traj.joint_names = [self.lift_joint_name]
        lift_point = JointTrajectoryPoint()
        lift_point.positions = [self.target_position]
        lift_point.time_from_start.sec = int(self.duration)
        lift_traj.points.append(lift_point)
        self.lift_pub.publish(lift_traj)

        target_str = str(self.target_position)
        self.log_info(f'Lift trajectory published: target={target_str}')

        timeout_count = 0
        max_timeout = MOVE_LIFT_TIMEOUT_TICKS  # noqa: F405
        while not self._stop_event.is_set() and timeout_count < max_timeout:
            if self.joint_state is not None:
                current_pos = self._get_joint_position(self.lift_joint_name)
                if current_pos is not None:
                    error = abs(current_pos - self.target_position)
                    if error <= self.position_threshold:
                        pos_str = f'{current_pos:.3f}'
                        self.log_info(
                            f'Lift reached target position: {pos_str}'
                        )
                        with self._lock:
                            self._result = True
                        return
                else:
                    self.log_warn(
                        f"Joint '{self.lift_joint_name}' not found in "
                        f'/joint_states'
                    )

            time.sleep(rate_sleep)
            timeout_count += 1

        with self._lock:
            self._result = False
        self.log_error('Lift timeout waiting for target position')

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
            target_str = str(self.target_position)
            self.log_info(f'MoveLift thread started: target={target_str}')
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
