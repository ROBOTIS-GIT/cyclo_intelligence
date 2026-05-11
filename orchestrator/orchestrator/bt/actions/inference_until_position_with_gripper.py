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

"""Inference action that runs until arms reach target positions and gripper state changes."""

import math
import time
from typing import List
from typing import TYPE_CHECKING

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.constants import *  # noqa: F403
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState

if TYPE_CHECKING:
    from rclpy.node import Node

LEFT_JOINT_NAMES = [
    'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4',
    'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 'gripper_l_joint1',
]
RIGHT_JOINT_NAMES = [
    'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4',
    'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1',
]


class InferenceUntilPositionWithGripper(BaseAction):
    """Action that runs until arms reach target positions AND gripper state changes.

    Returns SUCCESS when:
    1. Gripper state change detected (closed -> open or open -> closed)
    2. AND Euclidean distance between current and target positions <= tolerance

    Position checking starts after check_delay seconds to allow initial movement.
    """

    def __init__(
        self,
        node: 'Node',
        left_positions: List[float] = None,
        right_positions: List[float] = None,
        tolerance: float = 0.1,
        gripper_closed_threshold: float = GRIPPER_CLOSED_THRESHOLD,  # noqa: F405
        gripper_open_threshold: float = GRIPPER_OPEN_THRESHOLD,  # noqa: F405
        check_delay: float = 5.0,
    ):
        super().__init__(node, name='InferenceUntilPositionWithGripper')

        default_positions = [0.0] * 8
        self.left_positions = left_positions or default_positions
        self.right_positions = right_positions or default_positions

        if len(self.left_positions) != 8:
            raise ValueError(
                f'left_positions must have 8 values, '
                f'got {len(self.left_positions)}'
            )
        if len(self.right_positions) != 8:
            raise ValueError(
                f'right_positions must have 8 values, '
                f'got {len(self.right_positions)}'
            )

        self.tolerance = tolerance
        self.gripper_closed_threshold = gripper_closed_threshold
        self.gripper_open_threshold = gripper_open_threshold
        self.check_delay = check_delay

        # Joint state tracking
        self.joint_state = None

        # Gripper state tracking
        self.gripper_state_changed = False
        self.initial_gripper_state = None
        self.previous_gripper_state = None

        # Time tracking
        self.start_time = None
        self._tick_count = 0

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,  # noqa: F405
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile,
        )

        self.log_info(
            f'Initialized with tolerance={tolerance:.3f}, '
            f'closed_thr={gripper_closed_threshold:.3f}, '
            f'open_thr={gripper_open_threshold:.3f}, '
            f'check_delay={check_delay:.1f}s'
        )

    def _joint_state_callback(self, msg):
        self.joint_state = msg

    def _get_gripper_state(self) -> str:
        if self.joint_state is None:
            return 'unknown'

        name_to_idx = {
            name: i for i, name in enumerate(self.joint_state.name)
        }

        right_idx = name_to_idx.get('gripper_r_joint1')
        if right_idx is None:
            return 'unknown'

        right_pos = self.joint_state.position[right_idx]

        if right_pos < self.gripper_open_threshold:
            return 'open'
        elif right_pos > self.gripper_closed_threshold:
            return 'closed'
        else:
            return (
                self.previous_gripper_state
                if self.previous_gripper_state
                else 'unknown'
            )

    def _check_gripper_state_change(self):
        current_state = self._get_gripper_state()
        if current_state == 'unknown':
            return

        if self.initial_gripper_state is None:
            self.initial_gripper_state = current_state
            self.previous_gripper_state = current_state
            self.log_info(f'Initial gripper state: {current_state}')
            return

        if self.previous_gripper_state != current_state:
            self.log_info(
                f'Gripper state changed: '
                f'{self.previous_gripper_state} -> {current_state}'
            )
            self.previous_gripper_state = current_state

            if ((self.initial_gripper_state == 'open'
                 and current_state == 'closed')
                or (self.initial_gripper_state == 'closed'
                    and current_state == 'open')):
                self.gripper_state_changed = True
                self.log_info('Gripper state change detected!')

    def _calculate_euclidean_distance(self) -> float:
        if self.joint_state is None:
            return float('inf')

        name_to_idx = {
            name: i for i, name in enumerate(self.joint_state.name)
        }

        squared_sum = 0.0

        for joint_name, target_pos in zip(
            LEFT_JOINT_NAMES, self.left_positions
        ):
            idx = name_to_idx.get(joint_name)
            if idx is None:
                return float('inf')
            diff = self.joint_state.position[idx] - target_pos
            squared_sum += diff * diff

        for joint_name, target_pos in zip(
            RIGHT_JOINT_NAMES, self.right_positions
        ):
            idx = name_to_idx.get(joint_name)
            if idx is None:
                return float('inf')
            diff = self.joint_state.position[idx] - target_pos
            squared_sum += diff * diff

        return math.sqrt(squared_sum)

    def tick(self) -> NodeStatus:
        if self.start_time is None:
            self.start_time = time.monotonic()
            self.log_info(
                f'Started position and gripper monitoring '
                f'with {self.check_delay}s delay'
            )

        self._check_gripper_state_change()

        elapsed_time = time.monotonic() - self.start_time
        self._tick_count += 1

        if elapsed_time < self.check_delay:
            if self._tick_count % 30 == 0:
                remaining = self.check_delay - elapsed_time
                self.log_info(
                    f'Waiting {remaining:.1f}s before position check '
                    f'(gripper_changed: {self.gripper_state_changed})'
                )
            return NodeStatus.RUNNING

        distance = self._calculate_euclidean_distance()
        position_reached = distance <= self.tolerance

        if self.gripper_state_changed and position_reached:
            self.log_info(
                f'Both conditions met! Distance: {distance:.4f} '
                f'<= {self.tolerance:.4f} (after {elapsed_time:.1f}s)'
            )
            return NodeStatus.SUCCESS

        if self._tick_count % 30 == 0:
            self.log_info(
                f'Status - Distance: {distance:.4f} '
                f'(tolerance: {self.tolerance:.4f}), '
                f'Gripper changed: {self.gripper_state_changed}, '
                f'elapsed: {elapsed_time:.1f}s'
            )

        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.joint_state = None
        self.start_time = None
        self.gripper_state_changed = False
        self.initial_gripper_state = None
        self.previous_gripper_state = None
        self._tick_count = 0
