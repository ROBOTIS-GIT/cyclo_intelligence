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

"""Action that waits until arms reach target positions and gripper state changes."""

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

GRIPPER_CHECK_OPTIONS = ('none', 'left', 'right', 'both')

_SIDE_TO_JOINT = {
    'left': 'gripper_l_joint1',
    'right': 'gripper_r_joint1',
}


class WaitUntilPoseAndGripperChange(BaseAction):
    """Wait until arms reach target positions, optionally gated by gripper state change.

    Returns SUCCESS when:
    1. Euclidean distance between current and target positions <= tolerance
    2. AND every gripper side selected by `gripper_check` has transitioned
       between open and closed at least once

    `gripper_check` values:
    - 'none'  : ignore grippers; pose alone triggers SUCCESS
    - 'left'  : require left  gripper open<->closed transition
    - 'right' : require right gripper open<->closed transition
    - 'both'  : require BOTH grippers to transition (AND)

    Position checking starts after `check_delay` seconds to allow initial movement.
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
        gripper_check: str = 'none',
    ):
        super().__init__(node, name='WaitUntilPoseAndGripperChange')

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

        if gripper_check not in GRIPPER_CHECK_OPTIONS:
            raise ValueError(
                f'gripper_check must be one of {GRIPPER_CHECK_OPTIONS}, '
                f'got {gripper_check!r}'
            )

        self.tolerance = tolerance
        self.gripper_closed_threshold = gripper_closed_threshold
        self.gripper_open_threshold = gripper_open_threshold
        self.check_delay = check_delay
        self.gripper_check = gripper_check

        # Joint state tracking
        self.joint_state = None

        # Per-side gripper state tracking (only updated for monitored sides)
        self.initial_gripper_state = {'left': None, 'right': None}
        self.previous_gripper_state = {'left': None, 'right': None}
        self.gripper_state_changed = {'left': False, 'right': False}

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
            f'check_delay={check_delay:.1f}s, '
            f'gripper_check={gripper_check}'
        )

    def _joint_state_callback(self, msg):
        self.joint_state = msg

    def _sides_to_monitor(self) -> tuple:
        """Return the gripper sides that participate in the success condition."""
        if self.gripper_check == 'left':
            return ('left',)
        if self.gripper_check == 'right':
            return ('right',)
        if self.gripper_check == 'both':
            return ('left', 'right')
        return ()

    def _get_gripper_state(self, side: str) -> str:
        """Classify one side's gripper as 'open' / 'closed' / 'unknown'.

        Returns the previous state inside the dead-band between thresholds
        (when there is one), otherwise 'unknown'.
        """
        if self.joint_state is None:
            return 'unknown'
        joint_name = _SIDE_TO_JOINT.get(side)
        if joint_name is None:
            return 'unknown'

        name_to_idx = {
            name: i for i, name in enumerate(self.joint_state.name)
        }
        idx = name_to_idx.get(joint_name)
        if idx is None:
            return 'unknown'

        pos = self.joint_state.position[idx]
        if pos < self.gripper_open_threshold:
            return 'open'
        if pos > self.gripper_closed_threshold:
            return 'closed'
        prev = self.previous_gripper_state[side]
        return prev if prev else 'unknown'

    def _check_gripper_state_change(self):
        """Update per-side latched flags for every monitored gripper side."""
        for side in self._sides_to_monitor():
            state = self._get_gripper_state(side)
            if state == 'unknown':
                continue

            if self.initial_gripper_state[side] is None:
                self.initial_gripper_state[side] = state
                self.previous_gripper_state[side] = state
                self.log_info(f'Initial {side} gripper state: {state}')
                continue

            if self.previous_gripper_state[side] != state:
                self.log_info(
                    f'{side.capitalize()} gripper state changed: '
                    f'{self.previous_gripper_state[side]} -> {state}'
                )
                self.previous_gripper_state[side] = state

                initial = self.initial_gripper_state[side]
                if ((initial == 'open' and state == 'closed')
                        or (initial == 'closed' and state == 'open')):
                    if not self.gripper_state_changed[side]:
                        self.log_info(
                            f'{side.capitalize()} gripper transition detected!'
                        )
                    self.gripper_state_changed[side] = True

    def _gripper_condition_met(self) -> bool:
        sides = self._sides_to_monitor()
        if not sides:
            return True  # 'none' — pose-only success
        return all(self.gripper_state_changed[s] for s in sides)

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
                    f'(gripper_changed={dict(self.gripper_state_changed)})'
                )
            return NodeStatus.RUNNING

        distance = self._calculate_euclidean_distance()
        position_reached = distance <= self.tolerance
        gripper_ok = self._gripper_condition_met()

        if gripper_ok and position_reached:
            self.log_info(
                f'Conditions met (gripper_check={self.gripper_check})! '
                f'Distance: {distance:.4f} <= {self.tolerance:.4f} '
                f'(after {elapsed_time:.1f}s)'
            )
            return NodeStatus.SUCCESS

        if self._tick_count % 30 == 0:
            self.log_info(
                f'Status - Distance: {distance:.4f} '
                f'(tolerance: {self.tolerance:.4f}), '
                f'Gripper changed: {dict(self.gripper_state_changed)} '
                f'(check={self.gripper_check}), '
                f'elapsed: {elapsed_time:.1f}s'
            )

        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.joint_state = None
        self.start_time = None
        self.initial_gripper_state = {'left': None, 'right': None}
        self.previous_gripper_state = {'left': None, 'right': None}
        self.gripper_state_changed = {'left': False, 'right': False}
        self._tick_count = 0
