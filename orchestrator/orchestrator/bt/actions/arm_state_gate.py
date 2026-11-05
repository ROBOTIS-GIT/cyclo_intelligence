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

"""BT action gate for arm joint targets and optional gripper events."""

import time
from typing import TYPE_CHECKING

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.constants import QOS_QUEUE_DEPTH
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState

if TYPE_CHECKING:
    from rclpy.node import Node


def _coerce_names(value) -> list[str]:
    """Normalize XML/UI joint-name values into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_positions(value) -> list[float]:
    """Normalize XML/UI position values into floats."""
    if value is None:
        return []
    if isinstance(value, str):
        return [
            float(part.strip())
            for part in value.split(',')
            if part.strip()
        ]
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(value)]


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def _make_targets(side: str, joints, positions) -> list[tuple[str, float]]:
    joint_names = _coerce_names(joints)
    target_positions = _coerce_positions(positions)
    if len(joint_names) != len(target_positions):
        raise ValueError(
            'ArmStateGate: '
            f'{side}_target_joints and {side}_target_positions must have '
            f'the same length ({len(joint_names)} != '
            f'{len(target_positions)})'
        )
    return list(zip(joint_names, target_positions))


class ArmStateGate(BaseAction):
    """Gate a behavior tree until selected arm state conditions are met.

    Optional gripper detection gates SUCCESS behind a closed-then-opened
    sequence per selected gripper. This node only observes /joint_states; it
    never publishes trajectory commands.
    """

    def __init__(
        self,
        node: 'Node',
        left_target_joints: str = '',
        left_target_positions: str = '',
        right_target_joints: str = '',
        right_target_positions: str = '',
        joint_threshold: float = 0.01,
        detect_left_gripper: bool = False,
        detect_right_gripper: bool = False,
        left_gripper_joint: str = 'gripper_l_joint1',
        right_gripper_joint: str = 'gripper_r_joint1',
        gripper_closed_value: float = 1.0,
        gripper_open_value: float = 0.0,
        gripper_threshold: float = 0.05,
        timeout_sec: float = 0.0,
    ):
        super().__init__(node, name='ArmStateGate')

        self.left_targets = _make_targets(
            'left', left_target_joints, left_target_positions,
        )
        self.right_targets = _make_targets(
            'right', right_target_joints, right_target_positions,
        )
        if not self.left_targets or not self.right_targets:
            raise ValueError(
                'ArmStateGate: provide at least one left and one right '
                'target joint.'
            )

        self.joint_threshold = float(joint_threshold)
        self.detect_left_gripper = _coerce_bool(detect_left_gripper)
        self.detect_right_gripper = _coerce_bool(detect_right_gripper)
        self.left_gripper_joint = str(left_gripper_joint)
        self.right_gripper_joint = str(right_gripper_joint)
        self.gripper_closed_value = float(gripper_closed_value)
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_threshold = float(gripper_threshold)
        self.timeout_sec = float(timeout_sec)

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.joint_state = None
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile,
        )

        self._start_time = None
        self._left_gripper_closed = False
        self._left_gripper_opened = False
        self._right_gripper_closed = False
        self._right_gripper_opened = False
        self._missing_joints_warned = set()

    def _joint_state_callback(self, msg):
        self.joint_state = msg

    def _is_close(self, value: float, target: float, threshold: float) -> bool:
        return abs(value - target) <= threshold

    def _warn_missing_once(self, joint_name: str):
        if joint_name in self._missing_joints_warned:
            return
        self._missing_joints_warned.add(joint_name)
        self.log_warn(f"Joint '{joint_name}' not found in /joint_states")

    def _joint_value(self, joint_name: str, name_to_idx: dict):
        idx = name_to_idx.get(joint_name)
        positions = getattr(self.joint_state, 'position', [])
        if idx is None or idx >= len(positions):
            self._warn_missing_once(joint_name)
            return None
        return positions[idx]

    def _update_gripper(
        self,
        enabled: bool,
        joint_name: str,
        closed_seen: bool,
        opened_seen: bool,
        name_to_idx: dict,
    ) -> tuple[bool, bool]:
        if not enabled:
            return True, True

        value = self._joint_value(joint_name, name_to_idx)
        if value is None:
            return closed_seen, opened_seen

        if not closed_seen and self._is_close(
            value, self.gripper_closed_value, self.gripper_threshold,
        ):
            closed_seen = True
            self.log_info(f"Gripper '{joint_name}' closed detected")

        if closed_seen and not opened_seen and self._is_close(
            value, self.gripper_open_value, self.gripper_threshold,
        ):
            opened_seen = True
            self.log_info(f"Gripper '{joint_name}' opened detected")

        return closed_seen, opened_seen

    def _targets_reached(self, name_to_idx: dict) -> bool:
        for joint_name, target in self.left_targets + self.right_targets:
            value = self._joint_value(joint_name, name_to_idx)
            if value is None:
                return False
            if not self._is_close(value, target, self.joint_threshold):
                return False
        return True

    def _timed_out(self) -> bool:
        if self.timeout_sec <= 0.0 or self._start_time is None:
            return False
        return time.monotonic() - self._start_time >= self.timeout_sec

    def tick(self) -> NodeStatus:
        if self._start_time is None:
            self._start_time = time.monotonic()
            self._missing_joints_warned.clear()
            self.log_info('Waiting for arm joint targets')
            return NodeStatus.RUNNING

        if self.joint_state is None:
            if self._timed_out():
                self.log_error('Timeout waiting for /joint_states')
                return NodeStatus.FAILURE
            return NodeStatus.RUNNING

        name_to_idx = {
            name: idx for idx, name in enumerate(self.joint_state.name)
        }

        self._left_gripper_closed, self._left_gripper_opened = (
            self._update_gripper(
                self.detect_left_gripper,
                self.left_gripper_joint,
                self._left_gripper_closed,
                self._left_gripper_opened,
                name_to_idx,
            )
        )
        self._right_gripper_closed, self._right_gripper_opened = (
            self._update_gripper(
                self.detect_right_gripper,
                self.right_gripper_joint,
                self._right_gripper_closed,
                self._right_gripper_opened,
                name_to_idx,
            )
        )

        grippers_done = (
            self._left_gripper_opened and self._right_gripper_opened
        )
        if grippers_done and self._targets_reached(name_to_idx):
            self.log_info('Arm joint targets reached')
            return NodeStatus.SUCCESS

        if self._timed_out():
            self.log_error('Timeout waiting for arm joint targets')
            return NodeStatus.FAILURE
        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self._start_time = None
        self._left_gripper_closed = False
        self._left_gripper_opened = False
        self._right_gripper_closed = False
        self._right_gripper_opened = False
        self._missing_joints_warned.clear()
        self.joint_state = None
