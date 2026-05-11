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

"""Inference actions that run until arms stabilize AND gripper reaches target."""

import time
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


class InferenceUntilGripper(BaseAction):
    """Base action that runs until arms stabilize AND gripper reaches target.

    Returns SUCCESS when:
    1. Arms static for static_duration
    2. Current gripper state matches target (both grippers must match)
    """

    def __init__(
        self,
        node: 'Node',
        target_gripper_state: str = 'closed',
        position_change_threshold: float = INFERENCE_POSITION_CHANGE_THRESHOLD,  # noqa: F405
        static_duration: float = INFERENCE_STATIC_DURATION,  # noqa: F405
        history_window: float = INFERENCE_HISTORY_WINDOW,  # noqa: F405
        gripper_closed_threshold: float = GRIPPER_CLOSED_THRESHOLD,  # noqa: F405
        gripper_open_threshold: float = GRIPPER_OPEN_THRESHOLD,  # noqa: F405
        name: str = 'InferenceUntilGripper',
    ):
        super().__init__(node, name=name)
        self.target_gripper_state = target_gripper_state
        self.position_change_threshold = position_change_threshold
        self.static_duration = static_duration
        self.history_window = history_window
        self.gripper_closed_threshold = gripper_closed_threshold
        self.gripper_open_threshold = gripper_open_threshold

        # Position history tracking
        self.position_history = []
        self.static_start_time = None
        self.gripper_reached_time = None  # When gripper first reached target

        # Gripper state tracking
        self.previous_gripper_state = None

        # Joint names to monitor
        self.monitored_joints = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES
        self.joint_positions = {}
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
            f'Initialized (target={target_gripper_state}, '
            f'pos_threshold={position_change_threshold:.3f}, '
            f'static_dur={static_duration:.1f}s, '
            f'closed_thr={gripper_closed_threshold:.3f}, '
            f'open_thr={gripper_open_threshold:.3f})'
        )

    def _joint_state_callback(self, msg):
        try:
            self.joint_positions = dict(zip(msg.name, msg.position))
        except Exception as e:
            self.log_warn(f'Error in joint state callback: {e}')

    def _get_gripper_state(self) -> str:
        left_pos = self.joint_positions.get('gripper_l_joint1')
        right_pos = self.joint_positions.get('gripper_r_joint1')
        if left_pos is None or right_pos is None:
            return 'unknown'

        # 0.0 = fully open, 1.0 = fully closed
        left_open = left_pos < self.gripper_open_threshold
        left_closed = left_pos > self.gripper_closed_threshold
        right_open = right_pos < self.gripper_open_threshold
        right_closed = right_pos > self.gripper_closed_threshold

        if left_closed and right_closed:
            return 'closed'
        elif left_open and right_open:
            return 'open'
        else:
            return 'transitioning'

    def _log_gripper_state_change(self):
        current_state = self._get_gripper_state()

        if self.previous_gripper_state is None:
            self.previous_gripper_state = current_state
            left_pos = self.joint_positions.get('gripper_l_joint1')
            right_pos = self.joint_positions.get('gripper_r_joint1')
            self.log_info(
                f'Initial gripper state: {current_state} '
                f'(L={left_pos}, R={right_pos})'
            )
            return

        if self.previous_gripper_state != current_state:
            left_pos = self.joint_positions.get('gripper_l_joint1')
            right_pos = self.joint_positions.get('gripper_r_joint1')
            self.log_info(
                f'Gripper state changed: '
                f'{self.previous_gripper_state} -> {current_state} '
                f'(L={left_pos}, R={right_pos})'
            )
            self.previous_gripper_state = current_state

    def _update_position_history(self):
        if not self.joint_positions:
            return

        current_time = time.monotonic()
        current_positions = {}
        for joint_name in self.monitored_joints:
            if joint_name in self.joint_positions:
                current_positions[joint_name] = self.joint_positions[joint_name]

        if len(current_positions) == len(self.monitored_joints):
            self.position_history.append((current_time, current_positions))
            cutoff_time = current_time - self.history_window
            self.position_history = [
                (t, pos) for t, pos in self.position_history
                if t >= cutoff_time
            ]

    def _calculate_max_position_change(self) -> float:
        if len(self.position_history) < 2:
            return float('inf')

        _, oldest_pos = self.position_history[0]
        _, newest_pos = self.position_history[-1]

        max_change = 0.0
        for joint_name in self.monitored_joints:
            if joint_name in oldest_pos and joint_name in newest_pos:
                change = abs(newest_pos[joint_name] - oldest_pos[joint_name])
                max_change = max(max_change, change)

        return max_change

    def _is_static(self) -> bool:
        max_change = self._calculate_max_position_change()
        if max_change == float('inf'):
            return False
        return max_change < self.position_change_threshold

    def tick(self) -> NodeStatus:
        # Log gripper state changes
        self._log_gripper_state_change()

        # Update position history
        self._update_position_history()

        current_gripper = self._get_gripper_state()
        is_static = self._is_static()
        self._tick_count += 1

        now = time.monotonic()
        relaxed_duration = self.static_duration * 2

        # Periodic debug log
        if self._tick_count % 50 == 0:
            max_change = self._calculate_max_position_change()
            left_pos = self.joint_positions.get('gripper_l_joint1')
            right_pos = self.joint_positions.get('gripper_r_joint1')
            gripper_elapsed = (
                f'{now - self.gripper_reached_time:.1f}s'
                if self.gripper_reached_time else 'none'
            )
            self.log_info(
                f'[DEBUG] static={is_static} '
                f'(max_change={max_change:.4f}, '
                f'threshold={self.position_change_threshold}), '
                f'gripper={current_gripper} '
                f'(target={self.target_gripper_state}, '
                f'L={left_pos}, R={right_pos}), '
                f'gripper_timer={gripper_elapsed}, '
                f'static_timer={"active" if self.static_start_time else "none"}'
            )

        if current_gripper == self.target_gripper_state:
            # Track when gripper first reached target (relaxed path)
            if self.gripper_reached_time is None:
                self.gripper_reached_time = now
                self.log_info(
                    f'Gripper reached {self.target_gripper_state}, '
                    f'relaxed timer started ({relaxed_duration}s)'
                )

            # Fast path: gripper at target AND arms static
            if is_static:
                if self.static_start_time is None:
                    self.static_start_time = now
                    self.log_info(
                        f'Gripper is {self.target_gripper_state} and arms '
                        f'static, holding for {self.static_duration}s...'
                    )
                elif now - self.static_start_time >= self.static_duration:
                    self.log_info(
                        f'Fast path: static for {self.static_duration}s '
                        f'and gripper is {self.target_gripper_state}.'
                    )
                    return NodeStatus.SUCCESS
            else:
                self.static_start_time = None

            # Relaxed path: gripper at target for extended duration
            if now - self.gripper_reached_time >= relaxed_duration:
                self.log_info(
                    f'Relaxed path: gripper has been '
                    f'{self.target_gripper_state} for '
                    f'{relaxed_duration}s.'
                )
                return NodeStatus.SUCCESS
        else:
            # Gripper not at target - reset all timers
            if self.gripper_reached_time is not None:
                self.log_info(
                    f'Gripper left {self.target_gripper_state} '
                    f'(now={current_gripper}). Resetting timers.'
                )
            self.static_start_time = None
            self.gripper_reached_time = None

        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.static_start_time = None
        self.gripper_reached_time = None
        self.position_history = []
        self.joint_positions = {}
        self.previous_gripper_state = None
        self._tick_count = 0


class InferenceUntilStatic(BaseAction):
    """Action that runs until both arms stop moving (no gripper condition).

    Returns SUCCESS when arm position change stays below threshold
    for static_duration seconds.
    """

    def __init__(
        self,
        node: 'Node',
        position_change_threshold: float = INFERENCE_POSITION_CHANGE_THRESHOLD,  # noqa: F405
        static_duration: float = INFERENCE_STATIC_DURATION,  # noqa: F405
        history_window: float = INFERENCE_HISTORY_WINDOW,  # noqa: F405
    ):
        super().__init__(node, name='InferenceUntilStatic')
        self.position_change_threshold = position_change_threshold
        self.static_duration = static_duration
        self.history_window = history_window

        self.position_history = []
        self.static_start_time = None
        self.monitored_joints = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES
        self.joint_positions = {}
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
            f'Initialized (pos_threshold={position_change_threshold:.3f}, '
            f'static_dur={static_duration:.1f}s, '
            f'history_window={history_window:.1f}s)'
        )

    def _joint_state_callback(self, msg):
        try:
            self.joint_positions = dict(zip(msg.name, msg.position))
        except Exception as e:
            self.log_warn(f'Error in joint state callback: {e}')

    def _update_position_history(self):
        if not self.joint_positions:
            return

        current_time = time.monotonic()
        current_positions = {}
        for joint_name in self.monitored_joints:
            if joint_name in self.joint_positions:
                current_positions[joint_name] = self.joint_positions[joint_name]

        if len(current_positions) == len(self.monitored_joints):
            self.position_history.append((current_time, current_positions))
            cutoff_time = current_time - self.history_window
            self.position_history = [
                (t, pos) for t, pos in self.position_history
                if t >= cutoff_time
            ]

    def _calculate_max_position_change(self) -> float:
        if len(self.position_history) < 2:
            return float('inf')

        _, oldest_pos = self.position_history[0]
        _, newest_pos = self.position_history[-1]

        max_change = 0.0
        for joint_name in self.monitored_joints:
            if joint_name in oldest_pos and joint_name in newest_pos:
                change = abs(newest_pos[joint_name] - oldest_pos[joint_name])
                max_change = max(max_change, change)

        return max_change

    def _is_static(self) -> bool:
        max_change = self._calculate_max_position_change()
        if max_change == float('inf'):
            return False
        return max_change < self.position_change_threshold

    def tick(self) -> NodeStatus:
        self._update_position_history()

        is_static = self._is_static()
        self._tick_count += 1
        now = time.monotonic()

        if self._tick_count % 50 == 0:
            max_change = self._calculate_max_position_change()
            self.log_info(
                f'[DEBUG] static={is_static} '
                f'(max_change={max_change:.4f}, '
                f'threshold={self.position_change_threshold}), '
                f'static_timer={"active" if self.static_start_time else "none"}'
            )

        if is_static:
            if self.static_start_time is None:
                self.static_start_time = now
                self.log_info(
                    f'Arms static, holding for {self.static_duration}s...'
                )
            elif now - self.static_start_time >= self.static_duration:
                self.log_info(
                    f'Arms static for {self.static_duration}s. Done.'
                )
                return NodeStatus.SUCCESS
        else:
            self.static_start_time = None

        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.static_start_time = None
        self.position_history = []
        self.joint_positions = {}
        self._tick_count = 0


class InferenceUntilGripperClose(InferenceUntilGripper):
    """Inference that succeeds when grippers close AND arms stabilize."""

    def __init__(
        self,
        node: 'Node',
        position_change_threshold: float = INFERENCE_POSITION_CHANGE_THRESHOLD,  # noqa: F405
        static_duration: float = INFERENCE_STATIC_DURATION,  # noqa: F405
        history_window: float = INFERENCE_HISTORY_WINDOW,  # noqa: F405
        gripper_closed_threshold: float = GRIPPER_CLOSED_THRESHOLD,  # noqa: F405
        gripper_open_threshold: float = GRIPPER_OPEN_THRESHOLD,  # noqa: F405
    ):
        super().__init__(
            node=node,
            target_gripper_state='closed',
            position_change_threshold=position_change_threshold,
            static_duration=static_duration,
            history_window=history_window,
            gripper_closed_threshold=gripper_closed_threshold,
            gripper_open_threshold=gripper_open_threshold,
            name='InferenceUntilGripperClose',
        )


class InferenceUntilGripperOpen(InferenceUntilGripper):
    """Inference that succeeds when grippers open AND arms stabilize."""

    def __init__(
        self,
        node: 'Node',
        position_change_threshold: float = INFERENCE_POSITION_CHANGE_THRESHOLD,  # noqa: F405
        static_duration: float = INFERENCE_STATIC_DURATION,  # noqa: F405
        history_window: float = INFERENCE_HISTORY_WINDOW,  # noqa: F405
        gripper_closed_threshold: float = GRIPPER_CLOSED_THRESHOLD,  # noqa: F405
        gripper_open_threshold: float = GRIPPER_OPEN_THRESHOLD,  # noqa: F405
    ):
        super().__init__(
            node=node,
            target_gripper_state='open',
            position_change_threshold=position_change_threshold,
            static_duration=static_duration,
            history_window=history_window,
            gripper_closed_threshold=gripper_closed_threshold,
            gripper_open_threshold=gripper_open_threshold,
            name='InferenceUntilGripperOpen',
        )
