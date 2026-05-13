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

"""Rule-based joint-space control for head, arms, and lift.

Replaces the earlier MoveHead / MoveArms / MoveLift trio. A single node
can drive any combination of the three joint groups: just toggle the
matching enable_* flag and supply its positions. With multiple groups on
at once their trajectories fire simultaneously and the node succeeds
once all activated joints land within position_threshold.
"""

import threading
import time
from typing import List
from typing import Optional
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


# Per-group static config. Topic + default joint names + group-specific
# default duration & timeout live here so the channel-builder below stays
# data-driven.
GROUP_DEFAULTS = {
    'head': {
        'topic': '/leader/joystick_controller_left/joint_trajectory',
        'joint_names': ['head_joint1', 'head_joint2'],
        'duration': DEFAULT_MOVE_HEAD_DURATION_SEC,  # noqa: F405
        'timeout_ticks': MOVE_HEAD_TIMEOUT_TICKS,  # noqa: F405
    },
    'arm_left': {
        'topic': (
            '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory'
        ),
        'joint_names': [
            'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4',
            'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 'gripper_l_joint1',
        ],
        'duration': DEFAULT_MOVE_ARMS_DURATION_SEC,  # noqa: F405
        'timeout_ticks': MOVE_ARMS_TIMEOUT_TICKS,  # noqa: F405
    },
    'arm_right': {
        'topic': (
            '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory'
        ),
        'joint_names': [
            'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4',
            'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1',
        ],
        'duration': DEFAULT_MOVE_ARMS_DURATION_SEC,  # noqa: F405
        'timeout_ticks': MOVE_ARMS_TIMEOUT_TICKS,  # noqa: F405
    },
    'lift': {
        'topic': '/leader/joystick_controller_right/joint_trajectory',
        'joint_names': ['lift_joint'],
        'duration': DEFAULT_MOVE_LIFT_DURATION_SEC,  # noqa: F405
        'timeout_ticks': MOVE_LIFT_TIMEOUT_TICKS,  # noqa: F405
    },
}


def _coerce_positions(value) -> Optional[List[float]]:
    """Normalise a positions param into list[float].

    XML attributes arrive as int / float / list (from bt_nodes_loader's
    _convert_value) or None when omitted. Lift sends a scalar; head/arms
    send a list. Wrap scalars so the rest of the control loop only sees a
    sequence.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]


class JointControl(BaseAction):
    """Drive any combination of head / arms / lift to target positions.

    Each group is independently toggled via enable_head / enable_arms /
    enable_lift. At least one group must be enabled (the action raises
    ValueError on construction otherwise) so an empty Joint Control node
    can't silently succeed at runtime.
    """

    SUPPORTED_GROUPS = ('head', 'arm_left', 'arm_right', 'lift')

    def __init__(
        self,
        node: 'Node',
        enable_head: bool = False,
        head_positions: Optional[List[float]] = None,
        head_joint_names: Optional[List[str]] = None,
        enable_arms: bool = False,
        left_positions: Optional[List[float]] = None,
        right_positions: Optional[List[float]] = None,
        left_joint_names: Optional[List[str]] = None,
        right_joint_names: Optional[List[str]] = None,
        enable_lift: bool = False,
        lift_position=None,
        lift_joint_name: Optional[str] = None,
        duration: Optional[float] = None,
        position_threshold: float = POSITION_THRESHOLD_RAD,  # noqa: F405
    ):
        super().__init__(node, name='JointControl')

        self.position_threshold = position_threshold

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,  # noqa: F405
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Build one channel per enabled sub-group. Each channel owns its
        # own publisher + joint_names + target positions so the control
        # loop just iterates over them.
        self._channels = []  # [{name, pub, joint_names, positions}, ...]

        if enable_head:
            self._channels.append(self._make_channel(
                'head',
                joint_names=head_joint_names,
                positions=head_positions,
                default_positions=[0.0, 0.0],
                qos_profile=qos_profile,
            ))

        if enable_arms:
            self._channels.append(self._make_channel(
                'arm_left',
                joint_names=left_joint_names,
                positions=left_positions,
                default_positions=[0.0] * 8,
                qos_profile=qos_profile,
            ))
            self._channels.append(self._make_channel(
                'arm_right',
                joint_names=right_joint_names,
                positions=right_positions,
                default_positions=[0.0] * 8,
                qos_profile=qos_profile,
            ))

        if enable_lift:
            # lift_position is a scalar at the UI/XML layer — _coerce_positions
            # wraps it so the channel sees a list.
            lift_jn = [lift_joint_name] if lift_joint_name else None
            self._channels.append(self._make_channel(
                'lift',
                joint_names=lift_jn,
                positions=lift_position,
                default_positions=[0.0],
                qos_profile=qos_profile,
            ))

        if not self._channels:
            raise ValueError(
                'JointControl: enable at least one of enable_head, '
                'enable_arms, enable_lift.'
            )

        # Single duration shared across channels. If the user didn't pass
        # one, take the slowest enabled group's default — being a hair
        # too slow is always safer than racing to finish.
        if duration is not None:
            self.duration = float(duration)
        else:
            self.duration = max(
                GROUP_DEFAULTS[ch['group']]['duration']
                for ch in self._channels
            )
        self._timeout_ticks = max(
            GROUP_DEFAULTS[ch['group']]['timeout_ticks']
            for ch in self._channels
        )

        self.joint_state = None
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile,
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._result = None  # None=running, True=success, False=failure
        self._control_rate = CONTROL_RATE_HZ  # noqa: F405

    # -- Construction helpers ------------------------------------------------

    def _make_channel(self, group, joint_names, positions, default_positions,
                      qos_profile):
        cfg = GROUP_DEFAULTS[group]
        pub = self.node.create_publisher(
            JointTrajectory, cfg['topic'], qos_profile,
        )
        return {
            'group': group,
            'pub': pub,
            'joint_names': joint_names or cfg['joint_names'],
            'positions': (
                _coerce_positions(positions)
                if positions is not None
                else list(default_positions)
            ),
        }

    # -- ROS callbacks --------------------------------------------------------

    def _joint_state_callback(self, msg):
        self.joint_state = msg

    # -- Control loop ---------------------------------------------------------

    def _publish_channel(self, channel):
        traj = JointTrajectory()
        traj.joint_names = list(channel['joint_names'])
        point = JointTrajectoryPoint()
        point.positions = list(channel['positions'])
        point.time_from_start.sec = int(self.duration)
        traj.points.append(point)
        channel['pub'].publish(traj)

    def _channel_reached(self, channel, name_to_idx) -> bool:
        for jname, target in zip(channel['joint_names'], channel['positions']):
            idx = name_to_idx.get(jname)
            if idx is None:
                self.log_warn(f"Joint '{jname}' not found in /joint_states")
                return False
            if abs(self.joint_state.position[idx] - target) > self.position_threshold:
                return False
        return True

    def _control_loop(self):
        rate_sleep = RATE_SLEEP_SEC  # noqa: F405

        # Fire every active group's trajectory once up front, then watch
        # /joint_states until they all converge (or we time out).
        for ch in self._channels:
            self._publish_channel(ch)
        groups = ', '.join(ch['group'] for ch in self._channels)
        self.log_info(f'JointControl trajectory published [{groups}]')

        timeout_count = 0
        while (
            not self._stop_event.is_set()
            and timeout_count < self._timeout_ticks
        ):
            if self.joint_state is None:
                time.sleep(rate_sleep)
                timeout_count += 1
                continue

            name_to_idx = {
                n: i for i, n in enumerate(self.joint_state.name)
            }
            if all(
                self._channel_reached(ch, name_to_idx)
                for ch in self._channels
            ):
                self.log_info(
                    f'JointControl reached target positions [{groups}]'
                )
                with self._lock:
                    self._result = True
                return

            time.sleep(rate_sleep)
            timeout_count += 1

        with self._lock:
            self._result = False
        self.log_error(
            f'JointControl timeout waiting for target positions [{groups}]'
        )

    # -- BT plumbing ----------------------------------------------------------

    def tick(self) -> NodeStatus:
        if self._thread is None:
            self.joint_state = None
            self._stop_event.clear()
            with self._lock:
                self._result = None
            self._thread = threading.Thread(
                target=self._control_loop, daemon=True,
            )
            self._thread.start()
            groups = ', '.join(ch['group'] for ch in self._channels)
            self.log_info(f'JointControl thread started [{groups}]')
            return NodeStatus.RUNNING

        with self._lock:
            result = self._result
        if result is None:
            return NodeStatus.RUNNING
        return NodeStatus.SUCCESS if result else NodeStatus.FAILURE

    def reset(self):
        super().reset()
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)  # noqa: F405
        self._thread = None
        with self._lock:
            self._result = None
        self.joint_state = None
