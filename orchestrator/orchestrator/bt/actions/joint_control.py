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

"""Rule-based joint-space control backed by the active robot config.

The node supports two XML styles:

* Generic, robot-config driven:
  ``groups="arm" positions="0.0, 0.0, ..."``
  where each group maps to ``action.<group>`` in the robot yaml.

* Legacy FFW trees:
  ``enable_head``, ``enable_arms``, ``enable_lift`` plus the matching
  positions fields. These group names still resolve through the robot yaml,
  so no FFW topic or joint-name defaults live in this action anymore.
"""

import threading
import time
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


JOINT_STATE_TYPE = 'sensor_msgs/msg/JointState'
JOINT_TRAJECTORY_TYPE = 'trajectory_msgs/msg/JointTrajectory'


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _coerce_positions(value) -> Optional[list[float]]:
    """Normalise a positions param into ``list[float]``."""
    if value is None:
        return None
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(',') if x.strip()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value if str(x).strip()]
    return [float(value)]


def _parse_groups(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(',')
    return [str(part).strip() for part in parts if str(part).strip()]


def _parse_position_groups(value) -> list[list[float]]:
    if not _has_value(value):
        return []
    if isinstance(value, str):
        return [
            _coerce_positions(group_text) or []
            for group_text in value.split(';')
            if group_text.strip()
        ]
    if isinstance(value, (list, tuple)):
        if any(';' in str(item) for item in value):
            return _parse_position_groups(','.join(str(item) for item in value))
        return [_coerce_positions(value) or []]
    return [_coerce_positions(value) or []]


def _duration_for_group(group: str) -> float:
    if group == 'head':
        return DEFAULT_MOVE_HEAD_DURATION_SEC  # noqa: F405
    if group == 'lift':
        return DEFAULT_MOVE_LIFT_DURATION_SEC  # noqa: F405
    return DEFAULT_MOVE_ARMS_DURATION_SEC  # noqa: F405


def _timeout_for_group(group: str) -> int:
    if group == 'head':
        return MOVE_HEAD_TIMEOUT_TICKS  # noqa: F405
    if group == 'lift':
        return MOVE_LIFT_TIMEOUT_TICKS  # noqa: F405
    return MOVE_ARMS_TIMEOUT_TICKS  # noqa: F405


class JointControl(BaseAction):
    """Publish JointTrajectory commands and wait for configured state feedback."""

    @classmethod
    def from_xml_params(cls, context, name: str, params: dict):
        kwargs = {
            'node': context.node,
            'topic_config': context.topic_config,
        }

        if _has_value(params.get('groups')):
            kwargs['groups'] = params.get('groups')
            kwargs['positions'] = params.get('positions', '')
        else:
            enable_head = _as_bool(params.get('enable_head', False))
            enable_arms = _as_bool(params.get('enable_arms', False))
            enable_lift = _as_bool(params.get('enable_lift', False))

            kwargs.update({
                'enable_head': enable_head,
                'enable_arms': enable_arms,
                'enable_lift': enable_lift,
            })

            if enable_head:
                kwargs['head_positions'] = params.get(
                    'head_positions', [0.0, 0.0],
                )
            if enable_arms:
                default_positions = [0.0] * 8
                kwargs['left_positions'] = params.get(
                    'left_positions', default_positions,
                )
                kwargs['right_positions'] = params.get(
                    'right_positions', default_positions,
                )
            if enable_lift:
                kwargs['lift_position'] = params.get('lift_position', 0.0)

        duration = params.get('duration')
        if duration is not None:
            kwargs['duration'] = duration
        position_threshold = params.get('position_threshold')
        if position_threshold is not None:
            kwargs['position_threshold'] = position_threshold

        action = cls(**kwargs)
        action.name = name
        return action

    def __init__(
        self,
        node: 'Node',
        groups: str = '',
        positions: str = '',
        enable_head: bool = True,
        head_positions='0.0, 0.0',
        head_joint_names: Optional[list[str]] = None,
        enable_arms: bool = False,
        left_positions: Optional[list[float]] = None,
        right_positions: Optional[list[float]] = None,
        left_joint_names: Optional[list[str]] = None,
        right_joint_names: Optional[list[str]] = None,
        enable_lift: bool = False,
        lift_position: float = 0.0,
        lift_joint_name: Optional[str] = None,
        duration: Optional[float] = 2.0,
        position_threshold: float = POSITION_THRESHOLD_RAD,  # noqa: F405
        topic_config: Optional[dict] = None,
    ):
        super().__init__(node, name='JointControl')

        self.position_threshold = position_threshold
        self.topic_config = topic_config or {}
        self._missing_joint_names = set()

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,  # noqa: F405
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._channels = []
        group_names = _parse_groups(groups)
        if group_names:
            self._build_generic_channels(
                group_names,
                _parse_position_groups(positions),
                qos_profile,
            )
        else:
            self._build_legacy_channels(
                enable_head=enable_head,
                head_positions=head_positions,
                head_joint_names=head_joint_names,
                enable_arms=enable_arms,
                left_positions=left_positions,
                right_positions=right_positions,
                left_joint_names=left_joint_names,
                right_joint_names=right_joint_names,
                enable_lift=enable_lift,
                lift_position=lift_position,
                lift_joint_name=lift_joint_name,
                qos_profile=qos_profile,
            )

        if not self._channels:
            raise ValueError(
                'JointControl: set groups or enable at least one legacy '
                'group (enable_head, enable_arms, enable_lift).'
            )

        if duration is not None:
            self.duration = float(duration)
        else:
            self.duration = max(
                _duration_for_group(ch['group'])
                for ch in self._channels
            )
        self._timeout_ticks = max(
            _timeout_for_group(ch['group'])
            for ch in self._channels
        )

        self._joint_state_topics = self._configured_joint_state_topics()
        self._joint_states = self._empty_joint_states()
        self.joint_state_subscriptions = []
        for topic in self._joint_state_topics:
            sub = self.node.create_subscription(
                JointState,
                topic,
                lambda msg, topic_name=topic: self._joint_state_callback(
                    topic_name, msg
                ),
                qos_profile,
            )
            self.joint_state_subscriptions.append(sub)

        if not self.joint_state_subscriptions:
            self.log_warn(
                'JointControl: no JointState feedback topics configured; '
                'target convergence cannot be verified.'
            )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._result = None
        self._control_rate = CONTROL_RATE_HZ  # noqa: F405

    # -- Construction helpers ------------------------------------------------

    def _build_generic_channels(
        self,
        group_names: list[str],
        position_groups: list[list[float]],
        qos_profile,
    ) -> None:
        if len(position_groups) > len(group_names):
            raise ValueError(
                'JointControl: positions contains more groups than groups'
            )

        for index, group in enumerate(group_names):
            positions = (
                position_groups[index]
                if index < len(position_groups)
                else None
            )
            self._channels.append(self._make_channel(
                group=group,
                joint_names=None,
                positions=positions,
                default_positions=None,
                qos_profile=qos_profile,
            ))

    def _build_legacy_channels(
        self,
        *,
        enable_head,
        head_positions,
        head_joint_names,
        enable_arms,
        left_positions,
        right_positions,
        left_joint_names,
        right_joint_names,
        enable_lift,
        lift_position,
        lift_joint_name,
        qos_profile,
    ) -> None:
        if enable_head:
            self._channels.append(self._make_channel(
                group='head',
                joint_names=head_joint_names,
                positions=head_positions,
                default_positions=[0.0, 0.0],
                qos_profile=qos_profile,
            ))

        if enable_arms:
            self._channels.append(self._make_channel(
                group='arm_left',
                joint_names=left_joint_names,
                positions=left_positions,
                default_positions=[0.0] * 8,
                qos_profile=qos_profile,
            ))
            self._channels.append(self._make_channel(
                group='arm_right',
                joint_names=right_joint_names,
                positions=right_positions,
                default_positions=[0.0] * 8,
                qos_profile=qos_profile,
            ))

        if enable_lift:
            lift_joints = [lift_joint_name] if lift_joint_name else None
            self._channels.append(self._make_channel(
                group='lift',
                joint_names=lift_joints,
                positions=lift_position,
                default_positions=[0.0],
                qos_profile=qos_profile,
            ))

    def _resolve_action_group_key(self, group: str) -> str:
        topic_map = self.topic_config.get('topic_map', {})
        candidates = []
        if group.startswith('leader_'):
            candidates.append(group)
            logical_group = group[len('leader_'):]
        else:
            logical_group = group
            candidates.append(f'leader_{group}')
        candidates.append(group)

        for candidate in candidates:
            if candidate in topic_map:
                return candidate

        configured = sorted(
            key[len('leader_'):]
            for key in topic_map
            if key.startswith('leader_')
        )
        raise ValueError(
            f"JointControl: action group '{logical_group}' is not configured "
            f'(configured groups: {configured})'
        )

    def _make_channel(
        self,
        group,
        joint_names,
        positions,
        default_positions,
        qos_profile,
    ):
        group_key = self._resolve_action_group_key(group)
        topic = self.topic_config.get('topic_map', {}).get(group_key)
        msg_type = self.topic_config.get('topic_type_map', {}).get(
            group_key,
            JOINT_TRAJECTORY_TYPE,
        )
        if msg_type != JOINT_TRAJECTORY_TYPE:
            raise ValueError(
                f"JointControl: group '{group}' uses {msg_type}, "
                f'expected {JOINT_TRAJECTORY_TYPE}'
            )

        resolved_joint_names = list(
            joint_names
            or self.topic_config.get('joint_order', {}).get(group_key, [])
        )
        if not resolved_joint_names:
            raise ValueError(
                f"JointControl: no joint names configured for '{group_key}'"
            )

        if _has_value(positions):
            target_positions = _coerce_positions(positions) or []
        elif (
            default_positions is not None
            and len(default_positions) == len(resolved_joint_names)
        ):
            target_positions = list(default_positions)
        else:
            target_positions = [0.0] * len(resolved_joint_names)

        if len(target_positions) != len(resolved_joint_names):
            raise ValueError(
                f"JointControl: group '{group}' has "
                f'{len(resolved_joint_names)} joints but '
                f'{len(target_positions)} target positions'
            )

        pub = self.node.create_publisher(
            JointTrajectory,
            topic,
            qos_profile,
        )
        return {
            'group': (
                group_key[len('leader_'):]
                if group_key.startswith('leader_')
                else group
            ),
            'group_key': group_key,
            'topic': topic,
            'pub': pub,
            'joint_names': resolved_joint_names,
            'positions': target_positions,
        }

    def _configured_joint_state_topics(self) -> list[str]:
        topic_map = self.topic_config.get('topic_map', {})
        topic_type_map = self.topic_config.get('topic_type_map', {})
        topics = []
        for group, topic in topic_map.items():
            if (
                group.startswith('follower_')
                and topic_type_map.get(group) == JOINT_STATE_TYPE
                and topic not in topics
            ):
                topics.append(topic)
        return topics

    def _empty_joint_states(self) -> dict[str, Optional[JointState]]:
        return {topic: None for topic in self._joint_state_topics}

    # -- ROS callbacks --------------------------------------------------------

    def _joint_state_callback(self, topic_name, msg):
        if topic_name in self._joint_states:
            self._joint_states[topic_name] = msg

    # -- Control loop ---------------------------------------------------------

    def _publish_channel(self, channel):
        traj = JointTrajectory()
        traj.joint_names = list(channel['joint_names'])
        point = JointTrajectoryPoint()
        point.positions = list(channel['positions'])
        point.time_from_start.sec = int(self.duration)
        traj.points.append(point)
        channel['pub'].publish(traj)

    def _name_to_position(self) -> dict[str, float]:
        result = {}
        for msg in self._joint_states.values():
            if msg is None:
                continue
            for idx, name in enumerate(msg.name):
                if idx < len(msg.position):
                    result[name] = msg.position[idx]
        return result

    def _channel_reached(self, channel, name_to_position) -> bool:
        for jname, target in zip(channel['joint_names'], channel['positions']):
            if jname not in name_to_position:
                if jname not in self._missing_joint_names:
                    self._missing_joint_names.add(jname)
                    self.log_warn(
                        f"Joint '{jname}' not found in configured "
                        'JointState feedback topics'
                    )
                return False
            if abs(name_to_position[jname] - target) > self.position_threshold:
                return False
        return True

    def _control_loop(self):
        rate_sleep = RATE_SLEEP_SEC  # noqa: F405

        for ch in self._channels:
            self._publish_channel(ch)
        groups = ', '.join(ch['group'] for ch in self._channels)
        self.log_info(f'JointControl trajectory published [{groups}]')

        timeout_count = 0
        while (
            not self._stop_event.is_set()
            and timeout_count < self._timeout_ticks
        ):
            if not any(msg is not None for msg in self._joint_states.values()):
                time.sleep(rate_sleep)
                timeout_count += 1
                continue

            name_to_position = self._name_to_position()
            if all(
                self._channel_reached(ch, name_to_position)
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
            self._joint_states = self._empty_joint_states()
            self._missing_joint_names = set()
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
        self._joint_states = self._empty_joint_states()
