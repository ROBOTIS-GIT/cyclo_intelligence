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

"""Action node for rotating the mobile base by a specified angle."""

import math
import threading
import time
from typing import TYPE_CHECKING

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.constants import *  # noqa: F403
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

if TYPE_CHECKING:
    from rclpy.node import Node


class Rotate(BaseAction):
    """Action to rotate the mobile base by a target angle in degrees."""

    @staticmethod
    def angle_diff_deg(a, b):
        """Calculate the difference between two angles in degrees."""
        d = a - b
        while d > ANGLE_NORMALIZATION_180:  # noqa: F405
            d -= ANGLE_NORMALIZATION_360  # noqa: F405
        while d < -ANGLE_NORMALIZATION_180:  # noqa: F405
            d += ANGLE_NORMALIZATION_360  # noqa: F405
        return d

    def __init__(
            self,
            node: 'Node',
            angle_deg: float = DEFAULT_ROTATION_ANGLE_DEG,  # noqa: F405
            topic_config: dict = None
    ):
        """Initialize the Rotate action."""
        super().__init__(node, name='Rotate')
        self.angle_deg = angle_deg
        self.topic_config = topic_config or {}
        if not isinstance(self.topic_config, dict):
            self.topic_config = {}
        self.angular_velocity = ROTATION_ANGULAR_VELOCITY  # noqa: F405
        self.kp = ROTATION_KP  # noqa: F405
        self.min_angular_velocity = ROTATION_MIN_ANGULAR_VELOCITY  # noqa: F405

        qos_profile = QoSProfile(
            depth=QOS_QUEUE_DEPTH,  # noqa: F405
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.publishers = {}
        if self.topic_config and 'topic_map' in self.topic_config:
            topic_map = self.topic_config['topic_map']
            for joint_group, topic in topic_map.items():
                if joint_group == 'leader_mobile':
                    pub = self.node.create_publisher(
                        Twist,
                        topic,
                        qos_profile
                    )
                    self.publishers[joint_group] = pub

        self.odom_sub = self.node.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            qos_profile
        )
        self.odom_start_yaw = None
        self.odom_last_yaw = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._result = None  # None=running, True=success, False=failure
        self._control_rate = CONTROL_RATE_HZ  # noqa: F405

    def _odom_callback(self, msg):
        """Receive odometry updates and compute yaw angle."""
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if self.odom_start_yaw is None:
            self.odom_start_yaw = yaw
        self.odom_last_yaw = yaw

    def _control_loop(self):
        """Control loop that publishes velocity and monitors rotation."""
        rate_sleep = RATE_SLEEP_SEC  # noqa: F405

        timeout_count = 0
        max_init_timeout = ROTATE_INIT_TIMEOUT_TICKS  # noqa: F405
        while (
            self.odom_start_yaw is None and timeout_count < max_init_timeout
        ):
            time.sleep(0.01)
            timeout_count += 1

        if self.odom_start_yaw is None:
            self.log_error('Timeout waiting for odom data')
            with self._lock:
                self._result = False
            return

        while not self._stop_event.is_set():
            if self.odom_last_yaw is None:
                time.sleep(rate_sleep)
                continue

            start_deg = math.degrees(self.odom_start_yaw)
            last_deg = math.degrees(self.odom_last_yaw)
            delta_deg = self.angle_diff_deg(last_deg, start_deg)
            delta_deg_norm = ((delta_deg + ANGLE_NORMALIZATION_180) %  # noqa: F405, E501
                              ANGLE_NORMALIZATION_360) - ANGLE_NORMALIZATION_180  # noqa: F405, E501

            tolerance = ROTATION_TOLERANCE_DEG  # noqa: F405
            error = self.angle_deg - delta_deg_norm

            if abs(error) <= tolerance:
                self._stop_mobile()
                norm_str = f'{delta_deg_norm:.2f}'
                target_str = str(self.angle_deg)
                msg = (
                    f'[Thread] Rotation complete: {norm_str} deg '
                    f'(target: {target_str} deg)'
                )
                self.log_info(msg)
                with self._lock:
                    self._result = True
                return

            angular_z = self.kp * error
            angular_z = max(-self.angular_velocity,
                            min(self.angular_velocity, angular_z))
            if 0 < abs(angular_z) < self.min_angular_velocity:
                angular_z = (self.min_angular_velocity if angular_z > 0
                             else -self.min_angular_velocity)

            if 'leader_mobile' in self.publishers:
                twist_msg = Twist()
                twist_msg.linear.x = ZERO_VELOCITY  # noqa: F405
                twist_msg.linear.y = ZERO_VELOCITY  # noqa: F405
                twist_msg.angular.z = angular_z
                self.publishers['leader_mobile'].publish(twist_msg)

            time.sleep(rate_sleep)

    def tick(self) -> NodeStatus:
        """Execute the action and return its status."""
        if self._thread is None:
            self.odom_start_yaw = None
            self.odom_last_yaw = None
            self._stop_event.clear()
            with self._lock:
                self._result = None

            self._thread = threading.Thread(
                target=self._control_loop, daemon=True
            )
            self._thread.start()
            angle_str = str(self.angle_deg)
            self.log_info(f'Rotate thread started (target: {angle_str} deg)')
            return NodeStatus.RUNNING

        with self._lock:
            result = self._result

        if result is None:
            return NodeStatus.RUNNING
        return NodeStatus.SUCCESS if result else NodeStatus.FAILURE

    def _stop_mobile(self):
        """Stop the mobile base by publishing zero velocity."""
        if 'leader_mobile' in self.publishers:
            twist_msg = Twist()
            twist_msg.linear.x = ZERO_VELOCITY  # noqa: F405
            twist_msg.linear.y = ZERO_VELOCITY  # noqa: F405
            twist_msg.angular.z = ZERO_VELOCITY  # noqa: F405
            self.publishers['leader_mobile'].publish(twist_msg)
            self.log_info('Mobile base stopped')

    def reset(self):
        """Reset the action to its initial state."""
        super().reset()
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)  # noqa: F405
        self._thread = None
        with self._lock:
            self._result = None
        self.odom_start_yaw = None
        self.odom_last_yaw = None
