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

"""Constants for behavior tree actions."""

# Control Loop Parameters
CONTROL_RATE_HZ = 100  # Control loop frequency in Hz
RATE_SLEEP_SEC = 1.0 / CONTROL_RATE_HZ  # Sleep duration per control cycle
THREAD_JOIN_TIMEOUT_SEC = 1.0  # Timeout for thread cleanup

# QoS Configuration
QOS_QUEUE_DEPTH = 10  # Queue depth for ROS 2 publishers/subscribers

# Position Thresholds
POSITION_THRESHOLD_RAD = 0.01  # Joint position tolerance in radians

# Action Timeouts (in ticks at CONTROL_RATE_HZ)
MOVE_ARMS_TIMEOUT_TICKS = 1500  # 15 seconds at 100Hz
MOVE_HEAD_TIMEOUT_TICKS = 1000  # 10 seconds at 100Hz
MOVE_LIFT_TIMEOUT_TICKS = 2000  # 20 seconds at 100Hz
ROTATE_INIT_TIMEOUT_TICKS = 500  # 5 seconds at 100Hz

# Rotation Parameters
ROTATION_ANGULAR_VELOCITY = 0.6  # rad/s (max angular velocity)
ROTATION_KP = 0.02  # Proportional gain for rotation control
ROTATION_MIN_ANGULAR_VELOCITY = 0.05  # Minimum angular velocity to prevent stalling
ROTATION_TOLERANCE_DEG = 0.1  # degrees
DEFAULT_ROTATION_ANGLE_DEG = 90.0  # degrees
ANGLE_NORMALIZATION_180 = 180
ANGLE_NORMALIZATION_360 = 360

# Inference Until Gripper Parameters
GRIPPER_OPEN_THRESHOLD = 0.3  # Position below which gripper is open (0.0 = fully open)
GRIPPER_CLOSED_THRESHOLD = 0.7  # Position above which gripper is closed (1.0 = fully closed)
INFERENCE_POSITION_CHANGE_THRESHOLD = 0.05  # Max position change (rad) for stability
INFERENCE_STATIC_DURATION = 3.0  # Seconds arms must be stable
INFERENCE_HISTORY_WINDOW = 1.0  # Rolling window (seconds) for change calculation

# Default Durations
DEFAULT_MOVE_ARMS_DURATION_SEC = 2.0
DEFAULT_MOVE_HEAD_DURATION_SEC = 5.0
DEFAULT_MOVE_LIFT_DURATION_SEC = 5.0

# Hardware Commands
ZERO_VELOCITY = 0.0  # Zero velocity command for stopping
