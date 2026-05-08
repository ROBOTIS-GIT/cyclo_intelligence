#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Dongyun Kim

"""
robot_client - High-level abstraction for robot sensor data and control.

Provides RobotClient (sensor reading + action output) and
RobotServiceServer (training/inference service framework)
on top of zenoh_ros2_sdk.
"""
from .robot_client import RobotClient
from .service_server import RobotServiceServer, TrainingProgress

__all__ = ["RobotClient", "RobotServiceServer", "TrainingProgress"]
__version__ = "0.1.0"
