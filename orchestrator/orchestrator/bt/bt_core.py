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

"""Core classes for Behavior Tree implementation."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class NodeStatus(Enum):
    """Enum representing the status of a behavior tree node."""

    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class BTNode:
    """Base class for all behavior tree nodes."""

    def __init__(self, node: 'Node', name: str):
        """Initialize a behavior tree node."""
        self.node = node
        self.name = name
        self.uid = ''
        self.status = NodeStatus.RUNNING

    def tick(self) -> NodeStatus:
        """Execute the node's behavior and return status."""
        raise NotImplementedError('Subclasses must implement tick() method')

    def reset(self):
        """Reset the node to its initial state."""
        self.status = NodeStatus.RUNNING

    def get_active_node_ids(self):
        """Return UIDs of currently active nodes (leaf default)."""
        return [self.uid]

    def log_info(self, message: str):
        """Log an info message with the node name prefix."""
        self.node.get_logger().info(f'[{self.name}] {message}')

    def log_warn(self, message: str):
        """Log a warning message with the node name prefix."""
        self.node.get_logger().warn(f'[{self.name}] {message}')

    def log_error(self, message: str):
        """Log an error message with the node name prefix."""
        self.node.get_logger().error(f'[{self.name}] {message}')
