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

"""Sequence control node for behavior trees."""

from typing import TYPE_CHECKING

from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.controls.base_control import BaseControl

if TYPE_CHECKING:
    from rclpy.node import Node


class Sequence(BaseControl):
    """Execute children sequentially until one fails or all succeed."""

    def __init__(self, node: 'Node', name: str = 'Sequence'):
        """Initialize the Sequence control node."""
        super().__init__(node, name)
        self.current_child_index = 0

    def tick(self) -> NodeStatus:
        """Execute children in sequence."""
        while self.current_child_index < len(self.children):
            current_child = self.children[self.current_child_index]
            status = current_child.tick()

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                self.log_warn(f'Child {current_child.name} failed')
                current_child.reset()
                return NodeStatus.FAILURE
            else:
                self.log_info(f'Child {current_child.name} succeeded')
                current_child.reset()
                self.current_child_index += 1

        self.log_info('All children succeeded')
        return NodeStatus.SUCCESS

    def get_active_node_ids(self):
        """Return the currently active leaf node UID."""
        if self.current_child_index < len(self.children):
            return self.children[self.current_child_index].get_active_node_ids()
        return []

    def reset(self):
        """Reset the sequence to start from the first child."""
        super().reset()
        self.current_child_index = 0
