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

"""Loop control node for behavior trees."""

from typing import TYPE_CHECKING

from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.controls.base_control import BaseControl

if TYPE_CHECKING:
    from rclpy.node import Node


class Loop(BaseControl):
    """Repeat a single child forever, returning RUNNING on each success."""

    def __init__(self, node: 'Node', name: str = 'Loop'):
        """Initialize the Loop control node."""
        super().__init__(node, name)

    def get_active_node_ids(self):
        """Return the currently active leaf node UID."""
        if self.children:
            return self.children[0].get_active_node_ids()
        return []

    def tick(self) -> NodeStatus:
        """Tick the child; reset and repeat on SUCCESS, propagate FAILURE."""
        if not self.children:
            self.log_error('No child node')
            return NodeStatus.FAILURE

        child = self.children[0]
        status = child.tick()

        if status == NodeStatus.RUNNING:
            return NodeStatus.RUNNING
        elif status == NodeStatus.SUCCESS:
            self.log_info(f'Child {child.name} succeeded, restarting')
            child.reset()
            return NodeStatus.RUNNING
        else:
            self.log_warn(f'Child {child.name} failed, stopping')
            child.reset()
            return NodeStatus.FAILURE
