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
    """Repeat a Sequence-like run of children N times (or forever).

    Behaves like a Sequence within one iteration — ticks children left to
    right and bails out on the first FAILURE — but on completing the full
    children list once, that counts as one iteration. max_iterations <= 0
    keeps looping forever (the original behavior). Any positive integer
    caps the run to that many full passes; the Nth completed pass returns
    SUCCESS up to the parent.
    """

    def __init__(self, node: 'Node', name: str = 'Loop', max_iterations: int = 0):
        """Initialize the Loop control node."""
        super().__init__(node, name)
        try:
            self.max_iterations = int(max_iterations)
        except (TypeError, ValueError):
            self.max_iterations = 0
        self._iteration_count = 0
        self._current_child_index = 0

    def get_active_node_ids(self):
        """Return the currently active leaf node UID."""
        if not self.children:
            return []
        # Mid-iteration: highlight the child that's running. After a full
        # pass the index has rolled back to 0 so we point at the first
        # child by convention.
        idx = (
            self._current_child_index
            if self._current_child_index < len(self.children)
            else 0
        )
        return self.children[idx].get_active_node_ids()

    def tick(self) -> NodeStatus:
        """Sequence-tick children; one full pass = one iteration."""
        if not self.children:
            self.log_error('No child node')
            return NodeStatus.FAILURE

        while self._current_child_index < len(self.children):
            child = self.children[self._current_child_index]
            status = child.tick()

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            if status == NodeStatus.FAILURE:
                self.log_warn(f'Child {child.name} failed, stopping')
                child.reset()
                return NodeStatus.FAILURE

            # SUCCESS — advance to the next child within this iteration.
            child.reset()
            self._current_child_index += 1

        # Finished one full pass of all children.
        self._iteration_count += 1
        progress = (
            f'{self._iteration_count}/{self.max_iterations}'
            if self.max_iterations > 0
            else f'{self._iteration_count}'
        )
        if self.max_iterations > 0 and self._iteration_count >= self.max_iterations:
            self.log_info(f'Reached max_iterations={self.max_iterations}, stopping')
            return NodeStatus.SUCCESS

        # Roll back the index so the next tick starts the next iteration.
        self._current_child_index = 0
        self.log_info(f'Iteration {progress} complete, restarting')
        return NodeStatus.RUNNING

    def reset(self):
        """Reset children + iteration counter + child index."""
        super().reset()
        self._iteration_count = 0
        self._current_child_index = 0
