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

"""Loader for behavior trees from XML files."""

import xml.etree.ElementTree as ET  # noqa: I100
from typing import Dict  # noqa: I100
from typing import TYPE_CHECKING  # noqa: I100
from typing import Type  # noqa: I100

from orchestrator.bt.actions import JointControl
from orchestrator.bt.actions import Rotate
from orchestrator.bt.actions import SendCommandAction
from orchestrator.bt.actions import Wait
from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import BTNode
from orchestrator.bt.controls import Loop
from orchestrator.bt.controls import Sequence
from orchestrator.bt.controls.base_control import BaseControl

if TYPE_CHECKING:
    from rclpy.node import Node


class TreeLoader:
    """Loads behavior trees from XML files and instantiates nodes."""

    def __init__(
        self, node: 'Node', joint_names: list = None, topic_config: dict = None
    ):
        """Initialize the tree loader."""
        self.node = node
        self.joint_names = joint_names or []
        self.topic_config = topic_config or {}

        self._node_counter = 0

        self.control_types: Dict[str, Type[BaseControl]] = {
            'Sequence': Sequence,
            'Loop': Loop,
        }

        self.action_types: Dict[str, Type[BaseAction]] = {
            'Rotate': Rotate,
            'JointControl': JointControl,
            'SendCommand': SendCommandAction,
            'Wait': Wait,
        }

    def load_tree_from_string(
        self, xml_string: str, main_tree_id: str = None
    ) -> BTNode:
        """Load a behavior tree from an XML string."""
        self._node_counter = 0
        root = ET.fromstring(xml_string)
        return self._load_tree_from_root(root, main_tree_id)

    def load_tree_from_file(
        self, xml_path: str, main_tree_id: str = None
    ) -> BTNode:
        """Load a behavior tree from an XML file."""
        self._node_counter = 0
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return self._load_tree_from_root(root, main_tree_id)

    def _load_tree_from_root(
        self, root: ET.Element, main_tree_id: str = None
    ) -> BTNode:
        """Load a behavior tree from a parsed XML root element."""
        if main_tree_id is None:
            main_tree_id = root.get('main_tree_to_execute')
            if not main_tree_id:
                raise ValueError(
                    'No main_tree_to_execute specified in XML'
                )

        for behavior_tree in root.findall('BehaviorTree'):
            if behavior_tree.get('ID') == main_tree_id:
                return self._load_node(behavior_tree[0])

        raise ValueError(
            f"BehaviorTree with ID '{main_tree_id}' not found"
        )

    def _load_node(self, xml_node: ET.Element) -> BTNode:
        """Load a behavior tree node from an XML element."""
        node_type = xml_node.tag
        node_id = xml_node.get('ID', node_type)
        node_name = xml_node.get('name', node_id)

        uid = f'bt_{self._node_counter}'
        self._node_counter += 1

        if node_type in self.control_types:
            control_class = self.control_types[node_type]
            # Control nodes whose XML carries extra params (e.g. Loop's
            # max_iterations) need those threaded through the ctor. The
            # generic _parse_node_params skips 'ID' and 'name', so any
            # remaining attributes are valid control-node kwargs. UI-only
            # metadata (bt_x / bt_y / bt_collapsed) is filtered out here
            # so it never reaches the Python ctor signature.
            raw_params = self._parse_node_params(xml_node)
            ui_only = ('bt_x', 'bt_y', 'bt_collapsed')
            control_params = {k: v for k, v in raw_params.items() if k not in ui_only}
            control_node = control_class(self.node, name=node_name, **control_params)
            control_node.uid = uid

            for child_xml in xml_node:
                child_node = self._load_node(child_xml)
                control_node.add_child(child_node)

            return control_node

        elif node_id in self.action_types:
            action_class = self.action_types[node_id]
            params = self._parse_node_params(xml_node)
            action = self._create_action(action_class, node_name, params)
            action.uid = uid
            return action

        else:
            raise ValueError(
                f"Unknown node type '{node_type}' with ID '{node_id}'"
            )

    def _parse_node_params(self, xml_node: ET.Element) -> Dict:
        """Parse parameters from XML node attributes."""
        params = {}

        for key, value in xml_node.attrib.items():
            if key not in ['ID', 'name']:
                params[key] = self._convert_value(value)

        return params

    def _convert_value(self, value: str):
        """Convert string value to appropriate Python type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        if ',' in value:
            parts = [p.strip() for p in value.split(',')]
            try:
                return [float(p) if '.' in p else int(p) for p in parts]
            except ValueError:
                return parts

        return value

    def _get_joint_names_for_group(self, group_name: str) -> list:
        """Get joint names for a specific joint group from topic_config."""
        if not self.topic_config or 'joint_order' not in self.topic_config:
            return []

        joint_order = self.topic_config['joint_order']
        return joint_order.get(group_name, [])

    def _create_action(
        self, action_class: Type[BaseAction], name: str, params: Dict
    ) -> BaseAction:
        """Create an action node instance with the given parameters."""
        if action_class == Rotate:
            action = action_class(
                node=self.node,
                angle_deg=params.get('angle_deg', 90.0),
                topic_config=self.topic_config
            )
            action.name = name
            return action

        elif action_class == JointControl:
            # JointControl now toggles each sub-group (head / arms / lift)
            # independently via enable_* flags. We only forward kwargs for
            # the groups that are actually enabled so the constructor's
            # at-least-one-enabled invariant + ValueError surface clearly
            # if someone drops a misconfigured node.
            enable_head = bool(params.get('enable_head', False))
            enable_arms = bool(params.get('enable_arms', False))
            enable_lift = bool(params.get('enable_lift', False))

            kwargs = {
                'node': self.node,
                'enable_head': enable_head,
                'enable_arms': enable_arms,
                'enable_lift': enable_lift,
            }

            if enable_head:
                kwargs['head_positions'] = params.get(
                    'head_positions', [0.0, 0.0],
                )
                head_joints = self._get_joint_names_for_group('leader_head')
                if head_joints:
                    kwargs['head_joint_names'] = head_joints

            if enable_arms:
                default_positions = [0.0] * 8
                kwargs['left_positions'] = params.get(
                    'left_positions', default_positions,
                )
                kwargs['right_positions'] = params.get(
                    'right_positions', default_positions,
                )
                left_joints = self._get_joint_names_for_group('leader_left')
                right_joints = self._get_joint_names_for_group('leader_right')
                if left_joints:
                    kwargs['left_joint_names'] = left_joints
                if right_joints:
                    kwargs['right_joint_names'] = right_joints

            if enable_lift:
                kwargs['lift_position'] = params.get('lift_position', 0.0)
                lift_joints = self._get_joint_names_for_group('leader_lift')
                if lift_joints:
                    kwargs['lift_joint_name'] = lift_joints[0]

            duration = params.get('duration')
            if duration is not None:
                kwargs['duration'] = duration
            # position_threshold isn't exposed in the UI catalog, but if a
            # hand-edited XML supplies it we still honor it; otherwise the
            # JointControl class default (0.01) applies.
            position_threshold = params.get('position_threshold')
            if position_threshold is not None:
                kwargs['position_threshold'] = position_threshold

            action = action_class(**kwargs)
            action.name = name
            return action

        elif action_class == SendCommandAction:
            task_instruction = params.get('task_instruction', '')
            if isinstance(task_instruction, list):
                task_instruction = ', '.join(task_instruction)
            action = action_class(
                node=self.node,
                command=params.get('command', 'LOAD'),
                model=params.get('model', ''),
                policy_path=params.get('policy_path', ''),
                task_instruction=task_instruction,
                inference_hz=params.get('inference_hz', 15),
                control_hz=params.get('control_hz', 100),
                chunk_align_window_s=params.get('chunk_align_window_s', 0.3),
            )
            action.name = name
            return action

        elif action_class == Wait:
            action = action_class(
                node=self.node,
                duration=params.get('duration', 5.0),
            )
            action.name = name
            return action

        else:
            raise ValueError(f'Unknown action class: {action_class}')
