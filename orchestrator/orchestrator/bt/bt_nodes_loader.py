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
import inspect  # noqa: I100
from typing import TYPE_CHECKING  # noqa: I100

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import BTNode
from orchestrator.bt.controls.base_control import BaseControl
from orchestrator.bt.node_registry import build_registry

if TYPE_CHECKING:
    from rclpy.node import Node

_RESERVED_XML_ATTRS = frozenset({
    'ID',
    'name',
    'bt_x',
    'bt_y',
    'bt_collapsed',
})


class LoaderContext:
    """Runtime dependencies and helpers available to node factories."""

    def __init__(self, loader: 'TreeLoader'):
        self.node = loader.node
        self.joint_names = loader.joint_names
        self.topic_config = loader.topic_config
        self.get_joint_names_for_group = loader._get_joint_names_for_group
        self.get_topic_for_group = loader._get_topic_for_group
        self.get_topic_type_for_group = loader._get_topic_type_for_group
        self.get_joint_state_topics = loader._get_joint_state_topics


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
        self._registry = {}
        self.context = LoaderContext(self)

    def load_tree_from_string(
        self, xml_string: str, main_tree_id: str = None
    ) -> BTNode:
        """Load a behavior tree from an XML string."""
        self._node_counter = 0
        self._registry = build_registry()
        root = ET.fromstring(xml_string)
        return self._load_tree_from_root(root, main_tree_id)

    def load_tree_from_file(
        self, xml_path: str, main_tree_id: str = None
    ) -> BTNode:
        """Load a behavior tree from an XML file."""
        self._node_counter = 0
        self._registry = build_registry()
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

        node_class = self._registry.get(node_type) or self._registry.get(node_id)
        if node_class is None:
            raise ValueError(
                f"Unknown node type '{node_type}' with ID '{node_id}'"
            )

        params = self._parse_node_params(xml_node)

        if issubclass(node_class, BaseControl):
            control_node = self._create_node(node_class, node_name, params)
            control_node.uid = uid

            for child_xml in xml_node:
                child_node = self._load_node(child_xml)
                control_node.add_child(child_node)

            return control_node

        elif issubclass(node_class, BaseAction):
            action = self._create_node(node_class, node_name, params)
            action.uid = uid
            return action

        raise ValueError(f"Unsupported BT node class for '{node_type}'")

    def _parse_node_params(self, xml_node: ET.Element) -> dict:
        """Parse parameters from XML node attributes."""
        params = {}

        for key, value in xml_node.attrib.items():
            if key not in _RESERVED_XML_ATTRS:
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

    def _get_topic_for_group(self, group_name: str) -> str:
        """Get topic name for a specific runtime group from topic_config."""
        if not self.topic_config or 'topic_map' not in self.topic_config:
            return ''
        return self.topic_config['topic_map'].get(group_name, '')

    def _get_topic_type_for_group(self, group_name: str) -> str:
        """Get ROS message type for a specific runtime group."""
        if not self.topic_config or 'topic_type_map' not in self.topic_config:
            return ''
        return self.topic_config['topic_type_map'].get(group_name, '')

    def _get_joint_state_topics(self) -> list[str]:
        """Return configured state topics publishing JointState messages."""
        if not self.topic_config:
            return []
        topic_map = self.topic_config.get('topic_map', {})
        topic_type_map = self.topic_config.get('topic_type_map', {})
        topics = []
        for group, topic in topic_map.items():
            if (
                group.startswith('follower_')
                and topic_type_map.get(group) == 'sensor_msgs/msg/JointState'
                and topic not in topics
            ):
                topics.append(topic)
        return topics

    def _create_node(self, node_class, name: str, params: dict) -> BTNode:
        """Create a BT node using a class factory or generic ctor wiring."""
        factory = getattr(node_class, 'from_xml_params', None)
        if callable(factory):
            node = factory(self.context, name, params)
            node.name = name
            return node

        kwargs = dict(params)
        sig = inspect.signature(node_class.__init__)
        ctor_params = sig.parameters
        if 'node' in ctor_params:
            kwargs['node'] = self.node
        if 'name' in ctor_params:
            kwargs['name'] = name
        kwargs = {
            key: value for key, value in kwargs.items()
            if key in ctor_params
        }

        node = node_class(**kwargs)
        node.name = name
        return node
