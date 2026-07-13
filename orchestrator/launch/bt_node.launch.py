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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from shared.robot_configs import schema as robot_schema


SUPPORTED_BT_ROBOT_TYPE = 'ffw_sg2_rev1'


def launch_setup(context, *args, **kwargs):

    robot_type = (
        LaunchConfiguration('robot_type').perform(context)
        or SUPPORTED_BT_ROBOT_TYPE
    )
    if robot_type != SUPPORTED_BT_ROBOT_TYPE:
        raise RuntimeError(
            'BT currently supports only '
            f'{SUPPORTED_BT_ROBOT_TYPE}; got {robot_type}'
        )

    shared_share = get_package_share_directory('shared')

    config_dir = os.path.join(shared_share, 'robot_configs')
    robot_config_path = os.path.join(config_dir, f'{robot_type}_config.yaml')

    if not os.path.exists(robot_config_path):
        raise FileNotFoundError(f'Config file not found: {robot_config_path}')

    # Phase 4 adapter — bt_node + bt_nodes_loader still consume the legacy
    # flat params (`<robot>.joint_list`, `<robot>.joint_topic_list`,
    # `<robot>.joint_order.<group>`). Translate the VLA-semantic schema
    # back into that shape here so the BT internals keep working without
    # leaking the new structure into them.
    section = robot_schema.load_robot_section(
        robot_type, explicit_path=robot_config_path,
    )

    state_groups = robot_schema.get_state_groups(section)
    action_groups = robot_schema.get_action_groups(section)

    joint_list = [f'leader_{m}' for m in action_groups]
    joint_topic_list = [
        f'follower_{name}:{cfg["topic"]}'
        for name, cfg in state_groups.items()
    ] + [
        f'leader_{m}:{cfg["topic"]}'
        for m, cfg in action_groups.items()
    ]
    joint_topic_type_list = [
        f'follower_{name}:{cfg["msg_type"]}'
        for name, cfg in state_groups.items()
    ] + [
        f'leader_{m}:{cfg["msg_type"]}'
        for m, cfg in action_groups.items()
    ]

    bt_params = {
        'robot_type': robot_type,
        'tick_rate': 30.0,
        f'{robot_type}.joint_list': joint_list,
        f'{robot_type}.joint_topic_list': joint_topic_list,
        f'{robot_type}.joint_topic_type_list': joint_topic_type_list,
    }
    for modality, cfg in action_groups.items():
        bt_params[f'{robot_type}.joint_order.leader_{modality}'] = list(
            cfg['joint_names']
        )

    bt_node = Node(
        package='orchestrator',
        executable='bt_node',
        name='bt_node',
        output='screen',
        parameters=[
            bt_params
        ]
    )

    return [bt_node]


def generate_launch_description():

    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value=SUPPORTED_BT_ROBOT_TYPE,
        description=(
            'BT robot type. Currently only ffw_sg2_rev1 is supported.'
        )
    )

    return LaunchDescription([
        robot_type_arg,
        OpaqueFunction(function=launch_setup)
    ])
