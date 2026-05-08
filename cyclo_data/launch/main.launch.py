#!/usr/bin/env python3
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cyclo_data_node = Node(
        package='cyclo_data',
        executable='cyclo_data_node',
        name='cyclo_data',
        output='screen',
    )

    return LaunchDescription([
        cyclo_data_node,
    ])
