#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Single-command bringup for cyclo_intelligence.

Launches orchestrator (+ rosbridge / rosbag_recorder / web_video_server)
and cyclo_data_node together so cyclo_manager / s6-agent can treat the
pair as one unit. ``cyclo_data`` and ``orchestrator`` aliases still work
when only one half needs to come up (debugging).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('orchestrator')

    orchestrator_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'orchestrator_bringup.launch.py')
        )
    )

    cyclo_data_node = Node(
        package='cyclo_data',
        executable='cyclo_data_node',
        name='cyclo_data',
        output='screen',
    )

    return LaunchDescription([
        orchestrator_bringup,
        cyclo_data_node,
    ])
