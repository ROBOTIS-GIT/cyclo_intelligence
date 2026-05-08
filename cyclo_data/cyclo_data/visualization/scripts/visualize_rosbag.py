#!/usr/bin/env python3
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
# Author: Claude AI Assistant

"""CLI script for visualizing rosbag2 MCAP data.

Installed as a console_script via cyclo_data/setup.py, so once the
package is built (colcon) / installed (pip) you can run it directly:

    visualize_rosbag /path/to/rosbag/directory --detailed --intervals
    visualize_rosbag /path/to/file.mcap

Or as a module:
    python3 -m cyclo_data.visualization.scripts.visualize_rosbag ...
"""

import argparse
import os
from pathlib import Path

from cyclo_data.visualization.rosbag_visualizer import RosbagVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize rosbag2 MCAP data for data quality validation'
    )
    parser.add_argument(
        'mcap_path', help='Path to MCAP file or directory containing it')
    parser.add_argument('--output', '-o', help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument(
        '--detailed', '-d', action='store_true', help='Show detailed statistics')
    parser.add_argument(
        '--intervals', '-i', action='store_true', help='Plot interval distributions')
    parser.add_argument(
        '--camera-hz', type=float, default=15.0, help='Expected camera Hz (default: 15)')
    parser.add_argument(
        '--joint-hz', type=float, default=100.0, help='Expected joint Hz (default: 100)')

    args = parser.parse_args()

    visualizer = RosbagVisualizer(args.mcap_path)
    visualizer.load()
    visualizer.print_summary(detailed=args.detailed)

    output_dir = args.output or str(Path(visualizer.mcap_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    visualizer.plot_timeline(
        output_path=os.path.join(output_dir, 'rosbag_timeline.png'),
        show=args.show,
    )

    if args.intervals:
        visualizer.plot_topic_intervals(
            output_path=os.path.join(output_dir, 'topic_intervals.png'),
            show=args.show,
        )

    visualizer.save_stats_json(
        output_path=os.path.join(output_dir, 'rosbag_stats.json'),
    )

    report = visualizer.get_validation_report(
        expected_camera_hz=args.camera_hz,
        expected_joint_hz=args.joint_hz,
    )

    print('\n' + '=' * 60)
    print('VALIDATION REPORT')
    print('=' * 60)
    print(f"Valid: {'YES' if report['valid'] else 'NO'}")

    if report['issues']:
        print('\nIssues Found:')
        for issue in report['issues']:
            print(f'  - {issue}')
    else:
        print('\nNo issues found. Data meets requirements.')

    print('=' * 60)


if __name__ == '__main__':
    main()
