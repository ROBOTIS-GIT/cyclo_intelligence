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

"""
Rosbag Visualization Module.

Provides comprehensive visualization and analysis tools for rosbag2 MCAP files.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from mcap.reader import make_reader
except ImportError:
    make_reader = None

try:
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    DecoderFactory = None


@dataclass
class JointData:
    """Joint state data for a single message."""

    timestamp: float
    names: List[str]
    positions: List[float]


@dataclass
class TopicStats:
    """Statistics for a single topic."""

    topic: str
    count: int
    avg_hz: float
    duration: float
    min_interval_ms: float
    max_interval_ms: float
    std_interval_ms: float
    timestamps: List[float] = field(default_factory=list)


@dataclass
class RosbagStats:
    """Overall statistics for a rosbag."""

    total_duration: float
    total_messages: int
    topic_stats: Dict[str, TopicStats]
    start_time: float
    end_time: float


class RosbagVisualizer:
    """
    Visualizer for rosbag2 MCAP files.

    Provides various visualization methods for analyzing rosbag data quality,
    topic timing, and data distribution.

    Usage:
        visualizer = RosbagVisualizer("/path/to/rosbag.mcap")
        visualizer.load()
        visualizer.plot_timeline(output_path="timeline.png")
        visualizer.print_summary()
    """

    # Color schemes for different topic types
    COLORS = {
        'camera': '#e74c3c',      # Red
        'joint': '#3498db',        # Blue
        'camera_info': '#9b59b6',  # Purple
        'tf': '#2ecc71',           # Green
        'odom': '#f39c12',         # Orange
        'cmd_vel': '#1abc9c',      # Teal
        'other': '#95a5a6'         # Gray
    }

    def __init__(self, mcap_path: str):
        """
        Initialize the visualizer.

        Args:
            mcap_path: Path to the MCAP file or directory containing it.
        """
        self.mcap_path, self.output_dir = self._resolve_mcap_path(mcap_path)
        self.topic_timestamps: Dict[str, List[float]] = defaultdict(list)
        self.joint_data: Dict[str, List[JointData]] = defaultdict(list)
        self.stats: Optional[RosbagStats] = None
        self._loaded = False

    def _resolve_mcap_path(self, path: str) -> Tuple[str, str]:
        """
        Resolve the MCAP file path and output directory.

        Returns:
            Tuple of (mcap_file_path, output_directory)
        """
        path = Path(path)
        if path.is_dir():
            # Look for .mcap file in directory
            mcap_files = list(path.glob('*.mcap'))
            if mcap_files:
                # Output directory is the directory that was passed in
                return str(mcap_files[0]), str(path)
            raise FileNotFoundError(f'No MCAP file found in {path}')
        # If a file was passed, output to its parent directory
        return str(path), str(path.parent)

    def load(self) -> 'RosbagVisualizer':
        """
        Load and parse the MCAP file.

        Uses header.stamp for messages that have a header (JointState, CameraInfo,
        CompressedImage, Odometry, etc.) for accurate timing. Falls back to
        publish_time for messages without headers.

        Returns:
            self for method chaining.
        """
        if make_reader is None:
            raise ImportError('mcap package is required. Install with: pip install mcap')

        print(f'Loading rosbag: {self.mcap_path}')

        self.topic_timestamps.clear()

        # Check if we can use ROS2 decoder for header.stamp extraction
        use_ros2_decoder = DecoderFactory is not None
        if use_ros2_decoder:
            print('Using header.stamp for messages with headers')
            self._load_with_header_stamp()
        else:
            print('Warning: mcap_ros2 not available, using publish_time for all messages')
            print('Install with: pip install mcap-ros2-support')
            self._load_with_publish_time()

        self._calculate_stats()
        self._loaded = True
        print(f'Loaded {len(self.topic_timestamps)} topics, {self.stats.total_messages} messages')

        return self

    def _load_with_header_stamp(self):
        """Load using header.stamp from decoded ROS2 messages."""
        with open(self.mcap_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
                topic = channel.topic
                timestamp_sec = None

                # Try to get header.stamp from decoded message
                if decoded_msg is not None and hasattr(decoded_msg, 'header'):
                    header = decoded_msg.header
                    if hasattr(header, 'stamp'):
                        stamp = header.stamp
                        timestamp_sec = stamp.sec + stamp.nanosec / 1e9

                # Fallback to publish_time if no header.stamp
                if timestamp_sec is None:
                    timestamp_sec = message.publish_time / 1e9

                self.topic_timestamps[topic].append(timestamp_sec)

                # Store joint state data for comparison visualization
                if 'joint_states' in topic and decoded_msg is not None:
                    if hasattr(decoded_msg, 'name') and hasattr(decoded_msg, 'position'):
                        self.joint_data[topic].append(JointData(
                            timestamp=timestamp_sec,
                            names=list(decoded_msg.name),
                            positions=list(decoded_msg.position)
                        ))

    def _load_with_publish_time(self):
        """Load using publish_time (fallback when mcap_ros2 not available)."""
        with open(self.mcap_path, 'rb') as f:
            reader = make_reader(f)

            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                timestamp_sec = message.publish_time / 1e9
                self.topic_timestamps[topic].append(timestamp_sec)

    def _calculate_stats(self):
        """Calculate statistics for all topics."""
        topic_stats = {}
        all_timestamps = []

        for topic, timestamps in self.topic_timestamps.items():
            timestamps = np.array(sorted(timestamps))
            all_timestamps.extend(timestamps)

            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                avg_interval = np.mean(intervals)
                avg_hz = 1.0 / avg_interval if avg_interval > 0 else 0
                min_interval = np.min(intervals)
                max_interval = np.max(intervals)
                std_interval = np.std(intervals)
                duration = timestamps[-1] - timestamps[0]
            else:
                avg_hz = 0
                min_interval = 0
                max_interval = 0
                std_interval = 0
                duration = 0

            topic_stats[topic] = TopicStats(
                topic=topic,
                count=len(timestamps),
                avg_hz=avg_hz,
                duration=duration,
                min_interval_ms=min_interval * 1000,
                max_interval_ms=max_interval * 1000,
                std_interval_ms=std_interval * 1000,
                timestamps=list(timestamps)
            )

        t_min = min(all_timestamps) if all_timestamps else 0
        t_max = max(all_timestamps) if all_timestamps else 0

        self.stats = RosbagStats(
            total_duration=t_max - t_min,
            total_messages=sum(len(ts) for ts in self.topic_timestamps.values()),
            topic_stats=topic_stats,
            start_time=t_min,
            end_time=t_max
        )

    def _get_readable_topic_name(self, topic: str) -> str:
        """Get a readable short name for a topic while preserving uniqueness."""
        parts = topic.split('/')

        # Camera topics: /robot/camera/cam_xxx/image_raw/compressed -> cam_xxx/compressed
        # Camera info: /robot/camera/cam_xxx/.../camera_info -> cam_xxx/camera_info
        if 'camera' in topic:
            cam_name = None
            for part in parts:
                if part.startswith('cam_'):
                    cam_name = part
                    break
            if cam_name:
                if 'camera_info' in topic:
                    return f'{cam_name}/camera_info'
                elif 'compressed' in topic:
                    return f'{cam_name}/compressed'

        # Joint topics: /robot/arm_left_follower/joint_states -> arm_left_follower
        if 'joint_states' in topic:
            for part in parts:
                if 'follower' in part or 'leader' in part:
                    return part

        # TF, odom, cmd_vel - return as is
        if topic in ['/tf', '/odom', '/cmd_vel']:
            return topic.lstrip('/')

        # Default: return last meaningful part
        if len(parts) > 1:
            return parts[-1] if parts[-1] else parts[-2]
        return topic

    def _get_topic_type(self, topic: str) -> str:
        """Determine the type of a topic for coloring."""
        topic_lower = topic.lower()
        if 'camera_info' in topic_lower:
            return 'camera_info'
        if 'camera' in topic_lower and 'compressed' in topic_lower:
            return 'camera'
        if 'joint_states' in topic_lower:
            return 'joint'
        if '/tf' in topic_lower:
            return 'tf'
        if 'odom' in topic_lower:
            return 'odom'
        if 'cmd_vel' in topic_lower:
            return 'cmd_vel'
        return 'other'

    def _get_topic_color(self, topic: str, idx: int = 0) -> str:
        """Get color for a topic."""
        topic_type = self._get_topic_type(topic)
        return self.COLORS.get(topic_type, self.COLORS['other'])

    def _normalize_timestamps(self, timestamps: List[float]) -> List[float]:
        """Normalize timestamps to start from 0."""
        t_min = self.stats.start_time
        return [t - t_min for t in timestamps]

    def plot_timeline(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 14),
        show: bool = False,
        title: str = None
    ) -> str:
        """
        Create a timeline visualization of all topics.

        Args:
            output_path: Path to save the figure. If None, auto-generates.
            figsize: Figure size (width, height).
            show: Whether to display the plot.
            title: Custom title for the plot.

        Returns:
            Path to the saved figure.
        """
        if not self._loaded:
            raise RuntimeError('Data not loaded. Call load() first.')

        topics = sorted(self.topic_timestamps.keys())

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        # Plot 1: Event timeline
        for idx, topic in enumerate(topics):
            timestamps = self._normalize_timestamps(self.topic_timestamps[topic])
            color = self._get_topic_color(topic, idx)
            ax1.eventplot(
                [timestamps], lineoffsets=idx, linelengths=0.8,
                colors=[color], linewidths=0.5, alpha=0.7
            )

        # Configure timeline plot
        ax1.set_yticks(range(len(topics)))
        short_names = [self._get_readable_topic_name(t) for t in topics]
        ax1.set_yticklabels(short_names, fontsize=8)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Topics', fontsize=12)

        plot_title = title or 'Rosbag2 Topic Timeline - Message Arrival Events'
        ax1.set_title(plot_title, fontsize=14, fontweight='bold')
        ax1.set_xlim(-0.5, self.stats.total_duration + 0.5)
        ax1.grid(True, axis='x', alpha=0.3)

        # Create legend
        legend_elements = []
        for i, topic in enumerate(topics):
            stats = self.stats.topic_stats[topic]
            label = f'{topic} ({stats.count} msgs, {stats.avg_hz:.1f}Hz)'
            color = self._get_topic_color(topic, i)
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=2, label=label)
            )
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=6, ncol=2)

        # Plot 2: Message rate histogram
        self._plot_rate_histogram(ax2, topics)

        plt.tight_layout()

        # Save figure
        if output_path is None:
            base_dir = Path(self.output_dir)
            output_path = str(base_dir / 'rosbag_timeline.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Timeline saved to: {output_path}')

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def _plot_rate_histogram(self, ax, topics: List[str]):
        """Plot message rate histogram."""
        # Categorize topics
        camera_topics = [
            t for t in topics
            if 'camera' in t and 'compressed' in t and 'camera_info' not in t]
        joint_topics = [t for t in topics if 'joint_states' in t]
        other_topics = [t for t in topics if t not in camera_topics and t not in joint_topics]

        # Create bins
        bin_width = 0.5
        bins = np.arange(0, self.stats.total_duration + bin_width, bin_width)

        # Count messages per category
        camera_counts = np.zeros(len(bins) - 1)
        joint_counts = np.zeros(len(bins) - 1)
        other_counts = np.zeros(len(bins) - 1)

        for topic in camera_topics:
            hist, _ = np.histogram(
                self._normalize_timestamps(self.topic_timestamps[topic]), bins=bins
            )
            camera_counts += hist

        for topic in joint_topics:
            hist, _ = np.histogram(
                self._normalize_timestamps(self.topic_timestamps[topic]), bins=bins
            )
            joint_counts += hist

        for topic in other_topics:
            hist, _ = np.histogram(
                self._normalize_timestamps(self.topic_timestamps[topic]), bins=bins
            )
            other_counts += hist

        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(bin_centers, camera_counts, width=bin_width*0.8, alpha=0.7,
               label=f'Camera ({len(camera_topics)} topics)', color=self.COLORS['camera'])
        ax.bar(bin_centers, joint_counts, width=bin_width*0.8, alpha=0.7,
               bottom=camera_counts, label=f'Joint ({len(joint_topics)} topics)',
               color=self.COLORS['joint'])
        ax.bar(bin_centers, other_counts, width=bin_width*0.8, alpha=0.7,
               bottom=camera_counts + joint_counts,
               label=f'Other ({len(other_topics)} topics)', color=self.COLORS['other'])

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Messages per 0.5s', fontsize=12)
        ax.set_title('Message Rate Distribution Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xlim(-0.5, self.stats.total_duration + 0.5)

    def plot_topic_intervals(
        self,
        topics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        show: bool = False
    ) -> str:
        """
        Plot message interval distribution for specified topics.

        Args:
            topics: List of topics to plot. If None, plots all.
            output_path: Path to save the figure.
            figsize: Figure size.
            show: Whether to display the plot.

        Returns:
            Path to the saved figure.
        """
        if not self._loaded:
            raise RuntimeError('Data not loaded. Call load() first.')

        if topics is None:
            topics = list(self.topic_timestamps.keys())

        n_topics = len(topics)
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_topics == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]

        for idx, topic in enumerate(topics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]

            timestamps = np.array(sorted(self.topic_timestamps[topic]))
            if len(timestamps) > 1:
                intervals = np.diff(timestamps) * 1000  # Convert to ms

                ax.hist(
                    intervals, bins=50, color=self._get_topic_color(topic),
                    alpha=0.7, edgecolor='black', linewidth=0.5)

                # Add min, mean, max lines
                min_val = np.min(intervals)
                mean_val = np.mean(intervals)
                max_val = np.max(intervals)

                ax.axvline(
                    min_val, color='blue', linestyle=':', linewidth=2,
                    label=f'Min: {min_val:.2f}ms')
                ax.axvline(
                    mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_val:.2f}ms')
                ax.axvline(
                    max_val, color='orange', linestyle=':', linewidth=2,
                    label=f'Max: {max_val:.2f}ms')

            short_name = topic.split('/')[-1] if len(topic) > 25 else topic
            ax.set_title(short_name, fontsize=10)
            ax.set_xlabel('Interval (ms)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_topics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].set_visible(False)

        plt.suptitle('Message Interval Distribution by Topic', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path is None:
            base_dir = Path(self.output_dir)
            output_path = str(base_dir / 'topic_intervals.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Interval plot saved to: {output_path}')

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def print_summary(self, detailed: bool = False):
        """
        Print a summary of the rosbag statistics.

        Args:
            detailed: Whether to print detailed per-topic stats.
        """
        if not self._loaded:
            raise RuntimeError('Data not loaded. Call load() first.')

        print('\n' + '=' * 80)
        print('ROSBAG SUMMARY')
        print('=' * 80)
        print(f'File: {self.mcap_path}')
        print(f'Duration: {self.stats.total_duration:.2f} seconds')
        print(f'Total Messages: {self.stats.total_messages}')
        print(f'Topics: {len(self.topic_timestamps)}')

        # Group topics by type
        camera_topics = []
        joint_topics = []
        other_topics = []

        for topic in sorted(self.topic_timestamps.keys()):
            topic_type = self._get_topic_type(topic)
            if topic_type == 'camera':
                camera_topics.append(topic)
            elif topic_type == 'joint':
                joint_topics.append(topic)
            else:
                other_topics.append(topic)

        print(f'\nCamera Topics ({len(camera_topics)}):')
        for topic in camera_topics:
            stats = self.stats.topic_stats[topic]
            print(f'  {topic}: {stats.count} frames @ {stats.avg_hz:.1f} Hz')
            if detailed:
                print(
                    f'    Interval: {stats.min_interval_ms:.2f}ms - '
                    f'{stats.max_interval_ms:.2f}ms (std: {stats.std_interval_ms:.2f}ms)')

        print(f'\nJoint Topics ({len(joint_topics)}):')
        for topic in joint_topics:
            stats = self.stats.topic_stats[topic]
            print(f'  {topic}: {stats.count} msgs @ {stats.avg_hz:.1f} Hz')
            if detailed:
                print(
                    f'    Interval: {stats.min_interval_ms:.2f}ms - '
                    f'{stats.max_interval_ms:.2f}ms (std: {stats.std_interval_ms:.2f}ms)')

        print(f'\nOther Topics ({len(other_topics)}):')
        for topic in other_topics:
            stats = self.stats.topic_stats[topic]
            print(f'  {topic}: {stats.count} msgs @ {stats.avg_hz:.1f} Hz')
            if detailed:
                print(
                    f'    Interval: {stats.min_interval_ms:.2f}ms - '
                    f'{stats.max_interval_ms:.2f}ms (std: {stats.std_interval_ms:.2f}ms)')

        print('=' * 80)

    def get_validation_report(
        self, expected_camera_hz: float = 15.0,
        expected_joint_hz: float = 100.0
    ) -> Dict:
        """
        Generate a validation report for ROBOTIS format.

        Args:
            expected_camera_hz: Expected camera frame rate.
            expected_joint_hz: Expected joint state rate.

        Returns:
            Dictionary containing validation results.
        """
        if not self._loaded:
            raise RuntimeError('Data not loaded. Call load() first.')

        report = {
            'valid': True,
            'duration': self.stats.total_duration,
            'total_messages': self.stats.total_messages,
            'issues': [],
            'cameras': {},
            'joints': {},
            'other': {}
        }

        for topic, stats in self.stats.topic_stats.items():
            topic_type = self._get_topic_type(topic)

            entry = {
                'count': stats.count,
                'avg_hz': stats.avg_hz,
                'expected_hz': None,
                'hz_ok': True
            }

            if topic_type == 'camera':
                entry['expected_hz'] = expected_camera_hz
                if abs(stats.avg_hz - expected_camera_hz) > 1.0:
                    entry['hz_ok'] = False
                    report['issues'].append(
                        f'{topic}: Expected {expected_camera_hz}Hz, got {stats.avg_hz:.1f}Hz'
                    )
                report['cameras'][topic] = entry

            elif topic_type == 'joint':
                entry['expected_hz'] = expected_joint_hz
                if stats.avg_hz < expected_joint_hz * 0.5:
                    entry['hz_ok'] = False
                    report['issues'].append(
                        f'{topic}: Expected {expected_joint_hz}Hz, got {stats.avg_hz:.1f}Hz'
                    )
                report['joints'][topic] = entry
            else:
                report['other'][topic] = entry

        report['valid'] = len(report['issues']) == 0

        return report

    def save_stats_json(self, output_path: Optional[str] = None) -> str:
        """
        Save statistics to a JSON file.

        Args:
            output_path: Path to save JSON. If None, auto-generates.

        Returns:
            Path to saved JSON file.
        """
        if not self._loaded:
            raise RuntimeError('Data not loaded. Call load() first.')

        stats_dict = {
            'mcap_path': self.mcap_path,
            'total_duration': self.stats.total_duration,
            'total_messages': self.stats.total_messages,
            'start_time': self.stats.start_time,
            'end_time': self.stats.end_time,
            'topics': {}
        }

        for topic, stats in self.stats.topic_stats.items():
            stats_dict['topics'][topic] = {
                'count': stats.count,
                'avg_hz': round(stats.avg_hz, 2),
                'duration': round(stats.duration, 2),
                'min_interval_ms': round(stats.min_interval_ms, 2),
                'max_interval_ms': round(stats.max_interval_ms, 2),
                'std_interval_ms': round(stats.std_interval_ms, 2)
            }

        if output_path is None:
            base_dir = Path(self.output_dir)
            output_path = str(base_dir / 'rosbag_stats.json')

        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        print(f'Stats saved to: {output_path}')
        return output_path

    def plot_joint_comparison(
        self,
        output_path: Optional[str] = None,
        figsize_per_joint: Tuple[float, float] = (12, 3),
        show: bool = False
    ) -> List[str]:
        """
        Plot follower vs leader joint comparison for teleoperation validation.

        Creates comparison plots for each joint pair showing how well the follower
        tracks the leader in teleoperation scenarios.

        Args:
            output_path: Base path for output files. If None, auto-generates.
            figsize_per_joint: Figure size per joint subplot.
            show: Whether to display the plots.

        Returns:
            List of paths to saved figures.
        """
        if not self._loaded:
            raise RuntimeError('Data not loaded. Call load() first.')

        if DecoderFactory is None:
            raise RuntimeError('mcap_ros2 is required for joint comparison. '
                               'Install with: pip install mcap-ros2-support')

        if not self.joint_data:
            print('No joint state data found. Re-loading with joint data...')
            self._reload_joint_data()

        # Define follower-leader pairs
        joint_pairs = [
            ('/robot/arm_left_follower/joint_states', '/robot/arm_left_leader/joint_states'),
            ('/robot/arm_right_follower/joint_states', '/robot/arm_right_leader/joint_states'),
            ('/robot/head_follower/joint_states', '/robot/head_leader/joint_states'),
            ('/robot/lift_follower/joint_states', '/robot/lift_leader/joint_states'),
        ]

        saved_paths = []
        base_dir = Path(self.output_dir)

        for follower_topic, leader_topic in joint_pairs:
            if follower_topic not in self.joint_data or leader_topic not in self.joint_data:
                print(f'Skipping {follower_topic} - data not found')
                continue

            follower_data = self.joint_data[follower_topic]
            leader_data = self.joint_data[leader_topic]

            if not follower_data or not leader_data:
                continue

            # Get joint names from the first message
            joint_names = follower_data[0].names
            n_joints = len(joint_names)

            # Extract time-series data
            follower_times = np.array([d.timestamp for d in follower_data])
            leader_times = np.array([d.timestamp for d in leader_data])

            # Normalize to start from 0
            t_min = min(follower_times.min(), leader_times.min())
            follower_times = follower_times - t_min
            leader_times = leader_times - t_min

            # Extract positions for each joint
            follower_positions = np.array([d.positions for d in follower_data])
            leader_positions = np.array([d.positions for d in leader_data])

            # Create figure
            fig, axes = plt.subplots(
                n_joints, 1,
                figsize=(figsize_per_joint[0], figsize_per_joint[1] * n_joints),
                sharex=True
            )

            if n_joints == 1:
                axes = [axes]

            # Get pair name for title
            pair_name = follower_topic.split('/')[2].replace('_follower', '')

            for idx, joint_name in enumerate(joint_names):
                ax = axes[idx]

                # Plot leader and follower
                ax.plot(leader_times, leader_positions[:, idx],
                        'b-', label='Leader', alpha=0.8, linewidth=1)
                ax.plot(follower_times, follower_positions[:, idx],
                        'r-', label='Follower', alpha=0.8, linewidth=1)

                ax.set_ylabel(f'{joint_name}\n(rad)', fontsize=9)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

                # Calculate tracking error statistics
                # Interpolate leader to follower timestamps for error calc
                leader_interp = np.interp(follower_times, leader_times, leader_positions[:, idx])
                error = follower_positions[:, idx] - leader_interp
                rmse = np.sqrt(np.mean(error ** 2))
                max_error = np.max(np.abs(error))

                ax.text(0.02, 0.95, f'RMSE: {rmse:.4f} rad, Max: {max_error:.4f} rad',
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            axes[-1].set_xlabel('Time (seconds)', fontsize=10)

            fig.suptitle(f'Joint Comparison: {pair_name}\n(Leader vs Follower)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save figure
            if output_path is None:
                save_path = str(base_dir / f'joint_comparison_{pair_name}.png')
            else:
                save_path = f'{output_path}_{pair_name}.png'

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Joint comparison saved: {save_path}')
            saved_paths.append(save_path)

            if show:
                plt.show()
            else:
                plt.close()

        return saved_paths

    def _reload_joint_data(self):
        """Reload MCAP to extract joint state data."""
        self.joint_data.clear()
        with open(self.mcap_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
                topic = channel.topic
                if 'joint_states' in topic and decoded_msg is not None:
                    timestamp_sec = None
                    if hasattr(decoded_msg, 'header'):
                        stamp = decoded_msg.header.stamp
                        timestamp_sec = stamp.sec + stamp.nanosec / 1e9
                    else:
                        timestamp_sec = message.publish_time / 1e9

                    if hasattr(decoded_msg, 'name') and hasattr(decoded_msg, 'position'):
                        self.joint_data[topic].append(JointData(
                            timestamp=timestamp_sec,
                            names=list(decoded_msg.name),
                            positions=list(decoded_msg.position)
                        ))


def main():
    """Command-line interface for rosbag visualization."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize rosbag2 MCAP data')
    parser.add_argument('mcap_path', help='Path to MCAP file or directory')
    parser.add_argument('--output', '-o', help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    parser.add_argument('--intervals', action='store_true', help='Plot interval distributions')
    parser.add_argument('--joints', action='store_true', help='Plot joint follower-leader comparison')

    args = parser.parse_args()

    visualizer = RosbagVisualizer(args.mcap_path)
    visualizer.load()
    visualizer.print_summary(detailed=args.detailed)

    output_dir = args.output or visualizer.output_dir

    visualizer.plot_timeline(
        output_path=os.path.join(output_dir, 'rosbag_timeline.png'),
        show=args.show
    )

    if args.intervals:
        visualizer.plot_topic_intervals(
            output_path=os.path.join(output_dir, 'topic_intervals.png'),
            show=args.show
        )

    if args.joints:
        visualizer.plot_joint_comparison(
            output_path=os.path.join(output_dir, 'joint_comparison'),
            show=args.show
        )

    visualizer.save_stats_json(
        output_path=os.path.join(output_dir, 'rosbag_stats.json')
    )

    # Print validation report
    report = visualizer.get_validation_report()
    print('\nValidation Report:')
    print(f"  Valid: {report['valid']}")
    if report['issues']:
        print('  Issues:')
        for issue in report['issues']:
            print(f'    - {issue}')


if __name__ == '__main__':
    main()
