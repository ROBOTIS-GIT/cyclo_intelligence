"""Shared ROS2 message construction helpers for the policy runtime.

The two-process policy runtime publishes joint commands via
zenoh_ros2_sdk's generated message classes, which are stricter than rclpy:
every IDL field is required as a positional/keyword arg and nested
structs must be real generated instances (not dicts). On top of that,
ros2_control's joint_trajectory_controller has its own ergonomics
constraints — it rejects trajectories with non-empty velocities /
accelerations / effort unless the underlying controller has the
matching command interface. These helpers consolidate those rules so a
typo or missing field doesn't reappear in a new backend's
control_publisher.

The expected ``classes`` dict is produced once at ``ControlPublisher``
init time (cached on the instance as ``self._JointTrajectoryPoint`` etc)
to keep the 100 Hz hot path free of ``get_message_class`` lookups.
"""

from typing import Mapping, Sequence

import numpy as np


def make_joint_trajectory(
    classes: Mapping[str, type],
    joint_names: Sequence[str],
    positions: np.ndarray,
):
    """Build a ros2_control-compatible JointTrajectory header + points.

    Args:
        classes: Mapping of generated-class names to types. Must contain
            ``JointTrajectoryPoint``, ``Header``, ``Time``, ``Duration``.
        joint_names: Joint names in the publish order.
        positions: 1-D array of position commands, same length as ``joint_names``.

    Returns:
        ``(header, [point])`` ready to pass to a zenoh_ros2_sdk
        ``ROS2Publisher.publish(header=..., joint_names=..., points=...)``.

    Why empty arrays for velocities/accelerations/effort:
        ros2_control's joint_trajectory_controller rejects messages with
        populated ``effort`` unless the controller has an ``effort`` command
        interface (and similarly for ``velocities`` / ``accelerations`` on
        position-only controllers). rclpy's default-constructed
        ``JointTrajectoryPoint()`` leaves these as empty sequences, which
        is what we mirror here.
    """
    JointTrajectoryPoint = classes["JointTrajectoryPoint"]
    Header = classes["Header"]
    Time = classes["Time"]
    Duration = classes["Duration"]

    empty = np.zeros(0, dtype=np.float64)
    point = JointTrajectoryPoint(
        positions=np.asarray(positions, dtype=np.float64),
        velocities=empty,
        accelerations=empty,
        effort=empty,
        time_from_start=Duration(sec=0, nanosec=0),
    )
    header = Header(stamp=Time(sec=0, nanosec=0), frame_id="")
    return header, [point]
