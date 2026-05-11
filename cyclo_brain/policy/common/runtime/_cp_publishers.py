#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PublishersMixin — per-tick action fan-out + trajectory preview.

Extracted from ``control_publisher.py``. Holds the four methods that
turn a flat action vector (or a chunk) into ROS2 messages on the
per-modality command topics:

- ``_publish_action_locked`` — splits one action via ``split_action``
  and dispatches to ``_publish_twist`` / ``_publish_joint_trajectory``.
- ``_publish_twist`` — geometry_msgs/Twist for the mobile group.
- ``_publish_joint_trajectory`` — trajectory_msgs/JointTrajectory for
  joint groups, built via ``make_joint_trajectory``.
- ``_publish_trajectory_preview_locked`` — UI-only preview on
  ``/inference/trajectory_preview`` (the full predicted chunk, not the
  per-tick action).

See the S5 design note in ``docs/plans/2026-05-10-cyclo_brain-refactor.md``.
The cached message classes (``self._Vector3``, ``self._msg_classes``)
are populated by ``ControlPublisher.__init__`` and consumed here via
``self``. The ``*_locked`` helpers must be called with
``self._config_lock`` held.
"""

from __future__ import annotations

import numpy as np

from zenoh_ros2_sdk import ROS2Publisher, get_logger

from post_processing import split_action
from post_processing.ros_msg_helpers import make_joint_trajectory


logger = get_logger("control_publisher")


class PublishersMixin:
    """Action fan-out + trajectory preview for ControlPublisher."""

    def _publish_trajectory_preview_locked(self, chunk: np.ndarray) -> None:
        """Emit the full predicted action chunk as JointTrajectory.

        Publishes to ``/inference/trajectory_preview``
        (trajectory_msgs/msg/JointTrajectory). Consumer: the UI's 3D
        viewer in orchestrator. This is a UI-only auxiliary; the real
        100 Hz robot commands go out on per-modality topics built in
        ``_setup_robot_specific_locked``. ``chunk`` is (T, D) where D
        matches ``len(self._preview_joint_names)``. Caller holds
        ``_config_lock``.
        """
        if self._trajectory_preview_pub is None:
            return
        joint_names = self._preview_joint_names
        if not joint_names:
            return
        n_names = len(joint_names)
        JointTrajectoryPoint = self._msg_classes["JointTrajectoryPoint"]
        Header = self._msg_classes["Header"]
        Time = self._msg_classes["Time"]
        Duration = self._msg_classes["Duration"]
        try:
            empty = np.zeros(0, dtype=np.float64)
            zero_duration = Duration(sec=0, nanosec=0)
            points = []
            for row in chunk:
                # Defensive trim — D should already equal n_names but the
                # model is the source of truth and we'd rather emit
                # something than crash on a one-off mismatch.
                positions = np.asarray(row[:n_names], dtype=np.float64)
                points.append(
                    JointTrajectoryPoint(
                        positions=positions,
                        velocities=empty,
                        accelerations=empty,
                        effort=empty,
                        time_from_start=zero_duration,
                    )
                )
            self._trajectory_preview_pub.publish(
                header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
                joint_names=list(joint_names),
                points=points,
            )
        except Exception as e:
            logger.error(f"trajectory preview publish failed: {e}", exc_info=True)

    def _publish_action_locked(self, action: np.ndarray) -> None:
        try:
            segments = split_action(
                action, self._action_joint_map, self._joint_order
            )
        except Exception as e:
            logger.error(f"split_action failed: {e}", exc_info=True)
            return

        for publisher_key, values in segments.items():
            pub = self._command_pubs.get(publisher_key)
            if pub is None:
                continue

            try:
                msg_type = self._command_msg_types.get(publisher_key, "")
                if msg_type == "geometry_msgs/msg/Twist":
                    self._publish_twist(pub, values)
                else:
                    joint_names = self._joint_order.get(
                        f"joint_order.{publisher_key}", []
                    )
                    self._publish_joint_trajectory(pub, joint_names, values)
            except Exception as e:
                logger.error(
                    f"publish {publisher_key} failed: {e}", exc_info=True
                )

    def _publish_twist(self, pub: ROS2Publisher, values: np.ndarray) -> None:
        Vector3 = self._Vector3
        linear = Vector3(
            x=float(values[0]) if len(values) > 0 else 0.0,
            y=float(values[1]) if len(values) > 1 else 0.0,
            z=0.0,
        )
        angular = Vector3(
            x=0.0,
            y=0.0,
            z=float(values[2]) if len(values) > 2 else 0.0,
        )
        pub.publish(linear=linear, angular=angular)

    def _publish_joint_trajectory(
        self,
        pub: ROS2Publisher,
        joint_names,
        values: np.ndarray,
    ) -> None:
        header, points = make_joint_trajectory(
            self._msg_classes, joint_names, values
        )
        pub.publish(header=header, joint_names=list(joint_names), points=points)
