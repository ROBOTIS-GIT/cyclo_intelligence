"""Canonical SG2 recorder schema and explicit model-specific projections.

The recorder schema is the common source of truth, but policy interfaces are
not interchangeable. MultiTaskDiT consumes the full 22D vector and uses the
dataset camera order. GR00T/RLT uses its processor camera order and projects
the recorder vector to the 19D arm/gripper/odometry contract.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


SG2_ROBOT_TYPE: Final = "ffw_sg2_rev1"

SG2_RECORDER_ACTION_NAMES: Final = (
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "gripper_l_joint1",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
    "linear_x",
    "linear_y",
    "angular_z",
)
SG2_RECORDER_DIM: Final = len(SG2_RECORDER_ACTION_NAMES)

# ActionStep names intentionally differ from GR00T's processor names. The
# final transport group is called ``mobile`` while the processor calls the
# same three recorder columns ``odometry``.
SG2_TRANSPORT_ACTION_GROUPS: Final = (
    "arm_left",
    "arm_right",
    "head",
    "lift",
    "mobile",
)
SG2_TRANSPORT_ACTION_WIDTHS: Final = (8, 8, 2, 1, 3)
SG2_LIVE_STATE_GROUPS: Final = (
    ("follower_arm_left", 8),
    ("follower_arm_right", 8),
    ("follower_head", 2),
    ("follower_lift", 1),
)

# A camera has one RobotClient/GR00T name and one LeRobot feature key. Orders
# are declared below per consumer rather than treating one order as universal.
SG2_CAMERA_FEATURE_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "cam_left_head": "observation.images.rgb.cam_left_head",
        "cam_left_wrist": "observation.images.rgb.cam_left_wrist",
        "cam_right_wrist": "observation.images.rgb.cam_right_wrist",
    }
)

# MultiTaskDiT checkpoint and LeRobot dataset contract: full 22D, 16 steps.
MULTI_TASK_DIT_CAMERA_NAMES: Final = (
    "cam_left_wrist",
    "cam_left_head",
    "cam_right_wrist",
)
MULTI_TASK_DIT_CAMERA_KEYS: Final = tuple(
    SG2_CAMERA_FEATURE_KEYS[name] for name in MULTI_TASK_DIT_CAMERA_NAMES
)
MULTI_TASK_DIT_STATE_DIM: Final = SG2_RECORDER_DIM
MULTI_TASK_DIT_ACTION_DIM: Final = SG2_RECORDER_DIM
MULTI_TASK_DIT_ACTION_HORIZON: Final = 16

# GR00T processor contract. Its camera order is not the MultiTaskDiT order.
GROOT_RLT_CAMERA_NAMES: Final = (
    "cam_left_head",
    "cam_left_wrist",
    "cam_right_wrist",
)
GROOT_RLT_PROCESSOR_GROUPS: Final = ("arm_left", "arm_right", "odometry")
GROOT_REFERENCE_ACTION_HORIZON: Final = 16

RLT_ACTION_GROUP_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "arm_left": SG2_RECORDER_ACTION_NAMES[0:8],
        "arm_right": SG2_RECORDER_ACTION_NAMES[8:16],
        "odometry": SG2_RECORDER_ACTION_NAMES[19:22],
    }
)
RLT_SELECTED_RECORDER_NAMES: Final = tuple(
    name
    for group in GROOT_RLT_PROCESSOR_GROUPS
    for name in RLT_ACTION_GROUP_NAMES[group]
)
RLT_SELECTED_RECORDER_INDICES: Final = tuple(
    SG2_RECORDER_ACTION_NAMES.index(name) for name in RLT_SELECTED_RECORDER_NAMES
)
RLT_DROPPED_RECORDER_INDICES: Final = tuple(
    index
    for index in range(SG2_RECORDER_DIM)
    if index not in RLT_SELECTED_RECORDER_INDICES
)
RLT_DROPPED_RECORDER_NAMES: Final = tuple(
    SG2_RECORDER_ACTION_NAMES[index] for index in RLT_DROPPED_RECORDER_INDICES
)
RLT_ACTION_DIM: Final = len(RLT_SELECTED_RECORDER_NAMES)
RLT_ACTION_HORIZON: Final = 10


if len(set(SG2_RECORDER_ACTION_NAMES)) != SG2_RECORDER_DIM:
    raise RuntimeError("SG2 recorder names must be unique")
if sum(SG2_TRANSPORT_ACTION_WIDTHS) != SG2_RECORDER_DIM:
    raise RuntimeError("SG2 ActionStep groups do not cover the 22D recorder contract")
if set(MULTI_TASK_DIT_CAMERA_NAMES) != set(GROOT_RLT_CAMERA_NAMES):
    raise RuntimeError("SG2 model camera projections must cover the same three cameras")
if RLT_SELECTED_RECORDER_INDICES != (*range(16), 19, 20, 21):
    raise RuntimeError("RLT must project the SG2 recorder contract from 22D to 19D")
if RLT_DROPPED_RECORDER_INDICES != (16, 17, 18):
    raise RuntimeError("RLT must drop only SG2 head and lift recorder values")


__all__ = [
    "GROOT_REFERENCE_ACTION_HORIZON",
    "GROOT_RLT_CAMERA_NAMES",
    "GROOT_RLT_PROCESSOR_GROUPS",
    "MULTI_TASK_DIT_ACTION_DIM",
    "MULTI_TASK_DIT_ACTION_HORIZON",
    "MULTI_TASK_DIT_CAMERA_KEYS",
    "MULTI_TASK_DIT_CAMERA_NAMES",
    "MULTI_TASK_DIT_STATE_DIM",
    "RLT_ACTION_DIM",
    "RLT_ACTION_GROUP_NAMES",
    "RLT_ACTION_HORIZON",
    "RLT_DROPPED_RECORDER_INDICES",
    "RLT_DROPPED_RECORDER_NAMES",
    "RLT_SELECTED_RECORDER_INDICES",
    "RLT_SELECTED_RECORDER_NAMES",
    "SG2_CAMERA_FEATURE_KEYS",
    "SG2_LIVE_STATE_GROUPS",
    "SG2_RECORDER_ACTION_NAMES",
    "SG2_RECORDER_DIM",
    "SG2_ROBOT_TYPE",
    "SG2_TRANSPORT_ACTION_GROUPS",
    "SG2_TRANSPORT_ACTION_WIDTHS",
]
