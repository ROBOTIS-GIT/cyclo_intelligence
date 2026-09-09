"""Regression tests for the shared SG2 recorder/model contracts."""

from cyclo_brain.model.common.sg2 import (
    GROOT_REFERENCE_ACTION_HORIZON,
    GROOT_RLT_CAMERA_NAMES,
    MULTI_TASK_DIT_ACTION_DIM,
    MULTI_TASK_DIT_ACTION_HORIZON,
    MULTI_TASK_DIT_CAMERA_KEYS,
    MULTI_TASK_DIT_CAMERA_NAMES,
    MULTI_TASK_DIT_STATE_DIM,
    RLT_ACTION_DIM,
    RLT_ACTION_HORIZON,
    RLT_DROPPED_RECORDER_INDICES,
    RLT_DROPPED_RECORDER_NAMES,
    RLT_SELECTED_RECORDER_INDICES,
    RLT_SELECTED_RECORDER_NAMES,
    SG2_CAMERA_FEATURE_KEYS,
    SG2_RECORDER_ACTION_NAMES,
    SG2_RECORDER_DIM,
    SG2_TRANSPORT_ACTION_WIDTHS,
)


def test_recorder_and_dit_preserve_the_full_22d_axis() -> None:
    assert SG2_RECORDER_DIM == 22
    assert len(SG2_RECORDER_ACTION_NAMES) == 22
    assert sum(SG2_TRANSPORT_ACTION_WIDTHS) == 22
    assert MULTI_TASK_DIT_STATE_DIM == 22
    assert MULTI_TASK_DIT_ACTION_DIM == 22


def test_camera_aliases_keep_model_specific_orders_explicit() -> None:
    assert MULTI_TASK_DIT_CAMERA_NAMES == (
        "cam_left_wrist",
        "cam_left_head",
        "cam_right_wrist",
    )
    assert GROOT_RLT_CAMERA_NAMES == (
        "cam_left_head",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert MULTI_TASK_DIT_CAMERA_KEYS == tuple(
        SG2_CAMERA_FEATURE_KEYS[name] for name in MULTI_TASK_DIT_CAMERA_NAMES
    )
    assert set(MULTI_TASK_DIT_CAMERA_NAMES) == set(GROOT_RLT_CAMERA_NAMES)


def test_rlt_projection_drops_only_head_and_lift() -> None:
    assert RLT_SELECTED_RECORDER_INDICES == (*range(16), 19, 20, 21)
    assert RLT_DROPPED_RECORDER_INDICES == (16, 17, 18)
    assert RLT_DROPPED_RECORDER_NAMES == (
        "head_joint1",
        "head_joint2",
        "lift_joint",
    )
    assert RLT_SELECTED_RECORDER_NAMES == (
        *SG2_RECORDER_ACTION_NAMES[:16],
        *SG2_RECORDER_ACTION_NAMES[19:22],
    )
    assert RLT_ACTION_DIM == 19


def test_model_horizons_are_not_conflated() -> None:
    assert MULTI_TASK_DIT_ACTION_HORIZON == 16
    assert GROOT_REFERENCE_ACTION_HORIZON == 16
    assert RLT_ACTION_HORIZON == 10
