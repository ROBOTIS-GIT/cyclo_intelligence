"""Cyclo SG2 batch fixes applied before LeRobot MultiTaskDiT preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from .flow_sde_adapter import CYCLO_SG2_CAMERA_KEYS, DEFAULT_TASK_INSTRUCTION


# State and action share this exact recorder order.  Keeping the names next to
# the batch boundary prevents a dimension-only check from silently accepting a
# different 22-DoF contract.
CYCLO_SG2_ACTION_NAMES = (
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


def canonicalize_training_batch(
    batch: Mapping[str, Any],
    *,
    n_obs_steps: int = 1,
    image_size: tuple[int, int] = (256, 256),
    task_instruction: str = DEFAULT_TASK_INSTRUCTION,
) -> dict[str, Any]:
    """Validate the three-camera batch and add MultiTaskDiT's history axis.

    Camera resizing belongs in ``LeRobotDataset.image_transforms`` so it runs
    before different native resolutions are stacked.  This function only
    validates that result, normalizes uint8 images, and supplies UI language.
    """

    if not isinstance(batch, Mapping):
        raise TypeError("Cyclo MultiTaskDiT batch must be a mapping")
    if isinstance(n_obs_steps, bool) or not isinstance(n_obs_steps, int) or n_obs_steps < 1:
        raise ValueError("Cyclo MultiTaskDiT n_obs_steps must be positive")
    if (
        len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in image_size)
    ):
        raise ValueError("Cyclo MultiTaskDiT image_size must contain two positive integers")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise ValueError("Cyclo MultiTaskDiT task instruction must be non-empty")

    state = batch.get("observation.state")
    if not isinstance(state, Tensor) or state.ndim not in (2, 3):
        raise ValueError("Cyclo MultiTaskDiT state must have shape (B, S) or (B, T, S)")
    if state.ndim == 2:
        state = state.unsqueeze(1)
    if state.shape[1] != n_obs_steps:
        raise ValueError("Cyclo MultiTaskDiT state history does not match n_obs_steps")
    batch_size = state.shape[0]

    result = {
        key: value
        for key, value in batch.items()
        if not key.startswith("observation.images.")
    }
    result["observation.state"] = state
    for key in CYCLO_SG2_CAMERA_KEYS:
        image = batch.get(key)
        if not isinstance(image, Tensor) or image.ndim not in (4, 5):
            raise ValueError(f"Cyclo MultiTaskDiT camera {key!r} has an invalid shape")
        if image.ndim == 4:
            image = image.unsqueeze(1)
        if (
            image.shape[0] != batch_size
            or image.shape[1] != n_obs_steps
            or image.shape[2] != 3
            or tuple(image.shape[-2:]) != tuple(image_size)
        ):
            raise ValueError(f"Cyclo MultiTaskDiT camera {key!r} does not match the canonical shape")
        if image.dtype == torch.uint8:
            image = image.to(torch.float32).div_(255.0)
        elif not image.is_floating_point():
            raise TypeError(f"Cyclo MultiTaskDiT camera {key!r} must be uint8 or floating point")
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f"Cyclo MultiTaskDiT camera {key!r} contains non-finite values")
        result[key] = image

    action = result.get("action")
    if not isinstance(action, Tensor) or action.ndim != 3 or action.shape[0] != batch_size:
        raise ValueError("Cyclo MultiTaskDiT action must have shape (B, H, A)")
    result["task"] = [task_instruction.strip()] * batch_size
    return result


def canonicalize_dataset_stats(
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collapse converter-duplicated camera statistics without editing data.

    Current v3 conversion can repeat one global ``(3, 1, 1)`` statistic once
    per episode, producing ``(3, E, 1)``.  It is safe to collapse only when all
    repeated columns are identical; non-identical data is rejected.
    """

    if not isinstance(stats, Mapping):
        raise TypeError("Cyclo LeRobot stats must be a mapping")
    result = {key: dict(value) for key, value in stats.items()}
    for key in CYCLO_SG2_CAMERA_KEYS:
        if key not in result:
            raise ValueError(f"Cyclo LeRobot stats are missing camera {key!r}")
        for statistic in ("mean", "std", "min", "max"):
            if statistic not in result[key]:
                raise ValueError(f"Cyclo LeRobot camera stats are missing {statistic!r}")
            value = torch.as_tensor(result[key][statistic])
            if value.ndim != 3 or value.shape[0] != 3 or value.shape[2] != 1:
                raise ValueError(f"Cyclo LeRobot camera {key!r} has invalid {statistic} shape")
            if value.shape[1] > 1:
                first = value[:, :1, :]
                if not bool(torch.allclose(value, first.expand_as(value), rtol=0.0, atol=0.0)):
                    raise ValueError(
                        f"Cyclo LeRobot camera {key!r} has non-identical duplicated {statistic} stats"
                    )
                value = first
            result[key][statistic] = value.clone()
    return result
