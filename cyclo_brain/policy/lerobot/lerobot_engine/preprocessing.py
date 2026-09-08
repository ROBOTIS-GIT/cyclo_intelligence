#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""LeRobot preprocessing helpers.

Builds a policy-ready batch from RobotClient sensor/state reads.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from .constants import STATE_KEY as _STATE_KEY


logger = logging.getLogger("lerobot_engine")


class PreprocessingMixin:
    """RobotClient observation -> policy input batch."""

    def _build_observation(self, task_instruction: str) -> Dict[str, Any]:
        """Pull raw sensor data from RobotClient and build a policy batch."""
        assert self._robot is not None

        images = self._robot.get_images(format="rgb")
        if not images:
            return self._fail("No camera frames available")

        joint_dict = self._robot.get_joint_positions()
        if not joint_dict:
            return self._fail("No joint positions available")

        batch: Dict[str, Any] = {}

        for cam_name, policy_key in self._cameras.items():
            img = images.get(cam_name)
            if img is None:
                return self._fail(f"Missing camera frame: {cam_name}")
            cam_cfg = self._robot._config.get("cameras", {}).get(cam_name, {})
            try:
                tensor = self._image_preprocessing.apply(
                    img,
                    policy_key,
                    rotation_deg=cam_cfg.get("rotation_deg", 0),
                )
            except Exception as exc:
                return self._fail(f"Camera preprocessing failed for {cam_name}: {exc}")
            batch[policy_key] = tensor.to(self._device)

        state_parts: List[np.ndarray] = []
        for modality in self._state_modalities:
            if modality == "mobile":
                odom = self._robot.get_odom()
                if odom is None:
                    return self._fail("Missing odom for mobile state")
                state_parts.append(
                    np.array(
                        [
                            float(odom["linear_velocity"][0]),
                            float(odom["linear_velocity"][1]),
                            float(odom["angular_velocity"][2]),
                        ],
                        dtype=np.float32,
                    )
                )
                continue
            group = f"follower_{modality}"
            positions = joint_dict.get(group)
            if positions is None or len(positions) == 0:
                return self._fail(f"Missing joint group: {modality}")
            state_parts.append(np.asarray(positions, dtype=np.float32))

        flat_state = np.concatenate(state_parts)
        # TODO(ROBOTIS): replace zero-padding with real values. Some training
        # datasets carry extra state dimensions (e.g. EE pose) that the current
        # robot_config joint topics do not surface.
        try:
            expected = int(
                self._policy.config.input_features[_STATE_KEY].shape[0]
            )
        except Exception:
            expected = flat_state.size
        if flat_state.size < expected:
            pad = np.zeros(expected - flat_state.size, dtype=np.float32)
            logger.warning(
                "state dim mismatch: got %d, policy expects %d - padding %d zeros",
                flat_state.size,
                expected,
                expected - flat_state.size,
            )
            flat_state = np.concatenate([flat_state, pad])
        elif flat_state.size > expected:
            logger.warning(
                "state dim mismatch: got %d, policy expects %d - truncating to %d",
                flat_state.size,
                expected,
                expected,
            )
            flat_state = flat_state[:expected]
        batch[_STATE_KEY] = (
            torch.from_numpy(flat_state).unsqueeze(0).to(self._device)
        )

        batch["task"] = [task_instruction or ""]
        return batch

    def _validate_camera_shapes(self, batch):
        # Allow the saved processor to resize before checking the stack contract.
        policy_type = getattr(self._policy.config, "type", None)
        needs_equal_sizes = policy_type == "diffusion" or (
            policy_type == "xvla"
            and getattr(self._policy.config, "resize_imgs_with_padding", None) is None
        )
        if needs_equal_sizes:
            shapes = {key: tuple(batch[key].shape[-2:]) for key in self._cameras.values()}
            if len(set(shapes.values())) > 1:
                raise ValueError(
                    f"{policy_type} cameras must have equal sizes before stacking: {shapes}. "
                    f"Check {policy_type}.yaml against the training preprocessing."
                )
