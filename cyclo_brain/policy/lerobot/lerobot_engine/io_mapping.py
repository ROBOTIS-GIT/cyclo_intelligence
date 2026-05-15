#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine I/O mapping helpers (IoMappingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_init_robot``: create RobotClient + resolve camera / state mappings.
- ``_teardown_robot``: release the RobotClient.
- ``_policy_image_keys``: read the policy's expected image input keys.
"""

from __future__ import annotations

import logging

from .constants import IMAGE_KEY_PREFIX as _IMAGE_KEY_PREFIX

from robot_client import RobotClient


logger = logging.getLogger("lerobot_engine")


class IoMappingMixin:
    """Robot wiring — camera / state modality resolution and teardown."""

    def _init_robot(self, robot_type: str) -> None:
        """Create RobotClient + resolve camera / state mappings."""
        self._robot = RobotClient(robot_type)

        # Cameras: only those that match a policy input key
        # ``observation.images.<cam>``. Cameras advertised by the robot
        # but not consumed by the policy are silently ignored — same
        # behavior as GR00TInference.
        policy_image_keys = self._policy_image_keys()
        active = {}
        for cam in self._robot.camera_names:
            key = f"{_IMAGE_KEY_PREFIX}{cam}"
            if not policy_image_keys or key in policy_image_keys:
                active[cam] = key
        if not active and policy_image_keys:
            raise RuntimeError(
                "No cameras match the policy's expected input keys: "
                f"policy needs {sorted(policy_image_keys)}, robot has "
                f"{self._robot.camera_names}"
            )
        self._cameras = active

        # State modalities: sorted follower joint groups. We follow the
        # same convention groot uses (sorted modality names map to the
        # training-time concat order). Synthetic per-modality views (with
        # ``parent``) win over their leaf physical group; otherwise the
        # leaf group is used directly.
        groups = self._robot._config.get("joint_groups", {})
        parents = {cfg.get("parent") for cfg in groups.values() if cfg.get("parent")}
        modality_groups = []
        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            if cfg.get("parent"):
                modality_groups.append(name)
            elif name not in parents:
                modality_groups.append(name)
        modalities = sorted(name[len("follower_"):] for name in modality_groups)
        if not modalities:
            raise RuntimeError(
                f"No follower joint groups in robot_type={robot_type}"
            )

        # Mobile is sourced from sensors["odom"] in the new schema —
        # bridge it into observation.state alongside the joint states so
        # policies trained on the legacy physical_ai_server pipeline
        # (with mobile as a 3-vector modality) still see it.
        sensors = self._robot._config.get("sensors", {})
        self._has_mobile_state = "odom" in sensors
        if self._has_mobile_state:
            modalities = sorted(set(modalities) | {"mobile"})

        self._state_modalities = modalities
        self._action_keys = list(modalities)

        # Block until at least one frame from each sensor lands. 10 s is
        # generous — typical hardware comes up in <2 s.
        self._robot.wait_for_ready(timeout=10.0)
        logger.info(
            "Robot ready: cameras=%s state_modalities=%s",
            list(self._cameras.keys()),
            self._state_modalities,
        )

    def _teardown_robot(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None

    def _policy_image_keys(self) -> set:
        try:
            features = getattr(self._policy.config, "input_features", {}) or {}
            return {k for k in features.keys() if k.startswith(_IMAGE_KEY_PREFIX)}
        except Exception:
            return set()
