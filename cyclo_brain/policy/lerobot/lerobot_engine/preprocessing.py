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
import time
from typing import Any, Dict, List

import numpy as np
import torch

from .constants import STATE_KEY as _STATE_KEY
from .input_plan import latest_input_plan
from .adapters import resolve_adapter
from inference_context import LatestValues
from inference_context.inputs import ReceivedValues, ResolvedInputs
from inference_context.observation import ObservationSession


logger = logging.getLogger("lerobot_engine")


class PreprocessingMixin:
    """RobotClient observation -> policy input batch."""

    def _build_observation(
        self, task_instruction: str, *, require_received=False, observation_after_s=None,
    ) -> Dict[str, Any]:
        """Pull raw sensor data from RobotClient and build a policy batch."""
        assert self._robot is not None

        plan, session = self._observation_plan(require_received)
        if require_received or session.requires_execution_context or any(session.live_ages.values()):
            try:
                if session.execution is not None:
                    session.check_execution(time.monotonic())
                provider, anchor = self._read_timestamped_inputs(
                    plan.spec, task_instruction, observation_after_s,
                    session=session,
                )
                return plan.assemble(provider, anchor_s=anchor)
            except (ValueError, RuntimeError) as exc:
                return self._fail(str(exc))

        snapshot_api = getattr(type(self._robot), "get_input_snapshot", None)
        if snapshot_api is not None:
            try:
                if callable(getattr(type(self._robot), "get_required_input_snapshot", None)):
                    snapshot = self._robot.get_required_input_snapshot(session.live_sources)
                else:
                    snapshot = self._robot.get_input_snapshot()
            except ValueError as exc:
                return self._fail(str(exc))
            images = snapshot["images"]
            joint_dict = snapshot["joint_positions"]
            sensors = snapshot["sensors"]
        else:
            # Compatibility with external RobotClient implementations.
            images = self._robot.get_images(format="rgb")
            joint_dict = self._robot.get_joint_positions()
            sensors = {"odom": self._robot.get_odom()} if "mobile" in self._state_modalities else {}

        sources = self._snapshot_values({"images": images, "joint_positions": joint_dict, "sensors": sensors})
        try:
            return plan.assemble(session.bind(LatestValues(sources), task_instruction))
        except ValueError as exc:
            return self._fail(str(exc))

    def _observation_plan(self, require_received):
        plans = getattr(self, "_input_plans", None)
        if plans is None:
            plans = self._input_plans = {}
        if require_received not in plans:
            definition = getattr(self, "_adapter_definition", None)
            compile_plan = definition.input_plan_factory if definition is not None else latest_input_plan
            plans[require_received] = compile_plan(
                self._cameras, self._state_modalities, self._transform_image, self._transform_state,
                max_age_s=1.0 if require_received else None,
                policy_config=self._policy.config,
                model_path=getattr(self, "_loaded_model_path", None),
            )
        plan = plans[require_received]
        sessions = getattr(self, "_observation_sessions", None)
        if sessions is None:
            sessions = self._observation_sessions = {}
        if require_received not in sessions:
            definition = getattr(self, "_adapter_definition", None)
            options = {"max_history_bytes": definition.history_max_bytes} if definition is not None else {}
            sessions[require_received] = ObservationSession(self._robot, plan.spec, **options)
        return plan, sessions[require_received]

    @staticmethod
    def _snapshot_values(snapshot):
        sources = {f"camera:{k}": v for k, v in snapshot["images"].items()}
        sources.update({f"joint:{k}": v for k, v in snapshot["joint_positions"].items()})
        sources.update({f"sensor:{k}": v for k, v in snapshot["sensors"].items()})
        for name, sensor in snapshot["sensors"].items():
            if isinstance(sensor, dict):
                sources.update({f"sensor:{name}.{field}": value for field, value in sensor.items()})
        return sources

    def _read_timestamped_inputs(self, spec, instruction, after_s, *, session):
        if not callable(getattr(type(self._robot), "get_input_snapshot", None)):
            raise ValueError("input plan requires timestamped RobotClient snapshots")
        started_s = time.monotonic()
        deadline = started_s + session.read_timeout_s
        selected_snapshot = getattr(type(self._robot), "get_required_input_snapshot", None)
        ages = set(session.live_ages.values())
        options = {"after_s": after_s, "max_age_s": None}
        if len(ages) == 1 and None not in ages:
            options["max_age_s"] = next(iter(ages))
        elif session.live_ages:
            options["max_age_by_source"] = session.live_ages
        if session.history is not None:
            options["readiness_check"] = lambda anchor: session.check_history(anchor, after_s=after_s)
        while True:
            try:
                snapshot = (
                    self._robot.get_required_input_snapshot(
                        session.live_sources, **options,
                    ) if callable(selected_snapshot) else self._robot.get_input_snapshot()
                )
            except ValueError as exc:
                anchor = time.monotonic()
                if anchor >= deadline:
                    raise ValueError(f"Required observations are not ready: {exc}") from exc
                time.sleep(min(0.01, deadline - anchor))
                continue
            anchor = snapshot.get("captured_monotonic_s")
            if anchor is None:
                anchor = time.monotonic()
            sources = self._snapshot_values(snapshot)
            received = ReceivedValues(
                sources, snapshot.get("reception_monotonic_timestamps", {}), after_s=after_s,
            )
            provider = session.bind(received, instruction, after_s=after_s)
            try:
                # Resolve history only once and check all inputs before GPU work.
                resolved = ResolvedInputs(spec, provider, anchor)
                if session.history is not None:
                    self._observation_wait_s = max(0., anchor - started_s)
                session.mark_ready()
                return resolved, anchor
            except ValueError as exc:
                if anchor >= deadline:
                    raise ValueError(f"Required observations are not ready: {exc}") from exc
                time.sleep(min(0.01, deadline - anchor))

    def _transform_image(self, camera, policy_key, image):
        cfg = self._robot._config.get("cameras", {}).get(camera, {})
        try:
            return self._image_preprocessing.apply(
                image, policy_key, rotation_deg=cfg.get("rotation_deg", 0)
            ).to(self._device)
        except Exception as exc:
            raise ValueError(f"Camera preprocessing failed for {camera}: {exc}") from exc

    def _transform_state(self, values):
        state_parts: List[np.ndarray] = []
        for modality, value in zip(self._state_modalities, values):
            if modality == "mobile":
                odom = value
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
            positions = value
            if positions is None or len(positions) == 0:
                raise ValueError(f"Missing joint group: {modality}")
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
        if getattr(self, "_step_adapter", None) is not None and flat_state.size != expected:
            raise ValueError(f"step state dimension changed: {flat_state.size} != {expected}")
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
        return torch.from_numpy(flat_state).unsqueeze(0).to(self._device)

    def _validate_camera_shapes(self, batch):
        # Allow the saved processor to resize before checking the stack contract.
        definition = getattr(self, "_adapter_definition", None)
        if definition is None:
            definition = resolve_adapter(getattr(self._policy.config, "type", None))
        if definition.batch_validator is not None:
            definition.batch_validator(self._policy.config, batch, self._cameras.values())
