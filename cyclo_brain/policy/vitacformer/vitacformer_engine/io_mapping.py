"""SH5 sensor wiring and episode-start calibration."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

import numpy as np

from .sensor_client import ViTacFormerSensorClient

from .constants import (
    ACTION_KEYS,
    CAMERA_NAME,
    JOINT_NAMES,
    STATE_HISTORY_HZ,
    STATE_HISTORY_SIZE,
    TACTILE_BASELINE_SAMPLES,
    TACTILE_HISTORY_HZ,
    TACTILE_HISTORY_SIZE,
)


logger = logging.getLogger("vitacformer_engine")


class IoMappingMixin:
    def _init_robot(self, robot_type: str) -> None:
        if robot_type != "ffw_sh5_rev1":
            raise ValueError(
                "ViTacFormer backend supports only ffw_sh5_rev1, got "
                f"{robot_type!r}"
            )

        self._robot = ViTacFormerSensorClient(robot_type)
        if CAMERA_NAME not in self._robot.camera_names:
            raise RuntimeError(
                f"ViTacFormer requires {CAMERA_NAME}; available cameras are "
                f"{self._robot.camera_names}"
            )

        self._tactile_inputs = self._resolve_tactile_inputs(
            self._robot._config.get("tactile_modalities", [])
        )
        available_actions = set(self._robot.action_keys)
        missing_actions = [key for key in ACTION_KEYS if key not in available_actions]
        if missing_actions:
            raise RuntimeError(
                "ViTacFormer SH5 action groups are unavailable: "
                f"{missing_actions}; robot={sorted(available_actions)}"
            )
        action_dim = int(self._robot.action_dimension(list(ACTION_KEYS)))
        if action_dim != len(JOINT_NAMES):
            raise RuntimeError(
                "ViTacFormer action layout must be 54D, got "
                f"{action_dim}D for {list(ACTION_KEYS)}"
            )

        action_names = [
            name for key in ACTION_KEYS
            for name in self._robot._action_groups[key]["joint_names"]
        ]
        if action_names != list(JOINT_NAMES):
            raise RuntimeError("ViTacFormer action joint order does not match SH5 training")

        self._wait_for_inputs(timeout=10.0)
        self._tactile_baselines = self._calibrate_tactile_inputs()
        self._wait_for_histories()
        self._action_keys = list(ACTION_KEYS)
        logger.info(
            "ViTacFormer robot ready: camera=%s tactile=%s actions=%s",
            CAMERA_NAME,
            self._tactile_inputs,
            self._action_keys,
        )

    @staticmethod
    def _resolve_tactile_inputs(sensor_names) -> Dict[str, str]:
        sensors = [str(name) for name in sensor_names]
        resolved: Dict[str, str] = {}
        for side in ("left", "right"):
            matches = [name for name in sensors if side in name.lower()]
            if len(matches) != 1:
                raise RuntimeError(
                    f"ViTacFormer requires one {side} tactile sensor; "
                    f"matches={matches}, sensors={sensors}"
                )
            resolved[side] = matches[0]
        return resolved

    def _wait_for_inputs(self, timeout: float) -> None:
        assert self._robot is not None
        deadline = time.monotonic() + max(0.0, float(timeout))

        def missing() -> list[str]:
            result = []
            if not self._robot.is_image_ready(CAMERA_NAME):
                result.append(f"camera:{CAMERA_NAME}")
            for side, sensor_name in self._tactile_inputs.items():
                if not self._robot.is_sensor_ready(sensor_name):
                    result.append(f"tactile:{side}:{sensor_name}")
            return result

        while time.monotonic() < deadline:
            current = missing()
            if not current:
                return
            time.sleep(0.05)
        raise RuntimeError(f"ViTacFormer inputs not ready: {missing()}")

    def _calibrate_tactile_inputs(self) -> Dict[str, np.ndarray]:
        assert self._robot is not None
        sample_count = int(
            os.environ.get(
                "VITACFORMER_TACTILE_BASELINE_SAMPLES",
                str(TACTILE_BASELINE_SAMPLES),
            )
        )
        timeout = float(
            os.environ.get("VITACFORMER_TACTILE_BASELINE_TIMEOUT_S", "5.0")
        )
        if sample_count != TACTILE_BASELINE_SAMPLES:
            raise ValueError(
                "ViTacFormer tactile baseline must use exactly "
                f"{TACTILE_BASELINE_SAMPLES} samples, got {sample_count}"
            )

        baselines: Dict[str, np.ndarray] = {}
        for side, sensor_name in self._tactile_inputs.items():
            samples = self._robot.wait_for_tactile_samples(
                sensor_name,
                sample_count=sample_count,
                timeout=timeout,
            )
            if tuple(samples.shape) != (sample_count, 5, 3, 3):
                raise RuntimeError(
                    f"ViTacFormer {side} tactile calibration must be "
                    f"({sample_count}, 5, 3, 3), got {tuple(samples.shape)}"
                )
            baseline = np.rint(np.median(samples, axis=0)).astype(np.float32)
            if not np.isfinite(baseline).all():
                raise RuntimeError(
                    f"ViTacFormer {side} tactile baseline contains NaN or Inf"
                )
            baselines[side] = baseline
        return baselines

    def _wait_for_histories(self) -> None:
        assert self._robot is not None
        required_span = max(
            (STATE_HISTORY_SIZE - 1) / STATE_HISTORY_HZ,
            (TACTILE_HISTORY_SIZE - 1) / TACTILE_HISTORY_HZ,
        )
        timeout = max(2.0, required_span + 1.0)
        state = self._robot.wait_for_joint_position_history(
            list(JOINT_NAMES),
            history_size=STATE_HISTORY_SIZE,
            sample_hz=STATE_HISTORY_HZ,
            timeout=timeout,
        )
        if tuple(state.shape) != (STATE_HISTORY_SIZE, len(JOINT_NAMES)):
            raise RuntimeError(
                "ViTacFormer state history must be (6, 54), got "
                f"{tuple(state.shape)}"
            )
        for side, sensor_name in self._tactile_inputs.items():
            tactile = self._robot.wait_for_tactile_taxel_history(
                sensor_name,
                history_size=TACTILE_HISTORY_SIZE,
                sample_hz=TACTILE_HISTORY_HZ,
                timeout=timeout,
            )
            if tuple(tactile.shape) != (TACTILE_HISTORY_SIZE, 5, 3, 3):
                raise RuntimeError(
                    f"ViTacFormer {side} tactile history must be "
                    f"(18, 5, 3, 3), got {tuple(tactile.shape)}"
                )

    def _teardown_robot(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.close()
        finally:
            self._robot = None
