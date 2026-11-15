"""Independent SG2 arm command conditioning at the control rate.

The model-rate TT-RTC queue stays in the model domain; only delivered commands
are conditioned. This changes closed-loop execution and must be evaluated for
tracking lag as well as noise reduction. Grippers and other modalities bypass
this module. No transport or robot commands are created here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

import numpy as np


@dataclass(frozen=True)
class ArmSmoothingConfig:
    low_pass_tau_s: float = 0.04
    max_velocity_rad_s: float = 1.5
    max_acceleration_rad_s2: float = 8.0
    max_jerk_rad_s3: float = 100.0
    lookahead_s: float = 0.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "lookahead_s":
                if not math.isfinite(value) or not 0 <= value <= 0.1:
                    raise ValueError("arm lookahead must be in [0, 0.1] seconds")
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"arm smoothing {name} must be finite and positive")

    @classmethod
    def from_environment(cls):
        names = {
            "low_pass_tau_s": "CYCLO_ARM_SMOOTHING_TAU_S",
            "max_velocity_rad_s": "CYCLO_ARM_MAX_VELOCITY_RAD_S",
            "max_acceleration_rad_s2": "CYCLO_ARM_MAX_ACCELERATION_RAD_S2",
            "max_jerk_rad_s3": "CYCLO_ARM_MAX_JERK_RAD_S3",
            "lookahead_s": "CYCLO_ARM_LOOKAHEAD_S",
        }
        return cls(**{key: float(os.environ[env]) for key, env in names.items() if env in os.environ})


def smoothing_enabled() -> bool:
    # Tracking lag changes execution fidelity. A policy backend must explicitly
    # opt in; model numerical equivalence alone is not task-level validation.
    value = os.environ.get("CYCLO_SG2_ARM_SMOOTHING", "false").strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError("CYCLO_SG2_ARM_SMOOTHING must be true or false")
    return value in ("true", "1")


class ArmCommandSmoother:
    """Causal low-pass targets followed by bounded online trajectories.

    Ruckig synchronization is disabled: one joint/arm never slows the other.
    A new target preserves position, velocity, and acceleration state instead
    of resetting at chunk boundaries. Reset only at a stopped lifecycle boundary.
    """

    def __init__(self, indices, *, config=None, lower=None, upper=None):
        from ruckig import InputParameter, OutputParameter, Ruckig, Synchronization

        self.indices = np.asarray(indices, dtype=int)
        if self.indices.ndim != 1 or len(self.indices) == 0 or len(set(self.indices)) != len(self.indices) or np.any(self.indices < 0):
            raise ValueError("arm indices must be unique non-negative indices")
        self.config = config or ArmSmoothingConfig()
        n = len(self.indices)
        self.lower = np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float)
        if self.lower.shape != (n,) or self.upper.shape != (n,) or not np.all(self.lower < self.upper):
            raise ValueError("invalid arm position limits")
        self._otg = Ruckig(n, 0.01)
        self._input = InputParameter(n)
        self._output = OutputParameter(n)
        self._input.synchronization = Synchronization.No
        self._input.max_velocity = [self.config.max_velocity_rad_s] * n
        self._input.max_acceleration = [self.config.max_acceleration_rad_s2] * n
        self._input.max_jerk = [self.config.max_jerk_rad_s3] * n
        self._input.target_velocity = [0.0] * n
        self._input.target_acceleration = [0.0] * n
        self._filtered = None
        self._width = None

    @property
    def initialized(self):
        return self._filtered is not None

    @property
    def velocity(self):
        return np.asarray(self._input.current_velocity, dtype=float).copy()

    @property
    def acceleration(self):
        return np.asarray(self._input.current_acceleration, dtype=float).copy()

    def clear(self):
        self._filtered = None
        self._width = None
        self._otg.reset()

    def _validate(self, action):
        values = np.asarray(action, dtype=float)
        if values.ndim != 1 or values.size <= int(self.indices.max()) or not np.isfinite(values).all():
            raise ValueError("smoothing requires a finite flat action vector")
        if self._width is not None and len(values) != self._width:
            raise ValueError("action width changed while smoothing")
        arm = values[self.indices]
        if np.any(arm < self.lower - 1e-6) or np.any(arm > self.upper + 1e-6):
            raise ValueError("arm action is outside configured URDF position limits")
        return values

    def reset(self, anchor):
        values = self._validate(anchor)
        self._width = len(values)
        self._filtered = values[self.indices].copy()
        self._input.current_position = self._filtered.tolist()
        self._input.current_velocity = [0.0] * len(self.indices)
        self._input.current_acceleration = [0.0] * len(self.indices)
        self._otg.reset()

    def step(self, target, dt):
        values = self._validate(target)
        if not self.initialized:
            raise RuntimeError("arm smoother must be seeded from a stopped pose first")
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("smoothing time step must be in (0, 0.1] seconds")
        alpha = -math.expm1(-dt / self.config.low_pass_tau_s)
        self._filtered += alpha * (values[self.indices] - self._filtered)
        self._input.target_position = self._filtered.tolist()
        self._otg.delta_time = dt
        result = self._otg.update(self._input, self._output)
        if int(result) < 0:
            raise RuntimeError(f"arm trajectory generation failed: {result}")
        command = values.copy()
        command[self.indices] = self._output.new_position
        # Online target reversals can require braking past the target. Never
        # publish a step outside the configured physical position interval.
        self._validate(command)
        self._output.pass_to_input(self._input)
        return command
