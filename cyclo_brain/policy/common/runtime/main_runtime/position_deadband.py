"""Continuous position deadband for explicitly selected joints.

Hold the last command inside the tolerance; outside it, follow only the excess.
The reference is the delivered position, never the previous input, so slow
intentional motion eventually releases the hold instead of being lost forever.
"""
import math
import os

import numpy as np


def left_arm_deadband_enabled():
    value = os.environ.get("CYCLO_LEFT_ARM_DEADBAND", "false").strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError("CYCLO_LEFT_ARM_DEADBAND must be true or false")
    return value in ("true", "1")


class PositionDeadband:
    def __init__(self, indices, radius_rad, *, lower, upper):
        self.indices = np.asarray(indices, dtype=int)
        self.radius_rad = float(radius_rad)
        if (self.indices.ndim != 1 or not self.indices.size
                or np.any(self.indices < 0)
                or len(set(self.indices)) != len(self.indices)):
            raise ValueError("deadband indices must be unique non-negative indices")
        if not math.isfinite(self.radius_rad) or not 0 < self.radius_rad <= 0.02:
            raise ValueError("left arm deadband must be in (0, 0.02] radians")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if (self.lower.shape != self.indices.shape or self.upper.shape != self.indices.shape
                or not np.all(self.lower < self.upper)):
            raise ValueError("invalid deadband joint limits")
        self.clear()

    def clear(self):
        self._held = None
        self._width = None

    def step(self, action):
        value = np.asarray(action, dtype=np.float64)
        if (value.ndim != 1 or value.size <= int(self.indices.max())
                or not np.isfinite(value).all()):
            raise ValueError("deadband requires a finite flat action vector")
        if self._width is not None and value.size != self._width:
            raise ValueError("deadband action width changed")
        target = value[self.indices]
        if np.any(target < self.lower - 1e-6) or np.any(target > self.upper + 1e-6):
            raise ValueError("deadband target exceeds configured joint limits")
        if self._held is None:
            # Initial-pose synchronization and the existing policy handoff
            # own physical startup. Do not introduce a second startup target.
            self._held = target.copy()
            self._width = value.size
        else:
            # Projection onto the target's tolerance band is continuous at
            # both boundaries, unlike snapping to the raw target on release.
            self._held = np.clip(self._held, target - self.radius_rad,
                                 target + self.radius_rad)
        result = value.copy()
        result[self.indices] = self._held
        return result
