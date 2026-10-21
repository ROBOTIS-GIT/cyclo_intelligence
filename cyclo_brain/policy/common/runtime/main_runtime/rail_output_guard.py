"""Validate RAIL commands and feedback without changing requested positions."""
import math

import numpy as np


class RailOutputGuard:
    def __init__(self, max_interval_s=.1):
        if not math.isfinite(max_interval_s) or max_interval_s <= 0:
            raise ValueError('RAIL command interval must be positive and finite')
        self.max_interval_s = float(max_interval_s)
        self.reset()

    def reset(self):
        self.positions = None
        self.last_time = None

    def update(self, target, now, initial_positions=None):
        target = np.asarray(target, dtype=float)
        if (target.ndim != 1 or target.size not in (8, 16)
                or not np.isfinite(target).all() or not math.isfinite(now)):
            raise ValueError('Invalid RAIL target or command timestamp')
        if self.positions is None:
            initial = np.asarray(initial_positions, dtype=float)
            if initial.shape != target.shape or not np.isfinite(initial).all():
                raise ValueError('Fresh current joint positions required to initialize RAIL output')
            self.positions = initial.copy()
            self.last_time = now
            return initial.copy()
        if target.shape != self.positions.shape:
            raise ValueError('RAIL action dimensions changed without reset')
        dt = now-self.last_time
        if not 0 < dt <= self.max_interval_s:
            raise ValueError('RAIL command interval invalid or stale; stop required')
        self.positions = target.copy()
        self.last_time = now
        return self.positions.copy()
