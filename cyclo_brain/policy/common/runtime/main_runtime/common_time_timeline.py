"""Geometry-preserving TT-RTC execution with a shared, bounded segment clock.

A polyline corner cannot retain nonzero velocity and preserve exact geometry.
Use a rest-to-rest quintic time law on each segment (no dwell). All dimensions
share progress; selected arm joints determine duration. This conservative
retiming can slow even small, noisy segments. Source waypoints remain unchanged.
"""
from __future__ import annotations
import math
import threading
import time
import numpy as np
from .tt_rtc_timeline import TTActionTimeline


class CommonTimeTimeline(TTActionTimeline):
    def __init__(self, *, indices, max_velocity, max_acceleration, max_jerk,
                 source_hz=15., control_hz=100., lower=None, upper=None):
        super().__init__(source_hz, control_hz)
        self._lock = threading.RLock()
        self.indices = np.asarray(indices, dtype=int)
        if (self.indices.ndim != 1 or not len(self.indices)
                or len(set(self.indices)) != len(self.indices) or np.any(self.indices < 0)):
            raise ValueError('common time scaling requires unique arm indices')
        self.limits = np.asarray([max_velocity, max_acceleration, max_jerk], dtype=float)
        if not np.isfinite(self.limits).all() or np.any(self.limits <= 0):
            raise ValueError('common time limits must be finite and positive')
        self.lower = np.full(len(self.indices), -np.inf) if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.full(len(self.indices), np.inf) if upper is None else np.asarray(upper, dtype=float)
        if (self.lower.shape != self.indices.shape or self.upper.shape != self.indices.shape
                or not np.all(self.lower < self.upper)):
            raise ValueError('invalid common time joint limits')
        self._elapsed = 0.
        self._duration = 0.
        self._seed_pending = False

    def _validate_positions(self, actions):
        if (actions.ndim != 2 or actions.shape[1] <= max(self.indices)
                or not np.isfinite(actions).all()):
            raise ValueError('common time scaling requires finite action vectors')
        arm = actions[:, self.indices]
        if np.any(arm < self.lower - 1e-6) or np.any(arm > self.upper + 1e-6):
            raise ValueError('arm action is outside configured URDF position limits')

    def seed_action(self, action):
        action = np.asarray(action, dtype=float)
        self._validate_positions(action.reshape(1, -1))
        with self._lock:
            if self._last_output_action is not None:
                raise ValueError('cannot reseed an active common timeline')
            if self._action_dim is not None and len(action) != self._action_dim:
                raise ValueError('feedback dimension differs from action dimension')
            self._action_dim = len(action)
            self._anchor = action.copy()
            self._last_output_action = action.copy()
            self._seed_pending = True

    def segment_duration(self, start, end):
        distance = float(np.max(np.abs(np.asarray(end)[self.indices] - np.asarray(start)[self.indices])))
        # Exact maxima of s'=30u²(1-u)², |s''|, and |s'''| on [0,1].
        velocity, acceleration, jerk = self.limits
        return max(1./self._source_hz, 1.875*distance/velocity,
                   math.sqrt((10./math.sqrt(3.))*distance/acceleration),
                   (60.*distance/jerk)**(1./3.))

    @property
    def remaining_duration_s(self):
        with self._lock:
            points = list(self._buffer)
            if self._anchor is None:
                return sum(self.segment_duration(a,b) for a,b in zip(points,points[1:]))
            if not points:
                return 0.
            remaining = ((self._duration or self.segment_duration(self._anchor,points[0])) - self._elapsed)
            return max(0.,remaining) + sum(self.segment_duration(a,b) for a,b in zip(points,points[1:]))

    def push_actions(self, actions, scheduled_start_delay_s=None, align=True):
        chunk = np.asarray(actions,dtype=float)
        self._validate_positions(chunk)
        return super().push_actions(chunk,scheduled_start_delay_s,align)

    def replace_queued_actions(self, expected_actions, replacement_actions):
        replacement = np.asarray(replacement_actions,dtype=float)
        self._validate_positions(replacement)
        with self._lock:
            if self._duration and (not len(replacement) or not np.array_equal(replacement[0],self._buffer[0])):
                raise ValueError('cannot rewrite the active common time segment')
            return super().replace_queued_actions(expected_actions,replacement)

    def pop_action(self, elapsed_s=None):
        dt = 1./self._control_hz if elapsed_s is None else float(elapsed_s)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('control interval must be finite and positive')
        dt = min(dt,1./self._control_hz)
        with self._lock:
            if self._anchor is None:
                if not self._buffer:
                    return None
                self._anchor = self._buffer.popleft()
                self._seed_pending = True
            if self._seed_pending:
                self._seed_pending = False
                self._last_output_action = self._anchor.copy()
                return self._last_output_action.copy()
            if not self._buffer:
                self._underrun_ticks += 1
                if self._underrun_started_at is None:
                    self._underrun_started_at=time.monotonic();self._underrun_events+=1
                return None
            if self._underrun_started_at is not None:
                self._max_underrun_s=max(self._max_underrun_s,time.monotonic()-self._underrun_started_at)
                self._underrun_started_at=None
            target = self._buffer[0]
            if not self._duration:
                self._duration = self.segment_duration(self._anchor,target)
            self._elapsed = min(self._elapsed+dt,self._duration)
            u = self._elapsed/self._duration
            progress = u*u*u*(10.+u*(-15.+6.*u))
            output = self._anchor + progress*(target-self._anchor)
            if self._elapsed >= self._duration:
                self._anchor = self._buffer.popleft()
                output = self._anchor.copy()
                self._elapsed = self._duration = 0.
            self._last_output_action=output.copy()
            return output

    def preview_action(self, delay_s):
        raise RuntimeError('common time scaling does not use independent arm lookahead')

    def clear(self):
        with self._lock:
            super().clear()
            self._elapsed = self._duration = 0.
            self._seed_pending = False
