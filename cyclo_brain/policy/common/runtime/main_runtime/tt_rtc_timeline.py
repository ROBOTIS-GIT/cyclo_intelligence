#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Source-clock action timeline for TT-RTC robot control.

TT-RTC reasons about the model's discrete action tokens (15 Hz for the
current GR00T contract), while the robot command transport runs at 100 Hz.
This timeline deliberately keeps the queue in the source domain so
``peek_actions()`` exposes the exact clean prefix.  Only ``pop_action()``
resamples that trajectory for the robot clock.
"""

from __future__ import annotations

import collections
import threading
from fractions import Fraction
from typing import Optional

import numpy as np


class TTActionTimeline:
    """Append-only source action queue with control-rate interpolation."""

    def __init__(self, source_hz: float = 15.0, control_hz: float = 100.0):
        self._source_hz = float(source_hz)
        self._control_hz = float(control_hz)
        if self._source_hz <= 0.0:
            raise ValueError("source_hz must be positive")
        if self._control_hz < self._source_hz:
            raise ValueError("control_hz must be greater than or equal to source_hz")

        # Use an integer phase accumulator so 15/100 cannot drift over long
        # episodes.  A phase of denominator means the next source waypoint is
        # due on the current control tick.
        ratio = Fraction(str(self._source_hz)) / Fraction(str(self._control_hz))
        self._phase_step = ratio.numerator
        self._phase_denominator = ratio.denominator
        self._phase = 0

        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._anchor: Optional[np.ndarray] = None
        self._last_output_action: Optional[np.ndarray] = None
        self._action_dim: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def buffer_size(self) -> int:
        """Number of unconsumed model-rate actions, never control samples."""

        with self._lock:
            return len(self._buffer)

    @property
    def output_hz(self) -> float:
        return self._control_hz

    @property
    def last_action(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last_output_action is None:
                return None
            return self._last_output_action.copy()

    def peek_actions(self, count: Optional[int] = None) -> np.ndarray:
        """Copy the exact unconsumed source actions used as the clean prefix."""

        with self._lock:
            if count is not None and count < 0:
                raise ValueError("count must be non-negative")
            actions = list(self._buffer)
            if count is not None:
                actions = actions[:count]
            if not actions:
                return np.empty((0, self._action_dim or 0), dtype=np.float64)
            return np.stack([action.copy() for action in actions], axis=0)

    def push_actions(
        self,
        actions: np.ndarray,
        scheduled_start_delay_s: Optional[float] = None,
        align: bool = True,
    ) -> int:
        """Append raw source actions; timing/alignment is owned by TT-RTC."""

        del scheduled_start_delay_s, align
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"actions must be 2D (T, D); got shape {chunk.shape}")
        if not bool(np.isfinite(chunk).all()):
            raise ValueError("actions contain NaN or Inf")

        with self._lock:
            action_dim = int(chunk.shape[1])
            if self._action_dim is None:
                self._action_dim = action_dim
            elif action_dim != self._action_dim:
                raise ValueError(
                    f"action dimension changed from {self._action_dim} to {action_dim}"
                )
            for action in chunk:
                self._buffer.append(action.copy())
            return len(chunk)

    def pop_action(self) -> Optional[np.ndarray]:
        """Return one interpolated command at the control clock."""

        with self._lock:
            if self._anchor is None:
                if not self._buffer:
                    return None
                self._anchor = self._buffer.popleft()

            # Advance only at the beginning of a control tick.  This keeps a
            # waypoint in the clean-prefix queue until its exact source time.
            while self._phase >= self._phase_denominator and self._buffer:
                self._anchor = self._buffer.popleft()
                self._phase -= self._phase_denominator

            if self._buffer:
                alpha = self._phase / self._phase_denominator
                output = (1.0 - alpha) * self._anchor + alpha * self._buffer[0]
            else:
                # The last committed waypoint may be held for the remainder
                # of its source period while a bounded TT request completes.
                output = self._anchor.copy()

            self._phase += self._phase_step
            self._last_output_action = np.asarray(output, dtype=np.float64).copy()
            return self._last_output_action.copy()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._anchor = None
            self._last_output_action = None
            self._action_dim = None
            self._phase = 0

