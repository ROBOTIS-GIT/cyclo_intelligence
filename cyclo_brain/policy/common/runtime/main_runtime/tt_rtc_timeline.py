#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Source-clock action timeline for TT-RTC robot control.

TT-RTC reasons about the model's discrete action tokens (15 Hz for the
current GR00T contract), while the robot command transport uses Control Hz.
This timeline deliberately keeps the queue in the source domain so
``peek_actions()`` exposes the exact clean prefix.  Only ``pop_action()``
resamples that trajectory for the robot clock.
"""

from __future__ import annotations

import collections
import threading
import time
from fractions import Fraction
from typing import Optional

import numpy as np


class TTActionTimeline:
    """Source action queue with control-rate interpolation."""

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
        self._underrun_ticks = 0
        self._underrun_events = 0
        self._underrun_started_at = None
        self._max_underrun_s = 0.0

    @property
    def timing_stats(self) -> dict:
        """Measure starvation after startup, without logging on control ticks."""
        with self._lock:
            active_s = (0.0 if self._underrun_started_at is None else
                        max(0.0, time.monotonic() - self._underrun_started_at))
            return dict(underrun_ticks=self._underrun_ticks,
                        underrun_events=self._underrun_events,
                        max_underrun_s=max(self._max_underrun_s, active_s))

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

    def preview_action(self, delay_s: float) -> Optional[np.ndarray]:
        """Sample committed future motion relative to the last output tick.

        This is read-only: source phase, prefix and starvation counters do not
        advance. At the queue tail clamp to its last known target; never
        extrapolate an unavailable policy action. Used only by the explicitly
        enabled arm trajectory conditioner, not by the policy prefix builder.
        """
        if not np.isfinite(delay_s) or delay_s < 0:
            raise ValueError("preview delay must be finite and non-negative")
        with self._lock:
            if self._anchor is None or self._last_output_action is None:
                return None
            points = [self._anchor, *self._buffer]
            phase = max(0, self._phase - self._phase_step) / self._phase_denominator
            offset = min(phase + delay_s * self._source_hz, len(points) - 1)
            index = int(offset)
            if index >= len(points) - 1:
                return points[-1].copy()
            alpha = offset - index
            return (1 - alpha) * points[index] + alpha * points[index + 1]

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
            if len(chunk) and not self._buffer and self._phase >= self._phase_denominator:
                # Resume from the held target, without catching up wall-clock
                # time by skipping newly arrived source actions.
                self._phase = 0
            for action in chunk:
                self._buffer.append(action.copy())
            return len(chunk)

    def replace_queued_actions(
        self,
        expected_actions: np.ndarray,
        replacement_actions: np.ndarray,
    ) -> int:
        """Atomically replace the exact unconsumed source queue.

        TT-RTC temporal overlap revises future, not-yet-executed waypoints
        while the control thread is consuming the committed prefix.  The
        compare-and-replace guard prevents a response from overwriting a queue
        that advanced or changed after it was inspected.
        """

        expected = np.asarray(expected_actions, dtype=np.float64)
        replacement = np.asarray(replacement_actions, dtype=np.float64)
        for name, actions in (
            ("expected_actions", expected),
            ("replacement_actions", replacement),
        ):
            if actions.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D (T, D); got shape {actions.shape}"
                )
            if not bool(np.isfinite(actions).all()):
                raise ValueError(f"{name} contains NaN or Inf")

        with self._lock:
            action_dim = self._action_dim
            if action_dim is None:
                candidate_dims = {
                    int(actions.shape[1])
                    for actions in (expected, replacement)
                    if actions.shape[0]
                }
                if len(candidate_dims) > 1:
                    raise ValueError("replacement action dimensions do not match")
                if candidate_dims:
                    action_dim = candidate_dims.pop()
            for name, actions in (
                ("expected_actions", expected),
                ("replacement_actions", replacement),
            ):
                if action_dim is not None and actions.shape[1] != action_dim:
                    raise ValueError(
                        f"{name} action dimension must be {action_dim}; "
                        f"got {actions.shape[1]}"
                    )

            current = (
                np.stack(list(self._buffer), axis=0)
                if self._buffer
                else np.empty((0, action_dim or expected.shape[1]), dtype=np.float64)
            )
            if current.shape != expected.shape or not bool(
                np.array_equal(current, expected)
            ):
                raise RuntimeError("queued actions changed before replacement")

            self._buffer.clear()
            for action in replacement:
                self._buffer.append(action.copy())
            if self._action_dim is None and action_dim is not None:
                self._action_dim = action_dim
            return len(replacement)

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
                if self._phase >= self._phase_denominator:
                    # Position controllers retain their last target; the
                    # caller sends idle (zero velocity) for Twist commands.
                    self._underrun_ticks += 1
                    if self._underrun_started_at is None:
                        self._underrun_started_at = time.monotonic()
                        self._underrun_events += 1
                    return None
                output = self._anchor.copy()

            if self._underrun_started_at is not None:
                self._max_underrun_s = max(
                    self._max_underrun_s, time.monotonic() - self._underrun_started_at,
                )
                self._underrun_started_at = None
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
            self._underrun_ticks = 0
            self._underrun_events = 0
            self._underrun_started_at = None
            self._max_underrun_s = 0.0
