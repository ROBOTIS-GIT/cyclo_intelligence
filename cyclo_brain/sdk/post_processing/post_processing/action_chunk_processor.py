#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action chunk post-processing pipeline.

Extracted from ``orchestrator/inference/inference_manager.py`` (Step 4-A).
Pure numpy + stdlib — no ROS2, Zenoh, or rclpy dependency — so the control
loop (Process B) can own it, and so it can be unit-tested outside a ROS2
environment.

Pipeline on each incoming raw chunk (shape ``(T, D)`` at ``inference_hz``):

    1. L2 align against ``last_action`` on the first
       ``chunk_align_window_s * inference_hz`` raw waypoints — skips past
       waypoints the robot has already crossed while preventing "jump ahead"
       on loop trajectories (A→B→C→B→A).
    2. Linear interpolation from ``inference_hz`` up to ``control_hz``.
    3. Linear blend of the first ``BLEND_DURATION_S * control_hz`` waypoints
       toward ``last_action`` so chunk boundaries don't discontinuously jump.
    4. Append to an internal buffer the control loop pops from.
"""

from __future__ import annotations

import collections
import threading
from typing import Dict, List, Optional

import numpy as np


class ActionChunkProcessor:

    BLEND_DURATION_S = 0.2

    def __init__(
        self,
        inference_hz: float = 15.0,
        control_hz: float = 100.0,
        chunk_align_window_s: float = 0.3,
    ):
        self._inference_hz = float(inference_hz)
        self._control_hz = float(control_hz)
        self._chunk_align_window_s = max(0.0, float(chunk_align_window_s))
        self._blend_steps = max(1, int(self.BLEND_DURATION_S * self._control_hz))

        self._buffer: collections.deque = collections.deque()
        self._last_action: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def last_action(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_action is None else self._last_action.copy()

    def push_chunk(self, chunk: np.ndarray) -> int:
        """Align, interpolate, blend, and append ``chunk`` to the buffer.

        Returns the number of interpolated waypoints appended. May be 0 if
        the chunk collapses to empty after L2 alignment.
        """
        if chunk.ndim != 2:
            raise ValueError(f"chunk must be 2D (T, D); got shape {chunk.shape}")

        with self._lock:
            aligned = self._align(chunk)
            if len(aligned) == 0:
                return 0

            interpolated = self._interpolate(aligned)
            blended = self._blend(interpolated)

            for action in blended:
                self._buffer.append(action)

            if len(blended) > 0:
                self._last_action = blended[-1].copy()
            return len(blended)

    def pop_action(self) -> Optional[np.ndarray]:
        """Pop the next action.

        Returns the next buffered action, or a copy of ``last_action`` if
        the buffer has drained (hold last command), or ``None`` if nothing
        has been pushed yet.
        """
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
            return None if self._last_action is None else self._last_action.copy()

    def clear(self) -> None:
        """Drop all buffered waypoints and forget ``last_action``.

        The next ``push_chunk`` will behave as the first push — no alignment
        and no blending.
        """
        with self._lock:
            self._buffer.clear()
            self._last_action = None

    # -- internals ---------------------------------------------------------
    # All internals assume the caller holds ``self._lock``.

    def _align(self, chunk: np.ndarray) -> np.ndarray:
        if self._last_action is None or len(chunk) <= 1:
            return chunk
        search_n = int(round(self._chunk_align_window_s * self._inference_hz))
        search_n = max(1, min(search_n, len(chunk)))
        distances = np.linalg.norm(chunk[:search_n] - self._last_action, axis=1)
        best_idx = int(np.argmin(distances))
        start_idx = best_idx + 1
        if start_idx >= len(chunk):
            return chunk[:0]
        return chunk[start_idx:]

    def _interpolate(self, chunk: np.ndarray) -> np.ndarray:
        T, D = chunk.shape
        if T < 2:
            return chunk
        t_original = np.arange(T) / self._inference_hz
        duration = (T - 1) / self._inference_hz
        n_interp = int(round(duration * self._control_hz)) + 1
        t_interp = np.linspace(0, duration, n_interp)
        out = np.empty((n_interp, D))
        for d in range(D):
            out[:, d] = np.interp(t_interp, t_original, chunk[:, d])
        return out

    def _blend(self, chunk: np.ndarray) -> np.ndarray:
        if self._last_action is None or len(chunk) == 0:
            return chunk
        chunk = chunk.copy()
        n_blend = min(self._blend_steps, len(chunk))
        for i in range(n_blend):
            alpha = (i + 1) / (n_blend + 1)
            chunk[i] = (1 - alpha) * self._last_action + alpha * chunk[i]
        return chunk


def build_action_joint_map(
    action_keys: List[str],
    joint_order: Dict[str, List[str]],
) -> Dict[str, str]:
    """Map model action modality keys to joint_order leader groups.

    For each ``key`` in ``action_keys``, looks up
    ``"joint_order.leader_<key>"`` in ``joint_order``. Keys without a
    matching leader group are silently dropped — the caller (Process A's
    LOAD handler) is responsible for logging if that is unexpected.
    """
    action_joint_map: Dict[str, str] = {}
    for key in action_keys:
        leader_key = f"joint_order.leader_{key}"
        if leader_key in joint_order:
            action_joint_map[key] = leader_key
    return action_joint_map


def split_action(
    action: np.ndarray,
    action_joint_map: Dict[str, str],
    joint_order: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """Slice a flat action vector into per-publisher-key segments.

    Returns ``{publisher_key: values}`` where ``publisher_key`` is the
    leader group with the ``"joint_order."`` prefix stripped. The caller
    (Process B) decides whether to wrap each segment as ``Twist`` or
    ``JointTrajectory`` based on the key (e.g. "mobile" substring).
    """
    result: Dict[str, np.ndarray] = {}
    offset = 0
    for _modality_key, leader_group in action_joint_map.items():
        joint_names = joint_order.get(leader_group, [])
        n_joints = len(joint_names)
        if n_joints == 0:
            continue
        values = action[offset:offset + n_joints]
        offset += n_joints
        publisher_key = leader_group.removeprefix("joint_order.")
        result[publisher_key] = values
    return result
