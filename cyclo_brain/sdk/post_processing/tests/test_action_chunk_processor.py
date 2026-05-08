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

"""Unit tests for ActionChunkProcessor.

Uses only ``unittest`` + ``numpy`` so it runs without pytest or any ROS2
environment. Run with::

    python3 -m unittest tests.test_action_chunk_processor

from ``cyclo_brain/sdk/post_processing/``.
"""

import pathlib
import sys
import unittest

import numpy as np

# Add sibling post_processing/ package to sys.path
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from post_processing import (  # noqa: E402
    ActionChunkProcessor,
    build_action_joint_map,
    split_action,
)


def _ramp(n_steps: int, n_dim: int, start: float = 0.0, step: float = 1.0) -> np.ndarray:
    """Monotonic ramp chunk where each joint starts at `start` and increments by `step`."""
    base = np.arange(n_steps, dtype=float) * step + start
    return np.tile(base[:, None], (1, n_dim))


class InterpolateTests(unittest.TestCase):

    def test_upsamples_length_correctly(self):
        # 15 Hz → 100 Hz. T=16 raw waypoints span (T-1)/15 = 1.0 s of motion.
        # Expect int(round(1.0 * 100)) + 1 = 101 interpolated waypoints.
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        chunk = _ramp(16, 2)
        n = proc.push_chunk(chunk)
        self.assertEqual(n, 101)
        self.assertEqual(proc.buffer_size, 101)

    def test_endpoints_preserved(self):
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        chunk = _ramp(10, 3, start=5.0, step=2.0)  # range [5, 23]
        proc.push_chunk(chunk)
        first = proc.pop_action()
        # Drain the rest
        last = first
        while proc.buffer_size > 0:
            last = proc.pop_action()
        np.testing.assert_allclose(first, chunk[0])
        np.testing.assert_allclose(last, chunk[-1])

    def test_single_waypoint_passthrough(self):
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        chunk = np.array([[1.0, 2.0, 3.0]])
        n = proc.push_chunk(chunk)
        self.assertEqual(n, 1)
        np.testing.assert_allclose(proc.pop_action(), chunk[0])


class AlignTests(unittest.TestCase):

    def test_first_push_uses_chunk_as_is(self):
        # No last_action → no alignment, no blend. Ramp preserved.
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        chunk = _ramp(16, 2, start=0.0, step=1.0)
        proc.push_chunk(chunk)
        np.testing.assert_allclose(proc.pop_action(), chunk[0])

    def test_skips_past_waypoints(self):
        # last_action at raw waypoint 3 → aligner should start at waypoint 4.
        proc = ActionChunkProcessor(
            inference_hz=15.0, control_hz=100.0, chunk_align_window_s=1.0
        )
        first_chunk = _ramp(16, 2, start=0.0, step=1.0)
        proc.push_chunk(first_chunk)
        # Force last_action to raw waypoint 3 so alignment is deterministic.
        proc.clear()
        proc._last_action = first_chunk[3].copy()  # noqa: SLF001 — test hook

        second_chunk = _ramp(16, 2, start=0.0, step=1.0)
        n = proc.push_chunk(second_chunk)
        # Waypoints 4..15 (12 waypoints) → interpolated to 100 Hz.
        # Span = (12-1)/15 s → int(round(11/15 * 100)) + 1 = 74 waypoints.
        self.assertEqual(n, 74)

    def test_loop_trajectory_window_restriction(self):
        # Trajectory visits 0→1→2→3→2→1→0. last_action = 0.5 lies between
        # raw waypoints 0 and 1 AND between waypoints 5 and 6. Without the
        # window restriction, L2 would jump to waypoint 5 (second 0.5 crossing).
        # With a 0.3 s window at 15 Hz = 4 waypoints search, L2 stays in the
        # first segment.
        proc = ActionChunkProcessor(
            inference_hz=15.0, control_hz=100.0, chunk_align_window_s=0.3
        )
        chunk = np.array([[0.0], [1.0], [2.0], [3.0], [2.0], [1.0], [0.0]])
        proc._last_action = np.array([0.5])  # noqa: SLF001 — test hook
        proc.push_chunk(chunk)
        # After align (window=4, closest in [0,1,2,3] is idx 0), start at idx 1.
        # Remaining waypoints: [1,2,3,2,1,0] — if loop jump happened we'd get
        # fewer. Span (6-1)/15 s → int(round(5/15*100))+1 = 34 waypoints.
        self.assertEqual(proc.buffer_size, 34)


class BlendTests(unittest.TestCase):

    def test_no_blend_when_no_last_action(self):
        # Cold start: first waypoint published equals chunk[0].
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        chunk = np.array([[0.0], [1.0]])
        proc.push_chunk(chunk)
        np.testing.assert_allclose(proc.pop_action(), np.array([0.0]))

    def test_boundary_blends_from_last_action(self):
        # Seed last_action = 10.0 directly so the buffer is empty but
        # last_action is set — the state the control loop sees when it
        # has drained the previous chunk and a new chunk arrives.
        proc = ActionChunkProcessor(
            inference_hz=15.0, control_hz=100.0, chunk_align_window_s=0.0
        )
        proc._last_action = np.array([10.0])  # noqa: SLF001

        # Enough waypoints to exercise the full blend window.
        second = np.full((50, 1), 0.0)
        proc.push_chunk(second)

        first_blended = proc.pop_action()
        # alpha = 1 / (blend_steps + 1); expected = (1 - alpha) * 10 + alpha * 0.
        alpha = 1 / (proc._blend_steps + 1)  # noqa: SLF001
        expected = (1 - alpha) * 10.0
        np.testing.assert_allclose(first_blended, np.array([expected]))

    def test_blend_decays_monotonically_to_chunk(self):
        # After n_blend waypoints, subsequent values equal chunk-only values.
        proc = ActionChunkProcessor(
            inference_hz=15.0, control_hz=100.0, chunk_align_window_s=0.0
        )
        # First chunk sets last_action.
        proc._last_action = np.array([10.0])  # noqa: SLF001
        chunk = np.full((50, 1), 0.0)  # all zeros at 15 Hz
        proc.push_chunk(chunk)
        values = []
        while proc.buffer_size > 0:
            values.append(proc.pop_action()[0])
        values = np.array(values)
        # Expect first `n_blend` values to decay from ~(10*n/(n+1)) to 0, then
        # stay at 0 for the remainder.
        n_blend = proc._blend_steps  # noqa: SLF001
        # Monotonic decay over the blend region
        self.assertTrue(np.all(np.diff(values[:n_blend]) <= 0))
        # Post-blend region is zero
        np.testing.assert_allclose(values[n_blend:], 0.0)


class BufferAndStateTests(unittest.TestCase):

    def test_pop_before_push_returns_none(self):
        proc = ActionChunkProcessor()
        self.assertIsNone(proc.pop_action())

    def test_buffer_drain_holds_last_action(self):
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        chunk = np.array([[1.0], [2.0]])
        proc.push_chunk(chunk)
        # Drain
        last = None
        while proc.buffer_size > 0:
            last = proc.pop_action()
        # Now buffer is empty — next pop returns last_action (a copy)
        held = proc.pop_action()
        np.testing.assert_allclose(held, last)
        # Subsequent pops continue to return the held value
        np.testing.assert_allclose(proc.pop_action(), last)

    def test_clear_resets_last_action(self):
        proc = ActionChunkProcessor(inference_hz=15.0, control_hz=100.0)
        proc.push_chunk(np.array([[1.0], [2.0]]))
        self.assertIsNotNone(proc.last_action)
        proc.clear()
        self.assertIsNone(proc.last_action)
        self.assertEqual(proc.buffer_size, 0)
        # After clear, next push behaves as first push — no blend.
        proc.push_chunk(np.array([[0.0], [0.0]]))
        np.testing.assert_allclose(proc.pop_action(), np.array([0.0]))

    def test_1d_chunk_raises(self):
        proc = ActionChunkProcessor()
        with self.assertRaises(ValueError):
            proc.push_chunk(np.array([1.0, 2.0, 3.0]))


class ActionJointMapTests(unittest.TestCase):

    def test_maps_matching_keys(self):
        joint_order = {
            "joint_order.leader_arm_left": ["l_j1", "l_j2"],
            "joint_order.leader_arm_right": ["r_j1"],
        }
        result = build_action_joint_map(["arm_left", "arm_right"], joint_order)
        self.assertEqual(
            result,
            {
                "arm_left": "joint_order.leader_arm_left",
                "arm_right": "joint_order.leader_arm_right",
            },
        )

    def test_silently_drops_unmatched_keys(self):
        joint_order = {"joint_order.leader_arm_left": ["l_j1"]}
        result = build_action_joint_map(["arm_left", "nonexistent"], joint_order)
        self.assertEqual(result, {"arm_left": "joint_order.leader_arm_left"})


class SplitActionTests(unittest.TestCase):

    def test_splits_flat_action_by_group(self):
        joint_order = {
            "joint_order.leader_arm_left": ["a", "b"],
            "joint_order.leader_arm_right": ["c", "d", "e"],
        }
        action_joint_map = {
            "arm_left": "joint_order.leader_arm_left",
            "arm_right": "joint_order.leader_arm_right",
        }
        action = np.arange(5, dtype=float)  # [0, 1, 2, 3, 4]
        result = split_action(action, action_joint_map, joint_order)
        self.assertEqual(set(result), {"leader_arm_left", "leader_arm_right"})
        np.testing.assert_allclose(result["leader_arm_left"], [0.0, 1.0])
        np.testing.assert_allclose(result["leader_arm_right"], [2.0, 3.0, 4.0])

    def test_skips_zero_joint_groups(self):
        joint_order = {
            "joint_order.leader_arm_left": ["a"],
            "joint_order.leader_empty": [],
        }
        action_joint_map = {
            "arm_left": "joint_order.leader_arm_left",
            "empty": "joint_order.leader_empty",
        }
        action = np.array([42.0])
        result = split_action(action, action_joint_map, joint_order)
        self.assertEqual(set(result), {"leader_arm_left"})
        np.testing.assert_allclose(result["leader_arm_left"], [42.0])

    def test_mobile_key_passes_through_intact(self):
        # split_action doesn't interpret "mobile" — Process B does when it
        # decides Twist vs JointTrajectory. Here we just verify the slice.
        joint_order = {"joint_order.leader_mobile_base": ["x", "y", "theta"]}
        action_joint_map = {"mobile_base": "joint_order.leader_mobile_base"}
        action = np.array([0.5, -0.2, 0.1])
        result = split_action(action, action_joint_map, joint_order)
        self.assertEqual(set(result), {"leader_mobile_base"})
        np.testing.assert_allclose(result["leader_mobile_base"], [0.5, -0.2, 0.1])


if __name__ == "__main__":
    unittest.main()
