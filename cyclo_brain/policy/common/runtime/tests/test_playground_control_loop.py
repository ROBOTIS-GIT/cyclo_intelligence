#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_client_stub)

from main_runtime.playground_control_loop import ControlLoop  # noqa: E402
from main_runtime.tt_rtc_timeline import TTActionTimeline  # noqa: E402
from action_chunk_processing import ActionChunkProcessor  # noqa: E402


class FakeProcessor:
    output_hz = 100.0

    def __init__(self, actions=None, buffer_size=100) -> None:
        self._actions = list(actions or [])
        self.buffer_size = buffer_size
        self.clear_count = 0
        self.pushed_chunks = []
        self.scheduled_delays = []
        self.align_flags = []
        self.boundary_blend_source_steps = []

    def pop_action(self):
        if self._actions:
            return self._actions.pop(0)
        return None

    def clear(self) -> None:
        self.clear_count += 1
        self._actions.clear()
        self.buffer_size = 0

    def push_actions(
        self,
        chunk,
        scheduled_start_delay_s=None,
        align=True,
        boundary_blend_source_steps=None,
    ):
        data = np.asarray(chunk, dtype=np.float64)
        self.pushed_chunks.append(data.copy())
        self.scheduled_delays.append(scheduled_start_delay_s)
        self.align_flags.append(bool(align))
        self.boundary_blend_source_steps.append(boundary_blend_source_steps)
        self.buffer_size += len(data)
        return len(data)


class FakeRobot:
    def __init__(self) -> None:
        self.commands = []
        self.previews = []
        self.idles = []
        self.syncs = []
        self.holds = []
        self.handoff_requests = []
        self.action_keys = ["arm"]

    def get_action_joint_limits(self, action_keys):
        # Generic protocol fixtures do not model SG2 physical joints.
        return []

    def publish_action(self, action, action_keys) -> None:
        self.commands.append((np.asarray(action).copy(), list(action_keys)))

    def publish_action_preview(self, action, action_keys) -> None:
        self.previews.append((np.asarray(action).copy(), list(action_keys)))

    def publish_idle_action(self, action_keys) -> None:
        self.idles.append(list(action_keys))

    def publish_initial_pose_sync(
        self,
        action,
        action_keys,
        minimum_duration_s=5.0,
        maximum_duration_s=60.0,
    ) -> float:
        self.syncs.append((
            np.asarray(action).copy(),
            list(action_keys),
            minimum_duration_s,
            maximum_duration_s,
        ))
        return 8.0

    def publish_current_pose_hold(self, action_keys, duration_s=0.1) -> None:
        self.holds.append((list(action_keys), duration_s))
        self._hold_clock = time.monotonic()
        return {"positions": {"arm": 0.0}, "commanded_at": self._hold_clock}

    def current_hold_feedback(self):
        self._hold_clock += .021
        return {"positions": {"arm": 0.0}, "received": {"arm": self._hold_clock}}

    def build_policy_handoff_ramp(
        self,
        action,
        action_keys,
        sample_hz=15.0,
        maximum_duration_s=60.0,
    ):
        target = np.asarray(action, dtype=np.float64)
        self.handoff_requests.append((
            target.copy(),
            list(action_keys),
            float(sample_hz),
            float(maximum_duration_s),
        ))
        return np.stack((np.zeros_like(target), target)), 0.5

    def close(self) -> None:
        pass


class FakeRequester:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []
        self.keyword_calls = []

    def get_action(
        self,
        task_instruction,
        action_policy_mode="base",
        action_request_mode="async",
        **_kwargs,
    ):
        self.calls.append((task_instruction, action_policy_mode))
        self.keyword_calls.append({
            "action_request_mode": action_request_mode,
            **_kwargs,
        })
        return self.response


class ControlLoopSafetyTests(unittest.TestCase):
    def test_robot_command_precedes_optional_viewer_publication(self):
        events = []
        robot = FakeRobot()
        robot.publish_action = lambda *args: events.append("robot")
        robot.publish_action_preview = lambda *args: events.append("preview")
        loop = self._make_loop(FakeProcessor(actions=[np.array([.1, .2])]), robot)
        loop._publish_to_robot = True
        loop.tick()
        self.assertEqual(events, ["robot", "preview"])

    def test_overrun_waits_before_next_command_and_shutdown_interrupts_wait(self):
        loop = ControlLoop(requester=object())
        clock = [0.0]
        tick_times = []
        waits = []
        class Stop:
            def is_set(self):
                return len(tick_times) >= 2
            def wait(self, seconds):
                waits.append(seconds)
                clock[0] += seconds
        loop._shutdown = Stop()
        def tick():
            tick_times.append(clock[0])
            clock[0] += .025 if len(tick_times) == 1 else .001
        loop.tick = tick
        loop._tick_period = lambda: .01
        with patch("main_runtime.playground_control_loop.time.monotonic", side_effect=lambda: clock[0]):
            loop.run()
        self.assertAlmostEqual(tick_times[1], .035)
        self.assertTrue(all(delay > 0 for delay in waits))

    def _make_loop(self, processor: FakeProcessor, robot: FakeRobot) -> ControlLoop:
        loop = ControlLoop(requester=object())
        loop._running = True
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        return loop

    def test_dry_run_publishes_preview_without_robot_command(self) -> None:
        action = np.asarray([0.1, 0.2], dtype=np.float64)
        processor = FakeProcessor(actions=[action])
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)

        loop.set_publish_to_robot(False)
        loop.tick()

        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(len(robot.previews), 1)
        np.testing.assert_allclose(robot.previews[0][0], action)

    def test_robot_mode_publishes_preview_and_robot_command(self) -> None:
        action = np.asarray([0.3, 0.4], dtype=np.float64)
        processor = FakeProcessor(actions=[action])
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True

        loop.tick()

        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(len(robot.previews), 1)
        np.testing.assert_allclose(robot.commands[0][0], action)
        np.testing.assert_allclose(robot.previews[0][0], action)

    def test_sg2_gripper_threshold_is_per_command_for_both_hands(self) -> None:
        # Exercise final preview/command routing, not only the helper.
        values = [0.69, 0.7, 0.8, 0.95, 1.1, 0.1]
        for mode in ("base", "rlt"):
            for publish in (False, True):
                with self.subTest(mode=mode, publish=publish):
                    actions = [np.full(16, value) for value in values]
                    originals = [a.copy() for a in actions]
                    robot = FakeRobot()
                    loop = self._make_loop(FakeProcessor(actions=actions), robot)
                    loop._robot_type = "ffw_sg2_rev1"
                    loop._action_keys = ["arm_left", "arm_right"]
                    loop._publish_to_robot = publish
                    loop._active_action_policy_mode = mode
                    loop._active_rlt_robot_override = mode == "rlt"
                    for _ in values:
                        loop.tick()
                    expected_grippers = [0.69, 0.9, 0.9, 0.9, 0.9, 0.1]
                    for index in (7, 15):
                        np.testing.assert_allclose(
                            [p[0][index] for p in robot.previews], expected_grippers)
                    for original, action, (preview, _) in zip(
                        originals, actions, robot.previews
                    ):
                        np.testing.assert_array_equal(action, original)
                        arm_indices = [i for i in range(16) if i not in (7, 15)]
                        np.testing.assert_array_equal(preview[arm_indices], original[arm_indices])
                    self.assertEqual(len(robot.commands), len(values) if publish else 0)
                    for command, preview in zip(robot.commands, robot.previews):
                        np.testing.assert_array_equal(command[0], preview[0])

    def test_sg2_threshold_after_tt_rtc_interpolation_preserves_prefix(self) -> None:
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        chunk = np.full((32, 16), 0.2)
        chunk[:, 7] = np.linspace(0.6, 1.0, 32)
        chunk[:, 15] = np.linspace(1.0, 0.6, 32)
        processor.push_actions(chunk)
        reference = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        reference.push_actions(chunk)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._robot_type = "ffw_sg2_rev1"
        loop._action_keys = ["arm_left", "arm_right"]
        loop._action_request_mode = "tt_rtc"
        for _ in range(150):
            raw = reference.pop_action()
            loop.tick()
            expected = raw.copy()
            for i in (7, 15):
                if expected[i] >= 0.7:
                    expected[i] = 0.9
            np.testing.assert_array_equal(robot.previews[-1][0], expected)
            np.testing.assert_array_equal(processor.peek_actions(), reference.peek_actions())
        self.assertEqual(robot.commands, [])

    def test_sg2_threshold_respects_selected_action_layout(self) -> None:
        for keys, size, indices in (
            (["arm_right"], 8, [7]),
            (["head", "arm_right", "arm_left", "lift"], 19, [9, 17]),
            (["mobile", "arm_left"], 11, [10]),
        ):
            loop = ControlLoop(requester=object())
            loop._robot_type = "ffw_sg2_rev1"
            loop._action_keys = keys
            original = np.full(size, 0.8)
            expected = original.copy()
            expected[indices] = 0.9
            np.testing.assert_array_equal(
                loop._apply_gripper_command_threshold_locked(original), expected)
        loop._robot_type = "other_robot"
        np.testing.assert_array_equal(
            loop._apply_gripper_command_threshold_locked(original), original)

    def test_rlt_trace_records_the_final_command_passed_to_robot(self) -> None:
        class Trace:
            def __init__(self):
                self.events = []

            def submit(self, metadata, arrays=None, recording_id=None):
                self.events.append((metadata, arrays, recording_id))

        action = np.asarray([0.3, 0.4], dtype=np.float64)
        processor = FakeProcessor(actions=[action])
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        loop._active_action_policy_mode = "rlt"
        loop._active_rlt_robot_override = True
        loop._rlt_trace = Trace()

        loop.tick()

        metadata, arrays, _recording_id = loop._rlt_trace.events[0]
        self.assertEqual(metadata["event"], "action_commanded")
        self.assertTrue(metadata["publish_success"])
        self.assertEqual(metadata["action_policy_mode"], "rlt")
        np.testing.assert_array_equal(arrays["commanded_action"], action[None])

    def test_robot_publish_error_does_not_crash_tick(self) -> None:
        class FailingRobot(FakeRobot):
            def publish_action(self, action, action_keys) -> None:
                raise RuntimeError("publish failed")

        processor = FakeProcessor(actions=[np.asarray([0.5], dtype=np.float64)])
        robot = FailingRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True

        loop.tick()

        self.assertEqual(len(robot.previews), 0)
        self.assertFalse(loop._running)
        self.assertTrue(loop.stop())

    def test_robot_mode_publishes_idle_when_action_buffer_is_empty(self) -> None:
        processor = FakeProcessor(actions=[], buffer_size=100)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        loop._action_keys = ["mobile"]

        loop.tick()

        self.assertEqual(robot.idles, [["mobile"]])
        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(len(robot.previews), 0)

    def test_dry_run_does_not_publish_idle_when_action_buffer_is_empty(self) -> None:
        processor = FakeProcessor(actions=[], buffer_size=100)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = False
        loop._action_keys = ["mobile"]

        loop.tick()

        self.assertEqual(robot.idles, [])

    def test_mode_change_clears_buffer(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)

        loop.set_publish_to_robot(True)

        self.assertEqual(processor.clear_count, 1)

    def test_pause_clears_buffer(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)

        loop.pause()

        self.assertEqual(processor.clear_count, 1)

    def test_pause_rearms_pose_sync_and_stream_handoff(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._initial_pose_sync_completed = True

        loop.pause()

        self.assertFalse(loop._initial_pose_sync_completed)
        self.assertTrue(loop._stream_handoff_pending)

    def test_first_stream_chunk_is_prefixed_with_feedback_handoff(self) -> None:
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._stream_handoff_pending = True
        chunk = np.asarray([[1.0, 2.0], [3.0, 4.0]])

        combined, duration = loop._prepend_policy_handoff_locked(chunk)

        np.testing.assert_allclose(combined, [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        self.assertEqual(duration, 0.5)
        self.assertFalse(loop._stream_handoff_pending)

    def test_initial_pose_sync_discards_first_chunk_and_uses_effective_duration(self):
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2, 0.3, 0.4],
        )
        requester = FakeRequester(response)
        processor = FakeProcessor(buffer_size=3)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._initial_pose_sync_min_duration_s = 2.0

        started_at = __import__("time").monotonic()
        duration = loop.start()

        self.assertEqual(duration, 8.0)
        self.assertEqual(processor.clear_count, 1)
        self.assertEqual(processor.pushed_chunks, [])
        self.assertEqual(len(robot.syncs), 1)
        np.testing.assert_allclose(robot.syncs[0][0], [0.1, 0.2])
        self.assertGreaterEqual(loop._initial_pose_sync_deadline, started_at + 8.0)
        self.assertTrue(loop._initial_pose_sync_in_progress)
        self.assertTrue(loop._running)

    def test_tt_rtc_initial_pose_request_keeps_bootstrap_pending(self):
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        requester = FakeRequester(response)
        loop = ControlLoop(requester=requester)
        loop._robot = FakeRobot()
        loop._processor = FakeProcessor(buffer_size=0)
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._action_request_mode = "tt_rtc"
        loop._tt_rtc_action_dim = 2
        loop._tt_rtc_bootstrap_pending = True

        loop.start()

        request = requester.keyword_calls[0]
        self.assertEqual(request["action_request_mode"], "tt_rtc")
        self.assertEqual(request["rtc_delay_steps"], 0)
        self.assertEqual(request["rtc_prefix_action_list"], [])
        self.assertTrue(loop._tt_rtc_bootstrap_pending)

    def test_tt_rtc_sync_completion_bootstraps_from_dimensioned_empty_queue(self):
        chunk = np.zeros((32, 16), dtype=np.float64)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=32,
            action_dim=16,
            action_list=chunk.reshape(-1).tolist(),
        )
        requester = FakeRequester(response)
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm_left", "arm_right"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._action_request_mode = "tt_rtc"
        loop._tt_rtc_horizon = 32
        loop._tt_rtc_action_dim = 16
        loop._tt_rtc_min_delay_steps = 0
        loop._tt_rtc_max_delay_steps = 6
        loop._stream_handoff_pending = True

        loop.start()
        self.assertEqual(processor.peek_actions().shape, (0, 0))

        loop._initial_pose_sync_deadline = 0.0
        loop.tick()
        loop._request_thread.join(timeout=1.0)

        self.assertFalse(loop._request_thread.is_alive())
        self.assertTrue(loop._running)
        self.assertIsNone(loop.tt_rtc_failure_reason)
        self.assertEqual(len(requester.calls), 2)
        request = requester.keyword_calls[-1]
        self.assertEqual(request["rtc_delay_steps"], 0)
        self.assertEqual(request["rtc_action_dim"], 16)
        self.assertEqual(request["rtc_prefix_action_list"], [])
        self.assertGreater(processor.buffer_size, 0)

    def test_action_policy_switch_is_rejected_during_initial_pose_sync(self):
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._initial_pose_sync_in_progress = True
        loop._rlt_enabled = True

        success, message = loop.set_action_policy("rlt")

        self.assertFalse(success)
        self.assertIn("initial pose sync", message)

    def test_pause_during_sync_publishes_current_pose_hold(self):
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        loop._initial_pose_sync_in_progress = True

        self.assertTrue(loop.pause())

        self.assertEqual(robot.holds, [(["arm"], 0.1)])
        self.assertFalse(loop.initial_pose_sync_hold_required())

    def test_sync_completion_requests_a_fresh_chunk(self):
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        requester = FakeRequester(response)
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True

        loop.start()
        loop._initial_pose_sync_deadline = 0.0
        loop.tick()
        loop._request_thread.join(timeout=1.0)

        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(len(processor.pushed_chunks), 1)
        self.assertTrue(loop._initial_pose_sync_completed)
        self.assertFalse(loop._initial_pose_sync_in_progress)

    def test_deferred_sync_waits_for_second_start_without_new_policy_request(self):
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        requester = FakeRequester(response)
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._initial_pose_sync_deferred_activation = True

        self.assertEqual(loop.start(), 8.0)
        self.assertIsNone(loop._initial_pose_sync_deadline)
        loop.tick()
        self.assertEqual(len(requester.calls), 1)
        self.assertIsNone(loop._request_thread)

        activated_at = __import__("time").monotonic()
        self.assertEqual(loop.start(), 8.0)

        self.assertEqual(len(requester.calls), 1)
        self.assertGreaterEqual(
            loop._initial_pose_sync_deadline,
            activated_at + 8.0,
        )

    def test_failed_sync_hold_remains_retryable(self):
        class FailingHoldRobot(FakeRobot):
            def publish_current_pose_hold(self, action_keys, duration_s=0.1):
                raise RuntimeError("joint state unavailable")

        processor = FakeProcessor(buffer_size=0)
        robot = FailingHoldRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        loop._initial_pose_sync_in_progress = True

        self.assertFalse(loop.stop())
        self.assertTrue(loop.initial_pose_sync_hold_required())
        self.assertFalse(loop._running)

    def test_refill_threshold_includes_observed_request_latency(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._refill_margin_s = 0.25
        loop._request_latency_ema_s = 0.25

        self.assertEqual(loop._refill_threshold(processor), 50)

    def test_initial_latency_sample_is_ignored_for_warmup(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._latency_warmup_remaining = 1

        loop._record_request_latency(5.0)
        self.assertIsNone(loop._request_latency_ema_s)

        loop._record_request_latency(0.25)
        self.assertEqual(loop._request_latency_ema_s, 0.25)

    def test_refill_latency_outlier_is_ignored(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._latency_warmup_remaining = 0
        loop._max_refill_latency_s = 1.0

        loop._record_request_latency(0.2)
        loop._record_request_latency(5.0)

        self.assertEqual(loop._request_latency_ema_s, 0.2)

    def test_async_mode_requests_before_buffer_is_empty(self) -> None:
        processor = FakeProcessor(buffer_size=10)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._action_request_mode = "async"
        loop._refill_margin_s = 0.2
        loop._request_latency_ema_s = None

        self.assertTrue(loop._should_request_actions(processor))

        processor.buffer_size = 30
        self.assertFalse(loop._should_request_actions(processor))

    def test_sync_mode_waits_until_buffer_is_empty(self) -> None:
        processor = FakeProcessor(buffer_size=1)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._action_request_mode = "sync"

        self.assertFalse(loop._should_request_actions(processor))

        processor.buffer_size = 0
        self.assertTrue(loop._should_request_actions(processor))

    def test_tt_rtc_refill_threshold_tracks_latency_and_is_bounded(self) -> None:
        processor = FakeProcessor(buffer_size=6)
        loop = ControlLoop(
            requester=object(),
            tt_rtc_prefix_mode="latency_adaptive",
        )
        loop._running = True
        loop._robot = FakeRobot()
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._action_request_mode = "tt_rtc"
        cases = (
            (None, 0.0, 6),
            (0.01, 0.0, 1),
            (0.05, 0.0, 1),
            (0.05, 0.2, 4),
            (0.20, 0.2, 6),
            (0.55, 0.0, 6),
            (2.50, 0.2, 6),
        )
        for latency, margin, expected_threshold in cases:
            with self.subTest(latency=latency, margin=margin):
                loop._request_latency_ema_s = latency
                loop._refill_margin_s = margin
                self.assertEqual(
                    loop._tt_rtc_refill_threshold(),
                    expected_threshold,
                )
                processor.buffer_size = expected_threshold
                self.assertTrue(loop._should_request_actions(processor))
                processor.buffer_size = expected_threshold + 1
                self.assertFalse(loop._should_request_actions(processor))

    def test_tt_rtc_20x16_refill_threshold_stays_in_trained_delay_range(self):
        with patch(
            "main_runtime.playground_control_loop.RobotClient",
            return_value=FakeRobot(),
        ):
            loop = ControlLoop(
                requester=object(),
                tt_rtc_prefix_mode="latency_adaptive",
            )
            loop.configure(
                "f2",
                action_request_mode="tt_rtc",
                tt_rtc_horizon=20,
                tt_rtc_action_dim=16,
                tt_rtc_min_delay_steps=6,
                tt_rtc_max_delay_steps=8,
            )

        for latency, margin, expected in (
            (None, 0.0, 8),
            (0.01, 0.0, 6),
            (0.20, 0.0, 6),
            (0.40, 0.2, 8),
            (2.00, 0.0, 8),
        ):
            with self.subTest(latency=latency, margin=margin):
                loop._request_latency_ema_s = latency
                loop._refill_margin_s = margin
                self.assertEqual(loop._tt_rtc_refill_threshold(), expected)
        loop.deconfigure()

    def test_tt_rtc_fixed_first_six_is_default_and_ignores_latency(self) -> None:
        loop = ControlLoop(requester=object())

        for latency in (None, 0.01, 0.2, 2.5):
            with self.subTest(latency=latency):
                loop._request_latency_ema_s = latency
                self.assertEqual(loop._tt_rtc_refill_threshold(), 6)

    def test_tt_rtc_fixed_mode_captures_first_six_source_actions(self) -> None:
        source = np.arange(8 * 19, dtype=np.float64).reshape(8, 19)
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        processor.push_actions(source)
        loop = ControlLoop(requester=object())

        prefix = loop._capture_tt_rtc_prefix(processor)

        self.assertEqual(prefix.shape, (6, 19))
        np.testing.assert_array_equal(prefix, source[:6])
        np.testing.assert_array_equal(processor.peek_actions(), source)

    def test_tt_rtc_prefix_mode_can_restore_latency_adaptive_behavior(self) -> None:
        source = np.arange(4 * 19, dtype=np.float64).reshape(4, 19)
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        processor.push_actions(source)
        with patch.dict(
            "os.environ",
            {"CYCLO_TT_RTC_PREFIX_MODE": "latency_adaptive"},
        ):
            loop = ControlLoop(requester=object())
        loop._request_latency_ema_s = 0.01
        loop._refill_margin_s = 0.0

        self.assertEqual(loop._tt_rtc_refill_threshold(), 1)
        np.testing.assert_array_equal(
            loop._capture_tt_rtc_prefix(processor),
            source[:1],
        )

    def test_tt_rtc_temporal_overlap_replaces_same_time_future_actions(self):
        source = np.repeat(
            np.arange(20, dtype=np.float64)[:, None],
            16,
            axis=1,
        )
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        processor.push_actions(source)
        while processor.buffer_size > 12:
            processor.pop_action()

        loop = ControlLoop(
            requester=object(),
            tt_rtc_prefix_mode="fixed_first_8",
            tt_rtc_temporal_overlap_steps=4,
        )
        loop._running = True
        loop._processor = processor
        loop._tt_rtc_horizon = 20
        loop._tt_rtc_action_dim = 16
        loop._tt_rtc_min_delay_steps = 6
        loop._tt_rtc_max_delay_steps = 8

        self.assertEqual(loop._tt_rtc_refill_threshold("base"), 12)
        self.assertEqual(loop._tt_rtc_refill_threshold("rlt"), 8)
        captured = loop._capture_tt_rtc_actions(
            processor,
            action_policy_mode="base",
        )
        prefix = loop._capture_tt_rtc_prefix(processor)
        np.testing.assert_array_equal(prefix, captured[:8])

        new_postfix = np.repeat(
            np.arange(100, 112, dtype=np.float64)[:, None],
            16,
            axis=1,
        )
        response_chunk = np.concatenate((prefix, new_postfix), axis=0)
        requester = FakeRequester(SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=20,
            action_dim=16,
            action_list=response_chunk.reshape(-1).tolist(),
        ))
        loop._requester = requester

        loop._request_and_buffer(
            "drill",
            loop._generation,
            "tt_rtc",
            "base",
            rtc_prefix_actions=prefix,
            rtc_captured_actions=captured,
        )

        weights = loop._tt_rtc_overlap_blend_weights(4)
        expected_overlap = (
            (1.0 - weights[:, None]) * captured[8:12]
            + weights[:, None] * new_postfix[:4]
        )
        expected = np.concatenate((prefix, expected_overlap, new_postfix[4:]))
        np.testing.assert_allclose(processor.peek_actions(), expected)
        self.assertEqual(requester.keyword_calls[-1]["rtc_delay_steps"], 8)

    def test_tt_rtc_drained_timeline_sends_idle_while_request_is_pending(self) -> None:
        processor = TTActionTimeline()
        processor.push_actions(np.ones((6, 19)))
        for _ in range(60):
            processor.pop_action()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._action_request_mode = "tt_rtc"
        loop._publish_to_robot = True
        loop._request_thread = SimpleNamespace(is_alive=lambda: True)
        loop.tick()
        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.idles), 1)
        self.assertTrue(loop._running)

    def test_tt_rtc_requests_at_six_source_actions_or_bootstrap(self) -> None:
        processor = FakeProcessor(buffer_size=7)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._action_request_mode = "tt_rtc"

        self.assertFalse(loop._should_request_actions(processor))
        processor.buffer_size = 6
        self.assertTrue(loop._should_request_actions(processor))
        processor.buffer_size = 0
        self.assertTrue(loop._should_request_actions(processor))

    def test_sync_mode_buffers_chunk_without_scheduled_skip(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2, 0.3, 0.4],
        )
        processor = FakeProcessor(buffer_size=0)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer("pick", loop._generation, "sync")

        self.assertEqual(len(processor.pushed_chunks), 1)
        self.assertIsNone(processor.scheduled_delays[-1])
        self.assertEqual(processor.align_flags[-1], False)

    def test_tt_rtc_configures_15_hz_source_with_100_hz_output(self) -> None:
        loop = ControlLoop(
            requester=object(),
            inference_hz=30.0,
            control_hz=100.0,
            postprocess_actions=True,
        )

        with patch(
            "main_runtime.playground_control_loop.RobotClient",
            side_effect=lambda *_args, **_kwargs: FakeRobot(),
        ):
            loop.configure(
                robot_type="ffw_sg2_rev1",
                action_request_mode="tt_rtc",
            )

        self.assertEqual(loop._processor.output_hz, 100.0)
        source = np.arange(16, dtype=np.float64).reshape(16, 1)
        loop._processor.push_actions(source)
        self.assertEqual(loop._processor.buffer_size, 16)
        self.assertAlmostEqual(loop._tick_period(), 0.01)

        outputs = [loop._processor.pop_action() for _ in range(61)]
        self.assertEqual(loop._processor.buffer_size, 6)
        np.testing.assert_allclose(loop._processor.peek_actions(), source[10:])
        np.testing.assert_allclose(outputs[0], source[0])
        np.testing.assert_allclose(outputs[-1], source[9])

        remaining_outputs = [loop._processor.pop_action() for _ in range(40)]
        self.assertEqual(loop._processor.buffer_size, 0)
        np.testing.assert_allclose(remaining_outputs[-1], source[-1])
        loop.deconfigure()

    def test_tt_rtc_uses_requested_control_rate_without_changing_source_clock(self):
        for hz in (50.0, 100.0, 200.0):
            with self.subTest(control_hz=hz), patch(
                "main_runtime.playground_control_loop.RobotClient", return_value=FakeRobot()
            ):
                loop = ControlLoop(requester=object(), control_hz=100.0)
                loop.configure(robot_type="f2", action_request_mode="tt_rtc",
                               tt_rtc_horizon=32, tt_rtc_action_dim=16,
                               control_hz=hz)
                self.assertEqual(loop._processor.output_hz, hz)
                self.assertAlmostEqual(loop._tick_period(), 1.0 / hz)
                source = np.arange(32, dtype=np.float64).reshape(32, 1)
                loop._processor.push_actions(source)
                # At 0.4 seconds the same six source actions have been used,
                # independently of how many interpolated commands were sent.
                for _ in range(round(0.4 * hz) + 1):
                    loop._processor.pop_action()
                np.testing.assert_allclose(loop._processor.peek_actions(), source[7:])
                loop.deconfigure()

    def test_tt_rtc_refill_request_carries_six_action_prefix(self) -> None:
        prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        postfix = 1000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        response_chunk = np.concatenate((prefix, postfix), axis=0)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=response_chunk.reshape(-1).tolist(),
        )
        requester = FakeRequester(response)
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            prefix.copy(),
        )

        self.assertEqual(processor.buffer_size, 16)
        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((prefix, postfix), axis=0),
        )
        request_fields = requester.keyword_calls[-1]
        self.assertEqual(request_fields["action_request_mode"], "tt_rtc")
        self.assertEqual(request_fields["rtc_delay_steps"], 6)
        self.assertEqual(request_fields["rtc_action_dim"], 19)
        self.assertEqual(
            request_fields["rtc_prefix_action_list"],
            prefix.reshape(-1).tolist(),
        )

    def test_tt_rtc_32x16_bootstrap_and_refill_keep_single_prefix(self):
        source = np.arange(32 * 16, dtype=np.float64).reshape(32, 16)
        response = SimpleNamespace(success=True, message="ok", chunk_size=32,
                                   action_dim=16, action_list=source.reshape(-1).tolist())
        requester = FakeRequester(response)
        loop = ControlLoop(requester=requester)
        with patch("main_runtime.playground_control_loop.RobotClient", return_value=FakeRobot()):
            loop.configure("f2", action_keys=["arm_left", "arm_right"],
                           action_request_mode="tt_rtc", tt_rtc_horizon=32,
                           tt_rtc_action_dim=16)
        loop._running = True
        loop._request_and_buffer("drill", loop._generation, "tt_rtc", "base")
        self.assertEqual(requester.keyword_calls[-1]["rtc_action_dim"], 16)
        self.assertEqual(requester.keyword_calls[-1]["rtc_delay_steps"], 0)
        np.testing.assert_array_equal(loop._processor.peek_actions(), source)
        for _ in range(300):
            if loop._processor.buffer_size <= 6:
                break
            loop._processor.pop_action()
        prefix = loop._processor.peek_actions()
        self.assertEqual(prefix.shape, (6, 16))
        postfix = 1000 + np.arange(26 * 16, dtype=np.float64).reshape(26, 16)
        response.action_list = np.concatenate((prefix, postfix)).reshape(-1).tolist()
        loop._request_and_buffer("drill", loop._generation, "tt_rtc", "base", prefix)
        self.assertEqual(requester.keyword_calls[-1]["rtc_delay_steps"], 6)
        self.assertEqual(requester.keyword_calls[-1]["rtc_prefix_action_list"], prefix.reshape(-1).tolist())
        np.testing.assert_array_equal(loop._processor.peek_actions(), np.concatenate((prefix, postfix)))

    def test_tt_rtc_20x16_bootstrap_and_refill_use_six_action_prefix(self):
        source = np.arange(20 * 16, dtype=np.float64).reshape(20, 16)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=20,
            action_dim=16,
            action_list=source.reshape(-1).tolist(),
        )
        requester = FakeRequester(response)
        loop = ControlLoop(requester=requester)
        with patch(
            "main_runtime.playground_control_loop.RobotClient",
            return_value=FakeRobot(),
        ):
            loop.configure(
                "f2",
                action_keys=["arm_left", "arm_right"],
                action_request_mode="tt_rtc",
                tt_rtc_horizon=20,
                tt_rtc_action_dim=16,
                tt_rtc_min_delay_steps=6,
                tt_rtc_max_delay_steps=8,
            )
        loop._running = True

        loop._request_and_buffer("drill", loop._generation, "tt_rtc", "base")
        self.assertEqual(requester.keyword_calls[-1]["rtc_delay_steps"], 0)
        for _ in range(300):
            if loop._processor.buffer_size <= 6:
                break
            loop._processor.pop_action()
        prefix = loop._processor.peek_actions()
        self.assertEqual(prefix.shape, (6, 16))

        postfix = 1000 + np.arange(14 * 16, dtype=np.float64).reshape(14, 16)
        response.action_list = np.concatenate((prefix, postfix)).reshape(-1).tolist()
        loop._request_and_buffer(
            "drill",
            loop._generation,
            "tt_rtc",
            "base",
            prefix,
        )

        self.assertEqual(requester.keyword_calls[-1]["rtc_delay_steps"], 6)
        np.testing.assert_array_equal(
            loop._processor.peek_actions(),
            np.concatenate((prefix, postfix)),
        )
        loop.deconfigure()

    def test_tt_rtc_32x16_rlt_appends_ten_without_dropping_prefix_twice(self):
        for delay in (0, 3, 6):
            with self.subTest(delay=delay), patch(
                "main_runtime.playground_control_loop.RobotClient", return_value=FakeRobot()
            ), patch("main_runtime.playground_control_loop.create_rlt_trace_publisher"):
                chunk = np.arange(160, dtype=np.float64).reshape(10, 16)
                reference = 5000.0 + chunk
                response = SimpleNamespace(success=True, message="ok", chunk_size=10,
                    action_dim=16, action_list=chunk.reshape(-1).tolist(),
                    rtc_reference_chunk_size=10,
                    rtc_reference_action_list=reference.reshape(-1).tolist())
                requester = FakeRequester(response)
                loop = ControlLoop(
                    requester=requester,
                    control_hz=200,
                    tt_rtc_prefix_mode="latency_adaptive",
                )
                loop.configure("ffw_sg2_rev1", action_keys=["arm_left", "arm_right"],
                    action_request_mode="tt_rtc", rlt_enabled=True,
                    tt_rtc_horizon=32, tt_rtc_action_dim=16)
                loop._running = True
                prefix = np.full((delay, 16), -1.0)
                if delay:
                    loop._processor.push_actions(prefix)
                loop._request_and_buffer("drill", loop._generation, "tt_rtc", "rlt", prefix)
                self.assertEqual(requester.keyword_calls[-1]["rtc_delay_steps"], delay)
                self.assertEqual(requester.keyword_calls[-1]["rtc_action_dim"], 16)
                expected_chunk = loop._smooth_rlt_chunk_boundary(
                    chunk,
                    prefix[-1] if delay else None,
                    source_steps=2,
                )
                np.testing.assert_array_equal(loop._processor.peek_actions(),
                                              np.concatenate((prefix, expected_chunk)))
                loop.deconfigure()

    def test_tt_rtc_rlt_to_vla_request_uses_remaining_mlp_actions(self):
        for control_hz in (100, 200):
            with self.subTest(control_hz=control_hz), patch(
                "main_runtime.playground_control_loop.RobotClient", return_value=FakeRobot()
            ), patch("main_runtime.playground_control_loop.create_rlt_trace_publisher"):
                mlp = 1000.0 + np.arange(160).reshape(10, 16)
                reference = 5000.0 + np.arange(160).reshape(10, 16)
                response = SimpleNamespace(success=True, message="ok", chunk_size=10,
                    action_dim=16, action_list=mlp.reshape(-1).tolist(),
                    rtc_reference_chunk_size=10,
                    rtc_reference_action_list=reference.reshape(-1).tolist())

                class RouteRequester(FakeRequester):
                    def get_action(
                        self,
                        task_instruction,
                        action_policy_mode="base",
                        action_request_mode="async",
                        **kwargs,
                    ):
                        if action_policy_mode == "base":
                            delay = int(kwargs["rtc_delay_steps"])
                            prefix = np.asarray(
                                kwargs["rtc_prefix_action_list"]
                            ).reshape(delay, 16)
                            postfix = np.full((32 - delay, 16), -123.0)
                            self.response = SimpleNamespace(
                                success=True,
                                message="ok",
                                chunk_size=32,
                                action_dim=16,
                                action_list=np.concatenate(
                                    (prefix, postfix)
                                ).reshape(-1).tolist(),
                            )
                        return super().get_action(
                            task_instruction,
                            action_policy_mode=action_policy_mode,
                            action_request_mode=action_request_mode,
                            **kwargs,
                        )

                requester = RouteRequester(response)
                loop = ControlLoop(requester=requester, control_hz=control_hz)
                loop.configure("ffw_sg2_rev1", action_keys=["arm_left", "arm_right"],
                    action_request_mode="tt_rtc", rlt_enabled=True,
                    tt_rtc_horizon=32, tt_rtc_action_dim=16)
                loop._running = True
                try:
                    loop._request_and_buffer("drill", loop._generation, "tt_rtc", "rlt")
                    loop._request_latency_ema_s = 0.01
                    for _ in range(control_hz):
                        if loop._should_request_actions(loop._processor):
                            break
                        loop._processor.pop_action()
                    # The next VLA request conditions on the MLP commands
                    # still committed to execution, never the raw reference.
                    loop._active_action_policy_mode = "rlt"
                    loop._pending_action_policy_mode = "base"
                    expected_prefix = loop._processor.peek_actions().copy()
                    loop.tick()
                    self.assertIsNotNone(loop._request_thread)
                    loop._request_thread.join(timeout=2.0)
                    self.assertFalse(loop._request_thread.is_alive())
                    self.assertEqual(requester.calls[-1][1], "base")
                    sent = requester.keyword_calls[-1]
                    delay = sent["rtc_delay_steps"]
                    self.assertEqual(delay, 6)
                    np.testing.assert_array_equal(
                        np.asarray(sent["rtc_prefix_action_list"]).reshape(delay, 16),
                        expected_prefix,
                    )
                    self.assertEqual(len(loop._robot.handoff_requests), 0)
                    self.assertGreater(loop._processor.buffer_size, 0)
                    self.assertEqual(loop._active_action_policy_mode, "base")
                finally:
                    loop.deconfigure()

    def test_tt_rtc_dual_rate_refill_appends_only_new_postfix(self) -> None:
        source = np.repeat(
            np.arange(16, dtype=np.float64).reshape(16, 1),
            19,
            axis=1,
        )
        postfix = np.repeat(
            np.arange(16, 26, dtype=np.float64).reshape(10, 1),
            19,
            axis=1,
        )
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        processor.push_actions(source)
        for _ in range(61):
            processor.pop_action()
        prefix = processor.peek_actions()
        self.assertEqual(prefix.shape, (6, 19))

        response_chunk = np.concatenate((prefix, postfix), axis=0)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=response_chunk.reshape(-1).tolist(),
        )
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            prefix.copy(),
        )

        self.assertEqual(processor.buffer_size, 16)
        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((prefix, postfix), axis=0),
        )

    def test_tt_rtc_rlt_appends_complete_ten_action_postfix(self) -> None:
        prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        rlt_chunk = 2000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        reference = 3000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=10,
            action_dim=19,
            action_list=rlt_chunk.reshape(-1).tolist(),
            rtc_reference_chunk_size=10,
            rtc_reference_action_list=reference.reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "rlt",
            prefix.copy(),
        )

        expected_postfix = loop._smooth_rlt_chunk_boundary(
            rlt_chunk,
            prefix[-1],
            source_steps=2,
        )
        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((prefix, expected_postfix), axis=0),
        )

    def test_tt_rtc_switches_vla_and_mlp_without_draining_prefix(self) -> None:
        for active_mode, target_mode in (("base", "rlt"), ("rlt", "base")):
            with self.subTest(active=active_mode, target=target_mode):
                source = np.arange(7 * 19, dtype=np.float64).reshape(7, 19)
                source[:, 15] = 0.5
                prefix = source[1:]
                postfix = (
                    4000.0
                    + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
                )
                reference = (
                    8000.0
                    + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
                )
                response_chunk = (
                    np.concatenate((prefix, postfix), axis=0)
                    if target_mode == "base"
                    else postfix
                )
                response = SimpleNamespace(
                    success=True,
                    message="ok",
                    chunk_size=len(response_chunk),
                    action_dim=19,
                    action_list=response_chunk.reshape(-1).tolist(),
                    rtc_reference_chunk_size=(
                        10 if target_mode == "rlt" else 0
                    ),
                    rtc_reference_action_list=(
                        reference.reshape(-1).tolist()
                        if target_mode == "rlt"
                        else []
                    ),
                )
                requester = FakeRequester(response)
                processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
                processor.push_actions(source)
                robot = FakeRobot()
                loop = self._make_loop(processor, robot)
                loop._requester = requester
                loop._action_request_mode = "tt_rtc"
                loop._rlt_enabled = True
                loop._active_action_policy_mode = active_mode
                result = []

                switch_thread = threading.Thread(
                    target=lambda: result.append(
                        loop.set_action_policy(target_mode, timeout_s=1.0)
                    )
                )
                switch_thread.start()
                for _ in range(100):
                    if loop._pending_action_policy_mode == target_mode:
                        break
                    threading.Event().wait(0.001)

                loop.tick()
                self.assertIsNotNone(loop._request_thread)
                loop._request_thread.join(timeout=1.0)
                switch_thread.join(timeout=1.0)

                self.assertFalse(switch_thread.is_alive())
                self.assertEqual(result, [
                    (True, f"{target_mode.upper()} action active")
                ])
                self.assertEqual(loop._active_action_policy_mode, target_mode)
                self.assertIsNone(loop._pending_action_policy_mode)
                self.assertEqual(requester.calls[-1], ("", target_mode))
                if target_mode == "rlt":
                    expected_postfix = loop._smooth_rlt_chunk_boundary(
                        postfix,
                        prefix[-1],
                        source_steps=2,
                    )
                    np.testing.assert_allclose(
                        processor.peek_actions(),
                        np.concatenate((prefix, expected_postfix), axis=0),
                    )
                    self.assertEqual(len(robot.previews), 1)
                else:
                    sent = requester.keyword_calls[-1]
                    self.assertEqual(sent["rtc_delay_steps"], 6)
                    np.testing.assert_array_equal(
                        np.asarray(sent["rtc_prefix_action_list"]).reshape(6, 19),
                        prefix,
                    )
                    self.assertEqual(len(robot.handoff_requests), 0)
                    np.testing.assert_array_equal(
                        processor.peek_actions(), response_chunk,
                    )
                self.assertEqual(robot.idles, [])

    def test_tt_rtc_mlp_to_vla_preserves_model_gripper_values(
        self,
    ) -> None:
        prefix = np.zeros((6, 16), dtype=np.float64)
        prefix[:, 15] = 0.7
        first_postfix = np.ones((10, 16), dtype=np.float64) * 0.2
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        processor.push_actions(prefix)
        loop = ControlLoop(requester=object())
        loop._running = True
        loop._processor = processor
        loop._tt_rtc_horizon = 16
        loop._tt_rtc_action_dim = 16
        loop._active_action_policy_mode = "rlt"
        loop._pending_action_policy_mode = "base"

        committed, reason = loop._commit_tt_rtc_chunk_locked(
            chunk=np.concatenate((prefix, first_postfix), axis=0),
            captured_prefix=prefix,
            captured_actions=prefix,
            delay_steps=len(prefix),
            prefix_captured_at=time.monotonic(),
            action_policy_mode="base",
        )

        self.assertTrue(committed, reason)
        np.testing.assert_allclose(processor.peek_actions()[:6, 15], 0.7)
        np.testing.assert_allclose(processor.peek_actions()[6:, 15], 0.2)

        while processor.peek_actions().shape[0]:
            processor.pop_action()
        next_postfix = np.ones((16, 16), dtype=np.float64) * 0.1
        committed, reason = loop._commit_tt_rtc_chunk_locked(
            chunk=next_postfix,
            captured_prefix=np.empty((0, 16), dtype=np.float64),
            captured_actions=np.empty((0, 16), dtype=np.float64),
            delay_steps=0,
            prefix_captured_at=time.monotonic(),
            action_policy_mode="base",
        )

        self.assertTrue(committed, reason)
        np.testing.assert_allclose(processor.peek_actions()[:, 15], 0.1)

    def test_tt_rtc_mlp_to_vla_preserves_open_gripper_values(self) -> None:
        prefix = np.zeros((6, 16), dtype=np.float64)
        prefix[:, 15] = 0.69
        postfix = np.ones((10, 16), dtype=np.float64) * 0.2
        processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        processor.push_actions(prefix)
        loop = ControlLoop(requester=object())
        loop._running = True
        loop._processor = processor
        loop._tt_rtc_horizon = 16
        loop._tt_rtc_action_dim = 16
        loop._active_action_policy_mode = "rlt"
        loop._pending_action_policy_mode = "base"

        committed, reason = loop._commit_tt_rtc_chunk_locked(
            chunk=np.concatenate((prefix, postfix), axis=0),
            captured_prefix=prefix,
            captured_actions=prefix,
            delay_steps=len(prefix),
            prefix_captured_at=time.monotonic(),
            action_policy_mode="base",
        )

        self.assertTrue(committed, reason)
        np.testing.assert_allclose(processor.peek_actions()[6:, 15], 0.2)

    def test_tt_rtc_repeated_dual_rate_refills_preserve_each_action(self) -> None:
        # Exercise both wire formats, startup, and route changes against the
        # actual interpolating timeline. Model/transport latency uses a fake
        # clock; no robot, GPU, or wall-clock sleeps are involved.
        routes = (("base",) * 4, ("rlt",) * 4, ("base", "rlt", "base", "rlt"))
        for modes in routes:
            for latency_ticks in (0, 20, 35, 55, 120):
                with self.subTest(modes=modes, latency_ticks=latency_ticks):
                    processor = TTActionTimeline(source_hz=15.0, control_hz=100.0)
                    loop = self._make_loop(processor, FakeRobot())
                    loop._action_request_mode = "tt_rtc"
                    loop._rlt_enabled = True
                    now = [100.0]
                    outputs = []
                    next_waypoint = 0

                    def control_tick():
                        outputs.append(processor.pop_action())
                        now[0] += 0.01

                    for request_index, mode in enumerate(modes):
                        if request_index:
                            for _ in range(120):
                                if loop._should_request_actions(processor):
                                    break
                                control_tick()
                        prefix = processor.peek_actions()
                        delay = len(prefix)
                        if request_index == 0:
                            self.assertEqual(delay, 0)
                        else:
                            self.assertGreaterEqual(delay, 1)
                            self.assertLessEqual(
                                delay,
                                loop._tt_rtc_refill_threshold(),
                            )
                        fresh_count = 16 - delay if mode == "base" else 10
                        fresh = np.repeat(
                            np.arange(
                                next_waypoint, next_waypoint + fresh_count,
                                dtype=np.float64,
                            )[:, None], 19, axis=1,
                        )
                        next_waypoint += fresh_count
                        chunk = (
                            np.concatenate((prefix, fresh))
                            if delay and mode == "base" else fresh
                        )
                        response = SimpleNamespace(
                            success=True, message="ok", chunk_size=len(chunk),
                            action_dim=19, action_list=chunk.reshape(-1).tolist(),
                            rtc_reference_chunk_size=(10 if mode == "rlt" else 0),
                            rtc_reference_action_list=(
                                fresh.reshape(-1).tolist()
                                if mode == "rlt"
                                else []
                            ),
                        )

                        class DelayedRequester(FakeRequester):
                            def get_action(self, *args, **kwargs):
                                if delay:
                                    for _ in range(latency_ticks):
                                        control_tick()
                                return super().get_action(*args, **kwargs)

                        loop._requester = DelayedRequester(response)
                        loop._pending_action_policy_mode = mode
                        with patch(
                            "main_runtime.playground_control_loop.time.monotonic",
                            side_effect=lambda: now[0],
                        ):
                            loop._request_and_buffer(
                                "pick", loop._generation, "tt_rtc", mode, prefix,
                            )
                        self.assertIsNone(loop.tt_rtc_failure_reason)
                        self.assertEqual(loop._active_action_policy_mode, mode)
                        self.assertEqual(
                            loop._requester.keyword_calls[-1]["rtc_delay_steps"], delay,
                        )
                        np.testing.assert_array_equal(
                            processor.peek_actions()[-fresh_count:], fresh,
                        )

                    # A ramp makes duplicates, skipped waypoints, and phase
                    # resets visible across every VLA/MLP chunk boundary.
                    for _ in range(60):
                        control_tick()
                    if latency_ticks <= 35:
                        np.testing.assert_allclose(
                            np.asarray(outputs)[:, 0], np.arange(len(outputs)) * 0.15,
                            rtol=0.0, atol=1e-12,
                        )
                    else:
                        self.assertTrue(any(action is None for action in outputs))
                        values = np.asarray([a for a in outputs if a is not None])[:, 0]
                        self.assertTrue(np.all(np.diff(values) >= -1e-12))
                        self.assertTrue(np.all(np.diff(values) <= 0.15 + 1e-12))

    def test_tt_rtc_switch_during_inflight_request_uses_old_postfix_as_bridge(
        self,
    ) -> None:
        initial_prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        old_postfix = (
            6000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        )
        target_postfix = (
            7000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        )
        old_response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=np.concatenate(
                (initial_prefix, old_postfix), axis=0
            ).reshape(-1).tolist(),
        )
        target_response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=10,
            action_dim=19,
            action_list=target_postfix.reshape(-1).tolist(),
            rtc_reference_chunk_size=10,
            rtc_reference_action_list=(target_postfix + 1000.0).reshape(-1).tolist(),
        )
        request_started = threading.Event()
        release_old_response = threading.Event()

        class BlockingRequester:
            def get_action(self, _task_instruction, action_policy_mode, **_kwargs):
                if action_policy_mode == "base":
                    request_started.set()
                    release_old_response.wait(timeout=1.0)
                    return old_response
                return target_response

        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(initial_prefix)
        loop = ControlLoop(requester=BlockingRequester())
        loop._running = True
        loop._processor = processor
        loop._action_request_mode = "tt_rtc"
        loop._rlt_enabled = True
        generation = loop._generation
        old_thread = threading.Thread(
            target=lambda: loop._request_and_buffer(
                "pick",
                generation,
                "tt_rtc",
                "base",
                initial_prefix.copy(),
            )
        )
        loop._request_thread = old_thread
        old_thread.start()
        self.assertTrue(request_started.wait(timeout=1.0))
        switch_result = []
        switch_thread = threading.Thread(
            target=lambda: switch_result.append(
                loop.set_action_policy("rlt", timeout_s=1.0)
            )
        )
        switch_thread.start()
        for _ in range(100):
            if loop._pending_action_policy_mode == "rlt":
                break
            threading.Event().wait(0.001)

        self.assertEqual(loop._generation, generation)
        release_old_response.set()
        old_thread.join(timeout=1.0)

        self.assertEqual(loop._active_action_policy_mode, "base")
        self.assertEqual(loop._pending_action_policy_mode, "rlt")
        self.assertEqual(processor.buffer_size, 16)
        for _ in range(10):
            self.assertIsNotNone(processor.pop_action())
        target_prefix = processor.peek_actions()
        self.assertEqual(target_prefix.shape, (6, 19))

        loop._request_and_buffer(
            "pick",
            generation,
            "tt_rtc",
            "rlt",
            target_prefix,
        )
        switch_thread.join(timeout=1.0)

        self.assertEqual(switch_result, [(True, "RLT action active")])
        self.assertEqual(loop._active_action_policy_mode, "rlt")
        self.assertEqual(processor.buffer_size, 16)
        expected_postfix = loop._smooth_rlt_chunk_boundary(
            target_postfix,
            target_prefix[-1],
            source_steps=2,
        )
        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((target_prefix, expected_postfix), axis=0),
        )

    def test_tt_rtc_rejected_mlp_postfix_latches_safe_pause(
        self,
    ) -> None:
        prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        invalid_chunk = np.zeros((9, 19), dtype=np.float64)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=9,
            action_dim=19,
            action_list=invalid_chunk.reshape(-1).tolist(),
            rtc_reference_chunk_size=10,
            rtc_reference_action_list=np.zeros((10, 19)).reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor
        loop._rlt_enabled = True
        loop._active_action_policy_mode = "rlt"
        generation = loop._generation

        loop._request_and_buffer(
            "pick",
            generation,
            "tt_rtc",
            "rlt",
            prefix.copy(),
        )

        self.assertEqual(loop._active_action_policy_mode, "base")
        self.assertIsNone(loop._pending_action_policy_mode)
        self.assertEqual(loop._generation, generation + 1)
        self.assertFalse(loop._running)
        self.assertEqual(processor.buffer_size, 0)
        self.assertIn("MLP response rejected", loop.tt_rtc_failure_reason)

    def test_tt_rtc_failed_mlp_handoff_reports_failure_and_latches_pause(
        self,
    ) -> None:
        source = np.arange(7 * 19, dtype=np.float64).reshape(7, 19)
        invalid_chunk = np.zeros((9, 19), dtype=np.float64)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=9,
            action_dim=19,
            action_list=invalid_chunk.reshape(-1).tolist(),
            rtc_reference_chunk_size=10,
            rtc_reference_action_list=np.zeros((10, 19)).reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(source)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._requester = FakeRequester(response)
        loop._action_request_mode = "tt_rtc"
        loop._rlt_enabled = True
        result = []
        switch_thread = threading.Thread(
            target=lambda: result.append(
                loop.set_action_policy("rlt", timeout_s=1.0)
            )
        )
        switch_thread.start()
        for _ in range(100):
            if loop._pending_action_policy_mode == "rlt":
                break
            threading.Event().wait(0.001)

        loop.tick()
        loop._request_thread.join(timeout=1.0)
        switch_thread.join(timeout=1.0)

        self.assertFalse(switch_thread.is_alive())
        self.assertEqual(loop._active_action_policy_mode, "base")
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0][0])
        self.assertIn("MLP response rejected", result[0][1])
        self.assertFalse(loop._running)
        self.assertEqual(processor.buffer_size, 0)
        self.assertEqual(len(robot.previews), 1)
        self.assertEqual(robot.idles, [])

    def test_tt_rtc_accepts_consumed_suffix_of_captured_prefix(self) -> None:
        prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        postfix = 3000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        response_chunk = np.concatenate((prefix, postfix), axis=0)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=response_chunk.reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)

        class ConsumingRequester(FakeRequester):
            def get_action(self, *args, **kwargs):
                processor.pop_action()
                processor.pop_action()
                return super().get_action(*args, **kwargs)

        loop = ControlLoop(requester=ConsumingRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            prefix.copy(),
        )

        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((prefix[2:], postfix), axis=0),
        )

    def test_tt_rtc_accepts_response_after_prefix_window(self) -> None:
        prefix = np.zeros((6, 19), dtype=np.float64)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=np.zeros((16, 19), dtype=np.float64).reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        with patch(
            "main_runtime.playground_control_loop.time.monotonic",
            side_effect=[100.300, 100.310, 100.310, 100.410],
        ):
            loop._request_and_buffer(
                "pick",
                loop._generation,
                "tt_rtc",
                "base",
                prefix.copy(),
                100.0,
            )

        self.assertEqual(processor.buffer_size, 16)
        self.assertTrue(loop._running)
        self.assertIsNone(loop.tt_rtc_failure_reason)

    def test_tt_rtc_passes_remaining_normal_timeout_to_requester(self) -> None:
        prefix = np.zeros((6, 19), dtype=np.float64)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=np.zeros((16, 19), dtype=np.float64).reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)
        requester = FakeRequester(response)
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor

        with patch("main_runtime.playground_control_loop.time.monotonic", return_value=10.1):
            loop._request_and_buffer(
                "pick",
                loop._generation,
                "tt_rtc",
                "base",
                prefix.copy(),
                10.0,
            )

        self.assertAlmostEqual(requester.keyword_calls[-1]["timeout_s"], 4.9)
        self.assertTrue(loop._running)
        self.assertEqual(processor.buffer_size, 16)

    def test_tt_rtc_bootstrap_uses_normal_timeout_and_requires_explicit_restart(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=np.zeros((16, 19), dtype=np.float64).reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        requester = FakeRequester(response)
        requester.get_action_timeout_s = 3.0
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor

        clock = [20.0]
        get_action = requester.get_action

        def cold_get_action(*args, **kwargs):
            clock[0] += 2.204
            return get_action(*args, **kwargs)

        with patch("main_runtime.playground_control_loop.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(requester, "get_action", side_effect=cold_get_action):
            loop._request_and_buffer(
                "pick",
                loop._generation,
                "tt_rtc",
                "base",
                np.empty((0, 19), dtype=np.float64),
                20.0,
            )

        self.assertAlmostEqual(requester.keyword_calls[-1]["timeout_s"], 3.0)
        self.assertEqual(processor.buffer_size, 16)
        self.assertTrue(loop._running)
        self.assertFalse(loop._tt_rtc_bootstrap_pending)

        # A drained running queue must not acquire a fresh bootstrap budget,
        # even if START is repeated while it is already running.
        processor.clear()
        loop.start()
        loop._request_and_buffer(
            "pick", loop._generation, "tt_rtc", "base", np.empty((0, 19)),
        )
        self.assertEqual(len(requester.calls), 1)
        self.assertFalse(loop._running)
        self.assertIn("buffer exhausted", loop.tt_rtc_failure_reason)

        # Explicit START after the pause permits a new bounded bootstrap.
        loop.start()
        loop._request_and_buffer(
            "pick", loop._generation, "tt_rtc", "base", np.empty((0, 19)),
        )
        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(processor.buffer_size, 16)
        self.assertTrue(loop._running)
        self.assertIsNone(loop.tt_rtc_failure_reason)

    def test_tt_rtc_bootstrap_rejects_response_after_normal_timeout(self) -> None:
        requester = FakeRequester(SimpleNamespace(
            success=True, message="ok", chunk_size=16, action_dim=19,
            action_list=np.zeros((16, 19)).reshape(-1).tolist(),
        ))
        processor = ActionChunkProcessor(
            inference_hz=15.0, control_hz=100.0, postprocess=False,
        )
        loop = ControlLoop(requester=requester)
        loop._processor = processor
        loop.start()
        clock = [20.0]
        get_action = requester.get_action

        def late_get_action(*args, **kwargs):
            clock[0] += 5.001
            return get_action(*args, **kwargs)

        with patch("main_runtime.playground_control_loop.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(requester, "get_action", side_effect=late_get_action):
            loop._request_and_buffer(
                "pick", loop._generation, "tt_rtc", "base", np.empty((0, 19)), 20.0,
            )

        self.assertAlmostEqual(requester.keyword_calls[-1]["timeout_s"], 5.0)
        self.assertEqual(processor.buffer_size, 0)
        self.assertFalse(loop._running)
        self.assertIn("5.001s > 5.000s", loop.tt_rtc_failure_reason)

    def test_tt_rtc_expired_budget_latches_before_rpc(self) -> None:
        requester = FakeRequester(SimpleNamespace(success=True))
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        prefix = np.zeros((6, 19), dtype=np.float64)
        processor.push_actions(prefix)
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor

        with patch("main_runtime.playground_control_loop.time.monotonic", return_value=35.1):
            loop._request_and_buffer(
                "pick",
                loop._generation,
                "tt_rtc",
                "base",
                prefix,
                30.0,
            )

        self.assertEqual(requester.calls, [])
        self.assertFalse(loop._running)
        self.assertEqual(processor.buffer_size, 0)
        self.assertIn("expired before", loop.tt_rtc_failure_reason)

    def test_tt_rtc_request_failure_latches_pause_and_idles_robot(self) -> None:
        response = SimpleNamespace(
            success=False,
            message="deadline timeout",
            chunk_size=0,
            action_dim=0,
            action_list=[],
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(np.zeros((6, 19), dtype=np.float64))
        requester = FakeRequester(response)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor
        loop._robot = robot
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        generation = loop._generation

        loop._request_and_buffer(
            "pick",
            generation,
            "tt_rtc",
            "base",
            processor.peek_actions(),
        )

        self.assertFalse(loop._running)
        self.assertEqual(loop._generation, generation + 1)
        self.assertEqual(processor.buffer_size, 0)
        self.assertEqual(robot.idles, [["arm"]])
        self.assertIn("deadline timeout", loop.tt_rtc_failure_reason)
        loop.tick()
        self.assertEqual(len(requester.calls), 1)

        self.assertTrue(loop.stop())
        loop.start(publish_to_robot=True)
        self.assertTrue(loop._running)
        self.assertIsNone(loop.tt_rtc_failure_reason)

    def test_tt_rtc_invalid_request_prefix_latches_pause_before_rpc(self) -> None:
        response = SimpleNamespace(success=True)
        requester = FakeRequester(response)
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            np.zeros((7, 19), dtype=np.float64),
        )

        self.assertFalse(loop._running)
        self.assertEqual(requester.calls, [])
        self.assertIn("prefix invalid", loop.tt_rtc_failure_reason)

    def test_tt_rtc_fixed_mode_rejects_partial_six_action_prefix(self) -> None:
        requester = FakeRequester(SimpleNamespace(success=True))
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        prefix = np.zeros((5, 19), dtype=np.float64)
        processor.push_actions(prefix)
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            prefix,
        )

        self.assertFalse(loop._running)
        self.assertEqual(requester.calls, [])
        self.assertIn("exactly six", loop.tt_rtc_failure_reason)

    def test_tt_rtc_discards_when_queued_prefix_is_not_captured_suffix(
        self,
    ) -> None:
        prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        postfix = 5000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        response_chunk = np.concatenate((prefix, postfix), axis=0)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=response_chunk.reshape(-1).tolist(),
        )
        replacement = np.full((5, 19), -1.0, dtype=np.float64)
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)

        class ReplacingRequester(FakeRequester):
            def get_action(self, *args, **kwargs):
                processor.clear()
                processor.push_actions(replacement)
                return super().get_action(*args, **kwargs)

        loop = ControlLoop(requester=ReplacingRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            prefix.copy(),
        )

        self.assertEqual(processor.buffer_size, 0)
        self.assertFalse(loop._running)
        self.assertIn("VLA response rejected", loop.tt_rtc_failure_reason)

    def test_tt_rtc_discards_base_response_with_changed_prefix(self) -> None:
        prefix = np.zeros((6, 19), dtype=np.float64)
        mismatched_chunk = np.zeros((16, 19), dtype=np.float64)
        mismatched_chunk[0, 0] = 1.0
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=16,
            action_dim=19,
            action_list=mismatched_chunk.reshape(-1).tolist(),
        )
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            postprocess=False,
        )
        processor.push_actions(prefix)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer(
            "pick",
            loop._generation,
            "tt_rtc",
            "base",
            prefix.copy(),
        )

        self.assertEqual(processor.buffer_size, 0)
        self.assertFalse(loop._running)
        self.assertIn("VLA response rejected", loop.tt_rtc_failure_reason)

    def test_async_mode_buffers_chunk_with_latency_and_buffer_delay(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2, 0.3, 0.4],
        )
        processor = FakeProcessor(buffer_size=50)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer("pick", loop._generation, "async")

        self.assertEqual(len(processor.pushed_chunks), 1)
        self.assertIsNotNone(processor.scheduled_delays[-1])
        self.assertGreaterEqual(processor.scheduled_delays[-1], 0.5)
        self.assertEqual(processor.align_flags[-1], True)

    def test_rlt_switch_commits_only_after_buffer_boundary(self) -> None:
        processor = FakeProcessor(buffer_size=1)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._rlt_enabled = True
        result = []

        thread = threading.Thread(
            target=lambda: result.append(loop.set_action_policy("rlt", timeout_s=1.0))
        )
        thread.start()
        for _ in range(100):
            if loop._pending_action_policy_mode == "rlt":
                break
            threading.Event().wait(0.001)

        self.assertEqual(loop._active_action_policy_mode, "base")
        self.assertFalse(loop._should_request_actions(processor))
        processor.buffer_size = 0
        with loop._lock:
            loop._commit_pending_action_policy_locked(processor)
        thread.join(timeout=1.0)

        self.assertEqual(result, [(True, "RLT action active")])
        self.assertEqual(loop._active_action_policy_mode, "rlt")

    def test_rlt_switch_is_rejected_for_robot_publish(self) -> None:
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._rlt_enabled = True
        loop._publish_to_robot = True

        success, message = loop.set_action_policy("rlt", timeout_s=0.0)

        self.assertFalse(success)
        self.assertIn("rlt_robot_override", message)

    def test_rlt_switch_is_allowed_for_robot_with_explicit_override(self) -> None:
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._rlt_enabled = True
        loop._publish_to_robot = True
        result = []

        thread = threading.Thread(
            target=lambda: result.append(loop.set_action_policy(
                "rlt",
                allow_robot_rlt=True,
                timeout_s=1.0,
            ))
        )
        thread.start()
        for _ in range(100):
            if loop._pending_action_policy_mode == "rlt":
                break
            threading.Event().wait(0.001)
        with loop._lock:
            loop._commit_pending_action_policy_locked(processor)
        thread.join(timeout=1.0)

        self.assertEqual(result, [(True, "RLT action active")])
        self.assertEqual(loop._active_action_policy_mode, "rlt")
        self.assertTrue(loop._active_rlt_robot_override)

    def test_robot_can_confirm_rlt_then_switch_base_and_retry_without_reload(
        self,
    ) -> None:
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._rlt_enabled = True
        loop._publish_to_robot = True
        loaded_robot = loop._robot
        loaded_processor = loop._processor

        denied, _message = loop.set_action_policy("rlt", timeout_s=0.0)
        self.assertFalse(denied)
        self.assertEqual(loop._active_action_policy_mode, "base")

        def switch(target: str, *, allow_robot_rlt: bool = False):
            result = []
            thread = threading.Thread(
                target=lambda: result.append(loop.set_action_policy(
                    target,
                    allow_robot_rlt=allow_robot_rlt,
                    timeout_s=1.0,
                ))
            )
            thread.start()
            for _ in range(100):
                if loop._pending_action_policy_mode == target:
                    break
                threading.Event().wait(0.001)
            with loop._lock:
                loop._commit_pending_action_policy_locked(processor)
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            return result[0]

        self.assertEqual(
            switch("rlt", allow_robot_rlt=True),
            (True, "RLT action active"),
        )
        self.assertTrue(loop._active_rlt_robot_override)

        self.assertEqual(switch("base"), (True, "BASE action active"))
        self.assertFalse(loop._active_rlt_robot_override)

        self.assertEqual(
            switch("rlt", allow_robot_rlt=True),
            (True, "RLT action active"),
        )
        self.assertTrue(loop._active_rlt_robot_override)
        self.assertTrue(loop._rlt_enabled)
        self.assertIs(loop._robot, loaded_robot)
        self.assertIs(loop._processor, loaded_processor)

    def test_enabling_robot_publish_drops_unapproved_sim_rlt(self) -> None:
        processor = FakeProcessor(buffer_size=10)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._rlt_enabled = True
        loop._active_action_policy_mode = "rlt"

        loop.set_publish_to_robot(True)

        self.assertEqual(loop._active_action_policy_mode, "base")
        self.assertIsNone(loop._pending_action_policy_mode)
        self.assertEqual(processor.clear_count, 1)

    def test_tick_never_publishes_unapproved_rlt_action_to_robot(self) -> None:
        processor = FakeProcessor(
            actions=[np.asarray([0.1, 0.2], dtype=np.float64)],
            buffer_size=1,
        )
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._rlt_enabled = True
        loop._active_action_policy_mode = "rlt"
        loop._publish_to_robot = True
        loop._request_thread = threading.current_thread()

        loop.tick()

        self.assertEqual(loop._active_action_policy_mode, "base")
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.previews, [])
        self.assertEqual(robot.idles, [["arm"]])

    def test_non_finite_base_chunk_is_not_buffered(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, float("nan")],
        )
        processor = FakeProcessor(buffer_size=0)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor

        loop._request_and_buffer("pick", loop._generation, "sync", "base")

        self.assertEqual(processor.pushed_chunks, [])

    def test_invalid_rlt_chunk_falls_back_to_base(self) -> None:
        responses = {
            "failed": SimpleNamespace(success=False, message="engine failed"),
            "empty": SimpleNamespace(
                success=True,
                message="ok",
                chunk_size=0,
                action_dim=19,
                action_list=[],
            ),
            "shape": SimpleNamespace(
                success=True,
                message="ok",
                chunk_size=2,
                action_dim=2,
                action_list=[0.1, 0.2, 0.3],
            ),
            "nan": SimpleNamespace(
                success=True,
                message="ok",
                chunk_size=1,
                action_dim=2,
                action_list=[0.1, float("nan")],
            ),
            "inf": SimpleNamespace(
                success=True,
                message="ok",
                chunk_size=1,
                action_dim=2,
                action_list=[0.1, float("inf")],
            ),
        }
        for name, response in responses.items():
            with self.subTest(name=name):
                processor = FakeProcessor(buffer_size=3)
                loop = ControlLoop(requester=FakeRequester(response))
                loop._running = True
                loop._processor = processor
                loop._active_action_policy_mode = "rlt"
                generation = loop._generation

                loop._request_and_buffer("pick", generation, "sync", "rlt")

                self.assertEqual(loop._active_action_policy_mode, "base")
                self.assertIsNone(loop._pending_action_policy_mode)
                self.assertEqual(loop._generation, generation + 1)
                self.assertEqual(processor.clear_count, 1)
                self.assertEqual(processor.pushed_chunks, [])

    def test_stale_rlt_failure_does_not_change_active_mode(self) -> None:
        response = SimpleNamespace(success=False, message="stale failure")
        processor = FakeProcessor(buffer_size=3)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._processor = processor
        loop._active_action_policy_mode = "rlt"

        loop._request_and_buffer("pick", loop._generation - 1, "sync", "rlt")

        self.assertEqual(loop._active_action_policy_mode, "rlt")
        self.assertEqual(processor.clear_count, 0)

    def test_get_action_request_carries_selected_policy_mode(self) -> None:
        response = SimpleNamespace(
            success=False,
            message="stop",
            chunk_size=0,
            action_dim=0,
            action_list=[],
        )
        requester = FakeRequester(response)
        loop = ControlLoop(requester=requester)
        loop._running = True
        loop._rlt_enabled = True

        loop._request_and_buffer("pick", loop._generation, "sync", "rlt")

        self.assertEqual(requester.calls, [("pick", "rlt")])

    def test_boundary_smoothing_is_enabled_only_for_rlt_policy(self) -> None:
        for policy_mode, expected_steps in (("base", None), ("rlt", 2)):
            with self.subTest(policy_mode=policy_mode):
                response = SimpleNamespace(
                    success=True,
                    message="ok",
                    chunk_size=2,
                    action_dim=2,
                    action_list=[0.1, 0.2, 0.3, 0.4],
                )
                processor = FakeProcessor(buffer_size=0)
                loop = ControlLoop(requester=FakeRequester(response))
                loop._running = True
                loop._processor = processor

                loop._request_and_buffer(
                    "pick",
                    loop._generation,
                    "sync",
                    policy_mode,
                )

                self.assertEqual(
                    processor.boundary_blend_source_steps,
                    [expected_steps],
                )


if __name__ == "__main__":
    unittest.main()
