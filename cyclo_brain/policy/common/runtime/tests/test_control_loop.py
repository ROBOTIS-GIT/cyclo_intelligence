#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
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

from main_runtime.control_loop import ControlLoop  # noqa: E402
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

    def pop_action(self):
        if self._actions:
            return self._actions.pop(0)
        return None

    def clear(self) -> None:
        self.clear_count += 1
        self._actions.clear()
        self.buffer_size = 0

    def push_actions(self, chunk, scheduled_start_delay_s=None, align=True):
        data = np.asarray(chunk, dtype=np.float64)
        self.pushed_chunks.append(data.copy())
        self.scheduled_delays.append(scheduled_start_delay_s)
        self.align_flags.append(bool(align))
        self.buffer_size += len(data)
        return len(data)


class FakeRobot:
    def __init__(self) -> None:
        self.commands = []
        self.previews = []
        self.idles = []
        self.action_keys = ["arm"]

    def publish_action(self, action, action_keys) -> None:
        self.commands.append((np.asarray(action).copy(), list(action_keys)))

    def publish_action_preview(self, action, action_keys) -> None:
        self.previews.append((np.asarray(action).copy(), list(action_keys)))

    def publish_idle_action(self, action_keys) -> None:
        self.idles.append(list(action_keys))

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

    def test_robot_publish_error_does_not_crash_tick(self) -> None:
        class FailingRobot(FakeRobot):
            def publish_action(self, action, action_keys) -> None:
                raise RuntimeError("publish failed")

        processor = FakeProcessor(actions=[np.asarray([0.5], dtype=np.float64)])
        robot = FailingRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True

        loop.tick()

        self.assertEqual(len(robot.previews), 1)

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

    def test_tt_rtc_configures_exact_15_hz_source_queue(self) -> None:
        loop = ControlLoop(
            requester=object(),
            inference_hz=30.0,
            control_hz=100.0,
            postprocess_actions=True,
        )

        with patch(
            "main_runtime.control_loop.RobotClient",
            side_effect=lambda *_args, **_kwargs: FakeRobot(),
        ):
            loop.configure(
                robot_type="ffw_sg2_rev1",
                action_request_mode="tt_rtc",
            )

        self.assertEqual(loop._processor.output_hz, 15.0)
        loop._processor.push_actions(np.zeros((16, 19), dtype=np.float64))
        self.assertEqual(loop._processor.buffer_size, 16)
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

    def test_tt_rtc_rlt_appends_complete_ten_action_postfix(self) -> None:
        prefix = np.arange(6 * 19, dtype=np.float64).reshape(6, 19)
        rlt_chunk = 2000.0 + np.arange(10 * 19, dtype=np.float64).reshape(10, 19)
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=10,
            action_dim=19,
            action_list=rlt_chunk.reshape(-1).tolist(),
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

        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((prefix, rlt_chunk), axis=0),
        )

    def test_tt_rtc_switches_vla_and_mlp_without_draining_prefix(self) -> None:
        for active_mode, target_mode in (("base", "rlt"), ("rlt", "base")):
            with self.subTest(active=active_mode, target=target_mode):
                source = np.arange(7 * 19, dtype=np.float64).reshape(7, 19)
                prefix = source[1:]
                postfix = (
                    4000.0
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
                )
                requester = FakeRequester(response)
                processor = ActionChunkProcessor(
                    inference_hz=15.0,
                    control_hz=100.0,
                    postprocess=False,
                )
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
                loop._request_thread.join(timeout=1.0)
                switch_thread.join(timeout=1.0)

                self.assertFalse(switch_thread.is_alive())
                self.assertEqual(result, [
                    (True, f"{target_mode.upper()} action active")
                ])
                self.assertEqual(loop._active_action_policy_mode, target_mode)
                self.assertIsNone(loop._pending_action_policy_mode)
                self.assertEqual(requester.calls[-1], ("", target_mode))
                np.testing.assert_allclose(
                    processor.peek_actions(),
                    np.concatenate((prefix, postfix), axis=0),
                )
                self.assertEqual(len(robot.previews), 1)
                self.assertEqual(robot.idles, [])

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
        np.testing.assert_allclose(
            processor.peek_actions(),
            np.concatenate((target_prefix, target_postfix), axis=0),
        )

    def test_tt_rtc_rejected_mlp_postfix_falls_back_without_dropping_prefix(
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
        np.testing.assert_allclose(processor.peek_actions(), prefix)

    def test_tt_rtc_failed_mlp_handoff_reports_failure_and_keeps_prefix(
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
        np.testing.assert_allclose(processor.peek_actions(), source[1:])
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

    def test_tt_rtc_deadline_starts_when_prefix_is_captured(self) -> None:
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
            "main_runtime.control_loop.time.monotonic",
            side_effect=[100.300, 100.310, 100.410],
        ):
            loop._request_and_buffer(
                "pick",
                loop._generation,
                "tt_rtc",
                "base",
                prefix.copy(),
                100.0,
            )

        self.assertEqual(processor.buffer_size, 6)

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

        np.testing.assert_allclose(processor.peek_actions(), replacement)

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

        self.assertEqual(processor.buffer_size, 6)

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

        loop._request_and_buffer("pick", loop._generation, "sync", "rlt")

        self.assertEqual(requester.calls, [("pick", "rlt")])


if __name__ == "__main__":
    unittest.main()
