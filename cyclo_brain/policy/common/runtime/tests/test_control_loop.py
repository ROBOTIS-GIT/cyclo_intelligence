#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_client_stub)

from main_runtime import control_loop as control_loop_module  # noqa: E402
from main_runtime.control_loop import ControlLoop  # noqa: E402


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
        self.sync_targets = []
        self.holds = []
        self.hold_failures_remaining = 0
        self.action_keys = ["arm"]

    def publish_action(self, action, action_keys) -> None:
        self.commands.append((np.asarray(action).copy(), list(action_keys)))

    def publish_action_preview(self, action, action_keys) -> None:
        self.previews.append((np.asarray(action).copy(), list(action_keys)))

    def publish_idle_action(self, action_keys) -> None:
        self.idles.append(list(action_keys))

    def publish_initial_pose_sync(self, action, action_keys, duration_s) -> None:
        self.sync_targets.append(
            (np.asarray(action).copy(), list(action_keys), float(duration_s))
        )

    def publish_current_pose_hold(self, action_keys, duration_s) -> None:
        if self.hold_failures_remaining > 0:
            self.hold_failures_remaining -= 1
            raise RuntimeError("hold publish failed")
        self.holds.append((list(action_keys), float(duration_s)))
        self._hold_stamp = time.monotonic()
        return {'commanded_at': self._hold_stamp, 'positions': {'arm_joint': 0.25},
                'mobile': 'mobile' in action_keys}

    def current_hold_feedback(self):
        self._hold_stamp += .02
        return {'positions': {'arm_joint': 0.25}, 'received': {'arm_joint': self._hold_stamp},
                'odom_received': self._hold_stamp,
                'odom': {'linear_velocity': [0, 0, 0], 'angular_velocity': [0, 0, 0]}}

    def close(self) -> None:
        pass


class FakeRequester:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def get_action(self, task_instruction):
        self.calls.append(task_instruction)
        return self.response


class SequenceRequester:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def get_action(self, task_instruction):
        self.calls.append(task_instruction)
        return self.responses.pop(0)


class ControlLoopSafetyTests(unittest.TestCase):
    def _make_loop(self, processor: FakeProcessor, robot: FakeRobot) -> ControlLoop:
        loop = ControlLoop(requester=object())
        loop._running = True
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        return loop

    def test_async_preserves_model_actions_and_policy_rate(self) -> None:
        loop = ControlLoop(requester=object(), inference_hz=15, control_hz=100,
                           postprocess_actions=True, target_chunk_size=100)
        with mock.patch.object(control_loop_module, "RobotClient", return_value=FakeRobot()):
            loop.configure(robot_type="ffw", action_request_mode="async")
        processor = loop._processor
        chunks = [np.array([[1., 9.], [20., -3.], [7., 2.]]),
                  np.array([[-30., 15.], [2., 40.]])]
        for chunk in chunks:
            processor.push_actions(chunk)
        self.assertEqual(processor.buffer_size, 5)
        self.assertEqual(loop._tick_period(), 1. / 15)
        actual = np.stack([processor.pop_action() for _ in range(5)])
        np.testing.assert_array_equal(actual, np.concatenate(chunks))
        self.assertIsNone(processor.pop_action())

    def test_sync_keeps_legacy_commands_without_rail(self):
        from action_chunk_processing import ActionChunkProcessor
        loop = ControlLoop(requester=object())
        with mock.patch.object(control_loop_module, "RobotClient", return_value=FakeRobot()):
            loop.configure(robot_type="ffw", action_request_mode="sync")
        self.assertIsNone(loop._rail)
        expected = ActionChunkProcessor(inference_hz=15, control_hz=100)
        self.assertEqual(loop._tick_period(), .01)
        for chunk in (np.arange(32.).reshape(16,2), -np.arange(32.).reshape(16,2)):
            expected.push_actions(chunk, scheduled_start_delay_s=None, align=False)
            loop._processor.push_actions(chunk, scheduled_start_delay_s=None, align=False)
            self.assertFalse(loop._should_request_actions(loop._processor))
            while expected.buffer_size:
                np.testing.assert_array_equal(loop._processor.pop_action(), expected.pop_action())
            self.assertTrue(loop._should_request_actions(loop._processor))


    def test_configure_applies_requested_timing(self) -> None:
        loop = ControlLoop(
            requester=object(),
            inference_hz=12,
            control_hz=80,
            chunk_align_window_s=0.4,
        )
        processor = FakeProcessor()
        with self.assertLogs(control_loop_module.logger, level="INFO") as logs:
            with (
                mock.patch.object(
                    control_loop_module, "RobotClient", return_value=FakeRobot()
                ),
                mock.patch.object(
                    control_loop_module,
                    "ActionChunkProcessor",
                    return_value=processor,
                ) as processor_factory,
            ):
                loop.configure(
                    robot_type="ffw",
                    control_hz=50,
                    inference_hz=20,
                    chunk_align_window_s=0.25,
                )

        processor_factory.assert_called_once_with(
            inference_hz=20.0,
            control_hz=50.0,
            chunk_align_window_s=0.25,
            postprocess=False,
            target_chunk_size=None,
            alignment_mode="none",
        )
        self.assertIn(
            "control_hz=50 inference_hz=20 chunk_align_window_s=0.25",
            "\n".join(logs.output),
        )
        self.assertEqual(loop._tick_period(), 1.0 / processor.output_hz)

    def test_configure_invalid_timing_uses_constructor_defaults(self) -> None:
        loop = ControlLoop(
            requester=object(),
            inference_hz=12,
            control_hz=80,
            chunk_align_window_s=0.4,
        )
        with (
            mock.patch.object(
                control_loop_module, "RobotClient", return_value=FakeRobot()
            ),
            mock.patch.object(
                control_loop_module,
                "ActionChunkProcessor",
                return_value=FakeProcessor(),
            ) as processor_factory,
        ):
            loop.configure(
                robot_type="ffw",
                control_hz=0,
                inference_hz=float("nan"),
                chunk_align_window_s=-1,
            )

        call_kwargs = processor_factory.call_args.kwargs
        self.assertEqual(call_kwargs["control_hz"], 80.0)
        self.assertEqual(call_kwargs["inference_hz"], 12.0)
        self.assertEqual(call_kwargs["chunk_align_window_s"], 0.4)

    def test_configure_zero_timing_uses_runtime_defaults(self) -> None:
        loop = ControlLoop(requester=object())
        with (
            mock.patch.object(
                control_loop_module, "RobotClient", return_value=FakeRobot()
            ),
            mock.patch.object(
                control_loop_module,
                "ActionChunkProcessor",
                return_value=FakeProcessor(),
            ) as processor_factory,
        ):
            loop.configure(
                robot_type="ffw",
                control_hz=0,
                inference_hz=0,
                chunk_align_window_s=0.0,
            )

        call_kwargs = processor_factory.call_args.kwargs
        self.assertEqual(call_kwargs["control_hz"], 100.0)
        self.assertEqual(call_kwargs["inference_hz"], 15.0)
        self.assertEqual(call_kwargs["chunk_align_window_s"], 0.3)

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

    def test_normal_physical_pause_holds_and_prevents_more_action_publish(self):
        robot = FakeRobot()
        processor = FakeProcessor(actions=[np.asarray([0.4])])
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        self.assertTrue(loop.pause())
        loop.tick()
        self.assertEqual(robot.holds, [(["arm"], .1)])
        self.assertFalse(loop._running)
        self.assertFalse(loop.initial_pose_sync_hold_required())
        self.assertEqual(robot.commands, [])
        self.assertEqual(processor.clear_count, 1)

    def test_failed_normal_hold_blocks_resume_until_retry_confirms(self):
        robot = FakeRobot(); robot.hold_failures_remaining = 1
        loop = self._make_loop(FakeProcessor(), robot)
        loop._publish_to_robot = True
        self.assertFalse(loop.pause())
        self.assertFalse(loop._running)
        with self.assertRaisesRegex(RuntimeError, "hold is still pending"):
            loop.start()
        self.assertTrue(loop.stop())
        self.assertFalse(loop.initial_pose_sync_hold_required())
        self.assertFalse(loop.start())
        self.assertTrue(loop._running)

    def test_late_chunk_after_pause_is_discarded(self):
        entered, release = threading.Event(), threading.Event()
        response = SimpleNamespace(success=True, message='ok', chunk_size=1, action_dim=1, action_list=[.9])
        class BlockingRequester:
            def get_action(self, instruction):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test did not release requester')
                return response
        processor = FakeProcessor(buffer_size=0)
        loop = self._make_loop(processor, FakeRobot())
        loop._requester = BlockingRequester()
        worker = threading.Thread(target=loop._request_and_buffer, args=('pick', loop._generation, 'async'))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(loop.pause())
        finally:
            release.set(); worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(processor.pushed_chunks, [])
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

    def test_async_mode_buffers_chunk_without_alignment_or_latency_skip(self) -> None:
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
        self.assertIsNone(processor.scheduled_delays[-1])
        self.assertEqual(processor.align_flags[-1], False)

    def test_initial_pose_sync_discards_first_chunk_and_requests_fresh_chunk(self) -> None:
        first = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2, 9.0, 9.0],
        )
        second = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.3, 0.4, 0.5, 0.6],
        )
        requester = SequenceRequester([first, second])
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._task_instruction = "pick"
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._initial_pose_sync_duration_s = 5.0

        self.assertTrue(loop.start())

        self.assertEqual(len(requester.calls), 1)
        self.assertEqual(len(robot.sync_targets), 1)
        np.testing.assert_allclose(robot.sync_targets[0][0], [0.1, 0.2])
        self.assertEqual(processor.pushed_chunks, [])

        loop.tick()
        self.assertEqual(len(requester.calls), 1)
        self.assertEqual(robot.idles, [["arm"]])

        loop._initial_pose_sync_deadline = 0.0
        loop.tick()
        loop._request_thread.join(timeout=1.0)

        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(len(processor.pushed_chunks), 1)
        np.testing.assert_allclose(processor.pushed_chunks[0], [[0.3, 0.4], [0.5, 0.6]])

    def test_pause_during_initial_pose_sync_holds_and_resume_retries(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        requester = SequenceRequester([response, response])
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True

        self.assertTrue(loop.start())
        self.assertTrue(loop.pause())
        self.assertEqual(robot.holds, [(["arm"], 0.1)])
        self.assertFalse(loop._initial_pose_sync_completed)

        self.assertTrue(loop.start())
        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(len(robot.sync_targets), 2)

    def test_failed_sync_hold_remains_retryable(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        robot = FakeRobot()
        robot.hold_failures_remaining = 1
        loop = ControlLoop(requester=FakeRequester(response))
        loop._robot = robot
        loop._processor = FakeProcessor()
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True

        self.assertTrue(loop.start())
        self.assertFalse(loop.pause())
        self.assertTrue(loop._initial_pose_sync_hold_pending)
        self.assertTrue(loop._initial_pose_sync_in_progress)
        self.assertFalse(loop._running)
        with self.assertRaisesRegex(RuntimeError, "hold is still pending"):
            loop.start()

        self.assertTrue(loop.pause())
        self.assertFalse(loop._initial_pose_sync_hold_pending)
        self.assertFalse(loop._initial_pose_sync_in_progress)
        self.assertEqual(robot.holds, [(["arm"], 0.1)])

    def test_repeated_sync_hold_failures_remain_retryable(self) -> None:
        robot = FakeRobot()
        robot.hold_failures_remaining = 2
        loop = self._make_loop(FakeProcessor(), robot)
        loop._publish_to_robot = True
        loop._initial_pose_sync_in_progress = True

        self.assertFalse(loop.pause())
        self.assertFalse(loop.pause())
        self.assertTrue(loop._initial_pose_sync_hold_pending)
        self.assertTrue(loop.pause())
        self.assertFalse(loop._initial_pose_sync_hold_pending)

    def test_resume_after_completed_sync_does_not_sync_again(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=1,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        requester = FakeRequester(response)
        processor = FakeProcessor(buffer_size=100)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        loop._initial_pose_sync_completed = True

        self.assertFalse(loop.start())
        loop.pause()
        self.assertFalse(loop.start())

        self.assertEqual(requester.calls, [])
        self.assertEqual(robot.sync_targets, [])

    def test_explicit_resume_sync_realigns_after_stop(self) -> None:
        response = SimpleNamespace(success=True, message="ok", chunk_size=1,
                                   action_dim=2, action_list=[0.1, 0.2])
        requester = FakeRequester(response)
        robot = FakeRobot()
        loop = ControlLoop(requester=requester)
        loop._robot = robot
        loop._processor = FakeProcessor(buffer_size=100)
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_completed = True
        self.assertTrue(loop.start(initial_pose_sync=True, initial_pose_sync_duration_s=2.0))
        self.assertEqual(len(robot.sync_targets), 1)
        self.assertFalse(loop._initial_pose_sync_completed)
        loop.stop()
        self.assertTrue(loop.start(initial_pose_sync=True, initial_pose_sync_duration_s=2.0))
        self.assertEqual(len(robot.sync_targets), 2)

    def test_explicit_resume_sync_is_ignored_for_simulation(self) -> None:
        requester = FakeRequester(None)
        loop = ControlLoop(requester=requester)
        loop._robot = FakeRobot()
        loop._processor = FakeProcessor()
        self.assertFalse(loop.start(publish_to_robot=False, initial_pose_sync=True))
        self.assertEqual(requester.calls, [])

    def test_simulation_ignores_initial_pose_sync(self) -> None:
        requester = FakeRequester(None)
        loop = ControlLoop(requester=requester)
        loop._robot = FakeRobot()
        loop._processor = FakeProcessor()
        loop._initial_pose_sync_enabled = True
        loop._publish_to_robot = False

        self.assertFalse(loop.start())
        self.assertEqual(requester.calls, [])

    def test_malformed_initial_pose_chunk_fails_before_publish(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2],
        )
        robot = FakeRobot()
        loop = ControlLoop(requester=FakeRequester(response))
        loop._robot = robot
        loop._processor = FakeProcessor()
        loop._action_keys = ["arm"]
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True

        with self.assertRaisesRegex(ValueError, "size mismatch"):
            loop.start()

        self.assertEqual(robot.sync_targets, [])
        self.assertFalse(loop._running)


if __name__ == "__main__":
    unittest.main()
