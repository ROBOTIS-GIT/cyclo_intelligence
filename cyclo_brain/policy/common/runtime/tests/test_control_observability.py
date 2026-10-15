"""Generic queue and startup trace contracts without model or robot execution."""
import unittest
from types import SimpleNamespace
from unittest import mock
from test_control_loop import ControlLoop, FakeProcessor, FakeRobot, control_loop_module


class MemoryTrace:
    def __init__(self):
        self.events = []

    def emit(self, event, **payload):
        self.events.append({"event": event, **payload})


class ControlObservabilityTests(unittest.TestCase):
    def _loop(self, requester):
        loop = ControlLoop(requester=requester)
        loop._robot = FakeRobot()
        loop._processor = FakeProcessor()
        loop._action_keys = ["arm"]
        loop._trace = MemoryTrace()
        loop._buffer_trace.trace = loop._trace
        return loop

    def test_baseline_marks_actual_appended_chunk_boundaries_and_underflow(self):
        response = SimpleNamespace(success=True, message="", timing_json="{}",
                                   action_list=[0.0, 1.0], chunk_size=2, action_dim=1)
        loop = self._loop(SimpleNamespace(get_action=lambda task: response))
        loop._processor = control_loop_module.ActionChunkProcessor(
            inference_hz=10.0, control_hz=10.0, postprocess=False)
        loop._running = True
        loop._publish_to_robot = True
        with mock.patch.object(loop, "_should_request_actions", return_value=False):
            loop.tick()  # Startup idle does not create an underflow interval.
            loop._request_and_buffer("pick", loop._generation, "sync")
            loop._request_and_buffer("pick", loop._generation, "sync")
            installed = [e for e in loop._trace.events if e["event"] == "chunk_installed"]
            self.assertEqual([e["queued_control_steps"] for e in installed], [0, 2])
            for _ in range(4):
                loop.tick()
            commands = [e for e in loop._trace.events if e["event"] == "command_published"]
            self.assertEqual([e["request_id"] for e in commands], [e["request_id"] for e in installed])
            self.assertEqual(len(loop._robot.commands), 4)
            loop.tick()
            loop.tick()
            self.assertEqual([e["phase"] for e in loop._trace.events if e["event"] == "queue_underflow"], ["start"])
            loop._request_and_buffer("pick", loop._generation, "sync")
            loop.tick()
        self.assertEqual([e["phase"] for e in loop._trace.events if e["event"] == "queue_underflow"], ["start", "end"])
        self.assertEqual(len(loop._buffer_trace._markers), 0)

    def test_initial_sync_retains_first_warmup_over_two_seconds(self):
        response = SimpleNamespace(success=True, message="", timing_json='{"inference_s":2.4}',
                                   action_list=[0.0, 1.0], chunk_size=2, action_dim=1)
        loop = self._loop(SimpleNamespace(get_action=lambda task: response))
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        with mock.patch.object(control_loop_module.time, "monotonic", side_effect=[10.0, 12.5, 12.5, 12.5]):
            self.assertTrue(loop.start())
        completed = [row for row in loop._trace.events if row["event"] == "request_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["latency_s"], 2.5)
        self.assertTrue(completed[0]["cold_start"])
        self.assertEqual(completed[0]["purpose"], "initial_pose_sync")
        self.assertEqual(completed[0]["timing"]["inference_s"], 2.4)

    def test_initial_sync_timeout_is_traced_and_does_not_publish_target(self):
        requester = SimpleNamespace(get_action=mock.Mock(side_effect=TimeoutError("engine timeout")))
        loop = self._loop(requester)
        loop._publish_to_robot = True
        loop._initial_pose_sync_enabled = True
        with self.assertRaises(TimeoutError):
            loop.start()
        completed = [row for row in loop._trace.events if row["event"] == "request_completed"]
        self.assertEqual(completed[0]["status"], "timeout")
        self.assertFalse(completed[0]["success"])
        self.assertTrue(completed[0]["cold_start"])
        self.assertEqual(loop._robot.sync_targets, [])
        self.assertFalse(loop._running)

