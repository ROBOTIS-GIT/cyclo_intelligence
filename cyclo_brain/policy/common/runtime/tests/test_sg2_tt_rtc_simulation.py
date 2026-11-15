"""Exercise SG2 TT-RTC lifecycle with synthetic chunks and no transport.

This validates the production service, request protocol and control timeline;
it deliberately does not stand in for a trained GR00T checkpoint forward pass.
"""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

RUNTIME = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
for path in (REPO, RUNTIME, REPO / "cyclo_brain/policy/groot",
             REPO / "shared/shared/robot_configs"):
    sys.path.insert(0, str(path))

# Importing the loop must not create a ROS or Zenoh connection on a test host.
if "robot_client" not in sys.modules:
    stub = types.ModuleType("robot_client")
    stub.RobotClient = object
    sys.modules["robot_client"] = stub

from engine_process.worker import EngineWorker
from main_runtime.playground_control_loop import ControlLoop
from main_runtime.inference_requester import InferenceRequester
from main_runtime.service_handler import ServiceHandler
from main_runtime.session_state import SessionState
from cyclo_brain.policy.groot.runtime.tt_rtc import parse_tt_rtc_request
import schema


class PreviewRobot:
    def __init__(self, robot_type, **_kwargs):
        assert robot_type == "ffw_sg2_rev1"
        self.groups = schema.get_action_groups(schema.load_robot_section(robot_type))
        self.action_keys = list(self.groups)
        self.previews = []
        self.physical_calls = []

    def get_action_joint_limits(self, action_keys):
        # This fixture checks transport/lifecycle; physical smoothing has separate tests.
        return []

    def publish_action_preview(self, action, keys):
        assert len(action) == sum(len(self.groups[k]["joint_names"]) for k in keys)
        self.previews.append(np.array(action, copy=True))

    def publish_action(self, *args, **kwargs):
        self.physical_calls.append("command")

    def publish_idle_action(self, *args, **kwargs):
        self.physical_calls.append("idle")

    def publish_current_pose_hold(self, *args, **kwargs):
        self.physical_calls.append("hold")

    def publish_initial_pose_sync(self, *args, **kwargs):
        self.physical_calls.append("initial_pose")
        return 5.0

    def close(self):
        pass


class SyntheticEngine:
    def __init__(self, horizon, dimension, delay_range):
        self.horizon, self.dimension = horizon, dimension
        self.delay_range = delay_range
        self.requests = []
        self.bad_output = False
        self.cleaned = False

    def load_policy(self, request):
        assert request.publish_to_robot is False
        assert request.robot_type == "ffw_sg2_rev1"
        keys = ["arm_left", "arm_right"]
        if self.dimension == 19:
            keys += ["head", "lift"]
        return dict(success=True, action_keys=keys,
                    chunk_size=self.horizon, action_dim=self.dimension,
                    rtc_min_delay_steps=self.delay_range[0],
                    rtc_max_delay_steps=self.delay_range[1])

    def get_action_chunk(self, request):
        parse_tt_rtc_request(request, action_dim=self.dimension,
                             min_delay_steps=self.delay_range[0],
                             max_delay_steps=self.delay_range[1])
        self.requests.append(request)
        chunk = np.full((self.horizon, self.dimension), 0.1)
        if self.dimension == 19:
            chunk[:, -1] = -0.3  # SG2 lift, in metres.
        delay = request.rtc_delay_steps
        if delay:
            chunk[:delay] = np.asarray(request.rtc_prefix_action_list).reshape(
                delay, self.dimension)
        if self.bad_output:
            chunk[-1, 0] = np.nan
        return dict(success=True, chunk_size=self.horizon,
                    action_dim=self.dimension, action_chunk=chunk)

    def cleanup(self):
        self.cleaned = True


class SG2TTRTCSimulationTests(unittest.TestCase):
    def exercise_profile(self, horizon, dimension, delay_range, hz, prefix_mode):
        engine = SyntheticEngine(horizon, dimension, delay_range)
        worker = EngineWorker(engine)
        requester = InferenceRequester(SimpleNamespace(
            call=lambda request, **kwargs: worker.handle(request)))
        loop = ControlLoop(requester, tt_rtc_prefix_mode=prefix_mode)
        session = SessionState()
        handler = ServiceHandler(session, requester, loop,
                                 lambda **kwargs: SimpleNamespace(**kwargs))

        def command(number, **kwargs):
            return handler.handle(SimpleNamespace(
                command=number, model_path="synthetic-fixture",
                robot_type="ffw_sg2_rev1", task_instruction="offline check",
                action_request_mode="tt_rtc", control_hz=hz,
                initial_pose_sync=True, **kwargs))

        def tick(count):
            for _ in range(count):
                loop.tick()
                if loop._request_thread is not None:
                    loop._request_thread.join(timeout=2)
                    self.assertFalse(loop._request_thread.is_alive())

        with patch("main_runtime.playground_control_loop.RobotClient", PreviewRobot):
            try:
                loaded = command(0, publish_to_robot=False)
                self.assertTrue(loaded.success, loaded.message)
                robot = loop._robot
                self.assertTrue(command(1, publish_to_robot=False).success)
                tick(round(hz * 4))
                self.assertIsNone(loop.tt_rtc_failure_reason)
                self.assertGreater(len(robot.previews), hz * 3)
                self.assertGreaterEqual(len(engine.requests), 3)
                self.assertEqual(engine.requests[0].rtc_delay_steps, 0)
                expected_delay = int(prefix_mode.rsplit("_", 1)[1])
                self.assertTrue(all(r.rtc_delay_steps == expected_delay
                                    for r in engine.requests[1:]))
                self.assertTrue(np.isfinite(robot.previews).all())
                self.assertEqual(robot.physical_calls, [])
                self.assertTrue(command(2).success)  # PAUSE
                before = len(robot.previews)
                tick(5)
                self.assertEqual(len(robot.previews), before)
                self.assertFalse(command(3, publish_to_robot=True).success)
                self.assertEqual(robot.physical_calls, [])
                self.assertTrue(command(3, publish_to_robot=False).success)
                tick(round(hz))
                self.assertGreater(len(robot.previews), before)
                # Malformed model output must pause without sending commands.
                engine.bad_output = True
                tick(round(hz * 3))
                self.assertIsNotNone(loop.tt_rtc_failure_reason)
                self.assertFalse(loop._running)
                self.assertTrue(command(4).success)  # STOP
                self.assertTrue(command(5).success)  # UNLOAD
                self.assertTrue(engine.cleaned)
                self.assertEqual(robot.physical_calls, [])
            finally:
                loop.shutdown()

    def test_supported_sg2_profiles_and_control_rates(self):
        for horizon, dimension, delay_range, prefix_mode in (
            (16, 19, (0, 6), "fixed_first_6"),
            (32, 16, (0, 6), "fixed_first_6"),
            (32, 16, (0, 8), "fixed_first_8"),
            (20, 16, (6, 8), "fixed_first_6"),
            (20, 16, (6, 8), "fixed_first_8"),
        ):
            for hz in (50, 100, 200):
                with self.subTest(horizon=horizon, dimension=dimension,
                                  hz=hz, prefix_mode=prefix_mode):
                    self.exercise_profile(horizon, dimension, delay_range, hz, prefix_mode)


if __name__ == "__main__":
    unittest.main()
