"""Policy transitions with inference latency and a continuously ticking robot."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from test_playground_control_loop import ControlLoop, FakeRobot, TTActionTimeline


class ContinuousPolicySwitchTests(unittest.TestCase):
    def test_both_directions_keep_publishing_during_350ms_request(self):
        for active, target in (("rlt", "base"), ("base", "rlt")):
            for hz in (100, 200):
                with self.subTest(active=active, hz=hz):
                    now = [100.0]
                    started, release = threading.Event(), threading.Event()
                    requests = []

                    class Requester:
                        def get_action(self, task, **kw):
                            requests.append(kw)
                            prefix = np.array(kw['rtc_prefix_action_list']).reshape(-1, 16)
                            count = 32 - len(prefix) if target == 'base' else 10
                            fresh = prefix[-1] + np.arange(1, count + 1)[:, None] * .001
                            chunk = np.concatenate((prefix, fresh)) if target == 'base' else fresh
                            started.set()
                            if not release.wait(2):
                                raise TimeoutError('test did not release inference')
                            return SimpleNamespace(success=True, chunk_size=len(chunk),
                                action_dim=16, action_list=chunk.ravel().tolist(),
                                rtc_reference_chunk_size=10,
                                rtc_reference_action_list=fresh[:10].ravel().tolist())

                    timeline = TTActionTimeline(control_hz=hz)
                    source = np.repeat(np.arange(7)[:, None] * .001, 16, axis=1)
                    timeline.push_actions(source)
                    robot = FakeRobot()
                    loop = ControlLoop(Requester(), control_hz=hz)
                    loop._processor, loop._robot = timeline, robot
                    loop._running = loop._publish_to_robot = True
                    loop._action_request_mode = 'tt_rtc'
                    loop._tt_rtc_horizon, loop._tt_rtc_action_dim = 32, 16
                    loop._active_action_policy_mode = active
                    loop._active_rlt_robot_override = True
                    loop._pending_action_policy_mode = target
                    loop._pending_rlt_robot_override = target == 'rlt'
                    try:
                        with patch('main_runtime.playground_control_loop.time.monotonic', side_effect=lambda: now[0]):
                            loop.tick()
                            self.assertTrue(started.wait(1))
                            for _ in range(round(.35 * hz)):
                                now[0] += 1 / hz
                                loop.tick()
                            self.assertEqual(loop._active_action_policy_mode, active)
                            release.set()
                            loop._request_thread.join(1)
                        self.assertEqual(loop._active_action_policy_mode, target)
                        self.assertIsNone(loop.tt_rtc_failure_reason)
                        self.assertEqual(len(robot.commands), 1 + round(.35 * hz))
                        self.assertEqual(robot.idles, [])
                        self.assertEqual(robot.handoff_requests, [])
                        self.assertEqual(timeline.timing_stats['underrun_events'], 0)
                        np.testing.assert_array_equal(
                            np.array(requests[0]['rtc_prefix_action_list']).reshape(6, 16), source[1:])
                    finally:
                        release.set()
                        if loop._request_thread:
                            loop._request_thread.join(2)

    def test_inflight_mlp_refill_survives_pending_base_switch(self):
        started, release = threading.Event(), threading.Event()
        prefix = np.zeros((6, 16))
        postfix = np.full((10, 16), .01)

        class Requester:
            def get_action(self, *args, **kwargs):
                started.set()
                release.wait(2)
                return SimpleNamespace(success=True, chunk_size=10, action_dim=16,
                    action_list=postfix.ravel().tolist(), rtc_reference_chunk_size=10,
                    rtc_reference_action_list=postfix.ravel().tolist())

        loop = ControlLoop(Requester())
        loop._processor = TTActionTimeline()
        loop._processor.push_actions(prefix)
        loop._running, loop._rlt_enabled = True, True
        loop._action_request_mode, loop._active_action_policy_mode = 'tt_rtc', 'rlt'
        loop._tt_rtc_horizon, loop._tt_rtc_action_dim = 32, 16
        generation = loop._generation
        worker = threading.Thread(target=lambda: loop._request_and_buffer(
            '', generation, 'tt_rtc', 'rlt', prefix.copy()))
        result = []
        switch = threading.Thread(target=lambda: result.append(loop.set_action_policy('base', timeout_s=2)))
        try:
            worker.start()
            self.assertTrue(started.wait(1))
            switch.start()
            deadline = time.monotonic() + 1
            while loop._pending_action_policy_mode != 'base' and time.monotonic() < deadline:
                threading.Event().wait(.001)
            self.assertEqual(loop._pending_action_policy_mode, 'base')
            self.assertEqual(loop._generation, generation)
            release.set()
            worker.join(1)
            self.assertEqual(loop._processor.buffer_size, 16)
            self.assertEqual(loop._active_action_policy_mode, 'rlt')
            self.assertEqual(loop._pending_action_policy_mode, 'base')
        finally:
            release.set()
            worker.join(2)
            with loop._mode_condition:
                loop._running = False
                loop._mode_condition.notify_all()
            switch.join(2)


if __name__ == '__main__':
    unittest.main()
