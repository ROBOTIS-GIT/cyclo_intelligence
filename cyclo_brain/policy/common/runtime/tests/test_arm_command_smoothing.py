"""Offline motion constraints and control-loop integration; no transport."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_runtime.arm_command_smoothing import ArmCommandSmoother, ArmSmoothingConfig, smoothing_enabled
from test_playground_control_loop import ControlLoop, FakeProcessor, FakeRobot
from main_runtime.tt_rtc_timeline import TTActionTimeline

INDICES = np.r_[0:7, 8:15]


class SmoothingRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.action_keys = ['arm_left', 'arm_right']
        self.pose = np.zeros(16)

    def get_action_joint_limits(self, keys):
        names = [name for side in ('l', 'r') for name in
                 [*[f'arm_{side}_joint{i}' for i in range(1, 8)], f'gripper_{side}_joint1']]
        return [dict(name=n, lower=-3.14, upper=3.14, velocity=4.8) for n in names]

    def get_current_action_positions(self, keys):
        return self.pose.copy()


class ArmSmoothingTests(unittest.TestCase):
    def test_step_and_reversal_obey_velocity_acceleration_and_jerk(self):
        for dt in (.005, .01, .02):
            with self.subTest(dt=dt):
                smoother = ArmCommandSmoother(INDICES)
                smoother.reset(np.zeros(16))
                outputs = [np.zeros(16)]
                accelerations = [np.zeros(14)]
                for t in np.arange(0, 6, dt):
                    target = np.full(16, .5 if t < 2 else -.4)
                    target[7], target[15] = .9, .3
                    before = target.copy()
                    outputs.append(smoother.step(target, dt))
                    accelerations.append(smoother.acceleration)
                    np.testing.assert_array_equal(target, before)
                    self.assertLessEqual(np.max(np.abs(smoother.velocity)), 1.5 + 1e-9)
                    self.assertLessEqual(np.max(np.abs(smoother.acceleration)), 8 + 1e-9)
                output = np.asarray(outputs)
                self.assertLessEqual(np.max(np.abs(np.diff(output[:, INDICES], axis=0) / dt)), 1.5 + 1e-9)
                self.assertLessEqual(np.max(np.abs(np.diff(accelerations, axis=0) / dt)), 100 + 1e-7)
                np.testing.assert_allclose(output[-1, INDICES], -.4, atol=1e-6)
                np.testing.assert_array_equal(output[1:, 7], .9)
                np.testing.assert_array_equal(output[1:, 15], .3)

    def test_other_arm_motion_does_not_change_left_arm_output(self):
        one, both = ArmCommandSmoother(INDICES), ArmCommandSmoother(INDICES)
        one.reset(np.zeros(16)); both.reset(np.zeros(16))
        for t in np.arange(0, 4, .01):
            a = np.zeros(16); a[:7] = .1 * np.sin(t)
            b = a.copy(); b[8:15] = np.sin(t * 17)
            np.testing.assert_allclose(one.step(a, .01)[:7], both.step(b, .01)[:7], atol=1e-12, rtol=0)

    def test_slow_precise_motion_is_not_frozen_and_static_target_converges(self):
        s = ArmCommandSmoother([0]); s.reset([0])
        for target in np.linspace(0, .02, 1000):
            out = s.step([target], .01)
        self.assertLess(abs(out[0] - .02), .0005)
        for _ in range(200):out = s.step([.02], .01)
        self.assertAlmostEqual(out[0], .02, places=8)

    def test_invalid_data_and_position_limits_fail_without_mutating_input(self):
        s = ArmCommandSmoother([0], lower=[-.5], upper=[.5]); s.reset([0])
        for value in [np.nan, np.inf, .6, -.6]:
            with self.assertRaises(ValueError):s.step([value], .01)
        for dt in [0, -.01, .5, np.nan]:
            with self.assertRaises(ValueError):s.step([.1], dt)
        for value in [0, -1, np.inf, np.nan]:
            with self.assertRaises(ValueError):ArmSmoothingConfig(max_velocity_rad_s=value)

    def make_loop(self, physical=False):
        robot = SmoothingRobot()
        loop = ControlLoop(requester=object())
        with patch('main_runtime.playground_control_loop.RobotClient', return_value=robot), patch.dict(
            os.environ, {'CYCLO_SG2_ARM_SMOOTHING': os.environ.get('CYCLO_SG2_ARM_SMOOTHING', 'true')}
        ):
            loop.configure('ffw_sg2_rev1', action_keys=robot.action_keys,
                           action_request_mode='tt_rtc', tt_rtc_horizon=32,
                           tt_rtc_action_dim=16, publish_to_robot=physical)
        loop._running = True
        return loop, robot

    def test_pipeline_keeps_model_prefix_and_grippers_but_smooths_both_outputs(self):
        for physical in (False, True):
            loop, robot = self.make_loop(physical)
            chunk = np.full((32, 16), .3); chunk[:, [7, 15]] = .7
            loop._processor.push_actions(chunk)
            reference = TTActionTimeline(); reference.push_actions(chunk)
            with patch.object(loop, '_should_request_actions', return_value=False):
                for i in range(150):
                    reference.pop_action()
                    with patch('main_runtime.playground_control_loop.time.monotonic', return_value=10 + i * .01):loop.tick()
                    np.testing.assert_array_equal(loop._processor.peek_actions(), reference.peek_actions())
            self.assertEqual(len(robot.commands), 150 if physical else 0)
            for preview in robot.previews:np.testing.assert_array_equal(preview[0][[7, 15]], [.9, .9])
            for command, preview in zip(robot.commands, robot.previews):np.testing.assert_array_equal(command[0], preview[0])
            if physical:
                np.testing.assert_array_equal(robot.commands[0][0][INDICES], 0)
                self.assertGreater(robot.commands[25][0][0], 0)
                self.assertLess(robot.commands[25][0][0], .3)

    def test_pause_clears_state_and_resume_uses_fresh_measured_pose(self):
        loop, robot = self.make_loop(True)
        with patch('main_runtime.playground_control_loop.time.monotonic', return_value=10):loop._smooth_arm_command_locked(np.full(16, .2))
        loop.pause(); robot.pose[:] = -.1
        with patch('main_runtime.playground_control_loop.time.monotonic', return_value=20):out = loop._smooth_arm_command_locked(np.full(16, .2))
        np.testing.assert_array_equal(out[INDICES], -.1)
        np.testing.assert_array_equal(out[[7, 15]], .2)

    def test_stale_feedback_or_bad_smoothing_pauses_before_any_publish(self):
        loop, robot = self.make_loop(True)
        loop._processor = FakeProcessor(actions=[np.zeros(16)])
        with patch.object(robot, 'get_current_action_positions', side_effect=RuntimeError('stale state')):loop.tick()
        self.assertFalse(loop._running)
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.previews, [])
        self.assertIn('stale state', loop.tt_rtc_failure_reason)

    def test_long_gap_requires_resume_instead_of_large_catchup(self):
        loop, robot = self.make_loop()
        with patch('main_runtime.playground_control_loop.time.monotonic', return_value=10):loop._smooth_arm_command_locked(np.zeros(16))
        with patch('main_runtime.playground_control_loop.time.monotonic', return_value=11):
            with self.assertRaisesRegex(RuntimeError, '200ms'):loop._smooth_arm_command_locked(np.ones(16))

    def test_disable_is_explicit_and_other_robots_are_unmodified(self):
        with patch.dict(os.environ, {'CYCLO_SG2_ARM_SMOOTHING': 'false'}):loop, robot = self.make_loop()
        self.assertIsNone(loop._arm_smoother)
        loop._robot_type = 'other'
        loop._configure_arm_smoothing_locked()
        self.assertIsNone(loop._arm_smoother)

    def test_smoothing_requires_explicit_runtime_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(smoothing_enabled())


if __name__ == '__main__':unittest.main()
