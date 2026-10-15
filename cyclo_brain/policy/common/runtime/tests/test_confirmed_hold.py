"""Hold confirmation requires fresh, stationary feedback; no robot services."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_runtime import confirmed_hold as h


class Clock:
    now = 0.0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class FeedbackRobot:
    def __init__(self, clock, mode='stable', mobile=False):
        self.clock, self.mode, self.mobile = clock, mode, mobile
        self.commands = 0
    def publish_current_pose_hold(self, keys, duration_s):
        self.commands += 1
        return {'positions': {'joint': .4}, 'commanded_at': 0.0, 'mobile': self.mobile}
    def current_hold_feedback(self):
        stamp = self.clock.now
        position = .4
        if self.mode == 'moving': position += stamp * .2
        if self.mode == 'offset': position += .2
        if self.mode == 'nan': position = float('nan')
        positions = {} if self.mode == 'missing_joint' else {'joint': position}
        received = {} if self.mode == 'missing_timestamp' else {'joint': -1 if self.mode == 'stale' else stamp}
        odom = {'linear_velocity': [.1 if self.mode == 'moving_base' else 0, 0, 0],
                'angular_velocity': [0, 0, 0]}
        if self.mode == 'missing_base': odom = {}
        if self.mode == 'empty_base': odom['linear_velocity'] = []
        if self.mode == 'short_base': odom['angular_velocity'] = [0, 0]
        if self.mode == 'nan_base': odom['linear_velocity'] = [0, float('nan'), 0]
        return {'positions': positions, 'received': received, 'odom': odom,
                'odom_received': -1 if self.mode == 'stale_base' else stamp}


class ConfirmedHoldTests(unittest.TestCase):
    def test_fresh_stationary_joint_and_base_feedback_confirms(self):
        for mobile in (False, True):
            clock = Clock(); robot = FeedbackRobot(clock, mobile=mobile)
            with patch.object(h.time, 'monotonic', clock.monotonic), patch.object(h.time, 'sleep', clock.sleep):
                h.hold_current_pose(robot, ['arm'], timeout=.3)
            self.assertEqual(robot.commands, 1)
            self.assertGreaterEqual(clock.now, .06)

    def test_stale_missing_moving_or_off_target_joint_feedback_never_confirms(self):
        for mode in ('stale', 'missing_joint', 'missing_timestamp', 'moving', 'offset', 'nan'):
            with self.subTest(mode=mode):
                clock = Clock(); robot = FeedbackRobot(clock, mode=mode)
                with patch.object(h.time, 'monotonic', clock.monotonic), patch.object(h.time, 'sleep', clock.sleep):
                    with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                        h.hold_current_pose(robot, ['arm'], timeout=.3)

    def test_mobile_hold_requires_fresh_stationary_odom(self):
        for mode in ('stale_base', 'missing_base', 'moving_base', 'empty_base', 'short_base', 'nan_base'):
            with self.subTest(mode=mode):
                clock = Clock(); robot = FeedbackRobot(clock, mode=mode, mobile=True)
                with patch.object(h.time, 'monotonic', clock.monotonic), patch.object(h.time, 'sleep', clock.sleep):
                    with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                        h.hold_current_pose(robot, ['arm', 'mobile'], timeout=.3)

    def test_legacy_publisher_ack_without_receipt_is_not_confirmation(self):
        class LegacyRobot:
            def publish_current_pose_hold(self, *args, **kwargs): return None
        with self.assertRaisesRegex(RuntimeError, 'does not support'):
            h.hold_current_pose(LegacyRobot(), ['arm'], timeout=.01)
