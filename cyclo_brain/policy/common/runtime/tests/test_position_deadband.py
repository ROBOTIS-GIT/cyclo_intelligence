"""Left-arm noise suppression, insertion isolation, and lifecycle checks."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_runtime.position_deadband import PositionDeadband, left_arm_deadband_enabled
from main_runtime.tt_rtc_timeline import TTActionTimeline
from test_playground_control_loop import ControlLoop, FakeRobot


def make_filter(indices=range(7), radius=.003):
    return PositionDeadband(indices, radius, lower=[-3.] * len(indices), upper=[3.] * len(indices))


def test_static_left_noise_is_held_while_right_micro_descent_and_grippers_pass_exactly():
    f = make_filter()
    f.step(np.zeros(16))
    for i in range(1000):
        value = np.zeros(16)
        value[:7] = .002 * np.sin(i * .2)
        value[8:15] = -i * 1e-6
        value[7], value[15] = .2 + i * 1e-5, .3
        before = value.copy()
        out = f.step(value)
        np.testing.assert_array_equal(out[:7], 0)
        np.testing.assert_array_equal(out[7:], value[7:])
        np.testing.assert_array_equal(value, before)


def test_slow_left_motion_accumulates_and_release_does_not_snap():
    f = make_filter([0]); f.step([0.])
    previous = 0.
    for target in np.linspace(0, .02, 2001):
        out = f.step([target])[0]
        assert abs(target - out) <= .003 + 1e-12
        assert -1e-12 <= out - previous <= .00001 + 1e-12
        previous = out
    assert out == pytest.approx(.017)
    for _ in range(100):
        assert f.step([.02])[0] == pytest.approx(.017)


def test_reversal_has_bounded_error_without_boundary_chatter():
    f = make_filter([0]); f.step([0.])
    assert f.step([.01])[0] == pytest.approx(.007)
    for target in [.009, .008, .005, .0041]:
        assert f.step([target])[0] == pytest.approx(.007)
    assert f.step([.00399])[0] == pytest.approx(.00699)
    assert f.step([-.01])[0] == pytest.approx(-.007)


def test_invalid_inputs_fail_without_corrupting_hold_and_clear_reanchors():
    f = make_filter([0]); f.step([.1])
    for value in [[np.nan], [np.inf], [3.1], [-3.1], [.1, .2]]:
        with pytest.raises(ValueError):f.step(value)
    assert f.step([.101])[0] == pytest.approx(.1)
    f.clear()
    assert f.step([-.2])[0] == pytest.approx(-.2)
    for radius in [0, -1, np.nan, np.inf, .021]:
        with pytest.raises(ValueError):make_filter([0], radius)


class NamedRobot(FakeRobot):
    def get_action_joint_limits(self, keys):
        names = []
        for key in keys:
            side = {'arm_left': 'l', 'arm_right': 'r'}[key]
            names += [f'arm_{side}_joint{i}' for i in range(1, 8)] + [f'gripper_{side}_joint1']
        return [dict(name=n, lower=-3., upper=3., velocity=4.8) for n in names]


def make_loop(keys=('arm_left', 'arm_right'), physical=False, enabled=True):
    robot = NamedRobot()
    loop = ControlLoop(requester=object())
    with patch('main_runtime.playground_control_loop.RobotClient', return_value=robot), patch.dict(
        os.environ, {'CYCLO_LEFT_ARM_DEADBAND': str(enabled).lower(),
                     'CYCLO_LEFT_ARM_DEADBAND_RAD': '.003', 'CYCLO_SG2_ARM_SMOOTHING': 'false'}):
        loop.configure('ffw_sg2_rev1', action_keys=list(keys),
                       action_request_mode='tt_rtc' if len(keys) == 2 else 'async',
                       tt_rtc_horizon=32, tt_rtc_action_dim=8 * len(keys), publish_to_robot=physical)
    loop._running = True
    return loop, robot


@pytest.mark.parametrize('keys', [('arm_left', 'arm_right'), ('arm_right', 'arm_left')])
def test_joint_names_resolve_left_only_with_reordered_groups(keys):
    loop, _ = make_loop(keys)
    offset = 8 * keys.index('arm_left')
    np.testing.assert_array_equal(loop._left_arm_deadband.indices, np.arange(offset, offset + 7))
    right, _ = make_loop(('arm_right',))
    assert right._left_arm_deadband is None


@pytest.mark.parametrize('physical', [False, True])
def test_pipeline_preserves_rtc_prefix_and_right_commands_in_preview_and_robot(physical):
    loop, robot = make_loop(physical=physical)
    chunk = np.zeros((32, 16))
    chunk[:, :7] = (.002 * np.sin(np.arange(32)))[:, None]
    chunk[:, 8:15] = (-np.arange(32) * .0001)[:, None]
    chunk[:, 7], chunk[:, 15] = .3, .4
    loop._processor.push_actions(chunk)
    reference = TTActionTimeline(); reference.push_actions(chunk)
    with patch.object(loop, '_should_request_actions', return_value=False):
        for _ in range(180):
            planned = reference.pop_action()
            loop.tick()
            np.testing.assert_array_equal(loop._processor.peek_actions(), reference.peek_actions())
            preview = robot.previews[-1][0]
            np.testing.assert_array_equal(preview[:7], 0)
            np.testing.assert_array_equal(preview[7:], planned[7:])
            if physical:np.testing.assert_array_equal(robot.commands[-1][0], preview)
    assert len(robot.commands) == (180 if physical else 0)


def test_pause_and_deconfigure_clear_hold_and_disabled_mode_is_unchanged():
    loop, _ = make_loop()
    f = loop._left_arm_deadband
    f.step(np.zeros(16)); f.step(np.full(16, .001))
    loop.pause()
    np.testing.assert_array_equal(f.step(np.full(16, -.2)), -.2)
    loop.deconfigure(); assert loop._left_arm_deadband is None
    disabled, _ = make_loop(enabled=False); assert disabled._left_arm_deadband is None
    with patch.dict(os.environ, {}, clear=True):assert not left_arm_deadband_enabled()


def test_bad_target_pauses_before_publishing():
    loop, robot = make_loop(physical=True)
    chunk = np.zeros((32, 16)); chunk[:, 0] = 3.1
    loop._processor.push_actions(chunk)
    loop.tick()
    assert not loop._running
    assert not robot.commands and not robot.previews
    assert 'joint limits' in loop.tt_rtc_failure_reason
