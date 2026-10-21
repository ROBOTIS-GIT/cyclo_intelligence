"""Nearest arm alignment must not rewind the nominal gripper clock."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from main_runtime.rail_execution import RailExecution
from main_runtime.rail_trajectory import CubicChunk, RailTrajectory
from test_rail import Trace


def linear_tcp(q):
    return np.stack((q[..., 0], np.zeros_like(q[..., 0]), np.zeros_like(q[..., 0])),
                    axis=-1)[..., None, :]


def make_chunk():
    t = np.arange(16)/15
    actions = np.zeros((16, 8))
    actions[:, 0] = .2*t
    actions[:, 7] = t
    return CubicChunk(actions, 1/15, ['arm_right'])


def make_executor():
    e = RailExecution(None, threading.RLock(), Trace(), tcp_positions=linear_tcp)
    e.capture_actions = True
    e.thread = SimpleNamespace(is_alive=lambda: True)
    c = make_chunk()
    e.trajectory = RailTrajectory(c, 10., playback_rate=.25)
    e.last_request_id = 'old'
    e.pending = (c, 10., 'nearest', {'request_round_trip_s': .65})
    return e, c


def test_nearest_arm_phase_and_nominal_gripper_phase_are_independent():
    e, c = make_executor()
    previous = e.trajectory
    now = 10.7
    boundary = previous.state(now)
    action = e.tick(now, 'reach', ['arm_right'], 1, enforce_envelope=False)
    tr = e.trajectory
    assert tr is not previous
    assert tr.offset < .3  # Actual progress is near policy phase .175, not .7.
    assert tr.gripper_offset == pytest.approx(.7)
    coverage = .65+e.blend+2*e.output_dt
    assert tr.gripper_time_scale == pytest.approx(.3/coverage)
    assert action[7] == c.grippers(.7)[0]
    assert action[7] != c.grippers(tr.offset)[0]
    for actual, expected in zip(tr.state(now), boundary):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    event = [x for name, x in e.trace.events if name == 'rail_chunk_installed'][-1]
    assert event['algorithm'] == 'nearest_tcp_handoff_v3'
    assert event['alignment_shift_s'] < 0
    assert event['gripper_alignment'] == 'nominal_observation_clock'
    assert event['output_hz'] == 15.


def test_nearest_gripper_clock_does_not_anticipate_events_in_command_window():
    e, c = make_executor()
    now = 10.7
    action = e.tick(now, 'reach', ['arm_right'], 1, enforce_envelope=False)
    q, v, a, offsets = e.command_window(now, action)
    np.testing.assert_array_equal(q[:, 7], np.full(len(offsets), action[7]))
    np.testing.assert_array_equal(v[:, 7], 0.)
    np.testing.assert_array_equal(a[:, 7], 0.)
    assert offsets[0] == 0. and offsets[-1] == pytest.approx(.2)
    for elapsed in (0., .1, .2, .5, 2.):
        expected_phase = min(c.duration, .7+elapsed*e.trajectory.gripper_time_scale)
        assert e.trajectory.gripper_policy_time(elapsed) == pytest.approx(expected_phase)
        assert e.trajectory.action(now+elapsed, enforce_envelope=False)[7] == c.grippers(expected_phase)[0]


def test_expired_chunk_keeps_existing_arm_and_gripper_clock():
    e, c = make_executor()
    e.pending = (c, 9., 'expired', {'request_round_trip_s': 1.7})
    previous = e.trajectory
    e.tick(10.7, 'reach', ['arm_right'], 1, enforce_envelope=False)
    assert e.trajectory is previous
    assert e.last_request_id == 'old'
    assert e.response_budget_s == 0.
    e.reset()
    assert e.trajectory is None and e.pending is None


def test_separate_gripper_clock_preserves_intentional_opening_and_arm_derivatives():
    c = make_chunk()
    c.actions[:, 7] = np.r_[np.linspace(0., 1., 8), np.linspace(1., 0., 8)]
    tr = RailTrajectory(c, 10., offset=.1, playback_rate=.5,
                        gripper_offset=.5, gripper_playback_rate=.25)
    arm_only = RailTrajectory(c, 10., offset=.1, playback_rate=.5)
    assert tr.action(10., enforce_envelope=False)[7] > tr.action(12., enforce_envelope=False)[7]
    assert tr.action(100., enforce_envelope=False)[7] == 0.
    assert tr.gripper_policy_time(-1.) == .5
    for now in (10., 10.1, 10.4, 11., 12.):
        for actual, expected in zip(tr.state(now), arm_only.state(now)):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('offset,rate', [(-.1, 1.), (1., 1.), (.1, 0.), (.1, float('nan'))])
def test_invalid_independent_gripper_clock_fails(offset, rate):
    with pytest.raises(ValueError, match='gripper timing'):
        RailTrajectory(make_chunk(), 10., gripper_offset=offset, gripper_playback_rate=rate)
