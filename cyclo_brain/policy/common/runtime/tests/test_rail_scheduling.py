"""Deterministic scheduling, sensor age and ROS command-window regressions."""
import json
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from test_control_loop import ControlLoop, FakeRobot, control_loop_module
from test_rail import Trace, response
from main_runtime.rail_execution import RailExecution
from main_runtime.rail_trajectory import CubicChunk, Quintic, RailTrajectory


def executor(length=16, policy_hz=15., interpolation_hz=None):
    c = CubicChunk(np.tile(np.linspace(0., .2, length)[:, None], (1, 8)), 1/policy_hz, ['arm_right'])
    e = RailExecution(None, threading.RLock(), Trace(), policy_hz=policy_hz,
                      interpolation_hz=interpolation_hz)
    e.thread = SimpleNamespace(is_alive=lambda: True)
    e.pending = (c, 10., 'first', {})
    e.tick(10., 'pick', ['arm_right'], 1, initial_positions=np.zeros(8))
    return e, c


@pytest.mark.parametrize('length', [4, 8, 16])
def test_paper_schedule_preserves_policy_duration_and_requests_without_wait(length):
    e, c = executor(length)
    assert e.next_request == pytest.approx(10.)
    assert e.trajectory.duration == pytest.approx(c.duration)
    assert e.trajectory.time_scale == pytest.approx(1.)


@pytest.mark.parametrize('policy_hz,interpolation_hz,rate', [(15., 15., 1.), (30., 15., 2.),
                                                          (15., 30., .5), (30., None, 1.)])
def test_explicit_interpolation_ratio_uses_policy_output_rate(policy_hz, interpolation_hz, rate):
    e, c = executor(policy_hz=policy_hz, interpolation_hz=interpolation_hz)
    assert e.playback_rate == rate
    assert e.trajectory.duration == pytest.approx(c.duration/rate)
    assert e.trajectory.time_scale == rate
    # Inside the cubic region, wall time advances the configured policy phase.
    elapsed = .5*c.duration/rate
    q, v, a = c.state(.5*c.duration)
    for actual, target in zip(e.trajectory.state(10.+elapsed), (q, v*rate, a*rate**2)):
        np.testing.assert_allclose(actual, target, atol=1e-10)


def test_fast_response_installs_immediately_during_head_and_preserves_c2():
    e, c = executor()
    old = e.trajectory
    e.pending = (c, 10.01, 'second', {})
    handoff = 10.02
    expected = old.state(handoff)
    e.tick(handoff, 'pick', ['arm_right'], 1)
    assert e.pending is None
    assert e.trajectory is not old
    for actual, boundary in zip(e.trajectory.state(handoff), expected):
        np.testing.assert_allclose(actual, boundary, atol=1e-10)
    event = [x for name, x in e.trace.events if name == 'rail_chunk_installed'][-1]
    assert event['nominal_policy_offset_s'] == pytest.approx(.01)
    assert event['alignment_shift_s'] == pytest.approx(event['offset_s']-.01)


@pytest.mark.parametrize('source,expected', [('local_monotonic_receive', 9.95), ('cache_read_fallback', 10.)])
def test_sensor_receive_timestamp_can_precede_request_dispatch(source, expected):
    _, c = executor()
    result = response(c.actions)
    result.timing_json = json.dumps({'sensor_snapshot_monotonic_s': 9.95, 'timestamp_source': source})
    e = RailExecution(SimpleNamespace(get_action=lambda _: result), threading.RLock(), Trace(), clock=lambda: 10.)
    timing = json.loads(result.timing_json)
    timing['clock_id'] = e.clock_id
    result.timing_json = json.dumps(timing)
    e._request('pick', ['arm_right'], 1, e.epoch, e.dt)
    assert e.pending[1] == expected


@pytest.mark.parametrize('boundary', ['head_midpoint', 'head_end', 'tail_start', 'tail_midpoint', 'end'])
def test_command_window_preserves_quintic_reference_and_does_not_anticipate_gripper(boundary):
    e, c = executor()
    trajectory = e.trajectory
    crossing = {'head_midpoint': trajectory.blend/2, 'head_end': trajectory.blend,
                'tail_start': trajectory.tail_start,
                'tail_midpoint': trajectory.tail_start+trajectory.blend/2, 'end': trajectory.duration}[boundary]
    now = trajectory.start_time+crossing-.015
    action = e.trajectory.action(now, enforce_envelope=False)
    action[7] = -.4
    q, v, a, offsets = e.command_window(now, action)
    assert offsets[0] == 0
    assert offsets[-1] == pytest.approx(.2)
    assert np.all(np.diff(offsets) > 0)
    np.testing.assert_array_equal(q[:, 7], -.4)
    np.testing.assert_array_equal(v[:, 7], 0.)
    np.testing.assert_array_equal(a[:, 7], 0.)
    assert np.any(np.abs(v[:, :7]) > 0)
    # Reproduce the receiving controller's quintic interpolation of q/v/a.
    for i in range(len(offsets)-1):
        interval = offsets[i+1]-offsets[i]
        interpolated = Quintic((q[i, :7], v[i, :7], a[i, :7]),
                               (q[i+1, :7], v[i+1, :7], a[i+1, :7]), interval)
        for fraction in (.25, .5, .75):
            expected = e.trajectory.state(now+offsets[i]+fraction*interval)
            for actual, target in zip(interpolated.state(fraction*interval), expected):
                np.testing.assert_allclose(actual, target, atol=2e-7)


def test_terminal_window_holds_at_fitted_final_target_with_zero_derivatives():
    e, c = executor()
    raw = c.actions.copy()
    raw[-1, :7] += .02
    c = CubicChunk(raw, c.dt, ['arm_right'])
    e.trajectory = RailTrajectory(c, 10.)
    now = e.trajectory.end+.01
    q, v, a, offsets = e.command_window(now, e.trajectory.action(now))
    expected = np.r_[c.state(c.duration)[0], c.grippers(c.duration)]
    np.testing.assert_allclose(q, np.tile(expected, (len(offsets), 1)), atol=1e-10)
    np.testing.assert_allclose(v, 0., atol=1e-9)
    np.testing.assert_allclose(a, 0., atol=1e-8)


def test_control_loop_publishes_timed_trajectory_before_preview_outside_lock():
    robot = FakeRobot()
    loop = ControlLoop(SimpleNamespace(get_action=mock.Mock()))
    events = []
    preview_ready = threading.Event()
    publish = robot.publish_action_trajectory
    def command(*args):
        events.append('command')
        publish(*args)
    def preview(*_):
        events.append('preview')
        assert not loop._lock._is_owned()
        preview_ready.set()
    robot.publish_action_trajectory = command
    robot.publish_action_preview = preview
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True, publish_to_robot=True)
        loop._running = True
        loop._rail.thread = SimpleNamespace(is_alive=lambda: True)
        _, c = executor()
        loop._rail.pending = (c, 10., 'first', {})
        with mock.patch.object(control_loop_module.time, 'monotonic', return_value=10.):
            loop.tick()
        assert preview_ready.wait(1.)
        assert events == ['command', 'preview']
        assert len(robot.trajectories) == 1
        q, v, a, offsets = robot.trajectories[0]
        np.testing.assert_allclose(q[0], 0.)
        np.testing.assert_allclose(v[0], 0.)
        np.testing.assert_allclose(a[0], 0.)
        assert offsets[-1] > 0
        loop.pause()
        assert robot.holds
        count = len(robot.trajectories)
        loop.tick()
        assert len(robot.trajectories) == count
    finally:
        loop.shutdown()


def test_foreign_monotonic_sensor_clock_uses_dispatch_fallback():
    _, c = executor()
    result = response(c.actions)
    result.timing_json = json.dumps({'sensor_snapshot_monotonic_s': 9.95,
                                    'timestamp_source': 'local_monotonic_receive',
                                    'clock_id': 'different-linux-boot'})
    e = RailExecution(SimpleNamespace(get_action=lambda _: result), threading.RLock(), Trace(), clock=lambda: 10.)
    e._request('pick', ['arm_right'], 1, e.epoch, e.dt)
    assert e.pending[1] == 10.
    assert e.pending[3]['rail_alignment_reference'] == 'clock_id_mismatch_dispatch_fallback'


def test_slow_preview_cannot_block_next_control_tick():
    robot = FakeRobot()
    loop = ControlLoop(SimpleNamespace(get_action=mock.Mock()))
    entered, release, second_tick_done = threading.Event(), threading.Event(), threading.Event()
    def preview(*_):
        entered.set()
        release.wait(2.)
    robot.publish_action_preview = preview
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True, publish_to_robot=True)
        loop._running = True
        loop._rail.thread = SimpleNamespace(is_alive=lambda: True)
        _, c = executor()
        loop._rail.pending = (c, 10., 'first', {})
        with mock.patch.object(control_loop_module.time, 'monotonic', return_value=10.):
            loop.tick()
        assert entered.wait(1.)
        def tick_again():
            with mock.patch.object(control_loop_module.time, 'monotonic', return_value=10.01):
                loop.tick()
            second_tick_done.set()
        thread = threading.Thread(target=tick_again)
        thread.start()
        assert second_tick_done.wait(1.)
        assert len(robot.trajectories) == 2
        thread.join(1.)
    finally:
        release.set()
        loop.shutdown()


@pytest.mark.parametrize('start', [10., 1e6])
@pytest.mark.parametrize('distance_to_knot', [1e-9, 5e-5])
def test_command_window_merges_near_coincident_knots_without_acceleration_spikes(start, distance_to_knot):
    e, _ = executor()
    t = np.linspace(0., 1., 16)
    c = CubicChunk(np.tile((.1*t+.01*t*t)[:, None], (1, 8)), 1/15, ['arm_right'])
    e.trajectory = RailTrajectory(c, start)
    now = start+e.trajectory.blend-distance_to_knot
    q, v, a, offsets = e.command_window(now, e.trajectory.action(now, enforce_envelope=False))
    assert np.min(np.diff(offsets)) >= min(1e-4, e.output_dt*.01)-1e-12
    for i in range(len(offsets)-1):
        interval = offsets[i+1]-offsets[i]
        segment = Quintic((q[i, :7], v[i, :7], a[i, :7]),
                          (q[i+1, :7], v[i+1, :7], a[i+1, :7]), interval)
        for fraction in (.25, .5, .75):
            state = segment.state(fraction*interval)
            expected = e.trajectory.state(now+offsets[i]+fraction*interval)
            np.testing.assert_allclose(state[0], expected[0], atol=1e-7)
            np.testing.assert_allclose(state[1], expected[1], atol=1e-5)
            np.testing.assert_allclose(state[2], expected[2], atol=.005)


@pytest.mark.parametrize('rate', [.5, 1., 2.])
def test_observation_latency_uses_wall_time_even_after_terminal_hold(rate):
    # Keep a complete head and tail available even at the 2x playback ratio.
    e, c = executor(length=32, interpolation_hz=15./rate)
    now = e.trajectory.end+1.
    # A fresh prediction starts from the held position. Replaying the original
    # origin-to-target chunk here would now be a rejected backward reset.
    actions = c.actions.copy()
    actions[:, :7] += c.state(c.duration)[0]
    fresh = CubicChunk(actions, c.dt, ['arm_right'])
    e.pending = (fresh, now-.15, 'after_hold', {})
    e.tick(now, 'pick', ['arm_right'], 1)
    event = [x for name, x in e.trace.events if name == 'rail_chunk_installed'][-1]
    assert event['request_id'] == 'after_hold'
    assert event['nominal_policy_offset_s'] == pytest.approx(.15*rate)
    assert event['observation_age_s'] == pytest.approx(.15)


def test_control_hz_does_not_change_rail_output_or_explicit_playback_ratio():
    loop = ControlLoop(SimpleNamespace(get_action=mock.Mock()), rail_interpolation_hz=15.)
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=FakeRobot()):
            for hz in (100., 200.):
                loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True, control_hz=hz)
                assert loop._rail.playback_rate == 1.
                assert loop._tick_period() == pytest.approx(1/15)
    finally:
        loop.shutdown()


@pytest.mark.parametrize('hz', [0., -1., np.inf, np.nan])
def test_invalid_interpolation_frequency_is_rejected(hz):
    with pytest.raises(ValueError):
        ControlLoop(SimpleNamespace(get_action=mock.Mock()), rail_interpolation_hz=hz)
    with pytest.raises(ValueError):
        RailExecution(None, threading.RLock(), Trace(), interpolation_hz=hz)


@pytest.mark.parametrize('policy_hz', [5., 15., 30.])
@pytest.mark.parametrize('control_hz', [50., 200.])
def test_policy_rate_publishing_covers_next_tick_without_changing_curve_speed(policy_hz, control_hz):
    robot = FakeRobot()
    loop = ControlLoop(SimpleNamespace(get_action=mock.Mock()))
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True,
                           publish_to_robot=True, inference_hz=policy_hz, control_hz=control_hz)
        loop._running = True
        e = loop._rail
        e.capture_actions = True
        e.trace = Trace()
        e.thread = SimpleNamespace(is_alive=lambda: True)
        raw = np.tile(np.linspace(0., .2, 16)[:, None], (1, 8))
        c = CubicChunk(raw, 1/policy_hz, ['arm_right'])
        e.pending = (c, 10., 'first', {})
        assert loop._tick_period() == pytest.approx(1/policy_hz)
        for step in range(5):
            with mock.patch.object(control_loop_module.time, 'monotonic', return_value=10.+step/policy_hz):
                loop.tick()
        assert loop._running
        assert len(robot.trajectories) == 5
        assert e.trajectory.duration == pytest.approx(c.duration)
        for step, (q, v, a, offsets) in enumerate(robot.trajectories):
            assert offsets[-1] == pytest.approx(3/policy_hz)
            # Adjacent windows describe the same continuous curve at handoff.
            if step:
                pq, pv, pa, po = robot.trajectories[step-1]
                index = np.argmin(abs(po-1/policy_hz))
                assert po[index] == pytest.approx(1/policy_hz)
                # ROS Duration rounds offsets to ns; jerk amplifies that tiny
                # time difference in the acceleration comparison.
                for previous, current, atol in zip((pq, pv, pa), (q, v, a), (1e-9, 1e-8, 5e-7)):
                    np.testing.assert_allclose(previous[index, :7], current[0, :7], atol=atol)
        published = [v for k, v in e.trace.events if k == 'command_published']
        assert len(published) == 5
        assert all(x['output_hz'] == policy_hz for x in published)
        for event in published[1:]:
            assert event['publish_interval_s'] == pytest.approx(1/policy_hz)
        # Normal policy ticks must pass, but a gap beyond the transmitted horizon fails.
        with pytest.raises(ValueError, match='interval invalid or stale'):
            e.prepare_action(np.zeros(8), 10.+8/policy_hz, 1)
    finally:
        loop.shutdown()


def test_reset_updates_policy_output_timing_and_default_playback_together():
    e, _ = executor()
    e.reset(5.)
    assert e.output_hz == 5.
    assert e.output_dt == pytest.approx(.2)
    assert e.playback_rate == 1.
    initial = np.zeros(8)
    e.output_guard.update(initial, 10., initial)
    e.output_guard.update(initial, 10.2)
    e.reset(30.)
    assert e.output_hz == 30.
    assert e.output_dt == pytest.approx(1/30)
    assert e.playback_rate == 1.
    assert e.output_guard.positions is None


@pytest.mark.parametrize('selected_scale', [.3, .5, .8])
@pytest.mark.parametrize('head_s,tail_s', [(.2, .1), (.24, .13)])
@pytest.mark.parametrize('head_mode', ['single', 'dual'])
def test_policy_rate_windows_preserve_retimed_curve_and_independent_tail(
        selected_scale, head_s, tail_s, head_mode):
    """JTC q/v/a reconstruction must preserve time-warped head, curve and tail."""
    times = np.arange(16)/15
    arms = (times[:, None]*np.linspace(.1, .3, 7)
            + times[:, None]**2*np.linspace(-.05, .04, 7))
    chunk = CubicChunk(np.column_stack((arms, times)), 1/15, ['arm_right'])
    previous = (np.full(7, .05), np.linspace(.04, .08, 7), np.linspace(-.01, .02, 7))
    trajectory = RailTrajectory(chunk, 10000., .665, previous, head_s,
                                selected_scale, head_mode=head_mode,
                                preserve_blend=True, tail_blend=tail_s)
    execution = RailExecution(None, threading.RLock(), Trace(), policy_hz=15.)
    execution.trajectory = trajectory
    assert execution.output_hz == 15.
    assert execution.output_dt == pytest.approx(1/15)
    assert trajectory.duration == pytest.approx((1.-.665)/selected_scale)
    assert trajectory.tail_start == pytest.approx(trajectory.duration-tail_s)
    # The selected rate transforms only the incoming reference. Existing q/v/a
    # remain exactly the handoff boundary even when its old rate was different.
    for actual, expected in zip(trajectory.state(trajectory.start_time), previous):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    middle = (head_s+trajectory.tail_start)/2
    incoming = chunk.state(.665+middle*selected_scale)
    for actual, expected in zip(trajectory.reference_state(middle),
                                (incoming[0], incoming[1]*selected_scale,
                                 incoming[2]*selected_scale**2)):
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    starts = (0., head_s/2-.025, head_s-.025,
              trajectory.tail_start-.035,
              trajectory.tail_start+tail_s/2-.025,
              trajectory.duration-.025, trajectory.duration+.1)
    for elapsed in starts:
        now = trajectory.start_time+elapsed
        action = trajectory.action(now, enforce_envelope=False)
        q, v, a, offsets = execution.command_window(now, action)
        assert offsets[0] == 0.
        assert offsets[-1] == pytest.approx(.2)
        assert np.all(np.diff(offsets) > 0)
        np.testing.assert_array_equal(q[:, 7], np.full(len(offsets), action[7]))
        np.testing.assert_array_equal(v[:, 7], 0.)
        np.testing.assert_array_equal(a[:, 7], 0.)
        # The live JTC uses quintic Hermite interpolation for transmitted q/v/a.
        # Reconstruct each interval to catch missing/wrong tail midpoint knots.
        for index in range(len(offsets)-1):
            duration = offsets[index+1]-offsets[index]
            segment = Quintic((q[index, :7], v[index, :7], a[index, :7]),
                              (q[index+1, :7], v[index+1, :7], a[index+1, :7]), duration)
            for fraction in (.13, .37, .61, .89):
                local = duration*fraction
                expected = trajectory.state(now+offsets[index]+local)
                for reconstructed, direct, atol in zip(segment.state(local), expected,
                                                        (1e-8, 1e-6, 2e-4)):
                    np.testing.assert_allclose(reconstructed, direct, atol=atol, rtol=1e-7)
    for elapsed in (trajectory.duration+.1, trajectory.duration+5.):
        state = trajectory.state(trajectory.start_time+elapsed)
        np.testing.assert_allclose(state[0], chunk.state(chunk.duration)[0], atol=1e-12)
        np.testing.assert_allclose(state[1:], 0., atol=1e-10)
