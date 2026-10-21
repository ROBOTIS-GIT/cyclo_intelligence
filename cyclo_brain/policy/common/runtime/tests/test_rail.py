"""Synthetic RAIL contracts; never connects to a robot."""
import json
import threading
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
from test_control_loop import FakeRobot, ControlLoop, control_loop_module
from main_runtime.rail_trajectory import ARM_COLUMNS, CubicChunk, Quintic, RailTrajectory, alignment_offset, validate_rail_keys
from main_runtime.rail_execution import RailExecution


KEYS = ['action.arm_left', 'action.arm_right']


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **data):
        self.events.append((event, data))


def test_capture_records_raw_chunk_transition_and_rejected_sample():
    trace = Trace()
    executor = RailExecution(SimpleNamespace(get_action=lambda _: response(chunk().actions)),
                             threading.RLock(), trace, clock=lambda: 10.)
    executor.capture_actions = True
    executor._request('pick', KEYS, 1, executor.epoch, executor.dt)
    executor.thread = SimpleNamespace(is_alive=lambda: True)
    executor.tick(10., 'pick', KEYS, 1)
    events = dict(trace.events)
    np.testing.assert_array_equal(events['rail_raw_chunk']['actions'], chunk().actions)
    assert events['rail_raw_chunk']['request_id'] == events['rail_transition']['request_id']
    c = CubicChunk(np.zeros((16, 8)), 1/15, ['arm_right'])
    executor.trajectory = RailTrajectory(c, 10, .6, (np.zeros(7), np.full(7, 2.), np.zeros(7)))
    with pytest.raises(ValueError, match='position envelope'):
        executor.tick(10.02, 'pick', ['arm_right'], 1)
    assert trace.events[-1][0] == 'rail_sample'
    assert max(trace.events[-1][1]['positions']) > .01


def response(actions):
    return SimpleNamespace(success=True, chunk_size=len(actions), action_dim=actions.shape[1],
                           action_list=actions.ravel().tolist(), timing_json='{}')


def curved_chunk(length=16):
    t = np.linspace(0., 1., length)
    raw = np.stack([.1*np.sin(np.pi*t)+.01*j+.02*t for j in range(16)], axis=1)
    raw[:, ARM_COLUMNS] += (.004*(-1.)**np.arange(length))[:, None]
    raw[:, 7] = np.arange(length) % 2
    raw[:, 15] = np.arange(length) % 3
    return raw


def chunk():
    return CubicChunk(curved_chunk(16), 1/15, KEYS)


def test_cubic_recovery_and_derivatives():
    t = np.linspace(0, 1, 16)
    data = np.tile((.2 + .3*t - .1*t*t + .04*t**3)[:, None], (1, 16))
    c = CubicChunk(data, 1/15, KEYS)
    q,v,a = c.state(.4)
    np.testing.assert_allclose(q, .2+.3*.4-.1*.4**2+.04*.4**3)
    np.testing.assert_allclose(v, .3-.2*.4+.12*.4**2)
    np.testing.assert_allclose(a, -.2+.24*.4)


def test_quintic_boundary_conditions():
    start = tuple(np.full(14, n) for n in (.1, .2, -.3))
    end = tuple(np.full(14, n) for n in (.3, -.1, .2))
    d = Quintic(start, end, .2)
    for actual, expected in zip(d.state(0), start):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    for actual, expected in zip(d.state(.2), end):
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_trajectory_c2_splices_hold_and_raw_grippers():
    c = chunk(); r = RailTrajectory(c, 10)
    for boundary in (r.blend, r.tail_start, r.duration):
        for left,right in zip(r.state(10+boundary-1e-8), r.state(10+boundary+1e-8)):
            np.testing.assert_allclose(left,right,atol=1e-4)
    np.testing.assert_allclose(r.state(100)[1:], 0, atol=1e-10)
    np.testing.assert_allclose(r.action(100),r.action(200))
    np.testing.assert_array_equal(r.action(10.31)[[7,15]],c.actions[4,[7,15]])
    replacement = RailTrajectory(c,10.1,.15,r.state(10.1))
    for actual,expected in zip(replacement.state(10.1),r.state(10.1)):
        np.testing.assert_allclose(actual,expected)


@pytest.mark.parametrize('keys', [['arm_right'], KEYS])
def test_short_remaining_chunk_preserves_scaled_derivatives(keys):
    raw = curved_chunk()[:, :8*len(keys)]
    c = CubicChunk(raw, 1/15, keys)
    r = RailTrajectory(c, 10, offset=.85, blend=.2, playback_rate=.15)
    assert r.duration == pytest.approx(1.)
    assert r.blend == pytest.approx(.2)
    assert r.time_scale == pytest.approx(.15)
    assert 0 < r.time_scale <= 1
    # Position, velocity and acceleration must use the same slowed phase.
    elapsed = .45
    q, v, a = c.state(.85+elapsed*.15)
    for actual, expected in zip(r.state(10+elapsed), (q, v*.15, a*.15**2)):
        np.testing.assert_allclose(actual, expected)
    # Check derivatives numerically, not just the chain-rule implementation.
    eps = 1e-5
    q0, v0, a0 = r.state(10+elapsed)
    qm, vm, _ = r.state(10+elapsed-eps)
    qp, vp, _ = r.state(10+elapsed+eps)
    np.testing.assert_allclose((qp-qm)/(2*eps), v0, atol=1e-8)
    np.testing.assert_allclose((vp-vm)/(2*eps), a0, atol=1e-8)
    # Compare the endpoint conditions directly: a narrow blend can have high
    # jerk, so two samples separated by epsilon need not have equal acceleration.
    for actual, expected in zip(r.head.state(r.blend), r.reference_state(r.blend)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    for actual, expected in zip(r.tail.state(0.), r.reference_state(r.tail_start)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    for actual, expected in zip(r.tail.state(r.blend), (c.state(c.duration)[0], np.zeros(len(c.arm_columns)), np.zeros(len(c.arm_columns)))):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_array_equal(r.action(10+elapsed, enforce_envelope=False)[7::8],
                                  c.grippers(.85+elapsed*.15))
    np.testing.assert_allclose(r.state(100)[1:], 0, atol=1e-10)
    np.testing.assert_allclose(r.action(100, enforce_envelope=False),
                               r.action(200, enforce_envelope=False))


def test_short_chunk_keeps_policy_rate_and_splice_from_slowed_state():
    c = CubicChunk(curved_chunk(4), 1/15, KEYS)
    r = RailTrajectory(c, 10, blend=.2)
    assert r.duration == pytest.approx(.2)
    assert r.blend == pytest.approx(.2/3)
    prior = r.state(10.3)
    new = RailTrajectory(c, 10.3, .05, prior, blend=.2, playback_rate=.5)
    for actual, expected in zip(new.state(10.3), prior):
        np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize('options', [dict(blend=0), dict(blend=np.nan),
                                    dict(playback_rate=0), dict(playback_rate=-1),
                                    dict(playback_rate=np.inf), dict(playback_rate=np.nan),
                                    dict(offset=-.1), dict(offset=np.nan), dict(offset=1.)])
def test_invalid_playback_timing_rejected(options):
    with pytest.raises(ValueError):
        RailTrajectory(chunk(), 10, **options)


def test_alignment_expiry_and_stationary_tie():
    c = chunk(); q,v,a = c.state(.1)
    assert alignment_offset(c, 2, (q,v,a)) is None
    flat = CubicChunk(np.zeros((16, 8)), 1/15, ['arm_right'])
    assert alignment_offset(flat, .2, flat.state(.2)) == .2


@pytest.mark.parametrize('keys', [['arm_right'], KEYS])
def test_inherited_derivatives_cannot_publish_polynomial_overshoot(keys):
    c = CubicChunk(np.zeros((16, 8*len(keys))), 1/15, keys)
    zeros = np.zeros(7*len(keys))
    r = RailTrajectory(c, 0, .6, (zeros, zeros+2., zeros))
    # C2 continuity alone still creates movement from an all-zero policy chunk.
    assert np.max(r.state(.02)[0]) > .01
    with pytest.raises(ValueError, match='position envelope'):
        r.action(.02)
    assert np.max(r.action(.02, enforce_envelope=False)) > .01


@pytest.mark.parametrize('publish_to_robot', [False, True])
def test_rail_preview_and_robot_commands_follow_reference_without_clipping(publish_to_robot):
    robot = FakeRobot()
    preview_ready = threading.Event()
    publish_preview = robot.publish_action_preview
    def preview(*args):
        publish_preview(*args)
        preview_ready.set()
    robot.publish_action_preview = preview
    loop = ControlLoop(SimpleNamespace(get_action=lambda _: response(chunk().actions)))
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True,
                           publish_to_robot=publish_to_robot)
        loop._running = True
        executor = loop._rail
        executor.thread = SimpleNamespace(is_alive=lambda: True)
        executor.trace = Trace()
        executor.capture_actions = True
        raw = np.tile(np.linspace(0., 3., 16)[:, None], (1, 8))
        c = CubicChunk(raw, 1/15, ['arm_right'])
        executor.pending = (c, 10., 'first', {})
        for step in range(21):
            preview_ready.clear()
            now = 10.+step*.01
            with mock.patch.object(control_loop_module.time, 'monotonic', return_value=now):
                loop.tick()
            assert preview_ready.wait(1.)
            assert loop._running
            expected = executor.trajectory.action(now, enforce_envelope=False)
            np.testing.assert_array_equal(robot.previews[-1][0], expected)
            if publish_to_robot:
                np.testing.assert_array_equal(robot.commands[-1][0], expected)
        samples = [data for event, data in executor.trace.events if event == 'rail_sample']
        assert max(np.max(np.abs(sample['velocities'])) for sample in samples) > .6
        assert max(np.max(np.abs(sample['accelerations'])) for sample in samples) > 1.
        assert any(event == 'rail_output_sample' for event, _ in executor.trace.events)
        assert len(robot.commands) == (21 if publish_to_robot else 0)
    finally:
        loop.shutdown()


def test_rail_start_uses_feedback_transition_instead_of_initial_pose_sync():
    robot = FakeRobot()
    requester = SimpleNamespace(get_action=mock.Mock())
    loop = ControlLoop(requester)
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True,
                           publish_to_robot=True)
        assert not loop.start(initial_pose_sync=True)
        assert loop._running
        assert not robot.sync_targets and not robot.commands
        requester.get_action.assert_not_called()
    finally:
        loop.shutdown()


def test_rejected_rail_output_never_reaches_robot_and_requests_confirmed_hold():
    robot = FakeRobot()
    loop = ControlLoop(SimpleNamespace(get_action=lambda _: response(chunk().actions)))
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw', action_keys=['arm_right'], vla_rail_enabled=True,
                           publish_to_robot=True)
        loop._running = True
        loop._rail.tick = mock.Mock(side_effect=ValueError('position envelope'))
        with mock.patch.object(control_loop_module.threading, 'Thread') as thread:
            loop.tick()
            thread.assert_called_once_with(target=loop.pause,
                                           kwargs={'expected_generation': loop._generation},
                                           daemon=True)
        assert not loop._running
        assert loop._initial_pose_sync_hold_pending
        assert not robot.commands and not robot.previews
        assert loop.pause()  # Existing feedback-confirmed hold path must settle.
        assert robot.holds
    finally:
        loop.shutdown()


@pytest.mark.parametrize('keys', [['arm_right'], KEYS])
def test_alignment_uses_direction_score_and_latency_only_to_break_ties(keys):
    data = np.tile(np.linspace(0, 1, 16)[:, None], (1, 8*len(keys)))
    c = CubicChunk(data, 1/15, keys)
    # Many candidates lie ahead of the old position. Eq.10 ties them and the
    # latency estimate wins; there is no distance or derivative penalty.
    previous = (np.full(7*len(keys), .1), np.ones(7*len(keys)), np.zeros(7*len(keys)))
    offset = alignment_offset(c, .4, previous)
    assert offset == pytest.approx(.4)
    # The earliest nonnegative displacement has a larger direction score than
    # all candidates behind the old position, including the latency estimate.
    previous = (np.full(7*len(keys), .46), previous[1], previous[2])
    offset = alignment_offset(c, .4, previous)
    assert offset == pytest.approx(.46, abs=1e-12)
    trajectory = RailTrajectory(c, 10, offset, previous)
    for actual, expected in zip(trajectory.state(10), previous):
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_latency_spike_is_observed_without_automatic_retiming():
    now = [10.]
    latencies = iter([.4, .7])
    def infer(_task):
        snapshot = now[0]+.05
        now[0] += next(latencies)
        result = response(chunk().actions)
        result.timing_json = json.dumps({'sensor_snapshot_monotonic_s': snapshot})
        return result
    trace = Trace()
    executor = RailExecution(SimpleNamespace(get_action=infer), threading.RLock(), trace,
                             clock=lambda: now[0])
    executor.thread = SimpleNamespace(is_alive=lambda: True)
    executor._request('pick', KEYS, 1, executor.epoch, executor.dt)
    assert executor.pending[3]['request_ready_latency_s'] == pytest.approx(.4)
    assert executor.pending[1] == pytest.approx(10.05)
    executor.tick(now[0], 'pick', KEYS, 1)
    event = trace.events[-1][1]
    assert event['observation_age_s'] == pytest.approx(.35)
    assert event['time_scale'] == 1.
    executor._request('pick', KEYS, 1, executor.epoch, executor.dt)
    assert executor.pending[3]['request_ready_latency_s'] == pytest.approx(.7)
    previous = executor.trajectory
    executor.tick(now[0], 'pick', KEYS, 1, enforce_envelope=False)
    event = [data for name, data in trace.events if name == 'rail_handoff_decision'][-1]
    assert event['diagnostics']['observation_age_s'] == pytest.approx(.65)
    assert event['diagnostics']['nominal_policy_offset_s'] == pytest.approx(.65)
    # A late response cannot fit the configured head and tail. Keep the old
    # reference instead of compressing the bridge or slowing policy playback.
    assert event['reason'] == 'insufficient_horizon'
    assert executor.trajectory is previous
    assert executor.next_request == now[0]
    trajectory = executor.trajectory
    assert 0 < trajectory.blend <= .2
    assert trajectory.duration == pytest.approx(trajectory.chunk.duration-trajectory.offset)
    assert trajectory.time_scale == 1.


@pytest.mark.parametrize('data', [np.zeros((3,16)), np.full((16,16),np.nan),np.zeros((16,15))])
def test_bad_chunks_rejected(data):
    with pytest.raises(ValueError): CubicChunk(data,1/15,KEYS)


@pytest.mark.parametrize('keys,columns', [
    (['arm_left'], list(range(8))),
    (['action.arm_right'], list(range(8, 16))),
    (KEYS, list(range(16))),
    (list(reversed(KEYS)), list(range(8, 16)) + list(range(8))),
])
def test_rail_selected_arms_preserve_order_grippers_and_c2(keys, columns):
    raw = curved_chunk()[:, columns]
    c = CubicChunk(raw, 1/15, keys)
    r = RailTrajectory(c, 10)
    np.testing.assert_allclose(r.action(10), raw[0])
    assert r.action(10.31).shape == (len(columns),)
    np.testing.assert_array_equal(r.action(10.31)[7::8], raw[4, 7::8])
    # Each selected arm must match its independent single-arm trajectory.
    for index, key in enumerate(keys):
        single = RailTrajectory(CubicChunk(raw[:, index*8:(index+1)*8], 1/15, [key]), 10)
        np.testing.assert_allclose(r.action(10.31)[index*8:(index+1)*8], single.action(10.31))
    for boundary in (r.blend, r.tail_start, r.duration):
        for left, right in zip(r.state(10+boundary-1e-8), r.state(10+boundary+1e-8)):
            np.testing.assert_allclose(left, right, atol=1e-4)
    replacement = RailTrajectory(c, 10.1, .15, r.state(10.1))
    for actual, expected in zip(replacement.state(10.1), r.state(10.1)):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(r.state(100)[1:], 0, atol=1e-10)
    np.testing.assert_allclose(r.action(100), r.action(200))


@pytest.mark.parametrize('keys', [[], ['head'], ['arm_right', 'action.arm_right'],
                                 ['arm_left', 'mobile']])
def test_rail_invalid_action_keys_rejected(keys):
    with pytest.raises(ValueError, match='one or both unique'):
        validate_rail_keys(keys)


@pytest.mark.parametrize('width', [7, 9, 16])
def test_single_arm_rejects_mismatched_action_width(width):
    with pytest.raises(ValueError, match='8 values per'):
        CubicChunk(np.zeros((16, width)), 1/15, ['arm_right'])


@pytest.mark.parametrize('keys', [['arm_left'], ['action.arm_right']])
def test_single_arm_controller_and_executor(keys):
    raw = curved_chunk()[:, :8]
    loop = ControlLoop(SimpleNamespace(get_action=lambda _: response(raw)))
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=FakeRobot()):
            loop.configure('ffw', action_keys=keys, action_request_mode='async', vla_rail_enabled=True)
        executor = loop._rail
        executor._request('pick', keys, loop._generation, executor.epoch, executor.dt)
        assert executor.pending is not None
        executor.thread = SimpleNamespace(is_alive=lambda: True)
        with executor.lock:
            actual = executor.tick(executor.clock(), 'pick', keys, loop._generation)
        assert actual.shape == (8,)
        np.testing.assert_allclose(actual, raw[0])
    finally:
        loop.shutdown()


def test_request_uses_ordinary_inference_reset_discards_and_failure_holds():
    c=chunk(); trace=Trace(); now=[10.]
    requester=SimpleNamespace(get_action=mock.Mock(return_value=response(c.actions)))
    e=RailExecution(requester,threading.RLock(),trace,clock=lambda:now[0])
    e._request('pick',KEYS,1,e.epoch,e.dt)
    requester.get_action.assert_called_once_with('pick')
    e.thread=SimpleNamespace(is_alive=lambda:True)
    with e.lock: actual=e.tick(10.,'pick',KEYS,1)
    assert np.isfinite(actual).all()
    assert e.pending is None
    old=e.epoch; e.reset(); e._request('pick',KEYS,1,old,e.dt)
    assert e.pending is None
    e.trajectory=RailTrajectory(c,10)
    requester.get_action.side_effect=RuntimeError('offline')
    e._request('pick',KEYS,1,e.epoch,e.dt)
    np.testing.assert_allclose(e.trajectory.action(100),e.trajectory.action(200))
    assert trace.events[-1][1]['success'] is False


def test_executor_rejects_expired_response_without_losing_old_schedule():
    c=chunk(); e=RailExecution(None,threading.RLock(),Trace())
    e.thread=SimpleNamespace(is_alive=lambda:True)
    e.trajectory=RailTrajectory(c,10); previous=e.trajectory
    e.pending=(c,1,'r',{})
    with e.lock: e.tick(10.2,'pick',KEYS,1)
    assert e.trajectory is previous


def test_control_loop_optin_mutual_exclusion_and_reset():
    loop=ControlLoop(SimpleNamespace(get_action=lambda _: response(chunk().actions)))
    try:
        with mock.patch.object(control_loop_module,'RobotClient',return_value=FakeRobot()):
            loop.configure(robot_type='ffw',action_keys=KEYS,action_request_mode='async',vla_rail_enabled=True)
            assert loop._rail is not None
            assert loop._tick_period() == 1/loop._inference_hz
            for invalid in ({'action_request_mode':'sync'},):
                options=dict(robot_type='ffw',action_keys=KEYS,action_request_mode='async',vla_rail_enabled=True)
                options.update(invalid)
                with pytest.raises(ValueError): loop.configure(**options)
            loop.configure(robot_type='ffw',action_keys=KEYS,action_request_mode='async')
            assert loop._rail is None
    finally: loop.shutdown()



def test_raw_policy_chunk_is_fitted_directly_without_prefilter():
    raw = curved_chunk(16)
    requester = SimpleNamespace(get_action=mock.Mock(return_value=response(raw)))
    executor = RailExecution(requester, threading.RLock(), Trace(), clock=lambda: 10.)
    executor._request('pick', KEYS, 1, executor.epoch, executor.dt)
    np.testing.assert_array_equal(executor.pending[0].actions, raw)
    requester.get_action.assert_called_once_with('pick')


def test_tick_outputs_rail_trajectory_without_second_regulator():
    executor = RailExecution(None, threading.RLock(), Trace())
    executor.thread = SimpleNamespace(is_alive=lambda: True)
    executor.trajectory = RailTrajectory(chunk(), 10)
    for now in np.linspace(10, 11.5, 80):
        with executor.lock:
            action = executor.tick(now, 'pick', KEYS, 1)
        np.testing.assert_array_equal(action, executor.trajectory.action(now))


@pytest.mark.parametrize('reset', ['pause', 'deconfigure', 'instruction', 'publish_mode'])
def test_controller_lifecycle_invalidates_inflight_rail_chunk(reset):
    loop = ControlLoop(SimpleNamespace(get_action=lambda _: response(chunk().actions)))
    with mock.patch.object(control_loop_module, 'RobotClient', return_value=FakeRobot()):
        loop.configure('ffw', action_keys=KEYS, vla_rail_enabled=True)
    executor = loop._rail
    epoch = executor.epoch
    executor.trajectory = RailTrajectory(chunk(), 10)
    executor.pending = (chunk(), 10, 'r', {})
    if reset == 'pause':
        assert loop.pause()
    elif reset == 'deconfigure':
        loop.deconfigure()
    elif reset == 'instruction':
        loop.set_task_instruction('new task')
    else:
        loop.set_publish_to_robot(True)
    assert executor.epoch > epoch
    assert executor.trajectory is None and executor.pending is None
    executor._request('old task', KEYS, loop._generation, epoch, executor.dt)
    assert executor.pending is None
    loop.shutdown()


@pytest.mark.parametrize('keys', [['arm_right'], KEYS])
def test_replanning_preserves_executed_c2_without_arm_output_lag(keys):
    width = 8*len(keys)
    raw = np.tile(np.linspace(0., .3, 16)[:, None], (1, width))
    c = CubicChunk(raw, 1/15, keys)
    trace = Trace()
    e = RailExecution(None, threading.RLock(), trace)
    e.capture_actions = True
    e.thread = SimpleNamespace(is_alive=lambda: True)
    initial = np.full(width, -.12)
    e.pending = (c, 10., 'first', {})
    command = e.tick(10., 'pick', keys, 1, enforce_envelope=False, initial_positions=initial)
    out = e.prepare_action(command, 10., 1, initial)
    np.testing.assert_allclose(out, initial)
    for tick in range(1, 201):
        now = 10.+tick*.01
        old = e.trajectory
        if tick % 50 == 0:
            changed = CubicChunk(raw * (-1 if tick % 100 else 1), 1/15, keys)
            e.pending = (changed, now-.4, str(tick), {})
        command = e.tick(now, 'pick', keys, 1, enforce_envelope=False)
        out = e.prepare_action(command, now, 1)
        np.testing.assert_array_equal(out[c.arm_columns], command[c.arm_columns])
        if old is not e.trajectory:
            for actual, previous in zip(e.trajectory.state(now), old.state(now)):
                np.testing.assert_allclose(actual, previous, atol=1e-10)
            installed = [data for event, data in trace.events if event == 'rail_chunk_installed'][-1]
            assert installed['nominal_policy_offset_s'] == pytest.approx(installed['observation_age_s'])
    for event, data in trace.events:
        if event == 'rail_output_sample':
            np.testing.assert_array_equal(np.asarray(data['action'])[c.arm_columns],
                                          np.asarray(data['target'])[c.arm_columns])


def test_arm_and_gripper_targets_pass_through_with_output_guards():
    e = RailExecution(None, threading.RLock(), Trace())
    e.capture_actions = True
    e.thread = SimpleNamespace(is_alive=lambda: True)
    c = CubicChunk(np.tile(np.linspace(0., 3., 16)[:, None], (1, 8)), 1/15, ['arm_right'])
    e.pending = (c, 10., 'first', {})
    initial = np.zeros(8)
    action = e.tick(10., 'pick', ['arm_right'], 1, enforce_envelope=False, initial_positions=initial)
    e.prepare_action(action, 10., 1, initial)
    assert e.trajectory.duration == pytest.approx(c.duration)
    for step in range(1, 21):
        now = 10.+step*.01
        action = e.tick(now, 'pick', ['arm_right'], 1, enforce_envelope=False)
        np.testing.assert_array_equal(e.prepare_action(action, now, 1), action)
    with pytest.raises(ValueError, match='interval'):
        e.prepare_action(action, 11., 1)
