"""Synthetic RAIL contracts; never connects to a robot."""
import threading
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
from test_control_loop import FakeRobot, ControlLoop, control_loop_module
from main_runtime.rail_trajectory import ARM_COLUMNS, CubicChunk, DualQuintic, RailTrajectory, alignment_offset
from main_runtime.rail_execution import RailExecution


KEYS = ['action.arm_left', 'action.arm_right']


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **data):
        self.events.append((event, data))


def response(actions):
    return SimpleNamespace(success=True, chunk_size=len(actions), action_dim=16,
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


def test_dual_quintic_boundary_conditions():
    start = tuple(np.full(14, n) for n in (.1, .2, -.3))
    end = tuple(np.full(14, n) for n in (.3, -.1, .2))
    d = DualQuintic(start, end, .2)
    for actual, expected in zip(d.state(0), start):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    for actual, expected in zip(d.state(.2), end):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    for left,right in zip(d.left.state(.1), d.right.state(0)):
        np.testing.assert_allclose(left,right,atol=1e-10)


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


def test_alignment_expiry_and_stationary_tie():
    c = chunk(); q,v,a = c.state(.1)
    assert alignment_offset(c, .2, (q,v*0,a)) == .2
    assert alignment_offset(c, 2, (q,v,a)) is None


@pytest.mark.parametrize('data', [np.zeros((3,16)), np.full((16,16),np.nan),np.zeros((16,15))])
def test_bad_chunks_rejected(data):
    with pytest.raises(ValueError): CubicChunk(data,1/15,KEYS)


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
            assert loop._tick_period() == 1/loop._control_hz
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
