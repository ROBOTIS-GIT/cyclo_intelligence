"""Admission, finite fallback and tracking diagnostics without robot motion."""
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from test_rail import Trace
from main_runtime.rail_execution import RailExecution
from main_runtime.rail_trajectory import CubicChunk, RailTrajectory
from test_control_loop import ControlLoop, FakeRobot, control_loop_module


KEYS = ['arm_right']


def chunk(start=0., speed=.2):
    t = np.arange(16)/15
    actions = np.tile((start+speed*t)[:, None], (1, 8))
    actions[:, 7] = 0.
    return CubicChunk(actions, 1/15, KEYS)


def execution():
    e = RailExecution(None, threading.RLock(), Trace())
    e.thread = SimpleNamespace(is_alive=lambda: True)
    e.capture_actions = True
    e.trajectory = RailTrajectory(chunk(), 10.)
    e.last_request_id = 'accepted'
    return e


def test_incompatible_response_keeps_reference_and_finite_terminal_hold():
    e = execution()
    previous = e.trajectory
    e.pending = (chunk(-1.), 10., 'behind', {})
    actual = e.tick(10.3, 'reach', KEYS, 1, enforce_envelope=False)
    assert e.trajectory is previous
    assert e.last_request_id == 'accepted'
    assert e.pending is None
    assert e.next_request == 10.3
    assert e.consecutive_rejections == 1
    np.testing.assert_allclose(actual, previous.action(10.3, enforce_envelope=False))
    rejected = [v for k, v in e.trace.events if k == 'rail_chunk_rejected'][-1]
    assert rejected['request_id'] == 'behind'
    assert rejected['fallback'] == 'previous_reference_then_terminal_hold'
    for now in (11., 12., 20.):
        action = e.tick(now, 'reach', KEYS, 1, enforce_envelope=False)
        np.testing.assert_allclose(action, previous.action(previous.end, enforce_envelope=False))
        np.testing.assert_allclose(e.trajectory.state(now)[1:], 0., atol=1e-9)


def test_tracking_observation_does_not_reset_boundary_to_feedback():
    e = execution()
    previous = e.trajectory
    e.output_guard.positions = np.zeros(8)
    expected = previous.state(10.3)
    e.pending = (chunk(.02), 10.1, 'forward', {})
    e.tick(10.3, 'reach', KEYS, 1, enforce_envelope=False, initial_positions=np.full(8, -.2))
    assert e.trajectory is not previous
    for actual, boundary in zip(e.trajectory.state(10.3), expected):
        np.testing.assert_allclose(actual, boundary, atol=1e-9)
    tracking = [v for k, v in e.trace.events if k == 'rail_tracking'][-1]
    assert tracking['reference_error_max_rad'] == pytest.approx(max(abs(expected[0]+.2)))
    assert tracking['projected_reference_lead_s'] > 0.
    installed = [v for k, v in e.trace.events if k == 'rail_chunk_installed'][-1]
    assert installed['offset_s'] >= installed['nominal_policy_offset_s']
    assert installed['algorithm'] == 'cartesian_bridge_admission_v2'
    assert installed['output_hz'] == 15.


def test_rejection_diagnostics_reset_with_generation():
    e = execution()
    e.pending = (chunk(-1.), 10., 'behind', {})
    e.tick(10.3, 'reach', KEYS, 1, enforce_envelope=False)
    assert e.consecutive_rejections == 1
    e.reset()
    assert e.consecutive_rejections == 0
    assert e.trajectory is None
    assert e.pending is None


def test_runtime_loads_cartesian_model_and_retains_it_across_reset():
    robot = FakeRobot()
    loop = ControlLoop(SimpleNamespace(get_action=mock.Mock()))
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            loop.configure('ffw_sg2_rev1', action_keys=KEYS, vla_rail_enabled=True)
        provider = loop._rail.tcp_positions
        assert provider(np.zeros((3, 7))).shape == (3, 1, 3)
        loop._rail.reset()
        assert loop._rail.tcp_positions is provider
    finally:
        loop.shutdown()


def test_missing_robot_model_fails_rail_configuration_without_joint_only_fallback(tmp_path):
    robot = FakeRobot()
    robot.urdf_path = str(tmp_path/'missing.urdf')
    robot.close = mock.Mock()
    loop = ControlLoop(SimpleNamespace(get_action=mock.Mock()))
    try:
        with mock.patch.object(control_loop_module, 'RobotClient', return_value=robot):
            with pytest.raises((OSError, ValueError)):
                loop.configure('ffw_sg2_rev1', action_keys=KEYS, vla_rail_enabled=True)
        robot.close.assert_called_once()
        assert loop._rail is None
        assert loop._robot is None
        assert not robot.commands
    finally:
        loop.shutdown()


def test_late_response_retimes_remaining_curve_from_current_progress():
    e = execution()
    e.tcp_positions = lambda q: np.stack((q[..., 0], np.zeros_like(q[..., 0]),
                                         np.zeros_like(q[..., 0])), axis=-1)[..., None, :]
    previous = e.trajectory
    now = 10.7
    expected = previous.state(now)
    e.pending = (chunk(), 10., 'late', {'request_round_trip_s': .65})
    e.tick(now, 'reach', KEYS, 1, enforce_envelope=False)
    assert e.trajectory is not previous
    assert e.last_request_id == 'late'
    # Here the reference and incoming path share their clock, so the nearest
    # phase agrees with elapsed time. Spatial search has finite precision.
    assert e.trajectory.offset == pytest.approx(.7, abs=1e-6)
    assert e.trajectory.time_scale < 1.
    assert e.trajectory.duration >= .65+e.blend+2*e.output_dt-1e-12
    assert e.output_hz == 15.
    for actual, boundary in zip(e.trajectory.state(now), expected):
        np.testing.assert_allclose(actual, boundary, atol=1e-9)
    np.testing.assert_allclose(e.trajectory.state(e.trajectory.end+10.)[1:], 0., atol=1e-9)


def test_expired_warmup_does_not_inflate_coverage_budget():
    e = execution()
    e.pending = (chunk(), 8., 'expired', {'request_round_trip_s': 2.3})
    e.tick(10.3, 'reach', KEYS, 1, enforce_envelope=False)
    assert e.response_budget_s == 0.
    assert e.last_request_id == 'accepted'
    e.response_budget_s = .7
    e.reset()
    assert e.response_budget_s == 0.
