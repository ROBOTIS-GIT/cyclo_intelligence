"""Handoff regressions: causal phase, invented reversals and real turns."""
import json
import numpy as np
import pytest

from main_runtime.rail_handoff import select_handoff, _reference_samples, _tcp_curve_metrics
from main_runtime.rail_alignment import nearest_tcp_phases
from main_runtime.rail_trajectory import CubicChunk, RailTrajectory, DualQuintic


class MovingReference:
    def __init__(self, q=0., v=.1, a=0.):
        self.q, self.v, self.a = (np.broadcast_to(np.asarray(value, dtype=float), (7,)).copy()
                                 for value in (q, v, a))

    def state(self, time):
        return self.q+self.v*time+.5*self.a*time**2, self.v+self.a*time, self.a.copy()


def chunk_from(function, length=16):
    times = np.arange(length)/15
    values = np.asarray(function(times), dtype=float)
    if values.ndim == 1:
        values = np.repeat(values[:, None], 7, axis=1)
    return CubicChunk(np.column_stack((values, np.zeros(length))), 1/15, ['arm_right'])


def choose(chunk, ref=None, age=.3, **kwargs):
    ref = MovingReference() if ref is None else ref
    return select_handoff(chunk, 0., age, ref.state(0.), ref, **kwargs)


def test_consistent_forward_motion_keeps_causal_phase_and_forward_velocity():
    result = choose(chunk_from(lambda t: .1*(t-.3)))
    assert result.reason == 'accepted'
    assert result.trajectory.offset >= .3
    assert result.trajectory.blend >= .2
    q, v, a = result.trajectory.head.state(np.linspace(0., result.trajectory.blend, 1001))
    assert np.min(v) >= -1e-12
    np.testing.assert_allclose(q[0], 0., atol=1e-12)
    np.testing.assert_allclose(v[0], .1, atol=1e-12)
    np.testing.assert_allclose(a[0], 0., atol=1e-12)
    json.dumps(result.diagnostics, allow_nan=False)


def test_entire_new_horizon_behind_same_moving_direction_is_rejected():
    result = choose(chunk_from(lambda t: -.2+.1*t))
    assert result.trajectory is None
    assert result.reason == 'incompatible_bridge'
    assert result.diagnostics['rejection_counts']['behind_continuation'] > 0


def test_actual_curve_check_advances_past_forward_endpoint_that_would_oscillate():
    # Baseline: q0=0, v0=.5, q1=.01, v1=.5 at B=.2 produces two
    # backwards lobes even though the endpoint lies in front of q0.
    baseline = DualQuintic((np.zeros(7), np.full(7, .5), np.zeros(7)),
                          (np.full(7, .01), np.full(7, .5), np.zeros(7)), .2)
    assert baseline.state(np.linspace(0., .2, 1001))[1].min() < -.3
    result = choose(chunk_from(lambda t: .5*(t-.3)-.09), MovingReference(v=.5))
    assert result.trajectory is not None
    assert result.trajectory.offset > .3
    assert result.trajectory.head.state(np.linspace(0., result.trajectory.blend, 1001))[1].min() >= -1e-10


def test_sustained_new_reverse_direction_is_allowed():
    result = choose(chunk_from(lambda t: .04-.1*t))
    assert result.trajectory is not None
    velocity = result.trajectory.head.state(np.linspace(0., result.trajectory.blend, 1001))[1][:, 0]
    assert velocity[0] > 0 and velocity[-1] < 0
    signs = np.sign(velocity[np.abs(velocity) > 1e-8])
    assert np.count_nonzero(np.diff(signs)) == 1


def test_joints_at_rest_can_approach_then_follow_a_new_path():
    result = choose(chunk_from(lambda t: .1*(t-.9)), MovingReference(v=0.))
    assert result.trajectory is not None
    assert result.diagnostics['checked_moving_joint_count'] == 0
    q, v, _ = result.trajectory.head.state(np.linspace(0., result.trajectory.blend, 1001))
    assert np.min(q) < 0
    assert v[-1, 0] > 0


def test_tiny_wrist_boundary_velocity_cannot_veto_forward_main_motion():
    def new_values(t):
        values = np.zeros((len(t), 7))
        values[:, 0] = .5*(t-.3)
        values[:, 1] = -.02+.025*(t-.3)
        return values
    boundary_velocity = np.array([.5, .001, 0., 0., 0., 0., 0.])
    result = choose(chunk_from(new_values), MovingReference(v=boundary_velocity))
    assert result.trajectory is not None
    assert result.diagnostics['checked_moving_joint_count'] == 1
    assert result.trajectory.head.state(np.linspace(0., result.trajectory.blend, 501))[1][:, 0].min() >= -1e-10


def test_requested_blend_is_not_shortened_to_fit_a_late_chunk():
    accepted = choose(chunk_from(lambda t: .1*(t-.55)), age=.55)
    assert accepted.trajectory is not None
    assert accepted.trajectory.blend == pytest.approx(.2)
    assert accepted.trajectory.tail_start >= accepted.trajectory.blend-1e-12
    rejected = choose(chunk_from(lambda t: .1*(t-.65)), age=.65)
    assert rejected.trajectory is None
    assert rejected.reason == 'insufficient_horizon'


def test_bootstrap_uses_feedback_and_zero_offset_but_rejects_expired_response():
    chunk = chunk_from(lambda t: .1*t)
    boundary = MovingReference(q=-.01, v=0.).state(0.)
    result = select_handoff(chunk, 0., .3, boundary)
    assert result.reason == 'bootstrap'
    assert result.trajectory.offset == 0.
    np.testing.assert_allclose(result.trajectory.state(0.)[0], boundary[0])
    assert select_handoff(chunk, 0., 1.01, boundary).reason == 'expired_horizon'


def test_short_fresh_bootstrap_preserves_existing_timing_adaptation():
    chunk = chunk_from(lambda t: .1*t, length=4)
    result = select_handoff(chunk, 0., .05, MovingReference(v=0.).state(0.))
    assert result.reason == 'bootstrap'
    assert result.trajectory.blend == pytest.approx(chunk.duration/3)


def test_static_goal_and_stationary_prediction_have_finite_costs():
    for goal in (0., .03):
        result = choose(chunk_from(lambda t: np.full_like(t, goal)), MovingReference(v=0.))
        assert result.trajectory is not None
        json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize('scale', [.001, 1., 100.])
def test_motion_scaling_does_not_encounter_a_hidden_velocity_limit(scale):
    result = choose(chunk_from(lambda t: scale*.1*(t-.3)), MovingReference(v=scale*.1))
    assert result.trajectory is not None
    np.testing.assert_allclose(result.trajectory.state(0.)[1], scale*.1, rtol=1e-10)


@pytest.mark.parametrize('field,value', [('observation_age', np.nan), ('observation_age', -.01),
                                        ('blend', 0.), ('playback_rate', np.inf), ('now', np.inf)])
def test_invalid_timing_is_rejected(field, value):
    kwargs = dict(chunk=chunk_from(lambda t: .1*t), now=0., observation_age=.3,
                  previous_state=MovingReference().state(0.))
    kwargs[field] = value
    with pytest.raises(ValueError):
        select_handoff(**kwargs)


def test_nonfinite_boundary_is_rejected():
    state = list(MovingReference().state(0.))
    state[1][2] = np.nan
    with pytest.raises(ValueError):
        select_handoff(chunk_from(lambda t: .1*t), 0., .3, state)


@pytest.mark.parametrize('mode', ['dual', 'single'])
def test_vectorized_reference_matches_scalar_head_middle_tail_and_hold(mode):
    chunk = chunk_from(lambda t: .1*t+.02*t*t)
    trajectory = RailTrajectory(chunk, 10., .2, MovingReference().state(0.), .2,
                                head_mode=mode, preserve_blend=True)
    times = np.linspace(9.9, 11.1, 121)
    vector = _reference_samples(trajectory, times)
    scalar = tuple(np.array([trajectory.state(t)[index] for t in times]) for index in range(3))
    for actual, expected in zip(vector, scalar):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_selected_bridge_and_terminal_stop_keep_c2_boundaries():
    result = choose(chunk_from(lambda t: .1*(t-.3)))
    trajectory = result.trajectory
    for left, right in ((trajectory.head.state(trajectory.blend), trajectory.reference_state(trajectory.blend)),
                        (trajectory.reference_state(trajectory.tail_start), trajectory.tail.state(0.))):
        for actual, expected in zip(left, right):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
    stopped = trajectory.state(trajectory.end+1.)
    np.testing.assert_allclose(stopped[1], 0., atol=1e-10)
    np.testing.assert_allclose(stopped[2], 0., atol=1e-10)


def linear_tcp(q):
    """A one-axis Cartesian robot, isolating progress from robot geometry."""
    q = np.asarray(q)
    xyz = np.zeros(q.shape[:-1]+(1, 3))
    xyz[..., 0, 0] = q[..., 0]
    return xyz


def tcp_path(values):
    q = np.zeros((1, len(values), 7))
    q[0, :, 0] = values
    return q


@pytest.mark.parametrize('excursion,rejected', [(.002, False), (.010, True)])
def test_tcp_admission_distinguishes_small_excursion_from_large_backtracking(excursion, rejected):
    boundary = MovingReference(v=np.array([.1, 0., 0., 0., 0., 0., 0.])).state(0.)
    metrics = _tcp_curve_metrics(linear_tcp, tcp_path([0., -excursion, .02]),
                                 tcp_path([0., .01, .02]), boundary, .2, .003, .02)
    assert bool(metrics['rejected'][0]) is rejected
    assert metrics['extra'][0, 0] == pytest.approx(excursion)


def test_tcp_prefix_check_cannot_borrow_a_later_intentional_return():
    boundary = MovingReference(v=np.array([.1, 0., 0., 0., 0., 0., 0.])).state(0.)
    metrics = _tcp_curve_metrics(linear_tcp, tcp_path([0., -.01, .01, -.04]),
                                 tcp_path([0., .02, .03, -.04]), boundary, .2, .003, .02)
    # Total backwards distance would pass: the model eventually returns even
    # farther. The extra 10 mm at the *beginning* must still be rejected.
    assert metrics['backward'][0, 0] < metrics['reference_backward'][0, 0]
    assert metrics['rejected'][0]


def test_tcp_admission_preserves_an_intentional_model_return():
    boundary = MovingReference(v=np.array([.1, 0., 0., 0., 0., 0., 0.])).state(0.)
    path = tcp_path([0., .01, -.02, -.04])
    metrics = _tcp_curve_metrics(linear_tcp, path, path, boundary, .2, .003, .02)
    assert metrics['backward'][0, 0] > .04
    assert not metrics['rejected'][0]
    result = select_handoff(chunk_from(lambda t: .04-.1*t), 0., .3,
                            MovingReference().state(0.), MovingReference(),
                            tcp_positions=linear_tcp, minimum_horizon_s=.9)
    assert result.trajectory is not None
    assert result.trajectory.state(0.)[1][0] > 0
    assert result.trajectory.reference_state(result.trajectory.blend)[1][0] < 0


def test_nearest_selected_remainder_stretches_to_cover_next_inference():
    age = .74
    reference = MovingReference()
    chunk = chunk_from(lambda t: .1*(t-age))
    result = select_handoff(chunk, 0., age, reference.state(0.), reference,
                            tcp_positions=linear_tcp, minimum_horizon_s=.9)
    trajectory = result.trajectory
    assert trajectory is not None
    assert trajectory.offset == pytest.approx(age, abs=1e-6)
    assert trajectory.duration >= .9-1e-12
    assert trajectory.time_scale <= (chunk.duration-trajectory.offset)/.9+1e-12
    assert trajectory.blend >= .2
    assert trajectory.tail_blend > 0
    assert trajectory.tail_start >= trajectory.blend-1e-12
    for actual, expected in zip(trajectory.state(0.), reference.state(0.)):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    # It still ends at the predicted endpoint, without extrapolation.
    stopped = trajectory.state(trajectory.end+10.)
    np.testing.assert_allclose(stopped[0], chunk.state(chunk.duration)[0], atol=1e-10)
    np.testing.assert_allclose(stopped[1], 0., atol=1e-10)
    np.testing.assert_allclose(stopped[2], 0., atol=1e-10)
    assert result.diagnostics['future_retimed']
    json.dumps(result.diagnostics, allow_nan=False)


def test_tcp_planner_rejects_large_whole_horizon_retreat_and_expired_responses():
    reference = MovingReference()
    chunk = chunk_from(lambda t: -.2+.1*t)
    result = select_handoff(chunk, 0., .3, reference.state(0.), reference,
                            tcp_positions=linear_tcp, minimum_horizon_s=.9)
    assert result.trajectory is None
    assert result.reason in ('incompatible_bridge', 'nearest_phase_has_no_future')
    if result.reason == 'incompatible_bridge':
        assert result.diagnostics['rejection_counts']['tcp_excess_backtrack'] > 0
    else:
        assert not result.diagnostics['nearest_has_future']
    expired = select_handoff(chunk, 0., 1.01, reference.state(0.),
                             tcp_positions=linear_tcp, minimum_horizon_s=10.)
    assert expired.trajectory is None
    assert expired.reason == 'expired_horizon'


def test_generic_reference_sampling_preserves_two_dimensional_time_shape():
    reference = MovingReference(q=.1, v=.2, a=.03)
    times = np.array([[0., .1, .2], [.3, .4, .5]])
    actual = _reference_samples(reference, times)
    for order in range(3):
        expected = np.array([reference.state(float(t))[order] for t in times.ravel()]).reshape(2, 3, 7)
        np.testing.assert_allclose(actual[order], expected, atol=1e-12)


def test_independent_head_tail_and_retiming_preserve_all_c2_boundaries():
    chunk = chunk_from(lambda t: .1*t+.02*t*t)
    trajectory = RailTrajectory(chunk, 10., .6, MovingReference().state(0.), .4, .4,
                                head_mode='single', preserve_blend=True, tail_blend=.2)
    assert trajectory.blend == .4
    assert trajectory.tail_blend == .2
    assert trajectory.tail_start == pytest.approx(.8)
    for left, right in ((trajectory.head.state(.4), trajectory.reference_state(.4)),
                        (trajectory.reference_state(.8), trajectory.tail.state(0.))):
        for actual, expected in zip(left, right):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
    times = np.linspace(9.9, 11.2, 131)
    vector = _reference_samples(trajectory, times)
    for order in range(3):
        np.testing.assert_allclose(vector[order], [trajectory.state(t)[order] for t in times], atol=1e-10)


def test_nearest_phase_avoids_age_catchup_without_slowing_the_body():
    # Representative of the captured e1d9d15a failure: the observation's first
    # point is near the current arm, while age*1 points far ahead of an old
    # reference already executing at approximately 0.3 policy seconds/second.
    reference = MovingReference(v=.04)
    chunk = chunk_from(lambda t: -.005+.14*t)
    age, coverage = .687, 1.
    result = select_handoff(chunk, 0., age, reference.state(0.), reference,
                            tcp_positions=linear_tcp, minimum_horizon_s=coverage)
    trajectory = result.trajectory
    old = RailTrajectory(chunk, 0., age, reference.state(0.), .4,
                         (chunk.duration-age)/coverage, head_mode='single',
                         preserve_blend=True, tail_blend=.2)
    assert trajectory is not None
    assert trajectory.offset == pytest.approx(.005/.14, abs=1e-6)
    assert trajectory.offset < .1 < age
    new_peak = np.max(trajectory.head.state(np.linspace(0., trajectory.blend, 1001))[1][:, 0])
    old_peak = np.max(old.head.state(np.linspace(0., old.blend, 1001))[1][:, 0])
    assert new_peak < .6*old_peak
    assert trajectory.time_scale > 2*old.time_scale
    assert result.diagnostics['selected_alignment_rank'] == 0
    assert result.diagnostics['planner'] == 'tcp_nearest_handoff_v3'
    for actual, expected in zip(trajectory.state(0.), reference.state(0.)):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    json.dumps(result.diagnostics, allow_nan=False)


def test_continuous_alignment_uses_both_arms_and_is_not_a_raw_index():
    times = np.arange(16)/15
    actions = np.zeros((16, 16))
    actions[:, 0], actions[:, 8] = times, 2*times
    chunk = CubicChunk(actions, 1/15, ['arm_left', 'arm_right'])
    anchor = np.zeros(14)
    anchor[0], anchor[7] = .2, .8

    def both_tcp(q):
        q = np.asarray(q)
        xyz = np.zeros(q.shape[:-1]+(2, 3))
        xyz[..., 0, 0], xyz[..., 1, 0] = q[..., 0], q[..., 7]
        return xyz

    phases, diagnostics = nearest_tcp_phases(chunk, anchor, both_tcp)
    # min (t-.2)^2+(2t-.8)^2 occurs at .36, between raw waypoints.
    assert phases[0] == pytest.approx(.36, abs=1e-6)
    assert abs(phases[0]/chunk.dt-round(phases[0]/chunk.dt)) > .1
    assert len(diagnostics['alignment_anchor_tcp_m']) == 2


def test_identical_tcp_positions_use_joint_posture_to_resolve_the_tie():
    def values(times):
        q = np.zeros((len(times), 7))
        q[:, 6] = times
        return q
    chunk = chunk_from(values)
    anchor = np.zeros(7)
    anchor[6] = .45
    phases, _ = nearest_tcp_phases(chunk, anchor, linear_tcp)
    assert phases[0] == pytest.approx(.45, abs=1e-6)


@pytest.mark.parametrize('phase', [.005, .995])
def test_continuous_minimum_inside_an_edge_grid_cell_is_refined(phase):
    chunk = chunk_from(lambda t: t)
    phases, _ = nearest_tcp_phases(chunk, np.full(7, phase), linear_tcp)
    assert phases[0] == pytest.approx(phase, abs=1e-6)


def test_equal_cartesian_crossings_prefer_the_near_joint_configuration():
    chunk = chunk_from(lambda t: 4*np.pi*t)
    anchor = np.full(7, 3*np.pi)

    def circular_tcp(q):
        q = np.asarray(q)
        xyz = np.zeros(q.shape[:-1]+(1, 3))
        xyz[..., 0, 0], xyz[..., 0, 1] = np.cos(q[..., 0]), np.sin(q[..., 0])
        return xyz

    phases, _ = nearest_tcp_phases(chunk, anchor, circular_tcp)
    assert phases[0] == pytest.approx(.75, abs=1e-6)


def test_terminal_nearest_point_uses_explicit_native_interval_fallback():
    chunk = chunk_from(lambda t: .02*t)
    reference = MovingReference(q=.02, v=0.)
    result = select_handoff(chunk, 0., .7, reference.state(0.), reference,
                            tcp_positions=linear_tcp, minimum_horizon_s=.9)
    assert result.trajectory is not None
    diagnostics = result.diagnostics
    assert diagnostics['nearest_policy_phase_s'] == pytest.approx(chunk.duration)
    assert not diagnostics['nearest_has_future']
    assert diagnostics['alignment_fallback_used']
    assert diagnostics['alignment_fallback_reason'] == 'nearest_phase_has_no_future'
    assert result.trajectory.offset <= chunk.duration-chunk.dt+1e-12
    np.testing.assert_allclose(result.trajectory.state(result.trajectory.end)[0], .02, atol=1e-10)


def test_alignment_fallbacks_cannot_jump_to_the_far_age_phase():
    chunk = chunk_from(lambda t: .5*t)
    phases, diagnostics = nearest_tcp_phases(chunk, np.zeros(7), linear_tcp)
    assert phases[0] == 0.
    assert np.max(phases) < .1
    assert diagnostics['alignment_candidate_tcp_gaps_m'][0] == pytest.approx([0.], abs=1e-12)


@pytest.mark.parametrize('retreat', [.023, .051, .190])
def test_nearest_alignment_still_rejects_large_invented_backward_bridges(retreat):
    # Every predicted point is behind the moving arm even though the model
    # itself continues forward. Nearest alignment must not waive admission.
    chunk = chunk_from(lambda t: -retreat-.002+.002*t)
    reference = MovingReference(v=.1)
    result = select_handoff(chunk, 0., .7, reference.state(0.), reference,
                            tcp_positions=linear_tcp, minimum_horizon_s=.9)
    assert result.trajectory is None
    assert result.reason == 'incompatible_bridge'
    assert result.diagnostics['rejection_counts']['tcp_excess_backtrack'] > 0


def test_arm_alignment_does_not_rewind_the_independent_gripper_clock():
    chunk = chunk_from(lambda t: -.005+.14*t)
    chunk.actions[:, chunk.gripper_columns] = np.arange(16)[:, None]/15
    reference = MovingReference(v=.04)
    result = select_handoff(chunk, 0., .7, reference.state(0.), reference,
                            tcp_positions=linear_tcp, minimum_horizon_s=1.)
    trajectory = result.trajectory
    assert trajectory.offset < .1
    assert trajectory.gripper_offset == pytest.approx(.7)
    assert trajectory.gripper_time_scale == pytest.approx(.3)
    assert trajectory.action(0.)[7] == pytest.approx(10/15)
    assert trajectory.action(.5)[7] == pytest.approx(12/15)
