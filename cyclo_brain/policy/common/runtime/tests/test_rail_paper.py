"""Independent numerical checks of VLA-RAIL arXiv:2512.24673v1 Eqs. 8–14.

The matrix reference solvers below intentionally use physical time and do not
reuse production coefficient generation. Local bootstrap/terminal holds are
checked separately from the paper's smoothing and fusion equations.
"""
import numpy as np
import pytest

from main_runtime.rail_trajectory import (
    CubicChunk, DualQuintic, RailTrajectory, alignment_offset, trajectory_timing,
)


def make_chunk(values, dt=1/15):
    values = np.asarray(values, dtype=float)
    return CubicChunk(np.tile(values[:, None], (1, 8)), dt, ['arm_right'])


def polynomial_state(coefficients, times):
    """Differentiate independent physical-time ascending-order coefficients."""
    times = np.asarray(times, dtype=float)
    result = []
    for order in range(3):
        derivative = np.polynomial.polynomial.polyder(coefficients, m=order, axis=0)
        powers = times[..., None]**np.arange(len(derivative))
        result.append(powers @ derivative)
    return tuple(result)


def reference_global_fit(actions, dt, arm_columns):
    times = np.arange(len(actions))*dt
    design = np.column_stack([times**power for power in range(4)])
    return np.linalg.lstsq(design, actions[:, arm_columns], rcond=None)[0]


def reference_quintic(start, end, duration):
    """Independent 6x6 boundary-value solve in physical seconds."""
    def row(time, order):
        result = []
        for power in range(6):
            if power < order:
                result.append(0.)
            else:
                factor = 1
                for offset in range(order):
                    factor *= power-offset
                result.append(factor*time**(power-order))
        return result

    matrix = np.array([row(time, order) for time in (0., duration) for order in range(3)])
    values = np.array(list(start)+list(end))
    return np.linalg.solve(matrix, values)


@pytest.mark.parametrize('keys', [['arm_right'], ['arm_right', 'arm_left']])
@pytest.mark.parametrize('length,dt', [(4, .03), (16, 1/15), (32, .1)])
def test_global_cubic_matches_independent_whole_chunk_least_squares(keys, length, dt):
    rng = np.random.default_rng(137)
    actions = rng.normal(0., .1, (length, 8*len(keys)))
    chunk = CubicChunk(actions, dt, keys)
    reference = reference_global_fit(actions, dt, chunk.arm_columns)
    times = np.linspace(0., chunk.duration, 127)
    for actual, expected in zip(chunk.state(times), polynomial_state(reference, times)):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_cubic_fit_attenuates_alternating_prediction_noise_instead_of_interpolating_it():
    times = np.arange(16)/15
    truth = .2+.1*times
    noise = .01*(-1.)**np.arange(len(times))
    chunk = make_chunk(truth+noise)
    fitted, velocity, acceleration = chunk.state(times)
    fitted_error = np.sqrt(np.mean((fitted[:, 0]-truth)**2))
    raw_error = np.sqrt(np.mean(noise**2))
    assert fitted_error < .25*raw_error
    assert np.max(np.abs(fitted[:, 0]-(truth+noise))) > .009
    assert np.min(velocity[:, 0]) > .05
    assert np.max(np.abs(acceleration[:, 0])) < .3
    # An interior perturbation also changes distant fitted values: the model
    # covers the whole chunk rather than fitting each waypoint interval.
    changed = truth+noise
    changed[8] += .04
    assert not np.isclose(make_chunk(changed).state(0.)[0][0], fitted[0, 0])


def test_global_fit_has_correct_physical_derivatives_for_nonunit_chunk_duration():
    times = np.arange(20)*.08
    values = .2+.3*times-.1*times**2+.04*times**3
    chunk = make_chunk(values, dt=.08)
    sample = np.linspace(0., chunk.duration, 101)
    q, v, a = chunk.state(sample)
    np.testing.assert_allclose(q[:, 0], .2+.3*sample-.1*sample**2+.04*sample**3, atol=1e-13)
    np.testing.assert_allclose(v[:, 0], .3-.2*sample+.12*sample**2, atol=1e-12)
    np.testing.assert_allclose(a[:, 0], -.2+.24*sample, atol=1e-11)


def test_arm_fit_excludes_grippers_and_preserves_action_order():
    rng = np.random.default_rng(9)
    actions = rng.normal(0., .1, (16, 16))
    chunk = CubicChunk(actions, .07, ['arm_right', 'arm_left'])
    altered = actions.copy()
    altered[:, [7, 15]] = rng.uniform(-100., 100., (16, 2))
    changed = CubicChunk(altered, .07, ['arm_right', 'arm_left'])
    for actual, expected in zip(changed.state(.31), chunk.state(.31)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(changed.grippers(.31), altered[4, [7, 15]])
    reference = reference_global_fit(actions, .07, [*range(7), *range(8, 15)])
    np.testing.assert_allclose(chunk.state(.31)[0], polynomial_state(reference, .31)[0], atol=1e-12)


@pytest.mark.parametrize('duration', [.12, .4, 1.3])
def test_dual_quintic_matches_independent_six_by_six_solves_and_midpoint_constraints(duration):
    rng = np.random.default_rng(21)
    start = tuple(rng.normal(0., .2, 7) for _ in range(3))
    end = tuple(rng.normal(0., .2, 7) for _ in range(3))
    midpoint = ((start[0]+end[0])/2, (start[1]+end[1])/2, np.zeros(7))
    left = reference_quintic(start, midpoint, duration/2)
    right = reference_quintic(midpoint, end, duration/2)
    trajectory = DualQuintic(start, end, duration)
    for time in np.linspace(0., duration, 51):
        coefficients = left if time <= duration/2 else right
        local_time = time if time <= duration/2 else time-duration/2
        for actual, expected in zip(trajectory.state(time), polynomial_state(coefficients, local_time)):
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    for time, expected in ((0., start), (duration/2, midpoint), (duration, end)):
        for actual, boundary in zip(trajectory.state(time), expected):
            np.testing.assert_allclose(actual, boundary, atol=1e-10)


def test_dual_quintic_vectorized_samples_match_scalar_evaluation_and_are_c2_at_midpoint():
    start = (np.zeros(7), np.full(7, .2), np.full(7, -.1))
    end = (np.full(7, .1), np.full(7, .3), np.full(7, .2))
    trajectory = DualQuintic(start, end, .2)
    times = np.array([0., .03, .1, .14, .2])
    for order, values in enumerate(trajectory.state(times)):
        expected = np.array([trajectory.state(time)[order] for time in times])
        np.testing.assert_allclose(values, expected, atol=1e-12)
    for left, right in zip(trajectory.state(.1-1e-11), trajectory.state(.1+1e-11)):
        np.testing.assert_allclose(left, right, atol=1e-6)


class AffineChunk:
    """Exact position functions keep Eq. 10 sign(0) checks unambiguous."""
    duration = 2.
    dt = .1

    def __init__(self, slopes):
        slopes = np.asarray(slopes, dtype=float)
        self.coefficients = np.array([np.zeros_like(slopes), self.duration*slopes,
                                      np.zeros_like(slopes), np.zeros_like(slopes)])

    def state(self, when):
        q, v, a = polynomial_state(self.coefficients, np.asarray(when)/self.duration)
        return q, v/self.duration, a/self.duration**2


@pytest.mark.parametrize('speed', [1., 1e-15])
def test_alignment_counts_zero_as_positive_and_has_no_moving_speed_threshold(speed):
    chunk = AffineChunk([1., -1., 0., 0., 0., 0., 0.])
    previous_q = np.array([.5, -.5, 0., 0., 0., 0., 0.])
    previous_v = np.array([speed, speed, 0., 0., 0., 0., 0.])
    # Both moving joints agree only at t=.5, where both products are exactly
    # zero. np.sign(0)=0 or a moving-speed threshold would lose this maximum.
    selected = alignment_offset(chunk, .2, (previous_q, previous_v, np.zeros(7)))
    assert selected == pytest.approx(.5, abs=1e-12)


def test_alignment_searches_full_valid_horizon_beyond_old_latency_window():
    chunk = AffineChunk(np.ones(7))
    previous = (np.full(7, 1.1), np.ones(7), np.zeros(7))
    selected = alignment_offset(chunk, .2, previous)
    assert selected >= 1.1-1e-12
    assert selected < chunk.duration-.2
    # Eq. 10 ranks direction, not distance magnitude or derivative mismatch.
    q = chunk.state(selected)[0]
    assert np.all((q-previous[0])*previous[1] >= -1e-12)


def test_alignment_finds_narrow_maximum_interval_between_polynomial_roots():
    chunk = AffineChunk(np.zeros(7))
    # q0=(t-.4)*(t-.401) agrees with negative old velocity only inside a
    # one-millisecond interval; a coarse full-horizon sample grid can miss it.
    chunk.coefficients[:, 0] = [.4*.401, -(.4+.401)*chunk.duration,
                                chunk.duration**2, 0.]
    previous = (np.zeros(7), np.array([-1., 0., 0., 0., 0., 0., 0.]), np.zeros(7))
    selected = alignment_offset(chunk, .1, previous)
    assert .4-1e-10 <= selected <= .401+1e-10


def test_alignment_equal_direction_scores_tie_at_nominal_without_distance_penalty():
    chunk = AffineChunk(np.ones(7))
    previous = (np.full(7, -100.), np.ones(7), np.full(7, 1000.))
    assert alignment_offset(chunk, .5, previous) == pytest.approx(.5)
    stationary = (np.full(7, 100.), np.zeros(7), np.full(7, -1000.))
    assert alignment_offset(chunk, .5, stationary) == pytest.approx(.5)


@pytest.mark.parametrize('rate', [.5, 1., 2.])
def test_explicit_playback_rate_sets_duration_and_physical_derivatives(rate):
    times = np.linspace(0., 1., 16)
    chunk = make_chunk(.2+.3*times-.1*times**2+.04*times**3)
    previous = (np.full(7, .15), np.full(7, .12), np.full(7, -.07))
    trajectory = RailTrajectory(chunk, 10., offset=.1, previous_state=previous,
                                blend=.1, playback_rate=rate)
    assert trajectory.duration == pytest.approx(.9/rate)
    assert trajectory.time_scale == pytest.approx(rate)
    assert trajectory.blend == pytest.approx(min(.1, trajectory.duration/3))
    for actual, expected in zip(trajectory.state(10.), previous):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    elapsed = trajectory.duration/2
    q, v, a = chunk.state(.1+elapsed*rate)
    for actual, expected in zip(trajectory.state(10.+elapsed), (q, v*rate, a*rate**2)):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    eps = 1e-6
    qm, vm, _ = trajectory.state(10.+elapsed-eps)
    qp, vp, _ = trajectory.state(10.+elapsed+eps)
    _, actual_v, actual_a = trajectory.state(10.+elapsed)
    np.testing.assert_allclose((qp-qm)/(2*eps), actual_v, rtol=2e-8, atol=1e-8)
    np.testing.assert_allclose((vp-vm)/(2*eps), actual_a, rtol=2e-8, atol=1e-8)


def test_paper_head_preserves_physical_boundary_and_dual_midpoint_with_rate_two():
    chunk = make_chunk(np.linspace(.1, .3, 16))
    previous = (np.zeros(7), np.full(7, .07), np.full(7, -.03))
    trajectory = RailTrajectory(chunk, 10., previous_state=previous, playback_rate=2.)
    target = trajectory.reference_state(trajectory.blend)
    midpoint = ((previous[0]+target[0])/2, (previous[1]+target[1])/2, np.zeros(7))
    for actual, expected in zip(trajectory.state(10.+trajectory.blend/2), midpoint):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    for actual, expected in zip(trajectory.state(10.+trajectory.blend), target):
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_local_terminal_hold_uses_fitted_endpoint_and_keeps_splices_c2():
    times = np.arange(16)/15
    chunk = make_chunk(.2+.1*times+.01*(-1.)**np.arange(16))
    trajectory = RailTrajectory(chunk, 10., playback_rate=2.)
    for boundary in (trajectory.blend/2, trajectory.blend, trajectory.tail_start,
                     trajectory.tail_start+trajectory.blend/2, trajectory.duration):
        left = trajectory.state(10.+boundary-1e-10)
        right = trajectory.state(10.+boundary+1e-10)
        for actual, expected in zip(left, right):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
    held_q, held_v, held_a = trajectory.state(100.)
    np.testing.assert_allclose(held_q, chunk.state(chunk.duration)[0], atol=1e-12)
    np.testing.assert_allclose(held_v, 0., atol=1e-12)
    np.testing.assert_allclose(held_a, 0., atol=1e-11)
    assert not np.isclose(held_q[0], chunk.actions[-1, 0])


def test_timing_vectorization_and_alignment_expiry_reserve_head_execution_time():
    chunk = make_chunk(np.linspace(0., .1, 16))
    offsets = np.array([0., .1, .7, .9])
    vector = trajectory_timing(chunk, offsets, blend=.15, playback_rate=2.)
    for index, offset in enumerate(offsets):
        trajectory = RailTrajectory(chunk, 0., offset=offset, blend=.15, playback_rate=2.)
        assert trajectory.duration == pytest.approx(vector[0][index])
        assert trajectory.blend == pytest.approx(vector[1][index])
        assert trajectory.time_scale == pytest.approx(np.broadcast_to(vector[2], offsets.shape)[index])
    last = chunk.duration-min(.2*2., chunk.duration/3)
    assert alignment_offset(chunk, last, None, blend=.2, playback_rate=2.) is None
    assert alignment_offset(chunk, last-1e-6, None, blend=.2, playback_rate=2.) == pytest.approx(last-1e-6)
