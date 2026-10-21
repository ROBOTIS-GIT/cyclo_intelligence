import numpy as np
import pytest

from main_runtime.rail_output_guard import RailOutputGuard


@pytest.mark.parametrize('width', [8, 16])
def test_commands_are_unchanged_through_large_jumps_jitter_and_reversals(width):
    guard = RailOutputGuard()
    rng = np.random.default_rng(42)
    now = 10.
    initial = np.linspace(-1., 1., width)
    np.testing.assert_array_equal(guard.update(np.ones(width)*5, now, initial), initial)
    for i in range(100):
        now += float(rng.uniform(.005, .025))
        target = rng.uniform(10., 100., width) * (-1 if i % 2 else 1)
        np.testing.assert_array_equal(guard.update(target, now), target)


@pytest.mark.parametrize('now', [0., -.1, .101, float('nan'), float('inf'), -float('inf')])
def test_invalid_or_stale_timestamps_preserve_last_valid_command(now):
    guard = RailOutputGuard()
    guard.update(np.ones(8), 0., np.zeros(8))
    with pytest.raises(ValueError):
        guard.update(np.ones(8), now)
    np.testing.assert_array_equal(guard.positions, np.zeros(8))
    assert guard.last_time == 0.
    np.testing.assert_array_equal(guard.update(np.ones(8), .1), np.ones(8))


@pytest.mark.parametrize('max_interval_s', [0., -.1, float('inf'), float('nan')])
def test_invalid_intervals_are_rejected(max_interval_s):
    with pytest.raises(ValueError, match='positive and finite'):
        RailOutputGuard(max_interval_s=max_interval_s)


@pytest.mark.parametrize('initial', [None, np.zeros(7), np.zeros(16), np.zeros((1, 8)),
                                   np.full(8, np.nan), np.full(8, np.inf)])
def test_initialization_requires_finite_feedback_of_matching_shape(initial):
    guard = RailOutputGuard()
    with pytest.raises(ValueError, match='Fresh current'):
        guard.update(np.ones(8), 0., initial)
    assert guard.positions is None
    assert guard.last_time is None


@pytest.mark.parametrize('bad', [np.full(8, np.nan), np.full(8, np.inf), np.ones(7),
                               np.ones((1, 8)), np.array(1.)])
@pytest.mark.parametrize('initialized', [False, True])
def test_invalid_targets_are_rejected_before_and_after_initialization(bad, initialized):
    guard = RailOutputGuard()
    if initialized:
        guard.update(np.ones(8), 0., np.zeros(8))
    with pytest.raises(ValueError, match='Invalid RAIL target'):
        guard.update(bad, .01, np.zeros(8))
    if initialized:
        np.testing.assert_array_equal(guard.positions, np.zeros(8))
        assert guard.last_time == 0.
    else:
        assert guard.positions is None
        assert guard.last_time is None


@pytest.mark.parametrize('now', [float('nan'), float('inf'), -float('inf')])
def test_initialization_rejects_nonfinite_timestamps(now):
    guard = RailOutputGuard()
    with pytest.raises(ValueError, match='Invalid RAIL target or command timestamp'):
        guard.update(np.ones(8), now, np.zeros(8))
    assert guard.positions is None
    assert guard.last_time is None


def test_dimension_change_requires_reset_and_fresh_feedback():
    guard = RailOutputGuard()
    guard.update(np.ones(8), 0., np.zeros(8))
    with pytest.raises(ValueError, match='dimensions changed'):
        guard.update(np.ones(16), .01)
    np.testing.assert_array_equal(guard.positions, np.zeros(8))
    assert guard.last_time == 0.
    guard.reset()
    assert guard.positions is None
    assert guard.last_time is None
    with pytest.raises(ValueError, match='Fresh current'):
        guard.update(np.ones(16), 20.)
    initial = np.full(16, -.2)
    np.testing.assert_array_equal(guard.update(np.ones(16), 20., initial), initial)
    np.testing.assert_array_equal(guard.update(np.ones(16), 20.01), np.ones(16))


def test_feedback_targets_and_returned_commands_do_not_alias_internal_state():
    guard = RailOutputGuard()
    initial = np.zeros(8)
    first_command = guard.update(np.ones(8), 0., initial)
    initial[:] = 10.
    first_command[:] = 20.
    np.testing.assert_array_equal(guard.positions, np.zeros(8))
    target = np.arange(8, dtype=float)
    expected = target.copy()
    command = guard.update(target, .01)
    target[:] = 30.
    np.testing.assert_array_equal(command, expected)
    command[:] = 40.
    np.testing.assert_array_equal(guard.positions, expected)
