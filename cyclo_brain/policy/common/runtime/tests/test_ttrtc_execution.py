"""TT-RTC queue/deadline tests with a fake clock and no hardware/model calls."""
import sys
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_runtime.ttrtc_execution import TtrtcExecution


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


def response(prefix, value=10.):
    chunk = np.full((32, 16), value)
    chunk[:len(prefix)] = prefix
    return SimpleNamespace(success=True, chunk_size=32, action_dim=16,
                           action_list=chunk.ravel(), timing_json='{}')


def executor(infer, prefix_steps=7):
    clock = [0.]
    trace = Trace()
    ex = TtrtcExecution(SimpleNamespace(get_action=infer), threading.RLock(), trace,
                       prefix_steps=prefix_steps, clock=lambda: clock[0])
    return ex, clock, trace


def tick(ex, clock, when):
    clock[0] = when
    with ex.lock:
        return ex.tick(when, 'pick', ['arm_left', 'arm_right'], 1)


def join(ex):
    ex.thread.join(2.)
    assert not ex.thread.is_alive()


def test_cold_start_empty_prefix_installs_exactly_24_actions():
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions.copy())
        return response(ttrtc_prefix_actions)
    ex, clock, trace = executor(infer)
    assert tick(ex, clock, 0.) is None
    join(ex)
    assert prefixes[0].shape == (0, 16)
    assert ex.buffer_size == 24
    np.testing.assert_array_equal(tick(ex, clock, 1/15), np.full(16, 10.))


def test_prefix_captured_before_publish_and_not_executed_twice():
    entered, release = threading.Event(), threading.Event()
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions.copy())
        entered.set()
        assert release.wait(2.)
        return response(ttrtc_prefix_actions, value=99.)
    ex, clock, trace = executor(infer)
    ex.actions.extend(np.full(16, i) for i in range(7))
    np.testing.assert_array_equal(tick(ex, clock, 0.), np.zeros(16))
    assert entered.wait(1.)
    for i in range(1, 4):
        np.testing.assert_array_equal(tick(ex, clock, i/15), np.full(16, i))
    clock[0] = .25
    release.set()
    join(ex)
    np.testing.assert_array_equal(prefixes[0][:, 0], np.arange(7))
    assert ex.buffer_size == 3 + 24
    for i in range(4, 7):
        np.testing.assert_array_equal(tick(ex, clock, i/15), np.full(16, i))
    np.testing.assert_array_equal(tick(ex, clock, 7/15), np.full(16, 99.))


@pytest.mark.parametrize('prefix_steps', range(9))
def test_all_supported_prefix_lengths(prefix_steps):
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions.copy())
        return response(ttrtc_prefix_actions)
    ex, clock, _ = executor(infer, prefix_steps)
    ex.actions.extend(np.zeros(16) for _ in range(prefix_steps))
    tick(ex, clock, 0.)
    join(ex)
    assert len(prefixes[0]) == prefix_steps
    assert ex.buffer_size == max(0, prefix_steps-1)+24


def test_late_response_is_discarded_and_next_request_cold_starts():
    entered, release = threading.Event(), threading.Event()
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions.copy())
        if len(prefixes) == 1:
            entered.set()
            assert release.wait(2.)
        return response(ttrtc_prefix_actions)
    ex, clock, trace = executor(infer)
    ex.actions.extend(np.zeros(16) for _ in range(7))
    tick(ex, clock, 0.)
    assert entered.wait(1.)
    for i in range(1, 8):
        tick(ex, clock, i/15)
    clock[0] = .51
    release.set()
    join(ex)
    assert ex.buffer_size == 0
    assert any(e == 'ttrtc_chunk_rejected' and f['reason'] == 'prefix_deadline'
               for e, f in trace.events)
    tick(ex, clock, .7)
    join(ex)
    assert prefixes[1].shape == (0, 16)
    assert ex.buffer_size == 24


def test_reset_discards_inflight_result():
    entered, release = threading.Event(), threading.Event()
    def infer(task, *, ttrtc_prefix_actions):
        entered.set()
        assert release.wait(2.)
        return response(ttrtc_prefix_actions)
    ex, clock, _ = executor(infer)
    tick(ex, clock, 0.)
    assert entered.wait(1.)
    with ex.lock:
        ex.reset()
    release.set()
    join(ex)
    assert ex.buffer_size == 0


def test_control_tick_gap_discards_old_actions_and_uses_empty_prefix():
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions.copy())
        return response(ttrtc_prefix_actions)
    ex, clock, trace = executor(infer)
    ex.actions.extend(np.ones(16) for _ in range(24))
    tick(ex, clock, 0.)
    assert tick(ex, clock, .5) is None
    join(ex)
    assert prefixes[0].shape == (0, 16)
    assert any(e == 'ttrtc_timeline_reset' for e, _ in trace.events)


@pytest.mark.parametrize('matching_clock, accepted', [(True, False), (False, True)])
def test_late_observation_start_uses_only_matching_monotonic_clock(matching_clock, accepted):
    def infer(task, *, ttrtc_prefix_actions):
        result = response(ttrtc_prefix_actions)
        result.timing_json = json.dumps({'clock_id': ex.clock_id if matching_clock else 'other-host',
                                         'sensor_snapshot_monotonic_s': .2})
        return result
    ex, clock, trace = executor(infer)
    ex.clock_id = 'local-host'
    ex.actions.extend(np.zeros(16) for _ in range(7))
    tick(ex, clock, 0.)
    join(ex)
    assert ex.buffer_size == (30 if accepted else 6)


@pytest.mark.parametrize('kind', ['nan', 'prefix_changed', 'wrong_horizon', 'wrong_dim'])
def test_malformed_response_preserves_existing_finite_tail(kind):
    def infer(task, *, ttrtc_prefix_actions):
        result = response(ttrtc_prefix_actions)
        if kind == 'nan':
            result.action_list[-1] = np.nan
        elif kind == 'prefix_changed':
            result.action_list[0] = 123.
        elif kind == 'wrong_horizon':
            result.chunk_size = 31
        else:
            result.action_dim = 8
        return result
    ex, clock, trace = executor(infer)
    ex.actions.extend(np.zeros(16) for _ in range(7))
    tick(ex, clock, 0.)
    join(ex)
    assert ex.buffer_size == 6
    assert any(e == 'ttrtc_chunk_rejected' for e, _ in trace.events)


@pytest.mark.parametrize('kwargs', [{'policy_hz': 30}, {'prefix_steps': 9},
                                    {'prefix_steps': -1}, {'prefix_steps': True}])
def test_rejects_invalid_timing(kwargs):
    with pytest.raises(ValueError):
        TtrtcExecution(object(), threading.RLock(), Trace(), **kwargs)
