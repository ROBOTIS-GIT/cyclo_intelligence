from pathlib import Path
from collections import deque
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_chunk_processing import ActionChunkProcessor
from action_chunk_processing.tracked_buffer import TrackedActionBuffer, PublicationEvent, ResetEvent


@pytest.mark.parametrize("limit", [None, 0, 1, 3, 4096])
def test_snapshot_plan_prefix_preserves_buffer_and_all_receipts(limit):
    tracked = TrackedActionBuffer(postprocess=False)
    tracked.enqueue(1, np.arange(20.).reshape(10, 2))
    command = tracked.take()
    tracked.finish(command.command_id, status="published", emitted_values=command.values)
    full = tracked.snapshot()
    partial = tracked.snapshot(pending_limit=limit)
    assert partial.pending == (full.pending if limit is None else full.pending[:limit])
    assert partial.events == full.events
    assert partial.latest_event_id == full.latest_event_id
    assert tracked.snapshot() == full
    assert tracked.take() == full.pending[0]


@pytest.mark.parametrize("limit", [-1, 4097, True, 1.0, "3"])
def test_snapshot_rejects_invalid_plan_prefix(limit):
    with pytest.raises(ValueError, match="pending limit"):
        TrackedActionBuffer().snapshot(pending_limit=limit)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("processing", [
    {}, {"postprocess": False}, {"alignment_mode": "none"},
    {"inference_hz": 50, "control_hz": 15}, {"target_chunk_size": 7},
    {"target_chunk_size": 1}, {"inference_hz": 10, "control_hz": 10},
])
def test_tracing_preserves_exact_legacy_values_and_timing(dtype, processing):
    rng = np.random.default_rng(42)
    legacy, tracked = ActionChunkProcessor(**processing), TrackedActionBuffer(**processing)
    chunks = {}
    for prediction_id in range(15):
        chunk = rng.normal(size=(rng.choice([1, 2, 8, 16, 32]), 3)).astype(dtype)
        chunks[prediction_id] = chunk
        delay = [None, .2, 20][prediction_id % 3]
        align = prediction_id % 4 != 0
        expected = legacy.push_chunk(chunk, delay, align)
        receipt = tracked.enqueue(prediction_id, chunk, delay, align)
        assert receipt.command_count == expected
        assert len(tracked.snapshot().pending) == legacy.buffer_size
        # Partial drainage checks anchoring against the *tail* of remaining plan.
        count = legacy.buffer_size if prediction_id % 2 else legacy.buffer_size // 2
        for _ in range(count):
            command = tracked.take()
            original = legacy.pop_action()
            assert command.dtype == str(original.dtype)
            np.testing.assert_array_equal(command.values, original)
            source = chunks[command.prediction_id]
            lo = int(np.floor(command.source_position))
            hi = min(lo + 1, len(source) - 1)
            ratio = command.source_position - lo
            incoming = (1 - ratio) * source[lo].astype(float) + ratio * source[hi].astype(float)
            if command.blend_weight < 1:
                assert command.anchor_command_id is not None
                incoming = ((1 - command.blend_weight) * np.array(command.anchor_values)
                            + command.blend_weight * incoming)
            np.testing.assert_allclose(command.values, incoming, atol=1e-6, rtol=1e-6)
            tracked.finish(command.command_id, status="published", emitted_values=command.values)
        if prediction_id % 5 == 0:
            tracked.clear("pause")
            legacy.clear()
    while legacy.buffer_size:
        command = tracked.take()
        np.testing.assert_array_equal(command.values, legacy.pop_action())
        tracked.finish(command.command_id, status="published", emitted_values=command.values)
    assert tracked.take() is None


def test_actual_alignment_decision_is_independent_of_interpolated_length():
    tracked = TrackedActionBuffer(inference_hz=10, control_hz=100)
    tracked.enqueue(1, np.array([[2.]]))
    receipt = tracked.enqueue(2, np.arange(6.).reshape(-1, 1))
    assert receipt.source_start == 3
    assert receipt.source_count == 6
    commands = tracked.snapshot().pending[1:]
    assert receipt.command_count == len(commands) == 20
    assert commands[0].source_position == 3
    assert commands[-1].source_position == 4.9
    assert commands[0].anchor_command_id == 0
    assert tracked.snapshot().events[-1].decision == receipt


def test_dequeue_does_not_claim_publication_and_emitted_values_can_differ():
    tracked = TrackedActionBuffer(postprocess=False)
    tracked.enqueue(1, np.array([[.001, 1.], [2., 3.]]))
    before = tracked.snapshot()
    command = tracked.take()
    during = tracked.snapshot()
    assert len(during.pending) == 1
    assert during.in_flight == command
    assert during.events == before.events
    assert during.revision > before.revision
    with pytest.raises(RuntimeError, match="outstanding"):
        tracked.take()
    with pytest.raises(RuntimeError, match="outstanding"):
        tracked.clear()
    with pytest.raises(ValueError, match="actual emitted"):
        tracked.finish(command.command_id, status="published")
    tracked.finish(command.command_id, status="published", emitted_values=[0., 1.])
    after = tracked.snapshot()
    assert after.in_flight is None
    assert after.events[-1].emitted_values == (0., 1.)
    assert after.events[-1].command.values == (.001, 1.)
    with pytest.raises(ValueError, match="already completed"):
        tracked.finish(command.command_id, status="failed")
    tracked.clear("stop")
    events = tracked.snapshot().events
    assert [e.status for e in events if isinstance(e, PublicationEvent)] == ["published", "discarded"]
    assert events[-2].reason == "stop"
    assert isinstance(events[-1], ResetEvent)


def test_failed_publication_is_not_success_and_clear_removes_anchor():
    tracked = TrackedActionBuffer(target_chunk_size=1)
    tracked.enqueue(1, np.array([[10.]]))
    command = tracked.take()
    tracked.finish(command.command_id, status="failed", reason="publisher disconnected")
    assert tracked.snapshot().error == "publisher disconnected"
    with pytest.raises(RuntimeError, match="clear required"):
        tracked.take()
    with pytest.raises(RuntimeError, match="clear required"):
        tracked.enqueue(2, np.array([[100.]]))
    tracked.clear("error")
    tracked.enqueue(2, np.array([[100.]]))
    snapshot = tracked.snapshot()
    assert snapshot.events[1].status == "failed"
    assert snapshot.events[1].emitted_values is None
    assert snapshot.pending[0].anchor_command_id is None
    assert snapshot.pending[0].values == (100.,)
    assert snapshot.error == ""


def test_lagging_consumer_cannot_silently_receive_incomplete_execution_history():
    tracked = TrackedActionBuffer(postprocess=False, max_events=2)
    tracked.enqueue(1, np.ones((3, 2)))
    for _ in range(3):
        command = tracked.take()
        tracked.finish(command.command_id, status="discarded", reason="preview-only")
    with pytest.raises(RuntimeError, match="history gap"):
        tracked.snapshot(after_event_id=0)
    assert [e.event_id for e in tracked.snapshot(after_event_id=2).events] == [3, 4]
    assert not tracked.snapshot(after_event_id=4).events


@pytest.mark.parametrize("unacknowledged", [0, 1, 20, 4096])
def test_delta_snapshot_does_not_scan_acknowledged_history(unacknowledged):
    class CountingDeque(deque):
        visited = 0

        def __iter__(self):
            for event in super().__iter__():
                self.visited += 1
                yield event

        def __reversed__(self):
            for event in super().__reversed__():
                self.visited += 1
                yield event

    tracked = TrackedActionBuffer(postprocess=False)
    # Reset-only events also exercise rollover without allocating model actions.
    for _ in range(5000):
        tracked.clear("generation reset")
    tracked._events = CountingDeque(tracked._events, maxlen=4096)
    cursor = 5000 - unacknowledged
    snapshot = tracked.snapshot(after_event_id=cursor)
    assert [event.event_id for event in snapshot.events] == list(range(cursor + 1, 5001))
    assert tracked._events.visited <= unacknowledged + 1
    assert snapshot.latest_event_id == 5000


def test_delta_snapshot_matches_full_history_filter_across_rollover_and_reset():
    tracked = TrackedActionBuffer(postprocess=False, max_events=32)
    for prediction_id in range(40):
        tracked.enqueue(prediction_id, np.ones((3, 2)))
        command = tracked.take()
        tracked.finish(command.command_id, status="published", emitted_values=[1., 1.])
        tracked.clear("stop")
        retained = tuple(tracked._events)
        for cursor in range(retained[0].event_id - 1, retained[-1].event_id + 1):
            expected = tuple(event for event in retained if event.event_id > cursor)
            assert tracked.snapshot(after_event_id=cursor).events == expected
        if retained[0].event_id > 1:
            with pytest.raises(RuntimeError, match="history gap"):
                tracked.snapshot(after_event_id=retained[0].event_id - 2)


def test_invalid_requests_are_atomic_and_repeated_prediction_is_rejected():
    tracked = TrackedActionBuffer(max_commands=5, max_dimensions=2, postprocess=False)
    tracked.enqueue(1, np.ones((2, 2)))
    before = tracked.snapshot()
    for values in (np.ones((6, 2)), np.ones((2, 3)), np.full((2, 2), np.nan), np.ones((4, 2))):
        with pytest.raises(ValueError):
            tracked.enqueue(2, values)
        assert tracked.snapshot() == before
    with pytest.raises(ValueError, match="stale"):
        tracked.enqueue(1, np.ones((2, 2)))
    command = tracked.take()
    for kwargs in ({"status": "executed"}, {"status": "published", "emitted_values": [1.]},
                   {"status": "failed", "emitted_values": [1., 1.]},
                   {"status": "published", "emitted_values": [float("inf"), 1.]}):
        with pytest.raises(ValueError):
            tracked.finish(command.command_id, **kwargs)
        assert tracked.snapshot().in_flight == command


def test_budget_checked_before_resampling_large_output():
    tracked = TrackedActionBuffer(max_commands=10, control_hz=1e8)
    with pytest.raises(ValueError, match="budget"):
        tracked.enqueue(1, np.ones((2, 2)))
    assert not tracked.snapshot().pending


def test_zoh_repeat_has_individual_receipt_and_preserves_prediction_provenance():
    tracked = TrackedActionBuffer(postprocess=False)
    assert tracked.take(repeat_last=True) is None
    tracked.enqueue(5, np.array([[1., 2.]]))
    first = tracked.take()
    with pytest.raises(RuntimeError, match="outstanding"):
        tracked.take(repeat_last=True)
    tracked.finish(first.command_id, status="published", emitted_values=[0., 2.])
    assert tracked.take() is None  # legacy no-repeat semantics remain unchanged
    repeat = tracked.take(repeat_last=True)
    assert repeat.command_id != first.command_id
    assert repeat.prediction_id == first.prediction_id
    assert repeat.source_position == first.source_position
    assert repeat.values == first.values
    tracked.finish(repeat.command_id, status="published", emitted_values=[0., 2.])
    assert [e.command.command_id for e in tracked.snapshot().events
            if isinstance(e, PublicationEvent)] == [first.command_id, repeat.command_id]
    tracked.clear("stop")
    assert tracked.take(repeat_last=True) is None


@pytest.mark.parametrize("status", ["failed", "discarded"])
def test_unsent_command_is_not_a_zoh_target(status):
    tracked = TrackedActionBuffer(postprocess=False)
    tracked.enqueue(5, np.array([[1., 2.]]))
    command = tracked.take()
    tracked.finish(command.command_id, status=status)
    if status == "failed":
        with pytest.raises(RuntimeError, match="clear required"):
            tracked.take(repeat_last=True)
        tracked.clear("retry")
    assert tracked.take(repeat_last=True) is None
