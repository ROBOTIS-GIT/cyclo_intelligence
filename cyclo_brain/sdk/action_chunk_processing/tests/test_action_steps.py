"""Action Steps selects aligned source waypoints, never control ticks."""

import numpy as np
import pytest

from action_chunk_processing import ActionChunkProcessor
from action_chunk_processing.tracked_buffer import TrackedActionBuffer


@pytest.mark.parametrize('steps,count', [(0, 7), (1, 1), (3, 3), (50, 7)])
def test_selection_follows_alignment_before_resampling(steps, count):
    processor = ActionChunkProcessor(inference_hz=10, control_hz=100)
    chunk = np.arange(10.).reshape(-1, 1)
    anchor = np.array([2.])
    result = processor.prepare_chunk(chunk, anchor, action_steps=steps)
    assert result.source_start == 3
    expected = processor.prepare_chunk(chunk[3:3 + count], anchor, align=False)
    np.testing.assert_array_equal(result.actions, expected.actions)
    np.testing.assert_array_equal(chunk[:, 0], np.arange(10.))


@pytest.mark.parametrize('postprocess', [True, False])
@pytest.mark.parametrize('steps', [0, 1, 3, 50])
def test_tracked_and_legacy_paths_match_with_limited_source_provenance(postprocess, steps):
    settings = dict(inference_hz=10, control_hz=100, postprocess=postprocess)
    legacy, tracked = ActionChunkProcessor(**settings), TrackedActionBuffer(**settings)
    legacy.push_chunk(np.array([[2.]]))
    tracked.enqueue(0, np.array([[2.]]))
    chunk = np.arange(10.).reshape(-1, 1)
    count = legacy.push_chunk(chunk, action_steps=steps)
    decision = tracked.enqueue(1, chunk, action_steps=steps)
    assert decision.command_count == count
    assert decision.source_count == 10
    selected = min(10 - decision.source_start, steps or 10)
    while legacy.buffer_size:
        command = tracked.take()
        np.testing.assert_array_equal(command.values, legacy.pop_action())
        if command.prediction_id == 1:
            assert decision.source_start <= command.source_position < decision.source_start + selected
        tracked.finish(command.command_id, status='published', emitted_values=command.values)
    assert tracked.request_ready('plan_terminal')
    repeated = tracked.take(repeat_last=True)
    tracked.finish(repeated.command_id, status='published', emitted_values=repeated.values)
    assert tracked.request_ready('plan_terminal')
    assert tracked.snapshot().pending == ()


@pytest.mark.parametrize('value', [-1, True, 1.5, '2', None, 2147483648])
def test_invalid_limits_do_not_change_pending_plan(value):
    buffer = TrackedActionBuffer()
    before = buffer.snapshot()
    with pytest.raises(ValueError, match='action_steps'):
        buffer.enqueue(0, np.ones((5, 2)), action_steps=value)
    assert buffer.snapshot() == before


def test_empty_chunk_is_not_padded():
    buffer = TrackedActionBuffer()
    decision = buffer.enqueue(0, np.empty((0, 2)), action_steps=5)
    assert decision.command_count == 0
    assert buffer.take() is None


def test_selection_precedes_output_memory_budget():
    buffer = TrackedActionBuffer(max_commands=100, inference_hz=1, control_hz=100)
    assert buffer.enqueue(0, np.ones((100, 2)), align=False, action_steps=2).command_count == 100


@pytest.mark.parametrize('settings,steps', [({}, 1), ({'target_chunk_size': 3}, 0)])
def test_no_resampling_path_owns_queued_values(settings, steps):
    processor = ActionChunkProcessor(**settings)
    chunk = np.arange(6, dtype=np.float32).reshape(3, 2)
    expected = chunk.copy()
    count = processor.push_chunk(chunk, action_steps=steps)
    chunk[:] = np.nan
    for index in range(count):
        np.testing.assert_array_equal(processor.pop_action(), expected[index])


@pytest.mark.parametrize('postprocess', [True, False])
@pytest.mark.parametrize('alignment_mode', ['none', 'l2'])
@pytest.mark.parametrize('rates', [(15, 100), (30, 100), (100, 30)])
@pytest.mark.parametrize('seed', [17, 89])
def test_seeded_partial_consumption_preserves_values_and_source_bounds(
    postprocess, alignment_mode, rates, seed,
):
    rng = np.random.default_rng(seed)
    settings = dict(inference_hz=rates[0], control_hz=rates[1],
                    postprocess=postprocess, alignment_mode=alignment_mode)
    legacy = ActionChunkProcessor(**settings)
    tracked = TrackedActionBuffer(max_commands=65536, **settings)
    bounds = {}
    command_ids = set()

    def consume(count):
        for _ in range(count):
            command = tracked.take()
            expected = legacy.pop_action()
            assert command is not None and expected is not None
            np.testing.assert_array_equal(command.values, expected)
            lower, upper = bounds[command.prediction_id]
            assert lower <= command.source_position < upper
            assert command.command_id not in command_ids
            command_ids.add(command.command_id)
            tracked.finish(command.command_id, status='published', emitted_values=command.values)
        assert tracked.buffer_size == legacy.buffer_size

    for prediction in range(30):
        if prediction and prediction % 7 == 0:
            legacy.clear()
            tracked.clear()
        length = int(rng.integers(1, 33))
        steps = int(rng.choice([0, 1, 2, 7, 64]))
        chunk = rng.normal(size=(length, 7)).astype(np.float32)
        original = chunk.copy()
        delay = None if prediction % 3 == 0 else float(rng.uniform(0, 2))
        count = legacy.push_chunk(chunk, scheduled_start_delay_s=delay, action_steps=steps)
        decision = tracked.enqueue(prediction, chunk, scheduled_start_delay_s=delay, action_steps=steps)
        assert decision.command_count == count
        selected = min(length - decision.source_start, steps or length)
        bounds[prediction] = (decision.source_start, decision.source_start + selected)
        np.testing.assert_array_equal(chunk, original)
        chunk[:] = np.nan
        consume(int(rng.integers(0, legacy.buffer_size + 1)))

    consume(legacy.buffer_size)
    assert tracked.take() is None
    assert legacy.pop_action() is None
