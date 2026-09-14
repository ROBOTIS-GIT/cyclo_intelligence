"""Bounded feedback input views preserve actual publication facts and ACK order."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_context import ExecutionQuery, InputAssembler, InputField, InputSpec
from inference_context.execution import ActionRecord, ExecutionContext, ResetRecord
from inference_context.execution_inputs import ExecutionInputs, ExecutionInputUnavailable


def make_inputs(*queries):
    spec = InputSpec(tuple(InputField(f"field{i}", (q,)) for i, q in enumerate(queries)))
    provider = ExecutionInputs(spec)
    ready = ExecutionContext("session", 0, 0, "ready")
    provider.update(ready)
    return provider, ready, InputAssembler(spec)


def published(event_id, value=1.):
    return ActionRecord("prediction", "published", "command", (value,), command_id=event_id,
                        event_id=event_id, recorded_s=100. + event_id / 100., planned_values=(2.,))


def test_only_requested_publications_are_retained_with_provenance_and_no_duplicates():
    query = ExecutionQuery("execution:published:command", count=2, min_count=0)
    provider, ready, assembler = make_inputs(query)
    assert assembler.assemble(provider)["field0"] == ()
    running = replace(ready, phase="running", revision=1, latest_event_id=4,
                      actions=(published(1), published(2), published(3), published(4)))
    provider.update(running)
    result = assembler.assemble(provider)["field0"]
    assert [r.event_id for r in result] == [3, 4]
    assert all(r.values == (1.,) and r.planned_values == (2.,) for r in result)
    assert provider.retained_record_count == 2
    provider.update(running)
    provider.update(replace(running, revision=2))  # Lost ACK, same facts retransmitted.
    assert assembler.assemble(provider)["field0"] == result
    assert provider.retained_record_count == 2
    provider.update(replace(running, revision=3, after_event_id=4, actions=()))
    assert assembler.assemble(provider)["field0"] == result  # Delta does not erase requested history.


def test_pending_plan_is_the_next_prefix_not_the_last_published_or_last_tail():
    query = ExecutionQuery("execution:pending:command", count=2, min_count=0)
    provider, ready, assembler = make_inputs(query)
    plan = tuple(ActionRecord("next", "planned", "command", (float(i),), command_id=i) for i in range(5))
    provider.update(replace(ready, revision=1, phase="running", actions=plan))
    assert [a.values for a in assembler.assemble(provider)["field0"]] == [(0.,), (1.,)]
    assert provider.retained_record_count == 2
    provider.update(replace(ready, revision=2, phase="running"))
    assert assembler.assemble(provider)["field0"] == ()


def test_reset_ignores_old_unacknowledged_commands_but_keeps_new_generation_events():
    query = ExecutionQuery("execution:published:command", count=3, min_count=0)
    events = ExecutionQuery("execution:events", count=4, min_count=0)
    provider, ready, assembler = make_inputs(query, events)
    running = replace(ready, phase="running", revision=1, latest_event_id=1, actions=(published(1),))
    provider.update(running)
    reset = ResetRecord(3, "stop", 100.03)
    resumed = replace(running, generation=1, revision=2, after_event_id=1, latest_event_id=4,
                      actions=(published(2), published(4, 9.)), resets=(reset,))
    provider.update(resumed)
    values = assembler.assemble(provider)
    assert [r.values for r in values["field0"]] == [(9.,)]
    assert values["field1"] == (reset, published(4, 9.))
    provider.clear()
    assert provider.retained_record_count == 0
    with pytest.raises(ExecutionInputUnavailable, match="LOAD context"):
        assembler.assemble(provider)


def test_context_view_exposes_the_validated_snapshot_without_an_extra_event_buffer():
    query = ExecutionQuery("execution:context")
    provider, ready, assembler = make_inputs(query)
    assert assembler.assemble(provider)["field0"] is ready
    assert provider.retained_record_count == 0


def test_missing_or_stale_publications_fail_without_fabricating_actions():
    query = ExecutionQuery("execution:published:command", count=2, min_count=1, max_age_s=.5)
    provider, ready, assembler = make_inputs(query)
    with pytest.raises(ExecutionInputUnavailable, match="have 0"):
        assembler.assemble(provider, anchor_s=100.)
    provider.update(replace(ready, revision=1, phase="running", latest_event_id=1, actions=(published(1),)))
    assert assembler.assemble(provider, anchor_s=100.1)["field0"] == (published(1),)
    with pytest.raises(ExecutionInputUnavailable, match="local monotonic clock"):
        assembler.assemble(provider, anchor_s=1.)
    with pytest.raises(ExecutionInputUnavailable, match="have 0"):
        assembler.assemble(provider, anchor_s=101.)


def test_stale_or_missing_feedback_is_not_accepted_as_complete_history():
    provider, ready, _ = make_inputs(ExecutionQuery("execution:published:command", min_count=0))
    running = replace(ready, revision=1, phase="running", latest_event_id=1, actions=(published(1),))
    provider.update(running)
    for invalid in (replace(running, session_id="other"), ready,
                    replace(running, phase="paused"),
                    replace(running, revision=2, after_event_id=5, latest_event_id=5, actions=())):
        with pytest.raises(ValueError):
            provider.update(invalid)


@pytest.mark.parametrize("kwargs", [
    {"source": "execution:executed"}, {"source": "execution:published"},
    {"count": 0}, {"count": 4097}, {"min_count": 2}, {"count": True},
    {"max_age_s": float("inf")}, {"max_age_s": -1},
    {"source": "execution:pending:command", "max_age_s": 1.},
])
def test_invalid_execution_queries_fail_at_load(kwargs):
    with pytest.raises(ValueError):
        ExecutionQuery(**{"source": "execution:published:command", **kwargs})
