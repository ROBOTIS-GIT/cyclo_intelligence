"""Bound pending-plan transport without altering execution or acknowledgement."""

import numpy as np
import pytest

from .test_control_loop import ControlLoop, FakeRobot, FakeRequester, control_loop_module
from unittest import mock
from inference_context.contract import ExecutionContract
from inference_context.execution import ExecutionContext
from main_runtime.execution_feedback import ExecutionFeedback


@pytest.mark.parametrize("waypoints", [64, 128])
@pytest.mark.parametrize("count", [0, 3])
def test_plan_budget_fits_payload_without_dropping_terminal_events(waypoints, count):
    initial = ExecutionContext("payload", 0, 0, "ready")
    feedback = ExecutionFeedback(initial, pending_command_count=count, control_hz=100, inference_hz=15)
    feedback.phase = "running"
    feedback.buffer.enqueue(1, np.random.default_rng(42).normal(size=(waypoints, 22)), align=False)
    full = feedback.buffer.snapshot()
    assert len(full.pending) > waypoints
    context = feedback.project(feedback.capture())
    assert len(context.actions) == count
    assert len(context.to_json().encode()) < 8192
    assert ExecutionContext.from_json(context.to_json()) == context
    assert feedback.buffer.snapshot() == full
    feedback.acknowledge(context)
    command = feedback.buffer.take()
    feedback.buffer.finish(command.command_id, status="published", emitted_values=command.values)
    published = feedback.project(feedback.capture())
    assert published.after_event_id == context.latest_event_id
    assert published.latest_event_id == context.latest_event_id + 1
    assert [a.command_id for a in published.actions if a.status == "published"] == [command.command_id]
    assert [a.command_id for a in published.actions if a.status == "planned"] == [
        cmd.command_id for cmd in full.pending[1:1 + count]]
    feedback.acknowledge(published)
    feedback.reset("pause", "paused")
    reset = feedback.project(feedback.capture())
    assert len(reset.actions) == len(full.pending) - 1  # Discards are receipts, not optional plan previews.
    assert reset.resets and reset.after_event_id == published.latest_event_id


def test_default_full_plan_roundtrips_above_compression_threshold():
    feedback = ExecutionFeedback(ExecutionContext("payload", 0, 0, "ready"), control_hz=100, inference_hz=15)
    feedback.buffer.enqueue(1, np.random.default_rng(42).normal(size=(128, 22)), align=False)
    context = feedback.project(feedback.capture())
    assert len(context.actions) == len(feedback.buffer.snapshot().pending)
    assert len(context.to_json().encode()) > 65536
    assert ExecutionContext.from_json(context.to_json()) == context


def test_control_loop_uses_negotiated_plan_count():
    loop = ControlLoop(FakeRequester(None), postprocess_actions=False)
    with mock.patch.object(control_loop_module, "RobotClient", return_value=FakeRobot()):
        loop.configure("test", publish_to_robot=True,
                       execution_context=ExecutionContext("runtime", 0, 0, "ready"),
                       execution_contract=ExecutionContract(pending_command_count=2))
    try:
        loop._feedback.buffer.enqueue(1, np.arange(10.).reshape(5, 2))
        assert len(loop._feedback.capture().snapshot.pending) == 2
        assert len(loop._feedback.buffer.snapshot().pending) == 5
    finally:
        loop.deconfigure()
