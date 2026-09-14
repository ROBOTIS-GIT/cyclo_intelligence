"""ControlLoop -> requester -> real Worker handler, without robot/network/model I/O."""

from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from .test_control_loop import ControlLoop, FakeRobot, FakeRequester, control_loop_module
from .test_execution_context import ContextEngine
from engine_process.worker import EngineWorker
from main_runtime.inference_requester import InferenceRequester
from inference_context.execution import ExecutionContext, ActionRecord, ResetRecord


class ReceiptRobot(FakeRobot):
    def publish_action_with_receipt(self, action, action_keys, *, zero_twist=False):
        self.publish_action(action, action_keys)
        emitted = np.array(action, copy=True)
        emitted[0] = 0.
        return emitted


class ChunkEngine(ContextEngine):
    def get_action_chunk(self, request):
        self.last_prediction_id = request.prediction_id
        super().get_action_chunk(request)
        return {"success": True, "chunk_size": 3, "action_dim": 2,
                "action_chunk": np.array([[1., 2.], [3., 4.], [5., 6.]])}


@pytest.fixture
def rig():
    engine = ChunkEngine()
    worker = EngineWorker(engine)
    requests = []

    class Client:
        def call(self, req, timeout_s):
            requests.append(req)
            return worker.handle(req)

    requester = InferenceRequester(Client())
    initial = ExecutionContext("test-session", 0, 0, "ready")
    assert requester.load_policy(SimpleNamespace(execution_context_json=initial.to_json())).success
    robot = ReceiptRobot()
    loop = ControlLoop(requester, postprocess_actions=False)
    with mock.patch.object(control_loop_module, "RobotClient", return_value=robot):
        loop.configure("test", publish_to_robot=True, execution_context=initial)
    loop.start()
    yield loop, robot, engine, requests
    for thread in (loop._request_thread, loop._feedback_thread):
        if thread is not None:
            thread.join(2)
            assert not thread.is_alive()
    loop.deconfigure()


def fill(loop):
    loop._request_and_buffer("move", loop._generation, "sync")


def tick_without_refill(loop):
    with mock.patch.object(loop, "_should_request_actions", return_value=False):
        loop.tick()


def flush(loop):
    thread = loop._feedback_thread
    if thread:
        thread.join(2)
        assert not thread.is_alive()


def test_actual_receipt_and_remaining_plan_reach_next_worker_prediction(rig):
    loop, robot, engine, _ = rig
    fill(loop)
    first_prediction_id = engine.last_prediction_id
    tick_without_refill(loop)
    fill(loop)
    ctx = engine.contexts[-1]
    published = [a for a in ctx.actions if a.status == "published"]
    pending = [a for a in ctx.actions if a.status == "planned"]
    assert published[0].values == (0., 2.)
    assert published[0].planned_values == (1., 2.)
    assert published[0].source_position == 0.
    assert published[0].command_id == 0
    assert published[0].prediction_id == str(first_prediction_id)
    assert [a.values for a in pending] == [(3., 4.), (5., 6.)]
    assert ctx.planning[0].source_start == 0
    assert ctx.after_event_id == 0 and ctx.latest_event_id == 2
    assert len(robot.commands) == 1
    assert ExecutionContext.from_json(ctx.to_json()) == ctx
    fill(loop)
    assert engine.contexts[-1].after_event_id == 2
    assert not any(a.status == "published" for a in engine.contexts[-1].actions)


def test_long_running_feedback_acks_keep_wire_deltas_bounded_after_retention_rollover(rig):
    loop, robot, engine, _ = rig
    cursor = 0
    wire_sizes = []
    for _ in range(1100):
        fill(loop)
        context = engine.contexts[-1]
        assert context.after_event_id == cursor
        terminal_ids = sorted(
            [a.event_id for a in context.actions if a.event_id is not None]
            + [p.event_id for p in context.planning]
            + [r.event_id for r in context.resets]
        )
        assert terminal_ids == list(range(cursor + 1, context.latest_event_id + 1))
        assert len(terminal_ids) <= 4  # One planning event and three publication receipts.
        cursor = context.latest_event_id
        raw = context.to_json()
        assert ExecutionContext.from_json(raw) == context
        if terminal_ids:
            wire_sizes.append(len(raw.encode("utf-8")))
        for _ in range(3):
            tick_without_refill(loop)
        assert len(loop._processor._events) <= 4096
    assert cursor > 4096
    assert len(robot.commands) == 3300
    # Only ID/timestamp widths change; old ACKed receipts do not accumulate on wire.
    assert max(wire_sizes) - min(wire_sizes) < 256


def test_pause_discards_remaining_plan_and_resets_before_resume(rig):
    loop, robot, engine, _ = rig
    fill(loop)
    tick_without_refill(loop)
    generation = engine.contexts[-1].generation
    assert loop.pause()
    flush(loop)
    paused = engine.contexts[-1]
    assert paused.phase == "paused"
    assert paused.generation > generation
    assert robot.holds
    assert [a.values for a in paused.actions if a.status == "discarded"] == [(3., 4.), (5., 6.)]
    assert paused.resets[-1].reason == "pause"
    loop.start()
    fill(loop)
    assert engine.contexts[-1].generation == paused.generation
    assert engine.contexts[-1].phase == "running"
    assert loop._processor.buffer_size == 3


def test_preview_is_not_recorded_as_robot_publication(rig):
    loop, robot, engine, _ = rig
    loop.set_publish_to_robot(False)
    fill(loop)
    tick_without_refill(loop)
    fill(loop)
    assert not robot.commands
    assert robot.previews
    assert any(a.reason == "preview-only" and a.status == "discarded" for a in engine.contexts[-1].actions)
    assert not any(a.status == "published" for a in engine.contexts[-1].actions)


def test_publication_failure_clears_plan_and_hold_failure_remains_retryable(rig):
    loop, robot, engine, _ = rig
    fill(loop)
    robot.hold_failures_remaining = 1
    robot.publish_action_with_receipt = mock.Mock(side_effect=RuntimeError("partial publish"))
    tick_without_refill(loop)
    flush(loop)
    assert not loop._running
    assert loop._processor.buffer_size == 0
    assert loop.initial_pose_sync_hold_required()
    assert engine.contexts[-1].phase == "error"
    assert any(a.status == "failed" for a in engine.contexts[-1].actions)
    with pytest.raises(RuntimeError, match="hold is still pending"):
        loop.start()
    assert loop.pause()
    flush(loop)
    assert engine.contexts[-1].phase == "paused"


def test_slow_old_prediction_cannot_block_stop_or_fault_a_resumed_session(rig):
    loop, robot, engine, _ = rig
    entered, release = threading.Event(), threading.Event()
    original = engine.get_action_chunk
    faults = []
    loop.set_fault_callback(lambda *args: faults.append(args))

    def slow(request):
        entered.set()
        assert release.wait(2)
        return {"success": False, "message": "late old request failed"}

    engine.get_action_chunk = slow
    request = threading.Thread(target=fill, args=(loop,))
    request.start()
    try:
        assert entered.wait(2)
        # Stop must complete before a blocked Worker can acknowledge it.
        assert loop.pause()
        assert robot.holds
        assert not release.is_set()
        loop.start()
    finally:
        release.set()
        request.join(2)
    flush(loop)
    assert not request.is_alive()
    assert loop._running
    assert not faults
    assert loop._processor.buffer_size == 0
    engine.get_action_chunk = original
    fill(loop)
    assert loop._processor.buffer_size == 3


def test_pose_sync_uses_a_new_generation_before_fresh_prediction(rig):
    loop, robot, engine, _ = rig
    loop._initial_pose_sync_enabled = True
    assert loop.start()
    initial_generation = engine.contexts[-1].generation
    assert robot.sync_targets
    assert loop._processor.buffer_size == 0
    loop._initial_pose_sync_deadline = 0
    tick_without_refill(loop)
    fill(loop)
    ctx = engine.contexts[-1]
    assert ctx.generation > initial_generation
    assert ctx.phase == "running"
    assert any(r.reason == "initial pose sync complete" for r in ctx.resets)


def test_oversized_feedback_fails_closed_without_silent_truncation(rig, monkeypatch):
    loop, robot, engine, _ = rig
    # Inject a small decoded budget to test the safety path without large allocation.
    monkeypatch.setattr("inference_context.execution.MAX_EXPANDED_CONTEXT_BYTES", 65536)
    loop._processor.enqueue(10, np.random.default_rng(42).normal(size=(300, 22)))
    fill(loop)
    flush(loop)
    assert not loop._running
    assert loop._processor.buffer_size == 0
    assert engine.calls == 0
    assert robot.holds


def test_feedback_event_ranges_and_provenance_are_validated_without_large_allocation():
    initial = ExecutionContext("s", 0, 0, "ready")
    with pytest.raises(ValueError, match="contiguous"):
        replace(initial, latest_event_id=10**18)
    with pytest.raises(ValueError, match="contiguous"):
        replace(initial, latest_event_id=2, resets=(ResetRecord(2, "stop", 1.),))
    with pytest.raises(ValueError, match="anchor"):
        ActionRecord("1", "planned", "command", (1.,), blend_weight=0.5)


def test_lost_update_ack_does_not_apply_events_twice(rig):
    loop, _, engine, _ = rig
    fill(loop)
    tick_without_refill(loop)
    original = loop._requester.update_context

    def lost_ack(context):
        assert original(context).success
        raise TimeoutError("ACK lost after Worker consumed events")

    with mock.patch.object(loop._requester, "update_context", side_effect=lost_ack):
        assert loop.stop()
        flush(loop)
    stopped = engine.contexts[-1]
    assert stopped.phase == "stopped"
    assert any(a.status == "published" for a in stopped.actions)
    loop.start()
    fill(loop)
    resumed = engine.contexts[-1]
    assert resumed.after_event_id == stopped.latest_event_id
    assert resumed.latest_event_id == stopped.latest_event_id
    assert not resumed.actions and not resumed.planning and not resumed.resets
    assert loop._processor.buffer_size == 3


def test_concurrent_ticks_cannot_reserve_two_requests_before_thread_start(rig):
    loop, _, engine, _ = rig
    scheduled = []

    class DeferredThread:
        def __init__(self, target, args=(), **kwargs):
            self.target, self.args = target, args
            scheduled.append(self)
        def start(self):
            pass
        def is_alive(self):
            return False
        def join(self, timeout):
            pass

    with mock.patch.object(control_loop_module.threading, "Thread", DeferredThread):
        loop.tick()
        loop.tick()
    assert len(scheduled) == 1
    assert loop._request_reserved
    scheduled[0].target(*scheduled[0].args)
    assert not loop._request_reserved
    assert engine.calls == 1


def test_sync_idle_publication_failure_uses_local_safety_path(rig):
    loop, robot, engine, _ = rig
    loop._initial_pose_sync_enabled = True
    assert loop.start()
    robot.publish_idle_action = mock.Mock(side_effect=RuntimeError("velocity publisher failed"))
    loop.tick()
    flush(loop)
    assert not loop._running
    assert robot.holds
    assert engine.contexts[-1].phase == "error"


def test_context_rejects_fabricated_ack_cursor(rig):
    loop, _, _, _ = rig
    initial = ExecutionContext("test-session", 0, 1, "running", after_event_id=10, latest_event_id=10)
    response = loop._requester.update_context(initial)
    assert not response.success
    assert "skips unacknowledged" in response.message


@pytest.mark.parametrize("postprocess", [True, False])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_contextual_control_loop_keeps_legacy_chunk_command_values(rig, postprocess, mode):
    loop, robot, _, _ = rig
    response = SimpleNamespace(success=True, chunk_size=3, action_dim=2,
                               action_list=[1., 2., 3., 4., 5., 6.], message="ok")
    legacy = ControlLoop(FakeRequester(response), postprocess_actions=postprocess)
    legacy_robot = FakeRobot()
    loop._postprocess_actions = postprocess
    initial = ExecutionContext("test-session", 0, 0, "ready")
    with mock.patch.object(control_loop_module, "RobotClient", return_value=robot):
        loop.configure("test", publish_to_robot=True, execution_context=initial)
    with mock.patch.object(control_loop_module, "RobotClient", return_value=legacy_robot):
        legacy.configure("test", publish_to_robot=True)
    loop.start()
    legacy.start()
    with mock.patch.object(control_loop_module.time, "monotonic", return_value=100.):
        for iteration in range(8):
            for current in (loop, legacy):
                current._request_and_buffer("move", current._generation, mode)
            assert loop._processor.buffer_size == legacy._processor.buffer_size
            assert loop._tick_period() == legacy._tick_period()
            for _ in range(max(1, legacy._processor.buffer_size // 2)):
                tick_without_refill(loop)
                tick_without_refill(legacy)
    np.testing.assert_array_equal([a for a, _ in robot.commands], [a for a, _ in legacy_robot.commands])
    legacy.deconfigure()


def test_feedback_projection_does_not_hold_the_robot_control_lock(rig):
    loop, robot, _, _ = rig
    entered, release = threading.Event(), threading.Event()
    original = loop._feedback.project

    def slow_projection(capture):
        entered.set()
        assert release.wait(2)
        return original(capture)

    with mock.patch.object(loop._feedback, "project", side_effect=slow_projection):
        request = threading.Thread(target=fill, args=(loop,))
        request.start()
        try:
            assert entered.wait(2)
            assert loop.stop()
            assert robot.holds
        finally:
            release.set()
            request.join(2)
            flush(loop)
    assert not request.is_alive()
    assert not loop._running


def test_new_instruction_invalidates_old_plan_and_worker_context(rig):
    loop, _, engine, _ = rig
    fill(loop)
    generation = engine.contexts[-1].generation
    loop.set_task_instruction("different task")
    assert loop._processor.buffer_size == 0
    loop._request_and_buffer("different task", loop._generation, "sync")
    assert engine.contexts[-1].generation > generation
    assert any(r.reason == "instruction changed" for r in engine.contexts[-1].resets)
