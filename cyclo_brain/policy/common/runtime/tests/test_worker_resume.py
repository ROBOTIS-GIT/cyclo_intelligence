"""Worker-only handoff tests: no model weights, ROS or robot commands."""

from dataclasses import replace
import threading

from engine_process.protocol import CMD_GET_ACTION, CMD_LOAD_POLICY, CMD_UPDATE_CONTEXT, EngineCommandRequest
from engine_process.worker import EngineWorker
from inference_context.contract import LoadedExecution
from inference_context.execution import ExecutionContext, ResetRecord


class ResettableEngine:
    def __init__(self):
        self.queue = []
        self.loads = 0
        self.context = None

    def load_policy(self, request):
        self.loads += 1
        return {"success": True}

    def update_execution_context(self, context):
        if self.context is None or context.generation != self.context.generation:
            self.queue.clear()
        self.context = context

    def get_action_chunk(self, request):
        self.queue.append(request.prediction_id)
        return {"success": True, "chunk_size": 1, "action_dim": 1,
                "action_chunk": [float(request.prediction_id)]}


def command(worker, code, context=None, seq=1):
    return worker.handle(EngineCommandRequest(command=code, seq_id=seq,
                         execution_context_json=context.to_json() if context else ""))


def test_context_capable_legacy_load_negotiates_session_without_feedback_metadata():
    engine = ResettableEngine()
    worker = EngineWorker(engine)
    loaded = command(worker, CMD_LOAD_POLICY)
    assert loaded.success, loaded.message
    execution = LoadedExecution.from_json(loaded.capabilities_json)
    assert execution.context is not None
    assert execution.context.feedback_schema == 1
    assert execution.contract.mode == "chunk"
    assert engine.loads == 1


def test_resume_waits_for_prediction_then_discards_old_queue_and_cached_response():
    engine = ResettableEngine()
    worker = EngineWorker(engine)
    ready = ExecutionContext("session", 0, 0, "ready")
    assert command(worker, CMD_LOAD_POLICY, ready).success
    running = replace(ready, phase="running", revision=1)
    entered, release, reset_started, reset_done = (threading.Event() for _ in range(4))
    original = engine.get_action_chunk

    def predict(request):
        entered.set()
        assert release.wait(2)
        return original(request)

    engine.get_action_chunk = predict
    old_result, reset_result = [], []
    old = threading.Thread(target=lambda: old_result.append(command(worker, CMD_GET_ACTION, running, 2)))
    resumed = replace(running, generation=1, revision=2, latest_event_id=1,
                      resets=(ResetRecord(1, "resume", 9000000.),))

    def reset():
        reset_started.set()
        reset_result.append(command(worker, CMD_UPDATE_CONTEXT, resumed, 3))
        reset_done.set()

    update = threading.Thread(target=reset)
    old.start()
    try:
        assert entered.wait(2)
        update.start()
        assert reset_started.wait(2)
        assert not reset_done.wait(.05)
    finally:
        release.set()
        old.join(2)
        if update.ident is not None:
            update.join(2)
    assert old_result[0].success and reset_result[0].success
    assert engine.queue == []
    assert not command(worker, CMD_GET_ACTION, running, 2).success
    assert command(worker, CMD_GET_ACTION, resumed, 2).success
    assert engine.queue == [2] and engine.loads == 1


def test_reset_clock_is_local_and_duplicate_context_does_not_move_barrier(monkeypatch):
    from inference_context import observation

    tracker = observation.ExecutionObservationBarrier()
    ready = ExecutionContext("session", 0, 0, "ready")
    assert tracker.update(ready)
    assert tracker.after_s is None
    running = replace(ready, phase="running", revision=1)
    assert not tracker.update(running)
    resume = replace(running, generation=1, revision=2, latest_event_id=1,
                     resets=(ResetRecord(1, "resume", 9000000.),))
    monkeypatch.setattr(observation.time, "monotonic", lambda: 20.)
    assert tracker.update(resume)
    assert tracker.after_s == 20.
    monkeypatch.setattr(observation.time, "monotonic", lambda: 30.)
    assert not tracker.update(resume)
    assert tracker.after_s == 20.
    paused = replace(resume, phase="paused", revision=3)
    assert tracker.update(paused)
    monkeypatch.setattr(observation.time, "monotonic", lambda: 40.)
    assert tracker.update(replace(paused, phase="running", revision=4))
    assert tracker.after_s == 40.


def test_lost_ack_replay_and_stale_context_cannot_shift_resume_barrier(monkeypatch):
    from inference_context import observation

    class Engine(ResettableEngine):
        def __init__(self):
            super().__init__()
            self.barrier = observation.ExecutionObservationBarrier()

        def update_execution_context(self, context):
            self.barrier.update(context)
            super().update_execution_context(context)

    engine = Engine()
    worker = EngineWorker(engine)
    loaded = LoadedExecution.from_json(command(worker, CMD_LOAD_POLICY).capabilities_json)
    stopped = replace(loaded.context, generation=1, revision=1, phase="stopped", latest_event_id=1,
                      resets=(ResetRecord(1, "control source changed", 9000000.),))
    monkeypatch.setattr(observation.time, "monotonic", lambda: 10.)
    assert command(worker, CMD_UPDATE_CONTEXT, stopped).success
    resumed = replace(stopped, phase="running", revision=2)
    monkeypatch.setattr(observation.time, "monotonic", lambda: 20.)
    assert command(worker, CMD_UPDATE_CONTEXT, resumed).success
    assert engine.barrier.after_s == 20.
    monkeypatch.setattr(observation.time, "monotonic", lambda: 30.)
    assert command(worker, CMD_UPDATE_CONTEXT, resumed).success
    assert command(worker, CMD_UPDATE_CONTEXT, replace(resumed, revision=3)).success
    assert not command(worker, CMD_UPDATE_CONTEXT, stopped).success
    assert engine.barrier.after_s == 20.
