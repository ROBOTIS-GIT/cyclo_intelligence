"""Runtime/Worker handoff contract, with no transport, model weights or ROS."""

import threading
import time

import pytest

from engine_process.protocol import CMD_GET_ACTION, CMD_LOAD_POLICY, EngineCommandRequest
from engine_process.worker import EngineWorker
from inference_context.contract import LoadedExecution
from inference_context.inputs import ReceivedValues, SampleQuery
from inference_context.observation import ExecutionObservationBarrier

from .test_control_loop import ControlLoop, control_loop_module
from .test_inference_handoff import HandoffRobot


class Observations:
    def __init__(self):
        self.values = {"camera:eye": 1., "joint:arm": 1.}
        self.stamps = dict.fromkeys(self.values, time.monotonic())
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stale = {source: threading.Event() for source in self.values}

    def receive(self, source, value):
        with self.lock:
            self.values[source] = value
            self.stamps[source] = time.monotonic()
            self.changed.notify_all()

    def wait_for_required_input_snapshot(self, sources, *, timeout_s, **options):
        deadline = time.monotonic() + timeout_s
        with self.changed:
            while True:
                try:
                    return self.get_required_input_snapshot(sources, **options)
                except ValueError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    self.changed.wait(remaining)

    def get_required_input_snapshot(self, sources, **options):
        with self.lock:
            received = ReceivedValues(self.values, self.stamps, after_s=options.get("after_s"))
            for source in sorted(sources):
                try:
                    received.resolve(SampleQuery(source), time.monotonic())
                except ValueError:
                    self.stale[source].set()
                    raise
            return self.values.copy()


class ResettableEngine:
    def __init__(self, observations):
        self.observations = observations
        self.barrier = ExecutionObservationBarrier()
        self.queue = []
        self.loads = 0
        self.predictions = []

    def load_policy(self, request):
        self.loads += 1
        return {"success": True}

    def update_execution_context(self, context):
        if self.barrier.update(context):
            self.queue.clear()

    def get_action_chunk(self, request):
        snapshot = self.barrier.read(self.observations, self.observations.values)
        if not self.queue:
            value = sum(snapshot.values())
            self.queue.extend([value, value + 100., value + 200.])
        value = self.queue.pop(0)
        self.predictions.append(value)
        return {"success": True, "chunk_size": 1, "action_dim": 1, "action_chunk": [value]}


class InProcessRequester:
    def __init__(self, worker, late_error):
        self.worker, self.late_error = worker, late_error
        self.requests = []
        self.omitted_updates = []
        self.old_response_ready = threading.Event()
        self.release_old_response = threading.Event()

    def get_action(self, instruction, *, context, **options):
        request = EngineCommandRequest(command=CMD_GET_ACTION, seq_id=len(self.requests) + 2,
                                       task_instruction=instruction, execution_context_json=context.to_json())
        self.requests.append(request)
        response = self.worker.handle(request)
        if len(self.requests) == 2:
            self.old_response_ready.set()
            assert self.release_old_response.wait(3)
            if self.late_error:
                raise RuntimeError("delayed old transport failure")
        return response

    def update_context(self, context):
        # Simulate a lost UPDATE_CONTEXT, without delivering or acknowledging it.
        self.omitted_updates.append(context)
        raise RuntimeError("UPDATE_CONTEXT omitted by test transport")


@pytest.mark.parametrize("late_error", [False, True], ids=["late-success", "late-error"])
@pytest.mark.parametrize("last_source", ["camera:eye", "joint:arm"])
def test_runtime_reset_reaches_worker_on_resume_get_action_without_paused_update(monkeypatch, late_error, last_source):
    observations = Observations()
    engine = ResettableEngine(observations)
    worker = EngineWorker(engine)
    loaded = worker.handle(EngineCommandRequest(command=CMD_LOAD_POLICY, seq_id=1))
    assert loaded.success, loaded.message
    execution = LoadedExecution.from_json(loaded.capabilities_json)
    assert execution.context is not None
    requester = InProcessRequester(worker, late_error)
    robot = HandoffRobot()
    monkeypatch.setattr(control_loop_module, "RobotClient", lambda *args, **kwargs: robot)
    faults = []
    loop = ControlLoop(requester, postprocess_actions=False, action_request_mode="sync",
                       fault_callback=lambda *args: faults.append(args))
    loop.configure("synthetic", action_keys=["arm"], publish_to_robot=True,
                   execution_context=execution.context, execution_contract=execution.contract)
    threads = []

    def predict_in_thread():
        thread = threading.Thread(target=loop._request_and_buffer,
                                  args=("pick", loop._generation, "sync"))
        loop._request_thread = thread
        threads.append(thread)
        thread.start()
        return thread

    try:
        loop.start()
        loop._request_and_buffer("pick", loop._generation, "sync")
        assert loop._processor.snapshot().pending[0].values == (2.,)
        initial_generation = loop._feedback.generation
        old = predict_in_thread()
        assert requester.old_response_ready.wait(2)
        assert engine.predictions == [2., 102.]

        robot.transition("yielded", "teleop")
        assert loop.stop(hold=False)
        feedback_thread = loop._feedback_thread
        assert loop._feedback.generation > initial_generation
        assert loop._processor.buffer_size == 0
        assert not robot.holds and not robot.commands
        robot.transition("ready")
        with pytest.raises(RuntimeError, match="previous prediction"):
            loop.start()
        requester.release_old_response.set()
        old.join(2)
        assert not old.is_alive()
        feedback_thread.join(2)
        assert not feedback_thread.is_alive()
        assert requester.omitted_updates
        assert worker._execution_context.generation == initial_generation
        assert engine.barrier.after_s is None
        assert loop._processor.buffer_size == 0 and not faults
        # A delayed successful response must not ACK away the Runtime reset.
        capture = loop._feedback.project(loop._feedback.capture())
        assert capture.resets and capture.after_event_id < capture.resets[-1].event_id

        loop.start()
        resume_started = time.monotonic()
        resumed = predict_in_thread()
        assert observations.stale["camera:eye"].wait(2)
        assert engine.barrier.after_s >= resume_started
        barrier = engine.barrier.after_s
        assert engine.queue == [] and engine.predictions == [2., 102.]
        first_source = "joint:arm" if last_source == "camera:eye" else "camera:eye"
        observations.receive(first_source, 5.)
        observations.stale[last_source].clear()
        assert observations.stale[last_source].wait(.5)
        assert resumed.is_alive() and loop._processor.buffer_size == 0
        observations.receive(last_source, 7.)
        resumed.join(2)
        assert not resumed.is_alive()
        assert loop._running and not faults
        assert engine.predictions == [2., 102., 12.]
        assert loop._processor.snapshot().pending[0].values == (12.,)
        assert worker._execution_context.generation == loop._feedback.generation > initial_generation
        assert worker._execution_context.phase == "running"
        assert engine.loads == 1
        assert not worker.handle(requester.requests[1]).success
        assert engine.barrier.after_s == barrier
        assert not robot.holds and not robot.commands
    finally:
        requester.release_old_response.set()
        for source in observations.values:
            observations.receive(source, 0.)
        for thread in threads:
            thread.join(3)
        feedback_thread = loop._feedback_thread
        if feedback_thread is not None:
            feedback_thread.join(3)
        loop.deconfigure()
