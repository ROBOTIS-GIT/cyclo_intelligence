from types import SimpleNamespace
from unittest import mock
import threading

import numpy as np
import pytest

from .test_control_loop import ControlLoop, FakeRobot, FakeRequester, control_loop_module
from inference_context.contract import ExecutionContract, LoadedExecution
from inference_context.execution import ExecutionContext
from main_runtime.execution_feedback import ExecutionFeedback


def test_feedback_schema_two_retains_ranges_and_legacy_omits_them():
    for schema in (1, 2):
        initial = ExecutionContext("s", 0, 0, "ready", feedback_schema=schema)
        contract = ExecutionContract(feedback_schema=schema)
        assert LoadedExecution.from_json(LoadedExecution(contract, initial).to_json()).contract == contract
        feedback = ExecutionFeedback(initial, postprocess=False)
        feedback.buffer.enqueue(1, np.ones((3, 2)))
        context = feedback.project(feedback.capture())
        raw = context.to_json()
        assert ("command_start_id" in raw) == (schema == 2)
        assert ExecutionContext.from_json(raw) == context


@pytest.mark.parametrize("condition,count,ticks", [("first_publication", 1, 1), ("published_count", 2, 2), ("plan_terminal", 1, 3)])
def test_request_gate_waits_locally_and_keeps_control_thread_free(condition, count, ticks):
    initial = ExecutionContext("s", 0, 0, "ready", feedback_schema=2)
    loop = ControlLoop(FakeRequester(SimpleNamespace(success=True)), postprocess_actions=False)
    with mock.patch.object(control_loop_module, "RobotClient", return_value=FakeRobot()):
        loop.configure("test", publish_to_robot=True, execution_context=initial,
                       execution_contract=ExecutionContract(feedback_schema=2, request_after=condition, request_after_count=count))
    try:
        buffer = loop._processor
        buffer.enqueue(1, np.ones((3, 2)))
        loop._action_request_mode = "async"
        with mock.patch.object(loop, "_refill_threshold", return_value=100):
            for i in range(ticks):
                assert not loop._should_request_actions(buffer)
                command = buffer.take()
                buffer.finish(command.command_id, status="published", emitted_values=command.values)
            assert loop._should_request_actions(buffer)
        # Neither operation performs Worker I/O or waits for a future callback.
        buffer.clear("stop")
        assert buffer.request_ready(condition, count)
    finally:
        loop.deconfigure()


def test_zero_plan_and_failed_publication_fail_closed():
    from action_chunk_processing.tracked_buffer import TrackedActionBuffer
    buffer = TrackedActionBuffer(postprocess=False)
    buffer.enqueue(1, np.empty((0, 2)))
    with pytest.raises(RuntimeError, match="discarded"):
        buffer.request_ready("plan_terminal")
    buffer.clear()
    buffer.enqueue(2, np.ones((2, 2)))
    command = buffer.take()
    buffer.finish(command.command_id, status="failed", reason="second topic failed")
    with pytest.raises(RuntimeError, match="failed"):
        buffer.request_ready("first_publication")


def test_waiting_for_plan_end_does_not_block_publication_status_or_stop():
    entered, release = threading.Event(), threading.Event()

    def update(context):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(success=True)

    class Robot(FakeRobot):
        def publish_action_with_receipt(self, action, action_keys):
            self.publish_action(action, action_keys)
            return np.array(action, copy=True)

    robot = Robot()
    requester = SimpleNamespace(update_context=update, get_action=mock.Mock())
    loop = ControlLoop(requester, postprocess_actions=False)
    with mock.patch.object(control_loop_module, "RobotClient", return_value=robot):
        loop.configure("test", publish_to_robot=True,
                       execution_context=ExecutionContext("s", 0, 0, "ready", feedback_schema=2),
                       execution_contract=ExecutionContract(feedback_schema=2, request_after="plan_terminal"))
    try:
        loop.start()
        loop._processor.enqueue(1, np.ones((10, 2)))
        loop._schedule_feedback_update()
        assert entered.wait(1)
        loop.tick()
        assert len(robot.commands) == 1
        assert loop.configuration_snapshot()["control_hz"] == 100
        requester.get_action.assert_not_called()
        assert loop.pause()
        assert robot.holds and loop._processor.buffer_size == 0
        assert not release.is_set()
    finally:
        thread = loop._feedback_thread
        release.set()
        if thread is not None:
            thread.join(3)
            assert not thread.is_alive()
        loop.deconfigure()
