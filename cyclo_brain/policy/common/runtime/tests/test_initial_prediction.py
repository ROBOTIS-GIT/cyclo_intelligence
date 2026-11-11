"""Opt-in first-prediction budgets never block Stop or change legacy deadlines."""

import threading
from unittest import mock

import pytest

from .test_control_loop_feedback import fill, flush, tick_without_refill
from .test_step_execution import step_rig, control_loop_module
from engine_process.protocol import CMD_GET_ACTION
from inference_context.contract import ExecutionContract, LoadedExecution
from inference_context.execution import ExecutionContext


@pytest.fixture
def preparing_rig(step_rig):
    loop, robot, engine, clock = step_rig
    with mock.patch.object(control_loop_module, "RobotClient", return_value=robot):
        loop.configure("test", publish_to_robot=True,
                       execution_context=ExecutionContext("step-session", 0, 0, "ready"),
                       execution_contract=ExecutionContract("step", 60.), inference_hz=15)
    yield loop, robot, engine, clock


@pytest.mark.parametrize("timeout", [False, "60", 0., -1., float("nan"), float("inf"), 121.])
def test_preparation_budget_validation(timeout):
    with pytest.raises(ValueError, match="timeout"):
        ExecutionContract("step", timeout)


def test_preparation_is_explicit_in_load_and_not_legal_for_legacy_chunk():
    contract = ExecutionContract("step", 60.)
    loaded = LoadedExecution(contract, ExecutionContext("s", 0, 0, "ready"))
    assert LoadedExecution.from_json(loaded.to_json()) == loaded
    assert LoadedExecution.from_json("").contract.initial_action_timeout_s is None
    with pytest.raises(ValueError):
        ExecutionContract("chunk", 60.)


def test_first_deadline_is_per_generation_and_later_calls_keep_five_seconds(preparing_rig):
    loop, _, _, clock = preparing_rig
    original = loop._requester._client.call
    budgets = []

    def record(request, timeout_s):
        if request.command == CMD_GET_ACTION:
            budgets.append(timeout_s)
        return original(request, timeout_s)

    with mock.patch.object(loop._requester._client, "call", side_effect=record):
        loop.start()
        assert loop.preparing()
        fill(loop)
        assert not loop.preparing()
        tick_without_refill(loop)
        clock[0] += .07
        fill(loop)
        assert loop.stop()
        flush(loop)
        loop.start()
        assert loop.preparing()
        fill(loop)
    assert budgets == [60., 5., 60.]


@pytest.mark.parametrize("pose_sync", [False, True])
def test_stop_during_first_prediction_is_local_and_discards_late_result(preparing_rig, pose_sync):
    loop, robot, engine, _ = preparing_rig
    loop._initial_pose_sync_enabled = pose_sync
    entered, release = threading.Event(), threading.Event()
    original = engine.policy.select_action
    faults = []
    loop.set_fault_callback(lambda *args: faults.append(args))
    robot.publish_initial_pose_sync = mock.Mock()

    def slow(observation):
        entered.set()
        assert release.wait(3)
        return original(observation)

    engine.policy.select_action = slow
    try:
        assert loop.start() == pose_sync
        if not pose_sync:
            loop.tick()
        assert entered.wait(2)
        assert loop.preparing()
        assert loop.prediction_pending()
        assert loop.stop()
        assert robot.holds
        assert not loop.preparing()
        assert not robot.commands
        with pytest.raises(RuntimeError, match="still finishing"):
            loop.start()
    finally:
        release.set()
        if loop._request_thread:
            loop._request_thread.join(3)
        flush(loop)
    assert not loop.prediction_pending()
    assert loop._processor.buffer_size == 0
    assert not faults
    robot.publish_initial_pose_sync.assert_not_called()
    assert not robot.commands


def test_pose_preparation_transitions_to_sync_then_prepares_after_reset(preparing_rig):
    loop, robot, _, clock = preparing_rig
    loop._initial_pose_sync_enabled = True
    assert loop.start()
    loop._request_thread.join(3)
    assert not loop._request_thread.is_alive()
    assert loop.initial_pose_sync_hold_required()
    assert not loop.preparing()
    clock[0] += 6
    tick_without_refill(loop)
    assert not loop.initial_pose_sync_hold_required()
    assert loop.preparing()
    fill(loop)
    assert not loop.preparing()
