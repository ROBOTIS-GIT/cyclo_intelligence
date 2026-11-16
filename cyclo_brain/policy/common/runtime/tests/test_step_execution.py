"""Step scheduling with real Worker/requester plumbing and a fake public policy."""

import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from .test_control_loop_feedback import ReceiptRobot, fill, flush, tick_without_refill
from .test_control_loop import ControlLoop, control_loop_module
from engine_process.worker import EngineWorker
from inference_context.contract import ExecutionContract
from inference_context.execution import ExecutionContext
from main_runtime.inference_requester import InferenceRequester
from main_runtime.step_schedule import StepSchedule


_path = Path(__file__).resolve().parents[3] / "lerobot/lerobot_engine/adapters/public_step.py"
_spec = importlib.util.spec_from_file_location("tested_step_adapter", _path)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)


class PublicPolicy:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, observation):
        self.calls += 1
        return np.array([[float(self.calls), 2.]])


class StepEngine:
    def __init__(self):
        self.policy = PublicPolicy()
        identity = lambda x: x
        self.adapter = _adapter.PublicStepAdapter(self.policy, identity, identity, np.asarray)
        self.contexts = []

    def load_policy(self, request):
        return {"success": True}

    def update_execution_context(self, context):
        self.adapter.update_execution_context(context)
        self.contexts.append(context)

    def get_action_chunk(self, request):
        chunk = self.adapter.predict({}, request.prediction_id)
        return {"success": True, "chunk_size": 1, "action_dim": 2, "action_chunk": chunk}


@pytest.fixture
def step_rig():
    engine = StepEngine()
    worker = EngineWorker(engine)

    class Client:
        def call(self, req, timeout_s):
            return worker.handle(req)

    requester = InferenceRequester(Client())
    context = ExecutionContext("step-session", 0, 0, "ready")
    assert requester.load_policy(SimpleNamespace(execution_context_json=context.to_json())).success
    robot = ReceiptRobot()
    # Force legacy settings that must NOT resample/blend a public single step.
    loop = ControlLoop(requester, postprocess_actions=True, target_chunk_size=32)
    clock = [100.0]
    with mock.patch.object(control_loop_module, "RobotClient", return_value=robot):
        loop.configure("test", publish_to_robot=True, execution_context=context,
                       execution_contract=ExecutionContract("step"), inference_hz=15, control_hz=100)
    with mock.patch.object(control_loop_module.time, "monotonic", side_effect=lambda: clock[0]):
        loop.start()
        yield loop, robot, engine, clock
        for thread in (loop._request_thread, loop._feedback_thread):
            if thread is not None:
                thread.join(2)
                assert not thread.is_alive()
        loop.deconfigure()


def test_step_waits_for_receipt_and_dataset_period_not_control_ticks(step_rig):
    loop, robot, engine, clock = step_rig
    assert loop._tick_period() == pytest.approx(.01)
    assert loop._inference_hz == 15
    assert loop._should_request_actions(loop._processor)
    fill(loop)
    assert loop._processor.buffer_size == 1
    assert not loop._should_request_actions(loop._processor)
    tick_without_refill(loop)
    for delta in (0.01, 0.03, 0.06):
        clock[0] = 100.0 + delta
        tick_without_refill(loop)
        assert not loop._should_request_actions(loop._processor)
    assert engine.policy.calls == 1
    assert len(robot.commands) == 4
    for values, _ in robot.commands:
        np.testing.assert_array_equal(values, [1., 2.])
    clock[0] = 100.07
    assert loop._should_request_actions(loop._processor)
    fill(loop)
    assert engine.policy.calls == 2
    receipts = [a for a in engine.contexts[-1].actions if a.status == "published"]
    assert len(receipts) == 4
    assert len({a.command_id for a in receipts}) == 4
    assert len({a.prediction_id for a in receipts}) == 1
    assert all(a.values == (0., 2.) and a.planned_values == (1., 2.) for a in receipts)
    tick_without_refill(loop)
    np.testing.assert_array_equal(robot.commands[-1][0], [2., 2.])
    assert not loop._should_request_actions(loop._processor)


def test_slow_step_holds_without_prefetch_and_pause_does_not_wait(step_rig):
    loop, robot, engine, clock = step_rig
    fill(loop)
    tick_without_refill(loop)
    clock[0] += .07
    entered, release = threading.Event(), threading.Event()
    original = engine.policy.select_action

    def slow(obs):
        entered.set()
        assert release.wait(2)
        return original(obs)

    engine.policy.select_action = slow
    loop.tick()
    assert entered.wait(2)
    try:
        for _ in range(3):
            clock[0] += .1
            loop.tick()
        assert engine.policy.calls == 1
        assert not loop._should_request_actions(loop._processor)
        assert loop.pause()
        count = len(robot.commands)
        loop.tick()
        assert len(robot.commands) == count
        assert robot.holds
    finally:
        release.set()
        loop._request_thread.join(2)
    flush(loop)
    assert loop._processor.buffer_size == 0
    assert engine.policy.resets >= 2
    assert engine.adapter._pending_id is None
    loop.start()
    assert loop._should_request_actions(loop._processor)
    fill(loop)
    assert loop._running


def test_step_failure_cannot_repeat_previous_target(step_rig):
    loop, robot, engine, _ = step_rig
    fill(loop)
    tick_without_refill(loop)
    robot.publish_action_with_receipt = mock.Mock(side_effect=RuntimeError("broken publisher"))
    tick_without_refill(loop)
    flush(loop)
    assert not loop._running
    assert loop._processor.take(repeat_last=True) is None
    assert engine.contexts[-1].phase == "error"


def test_sync_resets_step_cache_before_first_normal_prediction(step_rig):
    loop, _, engine, clock = step_rig
    loop._initial_pose_sync_enabled = True
    assert loop.start()
    assert engine.policy.calls == 1
    clock[0] += 6
    loop.tick()
    loop._request_thread.join(2)
    assert engine.policy.calls == 2
    assert engine.policy.resets >= 2
    assert loop._processor.buffer_size == 1


def test_preview_switch_is_explicitly_rejected(step_rig):
    loop, _, _, _ = step_rig
    with pytest.raises(ValueError, match="preview-only"):
        loop.set_publish_to_robot(False)
    assert loop._publish_to_robot


def test_step_contract_requires_feedback_before_robot_allocation():
    loop = ControlLoop(None)
    with mock.patch.object(control_loop_module, "RobotClient") as robot:
        with pytest.raises(ValueError, match="contextual"):
            loop.configure("test", execution_contract=ExecutionContract("step"))
        robot.assert_not_called()


def test_step_schedule_rejects_chunks_and_has_no_catchup_burst():
    schedule = StepSchedule(15)
    with pytest.raises(ValueError, match="exactly one"):
        schedule.accepted(1, np.zeros((2, 2)))
    schedule.accepted(1, np.zeros((1, 2)))
    assert not schedule.can_request(1000, 0)
    with pytest.raises(RuntimeError, match="not been published"):
        schedule.accepted(2, np.zeros((1, 2)))
    schedule.published(1, 1000)
    assert not schedule.can_request(1000.01, 0)
    assert schedule.can_request(2000, 0)
    schedule.accepted(2, np.zeros((1, 2)))
    assert not schedule.can_request(2000, 0)
    schedule.published(2, 2000)
    assert not schedule.can_request(2000.01, 0)
    schedule.reset()
    assert schedule.can_request(2000.01, 0)


def test_expired_step_requests_zero_velocity_without_modifying_prediction(step_rig):
    loop, robot, engine, clock = step_rig
    flags = []

    def publish(action, keys, *, zero_twist=False):
        flags.append(zero_twist)
        # Model a robot with one position and one velocity component.
        emitted = action.copy()
        if zero_twist:
            emitted[1] = 0.
        robot.commands.append((emitted.copy(), keys))
        return emitted

    robot.publish_action_with_receipt = publish
    fill(loop)
    tick_without_refill(loop)
    clock[0] += .06
    tick_without_refill(loop)
    clock[0] += .01
    tick_without_refill(loop)
    clock[0] += 1.
    tick_without_refill(loop)
    assert flags == [False, False, True, True]
    np.testing.assert_array_equal(robot.commands[-1][0], [1., 0.])
    fill(loop)
    ctx = engine.contexts[-1]
    expired = [a for a in ctx.actions if a.reason == "step period elapsed"]
    assert len(expired) == 2
    assert all(a.values == (1., 0.) and a.planned_values == (1., 2.) for a in expired)
    tick_without_refill(loop)
    assert flags[-1] is False  # A fresh prediction has a fresh lifetime.
    np.testing.assert_array_equal(robot.commands[-1][0], [2., 2.])


def test_step_expiry_is_scoped_to_the_current_prediction():
    schedule = StepSchedule(10)
    schedule.accepted(1, np.zeros((1, 2)))
    assert not schedule.velocity_expired(1, 100.)
    schedule.published(1, 100.)
    assert not schedule.velocity_expired(1, 100.05)
    assert schedule.velocity_expired(1, 100.1)
    schedule.accepted(2, np.zeros((1, 2)))
    assert not schedule.velocity_expired(2, 101.)
    assert not schedule.velocity_expired(1, 101.)
    schedule.reset()
    assert not schedule.velocity_expired(2, 1000.)
