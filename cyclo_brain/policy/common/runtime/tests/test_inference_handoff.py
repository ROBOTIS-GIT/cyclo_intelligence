"""Handoff must revoke output independently of lifecycle RPC latency."""

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from .test_control_loop import ControlLoop, FakeProcessor, FakeRobot
from .test_service_handler import FakeRequester as LifecycleRequester
from inference_context.execution import ExecutionContext
from main_runtime.execution_feedback import ExecutionFeedback
from main_runtime.service_handler import (
    CMD_PAUSE, CMD_RESUME, CMD_START, CMD_STOP, CMD_UNLOAD, CMD_UPDATE_INSTRUCTION,
    ServiceHandler,
)
from main_runtime.session_state import SessionState


class HandoffRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.status = SimpleNamespace(enabled=True, state="ready", reason="",
                                      revision=1, fault_revision=0, ready_since=1.0)
        self.arms = []

    def handoff_status(self):
        return self.status

    def arm_handoff(self, revision):
        assert self.status.state == "ready" and self.status.revision == revision
        self.arms.append(revision)
        return self.status

    def transition(self, state, reason=""):
        self.status = SimpleNamespace(
            enabled=True, state=state, reason=reason, revision=self.status.revision + 1,
            fault_revision=self.status.fault_revision + (state == "blocked"),
            ready_since=float(self.status.revision + 1),
        )


@pytest.fixture(params=["sync", "async"])
def mode(request):
    return request.param


def setup_session(*, publish=True, running=True, mode="async"):
    robot = HandoffRobot()
    loop = ControlLoop(LifecycleRequester())
    loop._robot = robot
    loop._processor = FakeProcessor(actions=[np.array([1.0])])
    loop._action_keys = ["arm"]
    loop._publish_to_robot = publish
    loop._action_request_mode = mode
    loop._handoff_context_supported = True
    session = SessionState()
    session.mark_loaded("sg2", "pick", ["arm"], model_path="/models/actual",
                        policy_id="lerobot:act", publish_to_robot=publish)
    handler = ServiceHandler(session, loop._requester, loop, lambda **kw: SimpleNamespace(**kw))
    if running:
        assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=publish)).success
    return handler, session, loop, robot


@pytest.mark.parametrize("source", ["teleop", "switching"])
def test_yield_clears_plan_without_hold_and_pauses_session(source, mode):
    handler, session, loop, robot = setup_session(mode=mode)
    generation = loop._generation
    robot.transition("yielded", source)
    loop.tick()
    handler.poll_handoff()
    assert not loop._running
    assert not loop._processor._actions
    assert loop._generation > generation
    assert not robot.commands and not robot.holds and not robot.idles
    assert session.paused and session.running and session.loaded
    assert handler.runtime_snapshot()["handoff_pending"]


def test_resume_keeps_loaded_session_without_load_or_new_ui_request(mode):
    handler, session, loop, robot = setup_session(mode=mode)
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    assert loop._running and not session.paused
    assert session.model_path == "/models/actual" and session.task_instruction == "pick"
    assert not handler.runtime_snapshot()["handoff_pending"]
    assert not robot.holds


@pytest.mark.parametrize("command", [CMD_PAUSE, CMD_STOP, CMD_UNLOAD])
def test_explicit_lifecycle_request_cancels_auto_resume(command):
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    assert handler.handle(SimpleNamespace(command=command)).success
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running
    assert not handler.runtime_snapshot()["handoff_pending"]
    assert not robot.holds


@pytest.mark.parametrize("command", [CMD_START, CMD_RESUME])
def test_manual_start_in_teleop_is_rejected(command):
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    response = handler.handle(SimpleNamespace(command=command, publish_to_robot=True, task_instruction="pick"))
    assert not response.success
    assert "model" in response.message.lower()
    assert not loop._running


def test_loaded_only_never_auto_starts():
    handler, session, loop, robot = setup_session(running=False)
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running and not session.running


@pytest.mark.parametrize("command", [CMD_START, CMD_RESUME])
def test_manual_start_prepares_local_model_and_records_started_revision(command):
    handler, session, loop, robot = setup_session(running=False)
    robot.transition("blocked", "source_missing")
    robot.request_model_start = Mock(side_effect=lambda: robot.transition("ready", "local_model"))
    if command == CMD_RESUME:
        session.mark_running()
        session.mark_paused()
    response = handler.handle(SimpleNamespace(command=command, publish_to_robot=True, task_instruction="pick"))
    assert response.success, response.message
    robot.request_model_start.assert_called_once_with()
    assert loop._running and not session.paused
    assert loop.handoff_status(started=True).reason == "local_model"
    assert handler._handoff_revision == robot.status.revision
    assert handler._handoff_fault_revision == robot.status.fault_revision


@pytest.mark.parametrize("publish", [True, False])
def test_poll_preview_and_automatic_resume_never_request_local_mode(publish):
    handler, session, loop, robot = setup_session(publish=publish)
    robot.request_model_start = Mock(side_effect=AssertionError("Not an explicit physical START"))
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    if not publish:
        loop.stop()
        loop.start(publish_to_robot=False)
    robot.request_model_start.assert_not_called()
    assert loop._running


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_real_guard_local_start_then_teleop_resume_and_feedback_fault(monkeypatch, mode):
    path = Path(__file__).resolve().parents[4] / "sdk/robot_client/robot_client/handoff.py"
    spec = importlib.util.spec_from_file_location("runtime_handoff_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    now = [10.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    guard = module.HandoffGuard({}, {"joint": {"a"}})
    joint = SimpleNamespace(name=["a"], position=[0.])
    control = SimpleNamespace(requested_control_mode=1, active_control_mode=1, state=1, transition_id=0)
    guard.update_joints("joint", joint)
    handler, session, loop, robot = setup_session(running=False, mode=mode)
    robot.handoff_status = guard.snapshot
    robot.arm_handoff = guard.arm
    robot.request_model_start = guard.request_model_start
    start = SimpleNamespace(command=CMD_START, publish_to_robot=True)
    assert handler.handle(start).success
    assert loop.handoff_status(started=True).reason == "local_model"
    handler.poll_handoff()
    assert loop._running

    guard.update_source(SimpleNamespace(data="teleop"))
    guard.update_status(control)
    guard.confirm_status(guard.confirmation_token(), control)
    handler.poll_handoff()
    assert session.paused and not loop._running and not robot.holds
    assert not handler.handle(start).success
    guard.update_source(SimpleNamespace(data="model_action"))
    guard.confirm_status(guard.confirmation_token(), control)
    handler.poll_handoff()
    assert loop._running and not session.paused

    now[0] += 1.
    guard.update_joints("joint", joint)
    guard.update_source(SimpleNamespace(data="model_action"))
    guard.confirm_status(guard.confirmation_token(), control)
    handler.poll_handoff()
    assert not loop._running and session.paused
    assert not handler.runtime_snapshot()["handoff_pending"]
    assert handler.handle(start).success


def test_preview_ignores_robot_source():
    handler, session, loop, robot = setup_session(publish=False)
    robot.transition("blocked", "source heartbeat lost")
    handler.poll_handoff()
    loop._processor.buffer_size = 100
    loop.tick()
    assert loop._running and session.running and not session.paused
    assert robot.previews and not robot.commands


def test_fast_roundtrip_discards_old_actions_even_if_poll_misses_teleop(mode):
    handler, session, loop, robot = setup_session(mode=mode)
    robot.transition("yielded", "teleop")
    robot.transition("ready")
    loop.tick()
    assert not robot.commands
    handler.poll_handoff()
    assert not loop._processor._actions


def test_fault_followed_by_recovery_never_auto_resumes():
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("blocked", "feedback stale")
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running
    assert not handler.runtime_snapshot()["handoff_pending"]
    assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success


def test_resume_waits_for_previous_prediction_thread(mode):
    handler, session, loop, robot = setup_session(mode=mode)
    entered, release = threading.Event(), threading.Event()
    def pending():
        entered.set()
        release.wait(5)
    thread = threading.Thread(target=pending)
    loop._request_thread = thread
    thread.start()
    assert entered.wait(1)
    try:
        robot.transition("yielded", "teleop")
        handler.poll_handoff()
        robot.transition("ready")
        handler.poll_handoff()
        assert not loop._running and session.paused
    finally:
        release.set()
        thread.join(2)
    handler.poll_handoff()
    assert loop._running


@pytest.mark.parametrize("successful", [True, False])
def test_late_response_cannot_restore_chunk_or_raise_fault(successful, mode):
    handler, session, loop, robot = setup_session(mode=mode)
    generation = loop._generation
    def predict(*args, **kwargs):
        robot.transition("yielded", "teleop")
        return SimpleNamespace(success=successful, message="old result", chunk_size=1,
                               action_dim=1, action_list=[99.0])
    loop._requester = SimpleNamespace(get_action=predict)
    loop._request_and_buffer("pick", generation, mode)
    handler.poll_handoff()
    assert not loop._processor.pushed_chunks
    assert not robot.holds
    assert session.paused and not session.error


def test_no_hold_stop_option_preserves_completed_slow_start():
    handler, session, loop, robot = setup_session()
    loop._initial_pose_sync_completed = True
    assert loop.stop(hold=False)
    assert loop._initial_pose_sync_completed
    assert not robot.holds


def test_shutdown_cancels_pending_auto_resume():
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    handler.begin_shutdown()
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running


def test_stop_default_still_publishes_hold_with_authority():
    handler, session, loop, robot = setup_session()
    assert loop.stop()
    assert len(robot.holds) == 1


def test_authority_loss_during_initial_prediction_cannot_publish_sync_target():
    handler, session, loop, robot = setup_session(running=False)
    loop._initial_pose_sync_enabled = True
    def predict(*args, **kwargs):
        robot.transition("yielded", "teleop")
        return SimpleNamespace(success=True, chunk_size=1, action_dim=1, action_list=[99.])
    loop._requester = SimpleNamespace(get_action=predict)
    result = handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True))
    assert result.success
    if loop._request_thread is not None:
        loop._request_thread.join(2)
    handler.poll_handoff()
    assert not robot.sync_targets and not robot.holds
    assert session.running and session.paused and handler._handoff_pending


def test_completed_sync_is_not_repeated_after_teleop():
    handler, session, loop, robot = setup_session()
    loop._initial_pose_sync_enabled = True
    loop._initial_pose_sync_completed = True
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    assert loop._running
    assert not robot.sync_targets


def test_fault_during_wait_cancels_resume_even_after_source_returns():
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    assert handler.fail_safe("worker heartbeat lost")
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running and session.error == "worker heartbeat lost"
    assert not robot.holds


def test_source_revocation_blocks_tick_even_while_lifecycle_lock_is_busy():
    handler, session, loop, robot = setup_session()
    entered, release = threading.Event(), threading.Event()
    def busy():
        with handler._lock:
            entered.set()
            release.wait(5)
    thread = threading.Thread(target=busy)
    thread.start()
    assert entered.wait(1)
    try:
        robot.transition("yielded", "teleop")
        handler.poll_handoff()
        loop.tick()
        assert not loop._running
        assert not robot.commands and not robot.idles and not robot.holds
    finally:
        release.set()
        thread.join(2)


def test_saved_pose_return_blocks_automatic_resume():
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    handler._pose_manager = SimpleNamespace(returning=True)
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running
    assert not handler._handoff_pending


def test_stop_after_fast_roundtrip_does_not_send_old_lease_hold():
    handler, session, loop, robot = setup_session()
    robot.transition("yielded", "teleop")
    robot.transition("ready")
    assert handler.handle(SimpleNamespace(command=CMD_STOP)).success
    assert not robot.holds


def test_fault_immediately_after_start_is_not_accepted_as_new_baseline():
    handler, session, loop, robot = setup_session(running=False)
    original = loop.start
    def start(**kwargs):
        result = original(**kwargs)
        robot.transition("blocked", "source lost")
        robot.transition("ready")
        return result
    loop.start = start
    assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success
    handler.poll_handoff()
    assert session.paused
    assert not loop._running and not handler._handoff_pending


def test_start_and_resume_explicitly_arm_only_the_current_ownership_revision():
    handler, session, loop, robot = setup_session()
    assert robot.arms == [1]
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    assert robot.arms == [1]
    robot.transition("ready")
    handler.poll_handoff()
    assert robot.arms == [1, 3]


def test_old_worker_without_reset_context_cannot_start_physical_handoff():
    handler, session, loop, robot = setup_session(running=False)
    loop._handoff_context_supported = False
    response = handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True))
    assert not response.success
    assert "Worker" in response.message and "context" in response.message
    assert not loop._running and not robot.arms


@pytest.mark.parametrize("command", [CMD_PAUSE, CMD_STOP, CMD_UNLOAD])
def test_rejected_user_cancel_cannot_be_rearmed_by_handoff(command, mode):
    handler, session, loop, robot = setup_session(mode=mode)
    robot.hold_failures_remaining = 1
    assert not handler.handle(SimpleNamespace(command=command)).success
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running
    assert not handler._handoff_pending
    # Only a new explicit start can restore eligibility.
    robot.hold_failures_remaining = 0
    assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    assert loop._running and not session.paused


@pytest.mark.parametrize("stage", ["before_start", "during_arm"])
def test_fault_during_auto_resume_cannot_become_new_baseline(stage, mode):
    handler, session, loop, robot = setup_session(mode=mode)
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    original = loop.prediction_pending if stage == "before_start" else robot.arm_handoff
    fired = False

    def race(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            robot.transition("blocked", "feedback stale")
            robot.transition("ready")
        return original(*args, **kwargs)

    if stage == "before_start":
        loop.prediction_pending = race
    else:
        robot.arm_handoff = race
    handler.poll_handoff()
    assert not loop._running and session.paused
    assert not handler._handoff_pending
    assert robot.arms == [1]


@pytest.mark.parametrize("restart", ["automatic", "manual"])
def test_delayed_control_fault_cannot_corrupt_a_new_run(restart, mode):
    handler, session, loop, robot = setup_session(mode=mode)
    entered = threading.Event()

    def callback(reason, hold_ok):
        entered.set()
        handler.on_control_fault(reason, hold_ok)

    loop.set_fault_callback(callback)
    generation = loop._generation
    thread = threading.Thread(target=loop._report_fault,
                              args=("old publication fault",), kwargs={"generation": generation})
    try:
        with handler._lock:
            thread.start()
            assert entered.wait(2)
            robot.transition("yielded", "teleop")
            robot.transition("ready")
            if restart == "manual":
                assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success
            else:
                handler.poll_handoff()
                assert not loop._running
    finally:
        thread.join(2)
        assert not thread.is_alive()
    if restart == "manual":
        assert loop._running and not session.error
    else:
        assert not loop._running
        assert not handler._handoff_pending


@pytest.mark.parametrize("paused", [False, True])
def test_instruction_update_preserves_current_fault_delivery(paused, mode, monkeypatch):
    handler, session, loop, robot = setup_session(mode=mode)
    loop._feedback = ExecutionFeedback(ExecutionContext("handoff-test", 0, 0, "ready"))
    loop._processor = loop._feedback.buffer
    monkeypatch.setattr(loop, "_schedule_feedback_update", lambda: None)
    if paused:
        assert handler.handle(SimpleNamespace(command=CMD_PAUSE)).success
    entered = threading.Event()

    def callback(reason, hold_ok):
        entered.set()
        handler.on_control_fault(reason, hold_ok)

    loop.set_fault_callback(callback)
    thread = threading.Thread(target=loop._report_fault, args=("current publication fault",),
                              kwargs={"generation": loop._generation})
    try:
        with handler._lock:
            thread.start()
            assert entered.wait(2)
            response = handler.handle(SimpleNamespace(
                command=CMD_UPDATE_INSTRUCTION, task_instruction="new task"))
            assert response.success
    finally:
        thread.join(2)
        assert not thread.is_alive()
    snapshot = handler.runtime_snapshot()
    assert snapshot["runtime_state"] == "error"
    assert snapshot["error"] == "current publication fault"
    assert session.task_instruction == "new task"
    assert not loop._running and not handler._handoff_eligible
    robot.transition("yielded", "teleop")
    handler.poll_handoff()
    robot.transition("ready")
    handler.poll_handoff()
    assert not loop._running and not handler._handoff_pending
    assert not robot.commands


@pytest.mark.parametrize("next_action", ["pause", "stop", "unload", "restart", "new_fault"])
def test_delayed_fault_preserves_newer_lifecycle_outcome(next_action, mode):
    handler, session, loop, robot = setup_session(mode=mode)
    entered = threading.Event()

    def callback(reason, hold_ok):
        entered.set()
        handler.on_control_fault(reason, hold_ok)

    loop.set_fault_callback(callback)
    thread = threading.Thread(target=loop._report_fault, args=("old publication fault",),
                              kwargs={"generation": loop._generation})
    try:
        with handler._lock:
            thread.start()
            assert entered.wait(2)
            if next_action == "new_fault":
                assert handler.fail_safe("new heartbeat fault")
            else:
                command = {"pause": CMD_PAUSE, "stop": CMD_STOP,
                           "unload": CMD_STOP, "restart": CMD_START}[next_action]
                assert handler.handle(SimpleNamespace(command=command, publish_to_robot=True)).success
                if next_action == "unload":
                    assert handler.handle(SimpleNamespace(command=CMD_UNLOAD)).success
            expected = handler.runtime_snapshot()
    finally:
        thread.join(2)
        assert not thread.is_alive()
    assert handler.runtime_snapshot() == expected
    assert loop._running == (next_action == "restart")
    assert not robot.commands


def test_repeated_start_before_fault_stop_preserves_current_fault(mode, monkeypatch):
    handler, session, loop, robot = setup_session(mode=mode)
    loop.set_fault_callback(handler.on_control_fault)
    emergency_stop = loop.emergency_stop

    def stop_after_start(reason, **kwargs):
        assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success
        return emergency_stop(reason, **kwargs)

    monkeypatch.setattr(loop, "emergency_stop", stop_after_start)
    loop._report_fault("current publication fault", generation=loop._generation)
    assert handler.runtime_snapshot()["runtime_state"] == "error"
    assert session.error == "current publication fault"
    assert not loop._running and not handler._handoff_eligible
    assert not robot.commands


def test_manual_start_after_missed_handoff_clears_previous_epoch_plan(mode):
    handler, session, loop, robot = setup_session(mode=mode)
    generation = loop._generation
    robot.transition("yielded", "teleop")
    robot.transition("ready")
    assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success
    assert loop._running
    assert loop._generation > generation
    assert not loop._processor._actions
    assert not robot.holds


def test_repeated_start_in_same_epoch_preserves_current_plan(mode):
    handler, session, loop, robot = setup_session(mode=mode)
    generation = loop._generation
    assert handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True)).success
    assert loop._generation == generation
    assert len(loop._processor._actions) == 1


@pytest.mark.parametrize("publish", [True, False])
@pytest.mark.parametrize("outcome", ["ready", "teleop", "publisher_changed"])
def test_clear_reload_waits_for_initial_handoff_but_start_still_checks(outcome, publish, monkeypatch):
    from main_runtime import control_loop as module

    loop = ControlLoop(LifecycleRequester())
    robots = []

    def make_robot(*args, **kwargs):
        robot = HandoffRobot()
        robot.transition("blocked", "publisher_changed")
        def wait():
            assert not loop._running
            assert not robot.commands and not robot.arms
            if outcome == "ready":
                robot.transition("ready", "model_action")
            elif outcome == "teleop":
                robot.transition("yielded", "teleop")
            return outcome != "publisher_changed"
        robot.wait_for_initial_handoff = Mock(side_effect=wait)
        robots.append(robot)
        return robot

    monkeypatch.setattr(module, "RobotClient", make_robot)
    for _ in range(10):
        loop.configure(robot_type="sg2", action_keys=["arm"], publish_to_robot=publish,
                       execution_context=ExecutionContext("reload", 0, 0, "ready"))
        robot = robots[-1]
        assert robot.wait_for_initial_handoff.call_count == int(publish)
        if publish and outcome != "ready":
            with pytest.raises(RuntimeError, match=outcome):
                loop.start(publish_to_robot=publish)
            assert not loop._running and not robot.arms
        else:
            loop.start(publish_to_robot=publish)
            assert loop._running
            loop.stop()
        loop.deconfigure()
        assert not robot.commands
    assert len(robots) == 10
