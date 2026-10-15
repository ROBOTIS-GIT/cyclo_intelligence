"""Supervised policy lifecycle tests with fake services, without model loading."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time
import sys

import pytest
from orchestrator.bt.actions.supervised_command import SupervisedCommand
from orchestrator.bt.inference_supervisor import InferenceSupervisor, HOLD_CAPABILITY
from orchestrator.bt.bt_core import NodeStatus
sys.path.insert(0, str(Path(__file__).parents[3] / 'docker'))
from supervisor_api.autonomy_runtime_control import validate_tree


def reply(message='ok', success=True):
    return SimpleNamespace(success=success, message=message, action_keys=[])


def done(value):
    future = Future(); future.set_result(value); return future


class Client:
    def __init__(self): self.requests = []; self.pending = {}; self.capability = HOLD_CAPABILITY
    def service_is_ready(self): return True
    def call_async(self, request):
        self.requests.append(request)
        return self.pending.get(request.command) or done(reply(self.capability if request.command == 7 else 'ok'))


class Node:
    robot_type = 'ffw_sg2_rev1'
    def __init__(self): self.client = Client()
    def create_client(self, *args, **kwargs): return self.client


class Owner:
    backend = 'groot'; target_str = 'INFERENCE'; inference_mode = 'robot'
    policy_path = '/workspace/model/groot/example'; task_instruction = 'pick'
    action_request_mode = 'sync'; acceleration_mode = 'pytorch'; acceleration_engine_path = ''
    control_hz = 100; inference_hz = 15; chunk_align_window_s = .3
    def __init__(self, command): self.command_str = command; self.log_error = Mock(); self._backend_job = None
    def _start_backend_job(self, operation):
        assert operation == 'START'
        self._backend_job = SimpleNamespace(snapshot=lambda: (True, 'ready'))


def run(action):
    for _ in range(25):
        result = action.tick()
        if result != NodeStatus.RUNNING: return result
    raise AssertionError('Action did not settle')


def loaded():
    node = Node(); supervisor = InferenceSupervisor(node)
    action = SupervisedCommand(Owner('LOAD'), supervisor)
    assert run(action) == NodeStatus.SUCCESS
    return node, supervisor


def test_load_never_starts_actuators_and_resume_is_explicit():
    node, supervisor = loaded()
    assert [r.command for r in node.client.requests] == [7, 4, 5, 0]
    assert supervisor.loaded and not supervisor.running
    assert node.client.requests[-1].publish_to_robot
    assert node.client.requests[-1].model_path == Owner.policy_path
    assert run(SupervisedCommand(Owner('RESUME'), supervisor)) == NodeStatus.SUCCESS
    assert node.client.requests[-1].command == 1 and supervisor.running
    assert node.client.requests[-1].initial_pose_sync is True
    assert node.client.requests[-1].initial_pose_sync_duration_s == 5.0


def test_pause_resume_preserves_policy_and_does_not_replay_load():
    node, supervisor = loaded()
    resume = SupervisedCommand(Owner('RESUME'), supervisor)
    assert run(resume) == NodeStatus.SUCCESS
    supervisor.pause(); assert not supervisor.running
    assert node.client.requests[-1].command == 4
    supervisor.resume(); assert supervisor.running
    assert node.client.requests[-1].command == 1
    assert [r.command for r in node.client.requests].count(0) == 1
    # Completed command is still complete, never reissued by resume.
    count = len(node.client.requests)
    assert resume.tick() == NodeStatus.SUCCESS
    assert len(node.client.requests) == count


def test_pause_during_load_does_not_wait_for_weights_or_start_after_reply():
    node = Node(); supervisor = InferenceSupervisor(node)
    pending = Future(); node.client.pending[0] = pending
    action = SupervisedCommand(Owner('LOAD'), supervisor)
    for _ in range(8): action.tick()
    assert supervisor.pending_command == 0
    before = time.monotonic(); supervisor.pause()
    assert time.monotonic() - before < .1
    pending.set_result(reply()); supervisor.resume()
    assert run(action) == NodeStatus.SUCCESS
    assert 1 not in [r.command for r in node.client.requests]


def test_pause_drains_pending_start_then_confirms_hold_once():
    node, supervisor = loaded(); pending = Future(); node.client.pending[1] = pending
    action = SupervisedCommand(Owner('RESUME'), supervisor)
    assert action.tick() == NodeStatus.RUNNING
    timer = threading.Timer(.02, lambda: pending.set_result(reply())); timer.start()
    supervisor.pause(); timer.join()
    assert [r.command for r in node.client.requests][-2:] == [1, 4]
    del node.client.pending[1]
    supervisor.resume()
    assert run(action) == NodeStatus.SUCCESS
    assert [r.command for r in node.client.requests].count(1) == 2


def test_stop_drains_pending_start_and_blocks_any_late_resume():
    node, supervisor = loaded(); pending = Future(); node.client.pending[1] = pending
    action = SupervisedCommand(Owner('RESUME'), supervisor); action.tick()
    timer = threading.Timer(.02, lambda: pending.set_result(reply())); timer.start()
    supervisor.stop(); timer.join()
    assert node.client.requests[-1].command == 4 and not supervisor.running
    with pytest.raises(RuntimeError, match='Stopped'): supervisor.resume()
    with pytest.raises(RuntimeError, match='late START'): supervisor.submit(1)


def test_clear_stops_before_unload_and_does_not_replay():
    node, supervisor = loaded()
    run(SupervisedCommand(Owner('RESUME'), supervisor))
    clear = SupervisedCommand(Owner('CLEAR'), supervisor)
    assert run(clear) == NodeStatus.SUCCESS
    assert [r.command for r in node.client.requests][-2:] == [4, 5]
    assert not supervisor.loaded and not supervisor.running
    before = len(node.client.requests); supervisor.complete()
    assert before == len(node.client.requests)


def test_old_main_fails_read_only_capability_before_load_or_start():
    node = Node(); node.client.capability = 'old runtime'
    owner = Owner('LOAD'); supervisor = InferenceSupervisor(node)
    assert run(SupervisedCommand(owner, supervisor)) == NodeStatus.FAILURE
    assert [r.command for r in node.client.requests] == [7]
    assert 'feedback-confirmed' in owner.log_error.call_args.args[0]


def test_home_mission_bt_files_all_pass_preflight():
    missions = Path(__file__).parents[3] / 'docker/workspace/navigation/missions/Home'
    for filename in missions.glob('*/locals/*/main.xml'):
        validate_tree(filename.read_text())


def test_unbounded_or_docker_policy_actions_rejected_before_execution():
    wrap = '<root main_tree_to_execute="M"><BehaviorTree ID="M"><Sequence>{}</Sequence></BehaviorTree></root>'
    load = '<SendCommand command="LOAD" model="groot:n17" policy_path="/model"/>'
    for body in (load + '<SendCommand command="RESUME"/>', '<SendCommand target="DOCKER" command="START" model="groot:n17"/>'):
        with pytest.raises(ValueError): validate_tree(wrap.format(body))


def test_timed_out_start_requests_remote_fault_hold_and_cannot_restart(monkeypatch):
    from orchestrator.bt import inference_supervisor as module
    node, supervisor = loaded()
    pending = Future(); node.client.pending[1] = pending
    action = SupervisedCommand(Owner('RESUME'), supervisor); action.tick()
    original_wait = module.wait
    def missing_start_reply(future, timeout=15):
        if future is pending and not future.done():
            raise RuntimeError('START response timed out')
        return original_wait(future, timeout)
    monkeypatch.setattr(module, 'wait', missing_start_reply)
    supervisor.stop()
    assert [r.command for r in node.client.requests][-2:] == [1, 8]
    assert supervisor.stopped and not supervisor.running
    assert pending in supervisor._revoked_requests and not pending.cancelled()
    pending.set_result(reply())  # Late reply must never restore running state.
    supervisor.poll(pending)
    assert not supervisor.running
    with pytest.raises(RuntimeError, match='Stopped'): supervisor.resume()
    with pytest.raises(RuntimeError, match='late START'): supervisor.submit(1)


def test_timed_out_pause_requires_restart_after_confirmed_fault_hold(monkeypatch):
    from orchestrator.bt import inference_supervisor as module
    node, supervisor = loaded()
    pending = Future(); node.client.pending[1] = pending
    SupervisedCommand(Owner('RESUME'), supervisor).tick()
    original_wait = module.wait
    def missing_start_reply(future, timeout=15):
        if future is pending and not future.done(): raise RuntimeError('START timed out')
        return original_wait(future, timeout)
    monkeypatch.setattr(module, 'wait', missing_start_reply)
    with pytest.raises(RuntimeError, match='START timed out'): supervisor.pause()
    assert node.client.requests[-1].command == 8 and supervisor.stopped
    assert not supervisor.restore_running


def test_unconfirmed_fault_stop_retains_request_and_retries_ack(monkeypatch):
    from orchestrator.bt import inference_supervisor as module
    node, supervisor = loaded()
    start, fault = Future(), Future()
    node.client.pending.update({1: start, 8: fault})
    SupervisedCommand(Owner('RESUME'), supervisor).tick()
    original_wait = module.wait
    def missing_reply(future, timeout=15):
        if not future.done(): raise RuntimeError('response timed out')
        return original_wait(future, timeout)
    monkeypatch.setattr(module, 'wait', missing_reply)
    with pytest.raises(RuntimeError, match='timed out'): supervisor.stop()
    assert supervisor._fault_future is fault
    assert not start.cancelled() and not fault.cancelled()
    fault.set_result(reply()); supervisor.stop()
    assert supervisor._fault_held and not supervisor.running
    assert [r.command for r in node.client.requests].count(8) == 1


def test_failed_start_and_failed_stop_replies_cannot_hide_running_output():
    node, supervisor = loaded()
    node.client.pending[1] = done(reply('partial start failed', False))
    future = supervisor.submit(1)
    with pytest.raises(RuntimeError, match='partial start failed'): supervisor.poll(future)
    assert supervisor.motion_uncertain and supervisor.pending is None
    node.client.pending[4] = done(reply('hold failed', False))
    supervisor.stop()
    assert node.client.requests[-1].command == 8
    assert supervisor._fault_held and not supervisor.motion_uncertain


def test_repeated_load_resume_stop_loop_never_starts_during_load():
    node = Node()
    supervisor = InferenceSupervisor(node)
    load = SupervisedCommand(Owner('LOAD'), supervisor)
    resume = SupervisedCommand(Owner('RESUME'), supervisor)
    stop = SupervisedCommand(Owner('STOP'), supervisor)
    for iteration in range(2):
        assert run(load) == NodeStatus.SUCCESS
        assert supervisor.loaded and not supervisor.running
        assert [r.command for r in node.client.requests].count(1) == iteration
        assert run(resume) == NodeStatus.SUCCESS
        assert supervisor.running
        assert run(stop) == NodeStatus.SUCCESS
        assert not supervisor.running
        for action in (load, resume, stop):
            action.reset()
    assert [r.command for r in node.client.requests].count(0) == 2
    assert [r.command for r in node.client.requests].count(1) == 2
    assert run(SupervisedCommand(Owner('CLEAR'), supervisor)) == NodeStatus.SUCCESS
    assert [r.command for r in node.client.requests][-2:] == [4, 5]
    assert not supervisor.loaded
