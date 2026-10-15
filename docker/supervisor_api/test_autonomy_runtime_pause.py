"""Real worker pause protocol, tested with fake ROS futures and no robot."""
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import threading
import time

import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from supervisor_api.autonomy_runtime import AutonomyRosRuntime, OperationCancelled
from supervisor_api.autonomy_runtime_control import PauseControl, validate_tree


def done(value):
    f = Future(); f.set_result(value); return f


def eventually(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > deadline: raise AssertionError('Timed out')
        time.sleep(.005)


class Goal:
    accepted = True
    def __init__(self):
        self.result = Future()
        self.cancels = 0
    def get_result_async(self): return self.result
    def cancel_goal_async(self):
        self.cancels += 1
        if not self.result.done(): self.finish(5)
        return done(SimpleNamespace(return_code=0))
    def finish(self, status=4):
        self.result.set_result(SimpleNamespace(status=status, result=SimpleNamespace(error_code=0)))


class Nav:
    def __init__(self): self.sent = []; self.goals = []
    def wait_for_server(self, **kw): return True
    def send_goal_async(self, msg):
        goal = Goal(); self.sent.append(msg); self.goals.append(goal); return done(goal)


def goal_message():
    return SimpleNamespace(pose=SimpleNamespace(header=SimpleNamespace(frame_id=''), pose=SimpleNamespace(
        position=SimpleNamespace(x=0, y=0), orientation=SimpleNamespace(z=0, w=0))))


@pytest.fixture
def ros():
    with patch.dict(sys.modules, {
        'nav2_msgs.action': SimpleNamespace(NavigateToPose=SimpleNamespace(Goal=goal_message)),
        'std_srvs.srv': SimpleNamespace(SetBool=SimpleNamespace(Request=lambda: SimpleNamespace(data=None))),
        'interfaces.srv': SimpleNamespace(LoadAndRunTree=SimpleNamespace(Request=lambda: SimpleNamespace(tree_xml=''))),
    }): yield


def worker(call):
    errors = []
    def run():
        try: call()
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=run, daemon=True); thread.start()
    return thread, errors


def test_navigation_pause_preserves_goal_and_blocks_completion(ros):
    runtime = AutonomyRosRuntime(Mock(), Mock()); runtime.nav = Nav(); runtime.arrival = Mock()
    thread, errors = worker(lambda: runtime.navigate({'x': 2, 'y': 3, 'yaw': 1}, threading.Event()))
    eventually(lambda: len(runtime.nav.goals) == 1)
    runtime.pause()
    assert runtime.nav.goals[0].cancels == 1
    assert thread.is_alive() and len(runtime.nav.goals) == 1
    runtime.resume()
    eventually(lambda: len(runtime.nav.goals) == 2)
    assert runtime.nav.sent[0] is runtime.nav.sent[1]
    runtime.nav.goals[1].finish()
    thread.join(2)
    assert not thread.is_alive() and errors == []


def test_stop_wakes_paused_navigation_and_prevents_late_resume(ros):
    runtime = AutonomyRosRuntime(Mock(), Mock()); runtime.nav = Nav(); runtime.arrival = Mock()
    thread, errors = worker(lambda: runtime.navigate({'x': 2, 'y': 3}, threading.Event()))
    eventually(lambda: len(runtime.nav.goals) == 1)
    runtime.pause(); runtime.cancel()
    with pytest.raises(OperationCancelled): runtime.resume()
    thread.join(2)
    assert not thread.is_alive() and isinstance(errors[0], OperationCancelled)
    assert len(runtime.nav.goals) == 1


def test_idle_pause_blocks_new_goal_until_resume(ros):
    runtime = AutonomyRosRuntime(Mock(), Mock()); runtime.nav = Nav(); runtime.arrival = Mock()
    runtime.pause()
    thread, errors = worker(lambda: runtime.navigate({'x': 2, 'y': 3}, threading.Event()))
    eventually(lambda: runtime.control.active)
    assert runtime.nav.sent == []
    runtime.resume(); eventually(lambda: len(runtime.nav.goals) == 1)
    runtime.nav.goals[0].finish(); thread.join(2)
    assert not thread.is_alive() and not errors


def test_stop_drains_late_goal_acceptance(ros):
    runtime = AutonomyRosRuntime(Mock(), Mock()); runtime.nav = Nav(); runtime.arrival = Mock()
    acceptance = Future(); goal = Goal()
    runtime.nav.send_goal_async = lambda msg: acceptance
    running, errors = worker(lambda: runtime.navigate({'x': 1, 'y': 1}, threading.Event()))
    eventually(lambda: runtime.goal_future is acceptance)
    stopping, stop_errors = worker(runtime.cancel)
    eventually(runtime.control.stopped.is_set)
    acceptance.set_result(goal)
    running.join(2); stopping.join(2)
    assert goal.cancels == 1 and not stop_errors
    assert isinstance(errors[0], OperationCancelled)


def test_bt_pause_resume_keeps_single_xml_load(ros):
    runtime = AutonomyRosRuntime(Mock(), Mock())
    runtime.load = Mock(); runtime.pause_bt = Mock()
    def load(req):
        runtime._status(SimpleNamespace(data='running'))
        return done(SimpleNamespace(success=True, message=''))
    runtime.load.call_async.side_effect = load
    runtime.pause_bt.call_async.side_effect = lambda req: done(SimpleNamespace(success=True, message=''))
    xml = '<root main_tree_to_execute="Main"><BehaviorTree ID="Main"><Wait duration="3"/></BehaviorTree></root>'
    thread, errors = worker(lambda: runtime.tree(xml, threading.Event()))
    eventually(lambda: runtime.running_sequence > 0)
    runtime.pause()
    assert runtime.load.call_async.call_count == 1
    runtime.resume()
    runtime._status(SimpleNamespace(data='completed')); thread.join(2)
    assert not thread.is_alive() and errors == []
    assert [c.args[0].data for c in runtime.pause_bt.call_async.call_args_list] == [True, False]
    assert runtime.load.call_async.call_count == 1


def test_bt_pause_failure_surfaces_without_automatic_resume(ros):
    runtime = AutonomyRosRuntime(Mock(), Mock())
    runtime.load = Mock(); runtime.pause_bt = Mock()
    def load(req):
        runtime._status(SimpleNamespace(data='running'))
        return done(SimpleNamespace(success=True, message=''))
    runtime.load.call_async.side_effect = load
    runtime.pause_bt.call_async.return_value = done(SimpleNamespace(success=False, message='hold not confirmed'))
    xml = '<root main_tree_to_execute="Main"><BehaviorTree ID="Main"><Wait/></BehaviorTree></root>'
    thread, errors = worker(lambda: runtime.tree(xml, threading.Event()))
    eventually(lambda: runtime.running_sequence > 0)
    with pytest.raises(RuntimeError, match='hold not confirmed'): runtime.pause()
    thread.join(2)
    assert errors and runtime.pause_bt.call_async.call_count == 1


def test_unsupported_tree_rejected_before_actuation():
    xml = '<root main_tree_to_execute="Main"><BehaviorTree ID="Main"><SendCommand command="RESUME"/></BehaviorTree></root>'
    with pytest.raises(ValueError, match='SendCommand'): validate_tree(xml)
    runtime = AutonomyRosRuntime(Mock(), Mock())
    with pytest.raises(ValueError): runtime.tree(xml, threading.Event())
    runtime.start_bt.assert_not_called()


def test_pause_control_stop_overrides_both_commands():
    control = PauseControl(); control.stop()
    for pause in (True, False):
        with pytest.raises(OperationCancelled): control.request(pause)


def test_patrol_redirect_drains_goal_without_stopping_session(ros):
    from supervisor_api.autonomy_direction import PatrolRedirect
    runtime = AutonomyRosRuntime(Mock(), Mock()); runtime.nav = Nav(); runtime.arrival = Mock()
    cancelled = threading.Event()
    thread, errors = worker(lambda: runtime.navigate({'x': 2, 'y': 3}, cancelled))
    eventually(lambda: len(runtime.nav.goals) == 1)
    runtime.patrol_redirect.set()
    thread.join(2)
    assert not thread.is_alive() and isinstance(errors[0], PatrolRedirect)
    assert runtime.nav.goals[0].cancels == 1
    assert runtime.goal is None and not runtime.control.stopped.is_set()
    runtime.patrol_redirect.clear()
    thread, errors = worker(lambda: runtime.navigate({'x': 0, 'y': 0}, cancelled))
    eventually(lambda: len(runtime.nav.goals) == 2)
    runtime.nav.goals[1].finish(); thread.join(2)
    assert not thread.is_alive() and not errors
