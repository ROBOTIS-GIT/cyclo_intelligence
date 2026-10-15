"""Progress and actuator hold regressions; no ROS node or robot is created."""
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading
import time

import pytest
from orchestrator.bt.actions.home_meal import HomeMeal
from orchestrator.bt.actions.motion_pause import JointPause, RotatePause
from orchestrator.bt.actions.wait import Wait
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.controls.sequence import Sequence
from orchestrator.bt.execution_pause import ExecutionPause
from orchestrator.bt.pause_control import pause_tree, resume_tree


class Node:
    def __init__(self): self.future = Future(); self.calls = 0
    def create_client(self, *args): return self
    def service_is_ready(self): return True
    def call_async(self, req): self.calls += 1; return self.future
    def get_logger(self): return Mock()


def test_sequence_pause_preserves_cursor_and_atomic_prop_action():
    node = Node(); tree = Sequence(node)
    prop = HomeMeal(node, 'put_sink'); wait = Wait(node, duration=10)
    tree.add_child(prop); tree.add_child(wait)
    assert tree.tick() == NodeStatus.RUNNING
    node.future.set_result(SimpleNamespace(success=True, message='applied'))
    assert tree.tick() == NodeStatus.RUNNING
    assert tree.current_child_index == 1 and node.calls == 1
    pause_tree(tree)
    assert tree.current_child_index == 1
    resume_tree(tree)
    assert tree.tick() == NodeStatus.RUNNING and node.calls == 1
    wait._start_time -= 11
    assert tree.tick() == NodeStatus.SUCCESS and node.calls == 1


def test_wait_does_not_consume_paused_duration():
    wait = Wait(Node(), duration=5)
    with patch('orchestrator.bt.actions.wait.time.monotonic', return_value=100): wait.tick()
    with patch('orchestrator.bt.pause_control.time.monotonic', return_value=102): wait.pause()
    with patch('orchestrator.bt.pause_control.time.monotonic', return_value=122): wait.resume()
    with patch('orchestrator.bt.actions.wait.time.monotonic', return_value=124):
        assert wait.tick() == NodeStatus.RUNNING
    with patch('orchestrator.bt.actions.wait.time.monotonic', return_value=126):
        assert wait.tick() == NodeStatus.SUCCESS


def test_pending_home_transition_settles_without_replaying():
    node = Node(); prop = HomeMeal(node, 'pickup_table'); prop.tick()
    complete = threading.Timer(.02, lambda: node.future.set_result(SimpleNamespace(success=True)))
    complete.start(); prop.pause(); complete.join()
    prop.resume()
    assert prop.tick() == NodeStatus.SUCCESS and node.calls == 1


class Engine(ExecutionPause):
    def __init__(self, root):
        self.root = root; self.tree_execution_mode = 'running'
        self._execution_lock = threading.RLock(); self._publish_status = Mock()


def test_resume_cannot_restart_stopped_or_failed_pause_tree():
    engine = Engine(Wait(Node()))
    engine.tree_execution_mode = 'stopped'
    response = engine._set_paused_callback(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success and engine.tree_execution_mode == 'stopped'
    engine.tree_execution_mode = 'pause_failed'
    response = engine._set_paused_callback(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success and engine.tree_execution_mode == 'pause_failed'


def test_pause_failure_freezes_ticks_and_never_reports_paused():
    engine = Engine(SimpleNamespace(name='unsafe', pause=Mock(side_effect=RuntimeError('hold failed'))))
    response = engine._set_paused_callback(SimpleNamespace(data=True), SimpleNamespace())
    assert not response.success and engine.tree_execution_mode == 'pause_failed'


def test_completion_racing_pause_does_not_resume_tree():
    engine = Engine(Mock()); engine.tree_execution_mode = 'completed'
    for paused in (True, False):
        response = engine._set_paused_callback(SimpleNamespace(data=paused), SimpleNamespace())
        assert response.success and engine.tree_execution_mode == 'completed'
    engine.root.reset.assert_not_called()


class Joint(JointPause):
    def __init__(self):
        self._thread = object(); self._result = None
        self._motion_lock = threading.RLock(); self._paused = threading.Event(); self._stop_event = threading.Event()
        self._channels = [{'joint_names': ['arm'], 'positions': [1.5]}]
        self._joint_received = {'arm': time.monotonic()}; self._joint_velocities = {'arm': 0.0}
        self._name_to_position = lambda: {'arm': .3}
        self.published = []
    def _publish_channel(self, channel, positions=None, duration=None):
        self.published.append((positions, duration))


def test_joint_pause_holds_feedback_and_resume_keeps_original_target():
    joint = Joint(); stop = threading.Event()
    def feedback():
        while not stop.is_set():
            joint._joint_received['arm'] = time.monotonic(); time.sleep(.005)
    thread = threading.Thread(target=feedback); thread.start()
    try:
        joint.pause()
        assert joint._paused.is_set()
        assert joint.published == [({'arm': .3}, .1)]
        joint.resume()
        assert not joint._paused.is_set()
        assert joint.published[-1] == (None, None)
        assert joint._channels[0]['positions'] == [1.5]
    finally:
        stop.set(); thread.join()


def test_joint_pause_rejects_missing_or_stale_feedback():
    joint = Joint(); joint._joint_received['arm'] = 0
    with pytest.raises(RuntimeError, match='fresh feedback'): joint.pause()
    assert joint._paused.is_set() and joint.published == []


def test_joint_stop_latch_prevents_republishing_saved_target():
    joint = Joint(); joint._paused.set(); joint._stop_event.set()
    with pytest.raises(RuntimeError, match='stopped'): joint.resume()
    assert joint.published == []


def _bt_engine():
    from types import MethodType
    from orchestrator.bt.bt_node import BehaviorTreeNode
    engine = SimpleNamespace(
        root=Mock(), inference_supervisor=Mock(), tree_execution_mode='running',
        _execution_lock=threading.RLock(), _publish_status=Mock(),
        _send_cleanup_command=Mock(), get_logger=lambda: Mock(), tree_loader=Mock(),
    )
    engine._set_running_callback = MethodType(BehaviorTreeNode._set_running_callback, engine)
    return BehaviorTreeNode, engine


def test_failed_bt_stop_preserves_owner_until_confirmed_retry():
    _, engine = _bt_engine()
    engine.inference_supervisor.stop.side_effect = [RuntimeError('hold pending'), None]
    first = engine._set_running_callback(SimpleNamespace(data=False), SimpleNamespace())
    assert not first.success and engine.tree_execution_mode == 'pause_failed'
    engine.root.reset.assert_not_called()
    second = engine._set_running_callback(SimpleNamespace(data=False), SimpleNamespace())
    assert second.success and engine.tree_execution_mode == 'stopped'
    assert engine.inference_supervisor.stop.call_count == 2
    engine.root.reset.assert_called_once()


def test_new_tree_cannot_replace_unconfirmed_stop_owner():
    cls, engine = _bt_engine()
    previous_root, owner = engine.root, engine.inference_supervisor
    owner.stop.side_effect = RuntimeError('hold pending')
    response = cls._load_and_run_callback(engine, SimpleNamespace(tree_xml='<root/>'), SimpleNamespace())
    assert not response.success and engine.tree_execution_mode == 'pause_failed'
    assert engine.root is previous_root and engine.inference_supervisor is owner
    engine.tree_loader.load_tree_from_string.assert_not_called()


def test_task_failure_does_not_release_unconfirmed_policy_hold():
    cls, engine = _bt_engine()
    engine.inference_supervisor.stop.side_effect = RuntimeError('hold pending')
    cls._handle_tree_completion(engine, NodeStatus.FAILURE)
    assert engine.tree_execution_mode == 'pause_failed'
    engine.root.reset.assert_not_called()
