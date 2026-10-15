"""Mission-local cancellation: drain execution, return, keep patrol and Qwen alive."""
import threading
from unittest.mock import Mock
import pytest
from supervisor_api.test_autonomy_continuous import harness, verdict
from supervisor_api.test_autonomy import Runtime, eventually
from supervisor_api.autonomy_runtime import OperationCancelled


@pytest.mark.parametrize('operation', ['admission', 'navigation', 'tree'])
def test_cancel_returns_to_trigger_spot_and_keeps_patrol_running(harness, operation):
    h = harness
    h.plan.mode = 'patrol'; h.plan.start_confirmations = 1
    h.points.append({**h.points[0], 'spot_id': 'c', 'label': 'C', 'pose': {'x': 3, 'y': 0}})
    old, replacement = h.runtime, Runtime()
    h.controller.runtime_factory = Mock(side_effect=[old, replacement])
    running, drained = threading.Event(), threading.Event()
    def cancel():
        old.calls.append(('cancel',)); drained.set()
    old.cancel = cancel
    def block():
        running.set()
        while not drained.wait(.002):
            if h.controller.cancelled.is_set(): raise OperationCancelled()
        raise OperationCancelled()
    navigate, tree = old.navigate, old.tree
    def nav(pose, stopped):
        if operation == 'navigation' and pose['x'] == 2:
            old.calls.append(('nav', 2)); block()
        navigate(pose, stopped)
    def bt(xml, stopped):
        if operation == 'tree' and xml == 'task':
            old.calls.append(('bt', xml)); block()
        tree(xml, stopped)
    old.navigate, old.tree = nav, bt
    requests = []
    def infer(req):
        requests.append(req)
        if req.reasoning_task == 'patrol_supervision': return verdict(h, 'continue', None)
        if req.reasoning_task == 'mission_supervision':
            return verdict(h, 'cancel' if operation == 'admission' or running.is_set() else 'continue')
        return verdict(h, 'start' if req.observation.get('at_spot_id') == 'b' and not req.recent_missions else 'wait')
    h.service.decide.side_effect = infer
    h.controller.start(h.request)
    eventually(lambda: ('nav', 3) in replacement.calls)
    assert replacement.calls.index(('nav', 1)) < replacement.calls.index(('nav', 3))
    assert ('close',) in old.calls and drained.is_set()
    if operation != 'tree': assert ('bt', 'task') not in old.calls
    assert ('bt', 'task') not in replacement.calls
    assert h.controller.state['recent_missions'] == [{'mission_id': 'task', 'status': 'cancelled'}]
    assert h.controller.state['active'] and not h.controller.cancelled.is_set()
    assert any(e['title'] == 'Mission cancelled' for e in h.controller.state['events'])
    h.service.stop.assert_not_called()
    returning = [r for r in requests if r.observation.get('phase') == 'returning']
    assert returning and all(r.reasoning_task == 'patrol_supervision' and r.active_mission is None for r in returning)


def test_cancel_does_not_release_manual_hold(harness):
    from supervisor_api.autonomy_mission_cancel import cancel_child_mission
    h = harness
    replacement = Runtime()
    h.controller.runtime = h.runtime
    h.controller.runtime_factory = lambda: replacement
    h.controller.state.update(active=True, mode='mission', phase='executing', map_name='Home',
        mission=h.selected, recent_missions=[], paused=True,
        pause_causes={'manual': 'Hold', 'ai': 'Obstacle'}, resume_checkpoint={'return_spot_id': 'a'})
    cancel_child_mission(h.controller, h.selected, 'Task no longer needed')
    assert h.controller.state['pause_causes'] == {'manual': 'Hold'}
    assert h.controller.state['paused'] and ('pause',) in replacement.calls
    assert ('resume',) not in replacement.calls


def test_failed_cancellation_never_starts_return(harness):
    h = harness
    h.plan.start_confirmations = 1
    def infer(req):
        if req.reasoning_task == 'patrol_supervision': return verdict(h, 'continue', None)
        if req.reasoning_task == 'mission_supervision': return verdict(h, 'cancel')
        return verdict(h, 'start')
    h.service.decide.side_effect = infer
    def fail(): raise RuntimeError('Actuator halt unconfirmed')
    h.runtime.cancel = fail
    h.controller.start(h.request)
    eventually(lambda: h.controller.state['phase'] == 'error')
    assert h.runtime.calls.count(('nav', 0)) == 1
    assert ('nav', 2) not in h.runtime.calls
    assert 'Actuator halt unconfirmed' in h.controller.state['error']
    h.runtime.cancel = lambda: None  # Permit teardown after checking the real failure.


def test_wrong_mission_or_stale_cancel_is_not_accepted(harness):
    from supervisor_api.autonomy_reasoner import SessionReasoner
    h = harness; h.controller.runtime = h.runtime
    h.controller.state.update(active=True, mode='mission', phase='executing', map_name='Home', mission=h.selected)
    h.controller.transition()
    reasoner = SessionReasoner(h.controller, h.request, h.plan, 'session')
    reasoner.infer = lambda _: {'context_version': h.controller.state['context_version'] - 1,
        'reasoning_task': 'mission_supervision', **verdict(h, 'cancel')}
    reasoner.supervise()
    assert not h.runtime.calls
    reasoner.infer = lambda _: {'context_version': h.controller.state['context_version'],
        'reasoning_task': 'mission_supervision', **verdict(h, 'cancel', 'different')}
    reasoner.supervise()
    assert ('pause',) in h.runtime.calls and ('cancel',) not in h.runtime.calls
