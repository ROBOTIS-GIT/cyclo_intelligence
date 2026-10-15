"""Continuous inference, frame admission and task resume regression scenarios."""
import json
import threading
import time
from unittest.mock import patch
import pytest
from supervisor_api import test_autonomy as fixtures
from supervisor_api.test_autonomy import eventually
from supervisor_api.autonomy_runtime import OperationCancelled
from supervisor_api.autonomy_context import ContextHistory
from supervisor_api.autonomy_config import AutonomyRepository, ObservationPlan, ObservationPoint
from supervisor_api import autonomy_config as config


@pytest.fixture
def harness():
    case = fixtures.ControllerTests('runTest')
    case.setUp()
    yield case
    case.doCleanups()


def verdict(case, action, task='task'):
    return {'action_allowed': True, 'frame_stamp': str(case.service.decide.call_count),
            'selected_mission': case.selected if action == 'start' else None,
            'decision': {'decision': action, 'mission_id': task if action not in ('wait', 'unknown') else None,
                         'reason': 'Scene evidence'}}


def test_transit_reasoning_never_starts_task_without_arrival_confirmation(harness):
    h = harness
    moving, arrive = threading.Event(), threading.Event()
    original = h.runtime.navigate
    def navigate(pose, cancelled):
        if pose['x'] == 0:
            moving.set()
            while not arrive.wait(.002):
                if cancelled.is_set(): raise OperationCancelled()
        original(pose, cancelled)
    h.runtime.navigate = navigate
    requests = []
    def infer(req):
        requests.append(req)
        return verdict(h, 'continue', task=None) if req.reasoning_task == 'patrol_supervision' else verdict(h, 'wait')
    h.service.decide.side_effect = infer
    h.controller.start(h.request)
    assert moving.wait(1)
    eventually(lambda: len(requests) >= 5)
    h.repo.mission.assert_not_called()
    arrive.set()
    eventually(lambda: any(r.observation.get('phase') == 'observing' for r in requests))
    transit = [r for r in requests if r.observation.get('phase') in ('moving', 'settling')]
    assert transit and all(r.reasoning_task == 'patrol_supervision' and r.allowed_mission_ids == [] and r.active_mission is None for r in transit)
    assert all(r.reasoning_task == 'mission_selection' for r in requests if r.observation.get('phase') == 'observing')
    h.repo.mission.assert_not_called()


def test_patrol_returns_to_trigger_spot_then_continues_next_spot(harness):
    h = harness
    h.plan.mode = 'patrol'
    h.plan.start_confirmations = 1
    h.points.append({**h.points[0], 'spot_id': 'c', 'label': 'C', 'pose': {'x': 3, 'y': 0}})
    contexts = []
    def infer(req):
        contexts.append(req)
        if req.reasoning_task == 'patrol_supervision': return verdict(h, 'continue', task=None)
        if req.reasoning_task == 'mission_supervision': return verdict(h, 'continue')
        at_b = req.observation.get('at_spot_id') == 'b'
        completed = ('bt', 'task') in h.runtime.calls
        return verdict(h, 'start' if at_b and not completed else 'wait')
    h.service.decide.side_effect = infer
    h.controller.start(h.request)
    eventually(lambda: ('nav', 3) in h.runtime.calls)
    assert [call for call in h.runtime.calls if call[0] == 'nav'][:5] == [
        ('nav', 0), ('nav', 1), ('nav', 2), ('nav', 1), ('nav', 3)]
    returning = [r for r in contexts if r.observation.get('phase') == 'returning']
    assert returning and all(r.reasoning_task == 'patrol_supervision' and r.active_mission is None for r in returning)
    h.service.stop.assert_not_called()


def test_invalid_supervision_holds_before_task_motion_and_keeps_reasoning(harness):
    h = harness
    h.plan.start_confirmations = 1
    def infer(req):
        if req.reasoning_task == 'patrol_supervision': return verdict(h, 'continue', task=None)
        if req.reasoning_task == 'mission_supervision':
            return {'error': 'Unsupported decision', 'action_allowed': False}
        return verdict(h, 'start')
    h.service.decide.side_effect = infer
    h.controller.start(h.request)
    eventually(lambda: h.controller.state['paused'])
    count = h.service.decide.call_count
    eventually(lambda: h.service.decide.call_count > count + 3)
    assert ('nav', 2) not in h.runtime.calls
    assert h.runtime.calls.count(('pause',)) == 1
    assert h.controller.state['pause_causes'].keys() == {'ai'}
    assert h.controller.state['active']
    assert not any(e['title'] == 'Mission started' for e in h.controller.state['events'])


def test_pause_resume_supervision_preserves_single_task_and_manual_hold(harness):
    h = harness
    h.plan.start_confirmations = 1
    started, finish = threading.Event(), threading.Event()
    original = h.runtime.navigate
    def navigate(pose, cancelled):
        if pose['x'] == 2:
            started.set()
            while not finish.wait(.002):
                if cancelled.is_set(): raise OperationCancelled()
        original(pose, cancelled)
    h.runtime.navigate = navigate
    issue = threading.Event()
    issue.set()
    def infer(req):
        if req.reasoning_task == 'patrol_supervision': return verdict(h, 'continue', task=None)
        if req.reasoning_task == 'mission_supervision':
            return verdict(h, 'pause' if started.is_set() and issue.is_set() else 'continue')
        return verdict(h, 'start' if not started.is_set() else 'wait')
    h.service.decide.side_effect = infer
    h.controller.start(h.request)
    assert started.wait(2)
    eventually(lambda: 'ai' in h.controller.state['pause_causes'])
    h.controller.pause('Operator pause', source='manual')
    issue.clear()
    eventually(lambda: 'ai' not in h.controller.state['pause_causes'])
    assert h.controller.state['paused']
    assert ('resume',) not in h.runtime.calls
    h.controller.resume(source='manual')
    assert h.runtime.calls.count(('resume',)) == 1
    finish.set()
    eventually(lambda: ('bt', 'task') in h.runtime.calls)
    assert h.runtime.calls.count(('nav', 2)) == 1
    assert h.runtime.calls.count(('bt', 'task')) == 1


@pytest.mark.parametrize("decision", ["continue", "cancel"])
def test_stop_during_supervision_discards_late_decision(harness, decision):
    h = harness
    h.plan.start_confirmations = 1
    pending = threading.Event()
    def infer(req):
        if req.reasoning_task == 'patrol_supervision': return verdict(h, 'continue', task=None)
        if req.reasoning_task == 'mission_supervision':
            pending.set()
            h.release.wait(2)
            return verdict(h, decision)
        return verdict(h, 'start')
    h.service.decide.side_effect = infer
    h.controller.start(h.request)
    assert pending.wait(2)
    h.controller.stop()
    h.controller.stop_thread.join(3)
    assert not h.controller.owned
    assert ('nav', 2) not in h.runtime.calls
    assert ('resume',) not in h.runtime.calls


def test_capture_context_uses_receive_interval_not_current_state():
    history = ContextHistory()
    with patch('supervisor_api.autonomy_context.time.time', return_value=10):
        history.record({'phase': 'moving'})
    with patch('supervisor_api.autonomy_context.time.time', return_value=20):
        history.record({'phase': 'observing'})
    with patch('supervisor_api.autonomy_context.time.time', return_value=30):
        previous = history.at(15)
        assert previous['phase'] == 'moving'
        assert previous['context_valid_from'] == 10
        assert previous['context_valid_until'] == 20
        assert history.at(20)['phase'] == 'observing'
        assert history.at(5) == {'runtime_source': 'unverified'}
        assert history.at(31) == {'runtime_source': 'unverified'}


def test_v1_empty_candidates_expand_but_v2_empty_stays_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config.spots, 'SPOTS_ROOT', tmp_path)
    monkeypatch.setattr(config.missions, 'collection_catalog', lambda _: [
        {'mission_id': 'task'}, {'mission_id': 'legacy-observation'}])
    repo = AutonomyRepository()
    file = repo.path('Home')
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps({'revision': 5, 'excluded_mission_ids': ['legacy-observation'],
                               'observations': [{'spot_id': 'a', 'mission_ids': []}]}))
    upgraded = repo.read('Home')
    assert upgraded.schema_version == 2
    assert upgraded.revision == 5
    assert upgraded.observations[0].mission_ids == ['task']
    file.write_text(ObservationPlan(observations=[ObservationPoint(spot_id='a')]).model_dump_json())
    assert repo.read('Home').observations[0].mission_ids == []


def test_unconfirmed_manual_pause_faults_and_requires_stop(harness):
    h = harness
    h.controller.start(h.request)
    eventually(lambda: h.controller.state['phase'] == 'observing')
    def fail(): raise TimeoutError('Hold was not acknowledged')
    h.runtime.pause = fail
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match='Hold was not acknowledged'):
        h.controller.pause()
    assert h.controller.state['phase'] == 'error'
    assert not h.controller.state['active']
    assert h.controller.owned
    assert ('cancel',) in h.runtime.calls
    assert h.controller.capture_context() == {'runtime_source': 'unverified'}


def test_capture_before_stop_keeps_historical_supervision_context(harness):
    h = harness
    h.controller.start(h.request)
    eventually(lambda: h.controller.state['phase'] == 'observing')
    h.controller.transition(mode='mission', mission=h.selected, runner={'phase': 'navigation'})
    stamp = time.time()
    expected = h.controller.capture_context(stamp)
    h.controller.stop()
    h.controller.stop_thread.join(3)
    historical = h.controller.capture_context(stamp)
    assert historical['reasoning_task'] == 'mission_supervision'
    assert historical['context_version'] == expected['context_version']
    assert historical['active_mission']['mission_id'] == 'task'
    assert h.controller.capture_context() is None


def test_resume_confirmation_does_not_cross_context_versions(harness):
    h = harness
    from supervisor_api.autonomy_reasoner import SessionReasoner
    h.controller.runtime = h.runtime
    h.controller.owned = True
    h.controller.state.update(active=True, mode='mission', map_name='Home', mission=h.selected,
                              phase='executing', paused=True, pause_causes={'ai': 'Hold'})
    h.controller.transition()
    reasoner = SessionReasoner(h.controller, h.request, h.plan, 'session')
    def response(stamp):
        return {'context_version': h.controller.state['context_version'], 'frame_stamp': stamp,
                'action_allowed': True, 'decision': {'decision': 'continue', 'mission_id': 'task'}}
    reasoner.infer = lambda _: response('1')
    reasoner.supervise()
    h.controller.instruction('Proceed only when clear')
    reasoner.infer = lambda _: response('2')
    reasoner.supervise()
    assert h.controller.state['paused']
    reasoner.infer = lambda _: response('3')
    reasoner.supervise()
    assert not h.controller.state['paused']
