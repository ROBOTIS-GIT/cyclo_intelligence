"""Execution snapshots and acknowledged controls without ROS, GPU or live services."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import sys
import threading

import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from supervisor_api.autonomy import AutonomyController
from supervisor_api.autonomy_reasoner import SessionReasoner
from supervisor_api.execution_state import execution_snapshot, transition_state


def controller():
    service = Mock()
    service.status.return_value = {'phase': 'ready', 'session_id': 'reasoner'}
    c = AutonomyController(service, Mock(), Mock(), Mock())
    c.runtime = Mock()
    c._commit(owned=True, record_context=True, phase='observing', active=True,
              session_id='session', mode='observation', map_name='Home', plan_revision=3,
              control_status='running', observation_context={'phase': 'observing', 'at_spot_id': 'a'})
    return c


def background(action):
    errors, values = [], []
    def run():
        try: values.append(action())
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors, values


def finish(thread, errors):
    thread.join(3)
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize('phase,runner,mode,activity', [
    ('starting', {}, 'observation', 'idle'),
    ('positioning', {'phase': 'patrol-navigation'}, 'observation', 'patrol_navigation'),
    ('settling', {'phase': 'observation-posture'}, 'observation', 'observation_posture'),
    ('observing', {}, 'observation', 'observing'),
    ('executing', {'phase': 'navigation'}, 'mission', 'mission_navigation'),
    ('executing', {'phase': 'waypoint-task'}, 'mission', 'mission_action'),
    ('returning', {'phase': 'return-navigation'}, 'observation', 'return_navigation'),
    ('returning', {'phase': 'return-posture'}, 'observation', 'return_posture'),
])
def test_snapshot_normalizes_existing_activities(phase, runner, mode, activity):
    snapshot = execution_snapshot({'phase': phase, 'runner': runner, 'mode': mode,
        'active': True, 'mission': {'mission_id': 'task'}, 'revision': 7}, True, 'boot')
    assert snapshot['activity'] == activity
    assert snapshot['schema_version'] == 1
    assert (snapshot['source'], snapshot['authority'], snapshot['epoch']) == ('operate', 'server', 'boot')
    assert snapshot['step_index'] is None


def test_pure_transition_copies_inputs_and_validates_control():
    previous = {'revision': 4, 'phase': 'observing'}
    changes = {'pause_causes': {'manual': 'Hold'}, 'control_status': 'paused'}
    result = transition_state(previous, changes, replace=True)
    changes['pause_causes'].clear()
    assert result == {'revision': 5, 'pause_causes': {'manual': 'Hold'}, 'control_status': 'paused'}
    assert previous == {'revision': 4, 'phase': 'observing'}
    with pytest.raises(ValueError): transition_state(previous, {'revision': 100})
    with pytest.raises(ValueError): transition_state(previous, {'control_status': 'unknown'})


def test_pause_and_resume_are_published_before_acknowledgement():
    c = controller()
    entered, release = threading.Event(), threading.Event()
    def acknowledge():
        entered.set()
        assert release.wait(3)
    c.runtime.pause.side_effect = acknowledge
    initial = c.status()
    thread, errors, _ = background(c.pause)
    try:
        assert entered.wait(3)
        pending = c.status()
        assert pending['execution']['control'] == pending['control_status'] == 'pausing'
        assert pending['paused'] is False
        assert pending['execution']['revision'] > initial['execution']['revision']
        assert pending['context_version'] == initial['context_version']
    finally:
        release.set(); finish(thread, errors)
    paused = c.status()
    assert paused['execution']['control'] == 'paused' and paused['paused'] is True
    assert paused['context_version'] > initial['context_version']
    entered.clear(); release.clear()
    c.runtime.resume.side_effect = acknowledge
    thread, errors, _ = background(c.resume)
    try:
        assert entered.wait(3)
        pending = c.status()
        assert pending['execution']['control'] == 'resuming' and pending['paused'] is True
        assert pending['execution']['pause_causes'] == {'manual': 'Operator pause'}
        assert pending['context_version'] == paused['context_version']
    finally:
        release.set(); finish(thread, errors)
    assert c.status()['execution']['control'] == 'running'
    assert c.status()['paused'] is False


def test_ai_resume_cannot_release_manual_pause():
    c = controller()
    c.pause(source='manual'); c.pause('Obstacle', source='ai')
    c.runtime.pause.assert_called_once()
    c.resume(source='ai')
    c.runtime.resume.assert_not_called()
    assert c.status()['execution']['control'] == 'paused'
    assert c.status()['execution']['pause_causes'] == {'manual': 'Operator pause'}
    c.resume(source='manual')
    c.runtime.resume.assert_called_once()
    assert c.status()['execution']['pause_causes'] == {}


def test_logs_advance_revision_without_invalidating_inflight_reasoning_or_capture():
    c = controller()
    entered, release = threading.Event(), threading.Event()
    def decide(_):
        entered.set()
        assert release.wait(3)
        return {'frame_stamp': 'frame', 'action_allowed': True, 'decision': {'decision': 'wait'}}
    c.reasoning.decide.side_effect = decide
    reasoner = SessionReasoner(c, SimpleNamespace(map_name='Home'), SimpleNamespace(resume_confirmations=4), 'reasoner')
    initial = c.status()
    thread, errors, values = background(reasoner.infer)
    try:
        assert entered.wait(3)
        c.event('Operator viewed status', persist=False)
        assert c.status()['revision'] > initial['revision']
        assert c.capture_context()['context_version'] == initial['context_version']
        assert len(c.history.entries) == 1
    finally:
        release.set(); finish(thread, errors)
    assert values[0]['context_version'] == initial['context_version']
    assert c.status()['result']['frame_stamp'] == 'frame'
    assert c.status()['revision'] > initial['revision'] + 1
    assert c.status()['context_version'] == initial['context_version']


def test_stop_keeps_ownership_until_cleanup_is_acknowledged():
    c = controller()
    entered, release = threading.Event(), threading.Event()
    def cancel():
        entered.set()
        assert release.wait(3)
    runtime = c.runtime
    runtime.cancel.side_effect = cancel
    c.stop()
    try:
        assert entered.wait(3)
        pending = c.status()['execution']
        assert pending['lifecycle'] == 'stopping'
        assert pending['control'] == 'cancelling' and pending['owns_execution']
        assert c.capture_context() == {'runtime_source': 'unverified'}
    finally:
        release.set(); c.stop_thread.join(3)
    assert not c.stop_thread.is_alive()
    stopped = c.status()['execution']
    assert stopped['lifecycle'] == 'stopped' and stopped['control'] == 'idle'
    assert stopped['owns_execution'] is False and c.status()['needsStop'] is False
    assert stopped['revision'] > pending['revision']
    assert c.capture_context() is None
    runtime.close.assert_called_once()


def test_failed_pause_retains_ownership_and_never_claims_paused():
    c = controller()
    c.runtime.pause.side_effect = RuntimeError('No actuator acknowledgement')
    with pytest.raises(Exception, match='Pause could not be confirmed'): c.pause()
    failed = c.status()
    assert failed['execution']['lifecycle'] == 'failed'
    assert failed['execution']['control'] == 'cancelling'
    assert failed['execution']['owns_execution'] is True
    assert failed['paused'] is False


def test_failed_stop_retains_ownership_until_retry():
    c = controller()
    c.runtime.cancel.side_effect = RuntimeError('Cancellation unconfirmed')
    c.stop(); c.stop_thread.join(3)
    assert c.status()['execution']['lifecycle'] == 'failed'
    assert c.status()['execution']['owns_execution'] is True
    c.runtime.cancel.side_effect = None
    c.stop(); c.stop_thread.join(3)
    assert c.status()['execution']['lifecycle'] == 'stopped'
    assert c.status()['execution']['owns_execution'] is False


def test_restart_changes_epoch_and_requires_confirmed_stop(tmp_path):
    path = tmp_path / 'session.json'
    path.write_text(json.dumps({'session_id': 'session', 'phase': 'executing', 'revision': 900,
        'paused': True, 'control_status': 'paused', 'map_name': 'Home'}))
    previous = controller()
    recovery, factory = Mock(), Mock()
    restored = AutonomyController(previous.reasoning, factory, Mock(), Mock(), path, recovery)
    snapshot = restored.status()['execution']
    assert snapshot['epoch'] != previous.status()['execution']['epoch']
    assert snapshot['session_id'] == 'session'
    assert snapshot['lifecycle'] == 'interrupted' and snapshot['control'] == 'idle'
    assert snapshot['owns_execution'] is True
    factory.assert_not_called()
    restored.stop(); restored.stop_thread.join(3)
    recovery.assert_called_once()
    assert restored.status()['execution']['owns_execution'] is False
    assert not path.exists()


@pytest.mark.parametrize('command', ['pause', 'resume'])
def test_stop_wins_a_delayed_control_failure(command):
    c = controller()
    if command == 'resume': c.pause()
    entered, release, stopping, stop_release = (threading.Event() for _ in range(4))
    def delayed_failure():
        entered.set()
        assert release.wait(3)
        raise RuntimeError('Late actuator failure')
    def delayed_stop():
        stopping.set()
        assert stop_release.wait(3)
    getattr(c.runtime, command).side_effect = delayed_failure
    c.runtime.cancel.side_effect = delayed_stop
    thread, errors, _ = background(getattr(c, command))
    try:
        assert entered.wait(3)
        c.stop()
        release.set()
        assert stopping.wait(3)
        finish(thread, errors)
        state = c.status()['execution']
        assert state['lifecycle'] == 'stopping' and state['control'] == 'cancelling'
        assert state['owns_execution'] is True and state['error'] is None
    finally:
        release.set(); stop_release.set()
        thread.join(3)
        if c.stop_thread: c.stop_thread.join(3)
    assert c.status()['execution']['lifecycle'] == 'stopped'


def test_recovery_ignores_unknown_metadata_and_malformed_nested_state(tmp_path):
    path = tmp_path / 'session.json'
    path.write_text(json.dumps({'session_id': 'session', 'owned': False,
        'persist': 'metadata', 'record_context': 'metadata', 'execution': {'authority': 'old'},
        'mission': 'invalid', 'runner': [], 'pause_causes': [], 'recent_missions': {},
        'resume_checkpoint': False}))
    c = AutonomyController(Mock(), Mock(), Mock(), Mock(), path)
    state = c.status()
    assert state['execution']['lifecycle'] == 'interrupted'
    assert state['execution']['owns_execution'] is True
    assert state['execution']['mission_id'] is None
    assert state['execution']['pause_causes'] == {}
    assert 'persist' not in state and 'record_context' not in state
