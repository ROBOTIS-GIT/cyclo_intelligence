"""Patrol motion has no mission selection authority, in live and training contracts."""
import json
import threading
from types import SimpleNamespace
import pytest
from supervisor_api import mission_decisions as contract
from supervisor_api.autonomy_context import context_for
from supervisor_api.autonomy_reasoner import SessionReasoner
from supervisor_api.autonomy_runtime_control import validate_tree
from supervisor_api.test_autonomy_continuous import harness, verdict
from supervisor_api.test_autonomy import eventually

PATROL = {'reasoning_task': 'patrol_supervision', 'active_mission': None,
          'observation': {'phase': 'moving', 'target_spot_id': 'laundry'}}
CATALOG = [{'mission_id': 'task', 'mission_name': 'DoLaundry'}]


@pytest.mark.parametrize('action', ['continue', 'pause'])
def test_patrol_prompt_and_training_have_no_task_missions(action):
    target = {'decision': action, 'mission_id': None, 'reason': 'A person is ahead.'}
    sample = contract.sample_for(PATROL, CATALOG, target, 'frame.jpg')
    user = json.loads(sample['messages'][1]['content'][1]['text'])
    assert user['missions'] == []
    assert user['context']['candidate_mission_ids'] == []
    assert 'patrol_supervision' in sample['messages'][0]['content'][0]['text']
    assert json.loads(sample['messages'][2]['content'][0]['text']) == target


@pytest.mark.parametrize('action,mid', [('start','task'),('wait',None),('continue','task'),('pause','task'),('cancel',None),('cancel','task')])
def test_patrol_rejects_mission_start_and_mission_id(action, mid):
    with pytest.raises(ValueError):
        contract.validate_target({'decision':action,'mission_id':mid,'reason':'Visible'}, PATROL, CATALOG)


def test_context_switches_to_selection_only_after_observation_arrival():
    state = {'mode':'observation','map_name':'house','observation_context':{'phase':'moving'}}
    for phase in ('moving','settling'):
        state['observation_context']['phase'] = phase
        context = context_for(state, ['task'])
        assert context['reasoning_task'] == 'patrol_supervision'
        assert context['candidate_mission_ids'] == []
    state['observation_context']['phase'] = 'observing'
    assert context_for(state,['task'])['reasoning_task'] == 'mission_selection'
    assert context_for(state,['task'])['candidate_mission_ids'] == ['task']


def test_invalid_patrol_start_holds_before_navigation(harness):
    h = harness
    h.service.decide.side_effect = lambda req: verdict(h, 'start')
    h.controller.start(h.request)
    eventually(lambda: h.controller.state['paused'])
    assert not any(c[0] == 'nav' for c in h.runtime.calls)
    assert h.controller.state['reasoning_task'] == 'patrol_supervision'
    h.repo.mission.assert_not_called()


def test_patrol_pause_resume_preserves_manual_hold(harness):
    h = harness
    h.controller.runtime = h.runtime
    h.controller.owned = True
    h.controller.state.update(active=True,mode='observation',phase='positioning',map_name='Home',
        mission=None,observation_context={'phase':'moving','target_spot_id':'a'})
    h.controller.transition()
    reasoner = SessionReasoner(h.controller,h.request,h.plan,'session')
    stamp = 0
    def reply(action):
        nonlocal stamp
        stamp += 1
        return {'context_version':h.controller.state['context_version'],'reasoning_task':'patrol_supervision',
                'frame_stamp':str(stamp),'action_allowed':True,
                'decision':{'decision':action,'mission_id':None,'reason':'Path'}}
    reasoner.infer = lambda _: reply('pause')
    reasoner.supervise()
    assert h.controller.state['paused'] and ('pause',) in h.runtime.calls
    h.controller.pause('Operator pause',source='manual')
    reasoner.infer = lambda _: reply('continue')
    reasoner.supervise(); reasoner.supervise()
    assert h.controller.state['pause_causes'] == {'manual':'Operator pause'}
    assert ('resume',) not in h.runtime.calls
    h.controller.resume(source='manual')
    assert ('resume',) in h.runtime.calls


@pytest.mark.parametrize('node', ['HomeLaundry','HomeTV'])
def test_inherited_atomic_prop_actions_are_admitted(node):
    validate_tree(f'<root main_tree_to_execute="Main"><BehaviorTree ID="Main"><Sequence><{node}/></Sequence></BehaviorTree></root>')
    with pytest.raises(ValueError):
        validate_tree('<root main_tree_to_execute="Main"><BehaviorTree ID="Main"><UnknownAction/></BehaviorTree></root>')
