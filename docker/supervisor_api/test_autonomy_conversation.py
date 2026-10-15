import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from fastapi import HTTPException
from supervisor_api.autonomy import AutonomyController
from supervisor_api.autonomy_dialogue import answer_message, direct_action, question_only
from supervisor_api.autonomy_direction import change_direction, next_route_index, PatrolRedirect
from supervisor_api.autonomy_session import PatrolSession


def controller():
    service=Mock();service.status.return_value={'phase':'ready'};service.observe_lock=threading.Lock()
    c=AutonomyController(service,Mock(),Mock(),Mock())
    c.runtime=Mock();c.runtime.patrol_redirect=threading.Event()
    c.route_points=[{'spot_id':str(i),'label':str(i)} for i in range(3)]
    c.plan_snapshot={'route_closed':True};c.owned=True
    c.transition(active=True,session_id='one',map_name='house',observation_mode='patrol',mode='observation',
                 phase='observing',patrol_index=0,patrol_direction=1,observation_context={'phase':'observing'})
    return c


def wait_done(chat):
    end=time.monotonic()+3
    while chat.pending and time.monotonic()<end:time.sleep(.005)
    assert not chat.pending


def test_reverse_closed_loop_and_forward_restore():
    c=controller();change_direction(c,'reverse_patrol')
    assert c.state['patrol_direction']==-1
    assert c.state['observation_context']['next_spot_id']=='2'
    assert c.state['observation_context']['can_advance']
    change_direction(c,'forward_patrol');assert c.state['observation_context']['next_spot_id']=='1'
    assert next_route_index(0,3,-1,False)==0
    assert next_route_index(2,3,1,False)==2


def test_direction_during_mission_is_deferred_without_cancel():
    c=controller();c.transition(mode='mission',phase='executing')
    assert 'after the mission' in change_direction(c,'reverse_patrol')
    assert not c.runtime.patrol_redirect.is_set();c.runtime.cancel.assert_not_called()


def test_moving_direction_requests_goal_only_redirect():
    c=controller();c.transition(phase='positioning',observation_context={'phase':'moving'})
    change_direction(c,'reverse_patrol');assert c.runtime.patrol_redirect.is_set()
    c.runtime.cancel.assert_not_called();assert not c.cancelled.is_set()


def test_inactive_direction_rejected():
    c=controller();c.update(active=False)
    with pytest.raises(HTTPException):change_direction(c,'reverse_patrol')


def test_no_actions_for_invalid_model_output():
    for value in ['{}', 'run_shell', 'pause extra']:
        with pytest.raises(ValueError): answer_message(Mock(return_value={'text':value}), 'Turn around now', {}, [])


def test_explanations_do_not_request_model_command_classification():
    request = Mock(return_value={'text':'It changes the visitation order.'})
    assert answer_message(request, '역방향 순회가 로봇을 뒤로 운전한다는 뜻이야? 설명만 해줘.', {}, [])['action']=='none'
    assert request.call_count==1
    assert 'history' in request.call_args.args[1]
    assert direct_action('사람있다 멈춰!')=='pause'
    assert question_only('Explain pause.')


def test_chat_calls_model_and_applies_command_once():
    c=controller()
    with patch('supervisor_api.reasoning.qwen_request',return_value={'text':'reverse_patrol'}) as infer:
        c.conversation.submit('Please reverse my observation route now','id1');wait_done(c.conversation)
        c.conversation.submit('Please reverse my observation route now','id1');wait_done(c.conversation)
    assert c.state['patrol_direction']==-1;assert infer.call_count==1
    assert infer.call_args.args[1]['use_adapter'] is False
    assert c.conversation.messages[-1]['action']=='reverse_patrol'


def test_chat_questions_preserve_motion_and_include_history():
    c=controller()
    with patch('supervisor_api.reasoning.qwen_request',return_value={'text':'Observing here.'}) as infer:
        c.conversation.submit('What are you doing?','id1');wait_done(c.conversation)
        c.conversation.submit('Why?','id2');wait_done(c.conversation)
    assert len(infer.call_args.args[1]['history'])==2
    assert c.state['patrol_direction']==1;c.runtime.pause.assert_not_called()


def test_changed_context_cannot_execute_old_command():
    c=controller()
    def answer(*args,**kwargs):
        c.transition(phase='executing')
        return {'text':'reverse_patrol'}
    with patch('supervisor_api.reasoning.qwen_request',side_effect=answer):
        c.conversation.submit('Please reverse my observation route now','id1');wait_done(c.conversation)
    assert c.state['patrol_direction']==1;assert 'changed' in c.conversation.error


def test_pause_bypasses_model_and_ai_pause_cannot_be_cleared_by_resume():
    c=controller()
    c.pause('Obstacle',source='ai')
    with patch('supervisor_api.reasoning.qwen_request') as infer:
        c.conversation.submit('pause','id1');wait_done(c.conversation)
        c.conversation.submit('resume','id2');wait_done(c.conversation)
    infer.assert_not_called();assert c.state['paused'];assert 'ai' in c.state['pause_causes'];assert 'manual' not in c.state['pause_causes']


def test_session_uses_changed_direction():
    c=controller();plan=SimpleNamespace(mode='patrol',route_closed=True,excluded_mission_ids=[])
    points=[{'spot_id':str(i),'mission_ids':[]} for i in range(3)]
    s=PatrolSession(c,Mock(),plan,points)
    assert s.next_index()==1
    change_direction(c,'reverse_patrol');assert s.next_index()==2
    plan.route_closed=False;assert not s.can_advance()


def test_chat_admission_race_does_not_pause_robot():
    from supervisor_api.autonomy_reasoner import SessionReasoner
    c = controller()
    c.reasoning.status.return_value = {'phase':'ready', 'session_id':'qwen'}
    c.reasoning.decide.side_effect = HTTPException(409, 'A Qwen request is already in progress')
    c.capture_context = lambda: {'reasoning_task':'mission_selection', 'active_mission':None,
        'operator_instruction':'', 'recent_missions':[], 'observation':{}, 'candidate_mission_ids':[]}
    c._wait = Mock()
    r = SessionReasoner(c, SimpleNamespace(map_name='house'), SimpleNamespace(resume_confirmations=2), 'qwen')
    assert r.infer() is None
    assert not c.state['paused']


def test_task_instruction_is_included_in_live_reasoning_without_starting_mission():
    c = controller()
    with patch('supervisor_api.reasoning.qwen_request',return_value={'text':'instruction'}):
        c.conversation.submit('식탁을 치워줘', 'task1'); wait_done(c.conversation)
    assert c.state['operator_instruction'] == '식탁을 치워줘'
    assert c.capture_context()['operator_instruction'] == '식탁을 치워줘'
    assert c.state['phase'] == 'observing'
    c.runtime.navigate.assert_not_called()
    c.runtime.tree.assert_not_called()
    c.conversation.submit('지령 해제', 'task2'); wait_done(c.conversation)
    assert c.state['operator_instruction'] == ''


def test_urgent_pause_does_not_wait_for_model_or_require_json():
    c = controller(); c.reasoning.status.return_value = {'phase':'loading'}
    with patch('supervisor_api.reasoning.qwen_request') as infer:
        c.conversation.submit('사람있다. 멈춰봐!', 'urgent'); wait_done(c.conversation)
    infer.assert_not_called()
    c.runtime.pause.assert_called_once()
    assert c.state['paused'] and 'manual' in c.state['pause_causes']
    assert c.conversation.messages[-1]['action'] == 'pause'


def test_repeated_reverse_requests_keep_reverse_order():
    c = controller()
    change_direction(c, 'reverse_patrol'); change_direction(c, 'reverse_patrol')
    assert c.state['patrol_direction'] == -1
