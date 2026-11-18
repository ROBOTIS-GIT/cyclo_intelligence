import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main_runtime.policy_time_timeline import PolicyTimeTimeline
from main_runtime.arm_command_smoothing import ArmCommandSmoother,ArmSmoothingConfig
from main_runtime.tt_rtc_timeline import TTActionTimeline


def timeline():
    return PolicyTimeTimeline(indices=[0,1],max_velocity=.7,max_acceleration=6,max_jerk=100)


def test_base_matches_original_timeline_and_independent_smoother():
    p=timeline();p.seed_action([0,0,0]);p.pop_action()
    raw=TTActionTimeline();raw.push_actions([[0,0,0]])
    s=ArmCommandSmoother([0,1],config=ArmSmoothingConfig(max_velocity_rad_s=.7,max_acceleration_rad_s2=6))
    s.reset([0,0,0])
    chunk=np.column_stack([np.linspace(.1,.8,24),np.linspace(.02,.4,24),np.full(24,.7)])
    p.push_actions(chunk,action_policy_mode='base');raw.push_actions(chunk)
    for _ in range(140):
        expected=s.step(raw.pop_action(),.01)
        np.testing.assert_allclose(p.pop_action(),expected,atol=1e-10)
    assert p.buffer_size<6


def test_mode_follows_queued_waypoints_not_latest_response():
    p=timeline();p.seed_action([0,0,0]);p.push_actions([[.1,.03,.2]]*6,action_policy_mode='base')
    p.push_actions([[.8,.4,.6]]*10,action_policy_mode='rlt')
    for _ in range(20):p.pop_action()
    assert p.executing_policy=='base'
    assert p.buffer_size<16
    for _ in range(600):
        p.pop_action()
        if p.executing_policy=='rlt':break
    assert p.executing_policy=='rlt'
    before=p.buffer_size
    for _ in range(30):p.pop_action()
    assert p.buffer_size==before


def test_switch_brakes_continuously_then_common_progress_and_returns_to_base():
    p=timeline();p.seed_action([0,0,0])
    p.push_actions([[.3,.1,.3]]*6,action_policy_mode='base')
    p.push_actions([[.8,.5,.8]],action_policy_mode='rlt')
    p.push_actions([[.1,-.1,.2]]*8,action_policy_mode='base')
    output=[];modes=[];common=[]
    for _ in range(1000):
        q=p.pop_action()
        if q is not None:
            output.append(q);modes.append(p.executing_policy)
            if p.executing_policy=='rlt':common.append(q.copy())
        if not p.buffer_size and len(output)>100:break
    assert 'braking' in modes and 'rlt' in modes and modes[-1]=='base'
    q=np.array(output);v=np.diff(q[:,:2],axis=0)/.01;a=np.diff(v,axis=0)/.01
    assert abs(v).max()<=.7+1e-7
    assert abs(a).max()<=6+1e-6
    assert abs(np.diff(a,axis=0)/.01).max()<=100+1e-3
    common=np.array(common);delta=common[-1]-common[0]
    ratios=(common-common[0])/delta
    np.testing.assert_allclose(ratios,np.repeat(ratios[:,[0]],3,axis=1),atol=1e-6)


def test_queue_trimming_keeps_policy_tags_and_clear_resets_them():
    p=timeline();p.seed_action([0,0,0]);p.push_actions([[.4,.2,.1]]*12,action_policy_mode='base')
    before=p.peek_actions();p.replace_queued_actions(before,before[:6])
    p.push_actions([[.8,.4,.3]]*10,action_policy_mode='rlt')
    assert list(p._policies)==['base']*6+['rlt']*10
    p.clear();assert not p._policies


def test_runtime_enqueues_mlp_with_ownership_and_preserves_old_base_prefix():
    import os,time
    from unittest.mock import patch
    from test_common_time_timeline import configured_loop
    with patch.dict(os.environ,{'CYCLO_SG2_COMMON_TIME_RLT_ONLY':'true'}):loop,robot=configured_loop(True)
    p=loop._processor
    assert isinstance(p,PolicyTimeTimeline)
    prefix=np.full((6,16),.2);p.seed_action(np.zeros(16));p.push_actions(prefix,action_policy_mode='base')
    loop._stream_handoff_pending=False
    chunk=np.full((10,16),.6)
    ok,reason=loop._commit_tt_rtc_chunk_locked(chunk=chunk,captured_prefix=prefix,
        captured_actions=prefix,delay_steps=6,prefix_captured_at=time.monotonic(),
        action_policy_mode='rlt',vla_reference_chunk=chunk.copy())
    assert ok,reason
    assert list(p._policies)==['base']*6+['rlt']*10
    with patch.object(loop,'_should_request_actions',return_value=False):
        for i in range(10):
            with patch('main_runtime.playground_control_loop.time.monotonic',return_value=10+i*.01):loop.tick()
    assert p.executing_policy=='base'
    assert len(robot.commands)==10
