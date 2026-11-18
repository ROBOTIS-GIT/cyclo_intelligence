"""Common clock geometry and dynamics; no robot transport."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_runtime.common_time_timeline import CommonTimeTimeline


def timeline(hz=100):
    return CommonTimeTimeline(source_hz=15, control_hz=hz, indices=[0,1],
                              max_velocity=.7, max_acceleration=6., max_jerk=100.)


def test_all_axes_share_progress_and_queue_waits_for_actual_arrival():
    p=timeline();p.seed_action(np.zeros(3));p.push_actions(np.array([[.74,-.3,.6],[.8,-.1,.2]]))
    before=p.peek_actions();values=[p.pop_action()]
    for _ in range(40):values.append(p.pop_action())
    np.testing.assert_array_equal(p.peek_actions(),before)
    assert p.remaining_duration_s>0.5
    while p.buffer_size==2:values.append(p.pop_action())
    values=np.array(values)
    np.testing.assert_allclose(values/np.array([.74,-.3,.6]),np.repeat(values[:,[0]]/.74,3,axis=1),atol=1e-12)
    np.testing.assert_array_equal(values[-1],before[0])
    assert p.buffer_size==1


@pytest.mark.parametrize('hz',[100,200])
def test_limits_at_large_jump_small_steps_reversals_and_starvation(hz):
    p=timeline(hz);p.seed_action(np.zeros(3))
    source=np.array([[.74,-.3,.6],[.75,-.29,.6],[-.2,.4,.1],[0,0,0]])
    p.push_actions(source);out=[p.pop_action()]
    while p.buffer_size:out.append(p.pop_action())
    out.extend([out[-1]]*4)
    assert p.pop_action() is None
    q=np.vstack([np.zeros((4,3)),out]);v=np.diff(q[:,:2],axis=0)*hz
    a=np.diff(v,axis=0)*hz;j=np.diff(a,axis=0)*hz
    assert abs(v).max()<=.7+1e-8
    assert abs(a).max()<=6+1e-8
    assert abs(j).max()<=100+1e-6
    np.testing.assert_array_equal(out[-1],source[-1])
    p.push_actions([[.1,.1,.1]])
    assert np.linalg.norm(p.pop_action())<.001


def test_inflight_prefix_and_active_target_are_not_rewritten():
    p=timeline();p.seed_action(np.zeros(3));p.push_actions([[.74,.2,0],[.5,.1,0]])
    for _ in range(20):p.pop_action()
    prefix=p.peek_actions();last=p.last_action
    p.replace_queued_actions(prefix,np.vstack([prefix,[.6,.2,0]]))
    np.testing.assert_array_equal(p.last_action,last)
    bad=p.peek_actions();changed=bad.copy();changed[0,0]=.8
    with pytest.raises(ValueError,match='active'):p.replace_queued_actions(bad,changed)
    p.clear();p.seed_action([-.2,-.1,0]);p.push_actions([[.1,.1,0]])
    np.testing.assert_array_equal(p.pop_action(),[-.2,-.1,0])


def test_late_tick_never_catches_up_and_validation():
    p=timeline();p.seed_action([0,0,0]);p.push_actions([[.7,.2,0]])
    p.pop_action();before=p.remaining_duration_s
    p.pop_action(elapsed_s=.1)
    assert before-p.remaining_duration_s==pytest.approx(.01)
    with pytest.raises(ValueError):p.pop_action(elapsed_s=float('nan'))
    with pytest.raises(ValueError):p.push_actions([[float('nan'),0,0]])


def configured_loop(physical=False, hz=100):
    from unittest.mock import patch
    import os
    from test_arm_command_smoothing import SmoothingRobot
    from test_playground_control_loop import ControlLoop
    robot=SmoothingRobot();loop=ControlLoop(requester=object())
    with patch('main_runtime.playground_control_loop.RobotClient',return_value=robot),patch.dict(os.environ,{
        'CYCLO_SG2_ARM_SMOOTHING':'true','CYCLO_SG2_COMMON_TIME_SCALING':'true',
        'CYCLO_ARM_MAX_VELOCITY_RAD_S':'.7','CYCLO_ARM_MAX_ACCELERATION_RAD_S2':'6',
        'CYCLO_ARM_MAX_JERK_RAD_S3':'100'}):
        loop.configure('ffw_sg2_rev1',action_keys=robot.action_keys,action_request_mode='tt_rtc',
                       tt_rtc_horizon=32,tt_rtc_action_dim=16,publish_to_robot=physical,control_hz=hz)
    loop._running=True
    return loop,robot


def test_integration_selects_common_clock_seeds_feedback_and_replaces_independent_filter():
    from unittest.mock import patch
    loop,robot=configured_loop(True)
    assert isinstance(loop._processor,CommonTimeTimeline)
    assert loop._arm_smoother is None
    robot.pose[:]=.1
    chunk=np.full((8,16),.3);chunk[:,8]=.8
    loop._processor.push_actions(chunk)
    with patch.object(loop,'_should_request_actions',return_value=False):
        for i in range(50):
            with patch('main_runtime.playground_control_loop.time.monotonic',return_value=10+i*.01):loop.tick()
    np.testing.assert_array_equal(robot.commands[0][0],robot.pose)
    last=robot.commands[-1][0]
    np.testing.assert_allclose((last-.1)/(chunk[0]-.1),np.repeat((last[8]-.1)/.7,16))
    assert loop._processor.buffer_size==8


def test_refill_waits_for_retimed_prefix_and_switch_only_trims_uncommitted_tail():
    loop,_=configured_loop()
    p=loop._processor
    p.seed_action(np.zeros(16));chunk=np.full((12,16),.74);chunk[:,8]=np.linspace(.74,1.5,12)
    p.push_actions(chunk);p.pop_action()
    assert not loop._should_request_actions(p)
    while p.buffer_size>6:p.pop_action()
    assert p.peek_actions().shape[0]==6
    assert loop._should_request_actions(p)
    p.clear();p.seed_action(np.zeros(16));p.push_actions(chunk);p.pop_action();p.pop_action()
    anchor=p.last_action.copy();prefix=p.peek_actions(6)
    loop._pending_action_policy_mode='rlt'
    assert loop._should_request_actions(p,action_policy_mode='rlt')
    np.testing.assert_array_equal(p.peek_actions(),prefix)
    np.testing.assert_array_equal(p.last_action,anchor)


def test_common_clock_failure_pauses_without_command_and_resume_clears_feedback():
    from unittest.mock import patch
    loop,robot=configured_loop(True)
    loop._processor.push_actions(np.full((8,16),.3))
    with patch.object(robot,'get_current_action_positions',side_effect=RuntimeError('stale feedback')):loop.tick()
    assert not loop._running
    assert not robot.commands
    assert 'stale feedback' in loop.tt_rtc_failure_reason


def test_delayed_mlp_response_keeps_slow_prefix_and_raw_chunk_geometry():
    import time
    loop,_=configured_loop();p=loop._processor
    p.seed_action(np.zeros(16))
    prefix=np.full((6,16),.5);prefix[:,8]=np.linspace(.74,.8,6)
    p.push_actions(prefix)
    captured_at=time.monotonic()
    for _ in range(60):p.pop_action()
    np.testing.assert_array_equal(p.peek_actions(),prefix)
    chunk=np.full((10,16),.25);chunk[:,8]=np.linspace(-.45,-.46,10)
    committed,reason=loop._commit_tt_rtc_chunk_locked(
        chunk=chunk,captured_prefix=prefix,captured_actions=prefix,delay_steps=6,
        prefix_captured_at=captured_at,action_policy_mode='rlt',vla_reference_chunk=chunk.copy())
    assert committed,reason
    np.testing.assert_array_equal(p.peek_actions(),np.vstack([prefix,chunk]))
    loop._pending_action_policy_mode='rlt';loop._pending_rlt_robot_override=True
    before=p.last_action.copy()
    with loop._lock:
        loop._commit_tt_rtc_action_policy_locked('rlt')
    np.testing.assert_array_equal(p.last_action,before)
    assert loop._active_action_policy_mode=='rlt'


def test_common_timeline_position_limits_and_terminal_hold_are_enforced():
    p=CommonTimeTimeline(source_hz=15,control_hz=100,indices=[0],max_velocity=.7,
                         max_acceleration=6,max_jerk=100,lower=[-1],upper=[1])
    with pytest.raises(ValueError,match='URDF'):p.push_actions([[1.1,0]])
    p.seed_action([0,0]);p.push_actions([[.2,.7]])
    while p.buffer_size:p.pop_action()
    final=p.last_action
    assert p.pop_action() is None
    np.testing.assert_array_equal(p.last_action,final)
    p.push_actions([[-.2,.2]])
    first=p.pop_action()
    assert np.max(abs(first-final))<.001


def test_late_control_gap_pauses_before_publishing_another_command():
    from unittest.mock import patch
    loop,robot=configured_loop(True);loop._processor.push_actions(np.full((6,16),.3))
    with patch.object(loop,'_should_request_actions',return_value=False):
        with patch('main_runtime.playground_control_loop.time.monotonic',return_value=10):loop.tick()
        count=len(robot.commands)
        with patch('main_runtime.playground_control_loop.time.monotonic',return_value=11):loop.tick()
    assert not loop._running
    assert len(robot.commands)==count
    assert '200ms' in loop.tt_rtc_failure_reason


def test_common_timeline_uses_requested_control_rate():
    loop,_=configured_loop(hz=200)
    assert loop._processor.output_hz==200
