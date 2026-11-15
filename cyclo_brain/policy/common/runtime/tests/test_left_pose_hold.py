"""Pose-gated hold: arrival evidence, release intent, queue/prefix consistency."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_runtime.left_pose_hold import LeftPoseHold, HoldConfig, REFERENCE, MODEL_NAME, enabled_for_model
from test_position_deadband import NamedRobot
from test_playground_control_loop import ControlLoop


def action(offset=0.):
    a = np.zeros(16)
    a[:7] = REFERENCE + offset
    a[7], a[15] = .9, .4
    return a


def hold():
    return LeftPoseHold(range(7), [-3.]*7, [3.]*7)


def ready(f, start=0.):
    for i in range(8):
        f.observe(action(), start+i*.05)
    assert f.feedback_ready


def test_hold_requires_measured_arrival_dwell_and_small_actual_motion():
    f = hold(); data=np.tile(action(), (15,1))
    out, state, events = f.plan(data, action(), 0.)
    assert not events
    ready(f)
    out, state, events = f.plan(data, action(.002), .35)
    assert events == [(0, 'align'), (1, 'hold')]
    np.testing.assert_allclose(out[:, :7], np.tile(REFERENCE, (15,1)), atol=1e-15, rtol=0)
    assert f.mode == 'moving'  # Candidate is not installed before enqueue.
    # Passing through the entry region is not stable arrival.
    f=hold()
    for i in range(8):f.observe(action(-.009+i*.0025), i*.05)
    assert not f.feedback_ready
    # Stale observations cannot authorize a new hold.
    f=hold();ready(f)
    assert f.plan(data, action(), 1.)[2] == []
    f.observe(action(), 1.); assert not f.feedback_ready


def test_outside_reference_is_bit_exact_even_for_tiny_intended_adjustments():
    f=hold();ready(f)
    data=np.tile(action(.1),(100,1));data[:,0]+=np.arange(100)*1e-6
    out, _, events=f.plan(data, action(.1), .35)
    np.testing.assert_array_equal(out,data); assert not events


def test_stationary_noise_is_fixed_without_changing_right_or_grippers():
    f=hold();ready(f)
    data=np.tile(action(),(100,1))
    data[:,:7]+=(.012*np.sin(np.arange(100)*.3))[:,None]
    data[:,8:15]=(-np.arange(100)*1e-6)[:,None]
    data[:,7]=np.arange(100)*.001;data[:,15]=.2
    out,f,events=f.plan(data, action(), .35)
    np.testing.assert_array_equal(out[:,:7],np.tile(REFERENCE,(100,1)))
    np.testing.assert_array_equal(out[:,7:],data[:,7:])
    assert events==[(0,'align'),(0,'hold')]


def test_release_requires_same_joint_same_direction_and_resumes_with_bounded_steps():
    f=hold();ready(f)
    _,f,_=f.plan(np.tile(action(),(3,1)),action(),.35)
    noise=np.tile(action(),(20,1));noise[:,0]+=.04*(-1.)**np.arange(20)
    out,f,events=f.plan(noise,action(),.4)
    assert not events;np.testing.assert_array_equal(out[:,:7],np.tile(REFERENCE,(20,1)))
    # One spike or two samples across the threshold cannot release.
    data=np.tile(action(),(20,1));data[:,0]+=.06
    out,f,events=f.plan(data,action(),.45)
    assert events==[(3,'release')]
    np.testing.assert_array_equal(out[:3,:7],np.tile(REFERENCE,(3,1)))
    assert np.max(np.abs(np.diff(np.vstack([REFERENCE,out[:,:7]]),axis=0))) <= .01+1e-12
    np.testing.assert_array_equal(out[-1],data[-1])


def test_release_evidence_survives_chunk_boundary_and_resets_on_pause():
    f=hold();ready(f)
    _,f,_=f.plan(np.tile(action(),(3,1)),action(),.35)
    a=action();a[0]+=.04
    out,f,events=f.plan(np.tile(a,(3,1)),action(),.4);assert not events
    out,f,events=f.plan(a[None,:],out[-1],.5);assert events==[(0,'release')]
    f.clear();assert f.mode=='moving' and not f.feedback_ready


def test_bad_data_is_rejected_transactionally():
    f=hold();ready(f)
    for value in [np.nan,np.inf,4.]:
        a=action();a[3]=value
        with pytest.raises(ValueError):f.plan(a[None,:],action(),.35)
    assert f.mode=='moving' and f.feedback_ready
    for kwargs in [dict(entry_rad=.03),dict(exit_rad=float('nan')),dict(stable_s=.01)]:
        with pytest.raises(ValueError):HoldConfig(**kwargs)


def make_loop(monkeypatch, keys=('arm_left','arm_right'), physical=False, model=MODEL_NAME, rlt=False):
    monkeypatch.setenv('CYCLO_LEFT_ARM_POSE_HOLD','true')
    monkeypatch.setenv('CYCLO_LEFT_ARM_DEADBAND','false')
    monkeypatch.setenv('CYCLO_SG2_ARM_SMOOTHING','false')
    robot=NamedRobot()
    robot.get_current_action_positions=lambda keys: action()
    loop=ControlLoop(requester=object())
    with patch('main_runtime.playground_control_loop.RobotClient',return_value=robot), \
            patch('main_runtime.playground_control_loop.create_rlt_trace_publisher',return_value=None):
        loop.configure('ffw_sg2_rev1', action_keys=list(keys),model_path='/models/'+model,
                       action_request_mode='tt_rtc',tt_rtc_horizon=32,tt_rtc_action_dim=16,
                       publish_to_robot=physical,rlt_enabled=rlt)
    loop._running=True
    loop._stream_handoff_pending=False
    return loop,robot


def test_rlt_preload_uses_bundle_hold_without_legacy_pose_filter(monkeypatch):
    loop, _ = make_loop(monkeypatch, rlt=True, physical=True)
    assert loop._rlt_enabled
    assert loop._left_pose_hold is None
    assert loop._active_action_policy_mode == 'base'
    # Disabling the extra pose filter must not bypass physical RLT routing consent.
    ok, reason = loop.set_action_policy('rlt', timeout_s=0)
    assert not ok and 'rlt_robot_override' in reason
    loop.deconfigure()
    base, _ = make_loop(monkeypatch)
    assert base._left_pose_hold is not None


@pytest.mark.parametrize('conflict', ['overlap', 'smoothing', 'deadband', 'async'])
def test_rlt_pose_filter_selection_keeps_other_compatibility_checks(monkeypatch, conflict):
    loop, _ = make_loop(monkeypatch)
    if conflict == 'overlap':
        loop._tt_rtc_temporal_overlap_steps = 1
    elif conflict == 'smoothing':
        loop._arm_smoother = object()
    elif conflict == 'deadband':
        loop._left_arm_deadband = object()
    else:
        loop._action_request_mode = 'async'
    with pytest.raises(ValueError, match='no overlap or other arm filters'):
        loop._configure_left_pose_hold_locked('/models/' + MODEL_NAME, True)


def test_scope_and_joint_mapping(monkeypatch):
    loop,_=make_loop(monkeypatch,keys=('arm_right','arm_left'))
    np.testing.assert_array_equal(loop._left_pose_hold.indices,np.arange(8,15))
    loop,_=make_loop(monkeypatch,model='another-model')
    assert loop._left_pose_hold is None
    monkeypatch.delenv('CYCLO_LEFT_ARM_POSE_HOLD')
    assert not enabled_for_model(MODEL_NAME)


@pytest.mark.parametrize('physical',[False,True])
def test_committed_prefix_matches_held_robot_commands_and_next_refill(monkeypatch,physical):
    loop,robot=make_loop(monkeypatch,physical=physical)
    ready(loop._left_pose_hold)
    prefix=np.tile(action(),(6,1));loop._processor.push_actions(prefix)
    chunk=np.tile(action(),(32,1));chunk[6:,:7]+=.004
    chunk[:,8:15]=np.arange(32)[:,None]*1e-5
    chunk[:6]=prefix
    with patch('main_runtime.playground_control_loop.time.monotonic',return_value=.35):
        ok,reason=loop._commit_tt_rtc_chunk_locked(chunk=chunk,captured_prefix=prefix,
            captured_actions=prefix,delay_steps=6,prefix_captured_at=.1,action_policy_mode='base')
    assert ok,reason
    queued=loop._processor.peek_actions()
    np.testing.assert_array_equal(queued[:,:7],np.tile(REFERENCE,(32,1)))
    np.testing.assert_array_equal(queued[:,7:],chunk[:,7:])
    with patch.object(loop,'_should_request_actions',return_value=False):
        for i in range(170):
            with patch('main_runtime.playground_control_loop.time.monotonic',return_value=.36+i*.01):loop.tick()
    assert robot.previews
    for a,_ in robot.previews:np.testing.assert_allclose(a[:7],REFERENCE,rtol=0,atol=1e-15)
    if physical:
        for a,_ in robot.commands:np.testing.assert_allclose(a[:7],REFERENCE,rtol=0,atol=1e-15)
    prefix=loop._capture_tt_rtc_prefix(loop._processor)
    captured=loop._capture_tt_rtc_actions(loop._processor)
    np.testing.assert_array_equal(prefix,loop._select_tt_rtc_vla_prefix(prefix))
    # The model returns the conditioned prefix, and the next queue commit accepts it.
    chunk=np.tile(action(),(32,1));chunk[:len(prefix)]=prefix
    with patch('main_runtime.playground_control_loop.time.monotonic',return_value=2.1):
        ok,reason=loop._commit_tt_rtc_chunk_locked(chunk=chunk,captured_prefix=prefix,
            captured_actions=captured,delay_steps=len(prefix),prefix_captured_at=2.,action_policy_mode='base')
    assert ok,reason
    loop.pause();assert not loop._left_pose_hold.feedback_ready
    assert loop._left_pose_hold.mode=='moving'
    loop.deconfigure();assert loop._left_pose_hold is None


def test_failed_enqueue_does_not_change_planner_state(monkeypatch):
    loop,_=make_loop(monkeypatch);ready(loop._left_pose_hold)
    prefix=np.tile(action(),(6,1));loop._processor.push_actions(prefix)
    with patch.object(loop._processor,'push_actions',side_effect=RuntimeError('enqueue failed')),patch('main_runtime.playground_control_loop.time.monotonic',return_value=.35):
        ok,_=loop._commit_tt_rtc_chunk_locked(chunk=np.tile(action(),(32,1)),captured_prefix=prefix,
            captured_actions=prefix,delay_steps=6,prefix_captured_at=.1,action_policy_mode='base')
    assert not ok and loop._left_pose_hold.mode=='moving'


def test_stale_feedback_pauses_before_any_command(monkeypatch):
    loop,robot=make_loop(monkeypatch,physical=True)
    robot.get_current_action_positions=lambda keys: (_ for _ in ()).throw(RuntimeError('stale feedback'))
    loop._processor.push_actions(np.tile(action(),(32,1)))
    loop.tick()
    assert not loop._running and not robot.commands and not robot.previews


def test_unobserved_future_arrival_cannot_latch():
    f=hold();ready(f)
    data=np.tile(action(),(30,1));data[:10,:7]+=.1
    out,_,events=f.plan(data,action(),.35)
    assert not events
    np.testing.assert_array_equal(out,data)
    out,_,events=f.plan(np.tile(action(),(10,1)),action(),.35,allow_entry=False)
    assert not events


def test_rlt_switch_is_rejected_while_pose_hold_is_configured(monkeypatch):
    loop,_=make_loop(monkeypatch)
    ok,reason=loop.set_action_policy('rlt')
    assert not ok and 'base policy' in reason


def test_alignment_reaches_exact_reference_without_claiming_unmeasured_arrival():
    f=hold()
    anchor=action();anchor[:7]+=np.array([.009,-.009,.008,-.007,.006,-.005,.004])
    for i in range(8):f.observe(anchor,i*.05)
    assert f.feedback_ready and not f.feedback_at_target
    raw=np.tile(anchor,(26,1))
    raw[:,8:15]=np.arange(26)[:,None]*1e-6
    raw[:,7]=.4;raw[:,15]=.3
    out,f,events=f.plan(raw,anchor,.35)
    assert events==[(0,'align')]
    assert f.mode=='aligning'  # Queued arrival alone is insufficient.
    np.testing.assert_array_equal(out[-1,:7],REFERENCE)
    np.testing.assert_array_equal(out[:,7:],raw[:,7:])
    trajectory=np.vstack([anchor[:7],out[:,:7]])
    assert np.max(np.abs(np.diff(trajectory,axis=0))) <= .03/15+1e-12
    error=np.abs(trajectory-REFERENCE)
    assert np.all(np.diff(error,axis=0) <= 1e-12)
    # Reference targets remain exact, even while actual feedback is still offset.
    for i in range(8):f.observe(anchor,.4+i*.05)
    out,f,events=f.plan(raw,out[-1],.75)
    assert not events and f.mode=='aligning'
    np.testing.assert_array_equal(out[:,:7],np.tile(REFERENCE,(26,1)))
    # Actual stable arrival authorizes HOLD on a subsequent committed postfix.
    for i in range(8):f.observe(action(),.8+i*.05)
    out,f,events=f.plan(raw,out[-1],1.15)
    assert events==[(0,'hold')] and f.mode=='holding'
    np.testing.assert_array_equal(out[:,:7],np.tile(REFERENCE,(26,1)))


def test_stale_arrival_feedback_cannot_confirm_hold():
    f=hold()
    for i in range(8):f.observe(action(.008),i*.05)
    out,f,_=f.plan(np.tile(action(.008),(26,1)),action(.008),.35)
    ready(f,start=.4)
    assert f.feedback_at_target
    _,f,events=f.plan(np.tile(action(),(26,1)),out[-1],2.)
    assert f.mode=='aligning' and not events


def test_sustained_departure_can_cancel_alignment_before_actual_arrival():
    f=hold()
    for i in range(8):f.observe(action(.008),i*.05)
    out,f,_=f.plan(np.tile(action(.008),(2,1)),action(.008),.35)
    assert f.mode=='aligning'
    raw=np.tile(action(.06),(26,1))
    out,f,events=f.plan(raw,out[-1],.4)
    assert events==[(3,'release')]
    np.testing.assert_array_equal(out[-1],raw[-1])
    assert f.mode=='moving'


def test_invalid_alignment_configuration_is_rejected():
    for kwargs in [dict(arrival_rad=.010),dict(arrival_rad=0),
                   dict(alignment_speed_rad_s=.16),dict(alignment_speed_rad_s=float('nan'))]:
        with pytest.raises(ValueError):HoldConfig(**kwargs)


def test_control_loop_simulated_tracking_aligns_and_confirms_actual_arrival(monkeypatch):
    loop,robot=make_loop(monkeypatch,physical=True)
    measured=action(.008)
    robot.get_current_action_positions=lambda keys: measured.copy()
    loop._processor.push_actions(np.tile(action(.008),(32,1)))
    seen_modes=set()
    commands=[]
    with patch.object(loop,'_should_request_actions',return_value=False):
        for i in range(600):
            now=i*.01
            with patch('main_runtime.playground_control_loop.time.monotonic',return_value=now):
                loop.tick()
                commands.append(robot.commands[-1][0].copy())
                # Test plant only: first-order position tracking, tau=165ms.
                measured += (commands[-1]-measured)*(1-np.exp(-.01/.165))
                if loop._processor.buffer_size<=6:
                    prefix=loop._capture_tt_rtc_prefix(loop._processor)
                    captured=loop._capture_tt_rtc_actions(loop._processor)
                    chunk=np.tile(action(.008),(32,1));chunk[:len(prefix)]=prefix
                    ok,reason=loop._commit_tt_rtc_chunk_locked(
                        chunk=chunk,captured_prefix=prefix,captured_actions=captured,
                        delay_steps=len(prefix),prefix_captured_at=now,action_policy_mode='base')
                    assert ok,reason
                    seen_modes.add(loop._left_pose_hold.mode)
    assert {'aligning','holding'} <= seen_modes
    np.testing.assert_allclose(commands[-1][:7],REFERENCE,rtol=0,atol=1e-15)
    assert np.max(np.abs(measured[:7]-REFERENCE)) < .002
    assert np.max(np.abs(np.diff(np.array(commands)[:,:7],axis=0))) <= .03/100+1e-12
    assert loop._processor.timing_stats['underrun_events']==0
