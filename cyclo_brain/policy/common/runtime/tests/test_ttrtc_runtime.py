"""TT-RTC lifecycle and wire integration, using fake robot/engine transports."""
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
robot_client_stub = types.ModuleType('robot_client')
robot_client_stub.RobotClient = object
sys.modules.setdefault('robot_client', robot_client_stub)
from engine_process.protocol import (EngineCommandRequest, EngineCommandResponse,
                                     request_from_message, request_to_message_kwargs)
from main_runtime.inference_requester import InferenceRequester
from main_runtime.control_loop import ControlLoop
from main_runtime import control_loop as loop_module
from main_runtime.service_handler import ServiceHandler
from main_runtime.session_state import SessionState


def test_requester_transports_empty_and_warm_prefix_with_actual_length():
    captured = []
    def call(request, timeout_s):
        captured.append(request)
        return EngineCommandResponse(success=True, seq_id=request.seq_id)
    requester = InferenceRequester(SimpleNamespace(call=call))
    requester.load_policy(SimpleNamespace(ttrtc_enabled=True, ttrtc_prefix_steps=8))
    for d in (0, 7, 8):
        prefix = np.arange(d*16, dtype=float).reshape(d, 16)
        assert requester.get_action('task', ttrtc_prefix_actions=prefix).success
        restored = request_from_message(SimpleNamespace(**request_to_message_kwargs(captured[-1])))
        assert restored.ttrtc_enabled
        assert restored.ttrtc_prefix_steps == d
        np.testing.assert_array_equal(restored.ttrtc_prefix_actions, prefix.ravel())
    assert captured[0].ttrtc_enabled
    assert captured[0].ttrtc_prefix_steps == 8


def test_legacy_wire_request_keeps_tt_rtc_disabled():
    request = request_from_message(SimpleNamespace(command=1))
    assert not request.ttrtc_enabled
    assert not request.ttrtc_prefix_actions


@pytest.mark.parametrize('prefix', [np.zeros((9, 16)), np.zeros((7, 8)), np.full((1, 16), np.nan)])
def test_bad_prefix_fails_before_transport(prefix):
    client = mock.Mock()
    requester = InferenceRequester(client)
    with pytest.raises(ValueError):
        requester.get_action('task', ttrtc_prefix_actions=prefix)
    client.call.assert_not_called()
    assert not requester.has_pending_get_action()


class Robot:
    action_keys = ['arm_left', 'arm_right']
    def __init__(self, *args, **kwargs):
        self.commands = []
    def close(self):
        pass
    def publish_action(self, action, keys):
        self.commands.append(action)
    def publish_initial_pose_sync(self, action, keys, duration_s):
        self.commands.append(action)


def test_tt_rtc_publishes_24_step_postfix_through_robot_command_path():
    chunk = np.arange(32*16, dtype=float).reshape(32, 16) / 1000
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions.copy())
        return SimpleNamespace(success=True, chunk_size=32, action_dim=16,
                               action_list=chunk.ravel(), timing_json='{}')
    loop = ControlLoop(SimpleNamespace(get_action=infer))
    with mock.patch.object(loop_module, 'RobotClient', Robot):
        loop.configure('fake', action_keys=Robot.action_keys, ttrtc_enabled=True,
                       action_request_mode='async', publish_to_robot=True)
    now = [0.]
    loop._ttrtc.clock = lambda: now[0]
    try:
        loop.start()
        with mock.patch.object(loop_module.time, 'monotonic', side_effect=lambda: now[0]):
            loop.tick()
            loop._ttrtc.thread.join(2.)
            assert not loop._ttrtc.thread.is_alive()
            assert prefixes[0].shape == (0, 16)
            assert loop._ttrtc.buffer_size == 24
            np.testing.assert_array_equal(np.asarray(loop._ttrtc.actions), chunk[:24])
            assert not loop._robot.commands
            now[0] = 1/15
            loop.tick()
            assert len(loop._robot.commands) == 1
            np.testing.assert_array_equal(loop._robot.commands[0], chunk[0])
            assert loop._ttrtc.buffer_size == 23
    finally:
        loop.shutdown()


@pytest.mark.parametrize('boundary', ['pause', 'instruction', 'subtask', 'deconfigure', 'publishing'])
def test_control_lifecycle_discards_inflight_prefix_and_outputs(boundary):
    entered, release = threading.Event(), threading.Event()
    def infer(task, *, ttrtc_prefix_actions):
        entered.set()
        assert release.wait(2.)
        chunk = np.zeros((32, 16))
        return SimpleNamespace(success=True, chunk_size=32, action_dim=16,
                               action_list=chunk.ravel(), timing_json='{}')
    loop = ControlLoop(SimpleNamespace(get_action=infer))
    with mock.patch.object(loop_module, 'RobotClient', Robot):
        loop.configure('fake', action_keys=Robot.action_keys, ttrtc_enabled=True)
    loop.start()
    loop.tick()
    ex = loop._ttrtc
    assert entered.wait(1.)
    if boundary == 'pause':
        assert loop.pause()
    elif boundary == 'instruction':
        loop.set_task_instruction('next task')
    elif boundary == 'subtask':
        loop.transition_subtask('next task')
    elif boundary == 'deconfigure':
        loop.deconfigure()
    else:
        loop.set_publish_to_robot(True)
    release.set()
    ex.thread.join(2.)
    assert not ex.thread.is_alive()
    assert ex.buffer_size == 0
    assert loop._request_thread is None
    assert loop._tick_period() == 1/15
    loop.shutdown()


@pytest.mark.parametrize('overrides', [dict(action_request_mode='sync'),
                                      dict(vla_rail_enabled=True), dict(inference_hz=30),
                                      dict(ttrtc_prefix_steps=9)])
def test_invalid_configuration_does_not_construct_robot(overrides):
    loop = ControlLoop(object())
    with mock.patch.object(loop_module, 'RobotClient') as robot:
        with pytest.raises(ValueError):
            loop.configure('fake', ttrtc_enabled=True, **overrides)
        robot.assert_not_called()
    loop.shutdown()


@pytest.mark.parametrize('horizon', [31, 32])
def test_initial_pose_sync_uses_empty_prefix_and_validates_tt_rtc_shape(horizon):
    prefixes = []
    def infer(task, *, ttrtc_prefix_actions):
        prefixes.append(ttrtc_prefix_actions)
        return SimpleNamespace(success=True, chunk_size=horizon, action_dim=16,
                               action_list=np.zeros(horizon*16), timing_json='{}')
    loop = ControlLoop(SimpleNamespace(get_action=infer))
    with mock.patch.object(loop_module, 'RobotClient', Robot):
        loop.configure('fake', action_keys=Robot.action_keys, ttrtc_enabled=True,
                       initial_pose_sync=True, publish_to_robot=True)
    if horizon == 32:
        assert loop.start()
        assert len(loop._robot.commands) == 1
        assert loop._ttrtc.buffer_size == 0
    else:
        with pytest.raises(ValueError, match='32, 16'):
            loop.start()
        assert not loop._robot.commands
    assert prefixes[0].shape == (0, 16)
    loop.shutdown()


def test_tt_rtc_default_rate_is_15_even_when_generic_default_is_30():
    loop = ControlLoop(object(), inference_hz=30, target_chunk_size=32)
    with mock.patch.object(loop_module, 'RobotClient', Robot):
        loop.configure('fake', action_keys=Robot.action_keys, ttrtc_enabled=True, inference_hz=0)
    assert loop._tick_period() == 1/15
    loop.shutdown()


def service(request_overrides, monkeypatch):
    monkeypatch.setenv('POLICY_BACKEND', 'groot')
    requester = mock.Mock()
    requester.load_policy.return_value = SimpleNamespace(success=True, message='loaded',
                                                         action_keys=Robot.action_keys)
    loop = mock.Mock()
    handler = ServiceHandler(SessionState(), requester, loop,
                             lambda **kwargs: SimpleNamespace(**kwargs))
    fields = dict(command=0, model_path='/nonexistent/ttrtc', robot_type='fake',
                  task_instruction='pick', ttrtc_enabled=True, inference_hz=0)
    fields.update(request_overrides)
    return handler, requester, loop, SimpleNamespace(**fields)


def test_service_default_zero_frequency_means_trained_15hz(monkeypatch):
    handler, requester, loop, request = service({}, monkeypatch)
    assert handler.handle(request).success
    assert loop.configure.call_args.kwargs['ttrtc_enabled']
    assert loop.configure.call_args.kwargs['ttrtc_prefix_steps'] == 7
    assert loop.configure.call_args.kwargs['inference_hz'] == 15.


@pytest.mark.parametrize('overrides', [dict(action_request_mode='sync'),
                                      dict(vla_rail_enabled=True), dict(inference_hz=30),
                                      dict(ttrtc_prefix_steps=9), dict(action_chunk_size=24)])
def test_service_rejects_conflicts_before_model_load(monkeypatch, overrides):
    handler, requester, loop, request = service(overrides, monkeypatch)
    assert not handler.handle(request).success
    requester.load_policy.assert_not_called()
    loop.configure.assert_not_called()


def test_service_rejects_other_backend_before_model_load(monkeypatch):
    handler, requester, loop, request = service({}, monkeypatch)
    monkeypatch.setenv('POLICY_BACKEND', 'lerobot')
    assert not handler.handle(request).success
    requester.load_policy.assert_not_called()
