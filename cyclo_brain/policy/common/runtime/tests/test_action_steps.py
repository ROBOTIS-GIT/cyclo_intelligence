"""Runtime limits plans without mutating model configs or calling model APIs."""

from types import SimpleNamespace
from unittest import mock
import threading

import numpy as np
import pytest

from .test_control_loop import ControlLoop, FakeRobot, FakeRequester, control_loop_module
from inference_context.contract import ExecutionContract
from inference_context.execution import ExecutionContext


def response():
    return SimpleNamespace(success=True, chunk_size=6, action_dim=1, action_list=list(range(6)))


def configured(**config):
    loop = ControlLoop(FakeRequester(response()), postprocess_actions=False)
    with mock.patch.object(control_loop_module, 'RobotClient', return_value=FakeRobot()):
        loop.configure('test', **config)
    return loop


@pytest.mark.parametrize('mode', ['sync', 'async'])
def test_limit_applies_to_both_request_modes_and_preview_is_not_publication(mode):
    loop = configured(action_steps=2)
    try:
        loop.start()
        loop._request_and_buffer('', loop._generation, mode)
        assert loop.observed_chunk_size() == 6
        assert loop._processor.buffer_size == 2
        with mock.patch.object(loop, '_should_request_actions', return_value=False):
            loop.tick()
            loop.tick()
        assert len(loop._robot.previews) == 2
        assert not loop._robot.commands
        assert loop._processor.buffer_size == 0
    finally:
        loop.deconfigure()


def test_running_change_is_rejected_and_pause_change_keeps_worker_loaded():
    loop = configured(action_steps=2)
    try:
        loop.start()
        loop._request_and_buffer('', loop._generation, 'sync')
        with pytest.raises(ValueError, match='Pause or Stop'):
            loop.start(action_steps=4)
        assert loop._processor.buffer_size == 2
        assert loop.configuration_snapshot()['action_steps'] == 2
        loop.pause()
        loop.start(action_steps=4)
        loop._request_and_buffer('', loop._generation, 'sync')
        assert loop._processor.buffer_size == 4
        loop.pause()
        loop.start(action_steps=0)
        loop._request_and_buffer('', loop._generation, 'sync')
        assert loop._processor.buffer_size == 6
    finally:
        loop.deconfigure()


def test_late_response_cannot_enter_the_new_plan():
    loop = configured(action_steps=2)
    entered, release = threading.Event(), threading.Event()
    def predict(*args):
        entered.set()
        assert release.wait(3)
        return response()
    loop._requester.get_action = predict
    loop.start()
    thread = threading.Thread(target=loop._request_and_buffer, args=('', loop._generation, 'async'))
    thread.start()
    try:
        assert entered.wait(1)
        loop.pause()
        loop.start(action_steps=3)
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert loop._processor.buffer_size == 0
        assert loop.observed_chunk_size() == 0
    finally:
        release.set()
        thread.join(3)
        loop.deconfigure()


def test_model_owned_queue_rejects_limit_before_creating_robot_client():
    loop = ControlLoop(object())
    with mock.patch.object(control_loop_module, 'RobotClient') as robot:
        with pytest.raises(ValueError, match='owns its action queue'):
            loop.configure('test', action_steps=2, publish_to_robot=True,
                           execution_context=ExecutionContext('s', 0, 0, 'ready'),
                           execution_contract=ExecutionContract('step'))
        robot.assert_not_called()


def test_step_policy_all_remains_supported_but_resume_limit_is_rejected():
    loop = configured(publish_to_robot=True,
                      execution_context=ExecutionContext('s', 0, 0, 'ready'),
                      execution_contract=ExecutionContract('step'))
    try:
        with pytest.raises(ValueError, match='owns its action queue'):
            loop.start(action_steps=1)
        assert not loop._running
        loop.start(action_steps=0)
        assert loop._running
    finally:
        loop.deconfigure()


def test_preview_switch_revalidates_publication_prerequisite():
    loop = configured(publish_to_robot=True,
                      execution_context=ExecutionContext('s', 0, 0, 'ready', feedback_schema=2),
                      execution_contract=ExecutionContract(feedback_schema=2, request_after='plan_terminal'))
    try:
        with pytest.raises(ValueError, match='requires command receipts'):
            loop.start(publish_to_robot=False, action_steps=3)
        assert loop.configuration_snapshot()['action_steps'] == 0
        assert loop._publish_to_robot
        assert not loop._running
    finally:
        loop.deconfigure()


def test_slow_start_target_is_unchanged_by_action_limit():
    loop = configured(action_steps=2, initial_pose_sync=True, publish_to_robot=True)
    try:
        assert loop.start()
        assert len(loop._robot.sync_targets) == 1
        assert loop.observed_chunk_size() == 6
        np.testing.assert_array_equal(loop._robot.sync_targets[0][0], [0.])
        assert loop._processor.buffer_size == 0
        with pytest.raises(ValueError, match='Pause or Stop'):
            loop.start(action_steps=3)
        assert loop.configuration_snapshot()['action_steps'] == 2
        assert len(loop._robot.sync_targets) == 1
    finally:
        loop.deconfigure()


def test_observed_length_tracks_variable_chunks_and_resets_on_unload():
    loop = configured(action_steps=2)
    try:
        assert loop.observed_chunk_size() == 0
        loop.start()
        for length in (6, 3, 9):
            loop._requester = FakeRequester(SimpleNamespace(
                success=True, chunk_size=length, action_dim=1, action_list=list(range(length))))
            loop._request_and_buffer('', loop._generation, 'sync')
            assert loop.observed_chunk_size() == length
            assert loop.configuration_snapshot()['action_steps'] == 2
        loop.pause()
        assert loop.observed_chunk_size() == 9
        loop.start(action_steps=1)
        assert loop.observed_chunk_size() == 9
    finally:
        loop.deconfigure()
    assert loop.observed_chunk_size() == 0


def test_invalid_prediction_does_not_supply_a_chunk_measurement():
    loop = configured()
    try:
        loop._requester = FakeRequester(SimpleNamespace(
            success=True, chunk_size=15, action_dim=1, action_list=[0.]))
        loop.start()
        loop._request_and_buffer('', loop._generation)
        assert loop.observed_chunk_size() == 0
    finally:
        loop.deconfigure()


def test_model_owned_queue_does_not_report_one_step_as_its_horizon():
    loop = configured(publish_to_robot=True, initial_pose_sync=True,
                      execution_context=ExecutionContext('s', 0, 0, 'ready'),
                      execution_contract=ExecutionContract('step'))
    try:
        result = SimpleNamespace(success=True, chunk_size=1, action_dim=1, action_list=[0.])
        with mock.patch.object(loop, '_get_action_with_feedback', return_value=result):
            loop.start()
        assert loop.observed_chunk_size() == 0
    finally:
        loop.deconfigure()


@pytest.mark.parametrize('failure', ['invalid_limit', 'running_limit', 'hold_pending', 'preview'])
def test_rejected_resume_keeps_instruction_generation_and_pending_commands(failure):
    loop = configured(action_steps=2, publish_to_robot=True,
                      execution_context=ExecutionContext('s', 0, 0, 'ready', feedback_schema=2),
                      execution_contract=ExecutionContract(feedback_schema=2, request_after='plan_terminal'))
    try:
        loop.start(task_instruction='pick')
        loop._processor.enqueue(0, np.arange(6.).reshape(-1, 1), action_steps=2)
        before = loop._processor.snapshot()
        generation = loop._generation
        if failure == 'hold_pending':
            loop._initial_pose_sync_hold_pending = True
        options = dict(task_instruction='place')
        if failure == 'invalid_limit':
            options['action_steps'] = -2
        elif failure == 'running_limit':
            options['action_steps'] = 3
        elif failure == 'preview':
            options['publish_to_robot'] = False
        with pytest.raises((ValueError, RuntimeError)):
            loop.start(**options)
        assert loop._task_instruction == 'pick'
        assert loop._generation == generation
        assert loop._processor.snapshot() == before
        assert loop.configuration_snapshot()['action_steps'] == 2
        assert loop._publish_to_robot
    finally:
        loop.deconfigure()
