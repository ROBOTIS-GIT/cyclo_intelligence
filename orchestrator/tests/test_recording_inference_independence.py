"""Native message integration; no ROS initialization, sockets, or real writers."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from . import test_orchestrator_initial_pose_sync as native_fixture
from interfaces.msg import TaskInfo
from interfaces.srv import RecordingCommand, SendCommand
from orchestrator.internal.communication.cyclo_data_client import CallResult, CycloDataClient


@pytest.fixture
def recording_stack(tmp_path, monkeypatch):
    # The native overlay supplies interfaces, not rosbag_recorder. Only that
    # transport and the unused image/tensor converter are import-time fakes.
    with monkeypatch.context() as imports:
        for name, symbol in (
            ('cyclo_data.recorder.rosbag_control', 'RosbagControl'),
            ('cyclo_data.converter.orchestrator', 'DataConverter'),
        ):
            module = ModuleType(name)
            setattr(module, symbol, object)
            imports.setitem(sys.modules, name, module)
        service_module = importlib.import_module('cyclo_data.services.recording_service')
        manager_module = importlib.import_module('cyclo_data.recorder.session_manager')

    import rclpy
    monkeypatch.setattr(rclpy, 'init', Mock(side_effect=AssertionError('No ROS init')))
    monkeypatch.setattr(rclpy.node.Node, '__init__',
                        Mock(side_effect=AssertionError('No ROS nodes')))
    raw_root = tmp_path / 'recordings'
    monkeypatch.setattr(manager_module, 'RECORDING_ROOT', raw_root)
    monkeypatch.setattr(manager_module, 'CPUChecker', object)
    monkeypatch.setattr(manager_module, 'DataConverter', object)
    monkeypatch.setattr(service_module, 'CPUChecker', object)
    monkeypatch.setattr(service_module.RecordingService, 'DEFAULT_SAVE_ROOT_PATH',
                        tmp_path / 'datasets')

    bag = Mock()
    bag.active = False
    bag.is_available.return_value = True

    def start_bag(*, rosbag_uri):
        directory = Path(rosbag_uri)
        directory.mkdir(parents=True)
        (directory / 'data.mcap').write_bytes(b'fake bag payload')
        (directory / 'metadata.yaml').write_text(yaml.safe_dump({
            'rosbag2_bagfile_information': {
                'relative_file_paths': ['data.mcap'],
                'starting_time': {'nanoseconds_since_epoch': 1000000000},
                'duration': {'nanoseconds': 50000000},
                'message_count': 1,
                'topics_with_message_count': [],
            },
        }))
        bag.active = True

    bag.start_rosbag.side_effect = start_bag
    bag.stop_rosbag.side_effect = lambda: setattr(bag, 'active', False)
    monkeypatch.setattr(service_module, 'RosbagControl', lambda _node: bag)
    service = service_module.RecordingService(Mock(), Mock())
    monkeypatch.setattr(service, '_resolve_video_topics', lambda _robot: ({}, {}, {}))

    # Reuse the native orchestrator fixture: __new__, locks and fake inference
    # client only. None of its node publishers/services are constructed.
    owner = native_fixture.InitialPoseSyncOrchestratorTest(methodName='runTest')
    owner.setUp()
    node = owner.node
    node.params = {}
    node.communicator.get_mcap_topics = lambda: ['/joint_states', '/action']
    node._current_task_name = f'{node.robot_type}_demo'
    node._loaded_inference_instruction = 'pick original'
    node._loaded_inference_initial_pose_sync = False
    service._status_pub.publish.side_effect = node._data_operation_status_callback
    requests = []

    def recording_rpc(_service, request, timeout, label):
        requests.append(request)
        response = service._callback(request, RecordingCommand.Response())
        return CallResult(response.success, response.message, response)

    node._cyclo_data = CycloDataClient.__new__(CycloDataClient)
    node._cyclo_data._recording = Mock()
    node._cyclo_data._call = recording_rpc
    try:
        yield SimpleNamespace(node=node, client=owner.client, service=service,
                              bag=bag, requests=requests, raw_root=raw_root,
                              manager_class=manager_module.DataManager)
    finally:
        thread = service._finish_episode_thread
        if thread is not None:
            thread.join(3)
            assert not thread.is_alive(), 'Archive thread did not finish'
        owner.tearDown()
        service.shutdown()


def _ui(stack, command, task=None):
    response = stack.node.user_interaction_callback(
        SendCommand.Request(command=command, task_info=task or TaskInfo()),
        SendCommand.Response())
    assert response.success, response.message


def _prepare(stack):
    task = TaskInfo(task_type='record', task_num='42', task_name='demo',
                    task_instruction=['pick original'])
    stack.node._cache_ui_task_info(task, 'PREPARE_SESSION')
    stack.node._cache_ui_task_info(TaskInfo(task_type='inference',
        task_num='different', task_name='model', policy_id='lerobot:act',
        policy_path='/models/policy', task_instruction=['model instruction']), 'SET_TASK_INFO')
    return task


def _start(stack, channel, task):
    if channel == 'leader':
        stack.node.handle_joystick_trigger('right')
    else:
        command = (SendCommand.Request.START_INFERENCE_RECORD
                   if channel == 'legacy' else SendCommand.Request.START_SEGMENT)
        _ui(stack, command, task)
    assert stack.node.on_recording
    assert stack.bag.active
    assert isinstance(stack.service._data_manager, stack.manager_class)
    assert stack.service._data_manager.is_recording()


def _save(stack, channel):
    if channel == 'leader':
        stack.node.handle_joystick_trigger('right')
    else:
        command = (SendCommand.Request.STOP_INFERENCE_RECORD
                   if channel == 'legacy' else SendCommand.Request.STOP_SEGMENT)
        _ui(stack, command)
    assert not stack.node.on_recording
    assert not stack.bag.active
    if channel == 'ui':
        _ui(stack, SendCommand.Request.FINISH_EPISODE)
    thread = stack.service._finish_episode_thread
    if thread is not None:
        thread.join(3)
        assert not thread.is_alive()
    root = Path(stack.service._data_manager._save_rosbag_path)
    episode = root / '0'
    assert (episode / '0_0.mcap').read_bytes() == b'fake bag payload'
    assert not (episode / 'segments').exists()
    assert not stack.service._data_manager.is_recording()
    assert stack.node._recording_task_info is None
    return json.loads((episode / 'episode_info.json').read_text())


@pytest.mark.parametrize('command', [SendCommand.Request.STOP, SendCommand.Request.FINISH])
def test_flat_start_freezes_metadata_until_async_archive_finishes(recording_stack, monkeypatch, command):
    stack = recording_stack
    task = _prepare(stack)
    _ui(stack, SendCommand.Request.START_RECORD, task)
    manager = stack.service._data_manager
    finish = manager.finish_full_episode
    entered, release = Event(), Event()

    def blocked_finish(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return finish(*args, **kwargs)

    monkeypatch.setattr(manager, 'finish_full_episode', blocked_finish)
    try:
        _ui(stack, command)
        assert entered.wait(1)
        assert stack.node._recording_finishing
        assert stack.node._recording_task_info == task
        edited = deepcopy(task)
        edited.task_instruction = ['unrelated new draft']
        _ui(stack, SendCommand.Request.SET_TASK_INFO, edited)
        assert manager._main_task_instruction == 'pick original'
        assert stack.node._recording_task_info == task
    finally:
        release.set()
        worker = stack.service._finish_episode_thread
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive()
    assert stack.node._recording_task_info is None
    metadata = json.loads((Path(manager._save_rosbag_path) / '0' / 'episode_info.json').read_text())
    assert 'pick original' in json.dumps(metadata)
    assert 'unrelated new draft' not in json.dumps(metadata)


def test_flat_start_archive_failure_leader_retries_without_extra_start(recording_stack, monkeypatch):
    stack = recording_stack
    task = _prepare(stack)
    _ui(stack, SendCommand.Request.START_RECORD, task)
    manager = stack.service._data_manager
    finish = manager.finish_full_episode
    monkeypatch.setattr(manager, 'finish_full_episode', Mock(side_effect=OSError('archive failed')))
    stack.node.handle_joystick_trigger('right')
    worker = stack.service._finish_episode_thread
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
    assert stack.node._recording_task_info == task
    assert not stack.node._recording_finishing
    assert stack.node._trigger_record_next_segment_index == 1
    monkeypatch.setattr(manager, 'finish_full_episode', finish)
    stack.node.handle_joystick_trigger('right')
    worker = stack.service._finish_episode_thread
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
    assert stack.node._recording_task_info is None
    assert sum(request.command == RecordingCommand.Request.START for request in stack.requests) == 1
    assert all(request.command != RecordingCommand.Request.START_SEGMENT for request in stack.requests)


@pytest.mark.parametrize('start_channel,save_channel', [
    ('ui', 'ui'), ('leader', 'leader'), ('ui', 'leader'), ('leader', 'ui'),
    ('legacy', 'ui'), ('legacy', 'leader'), ('ui', 'legacy'), ('legacy', 'legacy'),
])
@pytest.mark.parametrize('runtime_state', ['running', 'paused', 'ready', 'unloaded', 'error', 'unknown'])
def test_ui_and_leader_share_storage_without_inference_commands(
        recording_stack, start_channel, save_channel, runtime_state):
    stack = recording_stack
    task = _prepare(stack)
    stack.node._inference_status_snapshot.update(runtime_state=runtime_state)
    stack.node.on_inference = runtime_state == 'running'
    _start(stack, start_channel, task)
    folder = Path(stack.service._data_manager._save_rosbag_path)
    assert folder.parent == stack.raw_root
    assert folder.name == 'Task_42_demo_MCAP'

    metadata = _save(stack, save_channel)

    assert metadata['task_instruction'] == 'pick original'
    assert metadata['robot_type'] == stack.node.robot_type
    assert metadata['episode_index'] == 0
    assert 'inference' not in metadata
    assert stack.client.calls == []
    assert stack.node.on_inference == (runtime_state == 'running')
    assert stack.bag.publish_action_event.call_args_list[0].args == ('start',)


@pytest.mark.parametrize('start_channel', ['ui', 'legacy'])
def test_inference_clear_and_settings_edit_preserve_active_recorder_and_metadata(
        recording_stack, start_channel):
    stack = recording_stack
    task = _prepare(stack)
    _start(stack, start_channel, task)
    manager = stack.service._data_manager
    original = deepcopy(manager._task_info)
    request_count = len(stack.requests)

    _ui(stack, SendCommand.Request.FINISH, TaskInfo(task_type='inference'))
    assert stack.client.disconnected.wait(3), 'Inference cleanup did not finish'
    stack.node._cache_ui_task_info(TaskInfo(task_type='inference',
        policy_id='lerobot:act', policy_path='/models/other',
        task_instruction=['edited instruction']), 'START_INFERENCE')

    assert stack.node.on_recording and stack.bag.active and manager.is_recording()
    assert stack.service._data_manager is manager
    assert manager._task_info == original
    assert len(stack.requests) == request_count
    stack.bag.stop_rosbag.assert_not_called()
    assert stack.client.calls == [stack.client.CMD_STOP, stack.client.CMD_UNLOAD]
    metadata = _save(stack, 'leader')
    assert metadata['task_instruction'] == 'pick original'
    assert 'inference' not in metadata


def test_leader_retries_failed_archive_without_starting_another_segment(recording_stack, monkeypatch):
    stack = recording_stack
    task = _prepare(stack)
    _start(stack, 'leader', task)
    manager = stack.service._data_manager
    finish = manager.finish_full_episode
    attempts = []

    def finish_after_failure():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError('Archive temporarily unavailable')
        return finish()

    monkeypatch.setattr(manager, 'finish_full_episode', finish_after_failure)
    stack.node.handle_joystick_trigger('right')
    thread = stack.service._finish_episode_thread
    if thread is not None:
        thread.join(3)
        assert not thread.is_alive()
    assert stack.node._recording_task_info == task
    assert stack.node._trigger_record_next_segment_index == 1
    assert not stack.node._recording_finishing
    starts = sum(r.command == RecordingCommand.Request.START_SEGMENT for r in stack.requests)
    stack.node.handle_joystick_trigger('right')
    thread = stack.service._finish_episode_thread
    if thread is not None:
        thread.join(3)
        assert not thread.is_alive()
    assert len(attempts) == 2
    assert sum(r.command == RecordingCommand.Request.START_SEGMENT for r in stack.requests) == starts
    assert stack.node._recording_task_info is None
