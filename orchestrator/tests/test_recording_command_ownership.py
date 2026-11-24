"""Recording command ownership and bounded executor ingress, without ROS I/O."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from . import test_orchestrator_initial_pose_sync as native_fixture
from interfaces.msg import DataOperationStatus, TaskInfo
from interfaces.srv import RecordingCommand, SendCommand, SetRobotType


@pytest.fixture
def node():
    owner = native_fixture.InitialPoseSyncOrchestratorTest(methodName='runTest')
    owner.setUp()
    owner.node.params = {}
    owner.node.communicator.get_mcap_topics = lambda: []
    try:
        yield owner.node
    finally:
        owner.tearDown()


def _task(number='42'):
    return TaskInfo(task_type='record', task_num=number, task_name='demo',
                    subtask_instruction=['first', 'second', 'third'])


def _ack():
    return SimpleNamespace(success=True, response=SimpleNamespace(success=True, message='ok'))


def _terminal(status=DataOperationStatus.COMPLETED):
    return DataOperationStatus(operation_type=DataOperationStatus.OP_RECORDING,
                               stage='FINISH_EPISODE', status=status)


def test_encoded_index_discard_rejects_different_owner(node):
    node.on_recording = True
    node._recording_task_info = _task('new')
    node._forward_recording = Mock(return_value=_ack())
    result = node.user_interaction_callback(SendCommand.Request(
        command=SendCommand.Request.DISCARD_EPISODE, segment_index=1,
        task_info=_task('old')), SendCommand.Response())
    assert not result.success
    assert 'target' in result.message.lower()
    node._forward_recording.assert_not_called()
    assert node.on_recording


def test_move_to_next_uses_owned_cursor_not_default_request_index(node):
    node.on_recording = True
    node._recording_task_info = _task()
    node._recording_start_command = RecordingCommand.Request.START_SEGMENT
    node._trigger_record_active_segment_index = 1
    node._forward_recording = Mock(return_value=_ack())
    node._command_recording(RecordingCommand.Request.MOVE_TO_NEXT, segment_index=0)
    assert node._trigger_record_next_segment_index == 2


@pytest.mark.parametrize('command', [RecordingCommand.Request.STOP, RecordingCommand.Request.FINISH])
def test_flat_start_retains_archive_owner_until_terminal_status(node, command):
    node.on_recording = True
    task = _task()
    node._recording_task_info = task
    node._recording_start_command = RecordingCommand.Request.START
    node._forward_recording = Mock(return_value=_ack())
    node._command_recording(command)
    assert node._recording_task_info == task
    assert node._recording_finishing
    node._data_operation_status_callback(_terminal(DataOperationStatus.FAILED))
    assert node._recording_task_info == task
    assert not node._recording_finishing
    node._command_recording(RecordingCommand.Request.FINISH_EPISODE)
    node._data_operation_status_callback(_terminal())
    assert node._recording_task_info is None


@pytest.mark.parametrize('command', [RecordingCommand.Request.START, RecordingCommand.Request.START_SEGMENT])
def test_start_waits_for_ack_without_abandoning_command_ownership(node, command):
    entered, release = Event(), Event()
    timeouts = []

    def send(**kwargs):
        timeouts.append(kwargs['timeout_sec'])
        entered.set()
        assert release.wait(2)
        return _ack()

    node._cyclo_data = SimpleNamespace(send_recording_command=send)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(node._command_recording, command, _task())
        try:
            assert entered.wait(1)
            assert not node._recording_command_lock.acquire(blocking=False)
            assert timeouts == [5.0]
            assert not node.on_recording
        finally:
            release.set()
        assert future.result(timeout=1).response.success
    assert node.on_recording
    assert node._recording_task_info == _task()


@pytest.mark.parametrize('ingress', ['record', 'inference', 'leader', 'robot', 'status'])
def test_busy_ingress_does_not_occupy_executor_thread(node, ingress):
    entered, release = Event(), Event()
    lock = (node._inference_command_lock if ingress == 'inference'
            else node._recording_command_lock)
    node._recording_task_info = _task()
    node._recording_finishing = True

    def hold():
        with lock:
            entered.set()
            assert release.wait(2)

    def invoke():
        if ingress == 'leader':
            return node.handle_joystick_trigger('right')
        if ingress == 'robot':
            return node.set_robot_type_callback(
                SetRobotType.Request(robot_type='different'), SetRobotType.Response())
        if ingress == 'status':
            return node._data_operation_status_callback(_terminal())
        command = (SendCommand.Request.GET_INFERENCE_STATUS if ingress == 'inference'
                   else SendCommand.Request.FINISH_EPISODE)
        return node.user_interaction_callback(
            SendCommand.Request(command=command), SendCommand.Response())

    with ThreadPoolExecutor(max_workers=2) as pool:
        held = pool.submit(hold)
        assert entered.wait(1)
        pending = pool.submit(invoke)
        try:
            result = pending.result(timeout=0.5)
            if ingress in ('record', 'inference', 'robot'):
                assert not result.success
                assert 'progress' in result.message.lower() or 'busy' in result.message.lower()
        finally:
            release.set()
            held.result(timeout=1)
            pending.result(timeout=1)


def test_completion_before_archive_ack_is_drained_after_ack(node):
    node._recording_task_info = _task()
    node._recording_start_command = RecordingCommand.Request.START_SEGMENT

    def forward(*args, **kwargs):
        node._data_operation_status_callback(_terminal())
        assert node._recording_task_info is not None
        return _ack()

    node._forward_recording = forward
    node._command_recording(RecordingCommand.Request.FINISH_EPISODE)
    assert node._recording_task_info is None
    assert not node._recording_finishing


def test_failed_completion_before_ack_preserves_owner_for_retry(node):
    node._recording_task_info = _task()
    node._recording_start_command = RecordingCommand.Request.START

    def forward(*args, **kwargs):
        node._data_operation_status_callback(_terminal(DataOperationStatus.FAILED))
        return _ack()

    node._forward_recording = forward
    node._command_recording(RecordingCommand.Request.STOP)
    assert node._recording_task_info == _task()
    assert node._trigger_record_next_segment_index == 3
    assert not node._recording_finishing


def test_terminal_status_queue_is_bounded_and_drains_after_command(node):
    node._recording_task_info = _task()
    node._recording_finishing = True
    with node._command_scope(node._recording_command_lock):
        for _ in range(100):
            node._data_operation_status_callback(_terminal())
        assert node._recording_status_events.qsize() == 32
        assert node._recording_task_info is not None
    assert node._recording_task_info is None
    assert node._recording_status_events.empty()


def test_encoded_index_discard_keeps_same_owner_target(node):
    node._recording_task_info = _task()
    node._forward_recording = Mock(return_value=_ack())
    result = node.user_interaction_callback(SendCommand.Request(
        command=SendCommand.Request.DISCARD_EPISODE, segment_index=8,
        task_info=_task()), SendCommand.Response())
    assert result.success
    assert node._forward_recording.call_args.kwargs['segment_index'] == 8


def test_old_inference_flat_stop_does_not_wait_for_nonexistent_archive(node):
    node.on_recording = True
    node._recording_task_info = TaskInfo(task_type='inference', task_name='legacy')
    node._recording_start_command = RecordingCommand.Request.START
    node._forward_recording = Mock(return_value=_ack())
    node._command_recording(RecordingCommand.Request.STOP)
    assert not node.on_recording
    assert node._recording_task_info is None
    assert not node._recording_finishing


def test_executor_reserves_capacity_for_independent_rpc_responses():
    from orchestrator import orchestrator_node as module
    fake = Mock()
    with patch.object(module, 'rclpy'), patch.object(module, 'OrchestratorNode', return_value=fake), \
            patch.object(module, 'MultiThreadedExecutor') as executor:
        module.main()
    assert executor.call_args.kwargs['num_threads'] >= 4


@pytest.mark.parametrize('repeat', range(20))
def test_conversion_does_not_block_record_or_inference_commands(node, repeat):
    entered, release = Event(), Event()

    def dispatch(request, response):
        if request.command == SendCommand.Request.CONVERT_MP4:
            entered.set()
            assert release.wait(2)
        response.success = True
        return response

    node._dispatch_user_interaction = dispatch

    def call(command):
        return node.user_interaction_callback(
            SendCommand.Request(command=command), SendCommand.Response())

    with ThreadPoolExecutor(max_workers=2) as pool:
        converting = pool.submit(call, SendCommand.Request.CONVERT_MP4)
        try:
            assert entered.wait(1)
            for command in (SendCommand.Request.STOP_SEGMENT, SendCommand.Request.STOP_INFERENCE):
                assert pool.submit(call, command).result(timeout=.5).success
            assert not pool.submit(call, SendCommand.Request.CONVERT_MP4).result(timeout=.5).success
        finally:
            release.set()
        assert converting.result(timeout=1).success


@pytest.mark.parametrize('command', [RecordingCommand.Request.START_SEGMENT, RecordingCommand.Request.STOP_SEGMENT,
                                   RecordingCommand.Request.FINISH_EPISODE, RecordingCommand.Request.DISCARD_EPISODE])
@pytest.mark.parametrize('success', [True, False])
def test_lost_ack_reconciles_once_without_replaying_current_click(node, command, success):
    from orchestrator.internal.communication.cyclo_data_client import CallResult
    task = _task()
    node._trigger_record_next_segment_index = 0
    if command != RecordingCommand.Request.START_SEGMENT:
        node._recording_task_info = task
        node._recording_start_command = RecordingCommand.Request.START_SEGMENT
    unknown = CallResult(False, 'unconfirmed', request_id='receipt', outcome_unknown=True)
    node._cyclo_data = SimpleNamespace(send_recording_command=Mock(return_value=unknown),
                                      query_recording_result=Mock(return_value=unknown))
    result = node._command_recording(command, task)
    assert result.outcome_unknown
    assert node._pending_recording_command is not None
    with pytest.raises(ValueError, match='unconfirmed'):
        node._command_recording(command, task)
    node._cyclo_data.send_recording_command.assert_called_once()
    ack = CallResult(success, 'done', SimpleNamespace(success=success, message='done', archive_state=''),
                     request_id='receipt')
    node._cyclo_data.query_recording_result.return_value = ack
    with pytest.raises(ValueError, match='reconciled'):
        node._command_recording(command, task)
    assert node._pending_recording_command is None
    node._cyclo_data.send_recording_command.assert_called_once()
    if command == RecordingCommand.Request.START_SEGMENT:
        assert node.on_recording is success
    elif command == RecordingCommand.Request.STOP_SEGMENT:
        assert node._trigger_record_next_segment_index == (1 if success else 0)


def test_archive_completion_event_lost_is_recovered_by_receipt(node):
    from orchestrator.internal.communication.cyclo_data_client import CallResult
    node._recording_task_info = _task()
    node._recording_finishing = True
    node._recording_archive_request_id = 'saved'
    node._cyclo_data = SimpleNamespace(query_recording_result=Mock(return_value=CallResult(
        True, 'done', SimpleNamespace(archive_state='completed'), request_id='saved')))
    with node._command_scope(node._recording_command_lock):
        assert node._recording_task_info is None
        assert not node._recording_finishing


def test_old_archive_event_cannot_release_a_new_archive(node):
    node._recording_task_info = _task()
    node._recording_finishing = True
    node._recording_archive_request_id = 'current'
    msg = _terminal()
    msg.job_id = 'previous'
    node._data_operation_status_callback(msg)
    assert node._recording_finishing
    assert node._recording_task_info is not None
