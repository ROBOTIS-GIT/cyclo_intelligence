#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


_MODULE_BACKUPS = {}


def _install_module_stub(name: str, **attributes) -> None:
    _MODULE_BACKUPS[name] = sys.modules.get(name)
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)


class _Stub:
    pass


_install_module_stub("cyclo_data.recorder.session_manager", DataManager=_Stub)
_install_module_stub("cyclo_data.hub.endpoint_store", HFEndpointStore=_Stub)
_install_module_stub("cyclo_data.recorder.replay_handler", ReplayDataHandler=_Stub)
_install_module_stub(
    "cyclo_data.visualization.video_file_server",
    VideoFileServer=_Stub,
)

from interfaces.msg import InferenceStatus, TaskInfo  # noqa: E402
from interfaces.srv import SendCommand, RecordingCommand  # noqa: E402
from orchestrator.orchestrator_node import OrchestratorNode  # noqa: E402
from orchestrator.internal.inference_recording import InferenceRecordingSession
from orchestrator.internal.communication.container_service_client import ContainerServiceClient  # noqa: E402

for _module_name, _previous_module in _MODULE_BACKUPS.items():
    if _previous_module is None:
        sys.modules.pop(_module_name, None)
    else:
        sys.modules[_module_name] = _previous_module


class FakeCommunicator:
    def __init__(self) -> None:
        self.phases = []
        self.inferencing = threading.Event()
        self.snapshots = []
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg)
        fields = {key: getattr(msg, key) for key in msg.get_fields_and_field_types()}
        fields['phase'] = fields.pop('inference_phase')
        self.publish_inference_status(**fields)

    def publish_inference_status(self, *, phase, robot_type, error, **snapshot) -> None:
        self.phases.append((phase, robot_type, error))
        self.snapshots.append(dict(phase=phase, error=error, **snapshot))
        if phase == InferenceStatus.INFERENCING:
            self.inferencing.set()


class FakeLogger:
    def info(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass


class FakeInferenceClient:
    def __init__(self) -> None:
        self.calls = []
        self.pause_results = []
        self.stop_results = []
        self.unload_results = []
        self.status_results = []
        self.disconnected = threading.Event()
        self._cancelled = threading.Event()

    def inference_command(self, command, **_kwargs):
        self.calls.append(command)
        if command == self.CMD_PAUSE and self.pause_results:
            return self.pause_results.pop(0)
        if command == self.CMD_STOP and self.stop_results:
            return self.stop_results.pop(0)
        if command == self.CMD_UNLOAD and self.unload_results:
            return self.unload_results.pop(0)
        if command == self.CMD_STATUS and self.status_results:
            return self.status_results.pop(0)
        return SimpleNamespace(success=True, message="ok")

    def disconnect(self) -> None:
        self.disconnected.set()

    CMD_PAUSE = 2
    CMD_STOP = 4
    CMD_UNLOAD = 5
    CMD_STATUS = 7


class InitialPoseSyncOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeInferenceClient()
        self.node = OrchestratorNode.__new__(OrchestratorNode)
        self.node._state_lock = threading.RLock()
        self.node._recording_command_lock = threading.RLock()
        self.node._inference_recording = InferenceRecordingSession()
        self.node._inference_record_clear_pending = False
        self.node._loaded_inference_instruction = 'pick'
        self.node._inference_lifecycle_lock = threading.Lock()
        self.node.container_service_client = self.client
        self.node._initial_pose_sync_status_active = False
        self.node._init_inference_status_monitor()
        self.node._inference_status_client = self.client
        self.node._initial_pose_sync_hold_pending = False
        self.node.communicator = FakeCommunicator()
        self.node._inference_status_publisher = self.node.communicator
        self.node.robot_type = "ffw_sg2_rev1"
        self.node.get_logger = lambda: FakeLogger()
        self.node._loaded_inference_policy_path = "/models/policy"
        self.node._loaded_inference_policy_id = "lerobot:act"
        self.node._loaded_inference_policy_parameters_json = "{}"
        self.node._loaded_inference_publish_to_robot = True
        self.node._loaded_inference_acceleration_mode = "pytorch"
        self.node._loaded_inference_acceleration_engine_path = ""
        self.node._loaded_inference_action_request_mode = "async"
        self.node._loaded_inference_control_hz = 100
        self.node._loaded_inference_inference_hz = 15
        self.node._loaded_inference_chunk_align_window_s = 0.3
        self.node._loaded_inference_initial_pose_sync = True
        self.node._loaded_inference_initial_pose_sync_duration_s = 5.0
        self.node.on_recording = False
        self.node.on_inference = True

    def tearDown(self) -> None:
        self.node._stop_inference_status_monitor()
        self.node._cancel_initial_pose_sync_status()

    def test_recording_snapshot_reuse_and_failed_save(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            node = self.node
            node._inference_recording = InferenceRecordingSession(Path(root))
            task = TaskInfo(task_type='inference', policy_id='lerobot:act',
                            policy_path='/models/policy', task_instruction=['unsent edit'])
            node._cache_ui_task_info(task, 'SET_TASK_INFO')
            node._inference_status_snapshot.update(
                status_known=True, runtime_state='running', publish_to_robot=True)
            ok = SimpleNamespace(success=True, response=SimpleNamespace(success=True))
            bad = SimpleNamespace(success=True, response=SimpleNamespace(success=False))
            node._forward_recording = Mock(return_value=ok)
            node._command_inference_recording(RecordingCommand.Request.START)
            session_id = node._inference_recording.session_id
            recorded = node._forward_recording.call_args.kwargs['task_info']
            self.assertEqual(recorded.task_instruction, ['pick'])
            self.assertEqual(recorded.task_num, session_id)
            self.assertEqual(task.task_num, '')
            with self.assertRaisesRegex(ValueError, 'no active recording'):
                node._command_inference_recording(RecordingCommand.Request.START)
            with self.assertRaises(ValueError):
                node._select_inference_record_folder('')
            node._forward_recording.return_value = bad
            node._command_inference_recording(RecordingCommand.Request.STOP)
            self.assertTrue(node.on_recording)
            with self.assertRaisesRegex(ValueError, 'Save or Discard'):
                node._select_inference_record_folder('')
            self.assertEqual(node._inference_recording.session_id, session_id)
            node._forward_recording.return_value = ok
            node._command_inference_recording(RecordingCommand.Request.STOP)
            self.assertFalse(node.on_recording)
            # Old UI task_num values cannot replace the destination.
            task.task_num = 'stale-ui-value'
            node._cache_ui_task_info(task, 'SET_TASK_INFO')
            self.assertEqual(node._inference_recording.session_id, session_id)
            node._command_inference_recording(RecordingCommand.Request.START)
            self.assertEqual(node._forward_recording.call_args.kwargs['task_info'].task_num, session_id)
            node._command_inference_recording(RecordingCommand.Request.CANCEL)
            node._select_inference_record_folder('')
            self.assertEqual(node.communicator.messages[-1].task_info.task_num, '')
            self.assertEqual(node._inference_settings_task_info.task_num, 'stale-ui-value')
            self.assertEqual(len(list(Path(root).iterdir())), 1)
            node._command_inference_recording(RecordingCommand.Request.START)
            self.assertNotEqual(node._inference_recording.session_id, session_id)
            self.assertEqual(len(list(Path(root).iterdir())), 2)

    def test_recording_blocks_clear_but_not_pause(self):
        self.node.on_recording = True
        self.node._forward_recording = Mock()
        self.node._teardown_inference_client = Mock()
        response = self.node.user_interaction_callback(
            SendCommand.Request(command=SendCommand.Request.FINISH,
                                task_info=TaskInfo(task_type='inference')),
            SendCommand.Response())
        self.assertFalse(response.success)
        self.assertIn('Save or Discard', response.message)
        self.node._forward_recording.assert_not_called()
        self.node._teardown_inference_client.assert_not_called()
        response = self.node.user_interaction_callback(
            SendCommand.Request(command=SendCommand.Request.STOP_INFERENCE,
                                task_info=TaskInfo(task_type='inference')),
            SendCommand.Response())
        self.assertTrue(response.success, response.message)
        self.assertTrue(self.node.on_recording)

    def test_folder_command_allows_running_without_changing_inference(self):
        self.node._inference_status_snapshot.update(status_known=True, runtime_state='running')
        self.node._inference_recording.session_id = 'previous'
        self.node._forward_recording = Mock()
        self.node._teardown_inference_client = Mock()
        request = SendCommand.Request(command=SendCommand.Request.SET_INFERENCE_RECORD_FOLDER,
                                      task_info=TaskInfo(task_num=''))
        result = self.node.user_interaction_callback(request, SendCommand.Response())
        self.assertTrue(result.success, result.message)
        self.assertEqual(self.node._inference_recording.session_id, '')
        self.assertEqual(self.node.communicator.messages[-1].task_info.task_num, '')
        self.assertEqual(self.node._inference_status_snapshot['runtime_state'], 'running')
        self.assertTrue(self.node.on_inference)
        self.node._forward_recording.assert_not_called()
        self.node._teardown_inference_client.assert_not_called()

    def test_folder_command_rejects_active_recording_without_changing_selection(self):
        self.node._inference_status_snapshot.update(status_known=True, runtime_state='running')
        self.node._inference_recording.session_id = 'previous'
        self.node.on_recording = True
        request = SendCommand.Request(command=SendCommand.Request.SET_INFERENCE_RECORD_FOLDER,
                                      task_info=TaskInfo(task_num=''))
        result = self.node.user_interaction_callback(request, SendCommand.Response())
        self.assertFalse(result.success)
        self.assertIn('Save or Discard', result.message)
        self.assertEqual(self.node._inference_recording.session_id, 'previous')

    def test_trigger_uses_same_recording_command_path(self):
        result = SimpleNamespace(success=True, response=SimpleNamespace(success=True))
        self.node._command_inference_recording = Mock(return_value=result)
        self.node._toggle_inference_trigger_recording(False)
        self.node._toggle_inference_trigger_recording(True)
        self.node._cancel_inference_trigger_recording()
        self.assertEqual([call.args[0] for call in self.node._command_inference_recording.call_args_list],
                         [RecordingCommand.Request.START, RecordingCommand.Request.STOP,
                          RecordingCommand.Request.CANCEL])

    def test_simultaneous_record_requests_start_only_one_episode(self):
        from concurrent.futures import ThreadPoolExecutor
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            node = self.node
            node._inference_recording = InferenceRecordingSession(Path(root))
            node._cache_ui_task_info(TaskInfo(task_type='inference', policy_id='lerobot:act',
                                            policy_path='/models/policy'), 'SET_TASK_INFO')
            node._inference_status_snapshot.update(status_known=True, runtime_state='running', publish_to_robot=True)
            node._forward_recording = Mock(return_value=SimpleNamespace(
                success=True, response=SimpleNamespace(success=True)))
            barrier = threading.Barrier(2)
            def record():
                barrier.wait()
                try:
                    node._command_inference_recording(RecordingCommand.Request.START)
                    return True
                except ValueError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                attempts = [pool.submit(record) for _ in range(2)]
                self.assertEqual(sorted(result.result() for result in attempts), [False, True])
            node._forward_recording.assert_called_once()

    def prepare_async_load(self, client):
        self.node.container_service_client = None
        self.node._client_cb_group = object()
        self.node.init_robot_control_parameters_from_user_task = Mock()
        with patch('orchestrator.orchestrator_node.ContainerServiceClient') as factory, \
                patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            factory.return_value = client
            for name in ('CMD_LOAD', 'CMD_START', 'CMD_STOP', 'CMD_UNLOAD'):
                setattr(factory, name, getattr(ContainerServiceClient, name))
            response = self.node.user_interaction_callback(
                SendCommand.Request(command=SendCommand.Request.START_INFERENCE,
                                    task_info=TaskInfo(task_type='inference', policy_path='/models/new',
                                                       policy_id='lerobot:act', control_hz=80,
                                                       inference_hz=20, chunk_align_window_s=0.25)),
                SendCommand.Response(),
            )
        self.assertTrue(response.success, response.message)
        return thread.call_args.kwargs['target']

    def resume_request(self, command):
        self.client._service_prefix = '/lerobot'
        return self.node.user_interaction_callback(
            SendCommand.Request(command=command, task_info=TaskInfo(
                task_type='inference', policy_path='/models/policy', policy_id='lerobot:act',
                control_hz=100, inference_hz=15, chunk_align_window_s=0.3,
                initial_pose_sync=True, initial_pose_sync_duration_s=5.0,
                task_instruction=['pick'],
            )), SendCommand.Response())

    def test_resume_does_not_send_commands_during_reserved_cleanup(self):
        for command in (SendCommand.Request.START_INFERENCE, SendCommand.Request.RESUME_INFERENCE):
            with self.subTest(command=command):
                self.node._inference_cleanup_client = self.client
                self.client.calls.clear()
                result = self.resume_request(command)
                self.assertFalse(result.success)
                self.assertFalse(self.client.calls)

    def test_resume_reply_cannot_activate_replacement_session(self):
        for command in (SendCommand.Request.START_INFERENCE, SendCommand.Request.RESUME_INFERENCE):
            for message in ('ok', 'syncing', 'LOAD first'):
                with self.subTest(command=command, message=message):
                    replacement = FakeInferenceClient()
                    self.node.container_service_client = self.client
                    self.node.on_inference = True
                    self.node._initial_pose_sync_status_active = False
                    def replace_session(*args, **kwargs):
                        self.node.container_service_client = replacement
                        self.node.on_inference = False
                        return SimpleNamespace(success=message != 'LOAD first', message=message)
                    with patch.object(self.client, 'inference_command', side_effect=replace_session), \
                            patch.object(self.node, '_teardown_inference_client') as teardown, \
                            patch.object(self.node, '_publish_inference_phase') as publish:
                        result = self.resume_request(command)
                    self.assertFalse(result.success)
                    self.assertFalse(self.node.on_inference)
                    self.assertIs(self.node.container_service_client, replacement)
                    self.assertFalse(self.node._initial_pose_sync_status_active)
                    publish.assert_not_called()
                    teardown.assert_not_called()

    def test_resume_rejects_busy_lifecycle_without_waiting_or_rpc(self):
        self.node._inference_lifecycle_lock.acquire()
        try:
            for command in (SendCommand.Request.START_INFERENCE, SendCommand.Request.RESUME_INFERENCE):
                result = self.resume_request(command)
                self.assertFalse(result.success)
                self.assertIn('lifecycle operation in progress', result.message)
                self.assertFalse(self.client.calls)
        finally:
            self.node._inference_lifecycle_lock.release()

    def test_resume_transport_does_not_hold_state_lock_and_respects_new_cleanup(self):
        reserved = threading.Event()
        def reserve_cleanup():
            with self.node._state_lock:
                self.node._inference_cleanup_client = self.client
            reserved.set()

        def reply(*args, **kwargs):
            self.assertTrue(self.node._inference_lifecycle_lock.locked())
            thread = threading.Thread(target=reserve_cleanup)
            thread.start()
            try:
                self.assertTrue(reserved.wait(1.0), 'state lock blocked cleanup during RESUME RPC')
            finally:
                thread.join(1.0)
            return SimpleNamespace(success=True, message='ok')

        with patch.object(self.client, 'inference_command', side_effect=reply), \
                patch.object(self.node, '_publish_inference_phase') as publish:
            response = self.resume_request(SendCommand.Request.RESUME_INFERENCE)
        self.assertFalse(response.success)
        publish.assert_not_called()
        self.assertFalse(self.node._inference_lifecycle_lock.locked())

    def test_resume_shared_path_preserves_running_syncing_preparing_and_failure(self):
        for command in (SendCommand.Request.START_INFERENCE, SendCommand.Request.RESUME_INFERENCE):
            for phase in ('running', 'syncing', 'preparing', 'failure', 'exception'):
                with self.subTest(command=command, phase=phase):
                    result = SimpleNamespace(success=phase != 'failure', message=phase,
                                             data={'runtime_state': phase})
                    effect = RuntimeError('transport unavailable') if phase == 'exception' else None
                    with patch.object(self.client, 'inference_command', return_value=result, side_effect=effect) as rpc, \
                            patch.object(self.node, '_publish_inference_phase') as publish, \
                            patch.object(self.node, '_begin_initial_pose_sync_status') as sync, \
                            patch.object(self.node, '_apply_inference_runtime_status') as preparing:
                        response = self.resume_request(command)
                    self.assertEqual(response.success, phase not in ('failure', 'exception'))
                    self.assertEqual(rpc.call_count, 1)
                    self.assertEqual(rpc.call_args.args[0], ContainerServiceClient.CMD_RESUME)
                    self.assertEqual(rpc.call_args.kwargs['task_instruction'], 'pick')
                    self.assertFalse(self.node._inference_lifecycle_lock.locked())
                    self.assertEqual(publish.call_count, int(phase == 'running'))
                    self.assertEqual(sync.call_count, int(phase == 'syncing'))
                    self.assertEqual(preparing.call_count, int(phase == 'preparing'))

    def test_superseded_async_load_or_start_cannot_publish_or_retry_for_new_session(self):
        for stage in (ContainerServiceClient.CMD_LOAD, ContainerServiceClient.CMD_START):
            for outcome in ('failure', 'already_loaded', 'exception', 'success'):
                with self.subTest(stage=stage, outcome=outcome):
                    client = Mock()
                    load = self.prepare_async_load(client)
                    replacement = FakeInferenceClient()

                    def complete(command, **kwargs):
                        if command == stage:
                            self.node.container_service_client = replacement
                            self.node._loaded_inference_policy_path = '/models/replacement'
                            if outcome == 'exception':
                                raise RuntimeError('old request failed')
                            return SimpleNamespace(success=outcome == 'success', data={}, message=(
                                'already loaded; UNLOAD first' if outcome == 'already_loaded' else 'old response'))
                        return SimpleNamespace(success=True, message='ok', data={})

                    client.inference_command.side_effect = complete
                    before = list(self.node.communicator.phases)
                    load()
                    expected = [ContainerServiceClient.CMD_LOAD]
                    if stage == ContainerServiceClient.CMD_START:
                        expected.append(ContainerServiceClient.CMD_START)
                    self.assertEqual([c.args[0] for c in client.inference_command.call_args_list], expected)
                    self.assertEqual(self.node.communicator.phases, before)
                    self.assertIs(self.node.container_service_client, replacement)
                    self.assertEqual(replacement.calls, [])

    def test_clear_reserved_during_load_prevents_start(self):
        client = Mock()
        load = self.prepare_async_load(client)

        def complete(command, **kwargs):
            self.node._inference_cleanup_client = client
            return SimpleNamespace(success=True, message='ok', data={})

        client.inference_command.side_effect = complete
        load()
        self.assertEqual([c.args[0] for c in client.inference_command.call_args_list],
                         [ContainerServiceClient.CMD_LOAD])

    def test_load_retry_requires_stop_and_unload_success(self):
        for failed_command in (ContainerServiceClient.CMD_STOP, ContainerServiceClient.CMD_UNLOAD):
            with self.subTest(failed_command=failed_command):
                client = Mock()
                load = self.prepare_async_load(client)
                calls = []

                def complete(command, **kwargs):
                    calls.append(command)
                    if command == ContainerServiceClient.CMD_LOAD:
                        return SimpleNamespace(success=False, message='already loaded; UNLOAD first')
                    return SimpleNamespace(success=command != failed_command, message='hold/prediction pending')

                client.inference_command.side_effect = complete
                with patch.object(self.node, '_teardown_inference_client'):
                    load()
                expected = [ContainerServiceClient.CMD_LOAD, ContainerServiceClient.CMD_STOP]
                if failed_command == ContainerServiceClient.CMD_UNLOAD:
                    expected.append(ContainerServiceClient.CMD_UNLOAD)
                self.assertEqual(calls, expected)
                self.assertNotEqual(self.node._inference_status_snapshot['phase'], InferenceStatus.READY)

    def test_successful_load_retry_preserves_parameters_and_starts_once(self):
        client = Mock()
        load = self.prepare_async_load(client)
        client.inference_command.side_effect = [
            SimpleNamespace(success=False, message='already loaded; UNLOAD first'),
            SimpleNamespace(success=True, message='stopped'),
            SimpleNamespace(success=True, message='unloaded'),
            SimpleNamespace(success=True, message='loaded', data={'action_keys': ['arm']}),
            SimpleNamespace(success=True, message='running', data={}),
        ]
        load()
        calls = client.inference_command.call_args_list
        self.assertEqual([c.args[0] for c in calls], [0, 4, 5, 0, 1])
        self.assertEqual(calls[0].kwargs, calls[3].kwargs)
        self.assertEqual(calls[0].kwargs['control_hz'], 80)
        self.assertEqual(calls[0].kwargs['inference_hz'], 20)
        self.assertEqual(calls[0].kwargs['chunk_align_window_s'], 0.25)
        self.assertEqual(calls[0].kwargs['policy_id'], 'lerobot:act')
        self.assertEqual(self.node._loaded_inference_policy_path, '/models/new')
        self.assertEqual(self.node._inference_status_snapshot['phase'], InferenceStatus.INFERENCING)

    def test_replacement_during_load_retry_prevents_remaining_commands(self):
        for stage in (ContainerServiceClient.CMD_STOP, ContainerServiceClient.CMD_UNLOAD):
            with self.subTest(stage=stage):
                client = Mock()
                load = self.prepare_async_load(client)
                replacement = FakeInferenceClient()

                def complete(command, **kwargs):
                    if command == ContainerServiceClient.CMD_LOAD:
                        return SimpleNamespace(success=False, message='already loaded; UNLOAD first')
                    if command == stage:
                        self.node.container_service_client = replacement
                    return SimpleNamespace(success=True, message='ok', data={})

                client.inference_command.side_effect = complete
                before = list(self.node.communicator.phases)
                load()
                expected = [0, 4] if stage == ContainerServiceClient.CMD_STOP else [0, 4, 5]
                self.assertEqual([c.args[0] for c in client.inference_command.call_args_list], expected)
                self.assertEqual(self.node.communicator.phases, before)

    def test_failed_load_becomes_ready_only_after_cleanup_acknowledgement(self):
        client = Mock()
        client._cancelled = threading.Event()
        load = self.prepare_async_load(client)
        logger = FakeLogger()
        errors = []

        def log_error(message, **kwargs):
            allowed = {'throttle_duration_sec', 'throttle_time_source_type', 'skip_first', 'once'}
            if set(kwargs) - allowed:
                raise TypeError('unsupported ROS logger options')
            errors.append(message)

        logger.error = log_error
        self.node.get_logger = lambda: logger
        self.node._loaded_inference_initial_pose_sync = False
        client.inference_command.return_value = SimpleNamespace(success=False, message='load failed')
        with patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            load()
        self.assertTrue(any('Async LOAD/START error' in message for message in errors))
        self.assertEqual(self.node._inference_status_snapshot['phase'], InferenceStatus.LOADING)
        self.assertEqual(self.node._inference_status_snapshot['runtime_state'], 'error')
        self.assertIn('load failed', self.node._inference_status_snapshot['error'])
        self.assertIs(self.node._inference_cleanup_client, client)
        client.inference_command.return_value = SimpleNamespace(success=True, message='ok')
        thread.call_args.kwargs['target']()
        self.assertEqual(self.node._inference_status_snapshot['phase'], InferenceStatus.READY)
        self.assertIn('load failed', self.node._inference_status_snapshot['error'])
        client.disconnect.assert_called_once()
        self.assertIsNone(self.node.container_service_client)

    def test_inference_settings_survive_status_updates_without_touching_recording(self):
        task_info = TaskInfo(
            task_type="inference", policy_path="/models/draft", policy_id="lerobot:groot",
            inference_hz=30, control_hz=80, chunk_align_window_s=0.5,
            task_instruction=["Pick the ball"], initial_pose_sync=True,
            initial_pose_sync_duration_s=7.0,
        )
        with patch.object(self.node, '_forward_recording') as recording:
            response = self.node.user_interaction_callback(
                SendCommand.Request(command=SendCommand.Request.SET_TASK_INFO, task_info=task_info),
                SendCommand.Response(),
            )
        self.assertTrue(response.success, response.message)
        recording.assert_not_called()
        first = self.node.communicator.messages[-1]
        self.assertTrue(first.has_task_info)
        self.assertEqual(first.task_info, task_info)
        self.assertEqual(first.task_info_revision, 1)
        # Request/response objects must not alias the authoritative cache.
        task_info.policy_path = "/mutated/request"
        first.task_info.policy_path = "/mutated/message"
        self.node._prepared_inference_task_info.record_inference_mode = True
        self.node._publish_inference_phase(InferenceStatus.READY)
        restored = self.node.communicator.messages[-1]
        self.assertEqual(restored.task_info.policy_path, "/models/draft")
        self.assertEqual(restored.task_info.inference_hz, 30)
        self.assertFalse(restored.task_info.record_inference_mode)
        self.assertEqual(restored.task_info_revision, 1)
        self.node._cache_ui_task_info(TaskInfo(
            task_type="inference", policy_path="/models/updated",
        ), "SET_TASK_INFO")
        self.assertEqual(self.node.communicator.messages[-1].task_info_revision, 2)

    def test_status_without_saved_settings_is_explicitly_empty(self):
        self.node._publish_inference_phase(InferenceStatus.READY)
        snapshot = self.node.communicator.messages[-1]
        self.assertFalse(snapshot.has_task_info)
        self.assertEqual(snapshot.task_info_revision, 0)

    def test_task_info_settings_default_validate_and_copy(self) -> None:
        self.assertEqual(
            self.node._initial_pose_sync_from_task_info(SimpleNamespace()),
            (False, 5.0),
        )
        self.assertEqual(
            self.node._initial_pose_sync_from_task_info(
                SimpleNamespace(
                    initial_pose_sync=True,
                    initial_pose_sync_duration_s=7.5,
                )
            ),
            (True, 7.5),
        )
        with self.assertRaisesRegex(ValueError, "between 1.0 and 60.0"):
            self.node._initial_pose_sync_from_task_info(
                SimpleNamespace(
                    initial_pose_sync=True,
                    initial_pose_sync_duration_s=0.5,
                )
            )

        task_info = TaskInfo()
        task_info.initial_pose_sync = True
        task_info.initial_pose_sync_duration_s = 6.0
        task_info.policy_id = 'lerobot:act'
        task_info.policy_parameters_json = '{"gain":0.5}'
        copied = self.node._copy_task_info(task_info)
        self.assertTrue(copied.initial_pose_sync)
        self.assertEqual(copied.initial_pose_sync_duration_s, 6.0)
        self.assertEqual(copied.policy_id, 'lerobot:act')
        self.assertEqual(copied.policy_parameters_json, '{"gain":0.5}')

    def test_status_sequence_completes_for_the_active_client(self) -> None:
        self.client.status_results = [
            SimpleNamespace(
                success=True,
                message="syncing",
                data={"runtime_state": "syncing"},
            ),
            SimpleNamespace(
                success=True,
                message="running",
                data={"runtime_state": "running"},
            ),
        ]
        self.node._publish_inference_phase(InferenceStatus.LOADING)
        self.node._begin_initial_pose_sync_status(self.client, 0.01)

        self.node._poll_inference_status_once()
        self.node._poll_inference_status_once()
        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [
                InferenceStatus.LOADING,
                InferenceStatus.SYNCING,
                InferenceStatus.SYNCING,
                InferenceStatus.INFERENCING,
            ],
        )

    def test_sync_status_does_not_guess_completion_from_duration(self) -> None:
        self.client.status_results = [
            SimpleNamespace(
                success=True,
                message="syncing",
                data={"runtime_state": "syncing"},
            ),
            SimpleNamespace(
                success=True,
                message="syncing",
                data={"runtime_state": "syncing"},
            ),
        ]

        self.node._begin_initial_pose_sync_status(self.client, 0.01)
        self.node._poll_inference_status_once()
        self.node._poll_inference_status_once()

        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [InferenceStatus.SYNCING] * 3,
        )

    def test_running_runtime_status_is_returned_to_a_reconnected_ui(self) -> None:
        self.client.status_results = [SimpleNamespace(
            success=True,
            message="running",
            data={
                "runtime_state": "running",
                "loaded_model_path": "/models/act",
                "loaded_policy_id": "lerobot:act",
                "loaded_policy_parameters_json": "{}",
                "publish_to_robot": True,
                "loaded_action_request_mode": "sync",
                "loaded_acceleration_mode": "tensorrt_dit",
                "loaded_acceleration_engine_path": "/models/act/engine",
                "loaded_control_hz": 80,
                "loaded_inference_hz": 20,
                "loaded_chunk_align_window_s": 0.25,
                "loaded_initial_pose_sync": True,
                "loaded_initial_pose_sync_duration_s": 7.0,
            },
        )]

        self.node._poll_inference_status_once()
        response = self.node._handle_get_inference_status(
            TaskInfo(),
            SendCommand.Response(),
        )

        self.assertTrue(response.success)
        self.assertTrue(response.inference_status_known)
        self.assertEqual(response.inference_phase, InferenceStatus.INFERENCING)
        self.assertEqual(response.inference_runtime_state, "running")
        self.assertEqual(response.inference_model_path, "/models/act")
        self.assertEqual(response.inference_policy_id, "lerobot:act")
        self.assertTrue(response.inference_publish_to_robot)
        self.assertTrue(self.node.on_inference)
        self.assertEqual(self.node._loaded_inference_action_request_mode, "sync")
        self.assertEqual(
            self.node._loaded_inference_acceleration_mode, "tensorrt_dit"
        )
        self.assertEqual(
            self.node._loaded_inference_acceleration_engine_path,
            "/models/act/engine",
        )
        self.assertEqual(self.node._loaded_inference_control_hz, 80)
        self.assertEqual(self.node._loaded_inference_inference_hz, 20)
        self.assertEqual(self.node._loaded_inference_chunk_align_window_s, 0.25)
        self.assertTrue(self.node._loaded_inference_initial_pose_sync)
        self.assertEqual(
            self.node._loaded_inference_initial_pose_sync_duration_s, 7.0
        )

    def test_unloaded_runtime_status_restores_ready_without_mutation(self) -> None:
        self.client.status_results = [SimpleNamespace(
            success=True,
            message="unloaded",
            data={
                "runtime_state": "unloaded",
                "loaded_model_path": "",
                "loaded_policy_id": "",
                "loaded_policy_parameters_json": "{}",
                "publish_to_robot": False,
            },
        )]

        self.node._poll_inference_status_once()
        response = self.node._handle_get_inference_status(
            TaskInfo(),
            SendCommand.Response(),
        )

        self.assertTrue(response.success)
        self.assertTrue(response.inference_status_known)
        self.assertEqual(response.inference_phase, InferenceStatus.READY)
        self.assertEqual(response.inference_runtime_state, "unloaded")
        self.assertFalse(self.node.on_inference)

    def test_cancel_blocks_stale_completion(self) -> None:
        self.node._begin_initial_pose_sync_status(self.client, 0.02)
        def delayed_status(*args, **kwargs):
            self.node._cancel_initial_pose_sync_status()
            return SimpleNamespace(success=True, data={'runtime_state': 'running'})
        with patch.object(self.client, 'inference_command', side_effect=delayed_status):
            self.node._poll_inference_status_once()
        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [InferenceStatus.SYNCING],
        )

    def test_client_identity_blocks_stale_completion(self) -> None:
        self.node._begin_initial_pose_sync_status(self.client, 0.02)
        def delayed_status(*args, **kwargs):
            self.node.container_service_client = object()
            return SimpleNamespace(success=True, data={'runtime_state': 'running'})
        with patch.object(self.client, 'inference_command', side_effect=delayed_status):
            self.node._poll_inference_status_once()
        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [InferenceStatus.SYNCING],
        )

    def test_pause_hold_failure_stays_syncing_until_retry_succeeds(self) -> None:
        self.client.pause_results = [
            SimpleNamespace(success=False, message="joint state stale"),
            SimpleNamespace(success=True, message="paused"),
        ]
        self.node._begin_initial_pose_sync_status(self.client, 60.0)

        failed = self.node._pause_inference_client(self.client)

        self.assertFalse(failed.success)
        self.assertIs(self.node.container_service_client, self.client)
        self.assertTrue(self.node._initial_pose_sync_hold_pending)
        self.assertEqual(
            self.node.communicator.phases[-1],
            (InferenceStatus.SYNCING, "ffw_sg2_rev1", "joint state stale"),
        )

        succeeded = self.node._pause_inference_client(self.client)

        self.assertTrue(succeeded.success)
        self.assertFalse(self.node._initial_pose_sync_hold_pending)

    def test_many_ui_reads_only_use_one_cached_runtime_result(self) -> None:
        self.client.status_results = [SimpleNamespace(
            success=True, data={'runtime_state': 'running',
                                'loaded_policy_id': 'groot:n17'},
        )]
        self.node._poll_inference_status_once()
        task = TaskInfo()
        task.policy_id = 'lerobot:act'
        for _ in range(20):
            response = self.node._handle_get_inference_status(task, SendCommand.Response())
            self.assertTrue(response.inference_status_known)
            self.assertEqual(response.inference_policy_id, 'groot:n17')
        self.assertEqual(self.client.calls, [self.client.CMD_STATUS])
        snapshot = self.node.communicator.snapshots[-1]
        self.assertEqual(snapshot['policy_id'], 'groot:n17')
        self.assertTrue(snapshot['source_id'])
        self.assertEqual(snapshot['sequence'], 1)

    def test_publishes_without_robot_communicator_and_serializes_full_snapshot(self) -> None:
        from rclpy.serialization import serialize_message, deserialize_message
        publisher = self.node._inference_status_publisher
        self.node.communicator = None
        self.node._publish_inference_phase(InferenceStatus.INFERENCING)
        message = deserialize_message(serialize_message(publisher.messages[-1]), InferenceStatus)
        self.assertTrue(message.status_known)
        self.assertEqual(message.model_path, '/models/policy')
        self.assertEqual(message.policy_id, 'lerobot:act')
        self.assertEqual(message.sequence, 1)
        self.assertTrue(message.publish_to_robot)

    def test_unreachable_keeps_phase_model_and_hold_until_recovery(self) -> None:
        self.node._begin_initial_pose_sync_status(self.client, 5.0)
        self.node._mark_initial_pose_sync_hold_failed(self.client, 'hold failed')
        self.client.status_results = [SimpleNamespace(success=False, message='offline')]
        self.node._poll_inference_status_once()
        snapshot = self.node.communicator.snapshots[-1]
        self.assertFalse(snapshot['status_known'])
        self.assertEqual(snapshot['phase'], InferenceStatus.SYNCING)
        self.assertEqual(snapshot['model_path'], '/models/policy')
        self.assertTrue(self.node._initial_pose_sync_hold_pending)
        self.client.status_results = [SimpleNamespace(
            success=True, data={'runtime_state': 'paused'},
        )]
        self.node._poll_inference_status_once()
        self.assertTrue(self.node.communicator.snapshots[-1]['status_known'])
        self.assertEqual(self.node.communicator.snapshots[-1]['phase'], InferenceStatus.SYNCING)
        self.assertTrue(self.node._initial_pose_sync_hold_pending)
        self.assertNotIn(self.client.CMD_UNLOAD, self.client.calls)

    def test_command_in_flight_discards_earlier_status(self) -> None:
        self.node._publish_inference_phase(InferenceStatus.PAUSED)
        def delayed_status(*args, **kwargs):
            self.node._observe_inference_command(True)
            self.node._publish_inference_phase(InferenceStatus.INFERENCING)
            self.node._observe_inference_command(False)
            return SimpleNamespace(success=True, data={'runtime_state': 'paused'})
        with patch.object(self.client, 'inference_command', side_effect=delayed_status):
            self.node._poll_inference_status_once()
        self.assertEqual(self.node.communicator.snapshots[-1]['phase'], InferenceStatus.INFERENCING)
        self.assertEqual(len(self.node.communicator.snapshots), 2)

    def test_loading_republishes_progress_without_status_rpc(self) -> None:
        self.node._publish_inference_phase(InferenceStatus.LOADING)
        self.node._poll_inference_status_once()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.node.communicator.snapshots[-1]['phase'], InferenceStatus.LOADING)
        self.assertEqual(self.node.communicator.snapshots[-1]['sequence'], 2)

    def test_preparation_keeps_polling_until_runtime_really_runs(self):
        self.client.status_results = [
            SimpleNamespace(success=True, data={'runtime_state': 'preparing'}),
            SimpleNamespace(success=True, data={'runtime_state': 'running'}),
        ]
        self.node._poll_inference_status_once()
        first = self.node.communicator.snapshots[-1]
        self.assertEqual(first['phase'], InferenceStatus.LOADING)
        self.assertEqual(first['runtime_state'], 'preparing')
        self.assertFalse(self.node._inference_status_busy())
        self.node._poll_inference_status_once()
        self.assertEqual(self.node.communicator.snapshots[-1]['phase'], InferenceStatus.INFERENCING)
        self.assertEqual(self.client.calls, [self.client.CMD_STATUS, self.client.CMD_STATUS])

    def test_pending_cleanup_does_not_rediscover_old_policy(self) -> None:
        self.node._observe_inference_command(True)
        self.node._poll_inference_status_once()
        self.assertEqual(self.client.calls, [])
        self.node._observe_inference_command(False)

    def test_monitor_runs_without_ui_and_stops_without_lifecycle_commands(self) -> None:
        self.client.status_results = [SimpleNamespace(
            success=True, data={'runtime_state': 'running'},
        )]
        self.node._start_inference_status_monitor()
        self.assertTrue(self.node.communicator.inferencing.wait(timeout=1.0))
        self.node._stop_inference_status_monitor()
        self.assertFalse(self.node._inference_status_thread.is_alive())
        self.assertEqual(self.client.calls, [self.client.CMD_STATUS])
        self.assertTrue(self.client.disconnected.is_set())

    def test_invalid_result_does_not_publish_ready(self) -> None:
        self.node._publish_inference_phase(InferenceStatus.INFERENCING)
        self.client.status_results = [SimpleNamespace(
            success=True, data={'runtime_state': 'unexpected'},
        )]
        self.node._poll_inference_status_once()
        snapshot = self.node.communicator.snapshots[-1]
        self.assertFalse(snapshot['status_known'])
        self.assertEqual(snapshot['phase'], InferenceStatus.INFERENCING)

    def test_restart_recovers_active_runtime_independently_of_ui(self) -> None:
        self.node.container_service_client = None
        self.node._client_cb_group = object()
        self.client.status_results = [SimpleNamespace(
            success=True, data={'runtime_state': 'syncing',
                                'loaded_policy_id': 'groot:n17'},
        )]
        with patch('orchestrator.orchestrator_node.ContainerServiceClient') as factory:
            factory.CMD_STATUS = self.client.CMD_STATUS
            self.node._poll_inference_status_once()
            self.assertEqual(factory.call_args.kwargs['service_prefix'], '/groot')
            factory.return_value.connect.assert_called_once()
            self.assertIs(self.node.container_service_client, factory.return_value)
        self.assertTrue(self.node._initial_pose_sync_status_active)

    def test_status_reader_connects_once_to_unified_endpoint(self) -> None:
        self.node._inference_status_client = None
        self.node._client_cb_group = object()
        with patch('orchestrator.orchestrator_node.ContainerServiceClient') as factory:
            factory.return_value.inference_command.return_value = SimpleNamespace(
                success=True, data={'runtime_state': 'unloaded'},
            )
            self.node._poll_inference_status_once()
            self.node._poll_inference_status_once()
            factory.assert_called_once_with(
                node=self.node, service_prefix='/policy',
                callback_group=self.node._client_cb_group, inference_only=True,
            )
            factory.return_value.connect.assert_called_once()
            self.assertEqual(factory.return_value.inference_command.call_count, 2)

    def test_teardown_hold_failure_does_not_unload_or_disconnect(self) -> None:
        self.client.stop_results = [
            SimpleNamespace(success=False, message="joint state unavailable"),
        ]
        self.node._begin_initial_pose_sync_status(self.client, 60.0)

        with self.assertRaisesRegex(RuntimeError, "joint state unavailable"):
            self.node._teardown_inference_client()

        self.assertIs(self.node.container_service_client, self.client)
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])
        self.assertFalse(self.client.disconnected.is_set())
        self.assertTrue(self.node._initial_pose_sync_hold_pending)

    def test_teardown_retries_hold_before_stop_unload_and_disconnect(self) -> None:
        self.client.stop_results = [
            SimpleNamespace(success=False, message="temporary hold failure"),
            SimpleNamespace(success=True, message="stopped"),
        ]
        self.node._begin_initial_pose_sync_status(self.client, 60.0)
        with self.assertRaises(RuntimeError):
            self.node._teardown_inference_client()

        self.node._teardown_inference_client()

        self.assertTrue(self.client.disconnected.wait(timeout=1.0))
        self.assertIsNone(self.node.container_service_client)
        self.assertEqual(
            self.client.calls,
            [
                self.client.CMD_STOP,
                self.client.CMD_STOP,
                self.client.CMD_UNLOAD,
            ],
        )

    def test_elapsed_ui_timer_still_requires_verified_stop(self) -> None:
        self.client.stop_results = [
            SimpleNamespace(success=False, message="policy sync still active"),
        ]

        with self.assertRaisesRegex(RuntimeError, "policy sync still active"):
            self.node._teardown_inference_client()

        self.assertIs(self.node.container_service_client, self.client)
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])
        self.assertFalse(self.client.disconnected.is_set())

    def test_preverified_stop_is_not_sent_twice_during_teardown(self) -> None:
        self.node._begin_initial_pose_sync_status(self.client, 60.0)

        verified_client = self.node._prepare_active_initial_pose_sync_teardown()
        self.node._teardown_inference_client(
            stop_verified_client=verified_client,
        )

        self.assertTrue(self.client.disconnected.wait(timeout=1.0))
        self.assertEqual(
            self.client.calls,
            [self.client.CMD_STOP, self.client.CMD_UNLOAD],
        )

    def test_preverified_stop_expires_after_an_intervening_command(self):
        self.node._begin_initial_pose_sync_status(self.client, 60.0)
        verified = self.node._prepare_active_initial_pose_sync_teardown()
        # Real ContainerServiceClient reports both edges of mutating requests.
        self.node._observe_inference_command(True)
        self.node._observe_inference_command(False)
        self.node._teardown_inference_client(stop_verified_client=verified)
        self.assertTrue(self.client.disconnected.wait(1.0))
        self.assertEqual(self.client.calls, [self.client.CMD_STOP, self.client.CMD_STOP, self.client.CMD_UNLOAD])

    def test_stop_verification_expires_while_cleanup_waits_for_lifecycle(self):
        with patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            self.node._teardown_inference_client()
            cleanup = thread.call_args.kwargs['target']
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])
        self.node._observe_inference_command(True)
        self.node._observe_inference_command(False)
        cleanup()
        self.assertEqual(self.client.calls, [self.client.CMD_STOP, self.client.CMD_STOP, self.client.CMD_UNLOAD])

    def test_duplicate_teardown_does_not_repeat_sync_stop(self):
        with patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            self.node._teardown_inference_client()
            cleanup = thread.call_args.kwargs['target']
            self.node._teardown_inference_client()
        self.assertEqual(thread.call_count, 1)
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])
        cleanup()

    def test_replaced_client_cannot_verify_stop_or_unload(self):
        replacement = FakeInferenceClient()
        def replace_during_stop(command, **kwargs):
            self.client.calls.append(command)
            self.node.container_service_client = replacement
            return SimpleNamespace(success=True, message='stopped')
        with patch.object(self.client, 'inference_command', side_effect=replace_during_stop), \
                patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            with self.assertRaisesRegex(RuntimeError, 'session changed'):
                self.node._teardown_inference_client()
        thread.assert_not_called()
        self.assertIs(self.node.container_service_client, replacement)
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])

    def test_stop_during_busy_lifecycle_is_immediate_but_not_reused(self):
        thread_class = threading.Thread
        finished = threading.Event()
        errors = []
        def teardown():
            try:
                self.node._teardown_inference_client()
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        self.node._inference_lifecycle_lock.acquire()
        try:
            with patch('orchestrator.orchestrator_node.threading.Thread') as factory:
                runner = thread_class(target=teardown)
                runner.start()
                self.assertTrue(finished.wait(1.0), 'Stop waited for an unrelated lifecycle RPC')
                runner.join(1.0)
                self.assertFalse(errors)
                cleanup = factory.call_args.kwargs['target']
            self.assertEqual(self.client.calls, [self.client.CMD_STOP])
            self.assertIsNone(self.node._inference_verified_stop)
        finally:
            self.node._inference_lifecycle_lock.release()
        cleanup()
        self.assertEqual(self.client.calls, [self.client.CMD_STOP, self.client.CMD_STOP, self.client.CMD_UNLOAD])

    def test_async_stop_reply_for_replaced_client_never_unloads_replacement(self):
        self.node._loaded_inference_initial_pose_sync = False
        replacement = FakeInferenceClient()
        def replace_during_stop(command, **kwargs):
            self.client.calls.append(command)
            self.node.container_service_client = replacement
            return SimpleNamespace(success=True, message='stopped')
        with patch('orchestrator.orchestrator_node.threading.Thread') as factory:
            self.node._teardown_inference_client()
            cleanup = factory.call_args.kwargs['target']
        with patch.object(self.client, 'inference_command', side_effect=replace_during_stop):
            cleanup()
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])
        self.assertIs(self.node.container_service_client, replacement)
        self.assertTrue(self.client.disconnected.is_set())
        self.assertFalse(replacement.disconnected.is_set())

    def test_unload_rejection_preserves_client_for_clear_retry(self) -> None:
        self.client.unload_results = [SimpleNamespace(success=False, message='prediction pending')]
        self.node._teardown_inference_client()
        deadline = time.monotonic() + 1
        while getattr(self.node, '_inference_cleanup_client', None) is not None:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        self.assertIs(self.node.container_service_client, self.client)
        self.assertFalse(self.client.disconnected.is_set())
        self.assertFalse(self.client._cancelled.is_set())
        self.assertTrue(any('UNLOAD rejected' in s['error'] for s in self.node.communicator.snapshots))
        self.node._teardown_inference_client()
        self.assertTrue(self.client.disconnected.wait(1))
        self.assertIsNone(self.node.container_service_client)

    def test_non_sync_stop_rejection_prevents_unload_and_disconnect(self) -> None:
        self.node._loaded_inference_initial_pose_sync = False
        self.client.stop_results = [SimpleNamespace(success=False, message='hold failed')]
        self.node._teardown_inference_client()
        deadline = time.monotonic() + 1
        while getattr(self.node, '_inference_cleanup_client', None) is not None:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        self.assertEqual(self.client.calls, [self.client.CMD_STOP])
        self.assertIs(self.node.container_service_client, self.client)
        self.assertFalse(self.client.disconnected.is_set())

    def test_clear_retries_failed_load_cleanup_even_without_active_session(self):
        self.node.on_inference = False
        self.node.on_recording = False
        self.node.timer_manager = None
        self.node._loaded_inference_initial_pose_sync = False
        self.node._publish_inference_phase(InferenceStatus.LOADING)
        self.client.unload_results = [SimpleNamespace(success=False, message='worker unavailable')]
        with patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            self.node._teardown_inference_client()
        thread.call_args.kwargs['target']()
        self.assertIs(self.node.container_service_client, self.client)
        with patch.object(self.node, '_forward_recording', return_value=SimpleNamespace(success=False)), \
                patch('orchestrator.orchestrator_node.threading.Thread') as retry:
            response = self.node.user_interaction_callback(
                SendCommand.Request(command=SendCommand.Request.FINISH,
                                    task_info=TaskInfo(task_type='inference')),
                SendCommand.Response(),
            )
        self.assertTrue(response.success, response.message)
        self.assertNotEqual(self.node._inference_status_snapshot['phase'], InferenceStatus.READY)
        retry.call_args.kwargs['target']()
        self.assertIsNone(self.node.container_service_client)
        self.assertTrue(self.client.disconnected.is_set())
        self.assertEqual(self.node._inference_status_snapshot['phase'], InferenceStatus.READY)

    def test_idle_record_finish_does_not_clear_pending_inference(self):
        self.node.on_inference = False
        self.node.on_recording = False
        with patch.object(self.node, '_teardown_inference_client') as cleanup, \
                patch.object(self.node, '_forward_recording') as recording:
            response = self.node.user_interaction_callback(
                SendCommand.Request(command=SendCommand.Request.FINISH,
                                    task_info=TaskInfo(task_type='record')),
                SendCommand.Response(),
            )
        self.assertFalse(response.success)
        self.assertEqual(response.message, 'Not currently recording')
        cleanup.assert_not_called()
        recording.assert_not_called()
        self.assertIs(self.node.container_service_client, self.client)

    def test_delayed_cleanup_cannot_stop_or_unload_a_replacement_session(self) -> None:
        self.node._loaded_inference_initial_pose_sync = False
        with patch('orchestrator.orchestrator_node.threading.Thread') as thread:
            self.node._teardown_inference_client()
            cleanup = thread.call_args.kwargs['target']
        # Another LOAD/START wins the lifecycle lock before the queued cleanup.
        replacement = FakeInferenceClient()
        self.node.container_service_client = replacement
        self.node._loaded_inference_policy_path = '/models/replacement'
        self.node._publish_inference_phase(InferenceStatus.INFERENCING)
        cleanup()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(replacement.calls, [])
        self.assertTrue(self.client.disconnected.is_set())
        self.assertFalse(replacement.disconnected.is_set())
        self.assertIs(self.node.container_service_client, replacement)
        self.assertEqual(self.node._loaded_inference_policy_path, '/models/replacement')
        self.assertEqual(self.node.communicator.phases[-1][0], InferenceStatus.INFERENCING)
        self.assertIsNone(self.node._inference_cleanup_client)


if __name__ == "__main__":
    unittest.main()
