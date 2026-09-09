#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


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
from interfaces.srv import SendCommand  # noqa: E402
from orchestrator.orchestrator_node import OrchestratorNode  # noqa: E402

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


class FakeInferenceClient:
    def __init__(self) -> None:
        self.calls = []
        self.pause_results = []
        self.stop_results = []
        self.status_results = []
        self.disconnected = threading.Event()
        self._cancelled = threading.Event()

    def inference_command(self, command, **_kwargs):
        self.calls.append(command)
        if command == self.CMD_PAUSE and self.pause_results:
            return self.pause_results.pop(0)
        if command == self.CMD_STOP and self.stop_results:
            return self.stop_results.pop(0)
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


if __name__ == "__main__":
    unittest.main()
