#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace


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

    def publish_inference_status(self, *, phase, robot_type, error) -> None:
        self.phases.append((phase, robot_type, error))
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
        self.node._initial_pose_sync_status_timer = None
        self.node._initial_pose_sync_status_generation = 0
        self.node._initial_pose_sync_hold_pending = False
        self.node.communicator = FakeCommunicator()
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
        self.node._cancel_initial_pose_sync_status()

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

        self.assertTrue(self.node.communicator.inferencing.wait(timeout=0.5))
        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [
                InferenceStatus.LOADING,
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
        time.sleep(0.15)

        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [InferenceStatus.SYNCING],
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
        self.node._cancel_initial_pose_sync_status()

        time.sleep(0.05)
        self.assertEqual(
            [phase for phase, _robot_type, _error in self.node.communicator.phases],
            [InferenceStatus.SYNCING],
        )

    def test_client_identity_blocks_stale_completion(self) -> None:
        self.node._begin_initial_pose_sync_status(self.client, 0.02)
        self.node.container_service_client = object()

        time.sleep(0.05)
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
