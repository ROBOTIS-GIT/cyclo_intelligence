#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from main_runtime.service_handler import (  # noqa: E402
    CMD_LOAD,
    CMD_PAUSE,
    CMD_RESUME,
    CMD_START,
    CMD_STATUS,
    CMD_STOP,
    CMD_UNLOAD,
    ServiceHandler,
)
from main_runtime.session_state import SessionState  # noqa: E402
from catalog import load_catalog  # noqa: E402


POLICY_ROOT = RUNTIME_ROOT.parents[1]


class FakeRequester:
    def __init__(self):
        self.loaded_with = None
        self.unload_success = True
        self.unload_count = 0

    def load_policy(self, _request):
        self.loaded_with = _request
        return SimpleNamespace(
            success=True,
            message="loaded",
            action_keys=["arm"],
        )

    def unload_policy(self):
        self.unload_count += 1
        return SimpleNamespace(
            success=self.unload_success,
            message="unloaded" if self.unload_success else "worker unload failed",
        )


class FakeControlLoop:
    def __init__(self) -> None:
        self.configures = []
        self.starts = []
        self.task_instructions = []
        self.start_result = False
        self.start_error = None
        self.configure_error = None
        self.pause_result = True
        self.stop_result = True
        self.hold_pending = False
        self.emergency_stop_result = True
        self.emergency_stop_reasons = []
        self.deconfigure_count = 0

    def configure(self, **kwargs) -> None:
        self.configures.append(kwargs)
        if self.configure_error is not None:
            raise self.configure_error

    def start(self, publish_to_robot=None) -> bool:
        self.starts.append(publish_to_robot)
        if self.start_error is not None:
            raise self.start_error
        return self.start_result

    def set_task_instruction(self, task_instruction: str) -> None:
        self.task_instructions.append(task_instruction)

    def pause(self) -> bool:
        return self.pause_result

    def stop(self) -> bool:
        return self.stop_result

    def deconfigure(self) -> None:
        self.deconfigure_count += 1

    def configuration_snapshot(self) -> dict:
        config = self.configures[-1]
        return {
            "action_request_mode": config.get("action_request_mode") or "async",
            "control_hz": int(config.get("control_hz") or 100),
            "inference_hz": int(config.get("inference_hz") or 15),
            "chunk_align_window_s": float(
                config.get("chunk_align_window_s") or 0.3
            ),
            "initial_pose_sync": bool(config.get("initial_pose_sync", False)),
            "initial_pose_sync_duration_s": float(
                config.get("initial_pose_sync_duration_s") or 5.0
            ),
        }

    def initial_pose_sync_hold_required(self) -> bool:
        return self.hold_pending

    def emergency_stop(self, reason: str) -> bool:
        self.emergency_stop_reasons.append(reason)
        self.hold_pending = not self.emergency_stop_result
        return self.emergency_stop_result


def make_response(
    success,
    message="",
    action_keys=None,
    runtime_state="unloaded",
    loaded_model_path="",
    loaded_policy_id="",
    loaded_policy_parameters_json="{}",
    publish_to_robot=False,
    loaded_action_request_mode="async",
    loaded_acceleration_mode="pytorch",
    loaded_acceleration_engine_path="",
    loaded_control_hz=0,
    loaded_inference_hz=0,
    loaded_chunk_align_window_s=0.0,
    loaded_initial_pose_sync=False,
    loaded_initial_pose_sync_duration_s=5.0,
    runtime_error="",
):
    return SimpleNamespace(
        success=success,
        message=message,
        action_keys=list(action_keys or []),
        runtime_state=runtime_state,
        loaded_model_path=loaded_model_path,
        loaded_policy_id=loaded_policy_id,
        loaded_policy_parameters_json=loaded_policy_parameters_json,
        publish_to_robot=publish_to_robot,
        loaded_action_request_mode=loaded_action_request_mode,
        loaded_acceleration_mode=loaded_acceleration_mode,
        loaded_acceleration_engine_path=loaded_acceleration_engine_path,
        loaded_control_hz=loaded_control_hz,
        loaded_inference_hz=loaded_inference_hz,
        loaded_chunk_align_window_s=loaded_chunk_align_window_s,
        loaded_initial_pose_sync=loaded_initial_pose_sync,
        loaded_initial_pose_sync_duration_s=loaded_initial_pose_sync_duration_s,
        runtime_error=runtime_error,
    )


class ServiceHandlerPublishModeTests(unittest.TestCase):
    def test_worker_mutation_does_not_wait_for_load(self):
        handler, _, _ = self._handler(backend="lerobot")
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        result = []
        original = handler._requester.load_policy

        def slow_load(request):
            entered.set()
            release.wait(3)
            return original(request)

        handler._requester.load_policy = slow_load
        loader = threading.Thread(target=handler.handle, args=(SimpleNamespace(
            command=CMD_LOAD, model_path="/models/policy", robot_type="ffw",
            task_instruction="pick",
        ),))

        def mutate():
            result.append(handler.begin_worker_mutation("lerobot"))
            self.assertFalse(handler.can_mutate_worker("lerobot")[0])
            with self.assertRaisesRegex(RuntimeError, "busy"):
                handler.runtime_snapshot(blocking=False)
            finished.set()

        manager = threading.Thread(target=mutate)
        loader.start()
        try:
            self.assertTrue(entered.wait(1))
            manager.start()
            self.assertTrue(finished.wait(0.5), "management blocked behind LOAD")
            self.assertFalse(result[0][0])
            self.assertEqual(result[0][2], "")
        finally:
            release.set()
            loader.join(3)
            if manager.ident is not None:
                manager.join(3)
        self.assertEqual(handler._worker_mutations, {})

    def test_undelivered_response_does_not_release_another_token(self):
        handler, _, _ = self._handler()
        allowed, _, token = handler.begin_worker_mutation("lerobot")
        self.assertTrue(allowed)
        handler.release_undelivered_worker_mutation(
            {"operation": "begin_worker_mutation", "runtime_id": "lerobot"},
            {"ok": True, "allowed": True, "token": "old-token"},
        )
        self.assertFalse(handler.begin_worker_mutation("lerobot")[0])
        self.assertTrue(handler.end_worker_mutation("lerobot", token))

    def _handler(self, *, catalog=None, backend=""):
        session = SessionState()
        loop = FakeControlLoop()
        requester = FakeRequester()
        handler = ServiceHandler(
            session,
            requester,
            loop,
            make_response,
            catalog=catalog,
            backend=backend,
        )
        return handler, session, loop

    def test_status_reports_complete_session_lifecycle(self) -> None:
        handler, _session, loop = self._handler()

        unloaded = handler.handle(SimpleNamespace(command=CMD_STATUS))
        self.assertTrue(unloaded.success)
        self.assertEqual(unloaded.runtime_state, "unloaded")
        self.assertEqual(unloaded.loaded_model_path, "")

        loaded = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            policy_id="lerobot:act",
            policy_parameters_json="{}",
            publish_to_robot=True,
        ))
        self.assertEqual(loaded.runtime_state, "loaded")
        self.assertEqual(loaded.loaded_model_path, "/models/policy")
        self.assertEqual(loaded.loaded_policy_id, "lerobot:act")
        self.assertTrue(loaded.publish_to_robot)
        self.assertEqual(loaded.loaded_action_request_mode, "async")
        self.assertEqual(loaded.loaded_control_hz, 100)
        self.assertEqual(loaded.loaded_inference_hz, 15)
        self.assertEqual(loaded.loaded_chunk_align_window_s, 0.3)

        running = handler.handle(SimpleNamespace(
            command=CMD_START,
            publish_to_robot=True,
        ))
        self.assertEqual(running.runtime_state, "running")

        loop.hold_pending = True
        syncing = handler.handle(SimpleNamespace(command=CMD_STATUS))
        self.assertEqual(syncing.runtime_state, "syncing")

        loop.hold_pending = False
        paused = handler.handle(SimpleNamespace(command=CMD_PAUSE))
        self.assertEqual(paused.runtime_state, "paused")

        unloaded = handler.handle(SimpleNamespace(command=CMD_UNLOAD))
        self.assertEqual(unloaded.runtime_state, "unloaded")
        self.assertEqual(unloaded.loaded_model_path, "")
        self.assertFalse(unloaded.publish_to_robot)

    def test_unload_failure_preserves_loaded_session_and_control_loop(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler._requester.unload_success = False

        response = handler.handle(SimpleNamespace(command=CMD_UNLOAD))

        self.assertFalse(response.success)
        self.assertEqual(response.runtime_state, "loaded")
        self.assertTrue(session.loaded)
        self.assertEqual(session.model_path, "/models/policy")
        self.assertEqual(loop.deconfigure_count, 0)

    def test_load_configuration_failure_rolls_back_worker_and_session(self) -> None:
        handler, session, loop = self._handler()
        loop.configure_error = RuntimeError("robot config unavailable")

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))

        self.assertFalse(response.success)
        self.assertIn("robot config unavailable", response.message)
        self.assertEqual(response.runtime_state, "unloaded")
        self.assertFalse(session.loaded)
        self.assertEqual(handler._requester.unload_count, 1)
        self.assertEqual(loop.deconfigure_count, 1)

    def test_failed_load_rollback_keeps_recoverable_error_session(self) -> None:
        handler, session, loop = self._handler()
        loop.configure_error = RuntimeError("robot config unavailable")
        handler._requester.unload_success = False

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))

        self.assertFalse(response.success)
        self.assertEqual(response.runtime_state, "error")
        self.assertTrue(session.loaded)
        self.assertIn("worker rollback failed", response.runtime_error)

        handler._requester.unload_success = True
        recovered = handler.handle(SimpleNamespace(command=CMD_UNLOAD))
        self.assertTrue(recovered.success)
        self.assertEqual(recovered.runtime_state, "unloaded")

    def test_unload_is_blocked_while_policy_is_running(self) -> None:
        handler, session, _loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=False))

        response = handler.handle(SimpleNamespace(command=CMD_UNLOAD))

        self.assertFalse(response.success)
        self.assertIn("STOP or PAUSE", response.message)
        self.assertTrue(session.running)
        self.assertEqual(handler._requester.unload_count, 0)

    def test_load_canonicalizes_policy_selection_before_engine_request(self) -> None:
        catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})
        handler, _session, _loop = self._handler(
            catalog=catalog,
            backend="lerobot",
        )
        request = SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            policy_id="act",
            policy_parameters_json="",
        )

        response = handler.handle(request)

        self.assertTrue(response.success)
        self.assertEqual(request.policy_id, "lerobot:act")
        self.assertEqual(request.policy_parameters_json, "{}")
        self.assertIs(handler._requester.loaded_with, request)

    def test_load_rejects_unknown_policy_before_engine_request(self) -> None:
        catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})
        handler, _session, loop = self._handler(
            catalog=catalog,
            backend="lerobot",
        )

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            policy_id="groot:n17",
            policy_parameters_json="{}",
        ))

        self.assertFalse(response.success)
        self.assertIn("belongs to runtime", response.message)
        self.assertIsNone(handler._requester.loaded_with)
        self.assertEqual(loop.configures, [])

    def test_load_rejects_invalid_parameters_before_engine_request(self) -> None:
        catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})
        handler, _session, loop = self._handler(
            catalog=catalog,
            backend="lerobot",
        )

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            policy_id="lerobot:act",
            policy_parameters_json='{"unknown":true}',
        ))

        self.assertFalse(response.success)
        self.assertIn("unknown policy parameters", response.message)
        self.assertIsNone(handler._requester.loaded_with)
        self.assertEqual(loop.configures, [])

    def test_load_rejects_action_mode_not_supported_by_runtime(self) -> None:
        catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})
        catalog = deepcopy(catalog)
        lerobot_runtime = next(
            runtime for runtime in catalog["runtimes"] if runtime["id"] == "lerobot"
        )
        lerobot_runtime["capabilities"]["action_request_modes"] = ["sync"]
        handler, _session, loop = self._handler(
            catalog=catalog,
            backend="lerobot",
        )

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            policy_id="lerobot:act",
            policy_parameters_json="{}",
            action_request_mode="async",
        ))

        self.assertFalse(response.success)
        self.assertIn("does not support action request mode", response.message)
        self.assertIsNone(handler._requester.loaded_with)
        self.assertEqual(loop.configures, [])

    def test_load_rejects_unknown_action_mode_before_engine_request(self) -> None:
        catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})
        handler, _session, loop = self._handler(
            catalog=catalog,
            backend="lerobot",
        )

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            policy_id="lerobot:act",
            policy_parameters_json="{}",
            action_request_mode="rtc",
        ))

        self.assertFalse(response.success)
        self.assertIn("unsupported action request mode", response.message)
        self.assertIsNone(handler._requester.loaded_with)
        self.assertEqual(loop.configures, [])

    def test_empty_policy_id_uses_checkpoint_metadata_and_warns(self) -> None:
        catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})
        handler, _session, _loop = self._handler(
            catalog=catalog,
            backend="lerobot",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(
                '{"type":"act"}',
                encoding="utf-8",
            )
            request = SimpleNamespace(
                command=CMD_LOAD,
                model_path=temp_dir,
                robot_type="ffw",
                task_instruction="",
                policy_id="",
                policy_parameters_json="",
            )

            with self.assertLogs("main_runtime.service_handler", level="WARNING"):
                response = handler.handle(request)

        self.assertTrue(response.success)
        self.assertEqual(request.policy_id, "lerobot:act")

    def test_load_configures_dry_run_by_default(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.configures[0]["publish_to_robot"], False)
        self.assertEqual(loop.configures[0]["action_request_mode"], "async")

    def test_load_configures_robot_publish_when_requested(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            publish_to_robot=True,
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.configures[0]["publish_to_robot"], True)

    def test_load_configures_action_request_mode(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            action_request_mode="sync",
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.configures[0]["action_request_mode"], "sync")

    def test_load_forwards_action_processing_timing(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            control_hz=80,
            inference_hz=20,
            chunk_align_window_s=0.25,
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.configures[0]["control_hz"], 80)
        self.assertEqual(loop.configures[0]["inference_hz"], 20)
        self.assertEqual(loop.configures[0]["chunk_align_window_s"], 0.25)

    def test_load_uses_zero_timing_for_legacy_request(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.configures[0]["control_hz"], 0)
        self.assertEqual(loop.configures[0]["inference_hz"], 0)
        self.assertEqual(loop.configures[0]["chunk_align_window_s"], 0.0)

    def test_load_configures_initial_pose_sync(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            initial_pose_sync=True,
            initial_pose_sync_duration_s=7.5,
        ))

        self.assertTrue(response.success)
        self.assertTrue(loop.configures[0]["initial_pose_sync"])
        self.assertEqual(loop.configures[0]["initial_pose_sync_duration_s"], 7.5)

    def test_start_applies_publish_mode(self) -> None:
        handler, _session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            publish_to_robot=False,
        ))

        response = handler.handle(SimpleNamespace(
            command=CMD_START,
            publish_to_robot=True,
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.starts[-1], True)

    def test_start_reports_syncing_and_marks_session_running(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        loop.start_result = True

        response = handler.handle(SimpleNamespace(
            command=CMD_START,
            publish_to_robot=True,
        ))

        self.assertTrue(response.success)
        self.assertEqual(response.message, "syncing")
        self.assertTrue(session.running)

    def test_failed_initial_sync_does_not_mark_session_running(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        loop.start_error = RuntimeError("sync failed")

        response = handler.handle(SimpleNamespace(
            command=CMD_START,
            publish_to_robot=True,
        ))

        self.assertFalse(response.success)
        self.assertFalse(session.running)

    def test_resume_applies_publish_mode(self) -> None:
        handler, _session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=False))

        response = handler.handle(SimpleNamespace(
            command=CMD_RESUME,
            task_instruction="place",
            publish_to_robot=True,
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.starts[-1], True)
        self.assertEqual(loop.task_instructions[-1], "place")

    def test_pause_marks_session_only_after_hold_succeeds(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True))
        loop.pause_result = False

        failed = handler.handle(SimpleNamespace(command=CMD_PAUSE))
        self.assertFalse(failed.success)
        self.assertTrue(session.running)
        self.assertFalse(session.paused)

        loop.pause_result = True
        succeeded = handler.handle(SimpleNamespace(command=CMD_PAUSE))
        self.assertTrue(succeeded.success)
        self.assertTrue(session.paused)

    def test_stop_keeps_session_running_when_hold_fails(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True))
        loop.stop_result = False

        failed = handler.handle(SimpleNamespace(command=CMD_STOP))
        self.assertFalse(failed.success)
        self.assertTrue(session.running)

        loop.stop_result = True
        succeeded = handler.handle(SimpleNamespace(command=CMD_STOP))
        self.assertTrue(succeeded.success)
        self.assertFalse(session.running)

    def test_unload_is_blocked_while_current_pose_hold_is_pending(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        loop.hold_pending = True

        response = handler.handle(SimpleNamespace(command=CMD_UNLOAD))

        self.assertFalse(response.success)
        self.assertTrue(session.loaded)
        self.assertEqual(loop.deconfigure_count, 0)

    def test_worker_mutation_lease_blocks_load_until_released(self) -> None:
        handler, _session, _loop = self._handler(backend="lerobot")
        allowed, _reason, token = handler.begin_worker_mutation("lerobot")
        self.assertTrue(allowed)

        blocked = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        self.assertFalse(blocked.success)
        self.assertIn("being changed", blocked.message)

        self.assertTrue(handler.end_worker_mutation("lerobot", token))
        loaded = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        self.assertTrue(loaded.success)

    def test_fail_safe_records_runtime_error(self) -> None:
        handler, session, loop = self._handler(backend="lerobot")
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))

        self.assertTrue(handler.fail_safe("worker heartbeat stale"))
        status = handler.handle(SimpleNamespace(command=CMD_STATUS))

        self.assertEqual(status.runtime_state, "error")
        self.assertEqual(status.runtime_error, "worker heartbeat stale")
        self.assertEqual(loop.emergency_stop_reasons, ["worker heartbeat stale"])
        self.assertFalse(session.running)


if __name__ == "__main__":
    unittest.main()
