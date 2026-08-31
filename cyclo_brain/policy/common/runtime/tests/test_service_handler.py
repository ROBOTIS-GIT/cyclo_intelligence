#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
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
    CMD_STOP,
    CMD_UNLOAD,
    ServiceHandler,
)
from main_runtime.session_state import SessionState  # noqa: E402


class FakeRequester:
    def load_policy(self, _request):
        return SimpleNamespace(
            success=True,
            message="loaded",
            action_keys=["arm"],
        )

    def unload_policy(self):
        return SimpleNamespace(success=True, message="unloaded")


class FakeControlLoop:
    def __init__(self) -> None:
        self.configures = []
        self.starts = []
        self.task_instructions = []
        self.start_result = False
        self.start_error = None
        self.pause_result = True
        self.stop_result = True
        self.hold_pending = False
        self.deconfigure_count = 0

    def configure(self, **kwargs) -> None:
        self.configures.append(kwargs)

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

    def initial_pose_sync_hold_required(self) -> bool:
        return self.hold_pending


def make_response(success, message="", action_keys=None):
    return SimpleNamespace(
        success=success,
        message=message,
        action_keys=list(action_keys or []),
    )


class ServiceHandlerPublishModeTests(unittest.TestCase):
    def _handler(self):
        session = SessionState()
        loop = FakeControlLoop()
        handler = ServiceHandler(
            session,
            FakeRequester(),
            loop,
            make_response,
        )
        return handler, session, loop

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


if __name__ == "__main__":
    unittest.main()
