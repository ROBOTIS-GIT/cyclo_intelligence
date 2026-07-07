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
    CMD_RESUME,
    CMD_START,
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

    def configure(self, **kwargs) -> None:
        self.configures.append(kwargs)

    def start(self, publish_to_robot=None) -> None:
        self.starts.append(publish_to_robot)

    def set_task_instruction(self, task_instruction: str) -> None:
        self.task_instructions.append(task_instruction)

    def pause(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def deconfigure(self) -> None:
        pass


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

    def test_load_allows_empty_model_path_for_remote_backends(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="",
            robot_type="ffw",
            task_instruction="pick",
            remote_host="192.168.0.10",
            remote_port=5555,
            remote_timeout_ms=300000,
            publish_to_robot=False,
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.configures[0]["robot_type"], "ffw")
        self.assertEqual(loop.configures[0]["publish_to_robot"], False)

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


if __name__ == "__main__":
    unittest.main()
