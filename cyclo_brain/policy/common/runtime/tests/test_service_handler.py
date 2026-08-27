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
    CMD_SET_ACTION_POLICY,
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
        self.action_policy_modes = []

    def configure(self, **kwargs) -> None:
        self.configures.append(kwargs)

    def start(self, publish_to_robot=None) -> None:
        self.starts.append(publish_to_robot)

    def set_task_instruction(self, task_instruction: str) -> None:
        self.task_instructions.append(task_instruction)

    def set_action_policy(
        self,
        action_policy_mode: str,
        allow_robot_rlt=False,
        timeout_s=5.0,
    ):
        self.action_policy_modes.append(
            (action_policy_mode, bool(allow_robot_rlt), timeout_s)
        )
        return True, f"{action_policy_mode.upper()} action active"

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

    def test_load_configures_rlt_preload_flag(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            rlt_enabled=True,
        ))

        self.assertTrue(response.success)
        self.assertTrue(loop.configures[0]["rlt_enabled"])

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

    def test_running_session_forwards_action_policy_switch(self) -> None:
        handler, _session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=False))

        response = handler.handle(SimpleNamespace(
            command=CMD_SET_ACTION_POLICY,
            action_policy_mode="rlt",
            rlt_robot_override=True,
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.action_policy_modes, [("rlt", True, 5.0)])

    def test_action_policy_switch_defaults_robot_override_to_false(self) -> None:
        handler, _session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=True))

        response = handler.handle(SimpleNamespace(
            command=CMD_SET_ACTION_POLICY,
            action_policy_mode="rlt",
        ))

        self.assertTrue(response.success)
        self.assertEqual(loop.action_policy_modes, [("rlt", False, 5.0)])


if __name__ == "__main__":
    unittest.main()
