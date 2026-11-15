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


class PlaygroundServiceHandler(ServiceHandler):
    def _load(self, request):
        request.rlt_enabled = True
        return super()._load(request)


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

    def start(self, publish_to_robot=None, **options) -> float:
        self.starts.append(publish_to_robot)
        return 0.0

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

    def pause(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def initial_pose_sync_hold_required(self) -> bool:
        return False

    def deconfigure(self) -> None:
        pass


def make_response(
    success,
    message="",
    action_keys=None,
    initial_pose_sync_duration_s=0.0,
):
    return SimpleNamespace(
        success=success,
        message=message,
        action_keys=list(action_keys or []),
        initial_pose_sync_duration_s=float(initial_pose_sync_duration_s),
    )


class ServiceHandlerPublishModeTests(unittest.TestCase):
    def test_dataset_selection_is_forwarded_without_starting_inference(self):
        handler, session, loop = self._handler()
        calls = []
        handler._requester.policy_update = lambda command, bundle, **kwargs: (
            calls.append((command, bundle, kwargs)) or SimpleNamespace(success=True, message='selected'))
        session.mark_loaded(robot_type='ffw', task_instruction='pick', action_keys=['arm'])
        for paths in (['/data/a', '/data/b'], []):
            self.assertTrue(handler.handle(SimpleNamespace(command=27, rlt_bundle_path='/bundle',
                                                          rlt_dataset_paths=paths)).success)
            self.assertEqual(calls[-1], (27, '/bundle', {'dataset_paths': paths}))
        self.assertEqual(loop.starts, [])
        self.assertEqual(loop.action_policy_modes, [])
        self.assertTrue(handler.handle(SimpleNamespace(command=25, rlt_bundle_path='/bundle',
                                                      rlt_max_updates=750)).success)
        self.assertEqual(calls[-1], (25, '/bundle', {'max_updates': 750}))
        self.assertEqual(loop.starts, [])

    def test_policy_update_requires_loaded_session_and_preserves_action_mode(self):
        handler, session, loop = self._handler()
        calls = []
        handler._requester.policy_update = lambda command, bundle: (
            calls.append((command, bundle)) or SimpleNamespace(success=True, message='queued')
        )
        request = SimpleNamespace(command=24, rlt_bundle_path='/bundle')
        self.assertFalse(handler.handle(request).success)
        self.assertEqual(calls, [])
        session.mark_loaded(robot_type='ffw', task_instruction='pick', action_keys=['arm'])
        self.assertTrue(handler.handle(request).success)
        self.assertEqual(calls, [(24, '/bundle')])
        self.assertEqual(loop.action_policy_modes, [])
        self.assertEqual(loop.starts, [])

    def _handler(self):
        session = SessionState()
        loop = FakeControlLoop()
        handler = PlaygroundServiceHandler(
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

    def test_load_passes_model_action_shape_to_control_loop(self):
        handler, _session, loop = self._handler()
        handler._requester.load_policy = lambda _request: SimpleNamespace(
            success=True, message="loaded", action_keys=["arm_left", "arm_right"],
            chunk_size=32, action_dim=16,
        )
        result = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path="/model",
            robot_type="f2", task_instruction="drill", action_request_mode="tt_rtc",
            control_hz=200.0))
        self.assertTrue(result.success)
        self.assertEqual(loop.configures[-1]["tt_rtc_horizon"], 32)
        self.assertEqual(loop.configures[-1]["tt_rtc_action_dim"], 16)
        self.assertEqual(loop.configures[-1]["control_hz"], 200.0)

    def test_load_passes_20x16_delay_profile_to_control_loop(self):
        handler, _session, loop = self._handler()
        handler._requester.load_policy = lambda _request: SimpleNamespace(
            success=True,
            message="loaded",
            action_keys=["arm_left", "arm_right"],
            chunk_size=20,
            action_dim=16,
            rtc_min_delay_steps=6,
            rtc_max_delay_steps=8,
        )

        result = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/model",
            robot_type="f2",
            task_instruction="drill",
            action_request_mode="tt_rtc",
            control_hz=200.0,
        ))

        self.assertTrue(result.success)
        config = loop.configures[-1]
        self.assertEqual((config["tt_rtc_horizon"], config["tt_rtc_action_dim"]), (20, 16))
        self.assertEqual(
            (config["tt_rtc_min_delay_steps"], config["tt_rtc_max_delay_steps"]),
            (6, 8),
        )

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

    def test_load_configures_initial_pose_sync_minimum_duration(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            publish_to_robot=True,
            initial_pose_sync=True,
            initial_pose_sync_duration_s=3.5,
        ))

        self.assertTrue(response.success)
        self.assertTrue(loop.configures[0]["initial_pose_sync"])
        self.assertEqual(
            loop.configures[0]["initial_pose_sync_duration_s"],
            3.5,
        )
        self.assertFalse(
            loop.configures[0]["defer_initial_pose_sync_activation"]
        )

    def test_f2_robot_sync_defers_clock_until_arbiter_activation(self) -> None:
        handler, _session, loop = self._handler()

        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="f2",
            task_instruction="pick",
            publish_to_robot=True,
            initial_pose_sync=True,
            initial_pose_sync_duration_s=3.5,
            initial_pose_sync_defer_activation=True,
        ))

        self.assertTrue(response.success)
        self.assertTrue(
            loop.configures[0]["defer_initial_pose_sync_activation"]
        )

    def test_invalid_initial_pose_sync_duration_fails_before_model_load(self) -> None:
        handler, session, loop = self._handler()
        model_load_called = False

        def load_policy(_request):
            nonlocal model_load_called
            model_load_called = True
            return SimpleNamespace(success=True, message="loaded", action_keys=["arm"])

        handler._requester.load_policy = load_policy
        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            initial_pose_sync=True,
            initial_pose_sync_duration_s=61.0,
        ))

        self.assertFalse(response.success)
        self.assertIn("between 1.0 and 60.0", response.message)
        self.assertFalse(model_load_called)
        self.assertFalse(session.loaded)
        self.assertEqual(loop.configures, [])

    def test_start_returns_effective_initial_pose_sync_duration(self) -> None:
        handler, _session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            publish_to_robot=True,
        ))
        loop.start = lambda publish_to_robot=None, **options: 8.0

        response = handler.handle(SimpleNamespace(
            command=CMD_START,
            publish_to_robot=True,
        ))

        self.assertTrue(response.success)
        self.assertEqual(response.message, "syncing")
        self.assertEqual(response.initial_pose_sync_duration_s, 8.0)

    def test_start_rejects_publish_mode_elevation_after_simulation_load(self) -> None:
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

        self.assertFalse(response.success)
        self.assertIn("Deploy Target differs", response.message)
        self.assertEqual(loop.starts, [])

    def test_resume_rejects_publish_mode_elevation_after_simulation_load(self) -> None:
        handler, _session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            publish_to_robot=False,
        ))
        handler.handle(SimpleNamespace(command=CMD_START, publish_to_robot=False))

        response = handler.handle(SimpleNamespace(
            command=CMD_RESUME,
            task_instruction="place",
            publish_to_robot=True,
        ))

        self.assertFalse(response.success)
        self.assertIn("Deploy Target differs", response.message)
        self.assertEqual(loop.starts, [False])
        self.assertEqual(loop.task_instructions, [])

    def test_start_uses_robot_mode_when_policy_was_loaded_for_robot(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(
            command=CMD_LOAD,
            model_path="/models/policy",
            robot_type="ffw",
            task_instruction="pick",
            publish_to_robot=True,
        ))

        response = handler.handle(SimpleNamespace(
            command=CMD_START,
            publish_to_robot=True,
        ))

        self.assertTrue(response.success)
        self.assertTrue(session.publish_to_robot)
        self.assertEqual(loop.starts[-1], True)

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
            publish_to_robot=True,
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
