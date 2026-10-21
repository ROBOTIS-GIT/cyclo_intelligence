#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import unittest
import json
import tempfile
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from main_runtime.service_handler import (  # noqa: E402
    CMD_CAPABILITIES,
    CMD_FAULT_STOP,
    CMD_UPDATE_INSTRUCTION,
    CMD_LOAD,
    CMD_PAUSE,
    CMD_RESUME,
    CMD_START,
    CMD_STOP,
    CMD_UNLOAD,
    CMD_UPDATE_INSTRUCTION,
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
    def test_manual_and_detector_updates_share_idempotent_ordered_transition(self):
        handler, session, loop = self._handler()
        session.mark_loaded('ffw', 'phase 1', ['arm'])
        session.mark_running()
        handler._subtask_choices = [['phase 1'], ['phase 2'], ['phase 3']]
        loop.transition_subtask = mock.Mock()
        request = SimpleNamespace(command=CMD_UPDATE_INSTRUCTION, task_instruction='phase 2')
        self.assertTrue(handler.handle(request).success)
        self.assertEqual(session.task_instruction, 'phase 2')
        self.assertEqual(handler.subtask_state()['subtask_index'], 1)
        self.assertTrue(handler.handle(request).success)
        loop.transition_subtask.assert_called_once_with('phase 2')
        response = handler.handle(SimpleNamespace(command=CMD_UPDATE_INSTRUCTION, task_instruction='phase 1'))
        self.assertFalse(response.success)
        self.assertEqual(session.task_instruction, 'phase 2')
        self.assertEqual(loop.task_instructions, [])

    def test_failed_transition_keeps_confirmed_instruction(self):
        handler, session, loop = self._handler()
        session.mark_loaded('ffw', 'phase 1', ['arm'])
        session.mark_running()
        handler._subtask_choices = [['phase 1'], ['phase 2']]
        loop.transition_subtask = mock.Mock(side_effect=RuntimeError('hold failed'))
        result = handler.handle(SimpleNamespace(command=CMD_UPDATE_INSTRUCTION, task_instruction='phase 2'))
        self.assertFalse(result.success)
        self.assertEqual(session.task_instruction, 'phase 1')
        self.assertEqual(handler.subtask_state()['instruction'], 'phase 1')

    def test_regular_checkpoint_uses_legacy_instruction_update(self):
        handler, session, loop = self._handler()
        session.mark_loaded('ffw', 'whole task', ['arm'])
        session.mark_running()
        result = handler.handle(SimpleNamespace(command=CMD_UPDATE_INSTRUCTION, task_instruction='new whole task'))
        self.assertTrue(result.success)
        self.assertEqual(loop.task_instructions, ['new whole task'])

    def test_subtask_load_validates_prompt_before_loading_weights(self):
        handler, session, loop = self._handler()
        handler._requester.load_policy = mock.Mock()
        with mock.patch('main_runtime.service_handler.checkpoint_subtask_choices', return_value=[['phase 1'], ['phase 2']]):
            response = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/workspace/model/groot/example',
                                                     robot_type='ffw', task_instruction='whole task'))
        self.assertFalse(response.success)
        handler._requester.load_policy.assert_not_called()

    def test_chunk_override_selects_chunk_control_instead_of_saved_single_step(self):
        handler, session, loop = self._handler()
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(json.dumps({"type": "act", "chunk_size": 30,
                "n_action_steps": 1, "temporal_ensemble_coeff": None}))
            response = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path=root,
                robot_type="ffw", task_instruction="", inference_hz=15, action_chunk_size=100))
        self.assertTrue(response.success, response.message)
        self.assertFalse(loop.configures[-1].get("per_step_actions", False))

    def test_temporal_ensemble_keeps_per_step_control_with_chunk_override(self):
        handler, session, loop = self._handler()
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(json.dumps({"type": "act", "chunk_size": 30,
                "n_action_steps": 1, "temporal_ensemble_coeff": 0.01}))
            response = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path=root,
                robot_type="ffw", task_instruction="", inference_hz=15, action_chunk_size=8))
        self.assertTrue(response.success, response.message)
        self.assertTrue(loop.configures[-1]["per_step_actions"])

    def test_single_step_act_disables_chunk_processing_and_resets_on_start(self):
        handler, session, loop = self._handler()
        handler._requester.reset_policy = mock.Mock(return_value=SimpleNamespace(success=True))
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(json.dumps({"type": "act", "n_action_steps": 1}))
            Path(root, "inference_contract.json").write_text(json.dumps({"fps": 15}))
            response = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path=root,
                                      robot_type="ffw", task_instruction="", inference_hz=15))
        self.assertTrue(response.success, response.message)
        self.assertTrue(loop.configures[-1]["per_step_actions"])
        response = handler.handle(SimpleNamespace(command=CMD_START))
        self.assertTrue(response.success, response.message)
        handler._requester.reset_policy.assert_called_once()

    def test_single_step_act_rejects_wrong_dataset_cadence(self):
        handler, session, loop = self._handler()
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(json.dumps({"type": "act", "n_action_steps": 1}))
            Path(root, "inference_contract.json").write_text(json.dumps({"fps": 15}))
            response = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path=root,
                                      robot_type="ffw", task_instruction="", inference_hz=100))
        self.assertFalse(response.success)
        self.assertIn("Inference Hz=15", response.message)
        self.assertFalse(loop.configures)

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

    def test_load_passes_rail_flag_and_defaults_off(self):
        for flag in (None, False, True):
            handler, session, loop = self._handler()
            args = dict(command=CMD_LOAD, model_path='/models/policy', robot_type='ffw', task_instruction='pick')
            if flag is not None:
                args['vla_rail_enabled'] = flag
            response = handler.handle(SimpleNamespace(**args))
            self.assertTrue(response.success)
            self.assertIs(loop.configures[-1]['vla_rail_enabled'], flag is True)


    def test_capability_probe_is_read_only_without_loaded_policy(self):
        handler, session, loop = self._handler()
        response = handler.handle(SimpleNamespace(command=CMD_CAPABILITIES))
        self.assertTrue(response.success)
        self.assertEqual(response.message, 'confirmed_pose_hold_v2')
        self.assertFalse(session.loaded)
        self.assertFalse(session.running)
        self.assertEqual(loop.starts, [])
        self.assertEqual(loop.configures, [])

    def test_stop_waits_for_inflight_start_then_leaves_session_stopped(self):
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='pick'))
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        order, results = [], []
        def start(publish_to_robot=None):
            order.append('start_enter'); entered.set()
            if not release.wait(2):
                raise RuntimeError('test did not release start')
            order.append('start_done')
            return False
        def stop():
            order.append('stop'); stopped.set()
            return True
        loop.start, loop.stop = start, stop
        first = threading.Thread(target=lambda: results.append(handler.handle(SimpleNamespace(command=CMD_START))))
        second = threading.Thread(target=lambda: results.append(handler.handle(SimpleNamespace(command=CMD_STOP))))
        first.start()
        try:
            self.assertTrue(entered.wait(1)); second.start()
            self.assertFalse(stopped.wait(.05))
        finally:
            release.set(); first.join(2); second.join(2)
        self.assertFalse(first.is_alive()); self.assertFalse(second.is_alive())
        self.assertTrue(all(result.success for result in results))
        self.assertEqual(order, ['start_enter', 'start_done', 'stop'])
        self.assertFalse(session.running)

    def test_fault_stop_latches_before_waiting_for_inflight_start(self):
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='pick'))
        entered, release, held = threading.Event(), threading.Event(), threading.Event()
        order, results = [], {}
        def start(publish_to_robot=None):
            order.append('start_enter'); entered.set()
            if not release.wait(2):
                raise RuntimeError('test did not release START')
            order.append('start_done')
            return False
        def stop():
            order.append('hold'); held.set()
            return True
        loop.start, loop.stop = start, stop
        first = threading.Thread(target=lambda: results.update(start=handler.handle(SimpleNamespace(command=CMD_START))))
        fault = threading.Thread(target=lambda: results.update(fault=handler.handle(SimpleNamespace(command=CMD_FAULT_STOP))))
        late = threading.Thread(target=lambda: results.update(late=handler.handle(SimpleNamespace(command=CMD_START))))
        first.start()
        try:
            self.assertTrue(entered.wait(1)); fault.start()
            self.assertTrue(handler._fault_latched.wait(1))
            late.start()
            self.assertFalse(held.is_set())
        finally:
            release.set()
            for thread in (first, fault, late):
                if thread.ident is not None: thread.join(2)
        self.assertTrue(all(not thread.is_alive() for thread in (first, fault, late)))
        self.assertTrue(results['fault'].success)
        self.assertFalse(results['late'].success)
        self.assertEqual(order, ['start_enter', 'start_done', 'hold'])
        self.assertFalse(session.running)

    def test_fault_ack_rejects_late_actuating_commands_and_load_cannot_clear_latch(self):
        handler, session, loop = self._handler()
        load = SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                               robot_type='ffw', task_instruction='pick')
        self.assertTrue(handler.handle(load).success)
        self.assertTrue(handler.handle(SimpleNamespace(command=CMD_FAULT_STOP)).success)
        for command in (CMD_START, CMD_RESUME, CMD_UPDATE_INSTRUCTION, CMD_LOAD):
            with self.subTest(command=command):
                reply = handler.handle(SimpleNamespace(command=command, task_instruction='late',
                                      model_path='/models/replacement', robot_type='ffw'))
                self.assertFalse(reply.success)
                self.assertIn('fault', reply.message.lower())
        self.assertEqual(loop.starts, [])
        self.assertEqual(loop.task_instructions, [])
        self.assertEqual(len(loop.configures), 1)
        self.assertTrue(handler.handle(SimpleNamespace(command=CMD_STOP)).success)
        self.assertTrue(handler.handle(SimpleNamespace(command=CMD_UNLOAD)).success)
        self.assertFalse(handler.handle(load).success)
        self.assertFalse(session.loaded)
        self.assertEqual(handler.handle(SimpleNamespace(command=CMD_CAPABILITIES)).message,
                         'confirmed_pose_hold_v2')

    def test_failed_fault_hold_stays_latched_after_retry_succeeds(self):
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='pick'))
        handler.handle(SimpleNamespace(command=CMD_START))
        loop.stop_result = False
        failed = handler.handle(SimpleNamespace(command=CMD_FAULT_STOP))
        self.assertFalse(failed.success)
        self.assertTrue(handler._fault_latched.is_set())
        self.assertFalse(handler.handle(SimpleNamespace(command=CMD_START)).success)
        self.assertEqual(len(loop.starts), 1)
        loop.stop_result = True
        self.assertTrue(handler.handle(SimpleNamespace(command=CMD_STOP)).success)
        self.assertFalse(session.running)
        self.assertFalse(handler.handle(SimpleNamespace(command=CMD_START)).success)
        self.assertEqual(len(loop.starts), 1)

    def test_load_only_then_resume_starts_once_with_requested_instruction(self):
        handler, session, loop = self._handler()
        loaded = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                                robot_type='ffw', task_instruction='pick',
                                                publish_to_robot=True))
        self.assertTrue(loaded.success)
        self.assertTrue(session.loaded)
        self.assertFalse(session.running)
        self.assertEqual(loop.starts, [])
        resumed = handler.handle(SimpleNamespace(command=CMD_RESUME, task_instruction='place',
                                                 publish_to_robot=True))
        self.assertTrue(resumed.success)
        self.assertTrue(session.running)
        self.assertEqual(loop.starts, [True])
        self.assertEqual(loop.task_instructions, ['place'])
        self.assertEqual(session.task_instruction, 'place')

    def test_resume_loaded_idle_keeps_load_instruction_when_request_is_empty(self):
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='pick'))
        resumed = handler.handle(SimpleNamespace(command=CMD_RESUME, task_instruction='',
                                                 publish_to_robot=False))
        self.assertTrue(resumed.success)
        self.assertTrue(session.running)
        self.assertEqual(session.task_instruction, 'pick')
        self.assertEqual(loop.starts, [False])
        self.assertEqual(loop.configures[0]['task_instruction'], 'pick')

    def test_loaded_idle_resume_preserves_start_hold_failures(self):
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='pick'))
        loop.start_error = RuntimeError('current-pose hold is still pending')
        reply = handler.handle(SimpleNamespace(command=CMD_RESUME, task_instruction='',
                                               publish_to_robot=True))
        self.assertFalse(reply.success)
        self.assertIn('hold', reply.message)
        self.assertFalse(session.running)

    def test_stop_is_repeatable_and_resume_after_stop_requires_loaded_policy(self):
        handler, session, loop = self._handler()
        resume = SimpleNamespace(command=CMD_RESUME, task_instruction='', publish_to_robot=True)
        self.assertFalse(handler.handle(resume).success)
        self.assertEqual(loop.starts, [])
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='pick'))
        for _ in range(2):
            self.assertTrue(handler.handle(SimpleNamespace(command=CMD_STOP)).success)
        self.assertFalse(session.running)
        self.assertTrue(handler.handle(resume).success)
        self.assertEqual(loop.starts, [True])
        self.assertTrue(handler.handle(SimpleNamespace(command=CMD_STOP)).success)
        self.assertTrue(handler.handle(SimpleNamespace(command=CMD_UNLOAD)).success)
        self.assertFalse(session.loaded)
        self.assertFalse(handler.handle(resume).success)
        self.assertEqual(loop.starts, [True])
    def test_failed_configuration_keeps_session_unloaded_and_releases_engine(self):
        handler, session, loop = self._handler()
        loop.configure = mock.Mock(side_effect=ValueError("incompatible action layout"))
        handler._requester.unload_policy = mock.Mock(
            return_value=SimpleNamespace(success=True)
        )
        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD, model_path="/models/policy", robot_type="ffw",
            task_instruction="place",
        ))
        self.assertFalse(response.success)
        self.assertFalse(session.loaded)
        self.assertEqual(loop.deconfigure_count, 1)
        handler._requester.unload_policy.assert_called_once()

    def test_empty_model_action_keys_are_rejected_before_configuration(self):
        handler, session, loop = self._handler()
        handler._requester.load_policy = mock.Mock(return_value=SimpleNamespace(
            success=True, action_keys=[], message="loaded"
        ))
        handler._requester.unload_policy = mock.Mock(
            return_value=SimpleNamespace(success=True)
        )
        response = handler.handle(SimpleNamespace(
            command=CMD_LOAD, model_path="/models/policy", robot_type="ffw",
            task_instruction="place",
        ))
        self.assertFalse(response.success)
        self.assertFalse(session.loaded)
        self.assertEqual(loop.configures, [])
        handler._requester.unload_policy.assert_called_once()

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

    def test_start_updates_instruction_for_new_supervised_session(self) -> None:
        handler, session, loop = self._handler()
        handler.handle(SimpleNamespace(command=CMD_LOAD, model_path='/models/policy',
                                       robot_type='ffw', task_instruction='old task'))
        response = handler.handle(SimpleNamespace(command=CMD_START,
                                                   task_instruction='  new task  '))
        self.assertTrue(response.success)
        self.assertEqual(session.task_instruction, 'new task')
        self.assertEqual(loop.task_instructions, ['new task'])
        self.assertTrue(session.running)

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
