"""Run the real orchestrator callback with fake services, without ROS or hardware."""
import ast
import math
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
from typing import Optional
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'orchestrator/orchestrator_node.py'
SERVICE = ROOT.parent / 'interfaces/srv/SendCommand.srv'
CONSTANTS = {name: int(value) for name, value in re.findall(r'^uint8 (\w+) = (\d+)', SERVICE.read_text(), re.M)}
Request = type('Request', (), CONSTANTS)
PHASE = SimpleNamespace(READY=0, LOADING=1, INFERENCING=2, PAUSED=3, SYNCING=4)


class DeferredThread:
    pending = []

    def __init__(self, target, **kwargs):
        self.target, self.finished = target, False

    def start(self):
        self.pending.append(self)

    def join(self):
        if not self.finished:
            self.finished = True
            self.target()

    @classmethod
    def drain(cls, reverse=False):
        while cls.pending:
            cls.pending.pop(-1 if reverse else 0).join()


class Backend:
    def __init__(self):
        self.loaded = self.running = False
        self.generations = 0
        self.commands = []
        self.requests = []
        self.stop_ok = True
        self.syncing = False
        self.sync_duration_s = 0.0


class Client:
    CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = range(6)

    def __init__(self, node, service_prefix, callback_group):
        self.backend = node.backend
        self._service_prefix = service_prefix
        self._cancelled = threading.Event()

    def connect(self):
        return True

    def disconnect(self):
        pass

    def inference_command(self, command, **kwargs):
        backend = self.backend
        backend.commands.append(command)
        backend.requests.append((command, kwargs))
        if command == self.CMD_STOP and not backend.stop_ok:
            return SimpleNamespace(success=False, message='hold not confirmed', data={})
        if command == self.CMD_LOAD:
            if backend.loaded:
                return SimpleNamespace(success=False, message='policy already loaded - UNLOAD first', data={})
            backend.loaded = True
        elif command in (self.CMD_START, self.CMD_RESUME):
            if not backend.loaded:
                return SimpleNamespace(success=False, message='LOAD first', data={})
            backend.running = True
            backend.generations += 1
        elif command in (self.CMD_PAUSE, self.CMD_STOP):
            backend.running = False
        elif command == self.CMD_UNLOAD:
            backend.loaded = False
        data = {'action_keys': ['arm_left', 'arm_right']}
        message = 'ok'
        if command in (self.CMD_START, self.CMD_RESUME) and backend.syncing:
            data['initial_pose_sync_duration_s'] = backend.sync_duration_s
            message = 'syncing'
        return SimpleNamespace(success=True, message=message, data=data)


def real_methods():
    """Compile production methods unchanged; omit imports that initialize ROS."""
    tree = ast.parse(SOURCE.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OrchestratorNode')
    names = {'user_interaction_callback', '_set_session_active', '_snapshot_session_state', '_teardown_inference_client', '_pause_inference_client', '_inference_clear_error', '_effective_initial_pose_sync_duration', '_initial_pose_sync_from_task_info', '_mark_initial_pose_sync_started', '_select_policy_or_pause', '_activate_deferred_initial_pose_sync'}
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in names]
    environment = {
        'SendCommand': SimpleNamespace(Request=Request), 'InferenceStatus': PHASE,
        'ContainerServiceClient': Client, 'Optional': Optional, 'time': time, 'math': math,
        'threading': SimpleNamespace(Thread=DeferredThread),
        'publish_to_robot_from_task_info': lambda task: task.inference_mode == 'robot',
        'inference_timing_from_task_info': lambda task: (100, 15, .3),
        'inference_runtime_signature': lambda *args, **kwargs: (args, tuple(sorted(kwargs.items()))),
        'RecordingCommand': SimpleNamespace(Request=SimpleNamespace(FINISH=6)),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), 'exec'), environment)
    return {name: environment[name] for name in names}


class Node:
    def __init__(self):
        self.backend = Backend()
        self._state_lock = threading.RLock()
        self._inference_lifecycle_lock = threading.Lock()
        self._client_cb_group = None
        self.container_service_client = None
        self.on_recording = self.on_inference = False
        self._loaded_inference_policy_path = ''
        self._loaded_inference_publish_to_robot = False
        self._loaded_inference_acceleration_mode = 'pytorch'
        self._loaded_inference_acceleration_engine_path = ''
        self._loaded_inference_action_request_mode = 'async'
        self._loaded_inference_vla_rail_enabled = False
        self._loaded_inference_action_chunk_size = 0
        self._loaded_inference_rlt_enabled = False
        self._loaded_inference_rlt_bundle_path = ''
        self._inference_record_session_id = None
        self._loaded_inference_control_hz, self._loaded_inference_inference_hz = 100, 15
        self._loaded_inference_chunk_align_window_s = .3
        self._loaded_inference_initial_pose_sync = False
        self._loaded_inference_initial_pose_sync_duration_s = 5.
        self._initial_pose_sync_status_timer = None
        self._initial_pose_sync_hold_pending = False
        self._initial_pose_sync_status_generation = 0
        self.robot_type = 'ffw_sg2_rev1'
        self.timer_manager = None
        self.phases = []
        self.sync_durations = []
        self.arbiter_enabled = False
        self.arbiter_calls = []
        def arbiter_call(command):
            self.arbiter_calls.append(command)
            return SimpleNamespace(success=True, message='ok')
        self._command_arbiter = SimpleNamespace(
            call=arbiter_call,
            wait_select_policy=lambda **kwargs: arbiter_call('select_policy'),
        )
        self.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)

    def _requires_command_arbiter(self, task=None): return self.arbiter_enabled
    def _action_request_mode_error(self, *args): return ''
    def _acceleration_for_service(self, *args): return 'pytorch', ''
    def _begin_inference_record_session(self, task): self._inference_record_session_id = 'test'
    def _get_inference_record_task_info(self): return None
    def get_logger(self): return self.logger
    def _cache_ui_task_info(self, *args): pass
    def init_robot_control_parameters_from_user_task(self, *args): pass
    def _determine_service_prefix(self, task): return '/groot'
    def _acceleration_from_task_info(self, task): return 'pytorch', ''
    def _action_request_mode_from_task_info(self, task): return getattr(task, 'action_request_mode', 'async')
    def _normalize_policy_path(self, path): return path
    def _initial_pose_sync_is_active(self, client): return self._initial_pose_sync_hold_pending
    def _clear_initial_pose_sync_hold_pending(self, client): self._initial_pose_sync_hold_pending = False
    def _cancel_initial_pose_sync_status(self, expected_client=None): return False
    def _prepare_active_initial_pose_sync_teardown(self): return None
    def _publish_inference_phase(self, phase, **kwargs): self.phases.append((phase, kwargs))
    def _begin_initial_pose_sync_status(self, client, duration_s):
        self.sync_durations.append(duration_s)
        self._publish_inference_phase(PHASE.SYNCING)
    def _apply_cyclo_data_response(self, result, response):
        response.success = result.success and (result.response is None or result.response.success)
        response.message = result.response.message if result.response else result.message
    def _forward_recording(self, *args, **kwargs): return SimpleNamespace(success=True, message='ok', response=None)
    def _is_policy_already_loaded_message(self, message): return 'already loaded' in message


for name, method in real_methods().items():
    setattr(Node, name, method)


class LoadOnlyLifecycleTests(unittest.TestCase):
    def setUp(self):
        DeferredThread.pending = []
        self.node = Node()

    def command(self, command, drain=True, reverse=False, **task_fields):
        task = SimpleNamespace(task_instruction=['pick coffee'], policy_path='/model',
                               inference_mode='robot', task_type='inference', vla_rail_enabled=False, record_inference_mode=False)
        task.__dict__.update(task_fields)
        result = self.node.user_interaction_callback(SimpleNamespace(command=command, task_info=task), SimpleNamespace(success=False, message=''))
        self.assertTrue(result.success, result.message)
        if drain:
            DeferredThread.drain(reverse=reverse)
        return result

    def test_load_prepares_idle_without_start_or_action_generation(self):
        self.command(Request.LOAD_INFERENCE)
        self.assertEqual(self.node.backend.commands, [Client.CMD_STOP, Client.CMD_UNLOAD, Client.CMD_LOAD])
        self.assertTrue(self.node.backend.loaded)
        self.assertFalse(self.node.backend.running)
        self.assertEqual(self.node.backend.generations, 0)
        self.assertEqual(self.node.phases[-1][0], PHASE.PAUSED)
        self.assertTrue(self.node.on_inference)

    def test_resume_after_load_generates_exactly_once(self):
        self.command(Request.LOAD_INFERENCE)
        self.command(Request.RESUME_INFERENCE)
        self.assertEqual(self.node.backend.generations, 1)
        self.assertEqual(self.node.backend.commands[-1], Client.CMD_RESUME)
        self.assertEqual(self.node.phases[-1][0], PHASE.INFERENCING)

    def test_explicit_ui_start_keeps_load_and_start_behavior(self):
        self.command(Request.START_INFERENCE)
        self.assertEqual(self.node.backend.commands, [Client.CMD_LOAD, Client.CMD_START])
        self.assertEqual(self.node.backend.generations, 1)

    def test_repeated_loop_drains_old_cleanup_before_new_load_then_clear(self):
        for _ in range(2):
            self.command(Request.LOAD_INFERENCE, reverse=True)
            self.assertTrue(self.node.backend.loaded)
            self.assertFalse(self.node.backend.running)
            self.command(Request.RESUME_INFERENCE)
            self.command(Request.STOP_INFERENCE)
        self.assertEqual(self.node.backend.generations, 2)
        self.command(Request.FINISH)
        self.assertEqual(self.node.backend.commands[-2:], [Client.CMD_STOP, Client.CMD_UNLOAD])
        self.assertFalse(self.node.backend.loaded)
        self.assertFalse(self.node.on_inference)
        self.assertEqual(self.node.phases[-1][0], PHASE.READY)

    def test_failed_hold_before_load_does_not_unload_or_start(self):
        self.node.backend.stop_ok = False
        self.command(Request.LOAD_INFERENCE)
        self.assertEqual(self.node.backend.commands, [Client.CMD_STOP])
        self.assertEqual(self.node.backend.generations, 0)
        self.assertIn('hold not confirmed', self.node.phases[-1][1]['error'])

    def test_failed_hold_on_resident_policy_does_not_unload_or_replace_it(self):
        self.command(Request.LOAD_INFERENCE)
        self.command(Request.RESUME_INFERENCE)
        self.node.backend.commands.clear()
        self.node.backend.stop_ok = False
        self.command(Request.LOAD_INFERENCE, reverse=True)
        self.assertEqual(self.node.backend.commands, [Client.CMD_STOP])
        self.assertTrue(self.node.backend.loaded)
        self.assertEqual(self.node.backend.generations, 1)
        self.assertIn('hold not confirmed', self.node.phases[-1][1]['error'])

    def test_rail_load_is_forwarded_cached_and_toggle_requires_reload(self):
        self.command(Request.START_INFERENCE, vla_rail_enabled=True)
        loads = [params for command, params in self.node.backend.requests if command == Client.CMD_LOAD]
        self.assertTrue(loads[-1]['vla_rail_enabled'])
        self.assertTrue(self.node._loaded_inference_vla_rail_enabled)
        self.command(Request.START_INFERENCE, vla_rail_enabled=False)
        loads = [params for command, params in self.node.backend.requests if command == Client.CMD_LOAD]
        self.assertEqual(len(loads), 2)
        self.assertFalse(loads[-1]['vla_rail_enabled'])
        self.assertFalse(self.node._loaded_inference_vla_rail_enabled)
        self.command(Request.FINISH)
        self.assertFalse(self.node._loaded_inference_vla_rail_enabled)

    def test_playground_deploy_target_change_reloads(self):
        self.command(Request.START_INFERENCE, action_request_mode='tt_rtc', inference_mode='simulation')
        self.command(Request.START_INFERENCE, action_request_mode='tt_rtc', inference_mode='robot', reverse=True)
        loads = [parameters for command, parameters in self.node.backend.requests if command == Client.CMD_LOAD]
        self.assertEqual(len(loads), 2)
        self.assertFalse(loads[0]['publish_to_robot'])
        self.assertTrue(loads[1]['publish_to_robot'])
        self.assertTrue(self.node.backend.running)

    def test_runtime_change_drains_cleanup_even_when_new_load_runs_first(self):
        self.command(Request.START_INFERENCE, action_request_mode='async')
        self.command(Request.START_INFERENCE, action_request_mode='tt_rtc', reverse=True)
        self.assertTrue(self.node.backend.loaded)
        self.assertTrue(self.node.backend.running)
        self.assertEqual(self.node.backend.commands[-2:], [Client.CMD_LOAD, Client.CMD_START])

    def test_ttrtc_mode_changes_reload_and_finish_resets(self):
        self.command(Request.START_INFERENCE, action_request_mode='tt_rtc')
        loads = [params for command, params in self.node.backend.requests if command == Client.CMD_LOAD]
        self.assertEqual(loads[-1]['action_request_mode'], 'tt_rtc')
        self.assertNotIn('ttrtc_enabled', loads[-1])
        self.command(Request.START_INFERENCE, action_request_mode='async')
        loads = [params for command, params in self.node.backend.requests if command == Client.CMD_LOAD]
        self.assertEqual(len(loads), 2)
        self.assertEqual(loads[-1]['action_request_mode'], 'async')
        self.command(Request.FINISH)
        self.assertEqual(self.node._loaded_inference_action_request_mode, 'async')

    def test_fresh_start_arms_one_timer_for_runtime_extended_sync(self):
        self.node.backend.syncing = True
        self.node.backend.sync_duration_s = 47.5
        self.command(Request.START_INFERENCE, initial_pose_sync=True, initial_pose_sync_duration_s=5.)
        self.assertEqual(self.node.sync_durations, [47.5])
        self.assertEqual(self.node.phases[-1][0], PHASE.SYNCING)

    def test_resume_uses_runtime_extended_duration(self):
        self.command(Request.LOAD_INFERENCE)
        self.node.backend.syncing = True
        self.node.backend.sync_duration_s = 47.5
        self.command(Request.RESUME_INFERENCE, initial_pose_sync=True, initial_pose_sync_duration_s=5.)
        self.assertEqual(self.node.sync_durations, [47.5])
        self.assertEqual(self.node.phases[-1][0], PHASE.SYNCING)

    def test_reused_start_arms_only_one_timer_and_preserves_sync_request(self):
        self.node.backend.syncing = True
        self.node.backend.sync_duration_s = 47.5
        options = dict(initial_pose_sync=True, initial_pose_sync_duration_s=5.)
        self.command(Request.START_INFERENCE, **options)
        self.command(Request.START_INFERENCE, **options)
        self.assertEqual(self.node.sync_durations, [47.5, 47.5])
        command, parameters = self.node.backend.requests[-1]
        self.assertEqual(command, Client.CMD_RESUME)
        self.assertTrue(parameters['initial_pose_sync'])

    def test_ordinary_sync_uses_legacy_duration_without_double_start(self):
        self.node.arbiter_enabled = True
        self.node.backend.syncing = True
        self.command(Request.START_INFERENCE, initial_pose_sync=True, initial_pose_sync_duration_s=5.)
        self.assertEqual(self.node.sync_durations, [5.])
        self.assertEqual(self.node.backend.commands, [Client.CMD_LOAD, Client.CMD_START])

    def test_load_then_resume_selects_arbiter_and_activates_deferred_sync(self):
        self.node.arbiter_enabled = True
        self.command(Request.LOAD_INFERENCE, action_request_mode='tt_rtc', initial_pose_sync=True)
        self.node.backend.syncing = True
        self.node.backend.sync_duration_s = 47.5
        self.command(Request.RESUME_INFERENCE, initial_pose_sync=True, initial_pose_sync_duration_s=5.)
        self.assertEqual(self.node.arbiter_calls, ['select_none', 'select_policy'])
        self.assertEqual(self.node.backend.commands[-2:], [Client.CMD_RESUME, Client.CMD_START])
        self.assertEqual(self.node.sync_durations, [47.5])

    def test_clear_passes_cached_recording_identity_and_keeps_rejected_session(self):
        self.command(Request.LOAD_INFERENCE)
        cached_task = SimpleNamespace(task_num='record-session')
        self.node._get_inference_record_task_info = lambda: cached_task
        forwarded = []
        def reject_clear(*args, **kwargs):
            forwarded.append(kwargs['task_info'])
            return SimpleNamespace(success=True, response=SimpleNamespace(success=False, message='Archive still running'))
        self.node._forward_recording = reject_clear
        result = self.node.user_interaction_callback(
            SimpleNamespace(command=Request.FINISH, task_info=SimpleNamespace(task_type='inference')),
            SimpleNamespace(success=False, message=''),
        )
        self.assertFalse(result.success)
        self.assertEqual(forwarded, [cached_task])
        self.assertTrue(self.node.backend.loaded)
        self.assertIsNotNone(self.node.container_service_client)

    def test_bt_load_is_single_idle_stage(self):
        tree = ast.parse((ROOT / 'orchestrator/bt/actions/send_command.py').read_text())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ('LOAD_INFERENCE_COMMAND', 'INITIAL_POSE_SYNC_MAX_DURATION_SEC', 'PHASE_COMPLETION_GRACE_SEC', 'RESUME_PHASE_TIMEOUT_SEC', 'COMMAND_STAGES') for target in node.targets)]
        env = {'SendCommandSrv': SimpleNamespace(Request=Request), 'InferenceStatus': PHASE}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<bt stages>', 'exec'), env)
        stages = env['COMMAND_STAGES']['LOAD']
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]['command'], Request.LOAD_INFERENCE)
        self.assertEqual(stages[0]['target_phase'], PHASE.PAUSED)


if __name__ == '__main__':
    unittest.main()
