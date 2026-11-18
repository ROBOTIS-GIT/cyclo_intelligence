"""One policy lifecycle owner for supervised and policy-switch BTs.

LOAD never starts actuators. START/STOP replies are serialized by Main and STOP
acknowledges a feedback-confirmed hold. Tree pauses keep the same loaded policy.
"""
import time
import json
from pathlib import Path
from copy import deepcopy

from interfaces.srv import InferenceCommand
from rclpy.callback_groups import ReentrantCallbackGroup

LOAD, START, STOP, UNLOAD, CAPABILITIES, FAULT_STOP = 0, 1, 4, 5, 7, 8
SET_ACTION_POLICY = 20
HOLD_CAPABILITY = 'confirmed_pose_hold_v2'


def wait(future, timeout=15):
    deadline = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() >= deadline:
            raise RuntimeError('Policy lifecycle response timed out; output state is unconfirmed')
        time.sleep(.01)
    reply = future.result()
    if reply is None or not reply.success:
        raise RuntimeError(getattr(reply, 'message', '') or 'Policy command failed')
    return reply


class InferenceSupervisor:
    def __init__(self, node, *, strict_completion=True):
        self.node = node
        self.strict_completion = strict_completion
        self.load_request = None
        self.loaded_bundle_identity = None
        self.pending_bundle_identity = None
        self.desired_action_mode = 'base'
        self.client = None
        self.backend = None
        self.loaded = self.running = False
        self.motion_uncertain = False
        self.pending = None
        self.pending_command = None
        self.pending_owner = None
        self.restore_running = False
        self.stopped = False
        self.verified = False
        self.publish_to_robot = False
        self.instruction = ''
        self.sync_until = 0.0
        self._group = ReentrantCallbackGroup()
        self._revoked_requests = []
        self._fault_future = None
        self._fault_held = False

    def connect(self, owner):
        if self.backend and self.backend != owner.backend:
            raise RuntimeError('One supervised tree cannot switch policy backends')
        if self.client is None:
            self.backend = owner.backend
            self.client = self.node.create_client(
                InferenceCommand, f'/{self.backend}/inference_command',
                callback_group=self._group)

    def submit(self, command, owner=None):
        bundle_identity = None
        request = InferenceCommand.Request()
        request.command = command
        request.publish_to_robot = self.publish_to_robot
        request.task_instruction = self.instruction
        if command == LOAD:
            request.initial_pose_sync = getattr(owner, 'initial_pose_sync', True)
            request.initial_pose_sync_duration_s = float(getattr(owner, 'initial_pose_sync_duration_s', 5.0))
            request.model_path = owner.policy_path
            request.embodiment_tag = 'new_embodiment'
            request.robot_type = self.node.robot_type
            request.publish_to_robot = owner.inference_mode == 'robot'
            request.task_instruction = self._instruction(owner.task_instruction)
            rail = bool(getattr(owner, 'vla_rail_enabled', False))
            if rail and (owner.backend != 'groot' or owner.action_request_mode != 'async'):
                raise ValueError('VLA-RAIL requires ordinary GR00T Async')
            if rail and not hasattr(request, 'vla_rail_enabled'):
                raise RuntimeError('Rebuild interfaces to enable VLA-RAIL')
            if hasattr(request, 'vla_rail_enabled'):
                request.vla_rail_enabled = rail
            tt_rtc = owner.action_request_mode == 'tt_rtc'
            if tt_rtc:
                if (owner.backend != 'groot' or rail
                        or owner.acceleration_mode not in ('', 'pytorch', 'tensorrt_dit')):
                    raise ValueError('TT-RTC requires GR00T without VLA-RAIL')
                if not hasattr(request, 'action_request_mode'):
                    raise RuntimeError('Rebuild interfaces to enable TT-RTC')
            if hasattr(request, 'action_chunk_size'):
                # TT-RTC uses its trained checkpoint horizon, never the UI crop.
                request.action_chunk_size = 0 if tt_rtc else int(getattr(owner, 'action_chunk_size', 0))
            for field in ('action_request_mode', 'acceleration_mode', 'acceleration_engine_path'):
                setattr(request, field, getattr(owner, field))
            # ROS float64 serialization requires Python float even when field
            # checks are disabled. An int can serialize as a denormal double.
            request.control_hz = float(owner.control_hz)
            request.inference_hz = int(owner.inference_hz)
            request.chunk_align_window_s = float(owner.chunk_align_window_s)
            if tt_rtc:
                request.inference_hz = 15
            selection = getattr(owner, 'mlp_preload_selection', None)
            if selection is not None:
                from orchestrator.bt.actions.switch_action_policy import prepare_bundle
                bundle_identity = prepare_bundle(selection)
                request.rlt_enabled = True
                request.rlt_bundle_path = bundle_identity[0]
                request.action_policy_mode = 'base'
        elif command == START and owner is not None:
            request.initial_pose_sync = getattr(owner, 'initial_pose_sync', True)
            request.initial_pose_sync_duration_s = float(getattr(owner, 'initial_pose_sync_duration_s', 5.0))
            request.task_instruction = self._instruction(owner.task_instruction) or self.instruction
        elif command == SET_ACTION_POLICY:
            request.action_policy_mode = self.desired_action_mode
            request.rlt_robot_override = self.desired_action_mode == 'rlt'
        return self.submit_request(request, bundle_identity=bundle_identity)

    def submit_request(self, request, *, bundle_identity=None):
        """Submit a prepared request under the same lifecycle/cancellation guard."""
        command = request.command
        if self.stopped and command not in (STOP, UNLOAD):
            raise RuntimeError('Policy supervision stopped; late START is forbidden')
        if self.pending is not None:
            if not self.pending.done():
                raise RuntimeError('A policy lifecycle request is still pending')
            self.poll(self.pending)
        if not self.client or not self.client.service_is_ready():
            raise RuntimeError('Policy Main service unavailable')
        if command != CAPABILITIES and not self.verified:
            raise RuntimeError('Policy Main has not confirmed its pause capability')
        if command in (START, STOP):
            self.motion_uncertain = True
        self.pending_bundle_identity = bundle_identity
        self.pending_command, self.pending_owner = command, request
        self.pending = self.client.call_async(request)
        return self.pending

    @staticmethod
    def _instruction(value):
        return str(value[0] if isinstance(value, list) and value else value or '')

    def poll(self, future):
        if not future.done():
            return False
        try:
            reply = wait(future)
        except Exception:
            if future is self.pending and future.done():
                self.pending = self.pending_command = self.pending_owner = None
            raise
        if future is not self.pending:
            return True  # Supervisor already settled it during pause; never replay.
        command, request = self.pending_command, self.pending_owner
        self.pending = self.pending_command = self.pending_owner = None
        if command == CAPABILITIES:
            if reply.message != HOLD_CAPABILITY:
                raise RuntimeError('Policy Main lacks feedback-confirmed pause support; update its runtime')
            self.verified = True
        elif command == LOAD:
            identity = self.pending_bundle_identity
            if identity is not None:
                manifest = json.loads((Path(identity[0]) / 'manifest.json').read_text())
                if manifest.get('manifest_fingerprint') != identity[1]:
                    self.loaded = False
                    self.loaded_bundle_identity = None
                    raise RuntimeError('Selected bundle changed during LOAD; reload the tree')
            self.loaded = True
            self.load_request = deepcopy(request)
            self.loaded_bundle_identity = identity
            self.desired_action_mode = 'base'
            self.publish_to_robot = request.publish_to_robot
            self.instruction = request.task_instruction
        elif command == START:
            effective_duration = float(getattr(reply, 'initial_pose_sync_duration_s', 0.0) or 0.0)
            if effective_duration <= 0:
                effective_duration = request.initial_pose_sync_duration_s
            self.sync_until = (time.monotonic() + effective_duration
                               if reply.message == 'syncing' else 0.0)
            self.running = True
            self.motion_uncertain = False
            self.instruction = request.task_instruction
        elif command == STOP:
            self.sync_until = 0.0
            self.running = False
            self.motion_uncertain = False
        elif command == UNLOAD:
            self.loaded = self.running = False
            self.load_request = None
            self.loaded_bundle_identity = None
            self.desired_action_mode = 'base'
        elif command == SET_ACTION_POLICY:
            self.desired_action_mode = request.action_policy_mode
        return True

    def _settle_motion(self):
        # LOAD/UNLOAD cannot publish commands in this direct protocol. Preserve
        # a pending load through pause instead of blocking on weight loading.
        if self.pending is not None and self.pending_command not in (LOAD, UNLOAD, CAPABILITIES):
            future = self.pending
            try:
                wait(future)
            finally:
                if future.done():
                    self.poll(future)
        elif self.pending is not None and self.pending.done():
            self.poll(self.pending)

    def _command(self, command):
        future = self.submit(command)
        try:
            wait(future)
        finally:
            if future.done():
                self.poll(future)

    def _fault_stop(self):
        """Revoke all future starts remotely, including a delayed START reply.

        A missing response cannot prove that START never ran. FAULT_STOP latches
        Main before its lifecycle lock, then confirms physical hold. The latch
        requires a runtime restart; neither a late message nor a new LOAD can
        silently re-enable output. We retain in-flight futures without locally
        cancelling them, but revoked responses can never mutate running state.
        """
        self.stopped = True
        self.restore_running = False
        if self._fault_held:
            return
        if self.pending is not None:
            self._revoked_requests.append(self.pending)
            self.pending = self.pending_command = self.pending_owner = None
        if self._fault_future is None:
            if not self.client or not self.client.service_is_ready():
                raise RuntimeError('Policy emergency hold unconfirmed: Main service unavailable')
            request = InferenceCommand.Request()
            request.command = FAULT_STOP
            self._fault_future = self.client.call_async(request)
        future = self._fault_future
        try:
            wait(future)
        except Exception:
            if future.done():
                self._fault_future = None  # A failed hold may be retried.
            raise
        self._fault_held = True
        self.running = self.motion_uncertain = False
        self._revoked_requests[:] = [f for f in self._revoked_requests if not f.done()]

    def pause(self):
        try:
            self._settle_motion()
            if self.running:
                self.restore_running = True
                self._command(STOP)
        except Exception:
            if self.running or self.motion_uncertain:
                self._fault_stop()
            raise

    def resume(self):
        if self.stopped:
            raise RuntimeError('Stopped policy cannot resume')
        if self.restore_running:
            self._command(START)
            if self.desired_action_mode == 'rlt':
                self._command(SET_ACTION_POLICY)
            self.restore_running = False

    def stop(self):
        # Set before waiting, so failed/time-out STOP can never later resume.
        self.stopped = True
        self.restore_running = False
        if self._fault_future is not None or self._fault_held:
            self._fault_stop()
            return
        try:
            self._settle_motion()
            if self.running or self.motion_uncertain:
                self._command(STOP)
        except Exception:
            if self.running or self.motion_uncertain:
                self._fault_stop()
            else:
                raise

    def complete(self):
        self._settle_motion()
        if self.running:
            self.stop()
            if self.strict_completion:
                raise RuntimeError('Supervised tree completed with policy output still active')
