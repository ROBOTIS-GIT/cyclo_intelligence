"""One policy lifecycle owner for supervised BTs, without asynchronous /task LOAD.

LOAD never starts actuators. START/STOP replies are serialized by Main and STOP
acknowledges a feedback-confirmed hold. Tree pauses keep the same loaded policy.
"""
import time

from interfaces.srv import InferenceCommand
from rclpy.callback_groups import ReentrantCallbackGroup

LOAD, START, STOP, UNLOAD, CAPABILITIES, FAULT_STOP = 0, 1, 4, 5, 7, 8
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
    def __init__(self, node):
        self.node = node
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
        request = InferenceCommand.Request()
        request.command = command
        request.publish_to_robot = self.publish_to_robot
        request.task_instruction = self.instruction
        if command == LOAD:
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
            for field in ('action_request_mode', 'acceleration_mode', 'acceleration_engine_path',
                          'control_hz', 'inference_hz', 'chunk_align_window_s'):
                setattr(request, field, getattr(owner, field))
        elif command == START and owner is not None:
            request.initial_pose_sync = getattr(owner, 'initial_pose_sync', True)
            request.initial_pose_sync_duration_s = getattr(owner, 'initial_pose_sync_duration_s', 5.0)
            request.task_instruction = self._instruction(owner.task_instruction) or self.instruction
        if command in (START, STOP):
            self.motion_uncertain = True
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
            self.loaded = True
            self.publish_to_robot = request.publish_to_robot
            self.instruction = request.task_instruction
        elif command == START:
            self.sync_until = (time.monotonic() + request.initial_pose_sync_duration_s
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
            raise RuntimeError('Supervised tree completed with policy output still active')
