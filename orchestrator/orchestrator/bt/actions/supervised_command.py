"""Non-blocking direct policy actions used only by server-owned autonomy BTs."""
import time

from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.inference_supervisor import LOAD, START, STOP, UNLOAD, CAPABILITIES
from orchestrator.bt.pause_control import TimerPause


class SupervisedCommand(TimerPause):
    pause_timer_fields = ('deadline',)

    def __init__(self, owner, supervisor):
        self.owner = owner
        self.supervisor = supervisor
        self.phase = 'init'
        self.steps = []
        self.index = 0
        self.future = None
        self.deadline = None
        self.result = None

    def tick(self):
        if self.result is not None:
            return self.result
        try:
            return self._tick()
        except Exception as exc:
            self.owner.log_error(f'Supervised inference failed: {exc}')
            self.result = NodeStatus.FAILURE
            return self.result

    def _tick(self):
        if self.phase == 'init':
            command = self.owner.command_str
            if self.owner.target_str != 'INFERENCE':
                raise RuntimeError('Docker lifecycle actions are not resumable task actions')
            if command == 'LOAD':
                self.owner._start_backend_job('START')
                self.phase = 'backend'
                return NodeStatus.RUNNING
            if not self.supervisor.loaded:
                raise RuntimeError('Supervised inference requires LOAD in the same tree')
            self.steps = {'RESUME': [START], 'STOP': [STOP], 'CLEAR': [STOP, UNLOAD]}[command]
            self.phase = 'commands'
        if self.phase == 'backend':
            result, message = self.owner._backend_job.snapshot()
            if result is None:
                return NodeStatus.RUNNING
            self.owner._backend_job = None
            if not result:
                raise RuntimeError(message)
            self.supervisor.connect(self.owner)
            # Clear a previous resident session before loading the requested
            # checkpoint. No actuator START occurs as part of LOAD.
            self.steps = [CAPABILITIES, STOP, UNLOAD, LOAD]
            self.phase = 'commands'
        if self.index >= len(self.steps):
            if self.owner.command_str == 'RESUME' and time.monotonic() < getattr(self.supervisor, 'sync_until', 0.0):
                return NodeStatus.RUNNING
            self.result = NodeStatus.SUCCESS
            return self.result
        if self.future is None:
            command = self.steps[self.index]
            self.future = self.supervisor.submit(command, self.owner)
            self.deadline = time.monotonic() + (600 if command == LOAD else 20)
        if time.monotonic() > self.deadline:
            raise RuntimeError('Supervised inference command timed out')
        if self.supervisor.poll(self.future):
            self.future = None
            self.index += 1
        return NodeStatus.RUNNING

    def reset(self):
        # Future ownership remains in the supervisor until STOP has drained
        # any START. Cancelling a local future does not cancel robot service work.
        self.phase = 'init'
        self.steps = []
        self.index = 0
        self.future = self.deadline = self.result = None
        self._paused_at = None
