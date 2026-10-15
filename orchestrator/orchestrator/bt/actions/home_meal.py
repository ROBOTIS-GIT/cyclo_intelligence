"""Simulation-only Home meal prop transition, after a JointControl reach."""
import time
from std_srvs.srv import Trigger
from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.pause_control import TimerPause
from orchestrator.bt.bt_core import NodeStatus

class HomeMeal(TimerPause, BaseAction):
    """Ask the Gazebo scene to move dishes or serve food after reaching a station."""
    SERVICE_PREFIX = '/simulation/home_meal/'
    OPERATIONS = ('pickup_table', 'put_sink', 'pickup_sink', 'put_table', 'serve_food')

    def __init__(self, node, operation: str = 'pickup_table', timeout: float = 15.0, check_only: bool = False):
        super().__init__(node, name=type(self).__name__)
        if operation not in self.OPERATIONS:
            raise ValueError('Unknown HomeMeal operation: ' + operation)
        self.operation = operation
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 60:raise ValueError('HomeMeal timeout must be in (0, 60].')
        endpoint = ('check_' if check_only else '') + operation
        self.client = node.create_client(Trigger, self.SERVICE_PREFIX + endpoint)
        self.future = None
        self.started = None
        self.result = None

    pause_timer_fields = ('started',)

    def pause(self):
        # Prop service changes are atomic, not rewindable. Finish any already
        # accepted transition before acknowledging pause; never issue it twice.
        deadline = time.monotonic() + min(self.timeout, 10.0)
        while self.future is not None and not self.future.done():
            if time.monotonic() >= deadline:
                raise RuntimeError('Pending HomeMeal transition could not settle before pause')
            time.sleep(.01)
        super().pause()

    def tick(self):
        if self.result is not None:return self.result
        if self.started is None:
            self.started = time.monotonic()
        if time.monotonic() - self.started > self.timeout:
            self.log_error('HomeMeal service timed out: ' + self.operation)
            if self.future is not None:self.future.cancel()
            self.result = NodeStatus.FAILURE
        elif self.future is None:
            if self.client.service_is_ready():self.future = self.client.call_async(Trigger.Request())
        elif self.future.done():
            try:
                reply = self.future.result()
                self.result = NodeStatus.SUCCESS if reply.success else NodeStatus.FAILURE
                if not reply.success:self.log_error(reply.message)
            except Exception as exc:
                self.log_error(str(exc));self.result = NodeStatus.FAILURE
        return self.result if self.result is not None else NodeStatus.RUNNING

    def reset(self):
        super().reset()
        if self.future is not None and not self.future.done():self.future.cancel()
        self.future = None;self.started = None;self.result = None

    def destroy(self):
        self.reset()
        if self.client is not None:
            self.node.destroy_client(self.client);self.client = None
