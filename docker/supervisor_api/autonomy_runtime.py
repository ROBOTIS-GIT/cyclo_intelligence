"""Private ROS executor for server-owned navigation and waypoint BT execution."""
import threading
import time


from .autonomy_runtime_control import OperationCancelled, PauseControl, validate_tree, wait_future
from .autonomy_runtime_operations import RuntimeOperations


class AutonomyRosRuntime(RuntimeOperations):
    def __init__(self, start_bt, stop_bt):
        self.start_bt, self.stop_bt = start_bt, stop_bt
        self.lock = threading.RLock()
        self.goal_future = self.goal = self.load_future = None
        self.bt_owned = False
        self.sequence = self.running_sequence = self.terminal_sequence = 0
        self.terminal = self.bt_status = None
        self.node = self.executor = self.context = None
        self.closed = threading.Event()
        self.cancel_lock = threading.Lock()
        self.active_navigation = False
        self.control = PauseControl()
        self.patrol_redirect = threading.Event()

    def open(self, check_busy=True):
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose
        from std_msgs.msg import String
        from action_msgs.msg import GoalStatusArray
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from std_srvs.srv import SetBool
        from interfaces.srv import LoadAndRunTree
        self.context = Context(); self.context.init(args=[])
        self.node = Node('cyclo_autonomy_controller', context=self.context)
        from .autonomy_arrival import ArrivalGuard
        self.arrival = ArrivalGuard(self.node)
        self.nav = ActionClient(self.node, NavigateToPose, '/navigate_to_pose')
        self.load = self.node.create_client(LoadAndRunTree, '/bt/load_and_run')
        self.halt = self.node.create_client(SetBool, '/bt/set_running')
        self.pause_bt = self.node.create_client(SetBool, '/bt/set_paused')
        self.node.create_subscription(String, '/bt/status', self._status, 10)
        self.node.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status',
            lambda msg: setattr(self, 'active_navigation', any(s.status in (1, 2, 3) for s in msg.status_list)),
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()
        self.arrival.configure_clock()
        # Receive latched navigation state and the periodic BT heartbeat before admission.
        time.sleep(1.2)
        if check_busy and (self.active_navigation or self.bt_status in ('running', 'paused', 'pausing', 'pause_failed')):
            raise RuntimeError('Navigation or a waypoint task is already running. Stop it before autonomy.')

    def _spin(self):
        while not self.closed.is_set() and self.context.ok():
            self.executor.spin_once(timeout_sec=.1)

    def _status(self, message):
        with self.lock:
            self.sequence += 1
            self.bt_status = message.data
            if message.data == 'running': self.running_sequence = self.sequence
            if message.data in ('completed', 'failed', 'stopped', 'pause_failed'):
                self.terminal_sequence, self.terminal = self.sequence, message.data

    validate_tree = staticmethod(validate_tree)

    def pause(self):
        self.control.request(True)

    def resume(self):
        self.control.request(False)

    def cancel(self):
        self.control.stop()
        with self.cancel_lock: self._cancel()

    def _cancel(self):
        errors = []
        with self.lock: pending, goal, load = self.goal_future, self.goal, self.load_future
        try:
            if pending is not None: goal = wait_future(pending, 20)
            if goal is not None and goal.accepted:
                # A terminal goal needs no cancellation acknowledgment.
                terminal = goal.get_result_async()
                if not terminal.done():
                    response = wait_future(goal.cancel_goal_async(), 10)
                    if response.return_code not in (0, 3):
                        raise RuntimeError('Navigation cancellation was not confirmed')
                    wait_future(terminal, 15)
        except Exception as exc: errors.append(str(exc))
        try:
            if load is not None: wait_future(load, 30)
            if self.bt_owned:
                from std_srvs.srv import SetBool
                if not self.halt.wait_for_service(timeout_sec=5): raise RuntimeError('Task Engine stop service unavailable')
                req = SetBool.Request(); req.data = False
                response = wait_future(self.halt.call_async(req), 35)
                if not response.success: raise RuntimeError(response.message)
        except Exception as exc: errors.append(str(exc))
        if errors: raise RuntimeError('; '.join(errors))
        with self.lock: self.goal = self.goal_future = self.load_future = None

    def close(self, stop_service=True):
        self.control.stop()
        if stop_service and self.bt_owned:
            self.stop_bt(); self.bt_owned = False
        self.closed.set()
        if self.executor:
            self.spin_thread.join(timeout=3)
            self.executor.shutdown()
            self.node.destroy_node()
            self.context.try_shutdown()
