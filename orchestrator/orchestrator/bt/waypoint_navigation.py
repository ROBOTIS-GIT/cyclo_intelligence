"""Owned Nav2 goal lifecycle, including acknowledged cancellation and pause/resume."""
import math
import threading
import time

from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data

_NAV_LEASE = threading.Lock()


class WaypointNavigator:
    def __init__(self, node, target, timeout=180., odom_topic='/odom'):
        self.node, self.target, self.timeout = node, target, timeout
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.paused_at = None
        self.stopped = False
        self.pending = self.goal = self.result_future = self.cancel_future = None
        self.accept_done = False
        self.sent_at = None
        self.message = ''
        self.error = None
        self.odom = None
        self.odom_at = 0.
        self.stationary_count = 0
        self.last_stationary_sample = 0.
        self.finished_at = None
        self.nav_active = False
        self.closed = False
        if not _NAV_LEASE.acquire(blocking=False):
            raise RuntimeError('Another ReturnToWaypoint action owns Navigation')
        self.leased = True
        self.client = self.odom_sub = self.status_sub = None
        try:
            group = ReentrantCallbackGroup()
            self.client = ActionClient(node, NavigateToPose, '/navigate_to_pose', callback_group=group)
            self.odom_sub = node.create_subscription(Odometry, odom_topic, self.on_odom,
                                                      qos_profile_sensor_data, callback_group=group)
            self.status_sub = node.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status',
                self.on_status, QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=group)
        except Exception:
            for sub in (self.odom_sub, self.status_sub):
                if sub is not None: node.destroy_subscription(sub)
            if self.client is not None: self.client.destroy()
            self.leased = False
            _NAV_LEASE.release()
            raise

    def on_odom(self, message):
        with self.lock: self.odom, self.odom_at = message, time.monotonic()

    def on_status(self, message):
        with self.lock: self.nav_active = any(s.status in (1,2,3) for s in message.status_list)

    def slow(self, after=0.):
        if self.odom is None or self.odom_at <= after or time.monotonic()-self.odom_at > .5:
            return False
        v = self.odom.twist.twist
        return math.hypot(v.linear.x, v.linear.y) < .03 and abs(v.angular.z) < .03

    def send(self):
        message = NavigateToPose.Goal()
        message.pose.header.frame_id = 'map'  # stamp zero works with simulation and wall clocks
        pose = self.target['pose']
        message.pose.pose.position.x, message.pose.pose.position.y = pose['x'], pose['y']
        message.pose.pose.orientation.z = math.sin(pose['yaw']/2)
        message.pose.pose.orientation.w = math.cos(pose['yaw']/2)
        with self.lock:
            if self.stopped or self.paused_at is not None: return
            self.accept_done = False
            self.sent_at = time.monotonic()
            self.pending = self.client.send_goal_async(message)
            self.pending.add_done_callback(self.accept)

    def accept(self, future):
        # A Stop can precede goal acceptance. Keep the client alive and cancel that
        # exact late goal, rather than cancelling unrelated Navigation goals.
        with self.lock:
            try:
                self.goal = future.result()
                if not self.goal.accepted:
                    self.error = 'Nav2 rejected the waypoint goal'
                else:
                    self.result_future = self.goal.get_result_async()
            except Exception as exc: self.error = str(exc)
            self.accept_done = True
            if self.stopped or self.paused_at is not None or self.error:
                self.request_cancel()

    def terminal(self):
        if self.result_future is None or not self.result_future.done(): return None
        try: return self.result_future.result()
        except Exception as exc:
            self.error = 'Navigation result unavailable: '+str(exc)
            return None

    def request_cancel(self):
        with self.lock:
            if (self.goal is not None and self.goal.accepted
                    and self.terminal() is None and self.cancel_future is None):
                self.cancel_future = self.goal.cancel_goal_async()

    def drained(self):
        return ((self.pending is None or self.accept_done)
                and (self.goal is None or not self.goal.accepted or self.terminal() is not None))

    def poll(self):
        """None: running; True: arrived and stationary; False: failed."""
        with self.lock:
            now = time.monotonic()
            if self.paused_at is not None: return None
            if now-self.started > self.timeout:
                self.error = 'ReturnToWaypoint timed out'
            if self.error or self.stopped:
                self.stopped = True
                self.request_cancel()
                self.message = self.error or 'Waypoint navigation stopped'
                return False
            if self.pending is None:
                if not self.client.server_is_ready():
                    if now-self.started > 10.: self.error = 'Nav2 action server unavailable'
                    return None
                if now-self.started < .5: return None  # allow latched status discovery
                if self.nav_active:
                    self.error = 'Another Navigation goal is active; stop it before ReturnToWaypoint'
                    return None
                self.send()
                return None
            if not self.accept_done:
                if now-self.sent_at > 15.: self.error = 'Nav2 goal acceptance timed out'
                return None
            if self.result_future is None: return None
            if not self.result_future.done(): return None
            terminal = self.terminal()
            if terminal is None: return None
            if terminal.status != 4 or getattr(terminal.result, 'error_code', 0):
                self.error = f'Waypoint navigation failed (status {terminal.status}, error {getattr(terminal.result,"error_code",0)})'
                return None
            if self.finished_at is None: self.finished_at = now
            if self.slow() and self.odom_at != self.last_stationary_sample:
                self.stationary_count += 1
                self.last_stationary_sample = self.odom_at
            elif not self.slow(): self.stationary_count = 0
            if self.stationary_count >= 3:
                self.message = 'Arrived at '+self.target['label']
                return True
            if now-self.finished_at > 5.: self.error = 'Waypoint arrival was not confirmed by stationary odometry'
            return None

    def quiesce(self):
        started = time.monotonic()
        with self.lock:
            if self.goal is not None and self.goal.accepted and self.terminal() is None:
                if self.result_future is None or self.result_future.done():
                    self.result_future = self.goal.get_result_async()
        self.request_cancel()
        while time.monotonic()-started < 10.:
            with self.lock:
                if self.drained():
                    if self.goal is None or not self.goal.accepted: return
                    terminal = self.result_future.result()
                    if self.slow(after=started): return
                self.request_cancel()
            time.sleep(.025)
        raise RuntimeError('Waypoint Navigation stop was not confirmed; keep the action for Stop retry')

    def pause(self):
        with self.lock:
            if self.paused_at is None: self.paused_at = time.monotonic()
        self.quiesce()

    def resume(self):
        with self.lock:
            if self.stopped: raise RuntimeError('Waypoint Navigation is stopped')
            if self.paused_at is None: return
            self.started += time.monotonic()-self.paused_at
            self.paused_at = None
            # A completed goal stays completed; a cancelled goal reuses the same
            # resolved pose, never the destination waypoint's local BT.
            if self.result_future is not None and self.result_future.done():
                terminal = self.result_future.result()
                if terminal.status == 5:
                    self.pending = self.goal = self.result_future = self.cancel_future = None
                    self.accept_done = False
                    self.stationary_count = 0
                    self.finished_at = None

    def close(self):
        if self.closed: return
        with self.lock: self.stopped = True
        self.quiesce()  # On failure preserve clients/futures so Stop can be retried.
        self.client.destroy()
        self.node.destroy_subscription(self.odom_sub)
        self.node.destroy_subscription(self.status_sub)
        self.closed = True
        if self.leased: self.leased = False; _NAV_LEASE.release()
