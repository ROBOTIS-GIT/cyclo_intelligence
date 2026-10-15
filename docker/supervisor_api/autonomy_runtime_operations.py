"""Blocking navigation/BT workers with cooperative, acknowledged pause points."""
import math
import time
from xml.etree import ElementTree as ET

from .autonomy_runtime_control import OperationCancelled, wait_future
from .autonomy_direction import PatrolRedirect


class RuntimeOperations:
    def _idle_pause(self, cancelled):
        if self.control.requested:
            self.control.wait_resume(cancelled)
            self.control.resumed()
        self.control.check(cancelled)

    def _stop_goal(self, goal, result):
        if result.done():
            return result.result()
        reply = wait_future(goal.cancel_goal_async(), 10)
        if reply.return_code not in (0, 3):
            raise RuntimeError('Navigation pause cancellation was not confirmed')
        return wait_future(result, 15)

    def navigate(self, pose, cancelled):
        from nav2_msgs.action import NavigateToPose
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(pose['x'])
        goal_msg.pose.pose.position.y = float(pose['y'])
        yaw = float(pose.get('yaw', 0))
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2)
        self.control.begin(cancelled)
        error = None
        try:
            if not self.nav.wait_for_server(timeout_sec=10):
                raise RuntimeError('Nav2 action server unavailable')
            remaining = 300.0
            while True:
                self._idle_pause(cancelled)
                if getattr(self, 'patrol_redirect', None) and self.patrol_redirect.is_set():
                    raise PatrolRedirect()
                with self.lock:
                    self.control.check(cancelled)
                    self.goal_future = self.nav.send_goal_async(goal_msg)
                goal = wait_future(self.goal_future, 15)
                with self.lock:
                    self.goal = goal
                if not goal.accepted:
                    raise RuntimeError('Nav2 rejected the goal')
                self.control.check(cancelled)
                result = goal.get_result_async()
                last = time.monotonic()
                retry = False
                while True:
                    self.control.check(cancelled)
                    if getattr(self, 'patrol_redirect', None) and self.patrol_redirect.is_set():
                        with self.cancel_lock:
                            terminal = self._stop_goal(goal, result)
                        if terminal.status not in (4, 5):
                            raise RuntimeError('Patrol goal cancellation failed')
                        raise PatrolRedirect()
                    if self.control.requested:
                        # Cancel only this goal. Resume re-sends the preserved pose.
                        with self.cancel_lock:
                            self.control.check(cancelled)
                            terminal = self._stop_goal(goal, result)
                        if terminal.status not in (4, 5):
                            raise RuntimeError(f'Navigation failed during pause ({terminal.status})')
                        self.control.wait_resume(cancelled)
                        self.control.resumed()
                        last = time.monotonic()
                        if terminal.status == 5:
                            retry = True
                            break
                    if result.done():
                        terminal = result.result()
                        if terminal.status != 4 or getattr(terminal.result, 'error_code', 0):
                            raise RuntimeError(f'Navigation failed (action status {terminal.status})')
                        self.arrival.verify(pose, lambda: self._idle_pause(cancelled))
                        return
                    now = time.monotonic()
                    remaining -= now - last
                    last = now
                    if remaining <= 0:
                        raise TimeoutError('Navigation timed out')
                    time.sleep(.03)
                if retry:
                    with self.lock:
                        self.goal = self.goal_future = None
        except PatrolRedirect:
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            # Preserve in-flight references on error so Stop can drain them.
            if error is None:
                with self.lock:
                    self.goal = self.goal_future = None
            self.control.end(error)

    def _set_tree_paused(self, paused):
        from std_srvs.srv import SetBool
        if not self.pause_bt.wait_for_service(timeout_sec=5):
            raise RuntimeError('Task Engine pause service unavailable; rebuild/restart Task Engine')
        request = SetBool.Request()
        request.data = paused
        response = wait_future(self.pause_bt.call_async(request), 20)
        if not response.success:
            raise RuntimeError(response.message or 'Task Engine pause/resume failed')

    def tree(self, xml, cancelled):
        if not xml.strip():
            return
        self.validate_tree(xml)
        from interfaces.srv import LoadAndRunTree
        self.control.begin(cancelled)
        error = None
        try:
            self._idle_pause(cancelled)
            if not self.bt_owned:
                self.start_bt()
                self.bt_owned = True
            if not self.load.wait_for_service(timeout_sec=15):
                raise RuntimeError('Task Engine unavailable')
            # Do not start a tree on an older server without pause support.
            if not self.pause_bt.wait_for_service(timeout_sec=5):
                raise RuntimeError('Task Engine pause service unavailable; rebuild/restart Task Engine')
            request = LoadAndRunTree.Request()
            parsed = ET.fromstring(xml)
            parsed.set("autonomy_supervised", "true")
            request.tree_xml = ET.tostring(parsed, encoding="unicode")
            with self.lock:
                self.control.check(cancelled)
                baseline = self.sequence
                self.load_future = self.load.call_async(request)
            response = wait_future(self.load_future, 30)
            if not response.success:
                raise RuntimeError(response.message)
            remaining, last = 600.0, time.monotonic()
            while True:
                self.control.check(cancelled)
                if self.control.requested:
                    with self.cancel_lock:
                        self.control.check(cancelled)
                        self._set_tree_paused(True)
                    self.control.wait_resume(cancelled)
                    with self.cancel_lock:
                        self.control.check(cancelled)
                        self._set_tree_paused(False)
                    self.control.resumed()
                    last = time.monotonic()
                with self.lock:
                    if self.running_sequence > baseline and self.terminal_sequence > self.running_sequence:
                        if self.terminal != 'completed':
                            raise RuntimeError(f'Waypoint task {self.terminal}')
                        self.load_future = None
                        return
                now = time.monotonic()
                remaining -= now - last
                last = now
                if remaining <= 0:
                    raise TimeoutError('Waypoint task timed out')
                time.sleep(.03)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self.control.end(error)
