"""Navigate to a mission waypoint's saved pose; do not execute its local BT."""
from concurrent.futures import Future
from copy import deepcopy
import math
import threading

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.waypoint_targets import resolve_waypoint


class ReturnToWaypoint(BaseAction):
    @classmethod
    def from_xml_params(cls, context, name, params):
        action = cls(context.node, **params)
        action.name = name
        action.snapshot = deepcopy(getattr(context, 'mission_waypoints', []))
        return action

    def __init__(self, node, waypoint: str = '', map_name: str = '', mission_name: str = '',
                 timeout: float = 180., odom_topic: str = '/odom'):
        super().__init__(node, name='ReturnToWaypoint')
        self.waypoint, self.map_name, self.mission_name = [str(s).strip() for s in (waypoint,map_name,mission_name)]
        if len(self.waypoint) > 128 or len(self.map_name) > 128 or len(self.mission_name) > 128:
            raise ValueError('Waypoint and mission names must be at most 128 characters')
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or not 5 <= self.timeout <= 300:
            raise ValueError('timeout must be in [5,300] seconds')
        if not str(odom_topic).strip(): raise ValueError('odom_topic is required')
        self.odom_topic = odom_topic
        self.snapshot = []
        self.resolution = None
        self.runtime = None
        self.failure = False

    def tick(self):
        if self.failure: return NodeStatus.FAILURE
        if self.resolution is None:
            future = Future()
            self.resolution = future
            args = (deepcopy(self.snapshot),self.waypoint,self.map_name,self.mission_name)
            def resolve():
                try:
                    if not args[1]: raise ValueError('Select a destination waypoint')
                    future.set_result(resolve_waypoint(*args))
                except Exception as exc: future.set_exception(exc)
            threading.Thread(target=resolve,daemon=True).start()
            return NodeStatus.RUNNING
        try:
            if not self.resolution.done(): return NodeStatus.RUNNING
            if self.runtime is None:
                from orchestrator.bt.waypoint_navigation import WaypointNavigator
                self.runtime = WaypointNavigator(self.node,self.resolution.result(),self.timeout,self.odom_topic)
            outcome = self.runtime.poll()
            if outcome is None: return NodeStatus.RUNNING
            if not outcome: self.log_error(self.runtime.message)
            try:
                self.runtime.close()
            except RuntimeError as exc:
                # Keep this leaf and its goal owner alive until cancellation/hold
                # is acknowledged; a parent must not reset/replay the action yet.
                self.log_error(str(exc))
                return NodeStatus.RUNNING
            return NodeStatus.SUCCESS if outcome else NodeStatus.FAILURE
        except Exception as exc:
            self.log_error('ReturnToWaypoint: '+str(exc))
            self.failure = True
            return NodeStatus.FAILURE

    def pause(self):
        if self.runtime: self.runtime.pause()

    def resume(self):
        if self.runtime: self.runtime.resume()

    def reset(self):
        if self.runtime: self.runtime.close(); self.runtime = None
        self.resolution = None
        self.failure = False
        super().reset()

    def destroy(self):
        self.reset()
