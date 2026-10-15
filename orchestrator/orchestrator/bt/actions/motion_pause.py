"""Actuator-specific holds for state-preserving BT pause/resume."""
import math
import time


class JointPause:
    def pause(self):
        if self._thread is None or self._result is not None:
            return
        with self._motion_lock:
            self._paused.set()
            positions = self._name_to_position()
            joints = {name for ch in self._channels for name in ch['joint_names']}
            missing = joints - positions.keys()
            stale = [name for name in joints if time.monotonic() - self._joint_received.get(name, 0) > 1.0]
            if missing or stale:
                raise RuntimeError('Cannot hold joints without fresh feedback: ' + ', '.join(sorted(missing | set(stale))))
            if not all(math.isfinite(positions[n]) for n in joints):
                raise RuntimeError('Cannot hold non-finite joint positions')
            self._hold_positions = {name: positions[name] for name in joints}
            for channel in self._channels:
                self._publish_channel(channel, positions=self._hold_positions, duration=.1)
        # Require post-command feedback: publisher delivery alone is not hold
        # acknowledgment. Timer ticks must not block joint callbacks meanwhile.
        command_time, deadline = time.monotonic(), time.monotonic() + 3.0
        stable = 0
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                raise RuntimeError('Joint hold interrupted by Stop')
            observed = self._name_to_position()
            fresh = all(self._joint_received.get(n, 0) > command_time for n in joints)
            close = all(n in observed and abs(observed[n] - self._hold_positions[n]) <= .03 for n in joints)
            slow = all(abs(self._joint_velocities.get(n, float('inf'))) <= .08 for n in joints)
            stable = stable + 1 if fresh and close and slow else 0
            if stable >= 3:
                return
            time.sleep(.03)
        raise RuntimeError('Joint hold was not confirmed by fresh position/velocity feedback')

    def resume(self):
        if not self._paused.is_set():
            return
        with self._motion_lock:
            if self._stop_event.is_set():
                raise RuntimeError('Joint motion stopped; cannot resume')
            for channel in self._channels:
                self._publish_channel(channel)
            self._paused.clear()


class RotatePause:
    def pause(self):
        if self._thread is None or self._result is not None:
            return
        with self._motion_lock:
            self._paused.set()
            self._stop_mobile()
        started = time.monotonic()
        while time.monotonic() - started < 3.0:
            if self._stop_event.is_set():
                raise RuntimeError('Rotation hold interrupted by Stop')
            if (self._odom_received > started and abs(self._odom_angular_velocity) < .03):
                return
            time.sleep(.03)
        raise RuntimeError('Rotation pause was not confirmed by fresh odometry')

    def resume(self):
        if self._stop_event.is_set() and self._paused.is_set():
            raise RuntimeError('Rotation stopped; cannot resume')
        self._paused.clear()
