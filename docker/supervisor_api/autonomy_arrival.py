"""Reject false Nav2 success before observation or waypoint task execution."""
import math
import time

from .autonomy_runtime_control import wait_future


# Nav2 latches its 0.20 m position check before final heading alignment.
# Allow up to 0.10 m of alignment/localization drift in the independent guard.
# This does not change Nav2 goal tolerance or accept stale localization.
def check_arrival(pose, transform, now, xy_tolerance=.30, yaw_tolerance=.15):
    stamp = transform.header.stamp
    age = now - (stamp.sec + stamp.nanosec / 1e9)
    t, q = transform.transform.translation, transform.transform.rotation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
    distance = math.hypot(t.x - float(pose['x']), t.y - float(pose['y']))
    angle = abs(math.atan2(math.sin(yaw - float(pose.get('yaw', 0))),
                           math.cos(yaw - float(pose.get('yaw', 0)))))
    if not all(math.isfinite(v) for v in (age, distance, angle)):
        raise RuntimeError('Invalid localization values')
    # AMCL can postdate map->odom by its transform tolerance (normally 1s).
    if age > 1.5 or age < -1.5:
        raise RuntimeError(f'Localization clock mismatch or stale TF (age {age:.2f}s)')
    if distance > xy_tolerance or angle > yaw_tolerance:
        raise RuntimeError(f'Nav2 reported success before arrival: distance={distance:.3f}m, yaw error={angle:.3f}rad')


class ArrivalGuard:
    def __init__(self, node):
        from tf2_ros import Buffer, TransformListener
        self.node = node
        self.buffer = Buffer(node=node)
        self.listener = TransformListener(self.buffer, node)

    def configure_clock(self):
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.parameter import Parameter
        values = []
        for name in ('/bt_navigator', '/robot_state_publisher'):
            client = AsyncParameterClient(self.node, name)
            if not client.wait_for_services(timeout_sec=8):
                raise RuntimeError(f'Cannot verify navigation clock: {name} unavailable')
            reply = wait_future(client.get_parameters(['use_sim_time']), 8)
            if len(reply.values) != 1 or reply.values[0].type != 1:
                raise RuntimeError(f'{name} has no boolean use_sim_time')
            values.append(reply.values[0].bool_value)
        if values[0] != values[1]:
            raise RuntimeError('Nav2 and robot use different clocks. Restart navigation with the robot clock.')
        result = self.node.set_parameters([Parameter('use_sim_time', value=values[0])])
        if not all(r.successful for r in result):
            raise RuntimeError('Could not configure runtime navigation clock')

    def verify(self, pose, check_cancelled):
        from rclpy.time import Time
        deadline = time.monotonic() + 3
        error = 'Localization transform unavailable'
        while time.monotonic() < deadline:
            check_cancelled()
            try:
                transform = self.buffer.lookup_transform('map', 'base_link', Time())
                now = self.node.get_clock().now().nanoseconds / 1e9
                check_arrival(pose, transform, now)
                return
            except Exception as exc:
                error = str(exc)
            time.sleep(.05)
        raise RuntimeError(f'Arrival verification failed: {error}')
