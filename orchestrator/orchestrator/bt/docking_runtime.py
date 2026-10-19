"""ROS transport and lifecycle for local, image-feedback ArUco docking."""
import json
import math
import threading
import time

import cv2
import numpy as np
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .docking_rectification import DockRectifier
from .docking_core import dock_error, velocity, obstacle_in_motion
from .docking_vision import MarkerDetector, transform_pose

_LEASE = threading.Lock()


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class DockRuntime:
    def __init__(self, node, marker_id, marker_size, config, image_topic, camera_info_topic,
                 cmd_vel_topic, scan_topic, odom_topic, base_frame, image_mode="raw"):
        self.node, self.config, self.base_frame = node, config, base_frame
        self.detector = MarkerDetector(marker_id, marker_size)
        self.rectifier = DockRectifier(image_mode)
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.halt, self.paused = threading.Event(), threading.Event()
        self.thread = None
        self.result = None
        self.message = ''
        self.frame = self.info = self.scan = self.odom = self.clock = None
        self.frame_at = self.scan_at = self.odom_at = self.clock_at = 0.
        self.nav_active = False
        self.nav_received = False
        self.last_command = 0.
        self.leased = False
        self.buffer = Buffer(cache_time=Duration(seconds=5))
        self.listener = TransformListener(self.buffer, node, spin_thread=False)
        self.pub = node.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = node.create_publisher(String, '/bt/aruco_docking/status', 10)
        self.preview_pub = node.create_publisher(Image, '/bt/aruco_docking/debug_image', 1)
        self.group = ReentrantCallbackGroup()
        def sub(cls, topic, callback):
            return node.create_subscription(cls, topic, callback, qos_profile_sensor_data,
                                            callback_group=self.group)
        self.subs = [sub(Image, image_topic, self.on_image), sub(CameraInfo, camera_info_topic, self.on_info),
                     sub(LaserScan, scan_topic, self.on_scan), sub(Odometry, odom_topic, self.on_odom),
                     sub(Clock, '/clock', self.on_clock),
                     node.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.on_nav,
                                              QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                                              callback_group=self.group)]
        self.watchdog = node.create_timer(.05, self.watch, callback_group=self.group)

    def on_image(self, message):
        with self.lock:
            if self.frame and message.header.stamp == self.frame.header.stamp:
                return
            self.frame, self.frame_at = message, time.monotonic()

    def on_info(self, message):
        with self.lock: self.info = message

    def on_scan(self, message):
        with self.lock: self.scan, self.scan_at = message, time.monotonic()

    def on_odom(self, message):
        with self.lock: self.odom, self.odom_at = message, time.monotonic()

    def on_clock(self, message):
        with self.lock: self.clock, self.clock_at = stamp_seconds(message.clock), time.monotonic()

    def on_nav(self, message):
        with self.lock:
            self.nav_active = any(s.status in (1,2,3) for s in message.status_list)
            self.nav_received = True

    def source_fresh(self, stamp):
        seconds = stamp_seconds(stamp)
        # Sim image stamps require a progressing /clock; do not compare to host wall time.
        if self.clock is not None:
            if time.monotonic()-self.clock_at > .5:
                return False
            age = self.clock-seconds
        else:
            age = time.time()-seconds
        return -.05 <= age <= .5

    def publish(self, command):
        # Stop/Pause and velocity publication share this lock: no command after Stop.
        with self.lock:
            if not self.leased:
                return
            if self.halt.is_set() or self.paused.is_set() or self.nav_active:
                command = (0.,0.,0.)
            message = Twist()
            message.linear.x, message.linear.y, message.angular.z = map(float, command)
            self.pub.publish(message)
            self.last_command = time.monotonic()

    def watch(self):
        with self.lock:
            if self.leased and (time.monotonic()-self.last_command > .25
                                or time.monotonic()-self.frame_at > .5 or self.nav_active):
                self.publish((0.,0.,0.))

    def status(self, phase, detail='', error=None):
        payload = dict(marker_id=self.detector.marker_id, phase=phase, detail=detail)
        if error is not None: payload['error'] = list(map(float, error))
        self.status_pub.publish(String(data=json.dumps(payload)))

    def start(self):
        if not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another ArUco docking action owns the base')
        self.leased = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def tf(self, frame, stamp):
        if not frame:
            raise ValueError('Missing sensor frame_id')
        return self.buffer.lookup_transform(self.base_frame, frame, Time.from_msg(stamp),
                                            timeout=Duration(seconds=.12)).transform

    def scan_points(self, scan):
        if not self.source_fresh(scan.header.stamp):
            raise ValueError('Laser scan timestamp is stale')
        transform = self.tf(scan.header.frame_id, scan.header.stamp)
        distances = np.asarray(scan.ranges, dtype=float)
        valid = np.isfinite(distances) & (distances >= scan.range_min) & (distances <= scan.range_max)
        # Infinite returns mean free space; an all-NaN/zero scan is invalid.
        if not np.any(valid) and not np.any(np.isposinf(distances)):
            raise ValueError('No valid laser returns')
        angles = scan.angle_min + np.arange(len(distances))*scan.angle_increment
        from .docking_vision import rotation_matrix
        points = np.vstack((distances[valid]*np.cos(angles[valid]),
                            distances[valid]*np.sin(angles[valid]), np.zeros(valid.sum())))
        p = transform.translation
        points = rotation_matrix(transform.rotation) @ points + np.array([[p.x],[p.y],[p.z]])
        return points[:2].T

    def analyze(self, frame, info, scan):
        if (frame.width,frame.height) != (info.width,info.height) or frame.header.frame_id != info.header.frame_id:
            raise ValueError('Image and CameraInfo dimensions/frame do not match')
        if not self.source_fresh(frame.header.stamp):
            raise ValueError('Image timestamp is stale')
        gray = self.bridge.imgmsg_to_cv2(frame, desired_encoding='mono8')
        gray, intrinsic, optical_from_rectified = self.rectifier.prepare(gray, info)
        detection = self.detector.detect(gray, intrinsic, np.zeros(5))
        if detection is None:
            raise ValueError(f'Marker {self.detector.marker_id} not visible')
        position, normal, pixels, rms = detection
        position = optical_from_rectified @ position
        normal = optical_from_rectified @ normal
        position, normal = transform_pose(self.tf(frame.header.frame_id, frame.header.stamp), position, normal)
        error = dock_error(position, normal, self.config.distance)
        command, settled = velocity(error, self.config)
        if obstacle_in_motion(self.scan_points(scan), command):
            raise ValueError('Obstacle inside docking clearance')
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.polylines(preview, [pixels.astype(np.int32)], True, (0,255,0), 2)
        cv2.putText(preview, f'ID {self.detector.marker_id}  xy={math.hypot(*error[:2]):.3f}m yaw={error[2]:.3f}',
                    (12,30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)
        message = self.bridge.cv2_to_imgmsg(preview, encoding='bgr8'); message.header = frame.header
        self.preview_pub.publish(message)
        return error, command, settled

    def run(self):
        started = time.monotonic()
        last_good, stable_at, stable_count, previous = started, None, 0, None
        last_status = 0.
        self.status('acquiring')
        try:
            while not self.halt.wait(.05):
                now = time.monotonic()
                if self.paused.is_set():
                    self.publish((0.,0.,0.)); started += now-last_loop if 'last_loop' in locals() else 0
                    last_good, stable_at, stable_count, previous = now, None, 0, None
                    last_loop = now
                    continue
                last_loop = now
                if now-started > self.config.timeout:
                    raise TimeoutError('Docking timed out: '+self.message)
                with self.lock:
                    frame, info, scan, odom = self.frame, self.info, self.scan, self.odom
                    ages = [now-self.frame_at, now-self.scan_at, now-self.odom_at]
                    nav_active = self.nav_active
                if nav_active:
                    raise RuntimeError('Navigation goal active; stop Navigation before docking')
                try:
                    if self.node.count_publishers('/navigate_to_pose/_action/status') and not self.nav_received:
                        raise ValueError('Waiting for latched Navigation status')
                    if frame is None or info is None or scan is None or odom is None or max(ages) > .5:
                        raise ValueError('Waiting for fresh image, CameraInfo, scan and odometry')
                    if not self.source_fresh(odom.header.stamp):
                        raise ValueError('Odometry timestamp is stale')
                    key = (frame.header.stamp.sec,frame.header.stamp.nanosec)
                    if key == previous:
                        if now-last_good > .25: self.publish((0.,0.,0.))
                        continue
                    previous = key
                    error, command, settled = self.analyze(frame, info, scan)
                    # Never publish a command based on data that became stale during inference.
                    if not self.source_fresh(frame.header.stamp):
                        raise ValueError('Frame expired during detection')
                    last_good = time.monotonic()
                    self.message = ''
                    self.publish(command)
                    v = odom.twist.twist
                    slow = math.hypot(v.linear.x,v.linear.y) < .025 and abs(v.angular.z) < .025
                    if settled and slow:
                        stable_at = stable_at if stable_at is not None else now
                        stable_count += 1
                        if now-stable_at >= self.config.stable_seconds and stable_count >= 5:
                            self.result, self.message = True, 'Docked and stationary'
                            self.status('succeeded', self.message, error)
                            return
                    else: stable_at, stable_count = None, 0
                    if now-last_status > .25:
                        self.status('settling' if settled else 'docking', error=error); last_status = now
                except (ValueError, LookupError) as exc:
                    self.message = str(exc); self.publish((0.,0.,0.)); stable_at, stable_count = None, 0
                except Exception as exc:
                    # TF may arrive after the camera callback; retry only until the loss deadline.
                    self.message = str(exc); self.publish((0.,0.,0.)); stable_at, stable_count = None, 0
                if now-last_good > self.config.lost_timeout:
                    raise RuntimeError('Docking feedback unavailable: '+self.message)
                if now-last_status > .5:
                    self.status('holding', self.message); last_status = now
        except Exception as exc:
            self.result, self.message = False, str(exc)
            self.status('failed', self.message)
            self.node.get_logger().error('ArucoDock: '+self.message)
        finally:
            self.publish((0.,0.,0.))
            with self.lock:
                if self.leased: self.leased = False; _LEASE.release()

    def pause(self):
        if self.thread is None or self.result is not None: return
        with self.lock: self.paused.set(); self.publish((0.,0.,0.))
        started = time.monotonic()
        while time.monotonic()-started < 3.:
            with self.lock:
                if self.odom and self.odom_at > started:
                    v = self.odom.twist.twist
                    if math.hypot(v.linear.x,v.linear.y) < .025 and abs(v.angular.z) < .025: return
            if self.halt.wait(.03): break
        raise RuntimeError('Docking pause was not confirmed by odometry')

    def resume(self):
        with self.lock: self.paused.clear()

    def close(self):
        with self.lock: self.halt.set(); self.publish((0.,0.,0.))
        if self.thread: self.thread.join(timeout=3.)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Docking worker did not stop')
        self.node.destroy_timer(self.watchdog)
        for sub in self.subs: self.node.destroy_subscription(sub)
        self.listener.unregister()
        for pub in (self.pub,self.status_pub,self.preview_pub): self.node.destroy_publisher(pub)
