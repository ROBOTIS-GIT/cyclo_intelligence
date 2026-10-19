"""Short, odometry-measured retreat with the docking transport's stop/watchdog.

No image detection is used. Motion requires fresh scan/odometry and uses the
same base lease and footprint guard as docking. Navigation starts only after
stationary odometry has been observed.
"""
import math
import time
import numpy as np
from .docking_runtime import DockRuntime, _LEASE
from .docking_core import DockConfig
from .docking_vision import rotation_matrix


def retreat_progress(start, current):
    x,y,yaw = start
    dx,dy = current[0]-x,current[1]-y
    heading = math.atan2(math.sin(current[2]-yaw),math.cos(current[2]-yaw))
    return -(dx*math.cos(yaw)+dy*math.sin(yaw)), -dx*math.sin(yaw)+dy*math.cos(yaw), heading


class DockRetreatRuntime(DockRuntime):
    def __init__(self, node, distance, speed, timeout, scan_topic, odom_topic, cmd_vel_topic, base_frame, footprint_profile):
        super().__init__(node, 0, .18, DockConfig(timeout=timeout),
                         '/zed/left/image_raw', '/zed/left/camera_info',
                         cmd_vel_topic, scan_topic, odom_topic, base_frame,
                         footprint_profile=footprint_profile)
        self.distance, self.speed = distance, speed
        self.diagnostics.context.update(action='DockRetreat', retreat_distance_m=distance,
                                        footprint_profile=footprint_profile)

    def watch(self):
        with self.lock:
            if self.leased and (time.monotonic()-self.last_command > .25
                or time.monotonic()-self.scan_at > .5 or time.monotonic()-self.odom_at > .5
                or self.nav_active): self.publish((0.,0.,0.))

    @staticmethod
    def pose(odom):
        p,q = odom.pose.pose.position,odom.pose.pose.orientation
        rotation = rotation_matrix(q)
        if not all(math.isfinite(v) for v in (p.x,p.y)):
            raise ValueError('Invalid odometry position')
        return p.x,p.y,math.atan2(rotation[1,0],rotation[0,0])

    def rear_visible(self, scan):
        transform = self.tf(scan.header.frame_id, scan.header.stamp)
        rotation = rotation_matrix(transform.rotation)
        angles = scan.angle_min+np.arange(len(scan.ranges))*scan.angle_increment
        rays = np.column_stack((np.cos(angles),np.sin(angles),np.zeros(len(angles)))) @ rotation.T
        rear = (rays[:,0] < -.7) & (np.abs(rays[:,1]) < .71)
        ranges = np.asarray(scan.ranges)
        valid = np.isposinf(ranges) | (np.isfinite(ranges)&(ranges>=scan.range_min)&(ranges<=scan.range_max))
        if rear.sum() < 5 or not np.all(valid[rear]):
            raise ValueError('Rear scan coverage unavailable')

    def run(self):
        started=last=time.monotonic(); origin=previous=frame_id=None
        last_good=started; settled_at=None
        try:
            while not self.halt.wait(.05):
                now=time.monotonic()
                if self.paused.is_set():
                    self.publish((0.,0.,0.));started+=now-last;last_good=now;last=now;continue
                last=now
                if now-started>self.config.timeout:raise TimeoutError('DockRetreat timed out')
                with self.lock:
                    scan,odom=self.scan,self.odom
                    ages=(now-self.scan_at,now-self.odom_at)
                    nav_active=self.nav_active
                if nav_active:raise RuntimeError('Navigation goal active during retreat')
                try:
                    if scan is None or odom is None or max(ages)>.5 or not self.source_fresh(odom.header.stamp):
                        raise ValueError('Waiting for fresh scan and odometry')
                    if self.node.count_publishers('/navigate_to_pose/_action/status') and not self.nav_received:
                        raise ValueError('Waiting for Navigation status')
                    current=self.pose(odom)
                    if origin is None:origin=current;frame_id=odom.header.frame_id
                    if odom.header.frame_id!=frame_id:raise RuntimeError('Odometry frame changed')
                    if previous and math.hypot(current[0]-previous[0],current[1]-previous[1])>.10:
                        raise RuntimeError('Odometry jumped during retreat')
                    previous=current
                    progress,lateral,heading=retreat_progress(origin,current)
                    if abs(lateral)>.05 or abs(heading)>.15:raise RuntimeError('Retreat deviated from straight path')
                    self.rear_visible(scan)
                    done=progress>=self.distance-.005
                    command=(0.,0.,0.) if done else (-self.speed,0.,0.)
                    self.check_obstacles(scan,command)
                    if not self.source_fresh(odom.header.stamp):raise ValueError('Odometry expired')
                    self.publish(command);last_good=now
                    v=odom.twist.twist
                    if done and math.hypot(v.linear.x,v.linear.y)<.015 and abs(v.angular.z)<.02:
                        settled_at=settled_at or now
                        if now-settled_at>=.3:
                            self.result=True;self.message='Retreated and stationary';return
                    else:settled_at=None
                except ValueError as exc:
                    self.message=str(exc);self.publish((0.,0.,0.));settled_at=None
                    if now-last_good>3:raise RuntimeError(self.message)
        except Exception as exc:
            self.result=False;self.message=str(exc);self.publish((0.,0.,0.))
            self.node.get_logger().error('DockRetreat: '+self.message)
            self.diagnostics.save_async(self.message,self.node.get_logger())
        finally:
            self.publish((0.,0.,0.))
            with self.lock:
                if self.leased:self.leased=False;_LEASE.release()
