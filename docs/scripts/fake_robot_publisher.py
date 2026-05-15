"""Fake observation publisher for ffw_sg2_rev1 inference smoke test.

Publishes the 4 RGB cameras + /joint_states + /odom at 10 Hz with black images
and zero joint positions. Just enough to fill the Engine process RobotClient
buffer when Main requests GET_ACTION.
"""
import sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState
from nav_msgs.msg import Odometry


CAMERAS = [
    "/zed/zed_node/left/image_rect_color/compressed",
    "/zed/zed_node/right/image_rect_color/compressed",
    "/camera_left/camera_left/color/image_rect_raw/compressed",
    "/camera_right/camera_right/color/image_rect_raw/compressed",
]
JOINT_NAMES = [
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
    "head_joint1", "head_joint2", "lift_joint",
]


class FakeRobot(Node):
    def __init__(self):
        super().__init__("fake_robot_publisher")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Encode a single black 240x320 JPEG once and reuse.
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "FAKE", (60, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._jpeg = enc.tobytes()

        self._cam_pubs = {
            t: self.create_publisher(CompressedImage, t, qos)
            for t in CAMERAS
        }
        self._joint_pub = self.create_publisher(JointState, "/joint_states", qos)
        self._odom_pub = self.create_publisher(Odometry, "/odom", qos)

        self._timer = self.create_timer(0.1, self._tick)  # 10 Hz
        self.get_logger().info(
            f"Publishing 4 cams + /joint_states + /odom at 10Hz "
            f"(jpeg bytes={len(self._jpeg)})"
        )
        self._tick_count = 0

    def _tick(self):
        now = self.get_clock().now().to_msg()
        for topic, pub in self._cam_pubs.items():
            m = CompressedImage()
            m.header.stamp = now
            m.format = "jpeg"
            m.data = self._jpeg
            pub.publish(m)

        js = JointState()
        js.header.stamp = now
        js.header.frame_id = ""
        js.name = JOINT_NAMES
        js.position = [0.0] * len(JOINT_NAMES)
        js.velocity = [0.0] * len(JOINT_NAMES)
        js.effort = [0.0] * len(JOINT_NAMES)
        self._joint_pub.publish(js)

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        # explicitly fill pose / twist (zeros) to avoid empty CDR fields
        od.pose.pose.position.x = 0.0
        od.pose.pose.position.y = 0.0
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.x = 0.0
        od.pose.pose.orientation.y = 0.0
        od.pose.pose.orientation.z = 0.0
        od.pose.pose.orientation.w = 1.0
        od.pose.covariance = [0.0] * 36
        od.twist.covariance = [0.0] * 36
        self._odom_pub.publish(od)

        self._tick_count += 1
        if self._tick_count % 50 == 0:
            self.get_logger().info(f"published {self._tick_count} ticks")


def main():
    rclpy.init()
    n = FakeRobot()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
