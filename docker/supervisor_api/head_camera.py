"""A single ROS compressed-image subscription, retaining only the latest JPEG."""
import base64
import logging
import threading
import time

logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 6 * 1024 * 1024


class CameraUnavailable(RuntimeError):
    pass


class HeadCameraCache:
    def __init__(self, topic, max_age=2.0):
        self.topic = topic
        self.max_age = max_age
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._error = None

    def start(self):
        with self._condition:
            if self._stop.is_set():
                raise CameraUnavailable('Head camera subscriber is shutting down')
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name='head-camera-cache', daemon=True)
                self._thread.start()

    def close(self):
        self._stop.set()
        with self._condition:
            self._frame = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _receive(self, message):
        if len(message.data) > MAX_IMAGE_BYTES:
            self._reject('Head camera JPEG exceeds 6 MiB')
            return
        data = bytes(message.data)
        if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
            self._reject('Head camera must publish JPEG CompressedImage messages')
            return
        stamp = f'{message.header.stamp.sec}.{message.header.stamp.nanosec:09d}'
        with self._condition:
            if self._stop.is_set():
                return
            # Repeated delivery of a frozen frame must not make it fresh again.
            if self._frame is not None and self._frame[1] == stamp:
                return
            self._frame = (data, stamp, time.time(), time.monotonic())
            self._error = None
            self._condition.notify_all()

    def _reject(self, error):
        with self._condition:
            self._frame = None
            self._error = error
            self._condition.notify_all()

    def snapshot(self, timeout=2.0):
        self.start()
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                frame = self._frame
                if self._stop.is_set():
                    raise CameraUnavailable('Head camera subscriber is shutting down')
                if frame is not None and time.monotonic() - frame[3] <= self.max_age:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraUnavailable(self._error or f'No fresh head camera frame on {self.topic}. Check the camera publisher and ROS connection.')
                self._condition.wait(remaining)
        return {'image_base64': base64.b64encode(frame[0]).decode(), 'frame_stamp': frame[1],
                'captured_at': frame[2], 'camera_topic': self.topic}

    def _run(self):
        while not self._stop.is_set():
            try:
                self._spin()
            except Exception as exc:
                logger.exception('Head camera ROS subscriber failed')
                self._reject(f'Head camera ROS subscription failed ({type(exc).__name__})')
            if not self._stop.is_set():
                self._reject('Head camera ROS subscription disconnected; reconnecting')
                self._stop.wait(2.0)

    def _subscription_stalled(self, started):
        with self._condition:
            last_fresh = self._frame[3] if self._frame is not None else started
        # Give a new subscription time to discover the publisher. Repeated old
        # timestamps still count as stalled and never become fresh on reconnect.
        return time.monotonic() - max(started, last_fresh) >= max(5.0, self.max_age * 2)

    def _spin(self):
        # Private context: do not initialize/shut down the navigation subscriber's
        # global rclpy context. ROS imports stay lazy for non-ROS API/test hosts.
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import CompressedImage

        context = Context()
        node = executor = None
        try:
            context.init(args=[])
            node = Node('cyclo_reasoning_head_camera', context=context)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            node.create_subscription(CompressedImage, self.topic, self._receive, qos)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            started = time.monotonic()
            while context.ok() and not self._stop.is_set():
                executor.spin_once(timeout_sec=0.2)
                if self._subscription_stalled(started):
                    logger.warning("Head camera received no fresh frame; recreating ROS subscription")
                    break
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            context.try_shutdown()
