# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0.

"""Best-effort RLT telemetry; disk/network work never runs in the control tick.

RecordingService owns episode activation. Payloads are numeric NPZ archives,
not pickle, and carry the recording ID so late packets cannot enter a new episode.
"""

import io
import json
import logging
import queue
import threading
import time
import uuid

import numpy as np


class RLTTracePublisher:
    def __init__(self):
        from zenoh_ros2_sdk import ROS2Publisher, ROS2Subscriber, get_message_class

        self._session = {}
        self._producer = uuid.uuid4().hex
        self._queue = queue.Queue(maxsize=128)
        self._closed = False
        self._dropped = 0
        self._publisher = ROS2Publisher(
            topic="/inference/rlt_trace", msg_type="std_msgs/msg/UInt8MultiArray",
        )
        self._layout = get_message_class('std_msgs/msg/MultiArrayLayout')(
            dim=[], data_offset=0,
        )
        self._subscriber = ROS2Subscriber(
            topic="/inference/rlt_recording", msg_type="std_msgs/msg/String",
            callback=self._set_session,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _set_session(self, message):
        try:
            session = json.loads(message.data)
            self._session = session if (
                isinstance(session, dict)
                and isinstance(session.get('recording_id'), str)
                and type(session.get('valid_until_ns')) is int
            ) else {}
        except (ValueError, TypeError):
            self._session = {}

    @property
    def recording_id(self):
        session = self._session
        if self._closed or time.time_ns() >= session.get("valid_until_ns", 0):
            return ""
        return session.get("recording_id", "")

    def submit(self, metadata, arrays=None, *, recording_id=None):
        recording_id = self.recording_id if recording_id is None else recording_id
        if not recording_id or self._closed:
            return
        metadata = dict(
            metadata, format="cyclo.rlt.recording/v1", recording_id=recording_id,
            producer_id=self._producer, publisher_dropped=self._dropped,
        )
        try:
            self._queue.put_nowait((metadata, arrays or {}))
        except queue.Full:
            self._dropped += 1
            logging.getLogger(__name__).warning("RLT recording queue full; sample dropped")

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            metadata, arrays = item
            try:
                with io.BytesIO() as buffer:
                    np.savez(buffer, metadata=np.frombuffer(
                        json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8,
                    ), **arrays)
                    self._publisher.publish(
                        layout=self._layout,
                        data=np.frombuffer(buffer.getvalue(), dtype=np.uint8),
                    )
            except Exception:
                self._dropped += 1
                logging.getLogger(__name__).exception("RLT telemetry publish failed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._subscriber.close()
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            logging.getLogger(__name__).error("RLT telemetry worker did not drain")
            return
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._publisher.close()


def create_rlt_trace_publisher():
    try:
        return RLTTracePublisher()
    except Exception:
        logging.getLogger(__name__).exception("RLT recording transport unavailable")
        return None
