# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0.

"""Episode-scoped RLT sidecars, using the existing recording lifecycle."""

import io
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid

import numpy as np


def copy_rlt_sidecar(source, target):
    """Read-back verified copy before archiving/conversion may delete source."""
    source, target = Path(source), Path(target)
    if not source.is_dir():
        return False
    shutil.copytree(source, target, dirs_exist_ok=True)
    for original in source.rglob('*'):
        if original.is_file():
            if original.read_bytes() != (target / original.relative_to(source)).read_bytes():
                raise RuntimeError(f'RLT sidecar copy verification failed: {original}')
    return True


class RLTRecorder:
    def __init__(self, node):
        from std_msgs.msg import String, UInt8MultiArray

        self._node = node
        self._string_type = String
        self._lock = threading.Lock()
        self._recording_id = ""
        self._thread = None
        self._queue = None
        self._dropped = 0
        self._error = ""
        self._publisher = node.create_publisher(String, '/inference/rlt_recording', 10)
        self._subscription = node.create_subscription(
            UInt8MultiArray, '/inference/rlt_trace', self._receive, 128,
            callback_group=node.io_callback_group,
        )

    def publish_state(self):
        with self._lock:
            recording_id = self._recording_id
        message = self._string_type()
        message.data = json.dumps({
            'recording_id': recording_id,
            'valid_until_ns': time.time_ns() + 1_000_000_000,
        })
        self._publisher.publish(message)

    def start_episode(self, episode_dir):
        self.stop_episode()
        with self._lock:
            self._recording_id = uuid.uuid4().hex
            self._dropped = 0
            self._error = ""
            self._queue = queue.Queue(maxsize=128)
            self._thread = threading.Thread(
                target=self._write, args=(Path(episode_dir), self._recording_id, self._queue),
                daemon=True,
            )
            self._thread.start()
        self.publish_state()

    def _receive(self, message):
        with self._lock:
            if not self._recording_id:
                return
            if len(message.data) > 4 * 1024 * 1024:
                self._dropped += 1
                return
            try:
                self._queue.put_nowait(bytes(message.data))
            except queue.Full:
                self._dropped += 1

    def _write(self, episode_dir, recording_id, pending):
        root = episode_dir / 'rlt'
        count = 0
        while True:
            payload = pending.get()
            if payload is None:
                break
            try:
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    metadata = json.loads(archive['metadata'].tobytes())
                if metadata.get('recording_id') != recording_id:
                    continue
                if metadata.get('format') != 'cyclo.rlt.recording/v1':
                    raise ValueError('Unknown RLT trace format')
                root.mkdir(exist_ok=True)
                # Recorder chooses the filename; no path from the transport is used.
                name = f'{recording_id}_{count:06d}.npz'
                with (root / name).open('xb') as stream:
                    stream.write(payload)
                with (root / 'events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(metadata, file=name)) + '\n')
                count += 1
            except Exception as exc:
                self._error = str(exc)
                self._node.get_logger().error(f'RLT sidecar write failed: {exc}')
        if root.exists():
            try:
                (root / f'{recording_id}_summary.json').write_text(json.dumps({
                    'format': 'cyclo.rlt.recording_summary/v1',
                    'recording_id': recording_id, 'records': count,
                    'recorder_dropped': self._dropped, 'error': self._error,
                    'execution_verified': False,
                }), encoding='utf-8')
            except OSError as exc:
                self._error = str(exc)
                self._node.get_logger().error(f'RLT summary write failed: {exc}')

    def stop_episode(self):
        with self._lock:
            self._recording_id = ""
            worker, pending = self._thread, self._queue
        self.publish_state()
        if worker is not None:
            pending.put(None, timeout=5.0)
            worker.join(timeout=10.0)
            if worker.is_alive():
                raise RuntimeError('RLT recorder still writing; episode cannot be removed yet')
            self._thread = None

    def recording_warnings(self):
        return ([f'RLT recording dropped {self._dropped} packets'] if self._dropped else []) + (
            [f'RLT recording: {self._error}'] if self._error else []
        )

    def close(self):
        self.stop_episode()
        self._node.destroy_subscription(self._subscription)
        self._node.destroy_publisher(self._publisher)
