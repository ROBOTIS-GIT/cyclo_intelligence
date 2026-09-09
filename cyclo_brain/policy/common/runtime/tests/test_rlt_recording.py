"""Real publisher/recorder queues with an in-memory ROS/Zenoh transport."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'cyclo_brain/policy/common/runtime'))
sys.path.insert(0, str(REPO / 'cyclo_data'))
from rlt_recording import RLTTracePublisher
from cyclo_data.recorder.rlt_recorder import RLTRecorder


class RLTRecordingTests(unittest.TestCase):
    def setUp(self):
        self.callbacks = {}
        self.errors = []

        def subscribe(topic, callback, **_kwargs):
            self.callbacks.setdefault(topic, []).append(callback)
            return SimpleNamespace(close=lambda: self.callbacks[topic].remove(callback))

        def send(topic, message):
            for callback in tuple(self.callbacks.get(topic, ())):
                callback(message)

        sdk = ModuleType('zenoh_ros2_sdk')
        sdk.ROS2Subscriber = subscribe
        sdk.ROS2Publisher = lambda topic, **_kw: SimpleNamespace(
            publish=lambda **kwargs: send(topic, SimpleNamespace(**kwargs)), close=lambda: None,
        )
        sdk.get_message_class = lambda _name: SimpleNamespace
        messages = ModuleType('std_msgs.msg')
        messages.String = messages.UInt8MultiArray = SimpleNamespace
        self.modules = patch.dict(sys.modules, {'zenoh_ros2_sdk': sdk, 'std_msgs.msg': messages})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.node = SimpleNamespace(
            io_callback_group=None,
            create_publisher=lambda _type, topic, _qos: SimpleNamespace(publish=lambda msg: send(topic, msg)),
            create_subscription=lambda _type, topic, cb, _qos, **kw: subscribe(topic, cb),
            destroy_subscription=lambda sub: sub.close(), destroy_publisher=lambda pub: None,
            get_logger=lambda: SimpleNamespace(error=self.errors.append),
        )

    def test_round_trip_numeric_tensors_and_recording_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder, publisher = RLTRecorder(self.node), RLTTracePublisher()
            self.addCleanup(recorder.close)
            publisher.submit({'event': 'inference'}, {'z_rl': np.ones((1, 4))})
            self.assertEqual(publisher.recording_id, '')
            recorder.start_episode(root)
            recording_id = publisher.recording_id
            token = np.arange(32, dtype=np.float32).reshape(1, 32)
            publisher.submit({'event': 'inference', 'request_seq': 7}, {'z_rl': token})
            publisher.submit({'event': 'buffer_accepted', 'request_seq': 7})
            publisher.close()  # drain transport before finalizing recorder
            recorder.stop_episode()
            rows = [json.loads(row) for row in (root / 'rlt/events.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['inference', 'buffer_accepted'])
            self.assertTrue(all(row['recording_id'] == recording_id for row in rows))
            with np.load(root / 'rlt' / rows[0]['file'], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['z_rl'], token)
            summary = json.loads(next((root / 'rlt').glob('*_summary.json')).read_text())
            self.assertEqual(summary['records'], 2)
            self.assertFalse(summary['execution_verified'])
            recorder._receive(SimpleNamespace(data=b'late invalid data'))
            self.assertEqual(len(list((root / 'rlt').glob('*.npz'))), 2)

    def test_old_episode_packet_cannot_enter_new_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / 'first', root / 'second'
            first.mkdir()
            second.mkdir()
            recorder, publisher = RLTRecorder(self.node), RLTTracePublisher()
            self.addCleanup(recorder.close)
            recorder.start_episode(first)
            old_id = publisher.recording_id
            recorder.stop_episode()
            recorder.start_episode(second)
            publisher.submit({'event': 'inference'}, recording_id=old_id)
            publisher.close()
            recorder.stop_episode()
            self.assertFalse((second / 'rlt').exists())

    def test_corrupt_packet_is_reported_without_breaking_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = RLTRecorder(self.node)
            self.addCleanup(recorder.close)
            recorder.start_episode(Path(temporary))
            recorder._receive(SimpleNamespace(data=b'not an npz'))
            recorder.stop_episode()
            self.assertTrue(recorder.recording_warnings())
            self.assertTrue(self.errors)

    def test_expired_recording_heartbeat_disables_capture(self):
        publisher = RLTTracePublisher()
        self.addCleanup(publisher.close)
        publisher._set_session(SimpleNamespace(data=json.dumps({
            'recording_id': 'old', 'valid_until_ns': 1,
        })))
        self.assertEqual(publisher.recording_id, '')


if __name__ == '__main__':
    unittest.main()
