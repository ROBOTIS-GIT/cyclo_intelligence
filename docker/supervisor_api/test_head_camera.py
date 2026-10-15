import base64
import threading
import types
import unittest
from unittest.mock import patch
from head_camera import HeadCameraCache, CameraUnavailable, MAX_IMAGE_BYTES


def message(sec=1, data=b'\xff\xd8test\xff\xd9'):
    return types.SimpleNamespace(data=data, header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=sec,nanosec=123)))


class CameraCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache=HeadCameraCache('/head/compressed')
        self.addCleanup(self.cache.close)

    def test_latest_frame_replaces_previous_without_http(self):
        self.cache._receive(message(1)); self.cache._receive(message(2))
        with patch.object(self.cache,'start'), patch('urllib.request.urlopen',side_effect=AssertionError('HTTP used')):
            result=self.cache.snapshot(timeout=0)
        self.assertEqual(result['frame_stamp'],'2.000000123')
        self.assertEqual(base64.b64decode(result['image_base64']),message().data)
        self.assertEqual(result['camera_topic'],'/head/compressed')

    def test_frozen_frame_is_not_refreshed_and_new_frame_recovers(self):
        with patch('head_camera.time.monotonic',return_value=10):self.cache._receive(message())
        with patch('head_camera.time.monotonic',return_value=20),patch.object(self.cache,'start'):
            self.cache._receive(message())
            with self.assertRaisesRegex(CameraUnavailable,'No fresh'):self.cache.snapshot(timeout=0)
            self.cache._receive(message(2))
            self.assertEqual(self.cache.snapshot(timeout=0)['frame_stamp'],'2.000000123')

    def test_invalid_image_clears_previous_frame(self):
        for data in (b'png', b'x'*(MAX_IMAGE_BYTES+1)):
            self.cache._receive(message());self.cache._receive(message(2,data))
            with patch.object(self.cache,'start'),self.assertRaises(CameraUnavailable):
                self.cache.snapshot(timeout=0)

    def test_missing_camera_times_out(self):
        with patch.object(self.cache,'start'),self.assertRaisesRegex(CameraUnavailable,'No fresh'):
            self.cache.snapshot(timeout=0)

    def test_only_one_subscription_thread_for_concurrent_requests(self):
        entered=threading.Event()
        def spin():
            entered.set();self.cache._stop.wait(2)
        with patch.object(self.cache,'_spin',side_effect=spin) as worker:
            threads=[threading.Thread(target=self.cache.start) for _ in range(8)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(1)
            self.assertTrue(entered.wait(1));worker.assert_called_once()
            self.cache.close()
        self.assertFalse(self.cache._thread.is_alive())

    def test_close_wakes_waiting_capture(self):
        entered=threading.Event();errors=[]
        def snapshot():
            entered.set()
            try:self.cache.snapshot(timeout=10)
            except CameraUnavailable as e:errors.append(str(e))
        with patch.object(self.cache,'start'):
            thread=threading.Thread(target=snapshot);thread.start()
            self.assertTrue(entered.wait(1));self.cache.close();thread.join(1)
        self.assertFalse(thread.is_alive());self.assertTrue(errors)

    def test_stalled_subscription_reconnects_but_duplicate_does_not_refresh(self):
        with patch('head_camera.time.monotonic',return_value=10):
            self.cache._receive(message())
        with patch('head_camera.time.monotonic',return_value=16):
            self.cache._receive(message())
            self.assertTrue(self.cache._subscription_stalled(10))
            self.assertFalse(self.cache._subscription_stalled(15))
            self.cache._receive(message(2))
            self.assertFalse(self.cache._subscription_stalled(10))

    def test_initial_subscription_discovery_has_grace_period(self):
        with patch('head_camera.time.monotonic',return_value=14):
            self.assertFalse(self.cache._subscription_stalled(10))
        with patch('head_camera.time.monotonic',return_value=15):
            self.assertTrue(self.cache._subscription_stalled(10))

    def test_clock_reset_accepts_new_stamp(self):
        self.cache._receive(message(100));self.cache._receive(message(1))
        with patch.object(self.cache,'start'):
            self.assertEqual(self.cache.snapshot(timeout=0)['frame_stamp'],'1.000000123')

if __name__=='__main__':unittest.main()
