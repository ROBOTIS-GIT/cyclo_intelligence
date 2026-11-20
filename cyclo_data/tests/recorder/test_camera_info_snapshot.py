from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("rclpy")
from sensor_msgs.msg import CameraInfo

from cyclo_data.recorder import camera_info_snapshot
from cyclo_data.recorder.camera_info_snapshot import CameraInfoSnapshot


def _snapshot(topics):
    callbacks = {}
    node = Mock()

    def subscribe(msg_type, topic, callback, qos, **kwargs):
        callbacks[topic] = callback
        return object()

    node.create_subscription.side_effect = subscribe
    return CameraInfoSnapshot(node, topics), callbacks


def test_stop_wakes_for_last_camera_and_keeps_latest_across_episodes(tmp_path, monkeypatch):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    waiting = Event()
    wait = recorder._cache_ready.wait

    def signal_wait(timeout):
        waiting.set()
        return wait(timeout)

    monkeypatch.setattr(recorder._cache_ready, "wait", signal_wait)
    recorder.start_episode(tmp_path / "0")
    publish["/left"](CameraInfo(width=100))
    with ThreadPoolExecutor(max_workers=1) as pool:
        saved = pool.submit(recorder.stop_episode)
        assert waiting.wait(2)
        publish["/left"](CameraInfo(width=200))
        publish["/right"](CameraInfo(width=300))
        outputs = saved.result(timeout=2)
    assert {name: yaml.safe_load(path.read_text())["width"] for name, path in outputs.items()} == {
        "left": 200, "right": 300,
    }
    recorder.start_episode(tmp_path / "1")
    waiting.clear()
    again = recorder.stop_episode()
    assert not waiting.is_set()
    assert {name: path.read_bytes() for name, path in again.items()} == {
        name: path.read_bytes() for name, path in outputs.items()}


def test_timeout_writes_only_available_calibration_and_late_callback_fills_next_episode(
        tmp_path, monkeypatch):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    monkeypatch.setattr(recorder, "_STOP_WAIT_TIMEOUT_SEC", 0.02)
    publish["/left"](CameraInfo(width=400))
    recorder.start_episode(tmp_path / "0")
    assert set(recorder.stop_episode()) == {"left"}
    assert not (tmp_path / "0/camera_info/right.yaml").exists()
    publish["/right"](CameraInfo(width=500))
    recorder.start_episode(tmp_path / "1")
    assert set(recorder.stop_episode()) == {"left", "right"}


def test_old_subscription_cannot_replace_new_robot_calibration(tmp_path):
    recorder, publish = _snapshot({"head": "/old"})
    old_callback = publish["/old"]
    old_callback(CameraInfo(width=10))
    recorder.reconfigure({"head": "/new"})
    recorder.start_episode(tmp_path)
    publish["/new"](CameraInfo(width=20))
    old_callback(CameraInfo(width=30))
    assert yaml.safe_load(recorder.stop_episode()["head"].read_text())["width"] == 20


def test_high_rate_updates_only_convert_saved_snapshot_outside_lock(tmp_path, monkeypatch):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    convert = camera_info_snapshot._camera_info_to_dict
    started, release = Event(), Event()
    converted = []

    def slow_convert(msg):
        converted.append(msg.width)
        if len(converted) == 1:
            started.set()
            assert release.wait(3)
        return convert(msg)

    monkeypatch.setattr(camera_info_snapshot, "_camera_info_to_dict", slow_convert)
    for width in range(1, 101):
        publish["/right"](CameraInfo(width=width))
    publish["/left"](CameraInfo(width=10))
    recorder.start_episode(tmp_path / "0")
    assert not converted
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(recorder.stop_episode)
        try:
            assert started.wait(2)
            pool.submit(publish["/right"], CameraInfo(width=200)).result(timeout=2)
        finally:
            release.set()
        first = pending.result(timeout=2)
    assert converted == [10, 100]
    assert yaml.safe_load(first["right"].read_text())["width"] == 100
    recorder.start_episode(tmp_path / "1")
    second = recorder.stop_episode()
    assert yaml.safe_load(second["right"].read_text())["width"] == 200
