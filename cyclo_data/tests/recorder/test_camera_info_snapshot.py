from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("rclpy")
from sensor_msgs.msg import CameraInfo

from cyclo_data.recorder import camera_info_snapshot
from cyclo_data.recorder.camera_info_snapshot import CameraInfoSnapshot


def _cal(width):
    msg = CameraInfo(width=width, height=240)
    msg.k[0] = msg.k[4] = 450.0
    return msg


def _snapshot(topics):
    callbacks = {}
    node = Mock()

    def subscribe(msg_type, topic, callback, qos, **kwargs):
        callbacks[topic] = callback
        return object()

    node.create_subscription.side_effect = subscribe
    return CameraInfoSnapshot(node, topics), callbacks


def test_shutdown_of_interrupted_episode_preserves_late_calibration(tmp_path, monkeypatch):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    waiting = Event()
    wait = recorder._cache_ready.wait

    def signal_wait(timeout):
        waiting.set()
        return wait(timeout)

    monkeypatch.setattr(recorder._cache_ready, "wait", signal_wait)
    recorder.start_episode(tmp_path / "0")
    publish["/left"](_cal(100))
    with ThreadPoolExecutor(max_workers=1) as pool:
        closed = pool.submit(recorder.close)
        assert waiting.wait(2)
        publish["/left"](_cal(200))
        publish["/right"](_cal(300))
        closed.result(timeout=2)
    assert {p.stem: yaml.safe_load(p.read_text())["width"]
            for p in (tmp_path / "0/camera_info").glob("*.yaml")} == {
        "left": 100, "right": 300,
    }


def test_save_never_waits_and_late_calibration_only_fills_next_episode(
        tmp_path, monkeypatch):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    monkeypatch.setattr(recorder._cache_ready, "wait", lambda *_: pytest.fail("wait on Save"))
    publish["/left"](_cal(400))
    recorder.start_episode(tmp_path / "0")
    assert set(recorder.stop_episode()) == {"left"}
    assert not (tmp_path / "0/camera_info/right.yaml").exists()
    publish["/right"](_cal(500))
    recorder.start_episode(tmp_path / "1")
    assert set(recorder.stop_episode()) == {"left", "right"}
    assert not (tmp_path / "0/camera_info/right.yaml").exists()


def test_discard_neither_waits_nor_writes_and_retains_calibration(tmp_path, monkeypatch):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    publish["/left"](_cal(320))
    recorder.start_episode(tmp_path / "0")
    with monkeypatch.context() as patch:
        patch.setattr(recorder._cache_ready, "wait", lambda *_: pytest.fail("wait on Discard"))
        patch.setattr(camera_info_snapshot, "_camera_info_to_dict", lambda *_: pytest.fail("unexpected write"))
        recorder.discard_episode()
        recorder.discard_episode()
        assert recorder.stop_episode() == {}
    assert not list((tmp_path / "0/camera_info").iterdir())
    publish["/right"](_cal(640))
    recorder.start_episode(tmp_path / "1")
    assert {k: yaml.safe_load(v.read_text())["width"] for k, v in recorder.stop_episode().items()} == {
        "left": 320, "right": 640}


def test_old_subscription_cannot_replace_new_robot_calibration(tmp_path):
    recorder, publish = _snapshot({"head": "/old"})
    old_callback = publish["/old"]
    old_callback(_cal(10))
    recorder.reconfigure({"head": "/new"})
    recorder.start_episode(tmp_path)
    old_callback(_cal(30))
    publish["/new"](_cal(20))
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
        publish["/right"](_cal(width))
    publish["/left"](_cal(10))
    recorder.start_episode(tmp_path / "0")
    assert not converted
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(recorder.stop_episode)
        try:
            assert started.wait(2)
            pool.submit(publish["/right"], _cal(200)).result(timeout=2)
        finally:
            release.set()
        first = pending.result(timeout=2)
    assert converted == [10, 1]
    assert yaml.safe_load(first["right"].read_text())["width"] == 1
    recorder.start_episode(tmp_path / "1")
    second = recorder.stop_episode()
    assert yaml.safe_load(second["right"].read_text())["width"] == 1


def test_release_only_calibrated_camera_and_never_close_twice(tmp_path):
    recorder, publish = _snapshot({"left": "/left", "right": "/right"})
    left, right = recorder._subs.values()
    invalid = _cal(320)
    invalid.p[3] = float("nan")
    for msg in (CameraInfo(), _cal(0), CameraInfo(width=320, height=240), invalid):
        publish["/left"](msg)
    destroy = recorder._node.destroy_subscription
    destroy.assert_not_called()
    publish["/left"](_cal(320))
    publish["/left"](_cal(640))
    destroy.assert_called_once_with(left)
    assert recorder._subs == {"right": right}
    recorder.start_episode(tmp_path)
    assert set(recorder.stop_episode()) == {"left"}
    recorder.close()
    publish["/right"](_cal(800))
    assert [c.args[0] for c in destroy.call_args_list] == [left, right]
    assert not recorder._subs
