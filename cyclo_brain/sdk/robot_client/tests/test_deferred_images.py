"""Playground's lazy image API must keep atomic snapshot receive times."""
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from test_rail_transport import RobotClient, implementation


@pytest.fixture
def deferred(monkeypatch):
    monkeypatch.setattr(RobotClient, '_init_subscriptions', lambda self: None)
    robot = RobotClient('ffw_sg2_rev1', defer_image_decoding=True)
    yield robot
    robot.close()


def frame(value=80):
    ok, encoded = cv2.imencode('.png', np.full((4, 6, 3), value, np.uint8))
    assert ok
    return SimpleNamespace(data=encoded.tobytes())


def test_only_latest_frame_is_decoded_when_read(deferred):
    with mock.patch.object(implementation.cv2, 'imdecode', wraps=cv2.imdecode) as decode:
        deferred._update_image('head', frame(20))
        deferred._update_image('head', frame(90))
        assert decode.call_count == 0
        np.testing.assert_array_equal(deferred.get_images()['head'], np.full((4, 6, 3), 90))
        deferred.get_image('head')
        assert decode.call_count == 1


def test_snapshot_uses_receipt_time_not_decode_time(deferred, monkeypatch):
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 10.)
    monkeypatch.setattr(implementation.time, 'time', lambda: 100.)
    deferred._update_image('head', frame())
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 12.)
    result = deferred.get_observation_snapshot(camera_names=['head'], joint_groups=[])
    assert result['images']['head'].shape == (4, 6, 3)
    assert result['timing']['sensor_received_monotonic_s']['image:head'] == 10.
    assert result['timing']['sensor_age_s']['image:head'] == 2.
    assert deferred.get_image_timestamp('head') == 100.


def test_new_callback_during_decode_is_kept_for_next_read(deferred, monkeypatch):
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 10.)
    deferred._update_image('head', frame(20))
    decode = cv2.imdecode
    def receive_during_decode(data, flags):
        monkeypatch.setattr(implementation.time, 'monotonic', lambda: 11.)
        deferred._update_image('head', frame(90))
        return decode(data, flags)
    with mock.patch.object(implementation.cv2, 'imdecode', side_effect=receive_during_decode):
        image = deferred.get_image('head')
    assert image[0, 0, 0] == 20
    assert deferred._image_received_monotonic['head'] == 10.
    assert deferred.get_image('head')[0, 0, 0] == 90
    assert deferred._image_received_monotonic['head'] == 11.


def test_corrupt_latest_frame_cannot_reuse_older_valid_image(deferred):
    deferred._update_image('head', frame())
    assert deferred.is_image_ready('head')
    deferred._update_image('head', SimpleNamespace(data=b'broken image'))
    with pytest.raises(RuntimeError, match='decode image'):
        deferred.get_images()
    with pytest.raises(RuntimeError, match='unavailable'):
        deferred.get_observation_snapshot(camera_names=['head'], joint_groups=[])


def test_readiness_decodes_pending_selected_cameras(deferred):
    camera = next(iter(deferred.camera_names))
    deferred._config['cameras'] = {camera: deferred._config['cameras'][camera]}
    deferred._config['joint_groups'] = {}
    deferred._update_image(camera, frame())
    assert deferred._all_ready()
    assert deferred.get_status()['cameras'][camera]['ready']
    assert deferred._get_missing() == []


def test_default_client_keeps_immediate_decoding(monkeypatch):
    monkeypatch.setattr(RobotClient, '_init_subscriptions', lambda self: None)
    robot = RobotClient('ffw_sg2_rev1')
    try:
        robot._update_image('head', frame())
        assert 'head' in robot._images
    finally:
        robot.close()


def test_close_handles_incomplete_constructor():
    RobotClient.__new__(RobotClient).close()
