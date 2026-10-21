"""RAIL observation timing and derivative-preserving trajectory transport."""
import importlib.util
from pathlib import Path
import sys
import threading
import types
from unittest import mock

import cv2
import numpy as np
import pytest


class Message:
    def __init__(self, **values):
        self.__dict__.update(values)


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, **message):
        self.messages.append(message)


MODULE_PATH = Path(__file__).resolve().parents[1] / 'robot_client' / 'robot_client.py'
sys.path.insert(0, str(MODULE_PATH.parents[4] / 'shared' / 'shared' / 'robot_configs'))
sdk = types.ModuleType('zenoh_ros2_sdk')
sdk.ROS2Publisher = Publisher
sdk.ROS2Subscriber = object
sdk.get_message_class = lambda _name: Message
spec = importlib.util.spec_from_file_location('rail_robot_client_impl', MODULE_PATH)
implementation = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {'zenoh_ros2_sdk': sdk}):
    spec.loader.exec_module(implementation)
RobotClient = implementation.RobotClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(RobotClient, '_init_subscriptions', lambda self: None)
    robot = RobotClient('ffw_sg2_rev1')
    robot._closed = True
    for key, cfg in robot._action_groups.items():
        robot._command_publishers[f'leader_{key}'] = Publisher()
        robot._command_joint_names[f'leader_{key}'] = list(cfg.get('joint_names', []))
    return robot


def assert_no_publish(client):
    assert all(not publisher.messages for publisher in client._command_publishers.values())


def test_trajectory_serializes_selected_groups_derivatives_and_explicit_times(client):
    positions = np.arange(48, dtype=float).reshape(3, 16)
    velocities = positions*.1
    accelerations = -positions*.01
    client.publish_action_trajectory(positions, velocities, accelerations,
                                     [0., .025, 1.000000001], ['arm_right', 'arm_left'])
    for group, columns in [('arm_right', slice(0, 8)), ('arm_left', slice(8, 16))]:
        messages = client._command_publishers[f'leader_{group}'].messages
        assert len(messages) == 1
        message = messages[0]
        assert message['joint_names'] == client._action_groups[group]['joint_names']
        assert message['header'].stamp.sec == message['header'].stamp.nanosec == 0
        for index, (sec, nanosec) in enumerate([(0, 0), (0, 25_000_000), (1, 1)]):
            point = message['points'][index]
            np.testing.assert_array_equal(point.positions, positions[index, columns])
            np.testing.assert_array_equal(point.velocities, velocities[index, columns])
            np.testing.assert_array_equal(point.accelerations, accelerations[index, columns])
            assert point.time_from_start.sec == sec
            assert point.time_from_start.nanosec == nanosec
            assert len(point.effort) == 0
    positions[:] = 999.
    velocities[:] = 999.
    accelerations[:] = 999.
    point = client._command_publishers['leader_arm_right'].messages[0]['points'][0]
    np.testing.assert_array_equal(point.positions, np.arange(8))
    np.testing.assert_allclose(point.velocities, np.arange(8)*.1)
    np.testing.assert_allclose(point.accelerations, -np.arange(8)*.01)
    assert not client._command_publishers['leader_mobile'].messages


@pytest.mark.parametrize('offsets', [[-.1, .1], [0., 0.], [.1, 0.], [0., np.nan],
                                    [0., np.inf], [0.], [[0., .1]], [0., 1e-12],
                                    [0., 2**31]])
def test_invalid_offsets_reject_whole_trajectory(client, offsets):
    values = np.zeros((2, 16))
    with pytest.raises(ValueError, match='offsets'):
        client.publish_action_trajectory(values, values, values, offsets, ['arm_left', 'arm_right'])
    assert_no_publish(client)


@pytest.mark.parametrize('invalid', [np.zeros((2, 15)), np.zeros(16), np.zeros((0, 16)),
                                    np.full((2, 16), np.nan), np.full((2, 16), np.inf)])
@pytest.mark.parametrize('matrix', [0, 1, 2])
def test_invalid_matrices_reject_whole_trajectory(client, invalid, matrix):
    matrices = [np.zeros((2, 16)) for _ in range(3)]
    matrices[matrix] = invalid
    with pytest.raises(ValueError):
        client.publish_action_trajectory(*matrices, [0., .1], ['arm_left', 'arm_right'])
    assert_no_publish(client)


@pytest.mark.parametrize('keys,width', [(['arm_left', 'arm_left'], 16),
                                      (['arm_left', 'unknown'], 16),
                                      (['arm_left', 'odometry'], 11),
                                      (['arm_left', 'arm_right'], 15),
                                      (['arm_left', 'arm_right'], 17), ([], 8)])
def test_invalid_groups_reject_before_any_publish(client, keys, width):
    values = np.zeros((2, width))
    with pytest.raises(ValueError):
        client.publish_action_trajectory(values, values, values, [0., .1], keys)
    assert_no_publish(client)


def test_missing_second_publisher_does_not_publish_first_arm(client):
    del client._command_publishers['leader_arm_right']
    values = np.zeros((2, 16))
    with pytest.raises(RuntimeError, match='publisher unavailable'):
        client.publish_action_trajectory(values, values, values, [0., .1], ['arm_left', 'arm_right'])
    assert_no_publish(client)


def seed_observation(client):
    client._images = {'front': np.array([[[10, 20, 30]]], dtype=np.uint8),
                      'unused': np.zeros((1, 1, 3), dtype=np.uint8)}
    client._joint_positions = {'left': np.array([1., 2.]), 'right': np.array([3., 4.]),
                               'unused': np.array([999.])}
    client._image_received_monotonic = {'front': 9.95, 'unused': 1.}
    client._joint_received_monotonic = {'left': 9.9, 'right': 9.98, 'unused': 1.}


def test_snapshot_selects_sensors_and_exposes_receive_time_provenance(client, monkeypatch):
    seed_observation(client)
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 10.)
    snapshot = client.get_observation_snapshot(camera_names=['front'], joint_groups=['left', 'right'])
    np.testing.assert_array_equal(snapshot['images']['front'], [[[30, 20, 10]]])
    assert set(snapshot['images']) == {'front'}
    assert set(snapshot['joints']) == {'left', 'right'}
    timing = snapshot['timing']
    assert timing['sensor_snapshot_monotonic_s'] == 9.9
    assert timing['snapshot_read_monotonic_s'] == 10.
    assert timing['timestamp_source'] == 'local_monotonic_receive'
    assert timing['sensor_received_monotonic_s'] == {'image:front': 9.95, 'joint:left': 9.9, 'joint:right': 9.98}
    assert timing['sensor_age_s'] == pytest.approx({'image:front': .05, 'joint:left': .1, 'joint:right': .02})
    assert timing['sensor_skew_s'] == pytest.approx(.08)
    snapshot['images']['front'][:] = 0
    snapshot['joints']['left'][:] = 0
    timing['sensor_received_monotonic_s']['joint:left'] = 0
    np.testing.assert_array_equal(client._images['front'], [[[10, 20, 30]]])
    np.testing.assert_array_equal(client._joint_positions['left'], [1., 2.])
    assert client._joint_received_monotonic['left'] == 9.9


def test_snapshot_copies_values_and_timestamps_before_callbacks_can_update(client, monkeypatch):
    seed_observation(client)
    copying = threading.Event()
    attempting_update = threading.Event()

    class BlockingImage(np.ndarray):
        def copy(self, *args, **kwargs):
            copying.set()
            assert attempting_update.wait(2.)
            return super().copy(*args, **kwargs)

    client._images['front'] = client._images['front'].view(BlockingImage)
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 10.)

    def update():
        assert copying.wait(2.)
        attempting_update.set()
        with client._lock:
            client._images['front'][:] = 100
            client._joint_positions['left'][:] = 100
            client._image_received_monotonic['front'] = 10.
            client._joint_received_monotonic['left'] = 10.

    writer = threading.Thread(target=update)
    writer.start()
    snapshot = client.get_observation_snapshot(format='bgr', camera_names=['front'], joint_groups=['left'])
    writer.join(2.)
    assert not writer.is_alive()
    np.testing.assert_array_equal(snapshot['images']['front'], [[[10, 20, 30]]])
    np.testing.assert_array_equal(snapshot['joints']['left'], [1., 2.])
    assert snapshot['timing']['sensor_received_monotonic_s'] == {'image:front': 9.95, 'joint:left': 9.9}
    np.testing.assert_array_equal(client._joint_positions['left'], [100., 100.])


@pytest.mark.parametrize('bad_timestamp', [None, 'invalid', float('nan'), float('inf'), -1., 10.1])
def test_snapshot_rejects_invalid_receive_times(client, monkeypatch, bad_timestamp):
    seed_observation(client)
    client._joint_received_monotonic['left'] = bad_timestamp
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 10.)
    with pytest.raises(RuntimeError, match='timestamps are invalid'):
        client.get_observation_snapshot(camera_names=['front'], joint_groups=['left'])


@pytest.mark.parametrize('missing', ['image', 'image_timestamp', 'joint', 'joint_timestamp'])
def test_snapshot_requires_requested_values_and_their_receive_times(client, missing):
    seed_observation(client)
    if missing == 'image':
        del client._images['front']
    elif missing == 'image_timestamp':
        del client._image_received_monotonic['front']
    elif missing == 'joint':
        del client._joint_positions['left']
    else:
        del client._joint_received_monotonic['left']
    with pytest.raises(RuntimeError, match='unavailable'):
        client.get_observation_snapshot(camera_names=['front'], joint_groups=['left'])


def test_image_only_snapshot_uses_image_receipt_and_empty_snapshot_fails(client, monkeypatch):
    seed_observation(client)
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: 10.)
    snapshot = client.get_observation_snapshot(camera_names=['front'], joint_groups=[])
    assert snapshot['joints'] == {}
    assert snapshot['timing']['sensor_snapshot_monotonic_s'] == 9.95
    with pytest.raises(RuntimeError, match='at least one'):
        client.get_observation_snapshot(camera_names=[], joint_groups=[])


def test_callbacks_record_receive_times_with_parent_and_child_positions(client, monkeypatch):
    monotonic = [8.]
    monkeypatch.setattr(implementation.time, 'monotonic', lambda: monotonic[0])
    monkeypatch.setattr(implementation.time, 'time', lambda: 1_000.)
    client._config['joint_groups'] = {'left': {'joint_names': ['b']}}
    client._joint_children = {'all': ['left']}
    client._update_joint('all', Message(name=['a', 'b'], position=[1., 2.], velocity=[], effort=[]))
    assert client._joint_received_monotonic == {'all': 8., 'left': 8.}
    np.testing.assert_array_equal(client._joint_positions['left'], [2.])
    monotonic[0] = 9.
    client._update_joint('all', Message(name=['a', 'b'], position=[], velocity=[3., 4.], effort=[]))
    assert client._joint_received_monotonic == {'all': 8., 'left': 8.}

    def decode(_buffer, _flags):
        monotonic[0] = 10.
        return np.array([[[1, 2, 3]]], dtype=np.uint8)

    monkeypatch.setattr(cv2, 'imdecode', decode)
    client._update_image('front', Message(data=b'encoded'))
    assert client._image_received_monotonic == {'front': 9.}
    snapshot = client.get_observation_snapshot(camera_names=['front'], joint_groups=['left'])
    assert snapshot['timing']['sensor_snapshot_monotonic_s'] == 8.
    assert snapshot['timing']['sensor_skew_s'] == 1.
