"""Callback-driven readiness, including wakeup/close races without ROS I/O."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
from unittest.mock import patch

import pytest

from test_observation_capture import client, joint_message  # noqa: F401
from test_initial_pose_sync import robot_client_impl


@pytest.mark.parametrize('iteration', range(30))
def test_callback_between_snapshot_and_wait_is_not_lost(client, iteration):
    original = client.get_required_input_snapshot
    calls = []

    def snapshot(*args, **kwargs):
        calls.append(True)
        try:
            return original(*args, **kwargs)
        except robot_client_impl.ObservationNotReady:
            client._update_joint('parent', joint_message([1, 2]))
            raise

    with patch.object(client, 'get_required_input_snapshot', side_effect=snapshot):
        result = client.wait_for_required_input_snapshot({'joint:parent'}, timeout_s=.2)
    assert result['joint_positions']['parent'].tolist() == [1, 2]
    assert len(calls) == 2


def test_no_reception_means_one_read_not_a_polling_loop(client):
    with patch.object(client, 'get_required_input_snapshot', wraps=client.get_required_input_snapshot) as read:
        with pytest.raises(robot_client_impl.ObservationNotReady):
            client.wait_for_required_input_snapshot({'joint:parent'}, timeout_s=.04)
    assert read.call_count == 2  # Initial check and the deadline check.


@pytest.mark.parametrize('options', [{'max_age_s': -1}, {'after_s': -1}, {'max_age_by_source': {'joint:parent': 0}}])
def test_configuration_errors_are_not_retried(client, options):
    with patch.object(client, 'get_required_input_snapshot', wraps=client.get_required_input_snapshot) as read:
        with pytest.raises(ValueError):
            client.wait_for_required_input_snapshot({'joint:parent'}, timeout_s=5, **options)
    assert read.call_count == 1


def test_unknown_source_fails_before_waiting(client):
    with pytest.raises(ValueError, match='Unknown input'):
        client.wait_for_required_input_snapshot({'joint:typo'}, timeout_s=5)


@pytest.mark.parametrize('close', [False, True])
def test_wait_wakes_for_real_callback_or_close(client, close):
    reading = Event()
    original = client.get_required_input_snapshot

    def snapshot(*args, **kwargs):
        reading.set()
        return original(*args, **kwargs)

    with patch.object(client, 'get_required_input_snapshot', side_effect=snapshot):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.wait_for_required_input_snapshot,
                                 {'joint:parent'}, timeout_s=2, after_s=time.monotonic())
            try:
                assert reading.wait(.5)
                if close:
                    client.close()
                    with pytest.raises(RuntimeError, match='closed'):
                        future.result(timeout=.5)
                else:
                    client._update_joint('parent', joint_message([3, 4]))
                    assert future.result(timeout=.5)['joint_positions']['parent'].tolist() == [3, 4]
            finally:
                client.close()
