"""Reception-to-assembler integration without ROS, cameras or model weights."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from test_initial_pose_sync import RobotClient, robot_client_impl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "policy/common/runtime"))
from inference_context import InputAssembler, InputField, InputSpec, SampleQuery
from inference_context.reception import ReceptionHistory
from inference_context.observation import ObservationSession


@pytest.fixture
def client():
    with mock.patch.object(RobotClient, "_init_subscriptions"):
        result = RobotClient("ffw_sg2_rev1")
    result._config = {
        "cameras": {"eye": {}, "unused": {}},
        "joint_groups": {"parent": {}, "child": {"joint_names": ["b", "a"]}},
        "sensors": {"odom": {}},
    }
    result._joint_children = {"parent": ["child"]}
    yield result
    result.close()


def specification(source, offsets=(0.,), age=0.2):
    return InputSpec((InputField("input", (SampleQuery(source, offsets, age),), "stack"),))


def joint_message(values):
    return SimpleNamespace(name=["a", "b"], position=values, velocity=[], effort=[])


def test_snapshot_reception_times_are_not_refreshed_by_reads_or_empty_positions(client):
    assert client._observation_capture is None
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.):
        client._update_joint("parent", joint_message([1, 2]))
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=20.):
        client._update_joint("parent", joint_message([]))
        snapshot = client.get_input_snapshot()
    assert snapshot["captured_monotonic_s"] == 20.
    assert snapshot["reception_monotonic_timestamps"] == {"joint:parent": 10., "joint:child": 10.}
    snapshot["reception_monotonic_timestamps"].clear()
    assert client.get_input_snapshot()["reception_monotonic_timestamps"]["joint:child"] == 10.


def test_required_snapshot_rejects_stale_state_before_pixel_work(client):
    client._images = {"eye": np.zeros((480, 640, 3), dtype=np.uint8)}
    client._joint_positions = {"parent": np.ones(2)}
    client._input_received_monotonic = {"camera:eye": 10., "joint:parent": 5.}
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.1), \
            mock.patch.object(robot_client_impl.cv2, "cvtColor") as convert:
        for _ in range(100):
            with pytest.raises(ValueError, match="joint:parent: stale"):
                client.get_required_input_snapshot({"camera:eye", "joint:parent"}, max_age_s=1.)
        convert.assert_not_called()


def test_required_snapshot_respects_distinct_source_ages_and_numeric_sensor_fields(client):
    client._images = {"eye": np.zeros((2, 2, 3), dtype=np.uint8)}
    client._sensors = {"odom": {"linear_velocity": np.array([1., 2., 3.])}}
    client._input_received_monotonic = {"camera:eye": 9.9, "sensor:odom.linear_velocity": 9.}
    sources = {"camera:eye", "sensor:odom.linear_velocity"}
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.):
        snapshot = client.get_required_input_snapshot(sources, max_age_by_source={
            "camera:eye": .2, "sensor:odom.linear_velocity": 2.,
        })
        np.testing.assert_array_equal(snapshot["sensors"]["odom"]["linear_velocity"], [1., 2., 3.])
        snapshot["sensors"]["odom"]["linear_velocity"][:] = 0.
        np.testing.assert_array_equal(client._sensors["odom"]["linear_velocity"], [1., 2., 3.])
        with mock.patch.object(robot_client_impl.cv2, "cvtColor") as convert:
            with pytest.raises(ValueError, match="sensor:odom.linear_velocity: stale"):
                client.get_required_input_snapshot(sources, max_age_by_source={
                    "camera:eye": .2, "sensor:odom.linear_velocity": .5,
                })
            convert.assert_not_called()


def test_history_warmup_does_not_convert_live_cameras(client):
    spec = InputSpec((InputField("history", (SampleQuery("joint:parent", (-.1, 0.), .2),), "stack"),
                      InputField("image", (SampleQuery("camera:eye", max_age_s=1.),))))
    session = ObservationSession(client, spec)
    client._images = {"eye": np.zeros((480, 640, 3), dtype=np.uint8)}
    client._input_received_monotonic = {"camera:eye": 10.}
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.1), \
            mock.patch.object(robot_client_impl.cv2, "cvtColor") as convert:
        for _ in range(100):
            with pytest.raises(ValueError, match="Missing or stale input: joint:parent"):
                client.get_required_input_snapshot({"camera:eye"}, max_age_s=1.,
                                                   readiness_check=session.check_history)
        convert.assert_not_called()
    session.close()


def test_required_snapshot_converts_only_requested_images_outside_lock(client):
    class NoCopy(np.ndarray):
        def copy(self, *args, **kwargs):
            raise AssertionError("redundant BGR image copy")

    original = np.full((4, 6, 3), [1, 2, 3], dtype=np.uint8).view(NoCopy)
    client._images = {"eye": original, "unused": np.zeros((480, 640, 3), dtype=np.uint8)}
    client._input_received_monotonic = {"camera:eye": 10.}
    cvt = robot_client_impl.cv2.cvtColor

    def convert(image, mode):
        assert image is original
        # A different callback thread can acquire the data lock during conversion.
        acquired = []
        def callback():
            with client._lock:
                acquired.append(True)
                client._images["eye"] = np.zeros_like(original)
        thread = threading.Thread(target=callback)
        thread.start()
        thread.join(1)
        assert acquired
        return cvt(image, mode)

    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.1), \
            mock.patch.object(robot_client_impl.cv2, "cvtColor", side_effect=convert) as convert_mock:
        snapshot = client.get_required_input_snapshot({"camera:eye"}, max_age_s=1.)
        assert convert_mock.call_count == 1
    assert set(snapshot["images"]) == {"eye"}
    np.testing.assert_array_equal(snapshot["images"]["eye"][0, 0], [3, 2, 1])
    snapshot["images"]["eye"][:] = 99
    np.testing.assert_array_equal(original[0, 0], [1, 2, 3])


@pytest.mark.parametrize("stamp,after,reason", [(10., 10., "predates"), (11., None, "invalid"), (None, None, "invalid")])
def test_required_snapshot_rejects_invalid_or_prepublication_timestamps(client, stamp, after, reason):
    client._images = {"eye": np.zeros((2, 2, 3), dtype=np.uint8)}
    client._input_received_monotonic = {"camera:eye": stamp}
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.), \
            mock.patch.object(robot_client_impl.cv2, "cvtColor") as convert:
        with pytest.raises(ValueError, match=reason):
            client.get_required_input_snapshot({"camera:eye"}, max_age_s=1., after_s=after)
        convert.assert_not_called()


def test_capture_uses_real_frames_not_latest_reads_and_does_not_seed_cache(client):
    spec = specification("camera:eye", (-1., 0.))
    capture = ReceptionHistory(spec)
    client._images["eye"] = np.zeros((1, 1, 3), dtype=np.uint8)
    client.attach_observation_capture(capture)
    assembler = InputAssembler(spec, {"stack": np.stack})
    with pytest.raises(ValueError, match="Missing"):
        assembler.assemble(capture, anchor_s=2.)
    with mock.patch.object(robot_client_impl.cv2, "imdecode", return_value=np.array([[[1, 2, 3]]], dtype=np.uint8)):
        with mock.patch.object(robot_client_impl.time, "monotonic", return_value=1.):
            client._update_image("eye", SimpleNamespace(data=b"frame"))
        client.get_images()
        client.get_input_snapshot()
        with pytest.raises(ValueError, match="Missing"):
            assembler.assemble(capture, anchor_s=2.)
        with mock.patch.object(robot_client_impl.time, "monotonic", return_value=2.):
            client._update_image("eye", SimpleNamespace(data=b"frame"))
            client._update_image("unused", SimpleNamespace(data=b"frame"))
    result = assembler.assemble(capture, anchor_s=2.)["input"]
    np.testing.assert_array_equal(result, [[[[3, 2, 1]]], [[[3, 2, 1]]]])
    assert capture.bytes_used == 6
    assert client._observation_sequence == 2
    np.testing.assert_array_equal(client.get_image("eye"), [[[1, 2, 3]]])
    assert client.get_input_snapshot()["reception_monotonic_timestamps"]["camera:eye"] == 2.


def test_synthetic_joint_views_capture_only_positions_in_existing_order(client):
    capture = ReceptionHistory(specification("joint:child"))
    client.attach_observation_capture(capture)
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=1.):
        client._update_joint("parent", joint_message([10, 20]))
        client._update_joint("parent", joint_message([]))
    np.testing.assert_array_equal(capture.resolve(SampleQuery("joint:child"), 1.)[0], [20, 10])
    assert client._observation_sequence == 1
    assert capture.bytes_used == 8


def test_numeric_sensor_fields_do_not_store_raw_dicts(client):
    capture = ReceptionHistory(specification("sensor:odom.linear_velocity"))
    client.attach_observation_capture(capture)
    vector = SimpleNamespace(x=1., y=2., z=3., w=1.)
    msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=vector, orientation=vector)),
                          twist=SimpleNamespace(twist=SimpleNamespace(linear=vector, angular=vector)))
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=1.):
        client._update_sensor("odom", msg)
    np.testing.assert_array_equal(capture.resolve(SampleQuery("sensor:odom.linear_velocity"), 1.)[0], [1, 2, 3])
    assert capture.bytes_used == 12
    stamps = client.get_input_snapshot()["reception_monotonic_timestamps"]
    assert stamps["sensor:odom"] == stamps["sensor:odom.linear_velocity"] == 1.


def test_budget_failure_preserves_latest_updates_but_blocks_history_until_reset(client):
    query = SampleQuery("joint:parent", (-1., 0.), 0.2)
    capture = ReceptionHistory(InputSpec((InputField("state", (query,)),)), max_bytes=8)
    client.attach_observation_capture(capture)
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=1.):
        client._update_joint("parent", joint_message([1, 2]))
        client._update_joint("parent", joint_message([3, 4]))
    np.testing.assert_array_equal(client.get_joint_positions("parent"), [3, 4])
    with pytest.raises(RuntimeError, match="budget"):
        capture.resolve(query, 1.)
    client.reset_observation_capture(capture)
    with pytest.raises(ValueError, match="Missing"):
        capture.resolve(query, 1.)
    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=2.):
        client._update_joint("parent", joint_message([5, 6]))
    np.testing.assert_array_equal(capture.resolve(SampleQuery("joint:parent"), 2.)[0], [5, 6])


def test_attach_rejects_unknown_disabled_and_duplicate_captures(client):
    for source in ("instruction", "joint:missing", "sensor:odom", "sensor:odom.raw"):
        with pytest.raises(ValueError, match="Unavailable"):
            client.attach_observation_capture(ReceptionHistory(specification(source)))
    client._subscribe_images = False
    with pytest.raises(ValueError, match="Unavailable"):
        client.attach_observation_capture(ReceptionHistory(specification("camera:eye")))
    capture = ReceptionHistory(specification("joint:parent"))
    client.attach_observation_capture(capture)
    with pytest.raises(RuntimeError, match="already"):
        client.attach_observation_capture(capture)
    client.detach_observation_capture(object())
    assert client._observation_capture is capture
    client.detach_observation_capture(capture)
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.attach_observation_capture(capture)


@pytest.mark.parametrize("operation", ["detach", "reset"])
def test_lifecycle_change_is_a_callback_barrier(client, operation):
    capture = ReceptionHistory(specification("joint:parent"))
    entered, release, detached = threading.Event(), threading.Event(), threading.Event()
    original = capture.record

    def slow_record(*args):
        entered.set()
        assert release.wait(2)
        original(*args)

    capture.record = slow_record
    client.attach_observation_capture(capture)
    update = threading.Thread(target=client._update_joint, args=("parent", joint_message([1, 2])))
    update.start()
    assert entered.wait(2)

    def detach():
        if operation == "detach":
            client.detach_observation_capture(capture)
        else:
            client.reset_observation_capture(capture)
        detached.set()

    removal = threading.Thread(target=detach)
    removal.start()
    assert not detached.wait(0.05)
    release.set()
    update.join(2)
    removal.join(2)
    assert detached.is_set()
    if operation == "detach":
        capture.reset()
        with pytest.raises(RuntimeError, match="not attached"):
            client.reset_observation_capture(capture)
    assert capture.bytes_used == 0
    client._update_joint("parent", joint_message([3, 4]))
    assert capture.bytes_used == (0 if operation == "detach" else 8)


def test_no_capture_means_no_sample_history_or_identity_allocation(client):
    client._update_joint("parent", joint_message([1, 2]))
    assert client._observation_capture is None
    assert client._observation_sequence == 0
