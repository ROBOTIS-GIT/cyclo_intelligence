"""Source-driven history bindings and lifecycle, without model/transport imports."""

import threading
from unittest import mock
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_context import ExecutionQuery, InputAssembler, InputField, InputSpec, LatestValues, SampleQuery
from inference_context.execution import ExecutionContext
from inference_context.inputs import ResolvedInputs
from inference_context.observation import ObservationSession
from inference_context.reception import ReceptionHistory


def spec():
    return InputSpec((
        InputField("history", (SampleQuery("joint:arm", (-.1, 0.), .09),), "stack"),
        InputField("image", (SampleQuery("camera:eye"),)),
        InputField("instruction", (SampleQuery("instruction"),)),
    ))


@pytest.mark.parametrize("queries,expected", [
    ((SampleQuery("joint:arm"),), 0),
    ((ExecutionQuery("execution:published:command", count=8),), 0),
    ((ExecutionQuery("execution:pending:command", count=3),
      ExecutionQuery("execution:pending:command", count=7)), 7),
    ((ExecutionQuery("execution:pending:command", count=3),
      ExecutionQuery("execution:context")), None),
])
def test_pending_plan_budget_comes_from_queries(queries, expected):
    session = ObservationSession(mock.Mock(), InputSpec((InputField("input", queries),)))
    assert session.pending_command_count == expected
    session.close()


def test_latest_session_allocates_no_capture_or_queues():
    robot = mock.Mock()
    current = InputSpec((InputField("state", (SampleQuery("joint:arm"),)),))
    session = ObservationSession(robot, current)
    assert session.history is None and not session.history_sources
    assert session.live_sources == {"joint:arm"}
    robot.attach_observation_capture.assert_not_called()
    result = InputAssembler(current).assemble(session.bind(LatestValues({"joint:arm": [1.]}), ""))
    assert result == {"state": [1.]}
    session.close()
    robot.detach_observation_capture.assert_not_called()


def test_execution_sources_are_not_robot_subscriptions_or_observation_history():
    base = spec()
    combined = InputSpec(base.fields + (InputField("plan", (
        ExecutionQuery("execution:pending:command", count=3, min_count=0),)),))
    session = ObservationSession(mock.Mock(), combined)
    session.update_execution_context(ExecutionContext("test", 0, 0, "ready"))
    assert session.history.sources == {"joint:arm"}
    assert session.live_sources == {"camera:eye"}
    assert session.required_observations == {"camera_names": ["eye"], "joint_groups": ["arm"], "sensor_names": []}
    session.history.record("joint:arm", 1, 10., np.array([1.]))
    session.history.record("joint:arm", 2, 10.1, np.array([2.]))
    batch = InputAssembler(combined, {"stack": np.stack}).assemble(
        session.bind(LatestValues({"camera:eye": "pixels"}), "move"), anchor_s=10.15)
    assert batch["plan"] == () and batch["instruction"] == "move"
    np.testing.assert_array_equal(batch["history"], [[1.], [2.]])


def test_history_binding_keeps_latest_sources_out_of_capture_and_resolves_once():
    robot = mock.Mock()
    session = ObservationSession(robot, spec())
    capture = session.history
    assert capture.sources == {"joint:arm"}
    assert session.live_sources == {"camera:eye"}
    capture.record("joint:arm", 1, 10., np.array([1.]))
    capture.record("joint:arm", 2, 10.1, np.array([2.]))
    capture.record("camera:eye", 3, 10.1, np.zeros((1000, 1000, 3)))
    assert capture.bytes_used == 16
    current = LatestValues({"camera:eye": "pixels"})
    assembler = InputAssembler(spec(), {"stack": np.stack})
    with mock.patch.object(capture, "resolve", wraps=capture.resolve) as resolve:
        ready = ResolvedInputs(spec(), session.bind(current, "move"), 10.15)
        batch = assembler.assemble(ready, anchor_s=10.15)
        assert resolve.call_count == 1
    np.testing.assert_array_equal(batch["history"], [[1.], [2.]])
    assert batch["image"] == "pixels" and batch["instruction"] == "move"
    robot.reset_observation_capture.side_effect = lambda value: value.reset()
    session.reset()
    assert capture.bytes_used == 0
    session.close()
    session.close()
    robot.detach_observation_capture.assert_called_once_with(capture)
    with pytest.raises(RuntimeError, match="closed"):
        session.bind(current, "move")


def test_history_latest_sample_must_cross_publication_barrier_but_older_offsets_need_not():
    session = ObservationSession(mock.Mock(), spec())
    session.history.record("joint:arm", 1, 10., np.array([1.]))
    session.history.record("joint:arm", 2, 10.1, np.array([2.]))
    assembler = InputAssembler(spec(), {"stack": np.stack})
    current = LatestValues({"camera:eye": "pixels"})
    assembler.assemble(session.bind(current, "", after_s=10.05), anchor_s=10.15)
    with pytest.raises(ValueError, match="publication barrier"):
        assembler.assemble(session.bind(current, "", after_s=10.1), anchor_s=10.15)


def test_reset_while_copying_history_invalidates_the_in_flight_snapshot():
    capture = ReceptionHistory(spec())
    capture.record("joint:arm", 1, 10., np.array([1.]))
    original = capture._store.resolve
    entered, release = threading.Event(), threading.Event()
    failures = []

    def slow(*args, **kwargs):
        values = original(*args, **kwargs)
        entered.set()
        assert release.wait(2)
        return values

    def read():
        try:
            capture.resolve(SampleQuery("joint:arm"), 10.)
        except Exception as error:
            failures.append(error)

    with mock.patch.object(capture._store, "resolve", side_effect=slow):
        thread = threading.Thread(target=read)
        thread.start()
        try:
            assert entered.wait(2)
            capture.reset()  # Must not wait for image copies/model transforms.
        finally:
            release.set()
            thread.join(2)
        assert not thread.is_alive()
    assert len(failures) == 1 and "reset during snapshot" in str(failures[0])


def test_missing_late_input_prevents_all_transforms():
    transform = mock.Mock()
    specification = InputSpec((InputField("first", (SampleQuery("ok"),), "tensor"),
                               InputField("second", (SampleQuery("missing"),))))
    with pytest.raises(ValueError, match="Missing"):
        InputAssembler(specification, {"tensor": transform}).assemble(LatestValues({"ok": [1]}))
    transform.assert_not_called()
