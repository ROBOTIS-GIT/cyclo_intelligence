from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from inference_inputs import Data, Graph
from inference_inputs.memory import Memory, register_memory
from inference_inputs.operators import default_registry
from inference_context.inputs import LatestValues
from inference_context.execution import ExecutionContext, PlanningRecord, ActionRecord


def context(revision=0, after=0, plans=(), actions=(), generation=0):
    return ExecutionContext("session", generation, revision, "running", tuple(actions), tuple(plans), (),
                            after, after + len(plans) + len(actions), feedback_schema=2)


@pytest.mark.parametrize("condition,required", [("prediction_success", 0), ("plan_accepted", 0),
                                                ("first_publication", 1), ("published_count", 2), ("plan_terminal", 3)])
def test_memory_waits_for_declared_fact_not_zoh(condition, required):
    options = {"event": condition}
    if condition == "published_count":
        options["count"] = 2
    memory = Memory({"feature": options})
    memory.update(context())
    memory.begin(7)
    memory.propose("feature", Data(np.array([42.]), (("cam", 10),), (2.,)))
    memory.success()
    fallback = Data(np.array([-1.]))
    if condition != "prediction_success":
        assert memory.read("feature", initial=fallback).value[0] == -1
    memory.update(context(1, plans=[PlanningRecord(7, 3, 0, 3, 1, 3., command_start_id=20)]))
    # A synthetic ZOH receipt outside the accepted plan cannot commit memory.
    memory.update(context(2, 1, actions=[ActionRecord("7", "published", "command", (0.,), command_id=99,
                                                     event_id=2, recorded_s=3.1)]))
    if required:
        assert memory.read("feature", initial=fallback).value[0] == -1
    for i in range(3):
        memory.update(context(i + 3, i + 2, actions=[ActionRecord("7", "published", "command", (0.,),
                    command_id=20 + i, event_id=i + 3, recorded_s=4. + i)]))
        assert memory.read("feature", initial=fallback).value[0] == (42 if i + 1 >= required else -1)
    assert memory.read("feature").received_s == (2.,)


def test_discard_failure_reset_and_budget():
    memory = Memory({"x": {"event": "plan_terminal"}}, max_bytes=32)
    memory.update(context())
    memory.begin(1)
    memory.propose("x", Data(np.array([1.])))
    memory.success()
    memory.update(context(1, plans=[PlanningRecord(1, 1, 0, 1, 1, 1., command_start_id=0)], actions=[
        ActionRecord("1", "failed", "command", (0.,), command_id=0, event_id=2, recorded_s=2.)]))
    with pytest.raises(ValueError, match="no committed"):
        memory.read("x")
    memory.begin(2)
    with pytest.raises(MemoryError):
        memory.propose("x", Data(np.ones(16)))
    with pytest.raises(RuntimeError, match="reset"):
        memory.begin(3)
    memory.update(context(2, after=2, generation=1))
    assert memory.bytes_used == 0
    memory.begin(3)


def test_real_encoder_combines_previous_and_current_features_without_mutation():
    # A real numerical encoder, with fixed learned-weight-shaped parameters.
    weights = np.random.default_rng(4).normal(size=(12, 4)).astype(np.float32)
    calls = []
    registry = default_registry()
    registry.register("tiny_encoder", lambda opts, ctx: lambda xs: calls.append(1) or np.tanh(xs[0].value.reshape(-1) @ weights))
    memory = Memory({"features": {"event": "prediction_success"}})
    register_memory(registry, memory)
    graph = Graph({"sources": {"image": {"source": "camera:head"}}, "nodes": {
        "encoded": {"op": "tiny_encoder", "inputs": ["image"]},
        "previous": {"op": "memory_read", "inputs": ["encoded"], "options": {"slot": "features", "initial": "input"}},
        "combined": {"op": "concat", "inputs": ["previous", "encoded"]},
        "remember": {"op": "memory_write", "inputs": ["encoded"], "options": {"slot": "features"}},
    }, "outputs": {"before": {"features": "combined"}}}, registry)
    for i in range(2):
        image = np.full((2, 2, 3), i + 1., dtype=np.float32)
        memory.begin(i)
        result = graph.assemble(LatestValues({"camera:head": Data(image, (("head", i),), (float(i),))}))
        expected_old = np.tanh(np.full(12, max(1, i)) @ weights)
        np.testing.assert_allclose(result["features"][:4], expected_old, atol=1e-6)
        memory.success()
        np.testing.assert_array_equal(image, np.full((2, 2, 3), i + 1.))
    assert calls == [1, 1]
    memory.reset()
    assert memory.bytes_used == 0
