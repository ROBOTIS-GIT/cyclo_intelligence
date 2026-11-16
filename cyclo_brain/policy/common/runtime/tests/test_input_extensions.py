from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from inference_inputs import Binding, Data, Graph
from inference_inputs.memory import Memory
from inference_inputs.operators import default_registry
from inference_inputs.resources import Budget
from inference_context.inputs import LatestValues, SampleQuery, InputSample
from inference_context.observation import ObservationSession


def test_prediction_result_cache_is_rejected_before_execution():
    registry = default_registry()
    registry.register("encoder", lambda opts, ctx: lambda values: values[0], cacheable=True)
    with pytest.raises(ValueError, match="result nodes"):
        Graph({"sources": {}, "nodes": {"prediction": {
            "op": "encoder", "stage": "result", "inputs": ["model_action"],
            "cache": {"dependencies": ["model_action"]}}},
            "outputs": {"before": {}, "after": {"*": "processed"}}}, registry)


def test_memory_identity_changes_on_commit_with_same_input_sample():
    memory = Memory({"feature": {}})
    frame = Data(np.ones(3), (("camera", 1),), (1.,))
    memory.begin(1)
    memory.propose("feature", frame)
    memory.success()
    first = memory.read("feature")
    memory.begin(2)
    memory.propose("feature", frame.derived(frame.value * 2))
    memory.success()
    second = memory.read("feature")
    assert first.sample_ids != second.sample_ids
    assert first.received_s == second.received_s == (1.,)
    assert second.computed_s != second.received_s[0]


def test_new_source_lifecycle_and_old_sample_block_without_robot_subscriptions():
    class Provider:
        def __init__(self):
            self.events = []
            self.stamp = 1.

        def start(self, queries, budget):
            self.events.append("start")
            assert queries == (SampleQuery("custom:audio", (0.,), .2),)

        def resolve_samples(self, query, anchor):
            return (InputSample(np.ones(3), "custom:audio", 1, self.stamp),)

        def reset(self):
            self.events.append("reset")

        def close(self):
            self.events.append("close")

    provider = Provider()
    graph = Graph({"sources": {"audio": {"binding": "audio"}}, "nodes": {},
                   "outputs": {"before": {"audio": "audio"}}}, default_registry(),
                  bindings={"audio": lambda _: Binding((SampleQuery("custom:audio", (0.,), .2),), provider=provider)})
    session = ObservationSession(object(), graph.spec, providers=graph.providers)
    assert not session.live_sources and all(not v for v in session.required_observations.values())
    batch = graph.assemble(session.bind(LatestValues({}), ""), anchor_s=1.1)
    np.testing.assert_array_equal(batch["audio"], np.ones(3))
    with pytest.raises(ValueError, match="stale"):
        graph.assemble(session.bind(LatestValues({}), ""), anchor_s=2.)
    with pytest.raises(ValueError, match="predates"):
        graph.assemble(session.bind(LatestValues({}), "", after_s=1.01), anchor_s=1.1)
    session.reset()
    session.close()
    session.close()
    assert provider.events == ["start", "reset", "close"]


def test_external_temporal_source_rewarms_after_generation_reset():
    from inference_context.inputs import InputField, InputSpec
    provider = SimpleNamespace(start=lambda queries, budget: None,
                               resolve_samples=lambda query, anchor: (),
                               reset=lambda: None, close=lambda: None)
    query = SampleQuery("custom:feature", (-2., 0.), .1)
    session = ObservationSession(object(), InputSpec((InputField("feature", (query,)),)),
                                 providers={"custom:feature": provider})
    assert session.read_timeout_s == 3.
    session._warming_up = False
    session.reset()
    assert session.read_timeout_s == 3.
    session.close()


def test_cached_encoder_instruction_invalidation_and_shared_memory_budget():
    registry = default_registry()
    calls = []
    registry.register("encoder", lambda opts, ctx: lambda values: calls.append(1) or values[0].value * 2, cacheable=True)
    budget = Budget(48)
    config = {"sources": {"image": {"source": "camera:head"}, "instruction": {"source": "instruction"}},
              "nodes": {"feature": {"op": "encoder", "inputs": ["image"],
                                    "cache": {"dependencies": ["image", "instruction"]}}},
              "outputs": {"before": {"x": "feature"}}}
    graph = Graph(config, registry, budget=budget)
    frame = Data(np.ones(3), (("head", 1),), (1.,))
    for instruction in ("a", "a", "b"):
        result = graph.assemble(LatestValues({"camera:head": frame, "instruction": instruction}))
        if result["x"].flags.writeable:
            result["x"][0] = -100
        else:
            with pytest.raises(ValueError):
                result["x"][0] = -100
    assert len(calls) == 2 and budget.used == 24
    result = graph.assemble(LatestValues({"camera:head": frame, "instruction": "b"}))
    assert result["x"][0] == 2
    memory = Memory({"x": {}}, budget=budget)
    memory.begin(1)
    with pytest.raises(MemoryError):
        memory.propose("x", Data(np.ones(4)))
    assert budget.used == 24
    graph.reset()
    assert budget.used == 0
    graph.assemble(LatestValues({"camera:head": frame, "instruction": "b"}))
    assert len(calls) == 3
    graph.close()


def test_declared_mutating_operator_gets_private_inputs():
    registry = default_registry()
    def compile_mutator(options, context):
        def mutate(values):
            values[0].value[:] = 0
            return values[0]
        return mutate
    registry.register("mutating", compile_mutator, mutates_inputs=True)
    graph = Graph({"sources": {"x": {"source": "joint:arm"}},
                   "nodes": {"y": {"op": "mutating", "inputs": ["x"]}},
                   "outputs": {"before": {"before": "x", "after": "y"}}}, registry)
    original = np.ones(3)
    result = graph.assemble(LatestValues({"joint:arm": original}))
    np.testing.assert_array_equal(result["before"], np.ones(3))
    np.testing.assert_array_equal(result["after"], np.zeros(3))


def test_retained_history_and_features_share_one_budget():
    from inference_context.history import HistoryStore
    from inference_context import InputSpec, InputField
    budget = Budget(32)
    history = HistoryStore(InputSpec((InputField("x", (SampleQuery("joint:arm", (-.1, 0.), .1),)),)), budget=budget)
    history.append("joint:arm", 1, 1., np.ones(3))
    assert budget.used == 24
    memory = Memory({"feature": {}}, budget=budget)
    memory.begin(1)
    with pytest.raises(MemoryError):
        memory.propose("feature", Data(np.ones(2)))
    history.reset()
    memory.reset()
    assert budget.used == 0
