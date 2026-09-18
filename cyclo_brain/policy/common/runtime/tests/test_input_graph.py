from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from inference_context.inputs import LatestValues
from inference_inputs import Data, Graph, query_from_config
from inference_inputs.operators import default_registry


def graph_config():
    return {
        "sources": {"raw": {"source": "joint:arm"}},
        "nodes": {"batch": {"op": "unsqueeze", "inputs": ["raw"], "options": {"axis": 0}},
                  "processed_state": {"op": "select", "stage": "after", "inputs": ["processed"],
                                      "options": {"key": "state"}}},
        "outputs": {"before": {"state": "batch"}, "after": {"state": "processed_state"}},
    }


def test_processor_stages_and_session_snapshot():
    config = graph_config()
    graph = Graph(config, default_registry())
    config["nodes"]["batch"]["options"]["axis"] = 100
    evaluation = graph.begin(LatestValues({"joint:arm": np.array([1., 2.])}))
    batch = evaluation.run("before")
    np.testing.assert_array_equal(batch["state"], [[1., 2.]])
    result = evaluation.run("after", {"state": batch["state"] * 2})
    np.testing.assert_array_equal(result["state"], [[2., 4.]])
    with pytest.raises(ValueError, match="once"):
        evaluation.run("after", {})


@pytest.mark.parametrize("change", [
    lambda c: c["nodes"]["batch"].update(inputs=["batch"]),
    lambda c: c["nodes"]["batch"].update(inputs=["missing"]),
    lambda c: c["nodes"]["batch"].update(inputs=["processed"]),
    lambda c: c["nodes"]["batch"].update(op="import.os"),
    lambda c: c.update(unknown=True),
])
def test_invalid_graph_rejected_at_compile(change):
    config = graph_config()
    change(config)
    with pytest.raises(ValueError):
        Graph(config, default_registry())


def test_shared_node_once_and_source_not_mutated():
    calls = []
    registry = default_registry()
    registry.register("encoder", lambda o, c: lambda inputs: calls.append(1) or inputs[0].value * 3)
    config = graph_config()
    config["nodes"]["batch"] = {"op": "encoder", "inputs": ["raw"]}
    config["outputs"]["before"] = {"a": "batch", "b": "batch"}
    source = np.array([1., 2.])
    result = Graph(config, registry).assemble(LatestValues({"joint:arm": source}))
    assert len(calls) == 1
    np.testing.assert_array_equal(result["a"], result["b"])
    np.testing.assert_array_equal(source, [1., 2.])


def test_identity_preserves_provenance_not_compute_time_as_receive_time():
    original = Data(np.array([1.]), ("frame:3",), (5.,), semantics=(("units", "rad"),))
    config = {"sources": {"x": {"source": "joint:arm"}},
              "nodes": {"y": {"op": "identity", "inputs": ["x"], "contract": {"shape": [1]}}},
              "outputs": {"before": {"state": "y"}}}
    evaluation = Graph(config, default_registry()).begin(LatestValues({"joint:arm": original}))
    assert evaluation.run("before")["state"] is original.value
    assert evaluation.values["y"].received_s == (5.,)
    assert evaluation.values["y"].sample_ids == ("frame:3",)


def test_temporal_configuration_is_explicit():
    query = query_from_config({"source": "camera:head", "frame_offsets": [-2, -1, 0], "fps": 20, "max_age_s": .02})
    assert query.offsets_s == (-.1, -.05, 0.)
    for config in ({"source": "camera:head", "frame_offsets": [-1, 0]},
                   {"source": "camera:head", "offsets_s": [-.1, 0]},
                   {"source": "camera:head", "fps": 10}):
        with pytest.raises(ValueError):
            query_from_config(config)
