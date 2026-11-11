"""Requirements-driven input assembly without importing any model backend."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_context import InputAssembler, InputField, InputSpec, LatestValues, SampleQuery
from inference_context.history import HistoryStore
from inference_context.inputs import RoutedInputs


def history_spec(offsets=(-1.0, 0.0), age=0.2):
    query = SampleQuery("camera:eye", offsets, age)
    return InputSpec((InputField("video", (query,), "stack"),))


def test_one_assembler_supports_state_action_and_instruction_without_model_names():
    spec = InputSpec((
        InputField("custom.state", (SampleQuery("joint:left"), SampleQuery("joint:right")), "concat", (3,)),
        InputField("previous_actions", (SampleQuery("actions:published:command"),), shape=(2, 3)),
        InputField("language", (SampleQuery("instruction"),)),
    ))
    assembler = InputAssembler(spec, {"concat": np.concatenate})
    result = assembler.assemble(LatestValues({
        "joint:left": np.array([1, 2]), "joint:right": np.array([3]),
        "actions:published:command": np.zeros((2, 3)), "instruction": "move",
        "unused": object(),
    }))
    np.testing.assert_array_equal(result["custom.state"], [1, 2, 3])
    assert result["language"] == "move"
    assert set(result) == {"custom.state", "previous_actions", "language"}
    assert "unused" not in spec.sources


def test_requested_inputs_can_use_separate_providers_without_model_branches():
    observation_spec = history_spec()
    store = HistoryStore(observation_spec)
    store.append("camera:eye", 1, 1., np.array([1.]))
    store.append("camera:eye", 2, 2., np.array([2.]))
    spec = InputSpec(observation_spec.fields + (
        InputField("instruction", (SampleQuery("instruction"),)),
        InputField("feedback", (SampleQuery("actions:published:command"),)),
    ))
    execution = LatestValues({"actions:published:command": np.array([[3., 4.]])})
    providers = {
        "camera:eye": store,
        "instruction": LatestValues({"instruction": "move"}),
        "actions:published:command": execution,
    }
    routed = RoutedInputs(spec, providers)
    result = InputAssembler(spec, {"stack": np.stack}).assemble(routed, anchor_s=2.)
    np.testing.assert_array_equal(result["video"], [[1.], [2.]])
    np.testing.assert_array_equal(result["feedback"], [[3., 4.]])
    assert result["instruction"] == "move"
    with pytest.raises(ValueError, match="Missing or stale"):
        InputAssembler(spec, {"stack": np.stack}).assemble(routed, anchor_s=5.)
    with pytest.raises(ValueError, match="exactly"):
        RoutedInputs(spec, {**providers, "unused": execution})
    with pytest.raises(ValueError, match="exactly"):
        RoutedInputs(spec, {"camera:eye": store})
    with pytest.raises(ValueError, match="Undeclared"):
        routed.resolve(SampleQuery("unused"), 2.)


def test_latest_values_do_not_fake_history_or_timestamps():
    latest = LatestValues({"camera:eye": np.zeros(2)})
    for q in (SampleQuery("camera:eye", (-1., 0.)), SampleQuery("camera:eye", max_age_s=1)):
        with pytest.raises(ValueError, match="timestamped"):
            latest.resolve(q, 1)


def test_history_only_allocates_declared_sources_and_selects_causally():
    spec = history_spec()
    store = HistoryStore(spec)
    store.append("unused", 0, 0, np.ones(1_000_000))
    assert store.bytes_used == 0
    store.append("camera:eye", 1, 1., np.array([1.]))
    store.append("camera:eye", 2, 2., np.array([2.]))
    store.append("camera:eye", 3, 2.05, np.array([3.]))
    result = InputAssembler(spec, {"stack": np.stack}).assemble(store, anchor_s=2.)
    np.testing.assert_array_equal(result["video"], [[1.], [2.]])
    assert store.sources == {"camera:eye"}


def test_latest_sources_are_independent_and_do_not_evict_each_other():
    spec = InputSpec(tuple(InputField(k, (SampleQuery(k),)) for k in ("a", "b")))
    store = HistoryStore(spec)
    store.append("a", 0, 1., np.array([1]))
    store.append("b", 0, 2., np.array([2]))
    assert set(InputAssembler(spec).assemble(store, anchor_s=2.)) == {"a", "b"}


def test_missing_stale_repeated_and_future_samples_fail():
    spec = history_spec()
    store = HistoryStore(spec)
    assembler = InputAssembler(spec, {"stack": np.stack})
    with pytest.raises(ValueError, match="Missing"):
        assembler.assemble(store, anchor_s=2.)
    store.append("camera:eye", 1, 2., np.array([2]))
    with pytest.raises(ValueError, match="Missing"):
        assembler.assemble(store, anchor_s=2.)
    with pytest.raises(ValueError, match="Missing"):
        assembler.assemble(store, anchor_s=3.5)
    with pytest.raises(ValueError, match="sequences"):
        store.append("camera:eye", 1, 2.1, np.array([2]))
    with pytest.raises(ValueError, match="backwards"):
        store.append("camera:eye", 2, 1., np.array([2]))


def test_distinct_temporal_offsets_cannot_reuse_one_frame():
    spec = history_spec((-0.1, 0.), age=1.)
    store = HistoryStore(spec)
    store.append("camera:eye", 1, 0., np.array([1]))
    with pytest.raises(ValueError, match="distinct"):
        InputAssembler(spec, {"stack": np.stack}).assemble(store, anchor_s=0.2)


def test_memory_exhaustion_is_latched_until_reset():
    spec = history_spec()
    store = HistoryStore(spec, max_bytes=16)
    store.append("camera:eye", 0, 0., np.array([1.]))
    store.append("camera:eye", 1, 0.1, np.array([1.]))
    with pytest.raises(MemoryError):
        store.append("camera:eye", 2, 0.2, np.array([1.]))
    with pytest.raises(RuntimeError):
        store.resolve(SampleQuery("camera:eye"), 0.2)
    store.reset()
    assert store.bytes_used == 0
    store.append("camera:eye", 0, 0., np.array([2.]))


def test_history_owns_values_and_returns_independent_arrays():
    store = HistoryStore(history_spec())
    value = np.array([1.])
    store.append("camera:eye", 0, 1., value)
    value[0] = 9
    q = SampleQuery("camera:eye")
    result = store.resolve(q, 1.)[0]
    result[0] = 8
    assert store.resolve(q, 1.)[0][0] == 1
    with pytest.raises(ValueError, match="exceeds declared"):
        store.resolve(SampleQuery("camera:eye", (-5., 0.), 1.), 1.)


@pytest.mark.parametrize("kwargs", [
    {"offsets_s": (1.,)}, {"offsets_s": (float("nan"),)},
    {"offsets_s": (0., -1.)}, {"offsets_s": (0., 0.)}, {"max_age_s": -1.},
])
def test_invalid_temporal_queries(kwargs):
    with pytest.raises(ValueError):
        SampleQuery("x", **kwargs)


def test_spec_errors_are_not_silently_fixed():
    field = InputField("x", (SampleQuery("x"),))
    with pytest.raises(ValueError, match="unique"):
        InputSpec((field, field))
    with pytest.raises(ValueError, match="Unknown"):
        InputAssembler(InputSpec((InputField("x", (SampleQuery("x"),), "unknown"),)))
    with pytest.raises(ValueError, match="expected shape"):
        InputAssembler(InputSpec((InputField("x", (SampleQuery("x"),), shape=(2,)),))).assemble(
            LatestValues({"x": np.zeros(3)})
        )
