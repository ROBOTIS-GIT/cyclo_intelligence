from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_chunks import act_execution_steps, effective_chunk_size, requested_chunk_size


def test_caps_at_trained_horizon_without_padding():
    assert effective_chunk_size(100, 30) == 30
    assert effective_chunk_size(8, 24) == 8
    assert effective_chunk_size(0, 24) == 24


@pytest.mark.parametrize("value", [-1, 1.5, "", None, float("nan"), 65536])
def test_rejects_invalid_chunk_size(value):
    with pytest.raises(ValueError):
        requested_chunk_size(value)


def test_act_runtime_override_and_legacy_per_step_behavior():
    config = {"chunk_size": 30, "n_action_steps": 1, "temporal_ensemble_coeff": None}
    assert act_execution_steps(config, 0) == 1
    assert act_execution_steps(config, 100) == 30
    assert act_execution_steps(config, 8) == 8
    config["temporal_ensemble_coeff"] = 0.01
    assert act_execution_steps(config, 8) == 1
