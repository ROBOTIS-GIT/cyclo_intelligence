"""CPU checks for the narrow TT-RTC TensorRT adapter (no engine/GPU required)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture
def adapter():
    path = Path(__file__).resolve().parents[1] / "runtime/tt_rtc_trt.py"
    spec = importlib.util.spec_from_file_location("tt_rtc_trt_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("time_shape", [(1,), (1, 40)])
def test_rejects_wrong_timestep_engine(adapter, time_shape):
    runner = Mock()
    runner.engine.get_tensor_shape.side_effect = [time_shape, (1, 41, 1536)]
    source = SimpleNamespace(TensorRTDiTWrapper=lambda path: runner)
    with patch.dict(sys.modules, {"scripts.deployment.standalone_inference_script": source}):
        with pytest.raises(ValueError, match="per-token"):
            adapter.install_tt_rtc_dit(SimpleNamespace(), "wrong.trt")


def test_installs_only_dit_and_forwards_per_token_inputs(adapter):
    runner = Mock(return_value="dit-output")
    runner.engine.get_tensor_shape.side_effect = [(1, 41), (1, 41, 1536)]
    dit = SimpleNamespace(forward=None)
    head = SimpleNamespace(model=dit, action_decoder=object())
    policy = SimpleNamespace(model=SimpleNamespace(action_head=head))
    decoder = head.action_decoder
    source = SimpleNamespace(TensorRTDiTWrapper=lambda path: runner)
    with patch.dict(sys.modules, {"scripts.deployment.standalone_inference_script": source}):
        adapter.install_tt_rtc_dit(policy, "tt.trt")
    hidden = torch.zeros(1, 41, 1536)
    times = torch.zeros(1, 41, dtype=torch.int64)
    mask = torch.ones(1, 208, dtype=torch.bool)
    result = dit.forward(hidden, "vl", times, image_mask=mask, backbone_attention_mask=mask)
    assert result == "dit-output"
    assert head.action_decoder is decoder
    runner.assert_called_once_with(hidden, "vl", times,
                                   image_mask=mask, backbone_attention_mask=mask)
    with pytest.raises(ValueError, match="per-token"):
        dit.forward(hidden, "vl", torch.zeros(1))
