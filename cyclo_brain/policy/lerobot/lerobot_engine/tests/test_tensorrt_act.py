"""TensorRT must preserve ACT input order and the native policy interface."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

engine_root = Path(__file__).parents[1]
runtime_root = engine_root.parents[1] / "common/runtime"
sys.path.insert(0, str(runtime_root))
spec = importlib.util.spec_from_file_location("act_trt_under_test", engine_root / "tensorrt_act.py")
trt_act = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trt_act)
opt_spec = importlib.util.spec_from_file_location("act_opt_under_test", engine_root / "optimization.py")
optimization = importlib.util.module_from_spec(opt_spec)
opt_spec.loader.exec_module(optimization)


def test_adapter_keeps_camera_order_and_full_chunk():
    keys = ["observation.images.right", "observation.images.left"]
    cfg = SimpleNamespace(input_features={keys[0]: SimpleNamespace(shape=(3, 4, 4)),
                                          "observation.state": SimpleNamespace(shape=(8,)),
                                          keys[1]: SimpleNamespace(shape=(3, 4, 4))},
                          image_features={key: None for key in keys})
    captured = {}
    def call(inputs):
        captured.update(inputs)
        return torch.ones(1, 30, 8)
    with patch.object(trt_act, "TensorRTSession", return_value=call):
        model = trt_act.TensorRTACT("unused", {"inputs": trt_act.input_spec(cfg)}, cfg)
    right = torch.full((1, 3, 4, 4), 2.)
    left = torch.full((1, 3, 4, 4), 3.)
    actions, latent = model({"observation.state": torch.zeros(1, 8), "observation.images": [right, left]})
    assert tuple(actions.shape) == (1, 30, 8)
    assert latent == (None, None)
    assert torch.equal(captured["input_1"], right)
    assert torch.equal(captured["input_2"], left)


def test_acceleration_request_resolves_per_model_path():
    resolver = optimization.OptimizationMixin._resolve_policy_acceleration
    assert resolver("/model", SimpleNamespace()) == ("pytorch", "")
    assert resolver("/model", SimpleNamespace(acceleration_mode="tensorrt_act")) == ("tensorrt_act", "/model/act_model_fp32.trt")
    assert resolver("/model", SimpleNamespace(acceleration_mode="tensorrt_act", acceleration_engine_path="other.trt")) == ("tensorrt_act", "/model/other.trt")


def test_full_fp32_reference_restores_native_precision_settings():
    original = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    with trt_act.full_fp32_reference():
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == original
