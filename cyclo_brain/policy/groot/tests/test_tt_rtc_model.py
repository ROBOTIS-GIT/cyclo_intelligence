"""CPU proof of tokenwise flow inference using the actual deployment modules.

No backbone weights, model downloads, robot clients or GPU contexts are needed.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
from transformers.feature_extraction_utils import BatchFeature

OVERLAY = Path(__file__).resolve().parents[1] / "model_overrides"


@pytest.fixture
def modules(monkeypatch):
    # Import the real action head without importing/loading Cosmos. Package
    # registries are isolated so this also coexists with the factory stub tests.
    for name in ("gr00t", "gr00t.configs", "gr00t.configs.model", "gr00t.model",
                 "gr00t.model.modules", "gr00t.model.gr00t_n1d7"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    sys.modules["gr00t.configs.model"].register_model_config = lambda *_args: None
    monkeypatch.setattr(transformers.AutoConfig, "register", lambda *_args: None)
    monkeypatch.setattr(transformers.AutoModel, "register", lambda *_args: None)
    loaded = []
    for name in ("gr00t.configs.model.gr00t_n1d7", "gr00t.model.modules.dit",
                 "gr00t.model.modules.embodiment_conditioned_mlp",
                 "gr00t.model.gr00t_n1d7.gr00t_n1d7"):
        spec = importlib.util.spec_from_file_location(name, OVERLAY / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded[0], loaded[-1]


def tiny_head(modules, *, alternate=True, enabled=True):
    config_module, head_module = modules
    config = config_module.Gr00tN1d7Config(
        training_time_rtc_enabled=enabled, training_time_rtc_max_delay_steps=8, logical_action_horizon=32, logical_action_dim=16,
        tt_rtc_max_delay_steps=8, num_inference_timesteps=4, num_timestep_buckets=1000,
        hidden_size=16, input_embedding_dim=16, backbone_embedding_dim=16,
        max_num_embodiments=1, max_state_dim=16, max_action_dim=132, action_horizon=40,
        max_seq_len=40, state_dropout_prob=0, tune_vlln=False,
        use_alternate_vl_dit=alternate,
        vl_self_attention_cfg=dict(num_attention_heads=2, attention_head_dim=8,
                                   num_layers=1, dropout=0, positional_embeddings=None),
        diffusion_model_cfg=dict(num_attention_heads=2, attention_head_dim=8,
                                 num_layers=2, output_dim=16, dropout=0,
                                 final_dropout=False, positional_embeddings=None,
                                 interleave_self_attention=True),
    )
    return head_module.Gr00tN1d7ActionHead(config).eval()


def features():
    return dict(backbone_features=torch.randn(2, 4, 16), state_features=torch.randn(2, 1, 16),
                embodiment_id=torch.zeros(2, dtype=torch.long), action_input=BatchFeature({}),
                backbone_output=BatchFeature(dict(image_mask=torch.tensor([[1, 1, 0, 0]] * 2).bool(),
                                                   backbone_attention_mask=torch.ones(2, 4).bool())))


@pytest.mark.parametrize("alternate", [True, False])
@pytest.mark.parametrize("delay", [0, 1, 7, 8])
def test_four_steps_clamp_clean_prefix_and_match_training_times(modules, alternate, delay):
    head = tiny_head(modules, alternate=alternate)
    prefix = torch.randn(2, delay, 16).clamp(-1, 1)
    encoder_inputs, dit_inputs = [], []
    head.action_encoder.register_forward_pre_hook(
        lambda _module, args: encoder_inputs.append((args[0].clone(), args[1].clone())))
    head.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: dit_inputs.append(kwargs["timestep"].clone()), with_kwargs=True)
    inputs = features()
    inputs.pop("action_input")
    padded_prefix = torch.nn.functional.pad(prefix, (0, 116))
    result = head.get_action_tt_rtc_with_features(
        **inputs, normalized_action_prefix=padded_prefix,
        delay_steps=delay, logical_action_horizon=32,
    )["action_pred"]
    assert result.shape == (2, 40, 132) and torch.isfinite(result).all()
    assert torch.equal(result[:, :delay, :16], prefix)
    assert torch.count_nonzero(result[:, :delay, 16:]) == 0
    assert len(encoder_inputs) == len(dit_inputs) == 4
    for step, ((action, codes), dit_codes) in enumerate(zip(encoder_inputs, dit_inputs)):
        assert torch.equal(action[:, :delay, :16], prefix)
        assert torch.count_nonzero(action[:, :delay, 16:]) == 0
        assert (codes[:, :delay] == 1000).all()
        assert (codes[:, delay:] == step * 250).all()
        assert torch.equal(dit_codes[:, 1:], codes)
        assert (dit_codes[:, 0] == step * 250).all()
        assert dit_codes.shape == (2, 41)



@pytest.mark.parametrize("alternate", [True, False])
def test_zero_prefix_matches_standard_flow(modules, alternate):
    head = tiny_head(modules, alternate=alternate)
    standard = tiny_head(modules, alternate=alternate, enabled=False)
    standard.load_state_dict(head.state_dict(), strict=True)
    inputs = features()
    torch.manual_seed(123)
    expected = standard.get_action_with_features(**inputs)["action_pred"]
    inputs.pop("action_input")
    torch.manual_seed(123)
    actual = head.get_action_tt_rtc_with_features(
        **inputs, normalized_action_prefix=torch.zeros(2, 0, 132),
        delay_steps=0, logical_action_horizon=32,
    )["action_pred"]
    assert torch.allclose(actual, expected, atol=2e-6)
