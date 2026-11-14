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
        training_time_rtc=enabled, logical_action_horizon=32, logical_action_dim=16,
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
    result = head.get_action_with_features(**features(), options={"ttrtc_prefix": prefix})["action_pred"]
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
def test_cold_start_matches_standard_flow_and_parameter_layout(modules, alternate):
    head = tiny_head(modules, alternate=alternate)
    standard = tiny_head(modules, alternate=alternate, enabled=False)
    standard.load_state_dict(head.state_dict(), strict=True)
    inputs = features()
    torch.manual_seed(123)
    expected = standard.get_action_with_features(**inputs)["action_pred"]
    torch.manual_seed(123)
    actual = head.get_action_with_features(**inputs,
                                           options={"ttrtc_prefix": torch.zeros(2, 0, 16)})["action_pred"]
    assert torch.allclose(actual, expected, atol=2e-6)


def test_nonempty_prefix_changes_postfix_and_invalid_prefix_rejected(modules):
    head = tiny_head(modules)
    inputs = features()
    torch.manual_seed(321)
    first = head.get_action_with_features(**inputs, options={"ttrtc_prefix": torch.zeros(2, 8, 16)})
    torch.manual_seed(321)
    second = head.get_action_with_features(**inputs, options={"ttrtc_prefix": torch.ones(2, 8, 16)})
    assert not torch.allclose(first["action_pred"][:, 8:32, :16], second["action_pred"][:, 8:32, :16])
    for prefix in (torch.zeros(2, 9, 16), torch.zeros(1, 7, 16), torch.zeros(2, 7, 15),
                   torch.full((2, 1, 16), float("nan"))):
        with pytest.raises(ValueError, match="prefix"):
            head.get_action_with_features(**inputs, options={"ttrtc_prefix": prefix})
    with pytest.raises(ValueError, match="explicit"):
        head.get_action_with_features(**inputs)
    standard = tiny_head(modules, enabled=False)
    with pytest.raises(ValueError, match="training_time_rtc"):
        standard.get_action_with_features(**inputs, options={"ttrtc_prefix": torch.zeros(2, 0, 16)})


def _policy_method(name):
    # Only the adapter methods need fake observations; numerical flow above uses
    # real modules. Compile their source to avoid unrelated VLM imports.
    path = OVERLAY / "gr00t/policy/gr00t_policy.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "Gr00tPolicy")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np, "torch": torch, "Any": object,
                 "MessageType": types.SimpleNamespace(EPISODE_STEP=types.SimpleNamespace(value="step")),
                 "_rec_to_dtype": lambda value, **_: value}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_policy_normalizes_in_trained_key_order_and_preserves_committed_physical_prefix():
    keys = ["left", "right"]
    prefix = {"right": np.full((1, 7, 8), 30., np.float32),
              "left": np.full((1, 7, 8), 10., np.float32)}
    calls = []
    def apply_action(action, tag, state):
        calls.append((action, tag, state))
        return {"left": action["left"] / 10 - 1, "right": action["right"] / 10 - 1}
    sap = types.SimpleNamespace(norm_params={"new_embodiment": {"action": {
        key: {"dim": np.array(8)} for key in keys}}}, apply_action=apply_action)
    def model_get_action(**kwargs):
        normalized = kwargs["options"]["ttrtc_prefix"]
        np.testing.assert_array_equal(normalized[:, :, :8], 0)
        np.testing.assert_array_equal(normalized[:, :, 8:], 2)
        return {"action_pred": torch.zeros(1, 40, 132)}
    class Processor:
        state_action_processor = sap
        def __call__(self, _messages):
            return {}
        def decode_action(self, *_args):
            return {key: np.ones((1, 32, 8), np.float32) for key in keys}
    policy = types.SimpleNamespace(
        model=types.SimpleNamespace(config=types.SimpleNamespace(training_time_rtc=True),
                                    get_action=model_get_action),
        processor=Processor(), embodiment_tag=types.SimpleNamespace(value="new_embodiment"),
        modality_configs={key: types.SimpleNamespace(modality_keys=keys) for key in ("action", "state")},
        _unbatch_observation=lambda _obs: [{}],
        _to_vla_step_data=lambda _obs: types.SimpleNamespace(states={key: np.zeros((1, 8)) for key in keys}),
        collate_fn=lambda _inputs: {"inputs": {}},
    )
    policy._ttrtc_model_options = types.MethodType(_policy_method("_ttrtc_model_options"), policy)
    result, _ = _policy_method("_get_action")(policy, {},
        options={"ttrtc_enabled": True, "ttrtc_prefix_actions": prefix})
    assert len(calls) == 1 and calls[0][1] == "new_embodiment"
    for key in keys:
        np.testing.assert_array_equal(result[key][:, :7], prefix[key])
        np.testing.assert_array_equal(result[key][:, 7:], 1)


@pytest.mark.parametrize("delay", [0, 7, 8])
def test_real_checkpoint_processor_minmax_clipping_and_roundtrip(delay):
    # GR00T's installed data utilities supply its exact normalization formulas;
    # load the processor from the deployed overlay, without any model imports.
    pytest.importorskip("gr00t.data.utils", reason="requires installed GR00T data dependencies")
    from gr00t.configs.data.embodiment_configs import (
        ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
    )
    path = OVERLAY / "gr00t/data/state_action/state_action_processor.py"
    spec = importlib.util.spec_from_file_location("ttrtc_deployment_state_action_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = ["arm_left", "arm_right"]
    action_config = ModalityConfig(
        delta_indices=list(range(0, 64, 2)), modality_keys=keys,
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                     type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)] * 2,
    )
    configs = {"new_embodiment": {"action": action_config,
                "state": ModalityConfig(delta_indices=[0], modality_keys=keys)}}
    def stats(low, high):
        # Percentiles deliberately differ so accidentally enabling them fails.
        return {"min": [low] * 8, "max": [high] * 8, "mean": [(low + high) / 2] * 8,
                "std": [1.] * 8, "q01": [low + 0.5] * 8, "q99": [high - 0.5] * 8}
    processor = module.StateActionProcessor(
        modality_configs=configs, use_percentiles=False, clip_outliers=True,
        use_relative_action=False, statistics={"new_embodiment": {
            "action": {"arm_left": stats(-2, 2), "arm_right": stats(10, 30)},
            "state": {key: stats(-1000, 1000) for key in keys},
        }},
    )
    # Wire order differs from trained order and groups have distinct units/ranges.
    prefix = {"arm_right": np.tile([0, 10, 15, 20, 25, 30, 40, 20], (1, delay, 1)).astype(np.float32),
              "arm_left": np.tile([-10, -2, -1, 0, 1, 2, 10, 0], (1, delay, 1)).astype(np.float32)}
    policy = types.SimpleNamespace(
        processor=types.SimpleNamespace(state_action_processor=processor),
        embodiment_tag=types.SimpleNamespace(value="new_embodiment"),
        modality_configs=configs["new_embodiment"],
    )
    normalized = _policy_method("_ttrtc_model_options")(policy,
        {"ttrtc_enabled": True, "ttrtc_prefix_actions": prefix},
        [{key: np.zeros((1, 8), np.float32) for key in keys}])["ttrtc_prefix"]
    expected = np.tile([-1, -1, -.5, 0, .5, 1, 1, 0] * 2, (1, delay, 1))
    np.testing.assert_allclose(normalized, expected, atol=1e-6)
    decoded = processor.unapply_action(
        {"arm_left": normalized[0, :, :8], "arm_right": normalized[0, :, 8:]}, "new_embodiment")
    np.testing.assert_allclose(decoded["arm_left"], np.clip(prefix["arm_left"][0], -2, 2), atol=1e-6)
    np.testing.assert_allclose(decoded["arm_right"], np.clip(prefix["arm_right"][0], 10, 30), atol=1e-6)
