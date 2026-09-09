"""Opt-in tests against real pinned LeRobot dependencies; no weight downloads.

Run separately from tests that stub LeRobot modules, inside the candidate Worker:
CYCLO_TEST_POLICY_DEPENDENCIES=1 python -m pytest tests/test_new_policy_processors.py
"""

import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("CYCLO_TEST_POLICY_DEPENDENCIES") != "1":
    pytest.skip("requires the candidate LeRobot Worker dependencies", allow_module_level=True)

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline


MODELS = ["eo1", "evo1", "wall_x", "pi0_fast", "groot"]
FEATURES = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
    "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 40)),
}
OUTPUT = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
STATS = {
    key: {"min": torch.zeros(6), "max": torch.ones(6), "mean": torch.zeros(6), "std": torch.ones(6)}
    for key in ("observation.state", "action")
}


def batch():
    return {
        "observation.state": torch.zeros(1, 6),
        "observation.images.head": torch.zeros(1, 3, 48, 64),
        "observation.images.wrist": torch.ones(1, 3, 32, 40),
        "task": ["pick up the object"],
    }


@pytest.mark.parametrize("name", MODELS)
def test_real_policy_import_and_saved_config_roundtrip(name, tmp_path):
    cls = get_policy_class(name)
    importlib.import_module(f"lerobot.policies.{name}.processor_{name}")
    kwargs = {}
    if name == "eo1":
        from transformers import Qwen2_5_VLConfig

        # Supply local config metadata; do not download the default Qwen config.
        kwargs["vlm_config"] = Qwen2_5_VLConfig().to_dict()
    config = cls.config_class(device="cpu", input_features=FEATURES, output_features=OUTPUT, **kwargs)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.type == name
    assert restored.input_features == config.input_features


@pytest.mark.parametrize("name", ["wall_x", "evo1"])
def test_saved_real_processors_preserve_mixed_cameras_and_chunk_shape(name, tmp_path):
    config = get_policy_class(name).config_class(device="cpu", input_features=FEATURES, output_features=OUTPUT)
    pre, post = make_pre_post_processors(config, dataset_stats=STATS)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre, post = make_pre_post_processors(
        config, pretrained_path=tmp_path, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    for _ in range(2):
        observed = pre(batch())
        assert observed["observation.images.head"].shape[-2:] == (48, 64)
        assert observed["observation.images.wrist"].shape[-2:] == (32, 40)
        result = post(torch.zeros(1, 4, config.max_action_dim if name == "evo1" else 6))
        assert result.shape == (1, 4, 6)
        assert torch.isfinite(result).all()


def test_eo1_real_conversation_keeps_camera_shapes_and_instruction():
    from lerobot.policies.eo1.processor_eo1 import EO1ConversationTemplateStep

    pre = PolicyProcessorPipeline(steps=[EO1ConversationTemplateStep(input_features=FEATURES, chunk_size=4)])
    result = pre(batch())
    content = result["messages"][0][1]["content"]
    assert content[0]["image"].shape == (3, 48, 64)
    assert content[1]["image"].shape == (3, 32, 40)
    assert "pick up the object" in content[2]["text"]


def test_groot_validation_accepts_real_serialized_pack_and_decode(tmp_path):
    from lerobot.policies.groot.processor_groot import (
        GrootN17PackInputsStep, GrootN17VLMEncodeStep, GrootActionUnpackUnnormalizeStep,
    )
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    pack = GrootN17PackInputsStep(stats=STATS)
    # Reproduce the pinned pack step's pre-resize stack requirement with real code.
    pack_only = PolicyProcessorPipeline(steps=[pack])
    with pytest.raises(ValueError, match="same shape"):
        pack_only(batch())
    aligned = batch()
    aligned["observation.images.wrist"] = torch.ones(1, 3, 48, 64)
    assert pack_only(aligned)["video"].shape == (1, 1, 2, 48, 64, 3)
    pre = PolicyProcessorPipeline(
        steps=[pack, GrootN17VLMEncodeStep()], name="policy_preprocessor"
    )
    post = PolicyProcessorPipeline(
        steps=[GrootActionUnpackUnnormalizeStep(env_action_dim=6, stats=STATS)],
        name="policy_postprocessor", to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "new_policy_validation", Path(__file__).resolve().parents[1] / "lerobot_engine/policy_validation.py"
    )
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    validation.validate_checkpoint({
        "type": "groot", "base_model_path": str(tmp_path), "embodiment_tag": "new_embodiment"
    }, tmp_path)
    result = post(torch.zeros(1, 4, 132))
    assert result.shape == (1, 4, 6)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("resize_first", [True, False])
def test_groot_engine_runs_saved_resize_before_camera_packing(tmp_path, resize_first):
    """Real serialized image/pack steps; model weights and robot I/O are excluded."""
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
    from lerobot.processor.hil_processor import ImageCropResizeProcessorStep
    from lerobot_engine.engine import LeRobotEngine

    steps = [ImageCropResizeProcessorStep(resize_size=(256, 256))] if resize_first else []
    steps.append(GrootN17PackInputsStep(stats=STATS))
    pre = PolicyProcessorPipeline(steps=steps, name="policy_preprocessor")
    pre.save_pretrained(tmp_path)
    engine = LeRobotEngine()
    engine._preprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="policy_preprocessor.json", local_files_only=True,
    )
    engine._postprocessor = lambda value: value
    engine._policy = SimpleNamespace(config=SimpleNamespace(type="groot"))
    engine._robot = object()
    engine._image_preprocessing = object()
    engine._cameras = {"head": "observation.images.head", "wrist": "observation.images.wrist"}
    def observation(_instruction):
        raw = batch()
        raw["observation.images.head"] = torch.zeros(1, 3, 376, 672)
        raw["observation.images.wrist"] = torch.ones(1, 3, 424, 240)
        return raw
    engine._build_observation = observation
    predicted = []
    def predict(processed):
        predicted.append(processed["video"].shape)
        return torch.zeros(1, 4, 6)
    engine._predict_chunk = predict
    for _ in range(2):
        result = engine.get_action_chunk(SimpleNamespace(task_instruction="pick"))
        if resize_first:
            assert result["success"], result
            assert predicted[-1] == (1, 1, 2, 256, 256, 3)
            assert result["chunk_size"] == 4
            assert result["action_dim"] == 6
        else:
            assert not result["success"]
            assert "same shape" in result["message"]
            assert not predicted


@pytest.mark.skipif(not os.environ.get("CYCLO_ACT_SMOKE_PATH"), reason="requires a local ACT checkpoint")
def test_actual_act_checkpoint_load_predict_clear_reload(monkeypatch):
    """Real weights/processors and synthetic observations; no ROS or robot commands."""
    import numpy as np
    from lerobot_engine.engine import LeRobotEngine

    engine = LeRobotEngine()

    def attach_synthetic_robot(_robot_type):
        config = engine._policy.config
        images = {
            key.removeprefix("observation.images."): np.zeros((*feature.shape[1:], 3), np.uint8)
            for key, feature in config.input_features.items() if key.startswith("observation.images.")
        }
        engine._robot = SimpleNamespace(
            _config={"cameras": {}}, close=lambda: None,
            get_images=lambda **_: images,
            get_joint_positions=lambda: {"follower_arm": np.zeros(config.input_features["observation.state"].shape[0])},
        )
        engine._cameras = {name: f"observation.images.{name}" for name in images}
        engine._state_modalities = ["arm"]
        engine._action_keys = ["arm"]

    monkeypatch.setattr(engine, "_init_robot", attach_synthetic_robot)
    request = SimpleNamespace(model_path=os.environ["CYCLO_ACT_SMOKE_PATH"], robot_type="synthetic")
    try:
        for _ in range(2):
            loaded = engine.load_policy(request)
            assert loaded["success"], loaded
            assert engine._policy.config.type == "act"
            for _ in range(2):
                action = engine.get_action_chunk(SimpleNamespace(task_instruction=""))
                assert action["success"], action
                assert action["chunk_size"] > 0
                assert action["action_dim"] == engine._policy.config.output_features["action"].shape[0]
                assert np.isfinite(action["action_chunk"]).all()
            engine.cleanup()
            assert not engine.is_ready
    finally:
        engine.cleanup()
