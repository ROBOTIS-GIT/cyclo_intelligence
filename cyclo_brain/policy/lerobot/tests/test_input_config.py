"""Public settings are sequential and never expose the internal input DAG."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from lerobot_engine.adapters.definition import AdapterDefinition
from lerobot_engine.input_config import ImageOperations, build_input_graph
from lerobot_engine.input_pipeline import CONFIG_DIR, load_input_pipeline
from lerobot_engine.image_preprocessing import ImagePreprocessing


HEAD = "observation.images.head"
WRIST = "observation.images.wrist"
FEATURES = {HEAD: {"shape": [3, 8, 10]}, WRIST: {"shape": [3, 6, 4]}}


def resize(backend="torch", size=None):
    return {"resize": {"backend": backend, "size": size or [8, 10], "interpolation": "bilinear"}}


def test_every_model_yaml_has_only_user_preprocessing():
    for path in CONFIG_DIR.glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert set(config) == {"preprocessing"}, path
        assert "sources:" not in path.read_text() and '"*"' not in path.read_text()
        build_input_graph(config, {"input_features": FEATURES})


@pytest.mark.parametrize("operations", ["identity", []])
def test_identity_preserves_pixels_with_basic_api_packing(operations):
    image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    original = image.copy()
    actual = ImageOperations(operations, {}, FEATURES).apply(image, HEAD)
    expected = torch.from_numpy(image.copy()).float().div(255).permute(2, 0, 1)[None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("backend", ["torch", "opencv"])
def test_operations_follow_written_order(backend):
    image = np.random.default_rng(3).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    crop = {"center_crop": {"backend": backend, "size": [12, 16]}}
    actual = ImageOperations([crop, resize(backend)], {}, FEATURES).apply(image, HEAD)
    prior = ImagePreprocessing({"backend": backend, "operations": [
        {"type": "center_crop", "size": [12, 16]},
        {"type": "resize", "size": [8, 10], "interpolation": "bilinear"},
    ]}, FEATURES)
    torch.testing.assert_close(actual, prior.apply(image, HEAD), rtol=0, atol=0)
    # Reversing these transforms is not silently reordered and cannot crop a larger image.
    with pytest.raises(ValueError, match="exceeds input"):
        ImageOperations([resize(backend), crop], {}, FEATURES).apply(image, HEAD)


def test_opencv_then_torch_does_not_quantize_between_steps():
    image = np.random.default_rng(4).integers(0, 256, (13, 17, 3), dtype=np.uint8)
    actual = ImageOperations([resize("opencv"), resize("torch", [4, 5])], {}, FEATURES).apply(image, HEAD)
    first = ImagePreprocessing({"backend": "opencv", "operations": [
        {"type": "resize", "size": [8, 10], "interpolation": "bilinear"}]}, FEATURES).apply(image, HEAD)
    expected = torch.nn.functional.interpolate(first, size=(4, 5), mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_camera_override_replaces_default_and_uses_each_checkpoint_size():
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    transform = ImageOperations([resize("opencv", "checkpoint")], {WRIST: "identity"}, FEATURES)
    assert transform.apply(image, HEAD).shape == (1, 3, 8, 10)
    assert transform.apply(image, WRIST).shape == (1, 3, 12, 20)
    all_cameras = ImageOperations([resize("opencv", "checkpoint")], {}, FEATURES)
    assert all_cameras.apply(image, WRIST).shape == (1, 3, 6, 4)


@pytest.mark.parametrize("config", [
    {}, {"sources": {}}, {"preprocessing": None}, {"preprocessing": "auto"},
    {"preprocessing": {"unknown": True}}, {"preprocessing": {"images": "auto"}},
    {"preprocessing": {"images": [{"resize": {"size": [8, 10]}}]}},
    {"preprocessing": {"images": [resize("torch"), resize("opencv")]}},
    {"preprocessing": {"cameras": {"typo": "identity"}}},
    {"preprocessing": {"images": [{"resize": {"backend": "opencv", "size": [8, 10],
                                               "interpolation": "area", "antialias": True}}]}},
    {"preprocessing": {"images": [{"resize": {"backend": "torch", "type": "identity"}}]}},
    {"preprocessing": {"custom": {"handler": "os.system"}}},
])
def test_invalid_settings_fail_before_model_allocation(config):
    with pytest.raises(ValueError):
        build_input_graph(config, {"input_features": FEATURES})


def test_named_handler_options_load_and_contract_are_model_scoped(tmp_path):
    def build(graph, options):
        if set(options) != {"after"} or options["after"] != "plan_terminal":
            raise ValueError("after must be plan_terminal")
        graph["execution"] = {"request_after": {"event": options["after"]}}
        graph["memory"] = {"slots": {"feature": {"event": options["after"]}}}
        return graph
    definition = AdapterDefinition(input_handlers={"feature_memory": build})
    (tmp_path / "config.json").write_text(json.dumps({"type": "act", "input_features": FEATURES}))
    config = {"preprocessing": {"custom": {"handler": "feature_memory", "options": {"after": "plan_terminal"}}}}
    path = tmp_path / "act.yaml"
    path.write_text(yaml.safe_dump(config))
    with patch("lerobot_engine.adapters.resolve_adapter", return_value=definition):
        pipeline = load_input_pipeline(tmp_path, tmp_path)
        assert pipeline.request_after.event == "plan_terminal"
        assert pipeline.requires_feedback
        assert pipeline.memory.conditions["feature"].event == "plan_terminal"
        config["preprocessing"]["custom"]["options"] = {"typo": True}
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="after must"):
            load_input_pipeline(tmp_path, tmp_path)
    path.write_text(yaml.safe_dump({"preprocessing": {"custom": {"handler": "feature_memory"}}}))
    with patch("lerobot_engine.adapters.resolve_adapter", return_value=AdapterDefinition()):
        with pytest.raises(ValueError, match="unregistered"):
            load_input_pipeline(tmp_path, tmp_path)


@pytest.mark.parametrize("handlers", [{"invalid.path": lambda g, o: g}, {"name": "module.py"}, []])
def test_handlers_must_be_explicitly_registered_code(handlers):
    with pytest.raises(ValueError, match="input_handlers"):
        AdapterDefinition(input_handlers=handlers)


def test_handler_registry_is_a_snapshot():
    handlers = {"example": lambda graph, options: graph}
    definition = AdapterDefinition(input_handlers=handlers)
    handlers.clear()
    assert "example" in definition.input_handlers
    with pytest.raises(TypeError):
        definition.input_handlers["injected"] = lambda g, o: g


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    (tmp_path / "config.json").write_text('{"type":"act"}')
    (tmp_path / "act.yaml").write_text("preprocessing: identity\npreprocessing: identity\n")
    with pytest.raises(ValueError, match="duplicate YAML"):
        load_input_pipeline(tmp_path, tmp_path)
