"""Verify real safetensors merging without loading a neural network."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/create_ttrtc_mock_checkpoint.py"
spec = importlib.util.spec_from_file_location("create_ttrtc_mock_checkpoint", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def inputs(tmp_path):
    base, run = tmp_path / "base", tmp_path / "run"
    base.mkdir()
    (run / "processor").mkdir(parents=True)
    (run / "experiment_cfg").mkdir()
    embedding = "backbone.model.model.language_model.embed_tokens.weight"
    lm_head = "backbone.model.lm_head.weight"
    values = {embedding: torch.zeros(512, 512), lm_head: torch.zeros(512, 512),
              "action_head.test.weight": torch.tensor([[1., 2.], [3., 4.]])}
    save_file(values, base / "original.safetensors")
    builder.write_json(base / "model.safetensors.index.json",
                       {"weight_map": {key: "original.safetensors" for key in values}})
    overlay = tmp_path / "vlm.safetensors"
    save_file({embedding: torch.full((512, 512), 7.)}, overlay)
    builder.write_json(run / "experiment_cfg/final_model_config.json", dict(
        model_type="Gr00tN1d7", training_time_rtc=True, logical_action_horizon=32,
        logical_action_dim=16, action_horizon=40, max_action_dim=132))
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        builder.write_json(run / "processor" / name, {"original": name})
    return base, overlay, run, tmp_path / "output", values


def test_builder_merges_full_overlay_restores_alias_and_marks_untrained(inputs):
    base, overlay, run, output, original = inputs
    result = builder.build_checkpoint(base, overlay, run, output, max_shard_size_mb=1)
    assert result["tensor_count"] == 3 and result["shards"] == 2
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    actual = {}
    for name in set(index.values()):
        with safe_open(output / name, framework="pt", device="cpu") as reader:
            assert reader.metadata()["format"] == "pt"
            for key in reader.keys():
                assert key not in actual
                actual[key] = reader.get_tensor(key)
    for key, value in actual.items():
        assert value.dtype == torch.bfloat16
        if key.startswith("backbone."):
            assert torch.equal(value, torch.full_like(value, 7.))
        else:
            assert torch.equal(value.float(), original[key])
    manifest = json.loads((output / "experiment_cfg/tt_rtc_training_manifest.json").read_text())
    config = json.loads((output / "config.json").read_text())
    for record in (manifest, config):
        assert record["mock_checkpoint"] is True
        assert record["trained"] is False and record["complete"] is False and record["step"] == 0
    assert config["load_bf16"] is True and config["dtype"] == "bfloat16"
    assert manifest["inference_contract"]["action_fps"] == 15
    assert (output / "statistics.json").read_bytes() == (run / "processor/statistics.json").read_bytes()
    with safe_open(base / "original.safetensors", framework="pt", device="cpu") as source:
        assert torch.equal(source.get_tensor("backbone.model.lm_head.weight"), torch.zeros(512, 512))


def test_builder_refuses_existing_output_and_incomplete_overlay(inputs):
    base, overlay, run, output, _original = inputs
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing artifact")
    with pytest.raises(FileExistsError):
        builder.build_checkpoint(base, overlay, run, output)
    assert marker.read_text() == "existing artifact"
    invalid = overlay.parent / "invalid.safetensors"
    save_file({"action_head.wrong": torch.zeros(1)}, invalid)
    with pytest.raises(ValueError, match="complete backbone"):
        builder.build_checkpoint(base, invalid, run, output.parent / "not_created")
    assert not (output.parent / "not_created").exists()
