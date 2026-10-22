from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_training.initialization import load_pretrained


def tiny_model(dim):
    def mlp(a, b):
        result = torch.nn.Module()
        result.layer1 = torch.nn.Linear(a, 4)
        result.layer2 = torch.nn.Linear(4, b)
        return result

    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(4, 4)
    model.action_model = torch.nn.Module()
    model.action_model.state_encoder = mlp(dim, 4)
    model.action_model.action_encoder = mlp(dim, 4)
    model.action_model.action_decoder = mlp(4, dim)
    return model


def test_only_mismatched_whole_heads_are_reinitialized():
    source, target = tiny_model(14), tiny_model(22)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    report = load_pretrained(target, source.state_dict())
    assert len(report["reinitialized_heads"]) == 3
    for key, value in target.state_dict().items():
        expected = before[key] if key.startswith("action_model.") else source.state_dict()[key]
        torch.testing.assert_close(value, expected)


def test_same_dimensions_load_all_weights():
    source, target = tiny_model(14), tiny_model(14)
    assert load_pretrained(target, source.state_dict())["reinitialized_heads"] == []
    for k, v in target.state_dict().items():
        torch.testing.assert_close(v, source.state_dict()[k])


@pytest.mark.parametrize("failure", ["missing_head", "extra", "backbone_shape", "hidden_head_shape"])
def test_backbone_and_missing_keys_fail_without_mutating_model(failure):
    source, target = tiny_model(14).state_dict(), tiny_model(22)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    if failure == "missing_head":
        source.pop("action_model.state_encoder.layer1.weight")
    elif failure == "extra":
        source["unknown.weight"] = torch.ones(1)
    elif failure == "backbone_shape":
        source["backbone.weight"] = torch.ones(8, 4)
    else:
        source["action_model.state_encoder.layer2.weight"] = torch.ones(8, 4)
    with pytest.raises(ValueError):
        load_pretrained(target, source)
    for k, v in target.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_disabled_vggt_is_explicitly_reported():
    source = tiny_model(14).state_dict()
    source["spatial_model.weight"] = torch.ones(1)
    report = load_pretrained(tiny_model(22), source)
    assert report["discarded_vggt_keys"] == ["spatial_model.weight"]
