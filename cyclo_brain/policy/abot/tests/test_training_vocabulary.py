"""Use the real Qwen3-VL resize API with small CPU-only models."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_training.initialization import load_pretrained


class Tokenizer:
    all_special_ids = [0, 1, 2]

    def get_vocab(self):
        return {str(i): i for i in range(16)}


def model(rows, dim=14, tied=False):
    config = Qwen3VLConfig(
        text_config={"vocab_size": rows, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
                     "tie_word_embeddings": tied},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "out_hidden_size": 16, "deepstack_visual_indexes": []},
        tie_word_embeddings=tied,
    )
    result = torch.nn.Module()
    result.qwen_vl_interface = torch.nn.Module()
    result.qwen_vl_interface.model = Qwen3VLForConditionalGeneration(config)
    result.qwen_vl_interface.processor = SimpleNamespace(tokenizer=Tokenizer())
    result.action_model = torch.nn.Module()
    for name, a, b in (("state_encoder", dim, 16), ("action_encoder", dim, 16),
                       ("action_decoder", 16, dim)):
        head = torch.nn.Module()
        head.layer1 = torch.nn.Linear(a, 16)
        head.layer2 = torch.nn.Linear(16, b)
        setattr(result.action_model, name, head)
    return result


@pytest.mark.parametrize("rows", [16, 24])
def test_preserves_all_pretrained_rows_and_resets_only_dimension_heads(rows):
    source, target = model(rows), model(16, 22)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    tokenizer = target.qwen_vl_interface.processor.tokenizer
    vocab_before = deepcopy(tokenizer.get_vocab())
    report = load_pretrained(target, source.state_dict(), restore_vocabulary=True)
    assert report["vocabulary"]["checkpoint_rows"] == rows
    assert report["vocabulary"]["restored_rows"] == rows - 16
    assert len(report["reinitialized_heads"]) == 3
    for key, value in target.state_dict().items():
        expected = before[key] if key.startswith("action_model.") else source.state_dict()[key]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert tokenizer is target.qwen_vl_interface.processor.tokenizer
    assert tokenizer.get_vocab() == vocab_before
    assert target.qwen_vl_interface.model.config.text_config.vocab_size == rows


@pytest.mark.parametrize("failure", ["smaller", "different_rows", "hidden_size", "unexpected_key",
                                     "bad_token", "bad_special_token", "unrelated_shape", "tied"])
def test_invalid_vocabulary_fails_before_resizing(failure):
    source, target = model(24), model(16)
    weights = source.state_dict()
    prefix = "qwen_vl_interface.model."
    vlm = target.qwen_vl_interface.model
    input_key = next(k for k, v in target.state_dict(keep_vars=True).items()
                     if v is vlm.get_input_embeddings().weight)
    output_key = next(k for k, v in target.state_dict(keep_vars=True).items()
                      if v is vlm.get_output_embeddings().weight)
    if failure == "smaller":
        weights[input_key] = weights[input_key][:8]
        weights[output_key] = weights[output_key][:8]
    elif failure == "different_rows":
        weights[output_key] = weights[output_key][:20]
    elif failure == "hidden_size":
        weights[input_key] = torch.ones(24, 20)
    elif failure == "unexpected_key":
        weights[prefix + "unexpected.weight"] = torch.ones(1)
    elif failure == "bad_token":
        target.qwen_vl_interface.processor.tokenizer.get_vocab = lambda: {"bad": 24}
    elif failure == "bad_special_token":
        target.qwen_vl_interface.processor.tokenizer.all_special_ids = [30]
    elif failure == "unrelated_shape":
        weights["action_model.state_encoder.layer2.weight"] = torch.ones(32, 16)
    else:
        vlm.get_output_embeddings().weight = vlm.get_input_embeddings().weight
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError):
        load_pretrained(target, weights, restore_vocabulary=True)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert vlm.config.text_config.vocab_size == 16


def test_resize_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="Shapes differ"):
        load_pretrained(model(16), model(24).state_dict())


def test_identical_pretrained_weights_preserve_tying_after_resize_and_reload(tmp_path):
    source, target = model(24, tied=True), model(16, 22, tied=True)
    weights = {k: v.clone() for k, v in source.state_dict().items()}
    report = load_pretrained(target, weights, restore_vocabulary=True)
    assert report["vocabulary"]["tied_embeddings_preserved"] is True
    vlm = target.qwen_vl_interface.model
    assert vlm.get_input_embeddings().weight is vlm.get_output_embeddings().weight
    for key, value in target.state_dict().items():
        if not key.startswith("action_model."):
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    vlm.save_pretrained(tmp_path)
    restored = Qwen3VLForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    torch.testing.assert_close(restored.get_input_embeddings().weight, vlm.get_input_embeddings().weight)


def test_saved_config_reconstructs_expanded_qwen(tmp_path):
    target = model(16)
    load_pretrained(target, model(24).state_dict(), restore_vocabulary=True)
    vlm = target.qwen_vl_interface.model
    vlm.save_pretrained(tmp_path)
    reloaded = Qwen3VLForConditionalGeneration.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.get_input_embeddings().weight.shape[0] == 24
    assert reloaded.get_output_embeddings().weight.shape[0] == 24
    for key, value in vlm.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)
    vlm.eval()
    reloaded.eval()
    with torch.no_grad():
        inputs = torch.tensor([[0, 2, 23]])
        expected = vlm(input_ids=inputs, use_cache=False).logits
        actual = reloaded(input_ids=inputs, use_cache=False).logits
    assert actual.shape == (1, 3, 24)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
