from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_engine.native import load_policy


@pytest.mark.parametrize("invalid_weights", [False, True])
def test_checkpoint_is_memory_mapped_and_loaded_strictly(monkeypatch, tmp_path, invalid_weights):
    calls = []
    weights = {"weight": object()}
    config = SimpleNamespace(
        framework=SimpleNamespace(qwenvl=SimpleNamespace(base_vlm="owner/base")),
        trainer=SimpleNamespace(pretrained_checkpoint="training-only.pt"),
    )

    class Policy:
        def __init__(self, cfg):
            assert cfg.trainer.pretrained_checkpoint is None

        def load_state_dict(self, value, *, strict):
            assert value is weights and strict is True
            calls.append("load_state_dict")
            if invalid_weights:
                raise RuntimeError("incompatible checkpoint")

        def to(self, *, dtype, device):
            assert dtype == "bfloat16" and device == "cuda"
            calls.append("to")
            return self

        def eval(self):
            calls.append("eval")
            return self

    checkpoint = tmp_path / "model.pt"

    def load(path, *, map_location, weights_only, mmap):
        assert path == checkpoint
        assert map_location == "cpu" and weights_only is True and mmap is True
        calls.append("load")
        return weights

    for name, attrs in {
        "torch": {"load": load, "bfloat16": "bfloat16"},
        "omegaconf": {"OmegaConf": SimpleNamespace(create=lambda _: config)},
        "ABot.model.framework.ABot_M0": {"ABot_M0": Policy},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("ABOT_BASE_VLM", raising=False)
    bundle = SimpleNamespace(config={}, checkpoint=checkpoint, path=tmp_path)
    if invalid_weights:
        with pytest.raises(RuntimeError, match="incompatible checkpoint"):
            load_policy(bundle)
        assert calls == ["load", "load_state_dict"]
    else:
        assert isinstance(load_policy(bundle), Policy)
        assert calls == ["load", "load_state_dict", "to", "eval"]
