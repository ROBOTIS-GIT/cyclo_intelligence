import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_training.checkpoint import retain_checkpoints, save_checkpoint, verify_checkpoint


class Processor:
    def save_pretrained(self, path):
        path.mkdir()
        (path / "processor_config.json").write_text('{"test": true}')


def save(root, step=1, processor=None):
    metadata = {name: {"fixture": True} for name in (
        "profile.json", "train_config.json", "model_config.json", "dataset_statistics.json",
        "cyclo_io_mapping.json", "initialization.json",
    )}
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(2)
        model.bias.fill_(3)
    return save_checkpoint(root, step=step, model=model, processor=processor or Processor(), metadata=metadata)


def test_complete_checkpoint_reload_in_fresh_process(tmp_path):
    saved = save(tmp_path)
    receipt = verify_checkpoint(saved)
    assert receipt["optimizer_step"] == 1
    assert receipt["resume_supported"] is False
    code = """
import json, sys, torch
from pathlib import Path
from abot_training.checkpoint import verify_checkpoint
root = Path(sys.argv[1])
receipt = verify_checkpoint(root)
model = torch.nn.Linear(2, 1)
model.load_state_dict(torch.load(root/receipt['weights'], weights_only=True, mmap=True), strict=True)
assert model(torch.ones(1,2)).item() == 7
assert json.loads((root/'dataset_statistics.json').read_text()) == {'fixture':True}
assert json.loads((root/'processor/processor_config.json').read_text()) == {'test':True}
print('fresh-process-checkpoint-ok')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-c", code, str(saved)], env=env, check=True,
                            capture_output=True, text=True)
    assert "fresh-process-checkpoint-ok" in result.stdout


def test_no_overwrite_and_retention_after_save(tmp_path):
    save(tmp_path, 2)
    save(tmp_path, 10)
    with pytest.raises(FileExistsError):
        save(tmp_path, 10)
    retain_checkpoints(tmp_path, save_total_limit=1)
    assert not (tmp_path / "checkpoint-2").exists()
    assert (tmp_path / "checkpoint-10").exists()


@pytest.mark.parametrize("failure", ["corruption", "missing", "extra", "symlink"])
def test_checksum_and_inventory_fail_closed(tmp_path, failure):
    saved = save(tmp_path)
    target = saved / "dataset_statistics.json"
    if failure == "corruption":
        target.write_text("{}")
    elif failure == "missing":
        target.unlink()
    elif failure == "extra":
        (saved / "extra.json").write_text("{}")
    else:
        target.unlink()
        target.symlink_to(saved / "profile.json")
    with pytest.raises(ValueError):
        verify_checkpoint(saved)


def test_failed_save_keeps_previous_checkpoint(tmp_path):
    class BrokenProcessor:
        def save_pretrained(self, path):
            raise OSError("disk full")

    previous = save(tmp_path)
    with pytest.raises(OSError, match="disk full"):
        save(tmp_path, 2, processor=BrokenProcessor())
    assert verify_checkpoint(previous)["optimizer_step"] == 1
    assert list(tmp_path.iterdir()) == [previous]
