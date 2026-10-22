from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_training_data import prepared

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_training.channels import read_selection, resolve_channels
from abot_training.checkpoint import save_checkpoint
from abot_training.dataset import PreparedDataset


def selected_data(prepared, selection=None, *, padding=None, **kwargs):
    root, profile, _ = prepared
    profile = deepcopy(profile)
    profile["padding"] = padding or {"state_dim": "selected", "action_dim": "selected"}
    return PreparedDataset(root, profile, horizon=2, image_size=[12, 8],
                           channel_selection=selection, **kwargs)


@pytest.mark.parametrize("states,actions", [
    ("all", "all"), (["s2"], "all"), ("all", ["a1"]),
    (["s1"], ["a2"]), (["s2", "s1"], ["a2", "a1"]),
])
def test_selection_controls_vectors_statistics_and_model_dimensions(prepared, states, actions):
    all_data = selected_data(prepared)
    source_before = deepcopy(prepared[1])
    data = selected_data(prepared, {"state_names": states, "action_names": actions})
    assert prepared[1] == source_before
    sample = data[1]
    for kind, feature in (("state", "observation.state"), ("action", "action")):
        indices = data.channels["indices"][feature]
        np.testing.assert_array_equal(sample[kind], all_data[1][kind][..., indices])
        for stat in ("mean", "std", "min", "max"):
            np.testing.assert_allclose(data.statistics[kind][stat], np.array(all_data.statistics[kind][stat])[indices])
        assert data.model_dimensions[kind + "_dim"] == len(indices)
    assert sample["action_mask"].all()
    assert sample["image"][0].tobytes() == all_data[1]["image"][0].tobytes()
    config = {"framework": {"name": "ABot_M0", "action_model": {"state_dim": 14, "action_dim": 14}}}
    before = deepcopy(config)
    resolved = data.configure_model(config)
    assert config == before
    assert resolved["framework"]["action_model"]["state_dim"] == sample["state"].shape[-1]
    assert resolved["framework"]["action_model"]["action_dim"] == sample["action"].shape[-1]


def test_explicit_padding_masks_only_unused_model_columns(prepared):
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": ["a1"]},
                         padding={"state_dim": 3, "action_dim": 4})
    sample = data[1]
    np.testing.assert_array_equal(sample["state"][..., 1:], 0)
    np.testing.assert_array_equal(sample["action"][..., 1:], 0)
    np.testing.assert_array_equal(sample["action_mask"], [True, False, False, False])
    assert len(data.statistics["action"]["mean"]) == 1


@pytest.mark.parametrize("selection", [
    {}, {"state_names": "all"}, {"state_names": [], "action_names": "all"},
    {"state_names": ["s1", "s1"], "action_names": "all"},
    {"state_names": ["bad"], "action_names": "all"},
    {"state_names": ["s1"], "action_names": ["bad"]},
    {"state_names": True, "action_names": "all"},
    {"state_names": [1], "action_names": "all"},
    {"state_names": [" s1"], "action_names": "all"},
    {"state_names": ["s1"], "action_names": "all", "extra": 1},
])
def test_invalid_selection_is_rejected(prepared, selection):
    with pytest.raises(ValueError):
        selected_data(prepared, selection)


def test_yaml_uses_same_fields_as_lerobot_and_preserves_explicit_order(prepared, tmp_path):
    path = tmp_path / "channels.yaml"
    path.write_text("state_names: [s2, s1]\naction_names: all\n")
    data = selected_data(prepared, read_selection(path))
    assert data.selected["state_names"] == ["s2", "s1"]
    assert data.selected["action_names"] == ["a1", "a2"]
    path.write_text("state_names: all\n")
    with pytest.raises(ValueError):
        read_selection(path)
    path.write_text("state_names: [s1]\nstate_names: all\naction_names: all\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_selection(path)


def test_nan_in_excluded_state_does_not_leak_into_selected_input(prepared):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = prepared[0] / "data/0/episode_0.parquet"
    table = pq.read_table(path).to_pydict()
    table["observation.state"][0][1] = float("nan")
    pq.write_table(pa.table(table), path)
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": "all"})
    assert np.isfinite(data[0]["state"]).all()
    with pytest.raises(ValueError, match="selected state"):
        selected_data(prepared)


def test_saved_stats_reject_same_size_different_channel_selection(prepared):
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": ["a1"]})
    with pytest.raises(ValueError, match="Saved statistics"):
        selected_data(prepared, {"state_names": ["s2"], "action_names": ["a2"]}, statistics=data.statistics)
    with pytest.raises(ValueError, match="Saved statistics"):
        changed = deepcopy(prepared)
        changed[1]["normalization"] = "mean_std"
        selected_data(changed, {"state_names": ["s1"], "action_names": ["a1"]}, statistics=data.statistics)


def metadata_for(data):
    config = data.configure_model({"framework": {"name": "ABot_M0", "action_model": {}}})
    report = {"reinitialized_heads": ["action_model.state_encoder.", "action_model.action_decoder."],
              "vocabulary": {"mode": "restore_checkpoint"}}
    return data.checkpoint_metadata({"dataset": {"repo_id": "test/data", "revision": "a" * 40}}, config, report)


class Processor:
    def save_pretrained(self, path):
        path.mkdir()
        (path / "processor_config.json").write_text('{"fixture":true}')


def test_selected_training_save_and_fresh_process_reload(prepared, tmp_path):
    torch = pytest.importorskip("torch")
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": ["a1"]})
    model = torch.nn.Linear(data.model_dimensions["state_dim"], data.model_dimensions["action_dim"])
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    sample = data[1]
    x = torch.from_numpy(sample["state"])
    target = torch.from_numpy(sample["action"][:1])
    (model(x) - target).square().mean().backward()
    optimizer.step()
    metadata = metadata_for(data)
    assert metadata["train_config.json"]["initialization"] == metadata["initialization.json"]
    assert metadata["cyclo_io_mapping.json"]["state_names"] == ["s1"]
    assert metadata["train_config.json"]["training_adapter"]["resolved"]["source"]["state_names"] == ["s1", "s2"]
    saved = save_checkpoint(tmp_path / "output", step=1, model=model, processor=Processor(), metadata=metadata)
    expected = model(x).item()
    code = """
import sys, torch
from pathlib import Path
from abot_training.dataset import PreparedDataset
from abot_training.profile import read_json
data = PreparedDataset.from_checkpoint(sys.argv[1],sys.argv[2])
assert data.selected['state_names'] == ['s1']
assert data.selected['action_names'] == ['a1']
assert data.model_dimensions == {'state_dim':1,'action_dim':1}
root = Path(sys.argv[2])
receipt = read_json(root/'checkpoint.json')
model = torch.nn.Linear(data.model_dimensions['state_dim'],data.model_dimensions['action_dim'])
model.load_state_dict(torch.load(root/receipt['weights'],weights_only=True,mmap=True),strict=True)
assert abs(model(torch.from_numpy(data[1]['state'])).item()-float(sys.argv[3])) < 1e-6
print('selected-reload-ok')
"""
    result = subprocess.run([sys.executable, "-c", code, str(prepared[0]), str(saved), str(expected)],
                            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
                            text=True, capture_output=True, check=True)
    assert "selected-reload-ok" in result.stdout


@pytest.mark.parametrize("field", ["train_config.json", "cyclo_io_mapping.json", "model_config.json", "initialization.json"])
def test_reload_rejects_inconsistent_metadata_even_with_valid_checksums(prepared, tmp_path, field):
    torch = pytest.importorskip("torch")
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": ["a1"]})
    metadata = metadata_for(data)
    if field == "train_config.json":
        metadata[field]["training_adapter"]["resolved"]["selected"]["state_names"] = ["s2"]
    elif field == "cyclo_io_mapping.json":
        metadata[field]["state_names"] = ["s2"]
    elif field == "model_config.json":
        metadata[field]["framework"]["action_model"]["state_dim"] = 22
    else:
        metadata[field]["reinitialized_heads"] = []
    saved = save_checkpoint(tmp_path / "output", step=1, model=torch.nn.Linear(1,1),
                            processor=Processor(), metadata=metadata)
    with pytest.raises(ValueError):
        PreparedDataset.from_checkpoint(prepared[0],saved)


def test_metadata_rejects_stale_run_selection_and_wrong_model_dimensions(prepared):
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": ["a1"]})
    config = data.configure_model({"framework": {"name": "ABot_M0", "action_model": {}}})
    with pytest.raises(ValueError, match="Run channel selection"):
        data.checkpoint_metadata({"channel_selection": {"state_names": "all", "action_names": "all"}}, config, {})
    config["framework"]["action_model"]["state_dim"] = 22
    with pytest.raises(ValueError, match="Model config"):
        data.checkpoint_metadata({}, config, {})


def test_exported_metadata_does_not_alias_other_files_or_live_dataset(prepared):
    data = selected_data(prepared, {"state_names": ["s1"], "action_names": ["a1"]})
    metadata = metadata_for(data)
    metadata["cyclo_io_mapping.json"]["state_names"][0] = "s2"
    metadata["dataset_statistics.json"]["state"]["mean"][0] = -999
    assert metadata["train_config.json"]["training_adapter"]["resolved"]["selected"]["state_names"] == ["s1"]
    assert data.selected["state_names"] == ["s1"]
    assert data.statistics["state"]["mean"][0] != -999
