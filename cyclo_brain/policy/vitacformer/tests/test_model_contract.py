#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "vitacformer_engine"
    / "model.py"
)
spec = importlib.util.spec_from_file_location("vitacformer_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def _train_config():
    return {
        "architecture": "ViTacFormer",
        "source_repo_commit": model._EXPECTED_SOURCE_COMMIT,
        "data_contract": {
            "action_chunk": 100,
            "action_dim": 54,
            "state_dim": 54,
            "state_history": 6,
            "state_stride": 3,
            "tactile_history": 18,
            "tactile_raw_dim": 90,
            "tactile_rep_dim": 180,
            "tactile_taxels_per_hand": 45,
            "image_height": 188,
            "image_width": 336,
            "fps": 30,
            "image_key": "observation.images.rgb.cam_left_head",
            "tactile_order": "left[5,3,3],right[5,3,3]",
            "baseline": (
                "already corrected per episode from first 20 raw 100Hz samples"
            ),
        },
    }


def test_contract_accepts_only_the_audited_sh5_layout():
    model._validate_contract(_train_config())

    bad = _train_config()
    bad["data_contract"]["action_dim"] = 53
    with pytest.raises(ValueError, match="action_dim"):
        model._validate_contract(bad)

    bad = _train_config()
    bad["source_repo_commit"] = "unknown"
    with pytest.raises(ValueError, match="source commit"):
        model._validate_contract(bad)


def test_contract_accepts_original_export_without_descriptive_annotations():
    original_export = _train_config()
    del original_export["data_contract"]["tactile_order"]
    del original_export["data_contract"]["baseline"]

    model._validate_contract(original_export)

    contradictory = _train_config()
    contradictory["data_contract"]["tactile_order"] = "right,left"
    with pytest.raises(ValueError, match="tactile_order"):
        model._validate_contract(contradictory)


def test_layout_resolves_run_and_numeric_checkpoint(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "1000"
    checkpoint.mkdir(parents=True)
    (run / "train_config.json").write_text(json.dumps(_train_config()))
    weights = checkpoint / "model.pt"
    weights.touch()

    assert model._resolve_vitacformer_layout(run) == (run, None)
    assert model._resolve_vitacformer_layout(checkpoint) == (run, weights)
    assert model._resolve_vitacformer_layout(weights) == (run, weights)
    assert model._resolve_vitacformer_layout("") is None


def test_stats_require_exact_finite_positive_normalization_vectors():
    stats = {
        "state_mean": torch.zeros(54),
        "state_std": torch.ones(54),
        "action_mean": torch.zeros(54),
        "action_std": torch.ones(54),
        "tactile_history_mean": torch.zeros(180),
        "tactile_history_std": torch.ones(180),
        "tactile_future_mean": torch.zeros(180),
        "tactile_future_std": torch.ones(180),
    }
    model._validate_stats(stats)

    bad = dict(stats, state_mean=torch.zeros(53))
    with pytest.raises(ValueError, match="state_mean.*shape"):
        model._validate_stats(bad)

    bad = dict(stats, action_std=torch.zeros(54))
    with pytest.raises(ValueError, match="action_std.*strictly positive"):
        model._validate_stats(bad)


def test_default_checkpoint_prefers_best_model(tmp_path):
    root = tmp_path / "run"
    best = root / "checkpoints" / "best_model.pt"
    best.parent.mkdir(parents=True)
    best.touch()
    (root / "model.pt").touch()

    assert model._checkpoint_path(root) == best
    (root / "train_config.json").write_text(json.dumps(_train_config()))
    assert model._resolve_vitacformer_layout(root) == (root, None)


def test_bounded_decoder_keeps_all_actions_inside_sh5_limits():
    logits = torch.zeros(1, 100, 54)
    # Reproduce the online failure: right-hand joint 3 is global index 36.
    logits[:, 12, 36] = -0.060661
    normalized, bounded = model._bound_action(
        logits,
        torch.zeros(54),
        torch.ones(54),
    )

    lower = model._JOINT_LOWER + 1e-5
    upper = model._JOINT_UPPER - 1e-5
    assert torch.all(bounded >= lower)
    assert torch.all(bounded <= upper)
    assert bounded[0, 12, 36] >= lower[36]
    assert torch.allclose(normalized, bounded)


def test_warm_start_ramps_only_arms_over_sixteen_rows():
    action = torch.zeros(1, 100, 54)
    action[:, :, :14] = 0.2
    action[:, :, 14:] = 0.4
    current = torch.zeros(1, 54)

    ramped = model._apply_warm_start_ramp(action, current)
    safe_current = torch.maximum(
        model._JOINT_LOWER + 1e-5,
        torch.minimum(model._JOINT_UPPER - 1e-5, current),
    )

    assert torch.allclose(ramped[:, 0, :14], safe_current[:, :14])
    assert torch.allclose(ramped[:, 15, :14], action[:, 15, :14])
    assert torch.allclose(ramped[:, :, 14:], action[:, :, 14:])
    assert torch.allclose(ramped[:, 16:], action[:, 16:])


def test_right_tactile_future_uses_exact_persistence_fallback():
    history = torch.zeros(1, 18, 180)
    history[:, -1, :90] = torch.arange(90, dtype=torch.float32)
    persistence, residual_scale = model._build_tactile_persistence(
        history,
        torch.zeros(180),
        torch.ones(180),
    )

    assert persistence.shape == (1, 18, 180)
    assert torch.allclose(persistence[:, :, :90], history[:, -1:, :90])
    assert torch.count_nonzero(persistence[:, :, 90:]) == 0
    assert torch.count_nonzero(residual_scale[45:90]) == 0
    assert torch.count_nonzero(residual_scale[135:180]) == 0
    assert torch.all(residual_scale[:45] == 1)
    assert torch.all(residual_scale[90:135] == 1)
