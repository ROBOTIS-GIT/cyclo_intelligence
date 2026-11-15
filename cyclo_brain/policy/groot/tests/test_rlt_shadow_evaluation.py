import json
import torch
from torch import nn

from cyclo_brain.policy.groot.runtime.rlt_shadow_evaluation import (
    evaluate_replay_shadow,
    write_shadow_evaluation,
)


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, z_rl, proprio, reference):
        del z_rl, proprio
        return reference + self.anchor + 0.25


class _Replay:
    def __init__(self):
        self.tensors = {
            "z_rl": torch.zeros(3, 4),
            "proprio": torch.zeros(3, 2),
            "reference_actions": torch.zeros(3, 10, 2),
        }

    def __len__(self):
        return 3


def test_shadow_evaluation_reports_actor_reference_correction():
    result = evaluate_replay_shadow(_Actor(), _Replay(), batch_size=2)

    assert result["structural_pass"] is True
    assert result["sample_count"] == 3
    assert result["correction_abs_mean"] == 0.25
    assert result["correction_abs_p99"] == 0.25
    assert result["behavioral_gate"] == "operator_review_required"


def test_shadow_evidence_is_bound_to_bundle_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"manifest_fingerprint": "a" * 64}),
        encoding="utf-8",
    )
    evaluation = {
        "format": "cyclo.groot.rlt.shadow_evaluation/v1",
        "structural_pass": True,
        "evaluation_fingerprint": "b" * 64,
    }

    path = write_shadow_evaluation(bundle, evaluation)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["bundle_manifest_fingerprint"] == "a" * 64
    assert payload["evaluation"] == evaluation
