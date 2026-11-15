# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0.

"""Offline shadow metrics for a trained RLT actor and its immutable replay."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


SHADOW_FORMAT = "cyclo.groot.rlt.shadow_evaluation/v1"


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@torch.inference_mode()
def evaluate_replay_shadow(actor, replay, *, batch_size: int = 256) -> dict[str, Any]:
    """Compare deterministic actor output with GR00T reference on all rows."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("RLT shadow batch_size must be a positive integer")
    count = len(replay)
    if count < 1:
        raise ValueError("RLT shadow evaluation requires a non-empty replay")
    parameter = next(actor.parameters())
    device, dtype = parameter.device, parameter.dtype
    corrections = []
    action_deltas = []
    reference_deltas = []
    finite = True
    actor.eval()
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        indices = torch.arange(start, stop, dtype=torch.long)
        z_rl = replay.tensors["z_rl"].index_select(0, indices).to(device, dtype)
        proprio = replay.tensors["proprio"].index_select(0, indices).to(device, dtype)
        reference = replay.tensors["reference_actions"].index_select(0, indices).to(
            device, dtype
        )
        action = actor(z_rl, proprio, reference)
        finite = finite and bool(torch.isfinite(action).all())
        corrections.append((action - reference).abs().reshape(-1).float().cpu())
        if action.shape[1] > 1:
            action_deltas.append(
                (action[:, 1:] - action[:, :-1]).abs().reshape(-1).float().cpu()
            )
            reference_deltas.append(
                (reference[:, 1:] - reference[:, :-1])
                .abs()
                .reshape(-1)
                .float()
                .cpu()
            )
    correction = torch.cat(corrections)
    action_delta = torch.cat(action_deltas) if action_deltas else torch.zeros(1)
    reference_delta = (
        torch.cat(reference_deltas) if reference_deltas else torch.zeros(1)
    )

    def metric(value: torch.Tensor, operation: str) -> float:
        if operation == "mean":
            result = value.mean()
        elif operation == "rms":
            result = value.square().mean().sqrt()
        elif operation == "max":
            result = value.max()
        else:
            result = torch.quantile(value, float(operation))
        number = float(result.item())
        if not math.isfinite(number):
            raise FloatingPointError("RLT shadow metric is non-finite")
        return number

    unsigned = {
        "format": SHADOW_FORMAT,
        "structural_pass": bool(finite),
        "behavioral_gate": "operator_review_required",
        "sample_count": count,
        "action_element_count": int(correction.numel()),
        "correction_abs_mean": metric(correction, "mean"),
        "correction_rms": metric(correction, "rms"),
        "correction_abs_p95": metric(correction, "0.95"),
        "correction_abs_p99": metric(correction, "0.99"),
        "correction_abs_max": metric(correction, "max"),
        "action_temporal_delta_abs_p99": metric(action_delta, "0.99"),
        "reference_temporal_delta_abs_p99": metric(reference_delta, "0.99"),
    }
    return {**unsigned, "evaluation_fingerprint": _fingerprint(unsigned)}


def write_shadow_evaluation(
    bundle: str | Path,
    evaluation: Mapping[str, Any],
) -> Path:
    """Write evidence bound to the immutable bundle manifest fingerprint."""

    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bundle_fingerprint = manifest.get("manifest_fingerprint")
    if not isinstance(bundle_fingerprint, str) or len(bundle_fingerprint) != 64:
        raise ValueError("RLT bundle manifest fingerprint is invalid")
    unsigned = {
        "format": "cyclo.groot.rlt.shadow_qualification/v1",
        "bundle_manifest_fingerprint": bundle_fingerprint,
        "evaluation": dict(evaluation),
    }
    payload = {**unsigned, "qualification_fingerprint": _fingerprint(unsigned)}
    output = root / "evaluation" / "shadow_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


__all__ = ["SHADOW_FORMAT", "evaluate_replay_shadow", "write_shadow_evaluation"]
