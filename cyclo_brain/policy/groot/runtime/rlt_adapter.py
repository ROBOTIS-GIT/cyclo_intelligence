#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GR00T N1.7 to PI-RLT inference adapter.

The adapter keeps the upstream GR00T package untouched.  It extracts the raw
Qwen3 token sequence before the action head mutates it, samples the frozen
GR00T reference chunk once, and decodes the RLT Action MLP result with the
same checkpoint processor used by normal GR00T inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


EXPECTED_CHUNK_LENGTH = 10
EXPECTED_ACTION_DIM = 19
EXPECTED_REFERENCE_HORIZON = 16
SIMULATION_ONLY_QUALIFICATION = "training_only_not_deployment_validated"
DEPLOYMENT_QUALIFICATION = "deployment_qualified"


def is_deployment_qualified(qualification: object) -> bool:
    """Fail closed unless the actor carries the exact deployment marker."""

    return str(qualification or "").strip().lower() == DEPLOYMENT_QUALIFICATION


@dataclass(frozen=True)
class RLTBundlePaths:
    root: Path
    encoder: Path
    actor: Path


def _regular_files(root: Path, name: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(name)
        if path.is_file() and not path.is_symlink()
    )


def _manifest_fingerprint(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("artifact", {}).get("artifact_fingerprint")
    return value if isinstance(value, str) else None


def _actor_encoder_fingerprint(actor_root: Path) -> str | None:
    manifests = _regular_files(actor_root, "rlt_stage2.pt.run.json")
    if len(manifests) != 1:
        return None
    try:
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = (
        payload.get("base_contract", {})
        .get("learner", {})
        .get("contract", {})
        .get("spec", {})
        .get("rl_token_artifact_fingerprint")
    )
    return value if isinstance(value, str) else None


def resolve_rlt_bundle(bundle_path: str | os.PathLike[str]) -> RLTBundlePaths:
    """Resolve one Stage-2 actor and its exact Stage-1 encoder.

    A normal UI selection points at a Stage-2 output directory.  Stage-1 and
    Stage-2 are sibling training runs, so the resolver uses their persisted
    artifact fingerprint rather than guessing from directory names.
    """

    text = os.fspath(bundle_path).strip()
    if not text:
        raise ValueError("RLT bundle path is empty")
    root = Path(os.path.abspath(Path(text).expanduser()))
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"RLT bundle must be a non-symlink directory: {root}")

    actors = _regular_files(root, "rlt_actor.pt")
    if len(actors) != 1:
        raise ValueError(
            f"RLT bundle must contain exactly one rlt_actor.pt; found {len(actors)}"
        )
    actor = actors[0]
    actor_root = actor.parent.parent if actor.parent.name == "artifacts" else root

    local_encoders = _regular_files(root, "rl_token_encoder.pt")
    search_root = root.parent
    encoders = local_encoders or _regular_files(search_root, "rl_token_encoder.pt")
    if not encoders:
        raise ValueError(
            "RLT bundle has no compatible rl_token_encoder.pt in the bundle "
            "or its sibling training runs"
        )

    expected = _actor_encoder_fingerprint(actor_root)
    if expected:
        matched: list[Path] = []
        for encoder in encoders:
            encoder_root = (
                encoder.parent.parent
                if encoder.parent.name == "artifacts"
                else encoder.parent
            )
            manifests = _regular_files(encoder_root, "rlt_stage1.pt.run.json")
            if len(manifests) == 1 and _manifest_fingerprint(manifests[0]) == expected:
                matched.append(encoder)
        encoders = matched

    unique = sorted(set(encoders))
    if len(unique) != 1:
        raise ValueError(
            "RLT bundle encoder resolution is ambiguous; expected exactly one "
            f"compatible encoder, found {len(unique)}"
        )
    return RLTBundlePaths(root=root, encoder=unique[0], actor=actor)


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, Tensor) and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_rec_to_dtype(item, dtype) for item in value)
    return value


def _checkpoint_weight_fingerprint(checkpoint: Path) -> str:
    """Reproduce the Stage-1 path-independent GR00T weight identity."""

    suffixes = {".safetensors", ".pth", ".pt", ".bin"}
    ignored = (
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
        "trainer_state",
        "training_args",
        "training_state",
        "replay_buffer",
    )
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"GR00T checkpoint contains a symlink: {path}")
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() not in suffixes or name.startswith(ignored):
            continue
        digest = sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise ValueError("GR00T checkpoint contains no model weight files")
    core = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class GR00TRLTInferenceAdapter:
    """Preloaded RLT policy that shares one frozen GR00T instance."""

    def __init__(self, policy: Any, shadow_policy: Any, bundle: RLTBundlePaths):
        self.policy = policy
        self.shadow_policy = shadow_policy
        self.bundle = bundle
        self.spec = shadow_policy.spec
        if self.spec.reference_horizon != EXPECTED_REFERENCE_HORIZON:
            raise ValueError(
                "RLT reference horizon must be "
                f"{EXPECTED_REFERENCE_HORIZON}, got {self.spec.reference_horizon}"
            )
        if (
            self.spec.chunk_length != EXPECTED_CHUNK_LENGTH
            or self.spec.action_dim != EXPECTED_ACTION_DIM
            or self.spec.proprio_dim != EXPECTED_ACTION_DIM
        ):
            raise ValueError("RLT runtime requires the showroom 10x19 contract")

    @classmethod
    def load(
        cls,
        policy: Any,
        bundle_path: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
    ) -> "GR00TRLTInferenceAdapter":
        from cyclo_brain.algorithm.rl.rlt import load_groot_rlt_shadow_policy

        bundle = resolve_rlt_bundle(bundle_path)
        shadow = load_groot_rlt_shadow_policy(
            bundle.encoder,
            bundle.actor,
            device=policy.model.device,
            dtype=torch.float32,
            expected_chunk_length=EXPECTED_CHUNK_LENGTH,
            expected_action_dim=EXPECTED_ACTION_DIM,
        )

        expected_weight = shadow.encoder.representation_contract.get(
            "policy_weight_fingerprint"
        )
        if expected_weight:
            actual_weight = _checkpoint_weight_fingerprint(
                Path(os.path.abspath(Path(model_path).expanduser()))
            )
            if actual_weight != expected_weight:
                raise ValueError(
                    "RLT bundle was trained from a different GR00T checkpoint"
                )
        return cls(policy, shadow, bundle)

    @property
    def qualification(self) -> str:
        return str(self.shadow_policy.actor_qualification)

    @property
    def deployment_qualified(self) -> bool:
        return is_deployment_qualified(self.qualification)

    def _prepare(self, observation: Mapping[str, object]):
        if getattr(self.policy, "strict", False):
            self.policy.check_observation(observation)
        unbatched = self.policy._unbatch_observation(observation)
        processed = []
        raw_states = []
        for item in unbatched:
            vla_step = self.policy._to_vla_step_data(item)
            raw_states.append(vla_step.states)
            processed.append(
                self.policy.processor(
                    [{"type": "episode_step", "content": vla_step}]
                )
            )
        collated = self.policy.collate_fn(processed)
        collated = _rec_to_dtype(
            collated,
            dtype=getattr(self.policy.model, "dtype", torch.bfloat16),
        )
        if not isinstance(collated, Mapping) or "inputs" not in collated:
            raise RuntimeError("GR00T RLT expected collator output with inputs")
        batched_states = {
            key: np.stack([state[key] for state in raw_states], axis=0)
            for key in self.policy.modality_configs["state"].modality_keys
        }
        return collated, batched_states

    @torch.inference_mode()
    def get_action(self, observation: Mapping[str, object]) -> dict[str, np.ndarray]:
        collated, batched_states = self._prepare(observation)
        model_inputs = collated["inputs"]
        backbone_inputs, action_inputs = self.policy.model.prepare_input(model_inputs)
        backbone_output = self.policy.model.backbone(backbone_inputs)

        tokens = backbone_output["backbone_features"].detach().clone()
        token_valid = backbone_output["backbone_attention_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        image_token = backbone_output["image_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        normalized_state = action_inputs["state"]
        proprio = normalized_state[:, -1, : self.spec.proprio_dim].float().detach()

        model_pred = self.policy.model.action_head.get_action(
            backbone_output,
            action_inputs,
            collated.get("options"),
        )
        reference = model_pred["action_pred"][
            :, : self.spec.reference_horizon, : self.spec.action_dim
        ].float().detach()
        expected_reference = (
            tokens.shape[0],
            self.spec.reference_horizon,
            self.spec.action_dim,
        )
        if tuple(reference.shape) != expected_reference:
            raise RuntimeError(
                f"GR00T RLT reference shape {tuple(reference.shape)} does not "
                f"match {expected_reference}"
            )

        candidate = self.shadow_policy(
            tokens,
            token_valid,
            image_token,
            proprio,
            reference,
        ).action_mean
        if not bool(torch.isfinite(candidate).all()):
            raise FloatingPointError("RLT Action MLP returned non-finite actions")

        decoded = self.policy.processor.decode_action(
            candidate.float().cpu().numpy(),
            self.policy.embodiment_tag,
            batched_states,
        )
        physical = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in decoded.items()
        }
        expected_keys = list(self.policy.modality_configs["action"].modality_keys)
        if list(physical) != expected_keys:
            raise RuntimeError(
                f"RLT decoder returned {list(physical)}, expected {expected_keys}"
            )
        if any(not np.isfinite(value).all() for value in physical.values()):
            raise FloatingPointError("Decoded RLT action contains non-finite values")
        return physical


__all__ = [
    "DEPLOYMENT_QUALIFICATION",
    "EXPECTED_ACTION_DIM",
    "EXPECTED_CHUNK_LENGTH",
    "GR00TRLTInferenceAdapter",
    "RLTBundlePaths",
    "SIMULATION_ONLY_QUALIFICATION",
    "is_deployment_qualified",
    "resolve_rlt_bundle",
]
