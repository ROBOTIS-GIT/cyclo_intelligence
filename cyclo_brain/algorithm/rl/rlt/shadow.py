"""Minimal, non-deploying GR00T RLT checkpoint bundle.

The bundle consumes already-extracted frozen GR00T tokens, proprioception and
the normalized GR00T reference chunk.  It never imports or mutates the GR00T
submodule and does not replace the live inference action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import torch
from torch import Tensor, nn

from cyclo_brain.model.mlp import RLTGaussianChunkActor

from .rl_token import FrozenRLTokenEncoder, load_frozen_rl_token_encoder


_ACTOR_ARTIFACT_FORMAT = "cyclo_brain.rlt.stage2_actor/v2"
_MAX_ACTOR_ARTIFACT_BYTES = 1024**3


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"RLT actor artifact {name} is invalid")
    return value


def _secure_load(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    lexical = Path(os.path.abspath(Path(path).expanduser()))
    metadata = lexical.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("RLT actor artifact must be a non-symlink regular file")
    if not 0 < metadata.st_size <= _MAX_ACTOR_ARTIFACT_BYTES:
        raise ValueError("RLT actor artifact file size is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("RLT actor artifact changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("RLT actor artifact root must be a mapping")
    return payload


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"RLT Stage-2 {name} must be a positive integer")


@dataclass(frozen=True)
class RLTStage2InferenceSpec:
    reference_contract_fingerprint: str
    rl_token_artifact_fingerprint: str
    rl_token_dim: int
    proprio_dim: int
    reference_horizon: int
    chunk_length: int
    action_dim: int
    action_hz: float
    action_normalization_id: str
    action_codec_id: str
    model_domain: str
    schema_version: int

    def __post_init__(self) -> None:
        _digest(self.reference_contract_fingerprint, "reference fingerprint")
        _digest(self.rl_token_artifact_fingerprint, "RL-token fingerprint")
        for value, name in (
            (self.rl_token_dim, "rl_token_dim"),
            (self.proprio_dim, "proprio_dim"),
            (self.reference_horizon, "reference_horizon"),
            (self.chunk_length, "chunk_length"),
            (self.action_dim, "action_dim"),
        ):
            _positive_integer(value, name)
        if self.chunk_length > self.reference_horizon:
            raise ValueError("RLT chunk length exceeds the GR00T reference horizon")
        if (
            isinstance(self.action_hz, bool)
            or not isinstance(self.action_hz, (int, float))
            or not math.isfinite(float(self.action_hz))
            or float(self.action_hz) <= 0.0
        ):
            raise ValueError("RLT action_hz must be finite and positive")
        if self.model_domain != "normalized":
            raise ValueError("RLT Action MLP must operate in the normalized domain")
        if self.schema_version != 1:
            raise ValueError("Unsupported RLT Stage-2 spec schema")


@dataclass(frozen=True)
class RLTShadowOutput:
    """Candidate output kept separate from the live GR00T action."""

    z_rl: Tensor
    action_mean: Tensor
    reference_prefix: Tensor


class GR00TRLTShadowPolicy(nn.Module):
    """Frozen RL-token encoder plus deterministic Stage-2 Action MLP."""

    def __init__(
        self,
        encoder: FrozenRLTokenEncoder,
        actor: RLTGaussianChunkActor,
        spec: RLTStage2InferenceSpec,
        *,
        actor_qualification: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.spec = spec
        self.actor_qualification = actor_qualification
        self.eval()

    def train(self, mode: bool = True) -> "GR00TRLTShadowPolicy":
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.inference_mode()
    def forward(
        self,
        tokens: Tensor,
        token_valid: Tensor,
        image_token: Tensor,
        proprio: Tensor,
        reference_actions: Tensor,
        *,
        reference_offset_steps: int = 0,
    ) -> RLTShadowOutput:
        if not isinstance(reference_actions, Tensor) or tuple(reference_actions.shape) != (
            tokens.shape[0],
            self.spec.reference_horizon,
            self.spec.action_dim,
        ):
            raise ValueError(
                "RLT GR00T reference_actions must have shape "
                f"(B, {self.spec.reference_horizon}, {self.spec.action_dim})"
            )
        if not isinstance(proprio, Tensor) or tuple(proprio.shape) != (
            tokens.shape[0],
            self.spec.proprio_dim,
        ):
            raise ValueError(
                f"RLT proprio must have shape (B, {self.spec.proprio_dim})"
            )
        if reference_actions.requires_grad or proprio.requires_grad:
            raise ValueError("RLT shadow inputs must be detached from frozen GR00T")
        if not reference_actions.is_floating_point() or not proprio.is_floating_point():
            raise TypeError("RLT shadow proprio and actions must be floating tensors")
        if not bool(torch.isfinite(reference_actions).all()) or not bool(
            torch.isfinite(proprio).all()
        ):
            raise ValueError("RLT shadow inputs must be finite")
        if (
            isinstance(reference_offset_steps, bool)
            or not isinstance(reference_offset_steps, int)
            or reference_offset_steps < 0
            or reference_offset_steps + self.spec.chunk_length
            > self.spec.reference_horizon
        ):
            raise ValueError(
                "RLT reference_offset_steps must select one complete "
                f"{self.spec.chunk_length}-step chunk inside the "
                f"{self.spec.reference_horizon}-step reference"
            )

        z_rl = self.encoder(tokens, token_valid, image_token)
        actor_parameter = next(self.actor.parameters())
        z_rl = z_rl.to(device=actor_parameter.device, dtype=actor_parameter.dtype)
        proprio_actor = proprio.detach().to(
            device=actor_parameter.device,
            dtype=actor_parameter.dtype,
        )
        reference_prefix = reference_actions[
            :,
            reference_offset_steps : reference_offset_steps
            + self.spec.chunk_length,
        ].detach().to(device=actor_parameter.device, dtype=actor_parameter.dtype)
        action_mean = self.actor(z_rl, proprio_actor, reference_prefix)
        expected = (tokens.shape[0], self.spec.chunk_length, self.spec.action_dim)
        if tuple(action_mean.shape) != expected:
            raise RuntimeError(
                "RLT Action MLP returned "
                f"{tuple(action_mean.shape)}, expected {expected}"
            )
        return RLTShadowOutput(
            z_rl=z_rl,
            action_mean=action_mean,
            reference_prefix=reference_prefix,
        )


def load_groot_rlt_shadow_policy(
    rl_token_encoder_artifact: str | os.PathLike[str],
    actor_artifact: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    expected_chunk_length: int = 10,
    expected_action_dim: int = 19,
) -> GR00TRLTShadowPolicy:
    """Load the existing showroom bundle without enabling live execution."""

    encoder = load_frozen_rl_token_encoder(
        rl_token_encoder_artifact,
        device=device,
        dtype=dtype,
    )
    payload = _secure_load(actor_artifact)
    required = {
        "format",
        "spec",
        "spec_fingerprint",
        "config",
        "actor_hidden_dims",
        "completed_critic_updates",
        "completed_actor_updates",
        "replay_artifact",
        "source_manifest_fingerprint",
        "actor",
        "diagnostic_count",
        "last_diagnostic",
        "qualification",
    }
    if set(payload) != required or payload.get("format") != _ACTOR_ARTIFACT_FORMAT:
        raise ValueError("RLT actor artifact fields are invalid")
    raw_spec = payload.get("spec")
    if not isinstance(raw_spec, Mapping) or set(raw_spec) != {
        field.name for field in fields(RLTStage2InferenceSpec)
    }:
        raise ValueError("RLT actor spec is invalid")
    spec = RLTStage2InferenceSpec(**dict(raw_spec))
    spec_fingerprint = _digest(payload.get("spec_fingerprint"), "spec fingerprint")
    if _canonical_fingerprint(raw_spec) != spec_fingerprint:
        raise ValueError("RLT actor spec fingerprint disagrees")
    if spec.rl_token_artifact_fingerprint != encoder.artifact_fingerprint:
        raise ValueError("RLT actor and RL-token encoder artifacts disagree")
    if spec.rl_token_dim != encoder.config.embedding_dim:
        raise ValueError("RLT actor and RL-token dimensions disagree")
    if spec.chunk_length != expected_chunk_length or spec.action_dim != expected_action_dim:
        raise ValueError(
            "RLT actor does not satisfy the required "
            f"{expected_chunk_length}x{expected_action_dim} action contract"
        )

    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("RLT actor config is invalid")
    fixed_standard_deviation = config.get("fixed_standard_deviation")
    hidden_dims = payload.get("actor_hidden_dims")
    if not isinstance(hidden_dims, Sequence) or isinstance(hidden_dims, (str, bytes)):
        raise ValueError("RLT actor hidden dimensions are invalid")
    with torch.random.fork_rng(devices=[], enabled=True):
        actor = RLTGaussianChunkActor(
            spec.rl_token_dim,
            spec.proprio_dim,
            spec.chunk_length,
            spec.action_dim,
            fixed_standard_deviation=fixed_standard_deviation,
            hidden_dims=tuple(hidden_dims),
        )
    actor_state = payload.get("actor")
    if not isinstance(actor_state, Mapping):
        raise ValueError("RLT actor state is invalid")
    try:
        actor.load_state_dict(actor_state, strict=True)
    except RuntimeError as error:
        raise ValueError("RLT actor artifact tensors disagree") from error
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in actor.state_dict().values()
    ):
        raise ValueError("RLT actor artifact contains non-finite tensors")
    actor.to(device=torch.device(device), dtype=dtype)
    actor.eval()
    qualification = payload.get("qualification")
    if not isinstance(qualification, str) or not qualification:
        raise ValueError("RLT actor qualification is invalid")
    return GR00TRLTShadowPolicy(
        encoder,
        actor,
        spec,
        actor_qualification=qualification,
    )


__all__ = [
    "GR00TRLTShadowPolicy",
    "RLTShadowOutput",
    "RLTStage2InferenceSpec",
    "load_groot_rlt_shadow_policy",
]
