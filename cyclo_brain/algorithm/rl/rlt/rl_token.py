"""Frozen PI RLT Stage-1 RL-token encoder artifact support.

This is intentionally the inference-only half of the original Stage-1
autoencoder: the reconstruction decoder is not constructed or loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Literal

import torch
from torch import Tensor, nn


TokenSelection = Literal["all", "image"]
_ARTIFACT_FORMAT = "cyclo_brain.rlt.frozen_encoder/v1"
_MAX_ARTIFACT_BYTES = 8 * 1024**3
_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"RLT {name} must be a positive integer")


@dataclass(frozen=True)
class RLTokenConfig:
    """Architecture persisted by the existing Stage-1 artifact."""

    embedding_dim: int
    max_tokens: int
    num_heads: int
    encoder_layers: int
    decoder_layers: int
    feedforward_dim: int
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    token_selection: TokenSelection = "image"
    loss_reduction: str = "paper_per_sample_sum"

    def __post_init__(self) -> None:
        for value, name in (
            (self.embedding_dim, "embedding_dim"),
            (self.max_tokens, "max_tokens"),
            (self.num_heads, "num_heads"),
            (self.encoder_layers, "encoder_layers"),
            (self.decoder_layers, "decoder_layers"),
            (self.feedforward_dim, "feedforward_dim"),
        ):
            _positive_integer(value, name)
        if self.embedding_dim % self.num_heads:
            raise ValueError("RLT embedding_dim must be divisible by num_heads")
        if self.embedding_dim > 8192 or self.max_tokens > 8192:
            raise ValueError("RLT encoder dimensions exceed the supported limit")
        if self.encoder_layers > 32 or self.feedforward_dim > 65536:
            raise ValueError("RLT encoder architecture exceeds the supported limit")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("RLT dropout must be finite in [0, 1)")
        if (
            isinstance(self.layer_norm_eps, bool)
            or not isinstance(self.layer_norm_eps, (int, float))
            or not math.isfinite(float(self.layer_norm_eps))
            or float(self.layer_norm_eps) <= 0.0
        ):
            raise ValueError("RLT layer_norm_eps must be finite and positive")
        if self.token_selection not in {"all", "image"}:
            raise ValueError("RLT token_selection must be 'all' or 'image'")


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
        raise ValueError(f"RLT encoder artifact {name} is invalid")
    return value


def _secure_load(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    candidate = Path(path).expanduser()
    lexical = Path(os.path.abspath(candidate))
    metadata = lexical.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("RLT encoder artifact must be a non-symlink regular file")
    if not 0 < metadata.st_size <= _MAX_ARTIFACT_BYTES:
        raise ValueError("RLT encoder artifact file size is invalid")
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
        raise RuntimeError("RLT encoder artifact changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("RLT encoder artifact root must be a mapping")
    return payload


def _artifact_fingerprint(
    config: RLTokenConfig,
    representation_contract_fingerprint: str,
    state: Mapping[str, Tensor],
) -> str:
    tensors = []
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
        )
    return _canonical_fingerprint(
        {
            "schema": _ARTIFACT_FORMAT,
            "model_config": asdict(config),
            "representation_contract": representation_contract_fingerprint,
            "tensors": tensors,
        }
    )


class FrozenRLTokenEncoder(nn.Module):
    """Decoder-free transformer that extracts one 2048D RL token."""

    def __init__(
        self,
        config: RLTokenConfig,
        *,
        representation_contract: Mapping[str, Any],
        representation_contract_fingerprint: str,
        artifact_fingerprint: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.representation_contract = dict(representation_contract)
        self.representation_contract_fingerprint = representation_contract_fingerprint
        self.artifact_fingerprint = artifact_fingerprint
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.embedding_dim, eps=float(config.layer_norm_eps)),
            enable_nested_tensor=False,
        )
        self.rl_embedding = nn.Parameter(
            torch.empty(1, 1, config.embedding_dim),
            requires_grad=False,
        )
        self.rl_position = nn.Parameter(
            torch.empty(1, 1, config.embedding_dim),
            requires_grad=False,
        )
        self.encoder_positions = nn.Parameter(
            torch.empty(1, config.max_tokens, config.embedding_dim),
            requires_grad=False,
        )
        super().train(False)
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenRLTokenEncoder":
        del mode
        super().train(False)
        return self

    def forward(
        self,
        tokens: Tensor,
        token_valid: Tensor,
        image_token: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(tokens, Tensor) or not tokens.is_floating_point():
            raise TypeError("RLT tokens must be a floating tensor")
        if tokens.requires_grad:
            raise ValueError("RLT tokens must be detached from the frozen GR00T model")
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.embedding_dim:
            raise ValueError(
                f"RLT tokens must have shape (B, M, {self.config.embedding_dim})"
            )
        mask_shape = tuple(tokens.shape[:2])
        if (
            not isinstance(token_valid, Tensor)
            or token_valid.dtype != torch.bool
            or tuple(token_valid.shape) != mask_shape
        ):
            raise ValueError("RLT token_valid must be boolean with shape (B, M)")
        selected = token_valid
        if self.config.token_selection == "image":
            if (
                not isinstance(image_token, Tensor)
                or image_token.dtype != torch.bool
                or tuple(image_token.shape) != mask_shape
            ):
                raise ValueError("RLT image_token must be boolean with shape (B, M)")
            selected = token_valid & image_token
        if not bool(selected.any(dim=1).all()):
            raise ValueError("RLT every sample must contain a selected valid token")

        counts = selected.sum(dim=1)
        token_count = int(counts.max().item())
        if token_count > self.config.max_tokens:
            raise ValueError("RLT selected token count exceeds configured max_tokens")
        parameter = self.rl_embedding
        source = tokens.detach().to(device=parameter.device, dtype=parameter.dtype)
        selected = selected.to(device=parameter.device)
        compact = source.new_zeros(
            (source.shape[0], token_count, self.config.embedding_dim)
        )
        rows, columns = selected.nonzero(as_tuple=True)
        packed_columns = selected.long().cumsum(dim=1)[rows, columns] - 1
        compact[rows, packed_columns] = source[rows, columns]
        compact_valid = (
            torch.arange(token_count, device=parameter.device).unsqueeze(0)
            < counts.to(device=parameter.device).unsqueeze(1)
        )

        positioned = compact + self.encoder_positions[:, :token_count]
        encoder_input = compact.new_zeros(
            (compact.shape[0], token_count + 1, self.config.embedding_dim)
        )
        encoder_input[:, :token_count] = positioned
        encoder_valid = torch.zeros(
            (compact.shape[0], token_count + 1),
            dtype=torch.bool,
            device=parameter.device,
        )
        encoder_valid[:, :token_count] = compact_valid
        rl_indices = compact_valid.sum(dim=1)
        batch_rows = torch.arange(compact.shape[0], device=parameter.device)
        encoder_input[batch_rows, rl_indices] = (
            self.rl_embedding[0, 0] + self.rl_position[0, 0]
        )
        encoder_valid[batch_rows, rl_indices] = True
        encoded = self.encoder(
            encoder_input,
            src_key_padding_mask=~encoder_valid,
        )
        z_rl = encoded[batch_rows, rl_indices]
        if tuple(z_rl.shape) != (tokens.shape[0], self.config.embedding_dim):
            raise RuntimeError("RLT encoder returned an invalid RL-token shape")
        if not bool(torch.isfinite(z_rl).all()):
            raise FloatingPointError("RLT encoder returned a non-finite RL token")
        return z_rl


def load_frozen_rl_token_encoder(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> FrozenRLTokenEncoder:
    """Strictly load and fingerprint-check an existing Stage-1 artifact."""

    payload = _secure_load(path)
    required = {
        "format",
        "config",
        "representation_contract",
        "representation_contract_fingerprint",
        "artifact_fingerprint",
        "encoder",
    }
    if set(payload) != required or payload.get("format") != _ARTIFACT_FORMAT:
        raise ValueError("RLT encoder artifact fields are invalid")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping) or set(raw_config) != {
        field.name for field in fields(RLTokenConfig)
    }:
        raise ValueError("RLT encoder artifact config is invalid")
    config = RLTokenConfig(**dict(raw_config))
    representation = payload.get("representation_contract")
    if not isinstance(representation, Mapping):
        raise ValueError("RLT encoder representation contract is invalid")
    contract_fingerprint = _digest(
        payload.get("representation_contract_fingerprint"),
        "representation contract fingerprint",
    )
    if _canonical_fingerprint(representation) != contract_fingerprint:
        raise ValueError("RLT encoder representation contract fingerprint disagrees")
    embeddings = representation.get("embeddings")
    if not isinstance(embeddings, Mapping) or embeddings.get("width") != config.embedding_dim:
        raise ValueError("RLT encoder representation width disagrees")

    raw_state = payload.get("encoder")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("RLT encoder state is invalid")
    state: dict[str, Tensor] = {}
    artifact_dtype: torch.dtype | None = None
    for name, value in raw_state.items():
        if (
            not isinstance(name, str)
            or "decoder" in name
            or not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.dtype not in _SUPPORTED_DTYPES
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"RLT encoder tensor {name!r} is invalid")
        if artifact_dtype is None:
            artifact_dtype = value.dtype
        elif artifact_dtype != value.dtype:
            raise ValueError("RLT encoder artifact tensor dtypes disagree")
        state[name] = value
    if artifact_dtype is None:
        raise ValueError("RLT encoder state is empty")
    expected_artifact_fingerprint = _digest(
        payload.get("artifact_fingerprint"),
        "fingerprint",
    )
    if (
        _artifact_fingerprint(config, contract_fingerprint, state)
        != expected_artifact_fingerprint
    ):
        raise ValueError("RLT encoder artifact fingerprint disagrees")

    with torch.random.fork_rng(devices=[], enabled=True):
        encoder = FrozenRLTokenEncoder(
            config,
            representation_contract=representation,
            representation_contract_fingerprint=contract_fingerprint,
            artifact_fingerprint=expected_artifact_fingerprint,
        ).to(dtype=artifact_dtype)
    try:
        encoder.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("RLT encoder artifact tensors disagree") from error
    target_dtype = artifact_dtype if dtype is None else dtype
    if target_dtype not in _SUPPORTED_DTYPES:
        raise ValueError("RLT encoder target dtype is unsupported")
    encoder.to(device=torch.device(device), dtype=target_dtype)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


__all__ = [
    "FrozenRLTokenEncoder",
    "RLTokenConfig",
    "TokenSelection",
    "load_frozen_rl_token_encoder",
]
