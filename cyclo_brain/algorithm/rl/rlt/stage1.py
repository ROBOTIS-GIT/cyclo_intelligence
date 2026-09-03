"""Compact trainer for PI RLT Stage-1 representation learning.

The caller extracts final-layer token embeddings with a frozen GR00T model.
This trainer owns only the RL-token encoder and reconstruction decoder, so no
GR00T parameter can accidentally enter its optimizer or checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor

from .rl_token import (
    RLTokenAutoencoder,
    build_frozen_rl_token_encoder_artifact,
    load_frozen_rl_token_encoder,
    rl_token_reconstruction_loss,
)


_STAGE1_CHECKPOINT_FORMAT = "cyclo_brain.rlt.stage1/v1"


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _finite_positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"RLT Stage 1 {name} must be finite and positive")


@dataclass(frozen=True)
class RLTokenStage1Config:
    """Optimizer settings; architecture remains in ``RLTokenConfig``."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        _finite_positive(self.learning_rate, "learning_rate")
        _finite_positive(self.adam_epsilon, "adam_epsilon")
        _finite_positive(self.max_grad_norm, "max_grad_norm")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
        ):
            raise ValueError(
                "RLT Stage 1 weight_decay must be finite and non-negative"
            )
        for value, name in (
            (self.adam_beta1, "adam_beta1"),
            (self.adam_beta2, "adam_beta2"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(f"RLT Stage 1 {name} must be finite in [0, 1)")


@dataclass(frozen=True)
class RLTokenStage1Metrics:
    """One optimizer update, suitable for JSON progress reporting."""

    completed_steps: int
    reconstruction_loss: float
    mean_token_l2: float
    element_mse: float
    valid_tokens: int
    grad_norm: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.lexists(temporary_path):
            os.unlink(temporary_path)


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


class RLTokenStage1Trainer:
    """Train and checkpoint only the Stage-1 encoder-decoder parameters."""

    def __init__(
        self,
        model: RLTokenAutoencoder,
        representation_contract: Mapping[str, Any],
        *,
        config: RLTokenStage1Config | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(model, RLTokenAutoencoder):
            raise TypeError("RLT Stage 1 trainer requires RLTokenAutoencoder")
        if not isinstance(representation_contract, Mapping):
            raise TypeError("RLT Stage 1 trainer requires a representation contract")
        representation = dict(representation_contract)
        embeddings = representation.get("embeddings")
        if (
            not isinstance(embeddings, Mapping)
            or embeddings.get("width") != model.config.embedding_dim
        ):
            raise ValueError("RLT Stage 1 representation width disagrees")
        self.model = model.to(device=torch.device(device))
        self.representation_contract = representation
        self.representation_contract_fingerprint = _canonical_fingerprint(
            representation
        )
        self.config = config or RLTokenStage1Config()
        self.optimizer = torch.optim.AdamW(
            tuple(self.model.parameters()),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            betas=(float(self.config.adam_beta1), float(self.config.adam_beta2)),
            eps=float(self.config.adam_epsilon),
        )
        self.completed_steps = 0
        self.last_metrics: RLTokenStage1Metrics | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train_step(
        self,
        tokens: Tensor,
        token_valid: Tensor,
        image_token: Tensor | None = None,
    ) -> RLTokenStage1Metrics:
        """Run one reconstruction update on detached frozen-GR00T tokens."""
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(tokens, token_valid, image_token)
        reconstruction = rl_token_reconstruction_loss(
            output,
            reduction=self.model.config.loss_reduction,
        )
        if not bool(torch.isfinite(reconstruction.loss)):
            raise FloatingPointError("RLT Stage 1 reconstruction loss is non-finite")
        reconstruction.loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float(self.config.max_grad_norm),
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.completed_steps += 1

        def scalar(value: Tensor, name: str) -> float:
            result = float(value.detach().float().cpu().item())
            if not math.isfinite(result):
                raise FloatingPointError(f"RLT Stage 1 {name} is non-finite")
            return result

        metrics = RLTokenStage1Metrics(
            completed_steps=self.completed_steps,
            reconstruction_loss=scalar(reconstruction.loss, "loss"),
            mean_token_l2=scalar(reconstruction.mean_token_l2, "mean_token_l2"),
            element_mse=scalar(reconstruction.element_mse, "element_mse"),
            valid_tokens=reconstruction.valid_tokens,
            grad_norm=scalar(grad_norm_tensor, "grad_norm"),
        )
        self.last_metrics = metrics
        return metrics

    def progress(self) -> dict[str, int | float | None]:
        if self.last_metrics is None:
            return {
                "completed_steps": self.completed_steps,
                "reconstruction_loss": None,
                "mean_token_l2": None,
                "element_mse": None,
                "valid_tokens": 0,
                "grad_norm": None,
            }
        return self.last_metrics.as_dict()

    def save_checkpoint(self, path: str | os.PathLike[str]) -> Path:
        checkpoint_path = Path(path).expanduser().resolve()
        payload = {
            "format": _STAGE1_CHECKPOINT_FORMAT,
            "completed_steps": self.completed_steps,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "representation_contract": self.representation_contract,
            "representation_contract_fingerprint": (
                self.representation_contract_fingerprint
            ),
            "model": _cpu_clone(self.model.state_dict()),
            "optimizer": _cpu_clone(self.optimizer.state_dict()),
        }
        _atomic_torch_save(checkpoint_path, payload)
        return checkpoint_path

    def load_checkpoint(self, path: str | os.PathLike[str]) -> int:
        checkpoint_path = Path(path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("format") != (
            _STAGE1_CHECKPOINT_FORMAT
        ):
            raise ValueError("RLT Stage 1 checkpoint format is invalid")
        required = {
            "format",
            "completed_steps",
            "model_config",
            "training_config",
            "representation_contract",
            "representation_contract_fingerprint",
            "model",
            "optimizer",
        }
        if set(payload) != required:
            raise ValueError("RLT Stage 1 checkpoint fields are invalid")
        if payload["model_config"] != asdict(self.model.config):
            raise ValueError("RLT Stage 1 model config disagrees")
        if payload["training_config"] != asdict(self.config):
            raise ValueError("RLT Stage 1 training config disagrees")
        if payload["representation_contract"] != self.representation_contract:
            raise ValueError("RLT Stage 1 representation contract disagrees")
        if payload["representation_contract_fingerprint"] != (
            self.representation_contract_fingerprint
        ):
            raise ValueError("RLT Stage 1 representation fingerprint disagrees")
        completed_steps = payload["completed_steps"]
        if (
            isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or completed_steps < 0
        ):
            raise ValueError("RLT Stage 1 completed step count is invalid")
        try:
            self.model.load_state_dict(payload["model"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
        except (KeyError, RuntimeError, ValueError) as error:
            raise ValueError("RLT Stage 1 checkpoint state disagrees") from error
        for state in self.optimizer.state.values():
            for name, value in tuple(state.items()):
                if isinstance(value, Tensor):
                    state[name] = value.to(device=self.device)
        self.completed_steps = completed_steps
        self.last_metrics = None
        return self.completed_steps

    def export_encoder(self, path: str | os.PathLike[str]) -> Path:
        """Export and immediately verify the decoder-free runtime artifact."""
        artifact_path = Path(path).expanduser().resolve()
        payload = build_frozen_rl_token_encoder_artifact(
            self.model,
            self.representation_contract,
        )
        _atomic_torch_save(artifact_path, payload)
        loaded = load_frozen_rl_token_encoder(artifact_path, device="cpu")
        if loaded.artifact_fingerprint != payload["artifact_fingerprint"]:
            raise RuntimeError("RLT Stage 1 encoder export verification failed")
        return artifact_path


__all__ = [
    "RLTokenStage1Config",
    "RLTokenStage1Metrics",
    "RLTokenStage1Trainer",
]
