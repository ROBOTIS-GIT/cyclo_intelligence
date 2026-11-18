"""Compact PI RLT Stage-2 actor/critic learner.

The frozen GR00T policy and frozen Stage-1 RL-token encoder stay outside this
module.  Stage 2 consumes only their detached outputs, which makes the
optimizer boundary explicit: the Action MLP and twin Q critics are the only
trainable networks owned here.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from typing import Any

import torch
from torch import Tensor, nn

from cyclo_brain.algorithm.common import (
    canonical_json_sha256,
    validate_lowercase_sha256,
)
from cyclo_brain.model.common import RLT_ACTION_HORIZON
from cyclo_brain.model.mlp import RLTGaussianChunkActor

from .shadow import RLTStage2InferenceSpec


RLTStage2Spec = RLTStage2InferenceSpec
_LEARNER_STATE_FORMAT = "cyclo_brain.rlt.stage2_learner/v1"
_LEGACY_CUDA_GENERATOR_STATE_BYTES = 16


def stage2_spec_fingerprint(spec: RLTStage2Spec) -> str:
    if not isinstance(spec, RLTStage2InferenceSpec):
        raise TypeError("RLT Stage 2 spec must be RLTStage2Spec")
    return canonical_json_sha256(asdict(spec))


def _digest(value: Any, name: str) -> str:
    return validate_lowercase_sha256(
        value,
        error_message=(
            f"RLT Stage 2 {name} must be a lowercase SHA-256 digest"
        ),
    )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"RLT Stage 2 {name} must be a positive integer")
    return value


def _finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"RLT Stage 2 {name} must be finite")
    return float(value)


def _actor_warmup_data_record(value: Any) -> dict[str, Any]:
    required = {
        "scope",
        "replay_transition_count",
        "eligible_transition_count",
        "eligible_indices_sha256",
    }
    if (not isinstance(value, Mapping) or set(value) not in (required, required | {"target_source"})
            or ("target_source" in value and value["target_source"] != "human")):
        raise ValueError("RLT Actor warmup data provenance is invalid")
    scope = value.get("scope")
    if scope not in {"all", "success_only"}:
        raise ValueError("RLT Actor warmup data scope is invalid")
    replay_count = _positive_integer(
        value.get("replay_transition_count"),
        "Actor warmup replay transition count",
    )
    eligible_count = _positive_integer(
        value.get("eligible_transition_count"),
        "Actor warmup eligible transition count",
    )
    if eligible_count > replay_count or (
        scope == "all" and value.get("target_source") != "human" and eligible_count != replay_count
    ):
        raise ValueError("RLT Actor warmup transition counts disagree")
    return {
        **({"target_source": "human"} if value.get("target_source") == "human" else {}),
        "scope": scope,
        "replay_transition_count": replay_count,
        "eligible_transition_count": eligible_count,
        "eligible_indices_sha256": _digest(
            value.get("eligible_indices_sha256"),
            "Actor warmup eligible indices fingerprint",
        ),
    }


def _hidden_dims(value: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"RLT Stage 2 {name} must contain positive integers")
    result = tuple(value)
    if not result or any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1
        for width in result
    ):
        raise ValueError(f"RLT Stage 2 {name} must contain positive integers")
    return result


def _restore_generator_state(
    generator: torch.Generator,
    value: Any,
    name: str,
) -> None:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise ValueError(f"RLT Stage 2 {name} state is invalid")
    state = value.detach().cpu().contiguous()
    try:
        generator.set_state(state)
        return
    except RuntimeError:
        if state.numel() != _LEGACY_CUDA_GENERATOR_STATE_BYTES:
            raise

    digest = sha256(
        b"cyclo_brain.rlt.stage2.legacy_cuda_rng/v1\0"
        + name.encode("utf-8")
        + b"\0"
        + bytes(state.tolist())
    ).digest()
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 4))


def _validate_stage2_spec(spec: RLTStage2Spec) -> None:
    if not isinstance(spec, RLTStage2InferenceSpec):
        raise TypeError("RLT Stage 2 spec must be RLTStage2Spec")
    # Preserve the legacy profile alongside the 32-step dual-arm GR00T profile.
    if (
        spec.chunk_length != RLT_ACTION_HORIZON
        or (spec.reference_horizon, spec.action_dim) not in ((16, 19), (32, 16))
        or spec.proprio_dim != spec.action_dim
    ):
        raise ValueError("RLT Stage 2 requires a 10-step chunk with a 16x19 or 32x16 reference")
    _digest(spec.reference_contract_fingerprint, "GR00T contract fingerprint")
    _digest(spec.rl_token_artifact_fingerprint, "RL-token artifact fingerprint")


@dataclass(frozen=True)
class RLTStage2FrozenSource:
    """Identity of inputs produced by frozen GR00T and the Stage-1 encoder."""

    groot_checkpoint: str
    groot_checkpoint_fingerprint: str
    representation_contract_fingerprint: str
    rl_token_artifact_fingerprint: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.groot_checkpoint, str)
            or not self.groot_checkpoint.strip()
            or self.groot_checkpoint != self.groot_checkpoint.strip()
        ):
            raise ValueError("RLT Stage 2 GR00T checkpoint must be a non-empty path")
        _digest(self.groot_checkpoint_fingerprint, "GR00T checkpoint fingerprint")
        _digest(
            self.representation_contract_fingerprint,
            "GR00T representation fingerprint",
        )
        _digest(self.rl_token_artifact_fingerprint, "RL-token artifact fingerprint")

    def validate_spec(self, spec: RLTStage2Spec) -> None:
        _validate_stage2_spec(spec)
        # The GR00T weight digest is separate provenance.  The Stage-2 spec is
        # bound to the token representation contract shared by GR00T and the
        # frozen encoder, not to the checkpoint file digest.
        if self.representation_contract_fingerprint != (
            spec.reference_contract_fingerprint
        ):
            raise ValueError(
                "RLT Stage 2 frozen GR00T representation disagrees with spec"
            )
        if self.rl_token_artifact_fingerprint != spec.rl_token_artifact_fingerprint:
            raise ValueError("RLT Stage 2 frozen RL-token identity disagrees with spec")


@dataclass(frozen=True)
class RLTStage2Config:
    """Optimization constants for the paper's constrained actor objective."""

    fixed_standard_deviation: float = 0.1
    policy_constraint_weight: float = 1.0
    target_update_rate: float = 0.005
    reference_dropout_probability: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    critic_updates_per_actor: int = 2
    actor_gradient_clip_norm: float | None = None
    critic_gradient_clip_norm: float | None = None
    mlp_normalization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mlp_normalization is not None:
            from .normalization import validate_normalization
            object.__setattr__(self, "mlp_normalization", validate_normalization(self.mlp_normalization))
        normalized = {
            name: _finite(getattr(self, name), name)
            for name in (
                "fixed_standard_deviation",
                "policy_constraint_weight",
                "target_update_rate",
                "reference_dropout_probability",
                "actor_learning_rate",
                "critic_learning_rate",
            )
        }
        if normalized["fixed_standard_deviation"] <= 0.0:
            raise ValueError("RLT Stage 2 fixed_standard_deviation must be positive")
        if normalized["policy_constraint_weight"] < 0.0:
            raise ValueError(
                "RLT Stage 2 policy_constraint_weight must be non-negative"
            )
        if not 0.0 < normalized["target_update_rate"] <= 1.0:
            raise ValueError("RLT Stage 2 target_update_rate must be in (0, 1]")
        if not 0.0 <= normalized["reference_dropout_probability"] < 1.0:
            raise ValueError(
                "RLT Stage 2 reference_dropout_probability must be in [0, 1)"
            )
        for name in ("actor_learning_rate", "critic_learning_rate"):
            if normalized[name] <= 0.0:
                raise ValueError(f"RLT Stage 2 {name} must be positive")
        _positive_integer(self.critic_updates_per_actor, "critic_updates_per_actor")
        for name in ("actor_gradient_clip_norm", "critic_gradient_clip_norm"):
            value = getattr(self, name)
            if value is not None and _finite(value, name) <= 0.0:
                raise ValueError(f"RLT Stage 2 {name} must be positive when set")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


def stage2_config_payload(config):
    result = asdict(config)
    if result.get("mlp_normalization") is None:
        result.pop("mlp_normalization", None)
    return result


def _validate_float_tensor(
    value: Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"RLT Stage 2 {name} must have shape {shape}")
    if not value.is_floating_point() or value.requires_grad:
        raise ValueError(f"RLT Stage 2 {name} must be detached floating data")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"RLT Stage 2 {name} dtype disagrees")
    if device is not None and value.device != device:
        raise ValueError(f"RLT Stage 2 {name} device disagrees")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"RLT Stage 2 {name} contains NaN or Inf")


@dataclass(frozen=True)
class RLTStage2Batch:
    """Precomputed chunk transition from frozen GR00T/RL-token features.

    ``reward`` is the discounted return within the executed chunk and
    ``bootstrap_discount`` is zero for terminal rows or gamma**duration for a
    valid next state. Reward construction remains outside this compact core.
    """

    spec_fingerprint: str
    z_rl: Tensor
    proprio: Tensor
    reference_actions: Tensor
    executed_actions: Tensor
    reward: Tensor | None = None
    bootstrap_discount: Tensor | None = None
    next_z_rl: Tensor | None = None
    next_proprio: Tensor | None = None
    next_reference_actions: Tensor | None = None
    committed_prefix: Tensor | None = None
    next_committed_prefix: Tensor | None = None
    # Episode-level provenance is deliberately kept outside the persisted
    # replay tensor contract.  Explicit RL training requires this mask so a
    # successful human intervention remains the Actor's behavior target.
    episode_success: Tensor | None = None

    imitation_mask: Tensor | None = None
    replay_role: str = "transition"
    mlp_normalization_id: str | None = None

    def validate(self, spec: RLTStage2Spec, *, actor_only: bool = False) -> None:
        if not actor_only and self.replay_role not in {"transition", "tt_rtc_transition/v1"}:
            raise ValueError("TT-RTC Actor-only replay cannot train Critic/RL: delayed transitions are required")
        _validate_stage2_spec(spec)
        if _digest(self.spec_fingerprint, "batch spec fingerprint") != (
            stage2_spec_fingerprint(spec)
        ):
            raise ValueError("RLT Stage 2 batch identity disagrees with learner")
        if not isinstance(self.z_rl, Tensor) or self.z_rl.ndim != 2:
            raise ValueError("RLT Stage 2 z_rl must have shape (B, Z)")
        batch_size = int(self.z_rl.shape[0])
        if batch_size < 1:
            raise ValueError("RLT Stage 2 batch must not be empty")
        dtype = self.z_rl.dtype
        device = self.z_rl.device
        chunk_shape = (batch_size, spec.chunk_length, spec.action_dim)
        for value, name, shape in (
            (self.z_rl, "z_rl", (batch_size, spec.rl_token_dim)),
            (self.proprio, "proprio", (batch_size, spec.proprio_dim)),
            (self.reference_actions, "reference_actions", chunk_shape),
            (self.executed_actions, "executed_actions", chunk_shape),
            (self.reward, "reward", (batch_size, 1)),
            (self.bootstrap_discount, "bootstrap_discount", (batch_size, 1)),
            (self.next_z_rl, "next_z_rl", (batch_size, spec.rl_token_dim)),
            (self.next_proprio, "next_proprio", (batch_size, spec.proprio_dim)),
            (self.next_reference_actions, "next_reference_actions", chunk_shape),
        ):
            if actor_only and name in {"reward", "bootstrap_discount", "next_z_rl", "next_proprio", "next_reference_actions"}:
                continue
            _validate_float_tensor(
                value,
                name,
                shape,
                dtype=dtype,
                device=device,
            )
        if not actor_only and (bool((self.bootstrap_discount < 0.0).any()) or bool(
            (self.bootstrap_discount > 1.0).any()
        )):
            raise ValueError("RLT Stage 2 bootstrap_discount must be in [0, 1]")
        if self.imitation_mask is not None:
            mask = self.imitation_mask
            if (tuple(mask.shape) != (batch_size, spec.chunk_length) or mask.dtype != torch.bool
                    or mask.device != device or mask.requires_grad):
                raise ValueError("Actor imitation_mask must be detached bool (B, C)")
        for value, name in (
            (self.committed_prefix, "committed_prefix"),
            (self.next_committed_prefix, "next_committed_prefix"),
        ):
            if value is not None:
                _validate_float_tensor(
                    value,
                    name,
                    (batch_size, 6, spec.action_dim),
                    dtype=dtype,
                    device=device,
                )
        if (
            not actor_only
            and self.replay_role == "tt_rtc_transition/v1"
            and (self.committed_prefix is None or self.next_committed_prefix is None)
        ):
            raise ValueError(
                "TT-RTC Critic/RL requires current and next committed prefixes"
            )
        if self.replay_role != "transition" and self.imitation_mask is None:
            raise ValueError("TT-RTC Actor replay requires an explicit human imitation mask")
        if self.episode_success is not None and (
            not isinstance(self.episode_success, Tensor)
            or tuple(self.episode_success.shape) != (batch_size,)
            or self.episode_success.dtype != torch.bool
            or self.episode_success.device != device
            or self.episode_success.requires_grad
        ):
            raise ValueError(
                "RLT Stage 2 episode_success must be detached bool shape (B,) "
                "on the batch device"
            )

    @property
    def batch_size(self) -> int:
        return int(self.z_rl.shape[0])


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class _RLTQFunction(nn.Module):
    def __init__(self, spec: RLTStage2Spec, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.network = _mlp(
            spec.rl_token_dim
            + spec.proprio_dim
            + spec.chunk_length * spec.action_dim,
            hidden_dims,
            1,
        )

    def forward(self, z_rl: Tensor, proprio: Tensor, actions: Tensor) -> Tensor:
        inputs = torch.cat((z_rl, proprio, actions.flatten(start_dim=1)), dim=-1)
        return self.network(inputs)


class RLTStage2TwinCritic(nn.Module):
    """Two parameter-independent scalar Q functions over one action chunk."""

    def __init__(
        self,
        spec: RLTStage2Spec,
        *,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        _validate_stage2_spec(spec)
        self.hidden_dims = _hidden_dims(hidden_dims, "critic hidden_dims")
        self.q1 = _RLTQFunction(spec, self.hidden_dims)
        self.q2 = _RLTQFunction(spec, self.hidden_dims)

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.q1(z_rl, proprio, actions), self.q2(z_rl, proprio, actions)


@dataclass(frozen=True)
class RLTStage2Update:
    critic_loss: float
    target_mean: float
    actor_updated: bool
    actor_loss: float | None
    actor_negative_q: float | None
    actor_reference_constraint: float | None
    completed_critic_updates: int
    completed_actor_updates: int
    target_abs_max: float | None = None
    policy_q_abs_max: float | None = None
    policy_deviation_max: float | None = None


def rl_update_diagnostics(update, *, reward_contract: str, normalized_actions: bool) -> dict[str, float]:
    """Report objective components and abort unmistakable finite divergence.

    Sparse terminal success returns are bounded by one; ten is a deliberately
    loose alarm, not a value clamp. The action alarm is relative to the selected
    behavior target and only applies to the independent MLP coordinate system.
    """
    metrics = {
        name: float(value)
        for name in (
            "target_mean", "target_abs_max", "actor_negative_q",
            "actor_reference_constraint", "policy_q_abs_max", "policy_deviation_max",
        )
        if (value := getattr(update, name)) is not None
    }
    bounded_return = reward_contract in {
        "terminal_success_plus_one_discounted_within_chunk/v1",
        "tt_rtc_decision_td_with_observed_terminal_return/v1",
    }
    for name, value in metrics.items():
        limited = (bounded_return and name in {"target_abs_max", "policy_q_abs_max"}) or (
            normalized_actions and name == "policy_deviation_max"
        )
        if not math.isfinite(value) or (limited and abs(value) > 10.0):
            raise FloatingPointError(
                f"RL divergence: {name}={value:.6g}; limit=10. "
                "No completed bundle will be published. Resume a validated Critic warmup bundle."
            )
    return metrics


@contextmanager
def _frozen_parameters(module: nn.Module):
    flags = tuple(parameter.requires_grad for parameter in module.parameters())
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[name] = value.to(device=device)


def _polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters(), strict=True
        ):
            target_parameter.lerp_(source_parameter, float(tau))


class RLTStage2Learner:
    """Action MLP plus twin-Q learner with exact resumable RNG state."""

    def __init__(
        self,
        actor: RLTGaussianChunkActor,
        critic: RLTStage2TwinCritic,
        spec: RLTStage2Spec,
        config: RLTStage2Config,
        *,
        actor_hidden_dims: Sequence[int],
        random_seed: int = 0,
    ) -> None:
        _validate_stage2_spec(spec)
        if not isinstance(actor, RLTGaussianChunkActor):
            raise TypeError("RLT Stage 2 actor must be RLTGaussianChunkActor")
        if not isinstance(critic, RLTStage2TwinCritic):
            raise TypeError("RLT Stage 2 critic must be RLTStage2TwinCritic")
        if not isinstance(config, RLTStage2Config):
            raise TypeError("RLT Stage 2 config must be RLTStage2Config")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or not (
            0 <= random_seed < 2**63 - 4
        ):
            raise ValueError("RLT Stage 2 random seed is invalid")
        self.actor_hidden_dims = _hidden_dims(actor_hidden_dims, "actor hidden_dims")
        actual_actor = (
            actor.rl_token_dim,
            actor.proprio_dim,
            actor.chunk_length,
            actor.action_dim,
        )
        expected = (
            spec.rl_token_dim,
            spec.proprio_dim,
            spec.chunk_length,
            spec.action_dim,
        )
        if (
            actual_actor != expected
            or tuple(actor.hidden_dims) != self.actor_hidden_dims
        ):
            raise ValueError("RLT Stage 2 actor dimensions disagree with spec")
        if not math.isclose(
            float(actor.fixed_standard_deviation.detach().cpu()),
            config.fixed_standard_deviation,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError("RLT Stage 2 actor standard deviation disagrees")

        devices = {parameter.device for parameter in actor.parameters()} | {
            parameter.device for parameter in critic.parameters()
        }
        dtypes = {parameter.dtype for parameter in actor.parameters()} | {
            parameter.dtype for parameter in critic.parameters()
        }
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("RLT Stage 2 actor and critics must share dtype/device")
        self.device = next(iter(devices))
        self.dtype = next(iter(dtypes))
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("RLT Stage 2 training supports float32 or float64")

        self.actor = actor.train().requires_grad_(True)
        # Captured once at the warmup -> RL boundary, never Polyak-updated.
        self.actor_anchor: RLTGaussianChunkActor | None = None
        self.critic = critic.train().requires_grad_(True)
        self.critic_target = copy.deepcopy(critic).eval().requires_grad_(False)
        self.spec = spec
        self.config = config
        if config.mlp_normalization is not None:
            from .normalization import validate_normalization
            validate_normalization(config.mlp_normalization, spec.action_dim)
        self.actor.mlp_normalization = config.mlp_normalization
        self.random_seed = random_seed
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.target_generator = torch.Generator(device=self.device).manual_seed(
            random_seed
        )
        self.actor_generator = torch.Generator(device=self.device).manual_seed(
            random_seed + 1
        )
        self.dropout_generator = torch.Generator(device=self.device).manual_seed(
            random_seed + 2
        )
        self.completed_critic_updates = 0
        self.completed_actor_updates = 0
        self.completed_actor_warmup_updates = 0
        self.completed_critic_warmup_updates = 0
        self.actor_warmup_data: dict[str, Any] | None = None
        self.completed_phases: list[str] = []
        self.training_phase = "legacy_rl"
        self._pending_update: tuple[RLTStage2Batch, float, float, float] | None = None

    @classmethod
    def create(
        cls,
        spec: RLTStage2Spec,
        config: RLTStage2Config,
        *,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        random_seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "RLTStage2Learner":
        _validate_stage2_spec(spec)
        actor_dims = _hidden_dims(actor_hidden_dims, "actor hidden_dims")
        critic_dims = _hidden_dims(critic_hidden_dims, "critic hidden_dims")
        with torch.random.fork_rng(devices=[], enabled=True):
            # Initialization is CPU-only; do not reseed a live VLA's CUDA RNG.
            torch.random.default_generator.manual_seed(random_seed)
            actor = RLTGaussianChunkActor(
                spec.rl_token_dim,
                spec.proprio_dim,
                spec.chunk_length,
                spec.action_dim,
                fixed_standard_deviation=config.fixed_standard_deviation,
                hidden_dims=actor_dims,
            )
            critic = RLTStage2TwinCritic(spec, hidden_dims=critic_dims)
        actor.to(device=torch.device(device), dtype=dtype)
        critic.to(device=torch.device(device), dtype=dtype)
        return cls(
            actor,
            critic,
            spec,
            config,
            actor_hidden_dims=actor_dims,
            random_seed=random_seed,
        )

    def _noise(self, shape: tuple[int, ...], generator: torch.Generator) -> Tensor:
        return torch.randn(
            shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def _sample_actor(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        reference: Tensor,
        generator: torch.Generator,
        committed_prefix: Tensor | None = None,
    ) -> Tensor:
        mean = self.actor(z_rl, proprio, reference)
        sampled = mean + self.actor.fixed_standard_deviation.to(mean) * self._noise(
            tuple(mean.shape), generator
        )
        return self._apply_mlp_action_policy(sampled, committed_prefix)

    def _validate_batch_normalization(
        self,
        batch: RLTStage2Batch,
        *,
        require_policy_context: bool,
    ) -> None:
        expected = (self.config.mlp_normalization or {}).get("fingerprint")
        if batch.mlp_normalization_id != expected:
            raise ValueError("RLT batch MLP normalization does not match the bundle")
        if expected is not None and require_policy_context and (
            batch.committed_prefix is None or batch.next_committed_prefix is None
        ):
            raise ValueError(
                "MLP Critic/RL requires current and next committed prefixes"
            )

    def _active_action_indices(self) -> list[int] | None:
        if self.config.mlp_normalization is None:
            return None
        from .normalization import active_joint_indices

        return active_joint_indices(self.config.mlp_normalization)

    def _apply_mlp_action_policy(
        self,
        actions: Tensor,
        committed_prefix: Tensor | None,
    ) -> Tensor:
        normalization = self.config.mlp_normalization
        if normalization is None:
            return actions
        if committed_prefix is None:
            raise ValueError("MLP action policy requires a committed prefix")
        hold_indices = tuple(normalization["hold_indices"])
        result = actions.clone()
        result[..., hold_indices] = committed_prefix[:, -1, hold_indices].unsqueeze(1)
        # Right gripper remains trainable, including when resuming legacy bundles.
        return result

    def _constraint_error(self, prediction: Tensor, target: Tensor) -> Tensor:
        indices = self._active_action_indices()
        if indices is None:
            return (prediction - target).square()
        return (prediction[..., indices] - target[..., indices]).square()

    @torch.no_grad()
    def bellman_target(self, batch: RLTStage2Batch) -> Tensor:
        batch.validate(self.spec)
        next_actions = self._sample_actor(
            batch.next_z_rl,
            batch.next_proprio,
            batch.next_reference_actions,
            self.target_generator,
            batch.next_committed_prefix,
        )
        q1, q2 = self.critic_target(
            batch.next_z_rl, batch.next_proprio, next_actions
        )
        return batch.reward + batch.bootstrap_discount * torch.minimum(q1, q2)

    def _clip(self, module: nn.Module, maximum: float | None, name: str) -> None:
        gradients = tuple(
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        if not gradients or any(
            not bool(torch.isfinite(value).all()) for value in gradients
        ):
            raise RuntimeError(
                f"RLT Stage 2 {name} gradients are missing or non-finite"
            )
        if maximum is not None:
            nn.utils.clip_grad_norm_(
                module.parameters(), float(maximum), error_if_nonfinite=True
            )

    def update(self, batch: RLTStage2Batch) -> RLTStage2Update:
        """Existing synchronous entry point, with unchanged optimization order."""
        if self.training_phase not in {"legacy_rl", "rl"}:
            raise ValueError("Use the selected warmup phase; joint RL updates are disabled")
        self.update_critic(batch)
        return self.finish_update()

    def begin_phase(self, phase: str) -> None:
        """Select an explicit phase; never infer it from new/resume or loss."""
        if self.update_pending:
            raise RuntimeError("Finish the pending RLT update before changing phase")
        if phase not in {"actor_warmup", "critic_warmup", "rl"}:
            raise ValueError("Unknown RLT training phase")
        legacy_trained = (
            self.completed_actor_updates > self.completed_actor_warmup_updates
            and self.completed_critic_updates > self.completed_critic_warmup_updates
        )
        if phase == "actor_warmup" and self.completed_critic_updates:
            raise ValueError("Actor warmup cannot resume an already trained Critic")
        if phase == "critic_warmup" and not (
            "actor_warmup" in self.completed_phases or legacy_trained
            or "critic_warmup" in self.completed_phases
        ):
            raise ValueError("Complete Actor warmup before Critic warmup")
        if phase == "rl" and not (
            "critic_warmup" in self.completed_phases or legacy_trained
        ):
            raise ValueError("Complete Critic warmup before RL")
        if phase == "rl" and self.actor_anchor is None:
            if self.completed_actor_updates > self.completed_actor_warmup_updates:
                raise ValueError(
                    "RL checkpoint has no frozen warmup Actor; restart from a "
                    "Critic warmup bundle"
                )
            self.actor_anchor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.training_phase = phase

    def configure_actor_warmup_data(self, provenance: Mapping[str, Any]) -> None:
        """Bind Actor warmup to one explicit, signed replay-row selection."""
        if self.training_phase != "actor_warmup":
            raise ValueError("Actor warmup data can be configured only in Actor warmup")
        validated = _actor_warmup_data_record(provenance)
        if (
            self.completed_actor_warmup_updates
            and self.actor_warmup_data != validated
        ):
            raise ValueError("Actor warmup data cannot change after Actor updates")
        self.actor_warmup_data = validated

    def complete_phase(self) -> None:
        counts = {
            "actor_warmup": self.completed_actor_warmup_updates,
            "critic_warmup": self.completed_critic_warmup_updates,
            "rl": self.completed_critic_updates - self.completed_critic_warmup_updates,
        }
        if self.update_pending or not counts.get(self.training_phase, 0):
            raise ValueError("Cannot complete an empty or pending training phase")
        if self.training_phase not in self.completed_phases:
            self.completed_phases.append(self.training_phase)

    def update_phase(self, batch: RLTStage2Batch) -> RLTStage2Update:
        if self.training_phase == "rl":
            return self.update(batch)
        if self.training_phase == "critic_warmup":
            self.update_critic(batch)
            _, loss, target, target_max = self._pending_update
            self.completed_critic_warmup_updates += 1
            _polyak_update(self.critic, self.critic_target, self.config.target_update_rate)
            self._pending_update = None
            return RLTStage2Update(loss, target, False, None, None, None,
                                   self.completed_critic_updates, self.completed_actor_updates,
                                   target_abs_max=target_max)
        if self.training_phase != "actor_warmup":
            raise ValueError("Select a training phase before updating")
        if self.update_pending:
            raise RuntimeError("Finish the pending RLT update before Actor warmup")
        batch.validate(self.spec, actor_only=True)
        if batch.z_rl.device != self.device or batch.z_rl.dtype != self.dtype:
            raise ValueError("RLT Stage 2 batch and learner must share dtype/device")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._validate_batch_normalization(batch, require_policy_context=False)
        prediction = self.actor(batch.z_rl, batch.proprio, batch.reference_actions)
        # The frozen VLA reference is an Actor input, not the behavior-cloning
        # label.  In intervention replay the recorded executed chunk contains
        # the human correction, so warm-up must imitate that stored behavior.
        # This also keeps warm-up entirely offline: no VLA forward pass or
        # replay regeneration is needed to recover its training target.
        if self.config.mlp_normalization is not None:
            from .normalization import active_joint_indices
            indices = active_joint_indices(self.config.mlp_normalization)
            error = (prediction[..., indices] - batch.executed_actions[..., indices]).square()
            mask = (batch.imitation_mask if batch.imitation_mask is not None else
                    torch.ones(error.shape[:2], dtype=torch.bool, device=error.device))
            if not bool(mask.any()):
                raise ValueError("Actor warmup batch contains no supervised targets")
            loss = error.masked_fill(~mask[..., None], 0).sum() / (mask.sum() * len(indices))
        elif batch.imitation_mask is None:
            loss = nn.functional.mse_loss(prediction, batch.executed_actions)
        else:
            mask = batch.imitation_mask[:, :, None]
            if not bool(mask.any()):
                raise ValueError("Actor warmup batch contains no human action targets")
            error = (prediction - batch.executed_actions).square()
            loss = error.masked_fill(~mask, 0.0).sum() / (mask.sum() * self.spec.action_dim)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite Actor warmup loss")
        loss.backward()
        self._clip(self.actor, self.config.actor_gradient_clip_norm, "actor")
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.completed_actor_updates += 1
        self.completed_actor_warmup_updates += 1
        return RLTStage2Update(0.0, 0.0, True, float(loss.detach().cpu()), None, None,
                               self.completed_critic_updates, self.completed_actor_updates)

    @property
    def update_pending(self) -> bool:
        return self._pending_update is not None

    def update_critic(self, batch: RLTStage2Batch) -> None:
        """Run one critic step, then yield before the optional actor/target step.

        A single caller must serialize these methods. Keep the batch tensors
        unchanged until finish_update(); no autograd graph spans this boundary.
        """
        if self.training_phase == "actor_warmup":
            raise ValueError("Critic updates are disabled during Actor warmup")
        if self.training_phase == "rl" and self.actor_anchor is None:
            raise ValueError(
                "RL requires a frozen warmup Actor; restart from a Critic warmup bundle"
            )
        if self.training_phase == "rl" and batch.episode_success is None:
            raise ValueError(
                "Explicit RL requires episode_success provenance so successful "
                "human behavior remains the Actor target"
            )
        if self.update_pending:
            raise RuntimeError("Finish the pending RLT update before starting another")
        batch.validate(self.spec)
        self._validate_batch_normalization(batch, require_policy_context=True)
        if batch.z_rl.device != self.device or batch.z_rl.dtype != self.dtype:
            raise ValueError("RLT Stage 2 batch and learner must share dtype/device")
        target = self.bellman_target(batch)
        q1, q2 = self.critic(batch.z_rl, batch.proprio, batch.executed_actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(
            q2, target
        )
        if not bool(torch.isfinite(critic_loss)):
            raise FloatingPointError('RLT Stage 2 critic loss became non-finite before optimizer step')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(
            self.critic,
            self.config.critic_gradient_clip_norm,
            "critic",
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.completed_critic_updates += 1
        self._pending_update = (
            batch,
            float(critic_loss.detach().cpu().item()),
            float(target.detach().mean().cpu().item()),
            float(target.detach().abs().max().cpu().item()),
        )

    def finish_update(self) -> RLTStage2Update:
        """Finish the pending update using its original batch, without resampling."""
        if self.training_phase not in {"legacy_rl", "rl"}:
            raise ValueError("Joint RL updates are disabled during warmup")
        if self._pending_update is None:
            raise RuntimeError("No pending RLT update to finish")
        batch, critic_loss, target_mean, target_abs_max = self._pending_update

        actor_due = (
            (self.completed_critic_updates - self.completed_critic_warmup_updates)
            % self.config.critic_updates_per_actor == 0
        )
        actor_loss: Tensor | None = None
        negative_q: Tensor | None = None
        reference_constraint: Tensor | None = None
        policy_q_abs_max: Tensor | None = None
        policy_deviation_max: Tensor | None = None
        if actor_due:
            rows = torch.rand(
                (batch.batch_size,),
                device=self.device,
                dtype=self.dtype,
                generator=self.dropout_generator,
            ) < self.config.reference_dropout_probability
            # Successful intervention rows use the same complete VLA input as
            # Actor warmup.  Reference dropout remains an RL regularizer only
            # for non-success behavior.
            if batch.episode_success is not None:
                rows = rows & ~batch.episode_success
            actor_reference = batch.reference_actions.masked_fill(
                rows[:, None, None], 0.0
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            with _frozen_parameters(self.critic):
                action_mean = self.actor(
                    batch.z_rl,
                    batch.proprio,
                    actor_reference,
                )
                sampled = self._apply_mlp_action_policy(
                    action_mean
                    + self.actor.fixed_standard_deviation.to(action_mean)
                    * self._noise(tuple(action_mean.shape), self.actor_generator),
                    batch.committed_prefix,
                )
                policy_q = self.critic.q1(batch.z_rl, batch.proprio, sampled)
                negative_q = -policy_q.mean()
                policy_q_abs_max = policy_q.detach().abs().max()
                if batch.episode_success is None:
                    # Backward-compatible legacy RL checkpoints have no signed
                    # episode provenance and retain their original objective.
                    constraint_error = self._constraint_error(
                        sampled, batch.reference_actions
                    )
                else:
                    success = batch.episode_success[:, None, None]
                    # On successful intervention data, VLA reference is an
                    # input only.  The deterministic Actor mean must imitate
                    # the recorded human execution, exactly as in warmup.
                    human_error = self._constraint_error(
                        action_mean, batch.executed_actions
                    )
                    # Failure states still need a behavior constraint: otherwise
                    # offline Q maximization can exploit unobserved actions.
                    # Evaluate the immutable warmup policy with full references,
                    # even when the trainable policy received reference dropout.
                    if self.actor_anchor is not None:
                        with torch.no_grad():
                            anchor_mean = self.actor_anchor(
                                batch.z_rl, batch.proprio, batch.reference_actions
                            )
                        failure_error = self._constraint_error(action_mean, anchor_mean)
                    else:
                        # Preserve the legacy (non-explicit-RL) checkpoint path.
                        failure_error = torch.zeros_like(human_error)
                    constraint_error = torch.where(
                        success, human_error, failure_error
                    )
                reference_constraint = constraint_error.sum(dim=(1, 2)).mean()
                policy_deviation_max = constraint_error.detach().max().sqrt()
                actor_loss = (
                    negative_q
                    + self.config.policy_constraint_weight * reference_constraint
                )
                if not bool(torch.isfinite(actor_loss)):
                    raise FloatingPointError('RLT Stage 2 actor loss became non-finite before optimizer step')
                actor_loss.backward()
            self._clip(self.actor, self.config.actor_gradient_clip_norm, "actor")
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.training_phase == "rl":
                # Include the updated inference policy, not only the policy that
                # produced this gradient: the final step may immediately publish.
                with torch.no_grad():
                    updated_mean = self.actor(batch.z_rl, batch.proprio, batch.reference_actions)
                    updated_actions = self._apply_mlp_action_policy(updated_mean, batch.committed_prefix)
                    updated_q1, updated_q2 = self.critic(batch.z_rl, batch.proprio, updated_actions)
                    policy_q_abs_max = torch.maximum(
                        policy_q_abs_max,
                        torch.maximum(updated_q1.abs(), updated_q2.abs()).max(),
                    )
                    behavior_target = torch.where(success, batch.executed_actions, anchor_mean)
                    policy_deviation_max = torch.maximum(
                        policy_deviation_max,
                        self._constraint_error(updated_mean, behavior_target).max().sqrt(),
                    )
            self.completed_actor_updates += 1
            _polyak_update(
                self.critic,
                self.critic_target,
                self.config.target_update_rate,
            )

        def scalar(value: Tensor | None) -> float | None:
            if value is None:
                return None
            result = float(value.detach().cpu().item())
            if not math.isfinite(result):
                raise FloatingPointError("RLT Stage 2 metric became non-finite")
            return result

        result = RLTStage2Update(
            critic_loss=critic_loss,
            target_mean=target_mean,
            actor_updated=actor_due,
            actor_loss=scalar(actor_loss),
            actor_negative_q=scalar(negative_q),
            actor_reference_constraint=scalar(reference_constraint),
            completed_critic_updates=self.completed_critic_updates,
            completed_actor_updates=self.completed_actor_updates,
            target_abs_max=target_abs_max,
            policy_q_abs_max=scalar(policy_q_abs_max),
            policy_deviation_max=scalar(policy_deviation_max),
        )
        self._pending_update = None
        return result

    def _contract(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "spec_fingerprint": stage2_spec_fingerprint(self.spec),
            "config": stage2_config_payload(self.config),
            "actor_hidden_dims": self.actor_hidden_dims,
            "critic_hidden_dims": self.critic.hidden_dims,
            "random_seed": self.random_seed,
            "dtype": str(self.dtype),
        }

    def state_dict(self) -> dict[str, Any]:
        if self.update_pending:
            raise RuntimeError("Finish the pending RLT update before saving")
        phase_state = {
            "version": 1,
            "training_phase": self.training_phase,
            "completed_phases": list(self.completed_phases),
            "actor_warmup_updates": self.completed_actor_warmup_updates,
            "critic_warmup_updates": self.completed_critic_warmup_updates,
        }
        if self.actor_warmup_data is not None:
            phase_state["actor_warmup_data"] = dict(self.actor_warmup_data)
        return _cpu_tree(
            {
                "format": _LEARNER_STATE_FORMAT,
                "contract": self._contract(),
                "actor": self.actor.state_dict(),
                **({"actor_anchor": self.actor_anchor.state_dict()}
                   if self.actor_anchor is not None else {}),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "completed_critic_updates": self.completed_critic_updates,
                "completed_actor_updates": self.completed_actor_updates,
                "phase_state": phase_state,
                "target_generator": self.target_generator.get_state(),
                "actor_generator": self.actor_generator.get_state(),
                "dropout_generator": self.dropout_generator.get_state(),
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.update_pending:
            raise RuntimeError("Finish the pending RLT update before restoring")
        required = {
            "format",
            "contract",
            "actor",
            "critic",
            "critic_target",
            "actor_optimizer",
            "critic_optimizer",
            "completed_critic_updates",
            "completed_actor_updates",
            "target_generator",
            "actor_generator",
            "dropout_generator",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) - {"phase_state", "actor_anchor"} != required
            or state.get("format") != _LEARNER_STATE_FORMAT
            or state.get("contract") != self._contract()
        ):
            raise ValueError("RLT Stage 2 learner checkpoint contract disagrees")
        critic_updates = state.get("completed_critic_updates")
        actor_updates = state.get("completed_actor_updates")
        phase = state.get("phase_state", {
            "version": 1, "training_phase": "legacy_rl", "completed_phases": [],
            "actor_warmup_updates": 0, "critic_warmup_updates": 0,
        })
        phase_fields = {"version", "training_phase", "completed_phases",
                        "actor_warmup_updates", "critic_warmup_updates"}
        if (not isinstance(phase, Mapping)
            or set(phase) not in (phase_fields, phase_fields | {"actor_warmup_data"})
            or type(phase.get("version")) is not int or phase["version"] != 1
            or phase.get("training_phase") not in {"legacy_rl", "actor_warmup", "critic_warmup", "rl"}
            or not isinstance(phase.get("completed_phases"), list)
            or any(p not in {"actor_warmup", "critic_warmup", "rl"} for p in phase["completed_phases"])
            or any(type(phase.get(k)) is not int or phase[k] < 0
                   for k in ("actor_warmup_updates", "critic_warmup_updates"))):
            raise ValueError("RLT phase checkpoint is invalid")
        actor_warmup_data = (
            _actor_warmup_data_record(phase["actor_warmup_data"])
            if "actor_warmup_data" in phase
            else None
        )
        actor_warmup = phase["actor_warmup_updates"]
        critic_warmup = phase["critic_warmup_updates"]
        if (len(set(phase["completed_phases"])) != len(phase["completed_phases"])
            or ("actor_warmup" in phase["completed_phases"] and actor_warmup == 0)
            or ("critic_warmup" in phase["completed_phases"] and critic_warmup == 0)):
            raise ValueError("RLT phase completion counters disagree")
        if (
            isinstance(critic_updates, bool)
            or not isinstance(critic_updates, int)
            or critic_updates < 0
            or critic_warmup > critic_updates
            or type(actor_updates) is not int
            or actor_updates != actor_warmup + (critic_updates - critic_warmup) // self.config.critic_updates_per_actor
        ):
            raise ValueError("RLT Stage 2 checkpoint update counters disagree")
        try:
            self.actor.load_state_dict(state["actor"], strict=True)
            self.actor_anchor = None
            if "actor_anchor" in state:
                self.actor_anchor = copy.deepcopy(self.actor).eval().requires_grad_(False)
                self.actor_anchor.load_state_dict(state["actor_anchor"], strict=True)
            self.critic.load_state_dict(state["critic"], strict=True)
            self.critic_target.load_state_dict(state["critic_target"], strict=True)
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            for generator, name in (
                (self.target_generator, "target_generator"),
                (self.actor_generator, "actor_generator"),
                (self.dropout_generator, "dropout_generator"),
            ):
                _restore_generator_state(generator, state[name], name)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("RLT Stage 2 learner tensor state disagrees") from error
        _move_optimizer(self.actor_optimizer, self.device)
        _move_optimizer(self.critic_optimizer, self.device)
        self.completed_critic_updates = critic_updates
        self.completed_actor_updates = actor_updates
        self.completed_actor_warmup_updates = actor_warmup
        self.completed_critic_warmup_updates = critic_warmup
        self.actor_warmup_data = actor_warmup_data
        self.completed_phases = list(phase["completed_phases"])
        self.training_phase = phase["training_phase"]
        self.actor.train().requires_grad_(True)
        self.critic.train().requires_grad_(True)
        self.critic_target.eval().requires_grad_(False)
        modules = [
            (self.actor, "actor"),
            (self.critic, "critic"),
            (self.critic_target, "target critic"),
        ]
        if self.actor_anchor is not None:
            modules.append((self.actor_anchor, "frozen warmup Actor"))
        for module, name in modules:
            if any(
                value.is_floating_point() and not bool(torch.isfinite(value).all())
                for value in module.state_dict().values()
            ):
                raise ValueError(f"RLT Stage 2 {name} checkpoint is non-finite")


__all__ = [
    "RLTStage2Batch",
    "RLTStage2Config",
    "RLTStage2FrozenSource",
    "RLTStage2Learner",
    "RLTStage2Spec",
    "RLTStage2TwinCritic",
    "RLTStage2Update",
    "stage2_spec_fingerprint",
]
