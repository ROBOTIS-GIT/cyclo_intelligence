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
import json
import math
from typing import Any

import torch
from torch import Tensor, nn

from cyclo_brain.model.mlp import RLTGaussianChunkActor

from .shadow import RLTStage2InferenceSpec


RLTStage2Spec = RLTStage2InferenceSpec
_LEARNER_STATE_FORMAT = "cyclo_brain.rlt.stage2_learner/v1"


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def stage2_spec_fingerprint(spec: RLTStage2Spec) -> str:
    if not isinstance(spec, RLTStage2InferenceSpec):
        raise TypeError("RLT Stage 2 spec must be RLTStage2Spec")
    return _canonical_fingerprint(asdict(spec))


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"RLT Stage 2 {name} must be a lowercase SHA-256 digest")
    return value


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


def _validate_stage2_spec(spec: RLTStage2Spec) -> None:
    if not isinstance(spec, RLTStage2InferenceSpec):
        raise TypeError("RLT Stage 2 spec must be RLTStage2Spec")
    # The active GR00T RLT deployment contract is deliberately not generic.
    if spec.chunk_length != 10 or spec.action_dim != 19:
        raise ValueError("RLT Stage 2 requires the 10x19 action-chunk contract")
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

    def __post_init__(self) -> None:
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
    reward: Tensor
    bootstrap_discount: Tensor
    next_z_rl: Tensor
    next_proprio: Tensor
    next_reference_actions: Tensor

    def validate(self, spec: RLTStage2Spec) -> None:
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
        chunk_shape = (batch_size, 10, 19)
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
            _validate_float_tensor(
                value,
                name,
                shape,
                dtype=dtype,
                device=device,
            )
        if bool((self.bootstrap_discount < 0.0).any()) or bool(
            (self.bootstrap_discount > 1.0).any()
        ):
            raise ValueError("RLT Stage 2 bootstrap_discount must be in [0, 1]")

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
            spec.rl_token_dim + spec.proprio_dim + 10 * 19,
            hidden_dims,
            1,
        )

    def forward(self, z_rl: Tensor, proprio: Tensor, actions: Tensor) -> Tensor:
        inputs = torch.cat((z_rl, proprio, actions.flatten(start_dim=1)), dim=-1)
        return self.network(inputs)


class RLTStage2TwinCritic(nn.Module):
    """Two parameter-independent scalar Q functions over one 10x19 chunk."""

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
        expected = (spec.rl_token_dim, spec.proprio_dim, 10, 19)
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
        self.critic = critic.train().requires_grad_(True)
        self.critic_target = copy.deepcopy(critic).eval().requires_grad_(False)
        self.spec = spec
        self.config = config
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
            torch.manual_seed(random_seed)
            actor = RLTGaussianChunkActor(
                spec.rl_token_dim,
                spec.proprio_dim,
                10,
                19,
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
    ) -> Tensor:
        mean = self.actor(z_rl, proprio, reference)
        return mean + self.actor.fixed_standard_deviation.to(mean) * self._noise(
            tuple(mean.shape), generator
        )

    @torch.no_grad()
    def bellman_target(self, batch: RLTStage2Batch) -> Tensor:
        batch.validate(self.spec)
        next_actions = self._sample_actor(
            batch.next_z_rl,
            batch.next_proprio,
            batch.next_reference_actions,
            self.target_generator,
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
        batch.validate(self.spec)
        if batch.z_rl.device != self.device or batch.z_rl.dtype != self.dtype:
            raise ValueError("RLT Stage 2 batch and learner must share dtype/device")
        target = self.bellman_target(batch)
        q1, q2 = self.critic(batch.z_rl, batch.proprio, batch.executed_actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(
            q2, target
        )
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

        actor_due = (
            self.completed_critic_updates % self.config.critic_updates_per_actor == 0
        )
        actor_loss: Tensor | None = None
        negative_q: Tensor | None = None
        reference_constraint: Tensor | None = None
        if actor_due:
            rows = torch.rand(
                (batch.batch_size,),
                device=self.device,
                dtype=self.dtype,
                generator=self.dropout_generator,
            ) < self.config.reference_dropout_probability
            actor_reference = batch.reference_actions.masked_fill(
                rows[:, None, None], 0.0
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            with _frozen_parameters(self.critic):
                sampled = self._sample_actor(
                    batch.z_rl,
                    batch.proprio,
                    actor_reference,
                    self.actor_generator,
                )
                policy_q = self.critic.q1(batch.z_rl, batch.proprio, sampled)
                negative_q = -policy_q.mean()
                reference_constraint = (
                    (sampled - batch.reference_actions)
                    .square()
                    .sum(dim=(1, 2))
                    .mean()
                )
                actor_loss = (
                    negative_q
                    + self.config.policy_constraint_weight * reference_constraint
                )
                actor_loss.backward()
            self._clip(self.actor, self.config.actor_gradient_clip_norm, "actor")
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)
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

        return RLTStage2Update(
            critic_loss=float(critic_loss.detach().cpu().item()),
            target_mean=float(target.detach().mean().cpu().item()),
            actor_updated=actor_due,
            actor_loss=scalar(actor_loss),
            actor_negative_q=scalar(negative_q),
            actor_reference_constraint=scalar(reference_constraint),
            completed_critic_updates=self.completed_critic_updates,
            completed_actor_updates=self.completed_actor_updates,
        )

    def _contract(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "spec_fingerprint": stage2_spec_fingerprint(self.spec),
            "config": asdict(self.config),
            "actor_hidden_dims": self.actor_hidden_dims,
            "critic_hidden_dims": self.critic.hidden_dims,
            "random_seed": self.random_seed,
            "dtype": str(self.dtype),
        }

    def state_dict(self) -> dict[str, Any]:
        return _cpu_tree(
            {
                "format": _LEARNER_STATE_FORMAT,
                "contract": self._contract(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "completed_critic_updates": self.completed_critic_updates,
                "completed_actor_updates": self.completed_actor_updates,
                "target_generator": self.target_generator.get_state(),
                "actor_generator": self.actor_generator.get_state(),
                "dropout_generator": self.dropout_generator.get_state(),
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
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
            or set(state) != required
            or state.get("format") != _LEARNER_STATE_FORMAT
            or state.get("contract") != self._contract()
        ):
            raise ValueError("RLT Stage 2 learner checkpoint contract disagrees")
        critic_updates = state.get("completed_critic_updates")
        actor_updates = state.get("completed_actor_updates")
        if (
            isinstance(critic_updates, bool)
            or not isinstance(critic_updates, int)
            or critic_updates < 0
            or actor_updates != critic_updates // self.config.critic_updates_per_actor
        ):
            raise ValueError("RLT Stage 2 checkpoint update counters disagree")
        try:
            self.actor.load_state_dict(state["actor"], strict=True)
            self.critic.load_state_dict(state["critic"], strict=True)
            self.critic_target.load_state_dict(state["critic_target"], strict=True)
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            for generator, name in (
                (self.target_generator, "target_generator"),
                (self.actor_generator, "actor_generator"),
                (self.dropout_generator, "dropout_generator"),
            ):
                value = state[name]
                if not isinstance(value, Tensor) or value.dtype != torch.uint8:
                    raise ValueError(f"RLT Stage 2 {name} state is invalid")
                generator.set_state(value.cpu())
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("RLT Stage 2 learner tensor state disagrees") from error
        _move_optimizer(self.actor_optimizer, self.device)
        _move_optimizer(self.critic_optimizer, self.device)
        self.completed_critic_updates = critic_updates
        self.completed_actor_updates = actor_updates
        self.actor.train().requires_grad_(True)
        self.critic.train().requires_grad_(True)
        self.critic_target.eval().requires_grad_(False)
        for module, name in (
            (self.actor, "actor"),
            (self.critic, "critic"),
            (self.critic_target, "target critic"),
        ):
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
