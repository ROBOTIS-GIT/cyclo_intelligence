"""Model-level Flow-SDE PPO collection and update runner.

The runner deliberately has no ROS, Zenoh, simulator, or UI dependency. A
live bridge only has to implement :class:`FlowSDEEpisodeSource` and return one
complete chunk-level episode.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .batch import FlowSDERollout
from .config import FlowSDEPPOConfig
from .functional import clipped_value_loss, ppo_clipped_actor_loss
from .on_policy import (
    FlowSDEEpisode,
    FlowSDEOnPolicyBuffer,
    FlowSDETransition,
    index_rollout,
    rollout_to,
)
from .sampler import recompute_flow_sde_log_probs, sample_flow_sde_chunk


@runtime_checkable
class FlowSDEPolicyAdapter(Protocol):
    """The subset of ``MultiTaskDiTFlowAdapter`` consumed by this runner."""

    horizon: int
    action_dim: int
    conditioning_dim: int

    def encode_conditioning(self, batch: Mapping[str, Tensor]) -> Tensor: ...

    def velocity(self, latent: Tensor, progress: Tensor, conditioning: Tensor) -> Tensor: ...

    def executed_action_mask(self, batch_size: int, *, device: torch.device | str) -> Tensor: ...

    def executed_actions(self, normalized_chunk: Tensor) -> Tensor: ...

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]: ...


@dataclass(frozen=True)
class FlowSDEActionDecision:
    """Action, likelihood metadata, and old value emitted during collection."""

    conditioning: Tensor
    rollout: FlowSDERollout
    executed_actions: Tensor
    old_values: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.conditioning, Tensor)
            or self.conditioning.ndim != 2
            or not self.conditioning.is_floating_point()
        ):
            raise ValueError("Flow-SDE decision conditioning must have floating shape (B, C)")
        batch_size = self.conditioning.shape[0]
        if self.rollout.chains.shape[0] != batch_size:
            raise ValueError("Flow-SDE decision rollout batch size mismatch")
        if (
            not isinstance(self.executed_actions, Tensor)
            or self.executed_actions.ndim != 3
            or self.executed_actions.shape[0] != batch_size
        ):
            raise ValueError("Flow-SDE executed actions must have shape (B, execution_horizon, A)")
        if (
            not isinstance(self.old_values, Tensor)
            or self.old_values.shape != (batch_size,)
            or not self.old_values.is_floating_point()
        ):
            raise ValueError("Flow-SDE old values must have floating shape (B,)")
        if not bool(torch.isfinite(self.conditioning).all()) or not bool(
            torch.isfinite(self.executed_actions).all()
        ) or not bool(torch.isfinite(self.old_values).all()):
            raise ValueError("Flow-SDE decision tensors must be finite")
        devices = {
            self.conditioning.device,
            self.rollout.chains.device,
            self.executed_actions.device,
            self.old_values.device,
        }
        if len(devices) != 1:
            raise ValueError("Flow-SDE decision tensors must share one device")

    def as_transition(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> FlowSDETransition:
        """Offload a single-environment decision into the episode buffer."""

        if self.conditioning.shape[0] != 1:
            raise ValueError("Flow-SDE episode transitions currently require a single environment")
        return FlowSDETransition(
            conditioning=self.conditioning[0].detach().float().cpu(),
            rollout=rollout_to(self.rollout, "cpu"),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            old_value=float(self.old_values[0].detach().cpu()),
        )


@dataclass(frozen=True)
class FlowSDEPPOUpdateMetrics:
    update_step: int
    episodes: int
    transitions: int
    minibatches: int
    reward_mean: float
    episode_return_mean: float
    actor_loss: float
    value_loss: float
    total_loss: float
    ratio: float
    approx_kl: float
    clip_fraction: float
    advantage_mean: float
    advantage_std: float
    actor_grad_norm: float
    value_grad_norm: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@runtime_checkable
class FlowSDEEpisodeSource(Protocol):
    """Injected simulator/robot collector boundary.

    A source calls ``runner.sample_*`` for each policy decision, executes the
    returned action chunk, and labels reward plus termination. No environment
    implementation is bundled with the algorithm package.
    """

    def collect_episode(self, runner: "FlowSDEPPOTrainer") -> FlowSDEEpisode: ...


class FlowSDEPPOTrainer:
    """Train a MultiTaskDiT action head and conditioning-value MLP with PPO."""

    CHECKPOINT_FORMAT = "cyclo.flow_sde_ppo.training.v1"
    EXPORT_FORMAT = "cyclo.flow_sde_ppo.actor.v1"

    def __init__(
        self,
        adapter: FlowSDEPolicyAdapter,
        value_head: nn.Module,
        *,
        config: FlowSDEPPOConfig | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        value_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        required_adapter_members = (
            "horizon",
            "action_dim",
            "conditioning_dim",
            "encode_conditioning",
            "velocity",
            "executed_action_mask",
            "executed_actions",
            "trainable_parameters",
        )
        if any(not hasattr(adapter, member) for member in required_adapter_members):
            raise TypeError("Flow-SDE trainer requires a compatible policy adapter")
        if not isinstance(value_head, nn.Module):
            raise TypeError("Flow-SDE trainer value head must be a torch module")
        self.adapter = adapter
        self.value_head = value_head
        self.config = config or FlowSDEPPOConfig()
        if not isinstance(self.config, FlowSDEPPOConfig):
            raise TypeError("Flow-SDE trainer config must be FlowSDEPPOConfig")

        self.actor_parameters = tuple(adapter.trainable_parameters())
        self.value_parameters = tuple(
            parameter for parameter in value_head.parameters() if parameter.requires_grad
        )
        if not self.actor_parameters:
            raise ValueError("Flow-SDE actor has no trainable parameters")
        if not self.value_parameters:
            raise ValueError("Flow-SDE value head has no trainable parameters")
        if set(map(id, self.actor_parameters)) & set(map(id, self.value_parameters)):
            raise ValueError("Flow-SDE actor and value optimizers must have disjoint parameters")
        actor_device = self.actor_parameters[0].device
        if any(parameter.device != actor_device for parameter in self.actor_parameters):
            raise ValueError("Flow-SDE actor parameters must share one device")
        if any(parameter.device != actor_device for parameter in self.value_parameters):
            raise ValueError("Flow-SDE actor and value head must share one device")
        self.device = actor_device

        self.actor_optimizer = actor_optimizer or torch.optim.AdamW(
            self.actor_parameters,
            lr=self.config.actor_learning_rate,
        )
        self.value_optimizer = value_optimizer or torch.optim.AdamW(
            self.value_parameters,
            lr=self.config.value_learning_rate,
        )
        self._validate_optimizer(self.actor_optimizer, self.actor_parameters, "actor")
        self._validate_optimizer(self.value_optimizer, self.value_parameters, "value")
        self.update_step = 0
        self._value_initialization_provenance: dict[str, Any] | None = None

    @property
    def value_initialization_provenance(self) -> dict[str, Any] | None:
        """Return a defensive copy of the critic initialization contract."""

        return copy.deepcopy(self._value_initialization_provenance)

    def record_value_initialization_provenance(self, provenance: Mapping[str, Any]) -> None:
        """Bind one immutable offline initialization record to this online run."""

        if self.update_step != 0:
            raise RuntimeError("value initialization provenance must be recorded before PPO updates")
        if self._value_initialization_provenance is not None:
            raise RuntimeError("value initialization provenance is already recorded")
        if not isinstance(provenance, Mapping) or not provenance:
            raise TypeError("value initialization provenance must be a non-empty mapping")
        # Checkpoint metadata must stay machine-readable and immutable even if
        # a caller mutates its local result after this boundary.
        try:
            canonical = json.loads(json.dumps(dict(provenance), allow_nan=False, sort_keys=True))
        except (TypeError, ValueError) as error:
            raise ValueError("value initialization provenance must be finite JSON data") from error
        self._value_initialization_provenance = canonical

    @staticmethod
    def _validate_optimizer(
        optimizer: torch.optim.Optimizer,
        expected_parameters: tuple[nn.Parameter, ...],
        name: str,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Flow-SDE {name} optimizer must be a torch optimizer")
        actual = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if actual != set(map(id, expected_parameters)):
            raise ValueError(f"Flow-SDE {name} optimizer parameter set mismatch")

    @torch.no_grad()
    def value(self, conditioning: Tensor) -> Tensor:
        resolved = self._resolve_conditioning(conditioning)
        return self.value_head(resolved).float()

    def _resolve_conditioning(self, conditioning: Tensor) -> Tensor:
        if (
            not isinstance(conditioning, Tensor)
            or conditioning.ndim != 2
            or conditioning.shape[1] != self.adapter.conditioning_dim
            or not conditioning.is_floating_point()
            or not bool(torch.isfinite(conditioning).all())
        ):
            raise ValueError("Flow-SDE conditioning must have finite shape (B, conditioning_dim)")
        return conditioning.detach().to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def sample_from_conditioning(
        self,
        conditioning: Tensor,
        *,
        generator: torch.Generator | None = None,
        initial_noise: Tensor | None = None,
        denoise_indices: Tensor | None = None,
    ) -> FlowSDEActionDecision:
        resolved = self._resolve_conditioning(conditioning)
        action_mask = self.adapter.executed_action_mask(
            resolved.shape[0],
            device=self.device,
        )
        rollout = sample_flow_sde_chunk(
            self.adapter.velocity,
            resolved,
            horizon=self.adapter.horizon,
            action_dim=self.adapter.action_dim,
            config=self.config,
            action_mask=action_mask,
            initial_noise=initial_noise,
            denoise_indices=denoise_indices,
            generator=generator,
        )
        old_values = self.value_head(resolved).float()
        executed_actions = self.adapter.executed_actions(rollout.actions).float()
        return FlowSDEActionDecision(
            conditioning=resolved.detach(),
            rollout=rollout,
            executed_actions=executed_actions.detach(),
            old_values=old_values.detach(),
        )

    @torch.no_grad()
    def sample_preprocessed_batch(
        self,
        batch: Mapping[str, Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> FlowSDEActionDecision:
        """Encode a canonicalized/preprocessed LeRobot batch, then sample."""

        conditioning = self.adapter.encode_conditioning(batch)
        return self.sample_from_conditioning(conditioning, generator=generator)

    def update(self, episodes: tuple[FlowSDEEpisode, ...] | list[FlowSDEEpisode]) -> FlowSDEPPOUpdateMetrics:
        """Consume complete on-policy episodes for multiple PPO epochs."""

        buffer = FlowSDEOnPolicyBuffer()
        buffer.extend(episodes)
        batch = buffer.build_batch(
            discount=self.config.discount,
            gae_lambda=self.config.gae_lambda,
        ).to(self.device)
        advantages = batch.advantages
        advantage_mean = float(advantages.mean().detach().cpu())
        advantage_std_tensor = advantages.std(unbiased=False)
        advantage_std = float(advantage_std_tensor.detach().cpu())
        if self.config.normalize_advantages and advantage_std > 1.0e-8:
            advantages = (advantages - advantages.mean()) / advantage_std_tensor

        batch_size = batch.conditioning.shape[0]
        accumulated: dict[str, float] = {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "ratio": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "value_grad_norm": 0.0,
        }
        minibatches = 0
        for _epoch in range(self.config.ppo_epochs):
            order = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                conditioning = batch.conditioning.index_select(0, indices)
                rollout = index_rollout(batch.rollout, indices)
                new_log_probs = recompute_flow_sde_log_probs(
                    self.adapter.velocity,
                    conditioning,
                    rollout,
                    config=self.config,
                )
                actor_loss, actor_metrics = ppo_clipped_actor_loss(
                    new_log_probs,
                    rollout.old_log_probs,
                    advantages.index_select(0, indices),
                    rollout.action_mask,
                    clip_ratio_low=self.config.clip_ratio_low,
                    clip_ratio_high=self.config.clip_ratio_high,
                )
                values = self.value_head(conditioning).float()
                value_loss = clipped_value_loss(
                    values,
                    batch.old_values.index_select(0, indices),
                    batch.returns.index_select(0, indices),
                    value_clip=self.config.value_clip,
                )
                total_loss = actor_loss + self.config.value_loss_coefficient * value_loss
                if not bool(torch.isfinite(total_loss)):
                    raise RuntimeError("Flow-SDE PPO produced a non-finite loss")

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_parameters,
                    self.config.actor_max_grad_norm,
                )
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_parameters,
                    self.config.value_max_grad_norm,
                )
                if not bool(torch.isfinite(actor_grad_norm)) or not bool(
                    torch.isfinite(value_grad_norm)
                ):
                    raise RuntimeError("Flow-SDE PPO produced non-finite gradients")
                self.actor_optimizer.step()
                self.value_optimizer.step()

                values_to_add = {
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    "total_loss": total_loss,
                    "ratio": actor_metrics["ratio"],
                    "approx_kl": actor_metrics["approx_kl"],
                    "clip_fraction": actor_metrics["clip_fraction"],
                    "actor_grad_norm": actor_grad_norm,
                    "value_grad_norm": value_grad_norm,
                }
                for name, value in values_to_add.items():
                    scalar = float(value.detach().cpu())
                    if not math.isfinite(scalar):
                        raise RuntimeError(f"Flow-SDE PPO metric {name} is non-finite")
                    accumulated[name] += scalar
                minibatches += 1

        if minibatches < 1:
            raise RuntimeError("Flow-SDE PPO did not execute a minibatch")
        averaged = {name: value / minibatches for name, value in accumulated.items()}
        self.update_step += 1
        return FlowSDEPPOUpdateMetrics(
            update_step=self.update_step,
            episodes=batch.episode_returns.numel(),
            transitions=batch_size,
            minibatches=minibatches,
            reward_mean=float(batch.rewards.mean().detach().cpu()),
            episode_return_mean=float(batch.episode_returns.mean().detach().cpu()),
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            **averaged,
        )

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "update_step": self.update_step,
            "actor": self._actor_module().state_dict(),
            "value_head": self.value_head.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "value_initialization_provenance": self.value_initialization_provenance,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        }

    def _actor_module(self) -> nn.Module:
        policy = getattr(self.adapter, "policy", None)
        actor = getattr(policy, "noise_predictor", None)
        if not isinstance(actor, nn.Module):
            raise TypeError("Flow-SDE adapter must expose policy.noise_predictor for checkpoints")
        return actor

    @property
    def policy(self) -> nn.Module:
        """Return the complete base policy for LeRobot-compatible exporting."""

        policy = getattr(self.adapter, "policy", None)
        if not isinstance(policy, nn.Module):
            raise TypeError("Flow-SDE adapter must expose its complete policy module")
        return policy

    @staticmethod
    def _checkpoint_path(path: str | Path, *, filename: str) -> Path:
        resolved = Path(path).expanduser()
        if resolved.suffix:
            return resolved
        return resolved / filename

    def save_checkpoint(self, path: str | Path) -> Path:
        resolved = self._checkpoint_path(path, filename="trainer_state.pt")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved.with_suffix(resolved.suffix + ".tmp")
        torch.save(self.training_state_dict(), temporary)
        temporary.replace(resolved)
        return resolved

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        strict_config: bool = True,
        load_optimizers: bool = True,
    ) -> int:
        resolved = self._checkpoint_path(path, filename="trainer_state.pt")
        checkpoint = torch.load(resolved, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("Not a Cyclo Flow-SDE PPO training checkpoint")
        checkpoint_config = checkpoint.get("config")
        if strict_config and checkpoint_config != asdict(self.config):
            raise ValueError("Flow-SDE PPO checkpoint config does not match the trainer")
        provenance = checkpoint.get("value_initialization_provenance")
        if provenance is not None and (not isinstance(provenance, Mapping) or not provenance):
            raise ValueError("Flow-SDE PPO checkpoint value provenance is invalid")
        update_step = checkpoint.get("update_step")
        if isinstance(update_step, bool) or not isinstance(update_step, int) or update_step < 0:
            raise ValueError("Flow-SDE PPO checkpoint has an invalid update step")
        rng_state = checkpoint.get("torch_rng_state")
        if rng_state is not None and not isinstance(rng_state, Tensor):
            raise ValueError("Flow-SDE PPO checkpoint torch RNG state is invalid")
        cuda_rng_state = checkpoint.get("torch_cuda_rng_state_all")
        if cuda_rng_state is not None and (
            not isinstance(cuda_rng_state, list)
            or not all(isinstance(state, Tensor) for state in cuda_rng_state)
        ):
            raise ValueError("Flow-SDE PPO checkpoint CUDA RNG state is invalid")
        if load_optimizers:
            optimizer_contracts = (
                (
                    "actor",
                    checkpoint.get("actor_optimizer"),
                    self.config.actor_learning_rate,
                ),
                (
                    "value",
                    checkpoint.get("value_optimizer"),
                    self.config.value_learning_rate,
                ),
            )
            for name, optimizer_state, expected_lr in optimizer_contracts:
                if not isinstance(optimizer_state, Mapping):
                    raise ValueError(f"Flow-SDE PPO checkpoint {name} optimizer is invalid")
                parameter_state = optimizer_state.get("state")
                if not isinstance(parameter_state, Mapping) or (
                    update_step > 0 and not parameter_state
                ):
                    raise ValueError(
                        f"Flow-SDE PPO checkpoint {name} optimizer state is invalid"
                    )
                param_groups = optimizer_state.get("param_groups")
                if not isinstance(param_groups, list) or not param_groups:
                    raise ValueError(
                        f"Flow-SDE PPO checkpoint {name} optimizer param groups are invalid"
                    )
                for group in param_groups:
                    learning_rate = group.get("lr") if isinstance(group, Mapping) else None
                    if (
                        isinstance(learning_rate, bool)
                        or not isinstance(learning_rate, (int, float))
                        or not math.isfinite(float(learning_rate))
                        or float(learning_rate) != float(expected_lr)
                    ):
                        raise ValueError(
                            f"Flow-SDE PPO checkpoint {name} optimizer learning rate "
                            "does not match the trainer config"
                        )

        # All metadata and requested learning-rate contracts are checked before
        # any live module is mutated.  Cross-job resume is deliberately
        # fail-closed rather than silently adopting checkpoint hyperparameters.
        self._actor_module().load_state_dict(checkpoint["actor"], strict=True)
        self.value_head.load_state_dict(checkpoint["value_head"], strict=True)
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self._value_initialization_provenance = (
            copy.deepcopy(dict(provenance)) if provenance is not None else None
        )
        self.update_step = update_step
        if isinstance(rng_state, Tensor):
            torch.set_rng_state(rng_state.cpu())
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        return self.update_step

    def export_actor(self, path: str | Path) -> Path:
        """Export only inference-required action-head weights and contract."""

        resolved = self._checkpoint_path(path, filename="flow_sde_actor.pt")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.EXPORT_FORMAT,
            "config": {
                "num_denoising_steps": self.config.num_denoising_steps,
                "noise_level": self.config.noise_level,
                "horizon": self.adapter.horizon,
                "action_dim": self.adapter.action_dim,
                "conditioning_dim": self.adapter.conditioning_dim,
            },
            "actor": self._actor_module().state_dict(),
            "source_update_step": self.update_step,
            "value_initialization_provenance": self.value_initialization_provenance,
        }
        torch.save(payload, resolved)
        return resolved

    def export_pretrained_policy(
        self,
        output_dir: str | Path,
        *,
        preprocessor: Any,
        postprocessor: Any,
    ) -> Path:
        """Write a directly loadable LeRobot ``pretrained_model`` directory.

        The caller owns processor construction/loading because normalization
        statistics belong to the base dataset checkpoint, not the PPO core.
        Both processors are mandatory: omitting either would produce a model
        directory that appears deployable but changes the action contract.
        """

        resolved = Path(output_dir).expanduser()
        if resolved.exists() and any(resolved.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty LeRobot policy directory: {resolved}"
            )
        for name, processor in (
            ("preprocessor", preprocessor),
            ("postprocessor", postprocessor),
        ):
            if not callable(getattr(processor, "save_pretrained", None)):
                raise TypeError(f"Flow-SDE {name} must implement save_pretrained(directory)")
        save_policy = getattr(self.policy, "save_pretrained", None)
        if not callable(save_policy):
            raise TypeError("Flow-SDE base policy does not implement LeRobot save_pretrained")

        resolved.mkdir(parents=True, exist_ok=True)
        save_policy(resolved)
        preprocessor.save_pretrained(resolved)
        postprocessor.save_pretrained(resolved)
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        missing = [name for name in required if not (resolved / name).is_file()]
        if missing:
            raise RuntimeError(
                "LeRobot policy export is incomplete; missing " + ", ".join(missing)
            )
        metadata = {
            "format": self.EXPORT_FORMAT,
            "source_update_step": self.update_step,
            "algorithm": "flow_sde_ppo",
            "ppo_config": asdict(self.config),
            "value_initialization_provenance": self.value_initialization_provenance,
        }
        (resolved / "flow_sde_ppo_export.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return resolved


def collect_one_episode_and_update(
    trainer: FlowSDEPPOTrainer,
    source: FlowSDEEpisodeSource,
) -> tuple[FlowSDEEpisode, FlowSDEPPOUpdateMetrics]:
    """Minimal real entrypoint used by a live or test rollout source."""

    if not isinstance(trainer, FlowSDEPPOTrainer):
        raise TypeError("Flow-SDE one-episode update requires FlowSDEPPOTrainer")
    if not isinstance(source, FlowSDEEpisodeSource):
        raise TypeError("Flow-SDE source must implement collect_episode(trainer)")
    episode = source.collect_episode(trainer)
    if not isinstance(episode, FlowSDEEpisode):
        raise TypeError("Flow-SDE source returned an invalid episode")
    return episode, trainer.update([episode])
