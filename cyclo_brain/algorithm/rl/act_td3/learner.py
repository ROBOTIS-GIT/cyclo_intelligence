"""Executed-prefix SMDP learner for official LeRobot ACT policies."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cyclo_brain.algorithm.rl.td3.functional import critic_loss, polyak_update_
from cyclo_brain.model.act import (
    ACT_TRAINABLE_GROUPS,
    ACTTwinChunkCritic,
    apply_act_trainable_groups,
    compute_act_bc_loss,
    differentiable_act_action_chunk,
)

from .batch import ACTTD3Batch
from .config import ACTTD3Config
from .functional import (
    actor_update_is_due,
    build_smdp_returns,
    masked_deterministic_bc_l1,
    q_weight_for_actor_update,
    smooth_target_action_chunks,
)


@dataclass(frozen=True)
class ACTTD3UpdateResult:
    critic_loss: float
    target_mean: float
    actor_updated: bool
    actor_loss: float | None
    cvae_bc_loss: float | None
    deterministic_bc_loss: float | None
    actor_q_loss: float | None
    actor_q_weight: float | None
    actor_q_full_row_count: int | None
    completed_critic_updates: int
    completed_actor_updates: int
    target_critic_updated: bool


def _module_device_and_dtype(module: nn.Module, name: str) -> tuple[torch.device, torch.dtype]:
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError(f"ACT-TD3 {name} must have parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError(f"ACT-TD3 {name} parameters must share dtype and device")
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError(f"ACT-TD3 {name} parameters must be floating point")
    return next(iter(devices)), dtype


@contextmanager
def _temporary_mode(module: nn.Module, *, training: bool):
    previous = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(previous)


@contextmanager
def _temporarily_freeze_parameters(module: nn.Module):
    flags = tuple((parameter, parameter.requires_grad) for parameter in module.parameters())
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)


@contextmanager
def _use_owned_torch_rng(generator: torch.Generator, device: torch.device):
    """Run stochastic ACT BC while advancing only learner-owned RNG state."""

    if generator.device != device:
        raise ValueError("ACT-TD3 actor RNG must share the actor device")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        fork_devices = [index]
    else:
        fork_devices = []
    with torch.random.fork_rng(devices=fork_devices):
        if device.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), device=device)
        else:
            torch.random.set_rng_state(generator.get_state())
        try:
            yield
        finally:
            if device.type == "cuda":
                state = torch.cuda.get_rng_state(device=device)
            else:
                state = torch.random.get_rng_state()
            generator.set_state(state)


def _gradient_clip(module: nn.Module, maximum: float | None, name: str) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"ACT-TD3 {name} objective produced no gradients")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError(f"ACT-TD3 {name} gradients contain NaN or Inf")
    if maximum is not None:
        nn.utils.clip_grad_norm_(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            max_norm=float(maximum),
            error_if_nonfinite=True,
        )


def _index_observations(
    observations: Mapping[str, Tensor],
    indices: Tensor,
) -> dict[str, Tensor]:
    return {name: value.index_select(0, indices) for name, value in observations.items()}


def _cpu_clone_tree(value: Any) -> Any:
    """Return an alias-free, portable learner snapshot."""

    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "ACT-TD3 state contains unsupported value "
        f"{type(value).__name__}"
    )


class ACTTD3Learner:
    """ACT+BC TD3 over the policy's actually executed action prefix.

    ACT predicts ``chunk_size`` actions but the official deployed policy
    executes only the first ``n_action_steps`` before replanning. Replay,
    SMDP returns, target smoothing, and Q-functions therefore use the latter
    execution horizon. Official CVAE BC still receives its configured full
    prediction horizon, with the unexecuted suffix marked as padding.

    Replay actions, deterministic actor actions, and target actor actions all
    remain in the normalized action coordinates produced by the immutable ACT
    checkpoint preprocessor. TD3 target noise and the actor's direct Q gradient
    apply to every action dimension. There is deliberately no additional action
    projection or clamp inside the learner; the saved ACT postprocessor alone
    maps deployed actions back to the robot's raw command coordinates.
    """

    STATE_FORMAT = "cyclo_brain.act_td3_learner/v4"
    LEGACY_ALL_TRAINABLE_STATE_FORMAT = "cyclo_brain.act_td3_learner/v3"
    BC_SUPPORT = "executed_prefix_zero_padded_to_prediction_horizon"
    ACTION_DOMAIN = "saved_act_preprocessor_mean_std_normalized"
    TARGET_POLICY_SMOOTHING = "clipped_noise_all_dimensions_no_action_clamp"
    ACTOR_Q_GRADIENT = "all_action_dimensions"

    def __init__(
        self,
        actor: nn.Module,
        critic: ACTTwinChunkCritic,
        config: ACTTD3Config | None = None,
        *,
        random_seed: int = 0,
    ) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs import NormalizationMode

        if not isinstance(actor, ACTPolicy):
            raise TypeError("ACT-TD3 actor must be the official LeRobot ACTPolicy")
        if not isinstance(critic, ACTTwinChunkCritic):
            raise TypeError("ACT-TD3 critic must be ACTTwinChunkCritic")
        self.config = config or ACTTD3Config()
        if not isinstance(self.config, ACTTD3Config):
            raise TypeError("ACT-TD3 config must be ACTTD3Config")
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, int)
            or not 0 <= random_seed < 2**63 - 2
        ):
            raise ValueError("ACT-TD3 random seed is invalid")
        actor_config = actor.config
        if not actor_config.use_vae:
            raise ValueError("ACT-TD3 BC anchor requires ACT use_vae=True")
        if actor_config.temporal_ensemble_coeff is not None:
            raise ValueError("ACT-TD3 does not support temporal ensembling")
        if actor_config.n_obs_steps != 1:
            raise ValueError("ACT-TD3 requires one observation step")
        if actor_config.normalization_mapping.get("ACTION") != NormalizationMode.MEAN_STD:
            raise ValueError("ACT-TD3 requires saved MEAN_STD action normalization")
        action_feature = actor_config.action_feature
        if action_feature is None or len(action_feature.shape) != 1:
            raise ValueError("ACT-TD3 actor must expose one vector action feature")
        self.prediction_horizon = int(actor_config.chunk_size)
        self.execution_horizon = int(actor_config.n_action_steps)
        if not 1 <= self.execution_horizon <= self.prediction_horizon:
            raise ValueError("ACT-TD3 actor execution horizon is invalid")
        self.action_dim = int(action_feature.shape[0])
        expected_observation_keys = tuple(actor_config.input_features or {})
        if (
            critic.prediction_horizon != self.prediction_horizon
            or critic.execution_horizon != self.execution_horizon
            or critic.action_dim != self.action_dim
            or set(critic.observation_keys) != set(expected_observation_keys)
        ):
            raise ValueError("ACT-TD3 actor and critic tensor contracts disagree")
        if not all(
            bool(q_function.observation_encoder.visual_initialization_complete)
            for q_function in (critic.q1, critic.q2)
        ):
            raise ValueError(
                "ACT-TD3 visual critics must be initialized from the ACT backbone"
            )

        actor_device, actor_dtype = _module_device_and_dtype(actor, "actor")
        critic_device, critic_dtype = _module_device_and_dtype(critic, "critic")
        if actor_device != critic_device or actor_dtype != critic_dtype:
            raise ValueError("ACT-TD3 actor and critic must share dtype and device")
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        critic_ids = {id(parameter) for parameter in critic.parameters()}
        if actor_ids & critic_ids:
            raise ValueError("ACT-TD3 actor and critic parameters must be independent")
        if {id(parameter) for parameter in critic.q1.parameters()} & {
            id(parameter) for parameter in critic.q2.parameters()
        }:
            raise ValueError("ACT-TD3 Q1 and Q2 parameters must be independent")

        self.actor = actor.eval()
        apply_act_trainable_groups(
            self.actor,
            self.config.actor_trainable_groups,
        )
        self.critic = critic.train().requires_grad_(True)
        self.actor_target = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)
        trainable_actor_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        ]
        self.actor_optimizer = torch.optim.AdamW(
            trainable_actor_parameters,
            lr=self.config.actor_learning_rate,
            weight_decay=self.config.actor_weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_learning_rate,
        )
        self.random_seed = random_seed
        self.target_noise_generator = torch.Generator(device=actor_device).manual_seed(
            random_seed
        )
        self.actor_generator = torch.Generator(device=actor_device).manual_seed(
            random_seed + 1
        )
        self.completed_critic_updates = 0
        self.completed_actor_updates = 0
        self.device = actor_device
        self.dtype = actor_dtype
        self._target_noise_mask = torch.ones(
            self.action_dim,
            dtype=torch.bool,
            device=self.device,
        )

    def _state_contract(self) -> dict[str, Any]:
        """Return tensor semantics that model shapes alone cannot recover."""

        return {
            "prediction_horizon": self.prediction_horizon,
            "execution_horizon": self.execution_horizon,
            "action_dim": self.action_dim,
            "observation_keys": tuple(self.critic.observation_keys),
            "bc_support": self.BC_SUPPORT,
            "action_domain": self.ACTION_DOMAIN,
            "target_policy_smoothing": self.TARGET_POLICY_SMOOTHING,
            "actor_q_gradient": self.ACTOR_Q_GRADIENT,
            "action_clamp": False,
            "actor_trainable_groups": self.config.actor_trainable_groups,
        }

    def _validate_batch(self, batch: ACTTD3Batch) -> None:
        if not isinstance(batch, ACTTD3Batch):
            raise TypeError("ACT-TD3 update requires ACTTD3Batch")
        if (
            batch.execution_horizon != self.execution_horizon
            or batch.action_dim != self.action_dim
        ):
            raise ValueError("ACT-TD3 batch action contract disagrees with ACT")
        if set(batch.observations) != set(self.critic.observation_keys):
            raise ValueError("ACT-TD3 batch observation keys disagree with ACT")
        if (
            batch.behavior_action_chunks.device != self.device
            or batch.behavior_action_chunks.dtype != self.dtype
        ):
            raise ValueError("ACT-TD3 batch must share actor dtype and device")

    @torch.no_grad()
    def compute_bellman_targets(
        self,
        batch: ACTTD3Batch,
        *,
        target_standard_normal_noise: Tensor | None = None,
    ) -> Tensor:
        """Compute duration-aware clipped-double-Q targets for bootstrap rows."""

        self._validate_batch(batch)
        smdp = build_smdp_returns(
            batch.rewards,
            batch.executed_mask,
            batch.step_durations_s,
            batch.bootstrap_allowed,
            discount=self.config.discount,
            discount_reference_hz=self.config.discount_reference_hz,
        )
        targets = smdp.discounted_returns.clone()
        indices = torch.nonzero(batch.bootstrap_allowed, as_tuple=False).flatten()
        if indices.numel() == 0:
            if target_standard_normal_noise is not None:
                raise ValueError("ACT-TD3 target noise was supplied without bootstrap rows")
            return targets
        next_observations = _index_observations(batch.next_observations, indices)
        target_policy_actions = differentiable_act_action_chunk(
            self.actor_target,
            next_observations,
        )
        prediction_shape = (
            int(indices.numel()),
            self.prediction_horizon,
            self.action_dim,
        )
        if target_policy_actions.shape != prediction_shape:
            raise RuntimeError("ACT-TD3 target actor returned an invalid chunk shape")
        target_execution_actions = target_policy_actions[
            :, : self.execution_horizon
        ]
        execution_shape = (
            int(indices.numel()),
            self.execution_horizon,
            self.action_dim,
        )
        if target_standard_normal_noise is None:
            noise = torch.randn(
                execution_shape,
                dtype=self.dtype,
                device=self.device,
                generator=self.target_noise_generator,
            )
        else:
            noise = target_standard_normal_noise
        smoothed_policy_actions = smooth_target_action_chunks(
            target_execution_actions,
            noise,
            self._target_noise_mask,
            noise_standard_deviation=self.config.target_policy_noise,
            noise_clip=self.config.target_policy_noise_clip,
        )
        full_mask = torch.ones(
            (int(indices.numel()), self.execution_horizon),
            dtype=torch.bool,
            device=self.device,
        )
        target_q1, target_q2 = self.critic_target(
            next_observations,
            smoothed_policy_actions,
            full_mask,
        )
        if target_q1.shape != (int(indices.numel()), 1) or target_q2.shape != target_q1.shape:
            raise RuntimeError("ACT-TD3 target critic returned invalid Q shapes")
        bootstrap = smdp.bootstrap_discounts.index_select(0, indices)
        targets[indices] += bootstrap * torch.minimum(target_q1, target_q2)
        if not bool(torch.isfinite(targets).all()):
            raise RuntimeError("ACT-TD3 Bellman targets contain NaN or Inf")
        return targets

    def _actor_step(
        self,
        batch: ACTTD3Batch,
    ) -> tuple[Tensor, Tensor, Tensor, float, int]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        bc_batch = dict(batch.observations)
        from lerobot.utils.constants import ACTION

        bc_actions = batch.behavior_action_chunks.new_zeros(
            batch.batch_size,
            self.prediction_horizon,
            self.action_dim,
        )
        bc_actions[:, : self.execution_horizon] = batch.behavior_action_chunks
        action_is_pad = torch.ones(
            (batch.batch_size, self.prediction_horizon),
            dtype=torch.bool,
            device=self.device,
        )
        action_is_pad[:, : self.execution_horizon] = ~batch.executed_mask
        bc_batch[ACTION] = bc_actions
        bc_batch["action_is_pad"] = action_is_pad
        with _use_owned_torch_rng(self.actor_generator, self.device):
            with _temporary_mode(self.actor, training=True):
                cvae_bc_loss, _metrics = compute_act_bc_loss(self.actor, bc_batch)
        weighted_cvae_bc_loss = self.config.cvae_bc_weight * cvae_bc_loss
        weighted_cvae_bc_loss.backward()
        cvae_bc_loss_value = cvae_bc_loss.detach()
        del weighted_cvae_bc_loss, cvae_bc_loss, bc_batch

        # Release the stochastic CVAE graph before constructing the deployed
        # zero-latent graph. Gradients still accumulate into the same actor
        # parameters, but full visual ACT does not retain both large graphs.
        with _temporary_mode(self.actor, training=False):
            policy_actions = differentiable_act_action_chunk(
                self.actor,
                batch.observations,
            )
        policy_execution_actions = policy_actions[:, : self.execution_horizon]
        deterministic_bc_loss = masked_deterministic_bc_l1(
            policy_execution_actions,
            batch.behavior_action_chunks,
            batch.executed_mask,
        )
        full_indices = torch.nonzero(
            batch.lengths == self.execution_horizon,
            as_tuple=False,
        ).flatten()
        full_count = int(full_indices.numel())
        q_weight = q_weight_for_actor_update(
            self.completed_actor_updates + 1,
            maximum=self.config.q_weight_max,
            ramp_updates=self.config.q_weight_ramp_actor_updates,
        )
        self.critic.zero_grad(set_to_none=True)
        with _temporarily_freeze_parameters(self.critic):
            if full_count:
                full_observations = _index_observations(
                    batch.observations,
                    full_indices,
                )
                full_policy_actions = policy_execution_actions.index_select(
                    0,
                    full_indices,
                )
                full_mask = torch.ones(
                    (full_count, self.execution_horizon),
                    dtype=torch.bool,
                    device=self.device,
                )
                q1_for_actor = self.critic.q1(
                    full_observations,
                    full_policy_actions,
                    full_mask,
                )
                actor_q_loss = -q_weight * q1_for_actor.mean()
            else:
                actor_q_loss = policy_execution_actions.sum() * 0.0
            actor_q_and_bc_loss = (
                self.config.deterministic_bc_weight * deterministic_bc_loss
                + actor_q_loss
            )
            actor_q_and_bc_loss.backward()
        if any(parameter.grad is not None for parameter in self.critic.parameters()):
            raise RuntimeError("ACT-TD3 actor objective reached critic gradients")
        _gradient_clip(
            self.actor,
            self.config.actor_gradient_clip_norm,
            "actor",
        )
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor.eval()
        actor_loss = (
            self.config.cvae_bc_weight * cvae_bc_loss_value
            + self.config.deterministic_bc_weight * deterministic_bc_loss.detach()
            + actor_q_loss.detach()
        )
        return (
            actor_loss,
            cvae_bc_loss_value,
            deterministic_bc_loss.detach(),
            float(actor_q_loss.detach()),
            full_count,
        )

    def update(
        self,
        batch: ACTTD3Batch,
        *,
        target_standard_normal_noise: Tensor | None = None,
    ) -> ACTTD3UpdateResult:
        """Perform one critic step and an optional delayed ACT+BC step."""

        targets = self.compute_bellman_targets(
            batch,
            target_standard_normal_noise=target_standard_normal_noise,
        )
        q1, q2 = self.critic(
            batch.observations,
            batch.behavior_action_chunks,
            batch.executed_mask,
        )
        loss_critic = critic_loss(q1, q2, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_critic.backward()
        _gradient_clip(
            self.critic,
            self.config.critic_gradient_clip_norm,
            "critic",
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.completed_critic_updates += 1

        actor_due = actor_update_is_due(
            self.completed_critic_updates,
            critic_warmup_updates=self.config.critic_warmup_updates,
            policy_update_period=self.config.policy_update_period,
        )
        target_critic_updated = False
        actor_values: tuple[Tensor, Tensor, Tensor, float, int] | None = None
        q_weight: float | None = None
        if actor_due:
            q_weight = q_weight_for_actor_update(
                self.completed_actor_updates + 1,
                maximum=self.config.q_weight_max,
                ramp_updates=self.config.q_weight_ramp_actor_updates,
            )
            actor_values = self._actor_step(batch)
            self.completed_actor_updates += 1
            polyak_update_(
                self.actor,
                self.actor_target,
                tau=self.config.target_update_rate,
            )
            polyak_update_(
                self.critic,
                self.critic_target,
                tau=self.config.target_update_rate,
            )
            target_critic_updated = True
        elif (
            self.completed_critic_updates <= self.config.critic_warmup_updates
            and self.completed_critic_updates % self.config.policy_update_period == 0
        ):
            polyak_update_(
                self.critic,
                self.critic_target,
                tau=self.config.target_update_rate,
            )
            target_critic_updated = True

        if actor_values is None:
            actor_loss_value = None
            cvae_value = None
            deterministic_value = None
            q_loss_value = None
            full_count = None
        else:
            actor_loss, cvae_loss, deterministic_loss, q_loss, resolved_full_count = (
                actor_values
            )
            actor_loss_value = float(actor_loss)
            cvae_value = float(cvae_loss)
            deterministic_value = float(deterministic_loss)
            q_loss_value = q_loss
            full_count = resolved_full_count
        return ACTTD3UpdateResult(
            critic_loss=float(loss_critic.detach()),
            target_mean=float(targets.mean()),
            actor_updated=actor_due,
            actor_loss=actor_loss_value,
            cvae_bc_loss=cvae_value,
            deterministic_bc_loss=deterministic_value,
            actor_q_loss=q_loss_value,
            actor_q_weight=q_weight,
            actor_q_full_row_count=full_count,
            completed_critic_updates=self.completed_critic_updates,
            completed_actor_updates=self.completed_actor_updates,
            target_critic_updated=target_critic_updated,
        )

    def state_dict(self) -> dict[str, Any]:
        """Capture models, optimizers, counters, and learner-owned RNGs."""

        return _cpu_clone_tree({
            "format": self.STATE_FORMAT,
            "contract": self._state_contract(),
            "config": asdict(self.config),
            "random_seed": self.random_seed,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "completed_critic_updates": self.completed_critic_updates,
            "completed_actor_updates": self.completed_actor_updates,
            "target_noise_generator": self.target_noise_generator.get_state(),
            "actor_generator": self.actor_generator.get_state(),
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an exact update boundary without changing the action contract."""

        state = self._normalize_legacy_all_trainable_state(state)
        if state.get("format") != self.STATE_FORMAT:
            raise ValueError("ACT-TD3 learner state format is invalid")
        if state.get("contract") != self._state_contract():
            raise ValueError("ACT-TD3 learner state tensor contract disagrees")
        if state.get("config") != asdict(self.config):
            raise ValueError("ACT-TD3 learner state config disagrees")
        if state.get("random_seed") != self.random_seed:
            raise ValueError("ACT-TD3 learner random seed disagrees")
        critic_updates = state.get("completed_critic_updates")
        actor_updates = state.get("completed_actor_updates")
        for name, value in (
            ("completed_critic_updates", critic_updates),
            ("completed_actor_updates", actor_updates),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ACT-TD3 learner state {name} is invalid")
        expected_actor_updates = max(
            0,
            (critic_updates - self.config.critic_warmup_updates)
            // self.config.policy_update_period,
        )
        if actor_updates != expected_actor_updates:
            raise ValueError("ACT-TD3 learner update counters disagree")
        self.actor.load_state_dict(state["actor"], strict=True)
        self.actor_target.load_state_dict(state["actor_target"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.critic_target.load_state_dict(state["critic_target"], strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.target_noise_generator.set_state(state["target_noise_generator"])
        self.actor_generator.set_state(state["actor_generator"])
        self.completed_critic_updates = critic_updates
        self.completed_actor_updates = actor_updates
        self.actor.eval()
        apply_act_trainable_groups(
            self.actor,
            self.config.actor_trainable_groups,
        )
        self.critic.train().requires_grad_(True)
        self.actor_target.eval().requires_grad_(False)
        self.critic_target.eval().requires_grad_(False)

    def _normalize_legacy_all_trainable_state(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Migrate v3 only when it retains its historical all-trainable meaning."""

        if not isinstance(state, Mapping):
            raise ValueError("ACT-TD3 learner state format is invalid")
        if state.get("format") != self.LEGACY_ALL_TRAINABLE_STATE_FORMAT:
            return state
        if self.config.actor_trainable_groups != ACT_TRAINABLE_GROUPS:
            raise ValueError(
                "Legacy ACT-TD3 v3 checkpoints can resume only with all ACT "
                "actor trainable groups"
            )
        legacy_contract = state.get("contract")
        legacy_config = state.get("config")
        if (
            not isinstance(legacy_contract, Mapping)
            or not isinstance(legacy_config, Mapping)
            or "actor_trainable_groups" in legacy_contract
            or "actor_trainable_groups" in legacy_config
        ):
            raise ValueError("ACT-TD3 legacy learner state contract is invalid")

        normalized = dict(state)
        normalized_contract = dict(legacy_contract)
        normalized_contract["actor_trainable_groups"] = ACT_TRAINABLE_GROUPS
        normalized_config = dict(legacy_config)
        normalized_config["actor_trainable_groups"] = ACT_TRAINABLE_GROUPS
        normalized["format"] = self.STATE_FORMAT
        normalized["contract"] = normalized_contract
        normalized["config"] = normalized_config
        return normalized


__all__ = ["ACTTD3Learner", "ACTTD3UpdateResult"]
