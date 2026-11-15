"""Model-independent Flow-SDE action-chunk sampler."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .batch import FlowSDERollout
from .config import FlowSDEPPOConfig
from .functional import flow_sde_transition_stats, gaussian_log_prob

VelocityFunction = Callable[[Tensor, Tensor, Tensor], Tensor]


@torch.no_grad()
def sample_flow_sde_chunk(
    velocity_function: VelocityFunction,
    conditioning: Tensor,
    *,
    horizon: int,
    action_dim: int,
    config: FlowSDEPPOConfig,
    action_mask: Tensor | None = None,
    initial_noise: Tensor | None = None,
    denoise_indices: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowSDERollout:
    """Sample a chunk while making one denoising transition stochastic."""

    if not callable(velocity_function):
        raise TypeError("Flow-SDE velocity_function must be callable")
    if not isinstance(conditioning, Tensor) or conditioning.ndim != 2 or not conditioning.is_floating_point():
        raise ValueError("Flow-SDE conditioning must be floating with shape (B, C)")
    if not bool(torch.isfinite(conditioning).all()):
        raise ValueError("Flow-SDE conditioning must be finite")
    for name, value in (("horizon", horizon), ("action_dim", action_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Flow-SDE {name} must be a positive integer")
    if not isinstance(config, FlowSDEPPOConfig):
        raise TypeError("Flow-SDE sampler requires FlowSDEPPOConfig")

    batch_size = conditioning.shape[0]
    shape = (batch_size, horizon, action_dim)
    if initial_noise is None:
        current = torch.randn(shape, device=conditioning.device, dtype=torch.float32, generator=generator)
    else:
        if (
            not isinstance(initial_noise, Tensor)
            or initial_noise.shape != shape
            or not initial_noise.is_floating_point()
            or initial_noise.device != conditioning.device
            or not bool(torch.isfinite(initial_noise).all())
        ):
            raise ValueError("Flow-SDE initial_noise must match (B, H, A) and conditioning device")
        current = initial_noise.detach().float().clone()

    if denoise_indices is None:
        resolved_indices = torch.randint(
            0,
            config.num_denoising_steps,
            (batch_size,),
            device=conditioning.device,
            generator=generator,
        )
    else:
        if (
            not isinstance(denoise_indices, Tensor)
            or denoise_indices.shape != (batch_size,)
            or denoise_indices.dtype != torch.long
            or denoise_indices.device != conditioning.device
            or bool((denoise_indices < 0).any())
            or bool((denoise_indices >= config.num_denoising_steps).any())
        ):
            raise ValueError("Flow-SDE denoise_indices must be valid int64 indices with shape (B,)")
        resolved_indices = denoise_indices.detach().clone()

    if action_mask is None:
        resolved_action_mask = torch.ones(shape, dtype=torch.bool, device=conditioning.device)
    else:
        if (
            not isinstance(action_mask, Tensor)
            or action_mask.shape != shape
            or action_mask.dtype != torch.bool
            or action_mask.device != conditioning.device
        ):
            raise ValueError("Flow-SDE action_mask must be boolean with shape (B, H, A)")
        resolved_action_mask = action_mask.detach().clone()

    chains = [current]
    selected_log_probs = torch.empty(shape, device=conditioning.device, dtype=torch.float32)
    selected_written = torch.zeros(batch_size, device=conditioning.device, dtype=torch.bool)
    for index in range(config.num_denoising_steps):
        progress = torch.full(
            (batch_size,),
            index / config.num_denoising_steps,
            device=conditioning.device,
            dtype=torch.float32,
        )
        velocity = velocity_function(current, progress, conditioning)
        stochastic_mask = resolved_indices == index
        step_indices = torch.full(
            (batch_size,), index, device=conditioning.device, dtype=torch.long
        )
        mean, std = flow_sde_transition_stats(
            current,
            velocity,
            step_indices,
            num_steps=config.num_denoising_steps,
            noise_level=config.noise_level,
            stochastic_mask=stochastic_mask,
        )
        step_noise = torch.randn(shape, device=current.device, dtype=torch.float32, generator=generator)
        following = mean + std * step_noise
        if bool(stochastic_mask.any()):
            selected_log_probs[stochastic_mask] = gaussian_log_prob(
                following[stochastic_mask],
                mean[stochastic_mask],
                std[stochastic_mask],
            )
            selected_written[stochastic_mask] = True
        current = following
        chains.append(current)

    if not bool(selected_written.all()):
        raise RuntimeError("Flow-SDE sampler did not record every selected stochastic transition")
    return FlowSDERollout(
        chains=torch.stack(chains, dim=1).detach(),
        denoise_indices=resolved_indices,
        old_log_probs=selected_log_probs.detach(),
        action_mask=resolved_action_mask,
    )


def recompute_flow_sde_log_probs(
    velocity_function: VelocityFunction,
    conditioning: Tensor,
    rollout: FlowSDERollout,
    *,
    config: FlowSDEPPOConfig,
) -> Tensor:
    """Re-evaluate the cached stochastic transition under current weights."""

    if not callable(velocity_function):
        raise TypeError("Flow-SDE velocity_function must be callable")
    if not isinstance(rollout, FlowSDERollout):
        raise TypeError("Flow-SDE recomputation requires FlowSDERollout")
    if not isinstance(config, FlowSDEPPOConfig):
        raise TypeError("Flow-SDE recomputation requires FlowSDEPPOConfig")
    if rollout.chains.shape[1] != config.num_denoising_steps + 1:
        raise ValueError("Flow-SDE rollout chain length does not match config")
    if (
        not isinstance(conditioning, Tensor)
        or conditioning.ndim != 2
        or conditioning.shape[0] != rollout.chains.shape[0]
        or not conditioning.is_floating_point()
        or conditioning.device != rollout.chains.device
        or not bool(torch.isfinite(conditioning).all())
    ):
        raise ValueError("Flow-SDE conditioning does not match the rollout batch")

    current, following = rollout.selected_transition()
    progress = rollout.denoise_indices.to(torch.float32) / config.num_denoising_steps
    velocity = velocity_function(current, progress, conditioning)
    mean, std = flow_sde_transition_stats(
        current,
        velocity,
        rollout.denoise_indices,
        num_steps=config.num_denoising_steps,
        noise_level=config.noise_level,
    )
    return gaussian_log_prob(following, mean, std)
