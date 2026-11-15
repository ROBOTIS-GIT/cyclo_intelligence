"""Tensor contracts stored between Flow-SDE rollout and PPO update."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowSDERollout:
    """One sampled denoising chain for a batch of action chunks.

    ``chains`` has shape ``(B, N + 1, H, A)``. Exactly one transition index
    per batch element is stochastic and contributes ``old_log_probs``. The
    mask identifies action coordinates that were actually executed and are
    therefore eligible for PPO credit.
    """

    chains: Tensor
    denoise_indices: Tensor
    old_log_probs: Tensor
    action_mask: Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, Tensor)
            for value in (
                self.chains,
                self.denoise_indices,
                self.old_log_probs,
                self.action_mask,
            )
        ):
            raise TypeError("Flow-SDE rollout values must be torch tensors")
        if self.chains.ndim != 4 or any(size < 1 for size in self.chains.shape):
            raise ValueError("Flow-SDE chains must have non-empty shape (B, N+1, H, A)")

        batch_size, chain_length, horizon, action_dim = self.chains.shape
        if chain_length < 3:
            raise ValueError("Flow-SDE chains must contain at least two transitions")
        if self.denoise_indices.shape != (batch_size,) or self.denoise_indices.dtype != torch.long:
            raise ValueError("Flow-SDE denoise_indices must be int64 with shape (B,)")
        expected_action_shape = (batch_size, horizon, action_dim)
        if self.old_log_probs.shape != expected_action_shape:
            raise ValueError("Flow-SDE old_log_probs must have shape (B, H, A)")
        if self.action_mask.shape != expected_action_shape or self.action_mask.dtype != torch.bool:
            raise ValueError("Flow-SDE action_mask must be boolean with shape (B, H, A)")
        if any(
            value.device != self.chains.device
            for value in (self.denoise_indices, self.old_log_probs, self.action_mask)
        ):
            raise ValueError("Flow-SDE rollout tensors must share one device")
        if not self.chains.is_floating_point() or self.old_log_probs.dtype != torch.float32:
            raise ValueError("Flow-SDE chains must be floating and log-probabilities must be float32")
        if not bool(torch.isfinite(self.chains).all()) or not bool(
            torch.isfinite(self.old_log_probs).all()
        ):
            raise ValueError("Flow-SDE rollout tensors must be finite")
        if bool((self.denoise_indices < 0).any()) or bool(
            (self.denoise_indices >= chain_length - 1).any()
        ):
            raise ValueError("Flow-SDE denoise indices are outside the stored chain")
        if not bool(self.action_mask.reshape(batch_size, -1).any(dim=1).all()):
            raise ValueError("Every Flow-SDE rollout sample needs at least one valid action coordinate")

    @property
    def actions(self) -> Tensor:
        """Return the final normalized action chunk ``(B, H, A)``."""

        return self.chains[:, -1]

    def selected_transition(self) -> tuple[Tensor, Tensor]:
        """Return the cached ``(x_t, x_next)`` pair chosen for PPO."""

        batch_indices = torch.arange(self.chains.shape[0], device=self.chains.device)
        current = self.chains[batch_indices, self.denoise_indices]
        following = self.chains[batch_indices, self.denoise_indices + 1]
        return current, following
