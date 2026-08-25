"""Thin Flow-SDE adapter around LeRobot's upstream MultiTaskDiT policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


OBSERVATION_STATE = "observation.state"
LANGUAGE_TOKENS = "observation.language.tokens"
LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
DEFAULT_TASK_INSTRUCTION = "ACT_dataset"

# This is the canonical Cyclo SG2 order. The checkpoint config must preserve
# the same order because MultiTaskDiT concatenates camera features.
CYCLO_SG2_CAMERA_KEYS = (
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_left_head",
    "observation.images.rgb.cam_right_wrist",
)


def with_default_task_instruction(
    batch: Mapping[str, Any],
    *,
    default_instruction: str = DEFAULT_TASK_INSTRUCTION,
) -> dict[str, Any]:
    """Fill an absent/blank raw LeRobot ``task`` field before preprocessing."""

    if not isinstance(batch, Mapping):
        raise TypeError("MultiTaskDiT batch must be a mapping")
    if not isinstance(default_instruction, str) or not default_instruction.strip():
        raise ValueError("MultiTaskDiT default instruction must be non-empty")
    result = dict(batch)
    state = result.get(OBSERVATION_STATE)
    if not isinstance(state, Tensor) or state.ndim < 2:
        raise ValueError("MultiTaskDiT raw batch requires observation.state with a batch dimension")
    batch_size = state.shape[0]
    task = result.get("task")
    if task is None:
        result["task"] = [default_instruction] * batch_size
        return result
    if isinstance(task, str):
        result["task"] = [task.strip() or default_instruction] * batch_size
        return result
    if not isinstance(task, Sequence) or len(task) != batch_size:
        raise ValueError("MultiTaskDiT task must be a string or one string per batch element")
    result["task"] = [
        value.strip() if isinstance(value, str) and value.strip() else default_instruction
        for value in task
    ]
    return result


class MultiTaskDiTFlowAdapter:
    """Expose a GR00T-compatible velocity interface without patching LeRobot.

    The adapter expects a *preprocessed* observation batch when encoding the
    condition. Raw task strings must first pass through the checkpoint's
    LeRobot preprocessor so CLIP token IDs and attention masks are present.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        expected_camera_keys: Sequence[str] | None = CYCLO_SG2_CAMERA_KEYS,
        freeze_observation_encoder: bool = True,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise TypeError("MultiTaskDiT policy must be a torch module")
        for attribute in ("config", "observation_encoder", "noise_predictor", "_prepare_batch"):
            if not hasattr(policy, attribute):
                raise ValueError(f"MultiTaskDiT policy is missing {attribute!r}")

        config = policy.config
        if getattr(config, "objective", None) != "flow_matching":
            raise ValueError("Flow-SDE requires MultiTaskDiT objective='flow_matching'")
        if float(getattr(config, "sigma_min", float("nan"))) != 0.0:
            raise ValueError("Flow-SDE currently requires MultiTaskDiT sigma_min=0.0")
        image_features = getattr(config, "image_features", None)
        if not isinstance(image_features, Mapping):
            raise ValueError("MultiTaskDiT config must expose image_features")
        camera_keys = tuple(image_features.keys())
        if len(camera_keys) != 3:
            raise ValueError("Cyclo MultiTaskDiT Flow-SDE requires exactly three cameras")
        if expected_camera_keys is not None and camera_keys != tuple(expected_camera_keys):
            raise ValueError(
                "MultiTaskDiT camera order mismatch: "
                f"expected {tuple(expected_camera_keys)!r}, got {camera_keys!r}"
            )
        state_feature = getattr(config, "robot_state_feature", None)
        action_feature = getattr(config, "action_feature", None)
        if state_feature is None or action_feature is None:
            raise ValueError("MultiTaskDiT config requires state and action features")
        if not isinstance(getattr(config, "horizon", None), int) or config.horizon < 1:
            raise ValueError("MultiTaskDiT config requires a positive action horizon")

        self.policy = policy
        self.camera_keys = camera_keys
        self.horizon = config.horizon
        self.n_obs_steps = config.n_obs_steps
        self.n_action_steps = config.n_action_steps
        self.action_dim = action_feature.shape[0]
        self.conditioning_dim = policy.observation_encoder.conditioning_dim
        self.freeze_observation_encoder = bool(freeze_observation_encoder)

        if self.freeze_observation_encoder:
            self.policy.observation_encoder.requires_grad_(False)
            self.policy.observation_encoder.eval()
        # Eval mode disables dropout but does not disable gradients. PPO needs
        # likelihood recomputation to be deterministic under unchanged weights.
        self.policy.noise_predictor.eval()

    def encode_conditioning(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Encode three cameras, state, and tokenized task instruction."""

        if not isinstance(batch, Mapping):
            raise TypeError("MultiTaskDiT preprocessed batch must be a mapping")
        required = (OBSERVATION_STATE, LANGUAGE_TOKENS, LANGUAGE_ATTENTION_MASK, *self.camera_keys)
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"MultiTaskDiT preprocessed batch is missing {missing!r}")
        state = batch[OBSERVATION_STATE]
        if (
            not isinstance(state, Tensor)
            or state.ndim != 3
            or state.shape[1] != self.n_obs_steps
        ):
            raise ValueError("MultiTaskDiT state must have shape (B, n_obs_steps, S)")
        batch_size = state.shape[0]
        for key in self.camera_keys:
            image = batch[key]
            if (
                not isinstance(image, Tensor)
                or image.ndim != 5
                or image.shape[0] != batch_size
                or image.shape[1] != self.n_obs_steps
            ):
                raise ValueError(f"MultiTaskDiT camera {key!r} has an invalid batch shape")

        prepared = self.policy._prepare_batch(dict(batch))
        if self.freeze_observation_encoder:
            with torch.no_grad():
                conditioning = self.policy.observation_encoder.encode(prepared)
        else:
            conditioning = self.policy.observation_encoder.encode(prepared)
        if conditioning.shape != (batch_size, self.conditioning_dim):
            raise RuntimeError(
                "MultiTaskDiT conditioning shape mismatch; verify task tokens and observation history"
            )
        if not bool(torch.isfinite(conditioning).all()):
            raise RuntimeError("MultiTaskDiT conditioning contains non-finite values")
        return conditioning.detach() if self.freeze_observation_encoder else conditioning

    def velocity(self, latent: Tensor, progress: Tensor, conditioning: Tensor) -> Tensor:
        """Predict noise-to-action velocity for Flow-SDE sampling or PPO."""

        if (
            not isinstance(latent, Tensor)
            or latent.shape[1:] != (self.horizon, self.action_dim)
            or not latent.is_floating_point()
        ):
            raise ValueError("MultiTaskDiT latent must have shape (B, horizon, action_dim)")
        batch_size = latent.shape[0]
        if (
            not isinstance(progress, Tensor)
            or progress.shape != (batch_size,)
            or not progress.is_floating_point()
            or progress.device != latent.device
            or bool((progress < 0.0).any())
            or bool((progress >= 1.0).any())
        ):
            raise ValueError("MultiTaskDiT progress must have shape (B,) in [0, 1)")
        if (
            not isinstance(conditioning, Tensor)
            or conditioning.shape != (batch_size, self.conditioning_dim)
            or not conditioning.is_floating_point()
            or conditioning.device != latent.device
        ):
            raise ValueError("MultiTaskDiT conditioning does not match the latent batch")

        parameter = next(self.policy.noise_predictor.parameters())
        if parameter.device != latent.device:
            raise ValueError("MultiTaskDiT noise predictor and latent must share one device")
        self.policy.noise_predictor.eval()
        velocity = self.policy.noise_predictor(
            latent.to(parameter.dtype),
            progress.to(parameter.dtype),
            conditioning.to(parameter.dtype),
        )
        if velocity.shape != latent.shape:
            raise RuntimeError("MultiTaskDiT velocity shape does not match its latent input")
        return velocity.float()

    def executed_action_mask(self, batch_size: int, *, device: torch.device | str) -> Tensor:
        """Mask the exact horizon slice emitted by ``MultiTaskDiTPolicy``."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("MultiTaskDiT action-mask batch_size must be positive")
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        if end > self.horizon:
            raise RuntimeError("MultiTaskDiT executed action slice exceeds its horizon")
        mask = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            dtype=torch.bool,
            device=device,
        )
        mask[:, start:end] = True
        return mask

    def executed_actions(self, normalized_chunk: Tensor) -> Tensor:
        """Return the exact action slice that the upstream policy would emit."""

        if (
            not isinstance(normalized_chunk, Tensor)
            or normalized_chunk.ndim != 3
            or normalized_chunk.shape[1:] != (self.horizon, self.action_dim)
        ):
            raise ValueError("MultiTaskDiT chunk must have shape (B, horizon, action_dim)")
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return normalized_chunk[:, start:end]

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return action-head parameters while keeping observation encoders frozen."""

        return tuple(
            parameter
            for parameter in self.policy.noise_predictor.parameters()
            if parameter.requires_grad
        )
