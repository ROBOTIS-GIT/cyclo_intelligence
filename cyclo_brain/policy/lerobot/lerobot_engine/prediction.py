#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""LeRobot prediction helpers."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
STATE_KEY = "observation.state"


logger = logging.getLogger("lerobot_engine")


class PredictionMixin:
    """Policy input batch -> action chunk."""

    def _predict_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk tensor of shape (1, T, A)."""
        assert self._policy is not None
        config = self._policy.config
        if getattr(config, "type", "") == "multi_task_dit":
            return self._predict_multi_task_dit_chunk(batch)
        if (getattr(config, "type", "") == "act"
                and getattr(config, "temporal_ensemble_coeff", None) is not None):
            # The official policy updates its temporal ensemble on each fresh
            # observation. Main executes this one action without chunk blending.
            horizon = getattr(self, "_runtime_chunk_size", None)
            if horizon is not None and horizon < config.chunk_size:
                actions = self._policy.predict_action_chunk(batch)[:, :horizon]
                return self._policy.temporal_ensembler.update(actions).unsqueeze(1)
            return self._policy.select_action(batch).unsqueeze(1)
        try:
            action = self._policy.predict_action_chunk(batch)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            if getattr(self._policy.config, "type", "") == "act":
                # Upstream ACT.select_action executes only n_action_steps from
                # each prediction before obtaining a fresh observation.
                steps = int(getattr(self, "_runtime_chunk_size", None) or self._policy.config.n_action_steps)
                if steps < 1 or steps > action.shape[1]:
                    raise ValueError(f"Invalid ACT n_action_steps={steps} for chunk {action.shape}")
                action = action[:, :steps]
            return action
        except (NotImplementedError, AttributeError):
            logger.debug(
                "predict_action_chunk unavailable; falling back to select_action"
            )
            action = self._policy.select_action(batch)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            return action.unsqueeze(1)

    def _predict_multi_task_dit_chunk(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Generate a MultiTaskDiT chunk from one current observation.

        The generic Cyclo engine supplies state as ``(B, S)`` and each camera
        as an individual ``(B, C, H, W)`` tensor. MultiTaskDiT expects an
        observation-history axis and one stacked camera tensor. Its public
        ``predict_action_chunk`` currently tries to stack its internal queues
        before they have been populated, which raises on an empty queue during
        chunk-at-a-time inference. Build the one-step history explicitly and
        use the policy's queue-free generation path instead.
        """

        config = self._policy.config
        n_obs_steps = getattr(config, "n_obs_steps", None)
        if n_obs_steps != 1:
            raise ValueError(
                "Cyclo MultiTaskDiT inference currently requires n_obs_steps=1"
            )

        state = batch.get(STATE_KEY)
        if not isinstance(state, torch.Tensor) or state.dim() != 2:
            raise ValueError(
                "MultiTaskDiT engine state must have shape (B, S)"
            )
        model_batch = dict(batch)
        model_batch[STATE_KEY] = state.unsqueeze(1)
        batch_size = state.shape[0]

        image_features = getattr(config, "image_features", None)
        if not image_features:
            raise ValueError("MultiTaskDiT checkpoint has no configured cameras")
        for key in image_features:
            image = batch.get(key)
            if (
                not isinstance(image, torch.Tensor)
                or image.dim() != 4
                or image.shape[0] != batch_size
            ):
                raise ValueError(
                    f"MultiTaskDiT camera {key!r} must have shape (B, C, H, W)"
                )
            model_batch[key] = image.unsqueeze(1)

        prepare_batch = getattr(self._policy, "_prepare_batch", None)
        generate_actions = getattr(self._policy, "_generate_actions", None)
        if not callable(prepare_batch) or not callable(generate_actions):
            raise RuntimeError(
                "Loaded multi_task_dit policy does not expose direct generation"
            )
        prepared = prepare_batch(model_batch)
        return generate_actions(prepared)

    @staticmethod
    def _to_numpy_chunk(action: torch.Tensor) -> np.ndarray:
        """(B, T, A) or (B, A) tensor -> (T, A) float64 numpy."""
        chunk = action.detach().cpu()
        if chunk.dim() == 3:
            chunk = chunk[0]
        elif chunk.dim() == 2:
            pass
        elif chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        else:
            raise ValueError(
                f"Unexpected action tensor shape: {tuple(chunk.shape)}"
            )
        return chunk.to(torch.float64).numpy()
