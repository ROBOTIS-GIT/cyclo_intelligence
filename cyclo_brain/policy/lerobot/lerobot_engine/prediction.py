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

from .constants import STATE_KEY


logger = logging.getLogger("lerobot_engine")

_MULTI_TASK_DIT_POLICY_TYPE = "multi_task_dit"


class PredictionMixin:
    """Policy input batch -> action chunk."""

    def _predict_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk tensor of shape (1, T, A)."""
        assert self._policy is not None
        config = self._policy.config
        policy_type = getattr(config, "type", None)
        if policy_type == _MULTI_TASK_DIT_POLICY_TYPE:
            action = self._predict_multi_task_dit_chunk(batch)
        else:
            predict_action_chunk = getattr(self._policy, "predict_action_chunk", None)
            if predict_action_chunk is None:
                return self._select_action_as_chunk(batch)
            try:
                action = predict_action_chunk(batch)
            except NotImplementedError:
                logger.debug(
                    "predict_action_chunk unavailable; falling back to select_action"
                )
                return self._select_action_as_chunk(batch)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if policy_type == "act":
            if getattr(config, "temporal_ensemble_coeff", None) is not None:
                raise ValueError(
                    "Cyclo chunk runtime does not support ACT temporal ensembling"
                )
            execution_horizon = getattr(config, "n_action_steps", None)
            if (
                isinstance(execution_horizon, bool)
                or not isinstance(execution_horizon, int)
                or execution_horizon < 1
                or action.dim() != 3
                or execution_horizon > action.shape[1]
            ):
                raise ValueError("ACT execution horizon disagrees with predicted chunk")
            action = action[:, :execution_horizon]
        return action

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

    def _select_action_as_chunk(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Use the explicit one-action policy API as a length-one chunk."""

        action = self._policy.select_action(batch)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return action.unsqueeze(1)

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
