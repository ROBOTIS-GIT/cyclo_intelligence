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


logger = logging.getLogger("lerobot_engine")


class PredictionMixin:
    """Policy input batch -> action chunk."""

    def _predict_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk tensor of shape (1, T, A)."""
        assert self._policy is not None
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
        config = self._policy.config
        if getattr(config, "type", None) == "act":
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
