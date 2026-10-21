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
        config = self._policy.config
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
