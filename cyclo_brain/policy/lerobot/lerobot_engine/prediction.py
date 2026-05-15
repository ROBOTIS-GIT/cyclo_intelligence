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
        try:
            action = self._policy.predict_action_chunk(batch)
            if action.dim() == 2:
                action = action.unsqueeze(1)
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
