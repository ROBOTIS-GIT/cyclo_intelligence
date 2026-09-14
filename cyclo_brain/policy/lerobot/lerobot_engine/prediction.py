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
        predictor = getattr(self, "_chunk_predictor", None)
        if predictor is not None:
            action = predictor(batch)
            return action.unsqueeze(1) if action.dim() == 2 else action
        chunk_method = getattr(self._policy, "predict_action_chunk", None)
        if callable(chunk_method):
            action = chunk_method(batch)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            return action
        logger.debug(
            "predict_action_chunk unavailable; falling back to select_action"
        )
        action = self._policy.select_action(batch)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return action.unsqueeze(1)

    @staticmethod
    def _to_numpy_chunk(action: torch.Tensor) -> np.ndarray:
        """Single-batch chunk or unbatched (T, A)/(A,) -> float64 numpy."""
        chunk = action.detach()
        if chunk.dim() == 3:
            if chunk.shape[0] != 1:
                raise ValueError(
                    f"Expected a single batch for one robot, got {tuple(chunk.shape)}"
                )
            chunk = chunk[0]
        elif chunk.dim() == 2:
            pass
        elif chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        else:
            raise ValueError(
                f"Unexpected action tensor shape: {tuple(chunk.shape)}"
            )
        return chunk.cpu().to(torch.float64).numpy()
