"""Checkpoint loading for the dedicated ViTacFormer backend."""

from __future__ import annotations

import logging

import torch

from .model import ViTacFormerPolicy, load_vitacformer_policy


logger = logging.getLogger("vitacformer_engine")


class LoadingMixin:
    def _load_policy_assets(
        self,
        model_path: str,
        device: torch.device,
    ) -> ViTacFormerPolicy:
        logger.info("Loading ViTacFormer policy from %s", model_path)
        return load_vitacformer_policy(model_path, device)
