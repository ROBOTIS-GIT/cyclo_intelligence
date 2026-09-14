#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine loading helpers (LoadingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Installed in the Worker as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_resolve_model_dir``: auto-descend lerobot training-output roots.
- ``_load_policy_assets``: load weights + stored pre/post processors.
- ``_prepare_policy_predictor``: prepare the adapter's optional batch predictor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy

from .policy_validation import validate_checkpoint
from .adapters import resolve_adapter


logger = logging.getLogger("lerobot_engine")


class LoadingMixin:
    """Policy load helpers — weights and saved processors."""

    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        """Auto-descend lerobot training-output roots.

        Users frequently paste the training-output root which contains
        ``pretrained_model/`` next to ``training_state/``. Strip that
        wrapper if needed so ``from_pretrained`` finds ``config.json``.
        """
        root = Path((model_path or "").strip())
        nested = root / "pretrained_model"
        if not (root / "config.json").exists() and (nested / "config.json").exists():
            logger.info("Descending into pretrained_model: %s", nested)
            return str(nested)
        return str(root)

    @staticmethod
    def _load_policy_assets(
        model_path: str, device: torch.device
    ) -> tuple[PreTrainedPolicy, Any, Any]:
        """Load policy weights + saved pre/post processors."""
        import json

        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                checkpoint_config = json.load(f)
            validate_checkpoint(checkpoint_config, model_path)
            policy_type = checkpoint_config.get("type", "act")
        else:
            # ACT was the original default; fall back to it for
            # checkpoints saved before ``type`` started being recorded.
            policy_type = "act"

        logger.info("Policy type: %s", policy_type)
        PolicyClass = get_policy_class(policy_type)

        loader = resolve_adapter(policy_type).policy_loader
        if loader is not None:
            policy = loader(PolicyClass, PreTrainedConfig, model_path, device)
        else:
            policy = PolicyClass.from_pretrained(model_path).to(device).eval()
            logger.info("Policy weights loaded on %s", device)

        # Restore serialized steps and statistics. Dataset-side transforms
        # outside this pipeline are not necessarily recorded in the checkpoint.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )
        logger.info("Pre/post processors loaded")
        return policy, preprocessor, postprocessor

    def _prepare_policy_predictor(self, request: Any) -> None:
        """Prepare once per weights load; cached LOAD keeps the predictor."""
        factory = resolve_adapter(self._policy.config.type).predictor_factory
        self._chunk_predictor = factory(self._policy, self._device, request) if factory else None
        if factory is not None and not callable(self._chunk_predictor):
            raise TypeError("adapter predictor_factory must return a callable")
