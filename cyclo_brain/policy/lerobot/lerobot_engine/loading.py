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
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_resolve_model_dir``: auto-descend lerobot training-output roots.
- ``_load_policy_assets``: load weights + stored pre/post processors.
- ``_infer_image_resize``: read input-image shape hint off the policy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .constants import IMAGE_KEY_PREFIX as _IMAGE_KEY_PREFIX

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy


logger = logging.getLogger("lerobot_engine")


class LoadingMixin:
    """Policy load helpers — weights, processors, resize hint."""

    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        """Auto-descend lerobot training-output roots.

        Users frequently paste the training-output root which contains
        ``pretrained_model/`` next to ``training_state/``. Strip that
        wrapper if needed so ``from_pretrained`` finds ``config.json``.
        """
        root = Path(model_path)
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
                policy_type = json.load(f).get("type", "act")
        else:
            # ACT was the original default; fall back to it for
            # checkpoints saved before ``type`` started being recorded.
            policy_type = "act"

        logger.info("Policy type: %s", policy_type)
        PolicyClass = get_policy_class(policy_type)

        # ``from_pretrained`` reads config.json, instantiates the policy
        # config, then loads safetensors. We don't pass ``config=`` — the
        # saved config is already what we want.
        policy = PolicyClass.from_pretrained(model_path)
        policy = policy.to(device).eval()
        logger.info("Policy weights loaded on %s", device)

        # Stored processor pipelines include the dataset-time normalizer
        # stats and image transforms so we don't re-derive (and de-sync)
        # them. Falling through to the default factory here would wipe
        # those stats and produce garbage actions.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )
        logger.info("Pre/post processors loaded")
        return policy, preprocessor, postprocessor

    def _infer_image_resize(self, policy: PreTrainedPolicy) -> Optional[tuple]:
        """Best-effort target (W, H) from policy.config.input_features.

        Many lerobot policies advertise the expected image shape under
        ``input_features['observation.images.<cam>'].shape = (C, H, W)``.
        Pre-resizing on the host saves the preprocessor a copy on every
        tick. Returning None means: leave at native resolution.
        """
        try:
            features = getattr(policy.config, "input_features", {}) or {}
            for key, feat in features.items():
                if not key.startswith(_IMAGE_KEY_PREFIX):
                    continue
                shape = getattr(feat, "shape", None)
                if shape and len(shape) == 3:
                    _, h, w = shape
                    return (int(w), int(h))
        except Exception:
            pass
        return None
