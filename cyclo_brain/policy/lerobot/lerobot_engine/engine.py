#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot inference engine.

Implements ``InferenceEngine`` (cyclo_brain.policy.common.runtime.engine)
on top of upstream LeRobot's pretrained-policy + processor-pipeline
APIs. Installed into the Worker image as the ``/app/lerobot_engine/`` package;
the common Engine process imports it via
``POLICY_ENGINE_MODULE=lerobot_engine`` (the package's ``__init__.py``
re-exports ``LeRobotEngine`` + ``create_engine``).

Mirrors groot's ``inference_engine.py`` structure (RobotClient owns
sensor subscriptions; engine builds observations on demand) so the central
Policy Runtime can route both Workers through the same Engine contract.

This file holds the ``LeRobotEngine`` core — ``__init__``, the
``InferenceEngine`` API surface (``is_ready``, ``load_policy``,
``get_action_chunk``, ``cleanup``), the ``_fail`` helper, and the
module-level ``create_engine()`` factory. The implementation details
live in three mixin siblings inside the ``lerobot_engine`` sub-package:

- ``loading.LoadingMixin``: policy weights + processor load helpers.
- ``optimization.OptimizationMixin``: optional backend optimization hook.
- ``io_mapping.IoMappingMixin``: RobotClient wiring / camera+state map.
- ``preprocessing.PreprocessingMixin``: RobotClient observation -> model input.
- ``prediction.PredictionMixin``: model input -> action chunk.

Upstream API used:

- ``PreTrainedPolicy.from_pretrained(model_path, config=cfg)`` — loads
  weights and the saved policy config (auto-detects type via
  ``config.json``).
- ``make_pre_post_processors(policy_cfg, pretrained_path=model_path)`` —
  loads the stored normalizer / image / device steps so we don't
  reinvent (and de-sync from) preprocessing.
- ``policy.predict_action_chunk(batch)`` for chunked inference;
  fallback to ``policy.select_action(batch)`` for non-chunked
  policies (TDMPC, SAC, …).
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np


# -- robot_client import shim --------------------------------------------------
# /robot_client_sdk is the bind-mount root; the package itself sits at
# /robot_client_sdk/robot_client/ so the parent dir goes onto sys.path.
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", "/robot_client_sdk")
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)


# Import order: engine ABC first (validates /policy_runtime is on PYTHONPATH),
# then heavy ML deps.
from engine import InferenceEngine  # noqa: E402

import torch  # noqa: E402

from robot_client import RobotClient  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402

# Mixins are sub-package siblings. The Engine process loads the package via
# importlib.import_module("lerobot_engine") — the package's __init__.py
# re-exports LeRobotEngine + create_engine, so relative imports here
# resolve against /app/lerobot_engine/.
from .loading import LoadingMixin  # noqa: E402
from .optimization import OptimizationMixin  # noqa: E402
from .io_mapping import IoMappingMixin  # noqa: E402
from .preprocessing import PreprocessingMixin  # noqa: E402
from .prediction import PredictionMixin  # noqa: E402
from .image_preprocessing import load_image_preprocessing  # noqa: E402


logger = logging.getLogger("lerobot_engine")


class LeRobotEngine(
    LoadingMixin,
    OptimizationMixin,
    IoMappingMixin,
    PreprocessingMixin,
    PredictionMixin,
    InferenceEngine,
):
    """Wraps a LeRobot ``PreTrainedPolicy`` + processors + ``RobotClient``."""

    def __init__(self) -> None:
        self._policy: Optional[PreTrainedPolicy] = None
        self._preprocessor = None
        self._postprocessor = None
        self._robot: Optional[RobotClient] = None
        self._device: Optional[torch.device] = None
        self._loaded_model_path: Optional[str] = None

        # Resolved after load: which cameras / joint groups feed which
        # policy keys. ``_cameras`` maps RobotClient camera name → policy
        # input key (``observation.images.<cam>``). ``_state_modalities``
        # is the sorted list of follower joint groups whose positions are
        # concatenated into ``observation.state``.
        self._cameras: Dict[str, str] = {}
        self._state_modalities: List[str] = []
        self._action_keys: List[str] = []
        self._has_mobile_state: bool = False
        # Cached robot_type for repeated LOAD requests before an explicit
        # UNLOAD. cleanup() must clear this together with the policy cache.
        self._loaded_robot_type: Optional[str] = None

        self._image_preprocessing = None

    # ------------------------------------------------------------------ #
    # InferenceEngine API
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        return (
            self._policy is not None
            and self._preprocessor is not None
            and self._postprocessor is not None
            and self._robot is not None
            and self._image_preprocessing is not None
        )

    def load_policy(self, request: Any) -> Dict[str, Any]:
        model_path = request.model_path
        robot_type = request.robot_type

        try:
            # Auto-descend into ``pretrained_model/`` if the user pasted
            # a training-output root containing ``training_state/``
            # alongside (lerobot-train layout).
            model_path = self._resolve_model_dir(model_path)
            # Validate before allocating weights, including on cached LOAD.
            image_preprocessing = load_image_preprocessing(model_path)

            # Skip weights load when a second LOAD arrives before UNLOAD and
            # we're just reattaching the robot client for the same model.
            cache_hit = (
                self._policy is not None
                and self._loaded_model_path == model_path
            )
            if cache_hit:
                logger.info("Reusing cached policy: %s", model_path)
                self._teardown_robot()
            else:
                # Release the old policy before constructing another large model.
                self.cleanup()
                logger.info("Loading LeRobot policy from: %s", model_path)
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                policy, preprocessor, postprocessor = self._load_policy_assets(
                    model_path, self._device
                )
                self._policy = policy
                self._preprocessor = preprocessor
                self._postprocessor = postprocessor
                self._loaded_model_path = model_path
                self._apply_policy_optimization(model_path, request)

            self._image_preprocessing = image_preprocessing
            self._init_robot(robot_type)
            self._loaded_robot_type = robot_type

            return {
                "success": True,
                "message": (
                    "LeRobot inference restarted (policy cached)"
                    if cache_hit
                    else f"loaded {model_path}"
                ),
                "action_keys": list(self._action_keys),
            }
        except Exception as e:
            logger.error("load_policy failed: %s", e, exc_info=True)
            self.cleanup()
            return self._fail(str(e))

    def get_action_chunk(self, request: Any) -> Dict[str, Any]:
        if not self.is_ready:
            return self._fail("Not in inference mode")
        try:
            obs = self._build_observation(getattr(request, "task_instruction", ""))
            if "success" in obs:
                return obs

            with torch.inference_mode():
                preprocessed = self._preprocessor(obs)
                self._validate_camera_shapes(preprocessed)
                action = self._predict_chunk(preprocessed)
                action = self._postprocessor(action)

            chunk = self._to_numpy_chunk(action)
            T, D = chunk.shape
            logger.info("Action chunk: T=%d, D=%d", T, D)
            return {
                "success": True,
                # Keep flat numpy — zenoh_ros2_sdk's publisher uses .view()
                # for fast CDR encoding and crashes on plain Python lists.
                "action_chunk": np.ascontiguousarray(
                    chunk.reshape(-1), dtype=np.float64
                ),
                "chunk_size": int(T),
                "action_dim": int(D),
            }
        except Exception as e:
            logger.error("get_action_chunk failed: %s", e, exc_info=True)
            return self._fail(str(e))

    def cleanup(self) -> None:
        """Release robot and policy resources for a true UNLOAD."""
        self._teardown_robot()

        had_policy = any(
            obj is not None
            for obj in (self._policy, self._preprocessor, self._postprocessor)
        )
        if had_policy:
            logger.info("Releasing LeRobot policy: %s", self._loaded_model_path)

        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
        self._loaded_model_path = None
        self._loaded_robot_type = None
        self._image_preprocessing = None

        self._cameras = {}
        self._state_modalities = []
        self._action_keys = []
        self._has_mobile_state = False

        if had_policy:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    @staticmethod
    def _fail(message: str) -> Dict[str, Any]:
        return {"success": False, "message": message}


# ----------------------------------------------------------------------------
# Entry point used by common/runtime/engine_process/worker.py (via the package's
# ``__init__.py`` re-export).
# ----------------------------------------------------------------------------


def create_engine() -> InferenceEngine:
    return LeRobotEngine()
