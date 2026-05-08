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
APIs. Bind-mounted into the policy container at ``/app/lerobot_engine.py``;
the common Process A imports it via ``POLICY_ENGINE_MODULE=lerobot_engine``.

Mirrors groot's ``inference_engine.py`` structure (RobotClient owns
sensor subscriptions; engine builds observations on demand) so the
upstream-agnostic two-process server can route both backends through
the same shape.

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

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# -- robot_client import shim --------------------------------------------------
# /robot_client_sdk is the bind-mount root; the package itself sits at
# /robot_client_sdk/robot_client/ so the parent dir goes onto sys.path.
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", "/robot_client_sdk")
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)


# Import order: engine ABC first (validates /policy_runtime is on
# PYTHONPATH), then heavy ML deps (only after the server has confirmed
# the engine module is loadable).
from engine import InferenceEngine  # noqa: E402

import torch  # noqa: E402

from robot_client import RobotClient  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402


logger = logging.getLogger("lerobot_engine")


# Image keys in lerobot policies follow ``observation.images.<cam_name>``;
# state is always ``observation.state``. The action key naming convention
# changed across versions but the policy emits a flat tensor either way.
_IMAGE_KEY_PREFIX = "observation.images."
_STATE_KEY = "observation.state"


class LeRobotEngine(InferenceEngine):
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
        # Cached robot_type (so re-LOAD with the same model_path can skip
        # the heavy weights load — mirrors GR00TInference.load_policy).
        self._loaded_robot_type: Optional[str] = None

        # Resize target for input cameras. The preprocessor's stored
        # ImageProcessorStep handles normalization and CHW reorder; we
        # only need to pre-resize to roughly the policy's expected size
        # so the bind-mounted JPEGs aren't 4 K. If config doesn't expose
        # a target shape we leave images at native resolution and let
        # the preprocessor resize.
        self._image_resize: Optional[tuple] = None

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
        )

    def load_policy(self, request: Any) -> Dict[str, Any]:
        model_path = request.model_path
        robot_type = request.robot_type

        try:
            # Auto-descend into ``pretrained_model/`` if the user pasted
            # a training-output root containing ``training_state/``
            # alongside (lerobot-train layout).
            model_path = self._resolve_model_dir(model_path)

            # Skip weights load when we're just reattaching the robot
            # client to a different robot for the same model (save 5–30 s).
            cache_hit = (
                self._policy is not None
                and self._loaded_model_path == model_path
            )
            if cache_hit:
                logger.info("Reusing cached policy: %s", model_path)
                self._teardown_robot()
            else:
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

            self._init_robot(robot_type)
            self._loaded_robot_type = robot_type
            self._image_resize = self._infer_image_resize(self._policy)

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
        """Release the robot client; keep policy cached for fast re-LOAD."""
        self._teardown_robot()
        self._cameras = {}
        self._state_modalities = []
        self._action_keys = []
        self._has_mobile_state = False

    # ------------------------------------------------------------------ #
    # Policy load helpers
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Robot wiring
    # ------------------------------------------------------------------ #

    def _init_robot(self, robot_type: str) -> None:
        """Create RobotClient + resolve camera / state mappings."""
        self._robot = RobotClient(robot_type)

        # Cameras: only those that match a policy input key
        # ``observation.images.<cam>``. Cameras advertised by the robot
        # but not consumed by the policy are silently ignored — same
        # behavior as GR00TInference.
        policy_image_keys = self._policy_image_keys()
        active = {}
        for cam in self._robot.camera_names:
            key = f"{_IMAGE_KEY_PREFIX}{cam}"
            if not policy_image_keys or key in policy_image_keys:
                active[cam] = key
        if not active and policy_image_keys:
            raise RuntimeError(
                "No cameras match the policy's expected input keys: "
                f"policy needs {sorted(policy_image_keys)}, robot has "
                f"{self._robot.camera_names}"
            )
        self._cameras = active

        # State modalities: sorted follower joint groups. We follow the
        # same convention groot uses (sorted modality names map to the
        # training-time concat order). Synthetic per-modality views (with
        # ``parent``) win over their leaf physical group; otherwise the
        # leaf group is used directly.
        groups = self._robot._config.get("joint_groups", {})
        parents = {cfg.get("parent") for cfg in groups.values() if cfg.get("parent")}
        modality_groups = []
        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            if cfg.get("parent"):
                modality_groups.append(name)
            elif name not in parents:
                modality_groups.append(name)
        modalities = sorted(name[len("follower_"):] for name in modality_groups)
        if not modalities:
            raise RuntimeError(
                f"No follower joint groups in robot_type={robot_type}"
            )

        # Mobile is sourced from sensors["odom"] in the new schema —
        # bridge it into observation.state alongside the joint states so
        # policies trained on the legacy physical_ai_server pipeline
        # (with mobile as a 3-vector modality) still see it.
        sensors = self._robot._config.get("sensors", {})
        self._has_mobile_state = "odom" in sensors
        if self._has_mobile_state:
            modalities = sorted(set(modalities) | {"mobile"})

        self._state_modalities = modalities
        self._action_keys = list(modalities)

        # Block until at least one frame from each sensor lands. 10 s is
        # generous — typical hardware comes up in <2 s.
        self._robot.wait_for_ready(timeout=10.0)
        logger.info(
            "Robot ready: cameras=%s state_modalities=%s",
            list(self._cameras.keys()),
            self._state_modalities,
        )

    def _teardown_robot(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None

    def _policy_image_keys(self) -> set:
        try:
            features = getattr(self._policy.config, "input_features", {}) or {}
            return {k for k in features.keys() if k.startswith(_IMAGE_KEY_PREFIX)}
        except Exception:
            return set()

    # ------------------------------------------------------------------ #
    # Observation construction
    # ------------------------------------------------------------------ #

    def _build_observation(self, task_instruction: str) -> Dict[str, Any]:
        """Pull raw sensor data from RobotClient → policy-ready batch."""
        assert self._robot is not None  # guarded by is_ready

        # Pull RGB (lerobot policies were trained on RGB) at native
        # resolution if no resize hint, else at the policy's expected
        # (W, H). One memcpy per camera per tick.
        images = self._robot.get_images(resize=self._image_resize, format="rgb")
        if not images:
            return self._fail("No camera frames available")

        joint_dict = self._robot.get_joint_positions()
        if not joint_dict:
            return self._fail("No joint positions available")

        batch: Dict[str, Any] = {}

        # Cameras → observation.images.<cam>: tensor (1, C, H, W) float32
        # in [0, 1]. Mirrors prepare_observation_for_inference's contract
        # — preprocessor normalizes from there.
        for cam_name, policy_key in self._cameras.items():
            img = images.get(cam_name)
            if img is None:
                return self._fail(f"Missing camera frame: {cam_name}")
            # (H, W, C) uint8 → (1, C, H, W) float32 [0, 1]
            tensor = torch.from_numpy(img.copy()).to(torch.float32) / 255.0
            tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
            batch[policy_key] = tensor.to(self._device)

        # State → observation.state: concatenated joint positions in
        # sorted-modality order (must match training-time concat order).
        state_parts: List[np.ndarray] = []
        for modality in self._state_modalities:
            if modality == "mobile":
                odom = self._robot.get_odom()
                if odom is None:
                    return self._fail("Missing odom for mobile state")
                state_parts.append(
                    np.array(
                        [
                            float(odom["linear_velocity"][0]),
                            float(odom["linear_velocity"][1]),
                            float(odom["angular_velocity"][2]),
                        ],
                        dtype=np.float32,
                    )
                )
                continue
            group = f"follower_{modality}"
            positions = joint_dict.get(group)
            if positions is None or len(positions) == 0:
                return self._fail(f"Missing joint group: {modality}")
            state_parts.append(np.asarray(positions, dtype=np.float32))

        flat_state = np.concatenate(state_parts)
        batch[_STATE_KEY] = (
            torch.from_numpy(flat_state).unsqueeze(0).to(self._device)
        )

        # Language conditioning. Many lerobot policies (smolvla, pi0,
        # eo1, …) read ``task`` from the batch — supplying an empty
        # string is harmless for non-language policies.
        batch["task"] = [task_instruction or ""]

        return batch

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _predict_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk tensor of shape (1, T, A).

        Tries ``predict_action_chunk`` first (Pi0, SmolVLA, ACT chunked,
        Diffusion …); falls back to a single-step ``select_action`` call
        for policies that don't implement chunking (TDMPC, SAC).
        """
        assert self._policy is not None
        try:
            action = self._policy.predict_action_chunk(batch)
            if action.dim() == 2:
                # Some policies return (B, A) when chunking is disabled —
                # promote to (B, 1, A) so downstream sees a uniform
                # (B, T, A) layout.
                action = action.unsqueeze(1)
            return action
        except (NotImplementedError, AttributeError):
            logger.debug(
                "predict_action_chunk unavailable; falling back to select_action"
            )
            action = self._policy.select_action(batch)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            return action.unsqueeze(1)  # (B, 1, A)

    @staticmethod
    def _to_numpy_chunk(action: torch.Tensor) -> np.ndarray:
        """(B, T, A) or (B, A) tensor → (T, A) float64 numpy."""
        chunk = action.detach().cpu()
        if chunk.dim() == 3:
            chunk = chunk[0]  # drop batch
        elif chunk.dim() == 2:
            # Rare: (T, A) without explicit batch dim — keep as-is.
            pass
        elif chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        else:
            raise ValueError(
                f"Unexpected action tensor shape: {tuple(chunk.shape)}"
            )
        return chunk.to(torch.float64).numpy()

    @staticmethod
    def _fail(message: str) -> Dict[str, Any]:
        return {"success": False, "message": message}


# ----------------------------------------------------------------------------
# Entry point used by common/runtime/inference_server.py
# ----------------------------------------------------------------------------


def create_engine() -> InferenceEngine:
    return LeRobotEngine()
