"""Dedicated ViTacFormer inference engine."""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Any, Dict, Optional

import numpy as np


_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", "/robot_client_sdk")
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

from engine import InferenceEngine  # noqa: E402

import torch  # noqa: E402

from robot_client import RobotClient  # noqa: E402

from .io_mapping import IoMappingMixin  # noqa: E402
from .loading import LoadingMixin  # noqa: E402
from .model import ViTacFormerPolicy  # noqa: E402
from .prediction import PredictionMixin  # noqa: E402
from .preprocessing import PreprocessingMixin  # noqa: E402


logger = logging.getLogger("vitacformer_engine")


class ViTacFormerEngine(
    LoadingMixin,
    IoMappingMixin,
    PreprocessingMixin,
    PredictionMixin,
    InferenceEngine,
):
    def __init__(self) -> None:
        self._policy: Optional[ViTacFormerPolicy] = None
        self._robot: Optional[RobotClient] = None
        self._device: Optional[torch.device] = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_robot_type: Optional[str] = None
        self._tactile_inputs: Dict[str, str] = {}
        self._tactile_baselines: Dict[str, np.ndarray] = {}
        self._action_keys: list[str] = []
        self._execution_horizon = int(os.environ.get("VITACFORMER_EXECUTION_HORIZON", "20"))
        if not 1 <= self._execution_horizon <= 100:
            raise ValueError("VITACFORMER_EXECUTION_HORIZON must be between 1 and 100")

    @property
    def is_ready(self) -> bool:
        return self._policy is not None and self._robot is not None

    def load_policy(self, request: Any) -> Dict[str, Any]:
        if self.is_ready:
            return self._fail("ViTacFormer policy already loaded - UNLOAD first")
        model_path = str(getattr(request, "model_path", "") or "").strip()
        robot_type = str(getattr(request, "robot_type", "") or "").strip()
        if robot_type != "ffw_sh5_rev1":
            return self._fail("ViTacFormer supports only ffw_sh5_rev1")
        acceleration = str(getattr(request, "acceleration_mode", "") or "pytorch")
        if acceleration != "pytorch":
            return self._fail("ViTacFormer supports only PyTorch inference")
        try:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._policy = self._load_policy_assets(model_path, self._device)
            self._loaded_model_path = model_path
            self._loaded_robot_type = robot_type
            self._policy.reset()
            self._init_robot(robot_type)
            return {
                "success": True,
                "message": f"loaded ViTacFormer {model_path}",
                "action_keys": list(self._action_keys),
            }
        except Exception as exc:
            logger.error("ViTacFormer load failed: %s", exc, exc_info=True)
            self.cleanup()
            return self._fail(str(exc))

    def get_action_chunk(self, request: Any) -> Dict[str, Any]:
        del request
        if not self.is_ready:
            return self._fail("ViTacFormer policy is not loaded")
        try:
            chunk = self._predict_chunk(self._build_observation())
            # Execute a short prefix before replanning; keep the trained
            # 100-row decoder intact and preserve the 30 Hz action time base.
            chunk = chunk[:self._execution_horizon]
            rows, action_dim = chunk.shape
            return {
                "success": True,
                "action_chunk": chunk.reshape(-1),
                "chunk_size": int(rows),
                "action_dim": int(action_dim),
            }
        except Exception as exc:
            logger.error("ViTacFormer inference failed: %s", exc, exc_info=True)
            return self._fail(str(exc))

    def cleanup(self) -> None:
        self._teardown_robot()
        had_policy = self._policy is not None
        self._policy = None
        self._device = None
        self._loaded_model_path = None
        self._loaded_robot_type = None
        self._tactile_inputs = {}
        self._tactile_baselines = {}
        self._action_keys = []
        if had_policy:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    @staticmethod
    def _fail(message: str) -> Dict[str, Any]:
        return {"success": False, "message": message}


def create_engine() -> InferenceEngine:
    return ViTacFormerEngine()
