"""Stateless ABot-M0 action chunks behind the existing Cyclo Worker protocol."""

import gc
import logging
import math
import os
from pathlib import Path
import sys

import yaml

from .bundle import Bundle
from .mapping import RobotMapping
from .native import load_policy

logger = logging.getLogger("abot_engine")


class ABotEngine:
    def __init__(self):
        self.policy = self.robot = self.mapping = None

    @property
    def is_ready(self):
        return all(x is not None for x in (self.policy, self.robot, self.mapping))

    def load_policy(self, request):
        try:
            self.cleanup()
            bundle = Bundle(request.model_path, request.robot_type)
            settings = yaml.safe_load(Path(os.environ.get(
                "ABOT_INFERENCE_CONFIG", str(Path(__file__).parents[1] / "configs/inference.yaml"))).read_text())
            expected = {"image_preprocessing": "identity", "rotation": "robot_config", "observation_offsets": [0]}
            if (set(settings) != set(expected) | {"max_observation_age_s"}
                    or any(settings[k] != v for k, v in expected.items())):
                raise ValueError("ABot currently supports identity image preprocessing and current observations only")
            self.max_age_s = float(settings["max_observation_age_s"])
            if not math.isfinite(self.max_age_s) or self.max_age_s <= 0:
                raise ValueError("max_observation_age_s must be finite and positive")
            from robot_client import RobotClient
            from robot_client.camera_mapping import resolve_camera_feature_sources

            self.robot = RobotClient(request.robot_type, defer_subscriptions=True)
            cameras = resolve_camera_feature_sources(bundle.cameras, self.robot.camera_names)
            self.mapping = RobotMapping(bundle, self.robot._config, self.robot._action_groups, cameras)
            self.robot.configure_joint_views(self.mapping.joint_views)
            self.policy = load_policy(bundle)
            self.robot.start_observation_subscriptions(**self.mapping.required)
            if not self.robot.wait_for_ready(timeout=10., **self.mapping.required):
                raise ValueError("Required observations are not ready: " + ", ".join(
                    self.robot.get_missing_observations(**self.mapping.required)))
            # Initialize CUDA under LOAD's timeout, without publishing commands.
            result = self.get_action_chunk(request)
            if not result["success"]:
                raise ValueError(f"ABot warmup failed: {result['message']}")
            return {"success": True, "message": "ABot-M0 loaded", "action_keys": self.mapping.action_keys}
        except Exception as exc:
            logger.exception("LOAD failed")
            self.cleanup()
            return {"success": False, "message": str(exc)}

    def get_action_chunk(self, request):
        if not self.is_ready:
            return {"success": False, "message": "ABot-M0 is not loaded"}
        try:
            snapshot = self.robot.get_required_input_snapshot(self.mapping.sources, max_age_s=self.max_age_s)
            example = self.mapping.observation(snapshot, str(request.task_instruction or ""))
            action = self.mapping.action(self.policy.predict_action([example]))
            return {"success": True, "action_chunk": action.ravel(),
                    "chunk_size": len(action), "action_dim": action.shape[1]}
        except Exception as exc:
            logger.exception("GET_ACTION failed")
            return {"success": False, "message": str(exc)}

    def cleanup(self):
        robot, self.robot = self.robot, None
        self.policy = self.mapping = None
        try:
            if robot is not None:
                robot.close()
        finally:
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def create_engine():
    return ABotEngine()
