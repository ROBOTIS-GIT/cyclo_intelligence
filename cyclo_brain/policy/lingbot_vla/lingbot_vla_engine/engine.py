"""LingBot-VLA current-observation chunks behind the shared Engine protocol."""

import gc
import logging
import math
import os
from pathlib import Path
import sys

import yaml

from .bundle import CheckpointBundle
from .mapping import RobotMapping
from .native import load_native, reset_native

logger = logging.getLogger("lingbot_vla_engine")


class LingBotVLAEngine:
    def __init__(self):
        self.policy = self.robot = self.mapping = self.bundle = None
        self.context_key = self.instruction = None

    @property
    def is_ready(self):
        return self.policy is not None and self.robot is not None and self.mapping is not None

    def load_policy(self, request):
        try:
            self.cleanup()
            self.bundle = CheckpointBundle(request.model_path)
            metadata = self.bundle.metadata
            if metadata["robot_type"] != request.robot_type:
                raise ValueError("Checkpoint robot_type differs from the selected robot")
            settings = yaml.safe_load(Path(os.environ.get(
                "LINGBOT_VLA_INFERENCE_CONFIG", str(Path(__file__).parents[1] / "configs/inference.yaml"))).read_text())
            if (set(settings) != {"image_preprocessing", "rotation", "max_observation_age_s", "precision"}
                    or settings["image_preprocessing"] != "identity" or settings["rotation"] != "robot_config"
                    or settings["precision"] not in ("float32", "bfloat16")):
                raise ValueError("Unsupported LingBot-VLA inference settings")
            self.max_age_s = float(settings["max_observation_age_s"])
            if not math.isfinite(self.max_age_s) or self.max_age_s <= 0:
                raise ValueError("max_observation_age_s must be finite and positive")
            from robot_client import RobotClient
            from robot_client.camera_mapping import resolve_camera_feature_sources

            self.robot = RobotClient(request.robot_type, defer_subscriptions=True)
            cameras = resolve_camera_feature_sources(metadata["cameras"], self.robot.camera_names)
            self.mapping = RobotMapping(metadata, self.robot._config, self.robot._action_groups, cameras)
            self.robot.configure_joint_views(self.mapping.joint_views)
            self.policy = load_native(self.bundle, settings["precision"])
            self.robot.start_observation_subscriptions(**self.mapping.required)
            if not self.robot.wait_for_ready(timeout=10., **self.mapping.required):
                raise ValueError("Required observations are not ready: " + ", ".join(
                    self.robot.get_missing_observations(**self.mapping.required)))
            # Use LOAD's long timeout for CUDA initialization. Never publish warmup.
            warmup = self.get_action_chunk(request)
            if not warmup["success"]:
                raise ValueError(f"LingBot-VLA warmup failed: {warmup['message']}")
            reset_native(self.policy)
            self.instruction = None
            return {"success": True, "message": "LingBot-VLA 2.0 loaded", "action_keys": self.mapping.action_keys}
        except Exception as exc:
            logger.exception("LOAD failed")
            self.cleanup()
            return {"success": False, "message": str(exc)}

    def update_execution_context(self, context):
        key = (context.session_id, context.generation)
        if key != self.context_key:
            if self.policy is not None:
                reset_native(self.policy)
            self.context_key = key
            self.instruction = None

    def get_action_chunk(self, request):
        if not self.is_ready:
            return {"success": False, "message": "LingBot-VLA is not loaded"}
        try:
            instruction = str(request.task_instruction or "")
            if instruction != self.instruction:
                reset_native(self.policy)
            snapshot = self.robot.get_required_input_snapshot(self.mapping.sources, max_age_s=self.max_age_s)
            # RobotClient returns private RGB arrays, separate from callback storage.
            observation = self.mapping.observation(snapshot, instruction, copy_images=False)
            actions = self.policy.infer(observation)
            chunk = self.mapping.action(actions, self.bundle.horizon)
            self.instruction = instruction
            return {"success": True, "action_chunk": chunk.ravel(),
                    "chunk_size": len(chunk), "action_dim": chunk.shape[1]}
        except Exception as exc:
            logger.exception("GET_ACTION failed")
            return {"success": False, "message": str(exc)}

    def cleanup(self):
        robot, self.robot = self.robot, None
        self.policy = self.mapping = self.bundle = None
        self.context_key = self.instruction = None
        try:
            if robot is not None:
                robot.close()
        finally:
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def create_engine():
    return LingBotVLAEngine()
