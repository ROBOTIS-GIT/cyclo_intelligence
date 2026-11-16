"""Embedded RLDXPolicy behind Cyclo's existing Engine Worker protocol.

No second model server, ROS publisher, or policy-specific control loop.
"""

import gc
import json
import logging
import math
import os
from pathlib import Path
import uuid

import yaml

from .mapping import RobotMapping


logger = logging.getLogger("rldx_engine")


class RLDXEngine:
    def __init__(self):
        self.policy = None
        self.robot = None
        self.mapping = None
        self.context_key = None
        self.instruction = None
        self.session_id = str(uuid.uuid4())

    @property
    def is_ready(self):
        return self.policy is not None and self.robot is not None and self.mapping is not None

    def load_policy(self, request):
        try:
            self.cleanup()
            path = Path(request.model_path)
            if not path.is_dir():
                raise ValueError("RLDX requires an exported local checkpoint directory")
            config = json.loads((path / "config.json").read_text())
            metadata = json.loads((path / "cyclo_input_metadata.json").read_text())
            if config.get("model_type") != "RLDX-1":
                raise ValueError("Checkpoint model_type must be RLDX-1")
            if metadata["robot_type"] != request.robot_type:
                raise ValueError("Checkpoint robot_type differs from the selected robot")
            if any(config.get(k, False) for k in ("use_memory", "use_physics", "use_motion")):
                raise ValueError("RLDX memory/physics/motion execution contracts are not integrated yet")
            if config.get("rtc_inference_mode", "none") != "none" or config.get("rtc_inference_delay", 0):
                raise ValueError("RLDX RTC requires an execution-feedback adapter; unsupported here")
            if config.get("video_length", 4) != 1:
                raise ValueError("This RLDX adapter currently requires PT-IMG (video_length=1)")
            if not (path / "processor/processor_config.json").is_file():
                raise ValueError("Checkpoint is missing its saved RLDX processor")
            settings_path = Path(os.environ.get(
                "RLDX_INFERENCE_CONFIG", str(Path(__file__).parents[1] / "configs/inference.yaml")))
            settings = yaml.safe_load(settings_path.read_text())
            expected = {"image_preprocessing": "identity", "rotation": "robot_config",
                        "video_offsets_frames": [0], "state_offsets_frames": [0]}
            if set(settings) != set(expected) | {"max_observation_age_s", "embodiment"}:
                raise ValueError("Unknown or missing RLDX inference settings")
            if any(settings.get(k) != v for k, v in expected.items()):
                raise ValueError("RLDX currently supports identity preprocessing and current observations only")
            self.max_age_s = float(settings["max_observation_age_s"])
            if not math.isfinite(self.max_age_s) or self.max_age_s <= 0:
                raise ValueError("max_observation_age_s must be finite and positive")

            from robot_client import RobotClient
            from robot_client.camera_mapping import resolve_camera_feature_sources
            from rldx.data.embodiment_tags import EmbodimentTag
            from rldx.policy.rldx_policy import RLDXPolicy

            self.robot = RobotClient(request.robot_type, defer_subscriptions=True)
            cameras = resolve_camera_feature_sources(metadata["cameras"], self.robot.camera_names)
            self.mapping = RobotMapping(metadata, self.robot._config, self.robot._action_groups, cameras)
            self.robot.configure_joint_views(self.mapping.joint_views)
            self.policy = RLDXPolicy(
                model_path=str(path), embodiment_tag=EmbodimentTag[settings["embodiment"]],
                device="cuda", strict=True,
            )
            mc = self.policy.get_modality_config()
            for name, keys in (("state", ["joint_position"]), ("action", ["joint_position"]),
                               ("video", list(metadata["cameras"]))):
                if list(mc[name].modality_keys) != keys:
                    raise ValueError(f"{name}: checkpoint modality order differs from exported metadata")
            for name in ("state", "video", "language"):
                if list(mc[name].delta_indices) != [0]:
                    raise ValueError(f"{name}: checkpoint needs an unsupported observation history")
            if len(mc["language"].modality_keys) != 1:
                raise ValueError("RLDX requires one instruction key")
            for kind, names in (("state", self.mapping.state_names), ("action", self.mapping.action_names)):
                dimensions = getattr(self.policy.validator, f"expected_{kind}_dims")
                if dimensions != {"joint_position": len(names)}:
                    raise ValueError(f"Checkpoint {kind} statistics differ from exported channel names")
            self.language_key = mc["language"].modality_keys[0]
            self.horizon = len(mc["action"].delta_indices)
            if list(mc["action"].delta_indices) != list(range(self.horizon)) or self.horizon < 1:
                raise ValueError("Action offsets must be consecutive from zero")
            self.robot.start_observation_subscriptions(**self.mapping.required)
            if not self.robot.wait_for_ready(timeout=10., **self.mapping.required):
                raise ValueError("Required observations are not ready: " + ", ".join(
                    self.robot.get_missing_observations(**self.mapping.required)))
            # First CUDA/processor initialization belongs to LOAD's long timeout.
            # No action from warmup leaves this Worker.
            warmup = self.get_action_chunk(request)
            if not warmup["success"]:
                raise ValueError(f"RLDX warmup failed: {warmup['message']}")
            self.policy.reset()
            self.instruction = None
            return {"success": True, "message": "RLDX-1 loaded", "action_keys": self.mapping.action_keys}
        except Exception as exc:
            logger.exception("LOAD failed")
            self.cleanup()
            return {"success": False, "message": str(exc)}

    def update_execution_context(self, context):
        key = (context.session_id, context.generation)
        if key != self.context_key:
            if self.policy is not None:
                self.policy.reset()
            self.context_key = key
            self.instruction = None
            self.session_id = str(uuid.uuid4())

    def get_action_chunk(self, request):
        if not self.is_ready:
            return {"success": False, "message": "RLDX is not loaded"}
        try:
            instruction = str(request.task_instruction or "")
            reset = self.instruction != instruction
            if reset:
                self.policy.reset()
            snapshot = self.robot.get_required_input_snapshot(self.mapping.sources, max_age_s=self.max_age_s)
            observation = self.mapping.observation(snapshot, instruction, self.language_key)
            actions, _ = self.policy.get_action(observation, options={
                "session_ids": [self.session_id], "reset_memory": [reset],
            })
            chunk = self.mapping.action(actions, self.horizon)
            self.instruction = instruction
            return {"success": True, "action_chunk": chunk.ravel(),
                    "chunk_size": len(chunk), "action_dim": chunk.shape[1]}
        except Exception as exc:
            logger.exception("GET_ACTION failed")
            return {"success": False, "message": str(exc)}

    def cleanup(self):
        robot, self.robot = self.robot, None
        self.policy = None
        self.mapping = None
        self.context_key = None
        self.instruction = None
        self.session_id = str(uuid.uuid4())
        try:
            if robot is not None:
                robot.close()
        finally:
            gc.collect()
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def create_engine():
    return RLDXEngine()
