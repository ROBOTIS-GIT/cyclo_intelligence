#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

"""GR00T N1.6 inference engine.

Encapsulates Gr00tPolicy loading, RobotClient setup, observation
preprocessing, and action chunk postprocessing. Imported by
runtime/inference_server.py (Process A) which slots it into the
cyclo_intelligence two-process pattern (LOAD srv → configure broadcast
→ Zenoh trigger/chunk).

Original Step 1 location: cyclo_brain/policy/groot/inference.py.
Moved to runtime/ as part of D10-groot (mirrors lerobot/runtime/ layout).
"""
import logging
import os
import sys
import tempfile
import time
import ast
from typing import Optional

import cv2
import numpy as np
import torch


# -- robot_client import shim --------------------------------------------------
# /robot_client_sdk/ is the bind-mount root; the package itself sits at
# /robot_client_sdk/robot_client/ so the parent dir goes onto sys.path.
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", "/robot_client_sdk")
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

import gr00t.model  # noqa: F401 - register custom models
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from robot_client import RobotClient  # noqa: E402
from robot_client.camera_mapping import resolve_camera_feature_sources  # noqa: E402

try:
    from hf_token_sync import sync_token_file  # noqa: E402
except Exception:  # pragma: no cover - helper is mounted in policy containers.
    sync_token_file = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logging.getLogger("groot_inference").warning(
            "Ignoring invalid %s=%r; using %s",
            name,
            value,
            default,
        )
        return default
    return parsed if parsed > 0 else default


ACCELERATION_PYTORCH = "pytorch"
ACCELERATION_TENSORRT_DIT = "tensorrt_dit"
ACCELERATION_TENSORRT_FULL_PIPELINE = "tensorrt_full_pipeline"
SUPPORTED_ACCELERATION_MODES = {
    ACCELERATION_PYTORCH,
    ACCELERATION_TENSORRT_DIT,
    ACCELERATION_TENSORRT_FULL_PIPELINE,
}

# Add GR00T root to sys.path for deployment script imports.
# After the move into runtime/, parents[0]=runtime, parents[1]=groot,
# so the legacy in-tree fallback is a layer deeper than before — but
# the /gr00t fallback (where the submodule lives in the container)
# still wins, which is what we want at runtime.
_GROOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_GROOT_ROOT, "scripts", "deployment")):
    _groot_path = _GROOT_ROOT
elif os.path.isdir("/gr00t/scripts/deployment"):
    _groot_path = "/gr00t"
else:
    _groot_path = None

if _groot_path and _groot_path not in sys.path:
    sys.path.insert(0, _groot_path)

# TensorRT DiT acceleration - reuse existing GR00T deployment code
from scripts.deployment.standalone_inference_script import (  # noqa: E402
    replace_dit_with_tensorrt,
)
try:  # GR00T N1.7
    from scripts.deployment.export_onnx_n1d7 import (  # noqa: E402
        DiTInputCapture,
        export_dit_to_onnx,
    )
except ImportError:  # GR00T N1.6 / N1.6.1
    from scripts.deployment.export_onnx_n1d6 import (  # noqa: E402
        DiTInputCapture,
        export_dit_to_onnx,
    )


def build_trt_engine(
    policy: Gr00tPolicy,
    observation: dict,
    engine_path: str,
    workspace_mb: Optional[int] = None,
):
    """Export DiT to ONNX and build TensorRT engine automatically.

    Uses DiTInputCapture and export_dit_to_onnx from GR00T deployment scripts,
    then builds the TRT engine via build_tensorrt_engine.build_engine().
    Takes ~3-5 min on Orin.
    """
    from scripts.deployment.build_tensorrt_engine import (
        build_engine,
        derive_shapes_with_hint,
    )

    logger = logging.getLogger("groot_inference")
    engine_dir = os.path.dirname(engine_path)
    if workspace_mb is None:
        workspace_mb = _env_int("GROOT_TRT_WORKSPACE_MB", 4096)

    # Step 1: Capture DiT input shapes via hook
    logger.info("Capturing DiT input shapes...")
    capture = DiTInputCapture()
    hook = policy.model.action_head.model.register_forward_pre_hook(
        capture.hook_fn, with_kwargs=True
    )
    with torch.inference_mode():
        policy.get_action(observation)
    hook.remove()

    if not capture.captured:
        raise RuntimeError("Failed to capture DiT inputs")

    # Step 2/3: Export DiT to ONNX in an isolated temporary directory, then
    # build the TensorRT engine into the checkpoint directory. GR00T's ONNX
    # exporter consolidates external-data files by deleting every non
    # .onnx/.json/.data file next to the ONNX path. Keeping ONNX artifacts out
    # of the checkpoint directory prevents model-*.safetensors from being
    # mistaken for temporary external data.
    with tempfile.TemporaryDirectory(prefix=".trt_export_", dir=engine_dir) as export_dir:
        onnx_path = os.path.join(export_dir, "dit_model_bf16.onnx")

        # Patch torch.onnx.export to force dynamo=False (avoids torch.export
        # failures with DiT's dynamic shapes, without modifying upstream GR00T).
        _orig_export = torch.onnx.export
        def _patched_export(*args, **kwargs):
            kwargs.setdefault("dynamo", False)
            return _orig_export(*args, **kwargs)
        torch.onnx.export = _patched_export
        try:
            export_dit_to_onnx(
                policy=policy,
                captured_inputs=capture,
                output_path=onnx_path,
                use_bf16=True,
            )
        finally:
            torch.onnx.export = _orig_export

        logger.info("ONNX exported: %s", onnx_path)

        # N1.7 exports only the visual-language sequence axis as dynamic. The
        # state/action sequence (sa_embs) is static for a given policy/action
        # horizon. Derive profiles from ONNX so fixed dims stay fixed and only
        # named dynamic dims get ranges.
        vl_seq_len = int(capture.vl_embs.shape[1])
        min_shapes, opt_shapes, max_shapes = derive_shapes_with_hint(
            onnx_path,
            opt_seq_lens={"vl_seq_len": vl_seq_len},
            max_batch=1,
        )

        # Keep the previous generous upper bound for language-conditioned
        # inputs. derive_shapes_with_hint uses ~2x opt by default, which can be
        # too tight if the BT sends a longer instruction than export captured.
        for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
            shape = max_shapes.get(name)
            if shape and len(shape) > 1 and shape[1] != opt_shapes[name][1]:
                widened = list(shape)
                widened[1] = max(widened[1], 512)
                max_shapes[name] = tuple(widened)

        build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision="bf16",
            workspace_mb=workspace_mb,
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
        )

    logger.info("Cleaned up temporary ONNX export directory")


class GR00TInference:
    """Encapsulates GR00T policy loading, observation building, and inference."""

    logger = logging.getLogger("groot_inference")
    DEFAULT_EMBODIMENT_TAG = "new_embodiment"
    ROTATE_MAP = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def __init__(self):
        self.policy: Optional[Gr00tPolicy] = None
        self.robot: Optional[RobotClient] = None
        self._loaded_model_path: Optional[str] = None  # track cached policy path
        self._loaded_acceleration_mode: str = ACCELERATION_PYTORCH
        self._loaded_acceleration_engine_path: str = ""
        self.policy_info: dict = {
            "video": [],       # e.g. ["cam_left_head", "cam_left_wrist", ...]
            "state": [],       # e.g. ["arm_left", "arm_right"]
            "action": [],      # e.g. ["arm_left", "arm_right"]
            "language": [],    # e.g. ["annotation.human.task_description"]
        }
        self.robot_info: dict = {
            # Policy camera keys to emit in the GR00T observation.
            "cameras": [],
            # policy_camera_key -> RobotClient camera key. This lets a model
            # trained with cam_left_head consume a robot stream named
            # cam_head_left without changing the checkpoint metadata.
            "camera_sources": {},
            "joints": {},          # modality_key -> yaml_group mapping
            "camera_rotations": {},  # camera_name -> rotation_deg
        }

    @property
    def is_ready(self) -> bool:
        return self.policy is not None and self.robot is not None

    def load_policy(self, request) -> dict:
        """Load GR00T policy and create RobotClient for sensor data."""
        model_path = request.model_path
        robot_type = request.robot_type

        try:
            acceleration_mode, acceleration_engine_path, strict_acceleration = (
                self._resolve_acceleration_request(request, model_path)
            )
            if acceleration_mode == ACCELERATION_TENSORRT_FULL_PIPELINE:
                raise RuntimeError(
                    "acceleration_mode=tensorrt_full_pipeline is not wired into "
                    "Cyclo GR00T runtime yet; use tensorrt_dit or pytorch"
                )

            # Reuse cached policy if the same model is already loaded.
            # Only reload robot client (subscribers) on restart.
            if (
                self.policy is not None
                and self._loaded_model_path == model_path
                and self._loaded_acceleration_mode == acceleration_mode
                and self._loaded_acceleration_engine_path == acceleration_engine_path
            ):
                self.logger.info(
                    "Reusing cached policy: %s (acceleration=%s)",
                    model_path,
                    acceleration_mode,
                )
                if self.robot is not None:
                    self.robot.close()
                    self.robot = None
                self.init_policy_info()
                self.init_robot_info(robot_type)
                self.robot.wait_for_ready(timeout=10.0)
                return {
                    "success": True,
                    "message": "GR00T inference restarted (policy cached)",
                    "action_keys": list(self.policy_info["action"]),
                }

            if self.policy is not None:
                self.logger.info(
                    "Reloading GR00T policy due to model/runtime change "
                    "(model=%s, acceleration=%s)",
                    model_path,
                    acceleration_mode,
                )
                if self.robot is not None:
                    self.robot.close()
                    self.robot = None
                self.policy = None
                self._loaded_model_path = None
                self._loaded_acceleration_mode = ACCELERATION_PYTORCH
                self._loaded_acceleration_engine_path = ""

            self.logger.info(
                "Loading GR00T policy from: %s (acceleration=%s)",
                model_path,
                acceleration_mode,
            )
            self._sync_hf_token_for_gated_backbones()

            self.policy = Gr00tPolicy(
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                model_path=model_path,
                device="cuda",
            )

            self.init_policy_info()
            self.init_robot_info(robot_type)
            self.robot.wait_for_ready(timeout=10.0)

            if acceleration_mode == ACCELERATION_TENSORRT_DIT:
                trt_applied = self._enable_dit_tensorrt(
                    request=request,
                    engine_path=acceleration_engine_path,
                    strict=strict_acceleration,
                )
                if not trt_applied:
                    acceleration_mode = ACCELERATION_PYTORCH
                    acceleration_engine_path = ""
            else:
                self.logger.info(
                    "TensorRT acceleration disabled by request; using PyTorch Eager"
                )

            self._loaded_model_path = model_path
            self._loaded_acceleration_mode = acceleration_mode
            self._loaded_acceleration_engine_path = acceleration_engine_path

            return {
                "success": True,
                "message": "GR00T inference started",
                "action_keys": list(self.policy_info["action"]),
            }
        except Exception as e:
            self._loaded_model_path = None
            self._loaded_acceleration_mode = ACCELERATION_PYTORCH
            self._loaded_acceleration_engine_path = ""
            message = self._format_load_error(e)
            self.logger.error("Failed to start inference: %s", message, exc_info=True)
            return self.fail(message)

    @staticmethod
    def _normalize_acceleration_mode(value: str) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"", "none", "off", "false", "pytorch", "eager"}:
            return ACCELERATION_PYTORCH
        if mode in {"trt", "tensorrt", "tensorrt_dit", "dit", "dit_only"}:
            return ACCELERATION_TENSORRT_DIT
        if mode in {
            "trt_full_pipeline",
            "tensorrt_full_pipeline",
            "full_pipeline",
        }:
            return ACCELERATION_TENSORRT_FULL_PIPELINE
        return mode

    def _resolve_acceleration_request(
        self,
        request,
        model_path: str,
    ) -> tuple[str, str, bool]:
        raw_mode = str(getattr(request, "acceleration_mode", "") or "").strip()
        strict = bool(raw_mode)
        if raw_mode:
            mode = self._normalize_acceleration_mode(raw_mode)
        elif _env_flag("GROOT_TRT_ENABLED", default=False):
            mode = ACCELERATION_TENSORRT_DIT
        else:
            mode = ACCELERATION_PYTORCH

        if mode not in SUPPORTED_ACCELERATION_MODES:
            raise RuntimeError(
                f"Unsupported acceleration_mode={raw_mode!r}; expected one of "
                f"{sorted(SUPPORTED_ACCELERATION_MODES)}"
            )

        if mode == ACCELERATION_PYTORCH:
            return mode, "", strict

        engine_path = str(
            getattr(request, "acceleration_engine_path", "") or ""
        ).strip()
        if engine_path:
            if not os.path.isabs(engine_path):
                engine_path = os.path.join(model_path, engine_path)
        else:
            engine_path = os.path.join(model_path, "dit_model_bf16.trt")
        return mode, os.path.normpath(engine_path), strict

    def _enable_dit_tensorrt(
        self,
        request,
        engine_path: str,
        strict: bool,
    ) -> bool:
        try:
            engine_dir = os.path.dirname(os.path.abspath(engine_path))
            os.makedirs(engine_dir, exist_ok=True)
            if not os.path.exists(engine_path):
                raise FileNotFoundError(
                    f"TRT engine not found: {engine_path}. "
                    "Build the TensorRT engine before starting inference."
                )
            if os.path.getsize(engine_path) <= 0:
                raise RuntimeError(f"TRT engine is empty: {engine_path}")

            replace_dit_with_tensorrt(self.policy, engine_path)
            self.logger.info("DiT accelerated with TensorRT: %s", engine_path)
            return True
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"TensorRT acceleration requested but unavailable: {e}"
                ) from e
            self.logger.warning(
                "TensorRT acceleration unavailable, using PyTorch Eager: %s", e
            )
            return False

    def _sync_hf_token_for_gated_backbones(self) -> None:
        if sync_token_file is None:
            self.logger.warning("HF token sync helper unavailable")
            return
        if not sync_token_file():
            self.logger.warning(
                "No Hugging Face token found. GR00T N1.7 may need a token "
                "to download the gated Cosmos-Reason2-2B backbone unless it "
                "is already cached."
            )

    def _format_load_error(self, error: Exception) -> str:
        message = str(error)
        marker_text = message.lower()
        gated_markers = (
            "cannot access gated repo",
            "access to model nvidia/cosmos-reason2-2b is restricted",
            "401 client error",
        )
        if "cosmos-reason2-2b" in marker_text and any(
            marker in marker_text for marker in gated_markers
        ):
            return (
                "GR00T N1.7 needs access to the gated Hugging Face repo "
                "nvidia/Cosmos-Reason2-2B. Register a Hugging Face token for "
                "an approved account before loading the model, or pre-cache "
                "the Cosmos backbone in the shared Hugging Face cache."
            )
        return message

    def _build_dummy_observation(self, task_instruction: str = "") -> dict:
        """Build a real observation from robot sensors for TRT engine building."""
        images = self.robot.get_images(format="rgb")
        joints = self.robot.get_joint_positions()
        return self.preprocess(images, joints, task_instruction)

    def build_synthetic_observation(self, task_instruction: str = "") -> dict:
        """Build a model-schema observation without live robot sensors."""
        if self.policy is None:
            return self.fail("Policy is not loaded")

        image_h, image_w = self._model_image_hw()
        image_c = self._model_image_channels()
        video_t = self._modality_horizon("video")
        state_t = self._modality_horizon("state")
        language_t = self._modality_horizon("language")

        video_obs = {
            key: np.zeros(
                (1, video_t, image_h, image_w, image_c),
                dtype=np.uint8,
            )
            for key in self.policy_info["video"]
        }
        state_obs = {}
        for key in self.policy_info["state"]:
            dim = self._model_state_dim(key)
            vector = self._model_state_vector(key)
            state_obs[key] = (
                np.broadcast_to(vector, (1, state_t, dim))
                .astype(np.float32)
                .copy()
            )

        language_obs = {
            key: [[str(task_instruction)] * language_t]
            for key in self.policy_info["language"]
        }

        return {
            "video": video_obs,
            "state": state_obs,
            "language": language_obs,
        }

    def _embodiment_tag_value(self) -> str:
        tag = getattr(self.policy, "embodiment_tag", None)
        if hasattr(tag, "value"):
            return str(tag.value)
        if tag:
            return str(tag)
        return self.DEFAULT_EMBODIMENT_TAG

    @staticmethod
    def _entry_value(entry, name: str, default=None):
        if isinstance(entry, dict):
            return entry.get(name, default)
        return getattr(entry, name, default)

    def _policy_modality_entry(self, modality: str):
        configs = getattr(self.policy, "modality_configs", None)
        if configs is None:
            all_configs = self.policy.processor.get_modality_configs()
            configs = all_configs.get(self._embodiment_tag_value(), {})
        entry = configs.get(modality) if isinstance(configs, dict) else None
        if entry is None:
            raise RuntimeError(f"Model is missing modality config: {modality}")
        return entry

    def _modality_horizon(self, modality: str) -> int:
        entry = self._policy_modality_entry(modality)
        delta_indices = self._entry_value(entry, "delta_indices")
        if not delta_indices:
            raise RuntimeError(
                f"Model modality '{modality}' has no delta_indices"
            )
        return len(delta_indices)

    @staticmethod
    def _coerce_int_pair(value, label: str) -> tuple[int, int]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as e:
                raise RuntimeError(f"Invalid {label}: {value!r}") from e
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise RuntimeError(f"Invalid {label}: {value!r}")
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise RuntimeError(f"Invalid {label}: {value!r}")
        return first, second

    def _model_image_hw(self) -> tuple[int, int]:
        candidates = [
            getattr(getattr(self.policy, "processor", None), "image_target_size", None),
            getattr(
                getattr(getattr(self.policy, "processor", None), "config", None),
                "image_target_size",
                None,
            ),
            getattr(
                getattr(getattr(self.policy, "model", None), "config", None),
                "image_target_size",
                None,
            ),
        ]
        for candidate in candidates:
            if candidate:
                return self._coerce_int_pair(candidate, "image_target_size")
        raise RuntimeError(
            "Model image_target_size is unavailable; cannot build synthetic "
            "TRT observation without hardcoded image dimensions"
        )

    def _model_image_channels(self) -> int:
        processor = getattr(self.policy, "processor", None)
        qwen_processor = getattr(processor, "processor", None)
        candidates = [
            getattr(getattr(qwen_processor, "image_processor", None), "image_mean", None),
            getattr(getattr(qwen_processor, "image_processor", None), "image_std", None),
            getattr(getattr(qwen_processor, "video_processor", None), "image_mean", None),
            getattr(getattr(qwen_processor, "video_processor", None), "image_std", None),
            getattr(processor, "image_mean", None),
            getattr(processor, "image_std", None),
        ]
        for candidate in candidates:
            if isinstance(candidate, (list, tuple)) and candidate:
                return len(candidate)

        num_channels = getattr(
            getattr(getattr(self.policy, "model", None), "config", None),
            "num_channels",
            None,
        )
        if num_channels:
            channels = int(num_channels)
            if channels > 0:
                return channels

        raise RuntimeError(
            "Model image channel count is unavailable; cannot build synthetic "
            "TRT observation without hardcoded image channels"
        )

    def _state_norm_params(self, key: str) -> dict:
        processor = getattr(self.policy, "processor", None)
        state_action_processor = getattr(processor, "state_action_processor", None)
        norm_params = getattr(state_action_processor, "norm_params", None)
        tag = self._embodiment_tag_value()
        try:
            return norm_params[tag]["state"][key]
        except (TypeError, KeyError) as e:
            raise RuntimeError(
                f"Model statistics are missing state dimension for '{key}' "
                f"under embodiment '{tag}'"
            ) from e

    def _model_state_dim(self, key: str) -> int:
        params = self._state_norm_params(key)
        dim = params.get("dim")
        try:
            result = int(np.asarray(dim).item())
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid state dimension for '{key}': {dim!r}") from e
        if result <= 0:
            raise RuntimeError(f"Invalid state dimension for '{key}': {result}")
        return result

    def _model_state_vector(self, key: str) -> np.ndarray:
        params = self._state_norm_params(key)
        dim = self._model_state_dim(key)
        mean = params.get("mean")
        if mean is None:
            return np.zeros((dim,), dtype=np.float32)

        vector = np.asarray(mean, dtype=np.float32)
        if vector.size != dim:
            raise RuntimeError(
                f"Model state mean for '{key}' has {vector.size} values, "
                f"expected {dim}"
            )
        return vector.reshape(dim)

    def init_policy_info(self) -> None:
        """Read video/state/action/language keys from the loaded policy."""
        mc = getattr(self.policy, "modality_configs", None)
        if mc is None:
            mc = self.policy.processor.get_modality_configs().get(
                self.DEFAULT_EMBODIMENT_TAG, {}
            )

        for modality in ("video", "state", "action", "language"):
            if modality not in mc:
                self.policy_info[modality] = []
                continue
            entry = mc[modality]
            self.policy_info[modality] = self._entry_value(
                entry,
                "modality_keys",
                [],
            )

        self.logger.info("Policy info: %s", self.policy_info)

    def init_robot_info(self, robot_type: str) -> None:
        """Create RobotClient and resolve active cameras/joints from YAML."""
        self.robot = RobotClient(robot_type)
        cam_config = self.robot._config.get("cameras", {})
        available_cameras = set(self.robot.camera_names)

        camera_sources = resolve_camera_feature_sources(
            self.policy_info["video"],
            available_cameras,
        )

        self.robot_info["cameras"] = list(camera_sources.keys())
        self.robot_info["camera_sources"] = camera_sources
        self.robot_info["camera_rotations"] = {
            name: cfg.get("rotation_deg", 0)
            for name, cfg in cam_config.items()
            if cfg.get("rotation_deg", 0) != 0
        }

        joints = {}
        for group in self.robot.joint_group_names:
            if "follower" not in group:
                continue
            modality_key = group.removeprefix("follower_")
            if modality_key in self.policy_info["state"]:
                joints[modality_key] = group
        self.robot_info["joints"] = joints

        # Sensor-backed state modalities. Training runs have used both
        # ``mobile`` and ``odometry`` for the same 3-dim base velocity slice,
        # while robot_client keeps /odom as a sensor instead of a joint group.
        # Bridge either policy key to the odom sensor here.
        sensor_states = {}
        sensors_cfg = self.robot._config.get("sensors", {})
        if "odom" in sensors_cfg:
            for modality_key in ("mobile", "odometry"):
                if modality_key in self.policy_info["state"]:
                    sensor_states[modality_key] = "odom"
        self.robot_info["sensor_states"] = sensor_states

        self.logger.info("Robot info: %s", self.robot_info)

    def get_action_chunk(self, request) -> dict:
        """Build observation from RobotClient, run inference, return action chunk."""
        if not self.is_ready:
            return self.fail("Not in inference mode")

        try:
            images = self.robot.get_images(format="rgb")
            joints = self.robot.get_joint_positions()
            task = request.task_instruction

            observation = self.preprocess(images, joints, task)
            if "success" in observation:
                return observation

            t0 = time.monotonic()
            self.logger.info("Running GR00T inference...")
            action, info = self.policy.get_action(observation)
            self.logger.info("GR00T inference completed in %.3fs", time.monotonic() - t0)
            return self.postprocess_action(action)

        except Exception as e:
            self.logger.error("Inference failed: %s", e, exc_info=True)
            return self.fail(str(e))

    def preprocess(self, images: dict, joints: dict, task: str) -> dict:
        """Build observation dict from raw sensor data. Returns fail dict on error."""
        if not images or not joints:
            return self.fail("No recent observations from sensors")

        video_obs = {}
        for cam_key in self.robot_info["cameras"]:
            source_key = self.robot_info.get("camera_sources", {}).get(cam_key, cam_key)
            img = images.get(source_key)
            if img is None:
                return self.fail(f"Missing camera: {source_key} for {cam_key}")
            rotation = self.robot_info["camera_rotations"].get(source_key)
            if rotation and rotation in self.ROTATE_MAP:
                img = cv2.rotate(img, self.ROTATE_MAP[rotation])
            video_obs[cam_key] = img[np.newaxis, np.newaxis, ...]  # (1,1,H,W,C)

        state_obs = {}
        for modality_key, yaml_group in self.robot_info["joints"].items():
            positions = joints.get(yaml_group)
            if positions is None or len(positions) == 0:
                return self.fail(f"Missing joint group: {modality_key}")
            state_obs[modality_key] = positions[np.newaxis, np.newaxis, :]  # (1,1,D)

        # Sensor-backed state modalities (e.g. mobile ← odom).
        for modality_key, sensor_name in self.robot_info.get("sensor_states", {}).items():
            if sensor_name == "odom":
                odom = self.robot.get_odom()
                if odom is None:
                    return self.fail(f"Missing sensor: {sensor_name}")
                vec = np.array([
                    float(odom["linear_velocity"][0]),
                    float(odom["linear_velocity"][1]),
                    float(odom["angular_velocity"][2]),
                ], dtype=np.float32)
                state_obs[modality_key] = vec[np.newaxis, np.newaxis, :]  # (1,1,3)
            else:
                return self.fail(f"Unsupported sensor modality: {sensor_name}")

        language_obs = {key: [[task]] for key in self.policy_info["language"]}

        return {
            "video": video_obs,
            "state": state_obs,
            "language": language_obs,
        }

    def postprocess_action(self, action: dict) -> dict:
        chunks = [
            action[key][0]  # remove batch dim
            for key in self.policy_info["action"]
            if key in action and isinstance(action[key], np.ndarray)
        ]
        if not chunks:
            return self.fail("No action output from policy")

        chunk = np.concatenate(chunks, axis=1)  # (T, D_total)
        T, D = chunk.shape
        self.logger.info("Action chunk: T=%d, D=%d", T, D)
        return {
            "success": True,
            # Keep as numpy — zenoh_ros2_sdk's publisher uses .view() for fast
            # CDR encoding and crashes on plain lists ('list' object has no
            # attribute 'view'). inference_server._publish_chunk passes this
            # straight through.
            "action_chunk": np.asarray(chunk.flatten(), dtype=np.float64),
            "chunk_size": T,
            "action_dim": D,
        }

    def cleanup(self) -> None:
        """Release robot resources. Policy is kept cached for fast restart."""
        if self.robot is not None:
            self.robot.close()
            self.robot = None

        self.policy_info = {k: [] for k in self.policy_info}
        self.robot_info = {
            "cameras": [],
            "joints": {},
            "camera_rotations": {},
        }

    @staticmethod
    def fail(message: str) -> dict:
        return {"success": False, "message": message}


def create_engine() -> GR00TInference:
    """Factory used by the shared Engine process."""
    return GR00TInference()
