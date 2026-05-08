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

"""
GR00T Training - Finetuning logic for GR00T N1.6.

Module-level functions that use RobotServiceServer public API only:
  - server.report_progress()
  - server.stop_requested.is_set()
  - server.progress
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger("groot_training")

# Module-level state for logging interceptor
_original_log_info: Optional[Callable] = None


def run_training(server, request: Any) -> None:
    """Run GR00T finetuning. Called by RobotServiceServer in background thread."""
    start_time = time.time()
    max_steps = getattr(request, "steps", 10000) or 10000
    server.report_progress(total_steps=max_steps, step=0, state="training")

    logger.info("Starting GR00T N1.6 finetuning...")

    try:
        _execute_training(server, request, start_time)
    finally:
        _restore_logging()

    if server.stop_requested.is_set():
        logger.info("Finetuning stopped by user")
    else:
        logger.info("Finetuning completed successfully")


def _build_finetune_config(request: Any):
    """Build GR00T FinetuneConfig from request."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment_tag_str = getattr(request, "policy_type", "new_embodiment")
    if not embodiment_tag_str:
        embodiment_tag_str = "new_embodiment"

    try:
        embodiment_tag = EmbodimentTag(embodiment_tag_str)
    except ValueError:
        logger.warning(f"Unknown embodiment tag '{embodiment_tag_str}', using NEW_EMBODIMENT")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    dataset_path = getattr(request, "dataset_path", "")
    if not dataset_path:
        raise ValueError("dataset_path is required")

    base_model = "nvidia/GR00T-N1.6-3B"

    workspace_dir = os.environ.get("GROOT_WORKSPACE", "/workspace")
    checkpoints_dir = f"{workspace_dir}/checkpoints"

    if hasattr(request, "output_dir") and request.output_dir:
        output_dir = request.output_dir
        if not output_dir.startswith("/"):
            output_dir = f"{checkpoints_dir}/{output_dir}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{checkpoints_dir}/groot_{timestamp}"

    max_steps = getattr(request, "steps", 10000) or 10000
    batch_size = getattr(request, "batch_size", 64) or 64
    learning_rate = getattr(request, "learning_rate", 1e-4) or 1e-4
    save_steps = getattr(request, "save_freq", 500) or 500
    use_wandb = bool(getattr(request, "wandb_project", ""))

    from gr00t.configs.finetune_config import FinetuneConfig

    finetune_config = FinetuneConfig(
        base_model_path=base_model,
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        output_dir=output_dir,
        max_steps=max_steps,
        global_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=save_steps,
        use_wandb=use_wandb,
    )

    return finetune_config


def _execute_training(server, request: Any, start_time: float) -> None:
    """Execute GR00T finetuning using Isaac-GR00T API directly."""
    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run

    _setup_logging_interceptor(server, start_time)

    ft_config = _build_finetune_config(request)
    server.report_progress(total_steps=ft_config.max_steps)
    embodiment_tag = ft_config.embodiment_tag.value

    # Register custom embodiment config from dataset's modality.json if needed
    _ensure_embodiment_config(embodiment_tag, ft_config.dataset_path)

    logger.info(f"Starting GR00T finetuning with config:")
    logger.info(f"  Dataset: {ft_config.dataset_path}")
    logger.info(f"  Embodiment: {embodiment_tag}")
    logger.info(f"  Output: {ft_config.output_dir}")
    logger.info(f"  Max steps: {ft_config.max_steps}")
    logger.info(f"  Batch size: {ft_config.global_batch_size}")
    logger.info(f"  Learning rate: {ft_config.learning_rate}")

    config = get_default_config().load_dict({
        "data": {
            "download_cache": False,
            "datasets": [{
                "dataset_paths": [ft_config.dataset_path],
                "mix_ratio": 1.0,
                "embodiment_tag": embodiment_tag,
            }],
        }
    })
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = True
    config.model.use_flash_attention = True
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    # Use decord for video loading (torchcodec not available on ARM64/Jetson)
    config.data.video_backend = getattr(ft_config, "video_backend", None) or "decord"

    run(config)


def _setup_logging_interceptor(server, start_time: float) -> None:
    """Setup logging interceptor to capture training progress from GR00T logs."""
    global _original_log_info
    _original_log_info = logging.Logger.info

    patterns = {
        "step": re.compile(r"'global_step':\s*(\d+)"),
        "loss": re.compile(r"'loss':\s*([\d.]+)"),
        "lr": re.compile(r"'learning_rate':\s*([\d.e+-]+)"),
        "grad_norm": re.compile(r"'grad_norm':\s*([\d.]+)"),
    }

    alt_patterns = {
        "step": re.compile(r"step\s*[=:]\s*(\d+)", re.IGNORECASE),
        "loss": re.compile(r"loss\s*[=:]\s*([\d.]+)", re.IGNORECASE),
        "lr": re.compile(r"lr\s*[=:]\s*([\d.e+-]+)", re.IGNORECASE),
    }

    original = _original_log_info

    def interceptor(self_logger, msg, *args, **kwargs):
        original(self_logger, msg, *args, **kwargs)

        try:
            log_msg = str(msg) % args if args else str(msg)

            if server.stop_requested.is_set():
                return

            for key, pattern in patterns.items():
                match = pattern.search(log_msg)
                if match:
                    value = match.group(1)
                    if key == "step":
                        server.report_progress(step=int(value))
                    elif key == "loss":
                        server.report_progress(loss=float(value))
                    elif key == "lr":
                        server.report_progress(learning_rate=float(value))
                    elif key == "grad_norm":
                        server.report_progress(gradient_norm=float(value))

            for key, pattern in alt_patterns.items():
                match = pattern.search(log_msg)
                if match:
                    value = match.group(1)
                    if key == "step":
                        server.report_progress(step=int(value))
                    elif key == "loss":
                        server.report_progress(loss=float(value))
                    elif key == "lr":
                        server.report_progress(learning_rate=float(value))

            elapsed = time.time() - start_time
            server.report_progress(elapsed_seconds=elapsed)
            step = server.progress.step
            if step > 0:
                total = server.progress.total_steps
                time_per_step = elapsed / step
                server.report_progress(eta_seconds=(total - step) * time_per_step)

        except Exception:
            pass

    logging.Logger.info = interceptor


def _restore_logging() -> None:
    """Restore original logging."""
    global _original_log_info
    if _original_log_info is not None:
        logging.Logger.info = _original_log_info
        _original_log_info = None


def _ensure_embodiment_config(embodiment_tag: str, dataset_path: str) -> None:
    """Register custom embodiment config from dataset's modality.json if not in registry."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    if embodiment_tag in MODALITY_CONFIGS:
        return

    import json
    from pathlib import Path
    from gr00t.data.types import ModalityConfig

    modality_path = Path(dataset_path) / "meta" / "modality.json"
    if not modality_path.exists():
        raise ValueError(
            f"Embodiment '{embodiment_tag}' not in registry and no modality.json found at {modality_path}"
        )

    with open(modality_path) as f:
        modality_json = json.load(f)

    configs = {}
    if "video" in modality_json:
        configs["video"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=list(modality_json["video"].keys()),
        )
    if "state" in modality_json:
        configs["state"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=list(modality_json["state"].keys()),
        )
    if "action" in modality_json:
        configs["action"] = ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=list(modality_json["action"].keys()),
        )
    if "annotation" in modality_json:
        configs["language"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=[f"annotation.{k}" for k in modality_json["annotation"].keys()],
        )

    MODALITY_CONFIGS[embodiment_tag] = configs
    logger.info(f"Registered custom embodiment config '{embodiment_tag}' from {modality_path}")


def cleanup_training() -> None:
    """Cleanup training resources."""
    _restore_logging()
