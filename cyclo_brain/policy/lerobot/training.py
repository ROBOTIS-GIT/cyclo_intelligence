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
LeRobot Training - Training logic for LeRobot.

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
from typing import Any, Callable, Optional

logger = logging.getLogger("lerobot_training")

# Module-level state for logging interceptor
_original_log_info: Optional[Callable] = None


def run_training(server, request: Any) -> None:
    """Run LeRobot training. Called by RobotServiceServer in background thread."""
    start_time = time.time()
    max_steps = getattr(request, "steps", 100000) or 100000
    server.report_progress(total_steps=max_steps, step=0, state="training")

    logger.info("Starting LeRobot training...")

    args = _build_training_args(server, request)
    logger.info(f"Training args: {args}")

    try:
        _execute_training(server, args, start_time)
    finally:
        _restore_logging()

    if server.stop_requested.is_set():
        logger.info("Training stopped by user")
    else:
        logger.info("Training completed successfully")


def _build_training_args(server, request: Any) -> list[str]:
    """Build training arguments from request."""
    args = []

    policy_type = getattr(request, "policy_type", "act")
    args.append(f"--policy.type={policy_type}")

    dataset_path = getattr(request, "dataset_path", "")
    if dataset_path:
        args.append(f"--dataset.repo_id={dataset_path}")

    args.append("--policy.device=cuda")

    workspace_dir = os.environ.get("LEROBOT_WORKSPACE", "/workspace")
    checkpoints_dir = f"{workspace_dir}/checkpoints"

    if hasattr(request, "output_dir") and request.output_dir:
        output_dir = request.output_dir
        if not output_dir.startswith("/"):
            output_dir = f"{checkpoints_dir}/{output_dir}"
        args.append(f"--output_dir={output_dir}")
    else:
        args.append(f"--output_dir={checkpoints_dir}")

    steps = getattr(request, "steps", 0)
    if steps and int(steps) > 0:
        args.append(f"--steps={steps}")
        server.report_progress(total_steps=int(steps))

    batch_size = getattr(request, "batch_size", 0)
    if batch_size and int(batch_size) > 0:
        args.append(f"--batch_size={batch_size}")

    learning_rate = getattr(request, "learning_rate", 0)
    if learning_rate and float(learning_rate) > 0:
        args.append(f"--optimizer.lr={learning_rate}")

    eval_freq = getattr(request, "eval_freq", 0)
    if eval_freq and int(eval_freq) > 0:
        args.append(f"--eval_freq={eval_freq}")

    log_freq = getattr(request, "log_freq", 0)
    if log_freq and int(log_freq) > 0:
        args.append(f"--log_freq={log_freq}")

    save_freq = getattr(request, "save_freq", 0)
    if save_freq and int(save_freq) > 0:
        args.append(f"--save_freq={save_freq}")
    else:
        args.append("--save_freq=500")

    wandb_project = getattr(request, "wandb_project", "")
    if wandb_project:
        args.append(f"--wandb.project={wandb_project}")

    push_to_hub = getattr(request, "push_to_hub", False)
    if not push_to_hub:
        args.append("--policy.push_to_hub=false")

    tolerance_s = getattr(request, "tolerance_s", 0.0)
    if tolerance_s and float(tolerance_s) > 0:
        args.append(f"--tolerance_s={tolerance_s}")
    else:
        args.append("--tolerance_s=0.04")

    return args


def _execute_training(server, args: list[str], start_time: float) -> None:
    """Execute LeRobot training with progress monitoring."""
    import draccus

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train as lerobot_train

    _setup_logging_interceptor(server, start_time)

    cfg = draccus.parse(TrainPipelineConfig, None, args=args)
    lerobot_train(cfg)


def _setup_logging_interceptor(server, start_time: float) -> None:
    """Setup logging interceptor to capture training progress from LeRobot logs."""
    global _original_log_info
    _original_log_info = logging.Logger.info

    patterns = {
        "step": re.compile(r"step:([\d.]+)([KMB]?)"),
        "loss": re.compile(r"loss:([\d.]+)"),
        "grad": re.compile(r"grdn:([\d.]+)"),
        "lr": re.compile(r"lr:([\d.e+-]+)"),
        "epoch": re.compile(r"epch:([\d.]+)"),
    }

    def parse_number_with_suffix(value_str, suffix):
        value = float(value_str)
        if suffix == "K":
            value *= 1000
        elif suffix == "M":
            value *= 1000000
        elif suffix == "B":
            value *= 1000000000
        return int(value)

    original = _original_log_info

    def interceptor(self_logger, msg, *args, **kwargs):
        original(self_logger, msg, *args, **kwargs)

        try:
            log_msg = str(msg) % args if args else str(msg)

            if "step:" in log_msg:
                for key, pattern in patterns.items():
                    match = pattern.search(log_msg)
                    if match:
                        value = match.group(1)
                        if key == "step":
                            suffix_str = match.group(2) if len(match.groups()) > 1 else ""
                            parsed_step = parse_number_with_suffix(value, suffix_str)
                            server.report_progress(step=parsed_step)
                        elif key == "loss":
                            server.report_progress(loss=float(value))
                        elif key == "grad":
                            server.report_progress(gradient_norm=float(value))
                        elif key == "lr":
                            server.report_progress(learning_rate=float(value))
                        elif key == "epoch":
                            server.report_progress(epoch=float(value))

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


def cleanup_training() -> None:
    """Cleanup training resources."""
    _restore_logging()
