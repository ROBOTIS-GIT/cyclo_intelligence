#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""LeRobot optional policy optimization hook.

LeRobot currently runs through its native PyTorch policy path. This mixin exists
so optional runtime optimization (TensorRT, ONNX Runtime, torch.compile, etc.)
has a clear class boundary without changing the engine lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

import torch


logger = logging.getLogger("lerobot_engine")


class OptimizationMixin:
    """Optional policy optimization extension point."""

    def _apply_policy_optimization(self, model_path: str, request: Any) -> None:
        """Attach optional optimizers after policy load.

        No-op except for FastWAM, which needs CPU offload to fit on a 24GB GPU.
        """
        if getattr(getattr(self._policy, "config", None), "type", "") == "fastwam":
            self._offload_fastwam(request)
            return
        logger.debug("No LeRobot optimizer configured for %s", model_path)

    def _offload_fastwam(self, request: Any) -> None:
        """Keep FastWAM's ~11GB text encoder on the CPU so the rest fits on a 24GB GPU.

        The encoder only turns the instruction into a context. Cache that context and
        refresh it only when the runtime instruction changes.
        """
        policy = self._policy
        model = policy.model
        cfg = policy.config
        device = self._device

        def normalize_task(value: Any) -> str:
            if isinstance(value, (list, tuple)):
                value = value[0] if value else ""
            return str(value or "").strip()

        task = normalize_task(getattr(request, "task_instruction", ""))
        if not task:
            raise RuntimeError(
                "FastWAM needs the task instruction at load time: the text encoder "
                "uses it to prepare the inference context."
            )

        template = getattr(cfg, "prompt_template", None)

        def encode_task(task_text: str) -> tuple[torch.Tensor, torch.Tensor]:
            prompt = template.format(task=task_text) if template else task_text
            previous_device = model.device
            model.device = torch.device("cpu")
            try:
                encoded, encoded_mask = model.encode_prompt([prompt])
            finally:
                model.device = previous_device
            return encoded.detach().to(device), encoded_mask.detach().to(device)

        # The initial policy load is pinned to CPU, so prompt encoding happens
        # before the non-text modules are moved to the runtime device.
        context, context_mask = encode_task(task)

        # Everything except the text encoder goes to the GPU.
        for name, child in model.named_children():
            if name != "text_encoder":
                child.to(device)
        if hasattr(model, "vae"):
            model.vae.to(device)
        model.device = torch.device(device)
        cfg.device = str(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                "FastWAM offload done, VRAM %.1f GB, text encoder on CPU",
                torch.cuda.memory_allocated() / 1e9,
            )

        proprio_dim = getattr(cfg, "proprio_dim", None)
        real_predict = policy.predict_action_chunk

        def predict(batch, *args, **kwargs):
            nonlocal task, context, context_mask
            b = dict(batch)
            incoming_task = normalize_task(b.get("task"))
            if incoming_task and incoming_task != task:
                context, context_mask = encode_task(incoming_task)
                task = incoming_task
                logger.info("FastWAM prompt context refreshed")
            b.pop("task", None)
            b.pop("prompt", None)
            b["context"], b["context_mask"] = context, context_mask
            state = b.get("proprio", b.get("observation.state"))
            if state is not None and proprio_dim is not None:
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                dim = state.shape[-1]
                if dim < proprio_dim:
                    pad = torch.zeros(
                        *state.shape[:-1], proprio_dim - dim,
                        dtype=state.dtype, device=state.device,
                    )
                    state = torch.cat([state, pad], dim=-1)
                elif dim > proprio_dim:
                    state = state[..., :proprio_dim]
                b["proprio"] = state
            return real_predict(b, *args, **kwargs)

        policy.predict_action_chunk = predict
