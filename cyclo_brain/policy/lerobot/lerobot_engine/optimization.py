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


logger = logging.getLogger("lerobot_engine")


class OptimizationMixin:
    """Optional policy optimization extension point."""

    def _apply_policy_optimization(self, model_path: str, request: Any) -> None:
        """Attach optional optimizers after policy load.

        No-op by default. Future optimizers may mutate ``self._policy`` to wrap
        or replace backend-specific model internals.
        """
        logger.debug("No LeRobot optimizer configured for %s", model_path)
