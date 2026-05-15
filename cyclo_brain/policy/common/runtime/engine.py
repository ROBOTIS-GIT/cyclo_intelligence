#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""InferenceEngine abstract base class.

Each opensource policy backend (LeRobot, GR00T, OpenVLA, ...) implements
this contract in a backend engine package such as ``lerobot_engine`` or
``groot_engine``. The common Engine process dynamically imports the engine
module, calls ``create_engine()``, and routes internal EngineCommand service
calls through the methods below.

The engine owns the parts that genuinely vary per policy:

- Model loading (path → in-memory policy + processors).
- Observation construction from RobotClient sensor reads.
- One synchronous inference call returning a ``(T, D)`` action chunk.
- Resource cleanup.

Everything else - external service handling, request timeouts, action-list
buffering, and robot command publishing - lives in ``common/runtime/`` and is
shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class InferenceEngine(ABC):
    """Per-policy inference backend invoked by the common Engine process.

    Result dict shapes (kept as plain dicts so engines never need to
    import zenoh_ros2_sdk message classes):

    ``load_policy`` →
        ``{"success": bool, "message": str, "action_keys": list[str]}``

    ``get_action_chunk`` (success) →
        ``{"success": True,
           "action_chunk": np.ndarray flat (T*D,) float64,
           "chunk_size": int (T),
           "action_dim": int (D)}``

    ``get_action_chunk`` (failure) →
        ``{"success": False, "message": str}``

    ``action_chunk`` is kept as a flat ``np.ndarray`` for compatibility with
    backend implementations that already return numpy chunks. The Engine
    process converts it to a plain ``float64[] action_list`` for the internal
    service response.
    """

    @abstractmethod
    def load_policy(self, request: Any) -> Dict[str, Any]:
        """Load policy weights + bring up RobotClient.

        ``request`` carries the InferenceCommand srv body:
        ``request.model_path``, ``request.robot_type``,
        ``request.task_instruction``.
        """

    @abstractmethod
    def get_action_chunk(self, request: Any) -> Dict[str, Any]:
        """Build observation from RobotClient, run inference once.

        ``request`` is a ``SimpleNamespace`` with ``.task_instruction``
        only — model_path / robot_type are baked in by ``load_policy``.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release the RobotClient + drop policy refs.

        Called on UNLOAD. Should be safe to call before ``load_policy``
        and idempotent on repeat calls so the Engine process can use it as a
        catch-all teardown.
        """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True once policy + robot client are wired and inference is callable."""
