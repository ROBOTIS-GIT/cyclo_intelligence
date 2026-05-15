#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from engine_process.protocol import (  # noqa: E402
    CMD_GET_ACTION,
    CMD_LOAD_POLICY,
    CMD_UNLOAD_POLICY,
    EngineCommandRequest,
)
from engine_process.worker import EngineWorker  # noqa: E402


class FakeEngine:
    def __init__(self) -> None:
        self.loaded_with = None
        self.action_requested_with = None
        self.cleaned = False

    @property
    def is_ready(self) -> bool:
        return self.loaded_with is not None and not self.cleaned

    def load_policy(self, request) -> dict:
        self.loaded_with = request
        self.cleaned = False
        return {
            "success": True,
            "message": f"loaded {request.model_path}",
            "action_keys": ["arm", "gripper"],
        }

    def get_action_chunk(self, request) -> dict:
        self.action_requested_with = request
        return {
            "success": True,
            "message": "ok",
            "action_chunk": np.arange(6, dtype=np.float64),
            "chunk_size": 2,
            "action_dim": 3,
        }

    def cleanup(self) -> None:
        self.cleaned = True


class EngineWorkerTests(unittest.TestCase):
    def test_load_policy_delegates_to_engine(self) -> None:
        engine = FakeEngine()
        worker = EngineWorker(engine)

        response = worker.handle(
            EngineCommandRequest(
                command=CMD_LOAD_POLICY,
                seq_id=11,
                model_path="/models/policy",
                robot_type="ffw",
                task_instruction="pick",
            )
        )

        self.assertTrue(response.success)
        self.assertEqual(response.seq_id, 11)
        self.assertEqual(response.action_keys, ["arm", "gripper"])
        self.assertEqual(engine.loaded_with.model_path, "/models/policy")
        self.assertEqual(engine.loaded_with.robot_type, "ffw")

    def test_get_action_returns_flat_action_list(self) -> None:
        engine = FakeEngine()
        worker = EngineWorker(engine)
        worker.handle(
            EngineCommandRequest(
                command=CMD_LOAD_POLICY,
                seq_id=1,
                model_path="/models/policy",
                robot_type="ffw",
            )
        )

        response = worker.handle(
            EngineCommandRequest(
                command=CMD_GET_ACTION,
                seq_id=12,
                task_instruction="move",
            )
        )

        self.assertTrue(response.success)
        self.assertEqual(response.seq_id, 12)
        self.assertEqual(response.chunk_size, 2)
        self.assertEqual(response.action_dim, 3)
        self.assertEqual(response.action_list, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertIsInstance(engine.action_requested_with, SimpleNamespace)
        self.assertEqual(engine.action_requested_with.task_instruction, "move")

    def test_unload_is_idempotent_cleanup(self) -> None:
        engine = FakeEngine()
        worker = EngineWorker(engine)

        response = worker.handle(
            EngineCommandRequest(command=CMD_UNLOAD_POLICY, seq_id=13)
        )

        self.assertTrue(response.success)
        self.assertEqual(response.seq_id, 13)
        self.assertTrue(engine.cleaned)


if __name__ == "__main__":
    unittest.main()
