#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from engine_process.protocol import EngineCommandResponse  # noqa: E402
from main_runtime.inference_requester import InferenceRequester  # noqa: E402


class FakeEngineClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def call(self, request, timeout_s: float):
        self.calls.append((request, timeout_s))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class InferenceRequesterTests(unittest.TestCase):
    def test_load_policy_default_timeout_is_long_enough_for_model_load(self) -> None:
        client = FakeEngineClient(
            [EngineCommandResponse(success=True, seq_id=1)]
        )
        requester = InferenceRequester(client)

        requester.load_policy(type("Req", (), {"model_path": "/models/policy"})())

        self.assertEqual(client.calls[0][1], 300.0)

    def test_get_action_uses_monotonic_seq_id(self) -> None:
        client = FakeEngineClient(
            [
                EngineCommandResponse(
                    success=True,
                    seq_id=1,
                    action_list=[1.0, 2.0, 3.0, 4.0],
                    chunk_size=2,
                    action_dim=2,
                )
            ]
        )
        requester = InferenceRequester(client, get_action_timeout_s=2.5)

        response = requester.get_action("open drawer")

        self.assertTrue(response.success)
        self.assertEqual(response.seq_id, 1)
        request, timeout_s = client.calls[0]
        self.assertEqual(request.seq_id, 1)
        self.assertEqual(request.task_instruction, "open drawer")
        self.assertEqual(timeout_s, 2.5)
        self.assertFalse(requester.has_pending_get_action())

    def test_timeout_clears_in_flight_and_next_request_advances_seq(self) -> None:
        client = FakeEngineClient(
            [
                TimeoutError("slow model"),
                EngineCommandResponse(success=True, seq_id=2),
            ]
        )
        requester = InferenceRequester(client, get_action_timeout_s=1.0)

        first = requester.get_action("pick")
        second = requester.get_action("place")

        self.assertFalse(first.success)
        self.assertIn("timed out", first.message)
        self.assertTrue(second.success)
        self.assertEqual([call[0].seq_id for call in client.calls], [1, 2])
        self.assertFalse(requester.has_pending_get_action())

    def test_stale_response_is_discarded_by_seq_id(self) -> None:
        client = FakeEngineClient(
            [
                EngineCommandResponse(
                    success=True,
                    seq_id=7,
                    action_list=[1.0],
                    chunk_size=1,
                    action_dim=1,
                )
            ]
        )
        requester = InferenceRequester(client, get_action_timeout_s=5.0)

        response = requester.get_action("current request")

        self.assertFalse(response.success)
        self.assertIn("stale", response.message)
        self.assertEqual(response.action_list, [])
        self.assertFalse(requester.has_pending_get_action())


if __name__ == "__main__":
    unittest.main()
