#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.tt_rtc import (  # noqa: E402
    TTRTCContractError,
    load_tt_rtc_capability,
    parse_tt_rtc_request,
)


def _manifest(*, include_rlt: bool = True) -> dict:
    payload = {
        "schema": "cyclo.training-time-rtc/v1",
        "training_time_rtc": {
            "trained": True,
            "action_horizon": 16,
            "action_dimension": 19,
            "action_hz": 15.0,
            "max_delay_steps": 6,
            "delay_sampling": {
                "type": "uniform_integer",
                "min_inclusive": 0,
                "max_inclusive": 6,
            },
            "prefix_input": "ground_truth_clean_action",
            "loss_region": "postfix_only",
            "per_action_timestep": True,
            "flow_convention": {
                "noise_endpoint": 0.0,
                "clean_endpoint": 1.0,
                "velocity_target": "action_minus_noise",
            },
        },
    }
    if include_rlt:
        payload["rlt"] = {
            "chunk_length": 10,
            "reference_horizon": 16,
            "reference_slice": "[d:d+10]",
        }
    return payload


class TTRTCRequestTests(unittest.TestCase):
    def test_normal_modes_do_not_create_a_tt_rtc_request(self) -> None:
        for mode in ("", "sync", "async"):
            with self.subTest(mode=mode):
                self.assertIsNone(
                    parse_tt_rtc_request(SimpleNamespace(action_request_mode=mode))
                )

    def test_exact_six_by_nineteen_prefix_is_accepted(self) -> None:
        request = parse_tt_rtc_request(
            SimpleNamespace(
                action_request_mode="tt_rtc",
                rtc_delay_steps=6,
                rtc_action_dim=19,
                rtc_prefix_action_list=[0.25] * (6 * 19),
            )
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.delay_steps, 6)
        self.assertEqual(len(request.prefix_actions), 6)
        self.assertTrue(all(len(row) == 19 for row in request.prefix_actions))

    def test_bootstrap_requires_an_explicit_empty_prefix(self) -> None:
        request = parse_tt_rtc_request(
            SimpleNamespace(
                action_request_mode="tt_rtc",
                rtc_delay_steps=0,
                rtc_action_dim=19,
                rtc_prefix_action_list=[],
            )
        )

        self.assertEqual(request.prefix_actions, ())

    def test_bad_delay_dimension_length_and_nonfinite_values_fail_closed(self) -> None:
        bad_requests = (
            SimpleNamespace(
                action_request_mode="tt_rtc",
                rtc_delay_steps=7,
                rtc_action_dim=19,
                rtc_prefix_action_list=[0.0] * (7 * 19),
            ),
            SimpleNamespace(
                action_request_mode="tt_rtc",
                rtc_delay_steps=6,
                rtc_action_dim=22,
                rtc_prefix_action_list=[0.0] * (6 * 22),
            ),
            SimpleNamespace(
                action_request_mode="tt_rtc",
                rtc_delay_steps=6,
                rtc_action_dim=19,
                rtc_prefix_action_list=[0.0] * (6 * 19 - 1),
            ),
            SimpleNamespace(
                action_request_mode="tt_rtc",
                rtc_delay_steps=1,
                rtc_action_dim=19,
                rtc_prefix_action_list=[float("nan")] + [0.0] * 18,
            ),
        )
        for request in bad_requests:
            with self.subTest(request=request):
                with self.assertRaises(TTRTCContractError):
                    parse_tt_rtc_request(request)


class TTRTCManifestTests(unittest.TestCase):
    def test_dedicated_manifest_and_rlt_binding_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_rtc_manifest.json"
            path.write_text(json.dumps(_manifest()), encoding="utf-8")

            capability = load_tt_rtc_capability(root, require_rlt=True)

            self.assertEqual(capability.source, path)

    def test_explicit_config_object_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            path.write_text(json.dumps(_manifest(include_rlt=False)), encoding="utf-8")

            capability = load_tt_rtc_capability(root)

            self.assertEqual(capability.source, path)

    def test_legacy_rtc_training_prefix_flag_is_not_a_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({"rtc_training_prefix_steps": 6}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TTRTCContractError, "capability is missing"):
                load_tt_rtc_capability(root)

    def test_rlt_mode_requires_the_shifted_reference_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tt_rtc_manifest.json").write_text(
                json.dumps(_manifest(include_rlt=False)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TTRTCContractError, "manifest rlt"):
                load_tt_rtc_capability(root, require_rlt=True)


if __name__ == "__main__":
    unittest.main()
