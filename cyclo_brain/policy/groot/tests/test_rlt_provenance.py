#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.rlt_provenance import (  # noqa: E402
    RLTProvenanceError,
    build_groot_rlt_provenance,
    canonical_sha256,
    rlt_action_codec_contract,
    rlt_action_codec_id,
    validate_stage1_encoder_provenance,
    validate_stage2_bundle_provenance,
)


def _modality(keys, *, delta=(0,), action_configs=None):
    return {
        "delta_indices": list(delta),
        "modality_keys": list(keys),
        "sin_cos_embedding_keys": None,
        "mean_std_embedding_keys": None,
        "action_configs": action_configs,
    }


def _processor() -> dict:
    action_config = {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": None,
    }
    return {
        "processor_class": "Gr00tN1d7Processor",
        "processor_kwargs": {
            "modality_configs": {
                "new_embodiment": {
                    "video": _modality(
                        ("cam_left_head", "cam_left_wrist", "cam_right_wrist")
                    ),
                    "state": _modality(("arm_left", "arm_right", "odometry")),
                    "action": _modality(
                        ("arm_left", "arm_right", "odometry"),
                        delta=tuple(range(16)),
                        action_configs=[deepcopy(action_config) for _ in range(3)],
                    ),
                    "language": _modality(
                        ("annotation.human.task_description",)
                    ),
                }
            },
            "use_percentiles": False,
            "use_mean_std": False,
            "clip_outliers": True,
            "apply_sincos_state_encoding": False,
            "use_relative_action": True,
            "exclude_state": False,
            "max_state_dim": 132,
            "max_action_dim": 132,
            "max_action_horizon": 40,
        },
    }


def _statistics(offset: float = 0.0) -> dict:
    groups = {"arm_left": 8, "arm_right": 8, "odometry": 3}
    return {
        "new_embodiment": {
            domain: {
                group: {
                    statistic: [
                        offset + float(index) / 10.0 for index in range(width)
                    ]
                    for statistic in ("min", "max", "mean", "std", "q01", "q99")
                }
                for group, width in groups.items()
            }
            for domain in ("state", "action")
        }
    }


def _write_checkpoint(
    root: Path,
    *,
    statistics_offset: float = 0.0,
    reverse_json_keys: bool = False,
) -> None:
    root.mkdir()
    payloads = {
        "config.json": {"model_type": "Gr00tN1d7", "action_horizon": 40},
        "processor_config.json": _processor(),
        "statistics.json": _statistics(statistics_offset),
        "embodiment_id.json": {"new_embodiment": 10},
    }
    for name, payload in payloads.items():
        text = json.dumps(
            payload,
            sort_keys=not reverse_json_keys,
            indent=2 if reverse_json_keys else None,
        )
        (root / name).write_text(text, encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"same frozen weights")


class RLTProvenanceTests(unittest.TestCase):
    def test_identity_is_path_and_json_format_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            _write_checkpoint(first)
            _write_checkpoint(second, reverse_json_keys=True)

            self.assertEqual(
                build_groot_rlt_provenance(first),
                build_groot_rlt_provenance(second),
            )

    def test_statistics_change_normalization_and_checkpoint_not_codec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            _write_checkpoint(first)
            _write_checkpoint(second, statistics_offset=1.0)

            left = build_groot_rlt_provenance(first)
            right = build_groot_rlt_provenance(second)

            self.assertEqual(left.weight_fingerprint, right.weight_fingerprint)
            self.assertNotEqual(left.processor_fingerprint, right.processor_fingerprint)
            self.assertNotEqual(left.checkpoint_fingerprint, right.checkpoint_fingerprint)
            self.assertNotEqual(left.action_normalization_id, right.action_normalization_id)
            self.assertEqual(left.action_codec_id, right.action_codec_id)

    def test_action_codec_records_exact_22_to_19_mapping(self) -> None:
        contract = rlt_action_codec_contract()
        mapping = contract["mapping"]

        self.assertEqual(len(contract["source"]["recorder_names"]), 22)
        self.assertEqual(mapping["selected_source_indices"], [*range(16), 19, 20, 21])
        self.assertEqual(mapping["dropped_source_indices"], [16, 17, 18])
        self.assertEqual(len(mapping["output_names"]), 19)
        self.assertEqual(
            rlt_action_codec_id(),
            "sha256:c1f464f0c386a4fdda02ace25a6394268f7000a745077c1541902efbde0c2308",
        )

    def test_invalid_processor_modality_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            _write_checkpoint(checkpoint)
            processor_path = checkpoint / "processor_config.json"
            processor = json.loads(processor_path.read_text(encoding="utf-8"))
            processor["processor_kwargs"]["modality_configs"]["new_embodiment"][
                "action"
            ]["modality_keys"] = ["arm_right", "arm_left", "odometry"]
            processor_path.write_text(json.dumps(processor), encoding="utf-8")

            with self.assertRaisesRegex(RLTProvenanceError, "modality order"):
                build_groot_rlt_provenance(checkpoint)

    def test_stage1_encoder_requires_processor_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            _write_checkpoint(checkpoint)
            provenance = build_groot_rlt_provenance(checkpoint)

            with self.assertRaisesRegex(RLTProvenanceError, "retrain Stage 1"):
                validate_stage1_encoder_provenance(
                    {"policy_weight_fingerprint": provenance.weight_fingerprint},
                    provenance,
                )

    def test_stage2_manifest_is_bound_to_runtime_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkpoint = base / "checkpoint"
            bundle = base / "bundle"
            _write_checkpoint(checkpoint)
            bundle.mkdir()
            provenance = build_groot_rlt_provenance(checkpoint)
            spec = SimpleNamespace(
                action_normalization_id=provenance.action_normalization_id,
                action_codec_id=provenance.action_codec_id,
            )
            round_unsigned = {
                "format": "cyclo_brain.rlt.stage2_training_round/v1",
            }
            training_round = {
                **round_unsigned,
                "round_fingerprint": canonical_sha256(round_unsigned),
            }
            unsigned = {
                "format": "cyclo_brain.rlt.stage2_bundle/v2",
                "source": {
                    "groot_checkpoint_fingerprint": provenance.checkpoint_fingerprint
                },
                "spec": {
                    "action_normalization_id": provenance.action_normalization_id,
                    "action_codec_id": provenance.action_codec_id,
                },
                "training_round": training_round,
            }
            manifest = {
                **unsigned,
                "manifest_fingerprint": canonical_sha256(unsigned),
            }
            (bundle / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            validate_stage2_bundle_provenance(
                bundle,
                spec=spec,
                provenance=provenance,
            )
            manifest["source"]["groot_checkpoint_fingerprint"] = "0" * 64
            unsigned = {
                key: value for key, value in manifest.items() if key != "manifest_fingerprint"
            }
            manifest["manifest_fingerprint"] = canonical_sha256(unsigned)
            (bundle / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(RLTProvenanceError, "different GR00T"):
                validate_stage2_bundle_provenance(
                    bundle,
                    spec=spec,
                    provenance=provenance,
                )


if __name__ == "__main__":
    unittest.main()
