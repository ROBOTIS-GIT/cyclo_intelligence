"""Pure metadata validation for RLT Stage-2 bundles; no tensor loading."""
import math
from collections.abc import Mapping
from typing import Any

from .fingerprints import canonical_json_sha256, validate_lowercase_sha256


def _digest(value: Any, name: str) -> str:
    return validate_lowercase_sha256(
        value,
        error_message=f"RLT Stage 2 bundle {name} is invalid",
    )


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"RLT Stage 2 {name} must be a {qualifier} integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"RLT Stage 2 {name} must be finite")
    return float(value)


def validate_training_round(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "format",
        "replay",
        "datasets",
        "reference_extraction",
        "optimization",
        "round_fingerprint",
    }:
        raise ValueError("RLT Stage 2 training round fields are invalid")
    if value.get("format") != "cyclo_brain.rlt.stage2_training_round/v1":
        raise ValueError("RLT Stage 2 training round format is invalid")
    unsigned = {key: item for key, item in value.items() if key != "round_fingerprint"}
    fingerprint = _digest(value.get("round_fingerprint"), "training round fingerprint")
    if canonical_json_sha256(unsigned, allow_nan=False) != fingerprint:
        raise ValueError("RLT Stage 2 training round fingerprint disagrees")

    replay = value.get("replay")
    if not isinstance(replay, Mapping) or set(replay) != {
        "manifest",
        "artifact",
        "manifest_fingerprint",
        "spec_fingerprint",
        "transition_count",
        "average_reward",
        "reward_contract",
    }:
        raise ValueError("RLT Stage 2 training round replay fields are invalid")
    for key in ("manifest", "artifact"):
        record = replay.get(key)
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise ValueError("RLT Stage 2 training round replay record is invalid")
        expected = "manifest.json" if key == "manifest" else "replay.pt"
        if record.get("relative_path") != expected:
            raise ValueError("RLT Stage 2 training round replay path is invalid")
        _positive_integer(record.get("byte_count"), "replay byte_count")
        _digest(record.get("sha256"), "replay SHA-256")
    _digest(replay.get("manifest_fingerprint"), "replay manifest fingerprint")
    _digest(replay.get("spec_fingerprint"), "replay spec fingerprint")
    _positive_integer(replay.get("transition_count"), "replay transition_count")
    _finite_float(replay.get("average_reward"), "replay average_reward")
    if not isinstance(replay.get("reward_contract"), str) or not replay[
        "reward_contract"
    ]:
        raise ValueError("RLT Stage 2 reward contract is invalid")

    datasets = value.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {
        "snapshot_fingerprint",
        "snapshots",
    }:
        raise ValueError("RLT Stage 2 training round datasets are invalid")
    _digest(datasets.get("snapshot_fingerprint"), "dataset snapshot fingerprint")
    summaries = datasets.get("snapshots")
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("RLT Stage 2 training round dataset summaries are invalid")
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping) or set(summary) != {
            "ordinal",
            "file_count",
            "total_byte_count",
            "content_fingerprint",
        }:
            raise ValueError("RLT Stage 2 dataset summary fields are invalid")
        if _positive_integer(summary.get("ordinal"), "dataset summary ordinal", allow_zero=True) != index:
            raise ValueError("RLT Stage 2 dataset summary order is invalid")
        _positive_integer(summary.get("file_count"), "dataset summary file_count")
        _positive_integer(
            summary.get("total_byte_count"),
            "dataset summary total_byte_count",
            allow_zero=True,
        )
        _digest(summary.get("content_fingerprint"), "dataset summary fingerprint")
    expected_collection = canonical_json_sha256(
        {
            "format": "cyclo.groot.rlt.dataset_collection/v1",
            "ordered_content_fingerprints": [
                summary["content_fingerprint"] for summary in summaries
            ],
        },
        allow_nan=False,
    )
    if datasets.get("snapshot_fingerprint") != expected_collection:
        raise ValueError("RLT Stage 2 dataset summary collection disagrees")

    reference = value.get("reference_extraction")
    if not isinstance(reference, Mapping) or set(reference) != {
        "seed",
        "feature_batch_size",
    }:
        raise ValueError("RLT Stage 2 reference extraction fields are invalid")
    _positive_integer(reference.get("seed"), "reference seed", allow_zero=True)
    _positive_integer(reference.get("feature_batch_size"), "feature batch size")

    optimization = value.get("optimization")
    if not isinstance(optimization, Mapping) or set(optimization) != {
        "sampling_seed",
        "batch_size",
        "steps",
        "starting_critic_updates",
    }:
        raise ValueError("RLT Stage 2 optimization provenance fields are invalid")
    _positive_integer(optimization.get("sampling_seed"), "sampling seed", allow_zero=True)
    _positive_integer(optimization.get("batch_size"), "optimizer batch size")
    _positive_integer(optimization.get("steps"), "optimizer steps")
    _positive_integer(
        optimization.get("starting_critic_updates"),
        "starting critic updates",
        allow_zero=True,
    )
