#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical GR00T processor and Cyclo action-contract identities for RLT.

Model weights alone do not identify an executable robot policy.  GR00T's
processor configuration and normalization statistics change both the frozen
features consumed by RLT and the physical meaning of a normalized action.
This module derives path-independent SHA-256 identities from those assets and
from the exact SG2 recorder-to-19D mapping used by the RLT Action MLP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from cyclo_brain.algorithm.common import canonical_json_sha256
from cyclo_brain.model.common.sg2 import (
    GROOT_REFERENCE_ACTION_HORIZON,
    GROOT_RLT_CAMERA_NAMES,
    GROOT_RLT_PROCESSOR_GROUPS,
    RLT_ACTION_DIM,
    RLT_ACTION_GROUP_NAMES,
    RLT_DROPPED_RECORDER_INDICES,
    RLT_DROPPED_RECORDER_NAMES,
    RLT_SELECTED_RECORDER_INDICES,
    RLT_SELECTED_RECORDER_NAMES,
    SG2_RECORDER_ACTION_NAMES,
    SG2_ROBOT_TYPE,
)


_PROVENANCE_SCHEMA = "cyclo.groot.rlt.runtime-provenance/v1"
_NORMALIZATION_SCHEMA = "cyclo.groot.rlt.action-normalization/v1"
_CODEC_SCHEMA = "cyclo.groot.rlt.action-codec/v1"
_STAGE2_BUNDLE_FORMAT = "cyclo_brain.rlt.stage2_bundle/v2"

GROOT_EMBODIMENT = "new_embodiment"
RLT_CAMERA_KEYS = GROOT_RLT_CAMERA_NAMES
RLT_ACTION_GROUPS = GROOT_RLT_PROCESSOR_GROUPS

_PROCESSOR_BEHAVIOR_KEYS = (
    "use_percentiles",
    "use_mean_std",
    "clip_outliers",
    "apply_sincos_state_encoding",
    "use_relative_action",
    "exclude_state",
    "max_state_dim",
    "max_action_dim",
    "max_action_horizon",
)
_STATISTIC_KEYS = ("min", "max", "mean", "std", "q01", "q99")
_MODEL_ASSET_NAMES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
)


class RLTProvenanceError(ValueError):
    """Raised when a GR00T checkpoint cannot satisfy the RLT contract."""


@dataclass(frozen=True)
class GR00TRLTProvenance:
    """Path-independent identities required to train or run an RLT bundle."""

    weight_fingerprint: str
    model_config_fingerprint: str
    processor_fingerprint: str
    checkpoint_fingerprint: str
    action_normalization_id: str
    action_codec_id: str


def _reject_constant(value: str) -> None:
    raise RLTProvenanceError(f"GR00T JSON contains non-finite value {value!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RLTProvenanceError(f"GR00T JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one deterministic encoding."""

    try:
        return canonical_json_sha256(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RLTProvenanceError("GR00T contract is not canonical JSON") from error


def _regular_file(path: Path, name: str, *, maximum_bytes: int) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RLTProvenanceError(f"GR00T {name} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RLTProvenanceError(f"GR00T {name} must be a non-symlink regular file")
    if not 0 < metadata.st_size <= maximum_bytes:
        raise RLTProvenanceError(f"GR00T {name} size is invalid")
    return metadata


def _read_json(path: Path, name: str, *, maximum_bytes: int = 64 * 1024**2) -> Any:
    _regular_file(path, name, maximum_bytes=maximum_bytes)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except RLTProvenanceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RLTProvenanceError(f"Cannot read GR00T {name}: {path}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RLTProvenanceError(f"GR00T {name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RLTProvenanceError(f"GR00T {name} must be a sequence")
    return tuple(value)


def rlt_action_codec_contract() -> dict[str, Any]:
    """Return the complete named recorder-to-MLP action mapping."""

    return {
        "schema": _CODEC_SCHEMA,
        "robot_type": SG2_ROBOT_TYPE,
        "source": {
            "state_feature": "observation.state",
            "action_feature": "action",
            "recorder_names": list(SG2_RECORDER_ACTION_NAMES),
        },
        "mapping": {
            "operation": "select-by-name-then-concatenate",
            "group_order": list(RLT_ACTION_GROUPS),
            "group_names": {
                group: list(RLT_ACTION_GROUP_NAMES[group]) for group in RLT_ACTION_GROUPS
            },
            "selected_source_indices": list(RLT_SELECTED_RECORDER_INDICES),
            "output_names": list(RLT_SELECTED_RECORDER_NAMES),
            "dropped_source_indices": list(RLT_DROPPED_RECORDER_INDICES),
            "dropped_names": list(RLT_DROPPED_RECORDER_NAMES),
            "output_dimension": RLT_ACTION_DIM,
        },
    }


def rlt_action_codec_id() -> str:
    return f"sha256:{canonical_sha256(rlt_action_codec_contract())}"


def _validate_modality_config(processor: Mapping[str, Any]) -> Mapping[str, Any]:
    processor_class = processor.get("processor_class")
    if processor_class != "Gr00tN1d7Processor":
        raise RLTProvenanceError(
            "RLT requires processor_class='Gr00tN1d7Processor'"
        )
    kwargs = _mapping(processor.get("processor_kwargs"), "processor_kwargs")
    modalities = _mapping(kwargs.get("modality_configs"), "modality_configs")
    embodiment = _mapping(
        modalities.get(GROOT_EMBODIMENT),
        f"modality_configs.{GROOT_EMBODIMENT}",
    )
    for key in ("video", "state", "action", "language"):
        _mapping(embodiment.get(key), f"{GROOT_EMBODIMENT}.{key}")

    video_keys = _sequence(
        embodiment["video"].get("modality_keys"), "RLT video modality_keys"
    )
    state_keys = _sequence(
        embodiment["state"].get("modality_keys"), "RLT state modality_keys"
    )
    action_keys = _sequence(
        embodiment["action"].get("modality_keys"), "RLT action modality_keys"
    )
    action_offsets = _sequence(
        embodiment["action"].get("delta_indices"), "RLT action delta_indices"
    )
    if video_keys != RLT_CAMERA_KEYS:
        raise RLTProvenanceError(
            f"RLT camera order must be {RLT_CAMERA_KEYS}, got {video_keys}"
        )
    if state_keys != RLT_ACTION_GROUPS or action_keys != RLT_ACTION_GROUPS:
        raise RLTProvenanceError(
            "RLT state/action modality order must be arm_left, arm_right, odometry"
        )
    if action_offsets != tuple(range(GROOT_REFERENCE_ACTION_HORIZON)):
        raise RLTProvenanceError("RLT GR00T reference horizon must be indices 0..15")
    action_configs = _sequence(
        embodiment["action"].get("action_configs"), "RLT action action_configs"
    )
    expected_action_config = {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": None,
    }
    if len(action_configs) != len(RLT_ACTION_GROUPS) or any(
        value != expected_action_config for value in action_configs
    ):
        raise RLTProvenanceError(
            "RLT action groups must use ABSOLUTE/NON_EEF/DEFAULT processing"
        )
    language_keys = _sequence(
        embodiment["language"].get("modality_keys"),
        "RLT language modality_keys",
    )
    if language_keys != ("annotation.human.task_description",):
        raise RLTProvenanceError("RLT GR00T language modality contract disagrees")
    missing_behavior = [key for key in _PROCESSOR_BEHAVIOR_KEYS if key not in kwargs]
    if missing_behavior:
        raise RLTProvenanceError(
            "GR00T processor behavior contract is incomplete: "
            + ", ".join(missing_behavior)
        )
    return kwargs


def _selected_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    embodiment = _mapping(
        statistics.get(GROOT_EMBODIMENT),
        f"statistics.{GROOT_EMBODIMENT}",
    )
    selected: dict[str, Any] = {}
    for domain in ("state", "action"):
        source = _mapping(embodiment.get(domain), f"statistics.{domain}")
        groups: dict[str, Any] = {}
        for group in RLT_ACTION_GROUPS:
            values = _mapping(source.get(group), f"statistics.{domain}.{group}")
            expected_width = len(RLT_ACTION_GROUP_NAMES[group])
            normalized: dict[str, Any] = {}
            for statistic in _STATISTIC_KEYS:
                vector = _sequence(
                    values.get(statistic),
                    f"statistics.{domain}.{group}.{statistic}",
                )
                if len(vector) != expected_width or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in vector
                ):
                    raise RLTProvenanceError(
                        f"GR00T statistics.{domain}.{group}.{statistic} must have "
                        f"{expected_width} finite values"
                    )
                normalized[statistic] = list(vector)
            groups[group] = normalized
        selected[domain] = groups
    return selected


def _weight_fingerprint(checkpoint: Path) -> str:
    suffixes = {".safetensors", ".pth", ".pt", ".bin"}
    ignored = (
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
        "trainer_state",
        "training_args",
        "training_state",
        "replay_buffer",
    )
    files: list[dict[str, Any]] = []
    for path in sorted(checkpoint.rglob("*")):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RLTProvenanceError(f"Cannot inspect GR00T checkpoint: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RLTProvenanceError(f"GR00T checkpoint contains a symlink: {path}")
        if not stat.S_ISREG(metadata.st_mode):
            continue
        name = path.name.lower()
        if path.suffix.lower() not in suffixes or name.startswith(ignored):
            continue
        digest = sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("GR00T checkpoint weight changed while hashing")
        files.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": before.st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise RLTProvenanceError("GR00T checkpoint contains no model weight files")
    return canonical_sha256(
        {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "file_count": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        }
    )


def build_groot_rlt_provenance(
    checkpoint: str | os.PathLike[str],
) -> GR00TRLTProvenance:
    """Validate assets and derive all identities for one GR00T checkpoint."""

    root = Path(os.path.abspath(Path(checkpoint).expanduser()))
    try:
        metadata = root.lstat()
    except OSError as error:
        raise RLTProvenanceError(f"GR00T checkpoint does not exist: {root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RLTProvenanceError("GR00T checkpoint must be a non-symlink directory")

    assets = {
        name: _read_json(root / name, name)
        for name in _MODEL_ASSET_NAMES
    }
    model_config = _mapping(assets["config.json"], "config.json")
    processor = _mapping(assets["processor_config.json"], "processor_config.json")
    statistics = _mapping(assets["statistics.json"], "statistics.json")
    embodiment_ids = _mapping(assets["embodiment_id.json"], "embodiment_id.json")
    kwargs = _validate_modality_config(processor)
    selected_statistics = _selected_statistics(statistics)
    embodiment_id = embodiment_ids.get(GROOT_EMBODIMENT)
    if isinstance(embodiment_id, bool) or not isinstance(embodiment_id, int):
        raise RLTProvenanceError("GR00T new_embodiment ID is invalid")

    behavior = {
        key: kwargs.get(key) for key in _PROCESSOR_BEHAVIOR_KEYS
    }
    normalization_contract = {
        "schema": _NORMALIZATION_SCHEMA,
        "processor_class": processor["processor_class"],
        "embodiment": GROOT_EMBODIMENT,
        "embodiment_id": embodiment_id,
        "behavior": behavior,
        "state_modality": kwargs["modality_configs"][GROOT_EMBODIMENT]["state"],
        "action_modality": kwargs["modality_configs"][GROOT_EMBODIMENT]["action"],
        "statistics": selected_statistics,
    }
    action_normalization_id = f"sha256:{canonical_sha256(normalization_contract)}"
    action_codec_id = rlt_action_codec_id()

    processor_contract = {
        "processor_config": processor,
        "statistics": statistics,
        "embodiment_id": embodiment_ids,
    }
    processor_fingerprint = canonical_sha256(processor_contract)
    model_config_fingerprint = canonical_sha256(model_config)
    weight_fingerprint = _weight_fingerprint(root)
    checkpoint_fingerprint = canonical_sha256(
        {
            "schema": _PROVENANCE_SCHEMA,
            "weight_fingerprint": weight_fingerprint,
            "model_config_fingerprint": model_config_fingerprint,
            "processor_fingerprint": processor_fingerprint,
        }
    )
    return GR00TRLTProvenance(
        weight_fingerprint=weight_fingerprint,
        model_config_fingerprint=model_config_fingerprint,
        processor_fingerprint=processor_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        action_normalization_id=action_normalization_id,
        action_codec_id=action_codec_id,
    )


def validate_stage2_bundle_provenance(
    bundle_root: str | os.PathLike[str],
    *,
    spec: Any,
    provenance: GR00TRLTProvenance,
) -> None:
    """Fail closed when a Stage-2 bundle and loaded GR00T runtime disagree."""

    root = Path(os.path.abspath(Path(bundle_root).expanduser()))
    manifest = _mapping(
        _read_json(root / "manifest.json", "RLT Stage 2 manifest", maximum_bytes=4 * 1024**2),
        "RLT Stage 2 manifest",
    )
    if manifest.get("format") != _STAGE2_BUNDLE_FORMAT:
        raise RLTProvenanceError(
            "RLT bundle lacks canonical Stage 2 provenance; retrain Stage 2"
        )
    saved_fingerprint = manifest.get("manifest_fingerprint")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    if saved_fingerprint != canonical_sha256(unsigned):
        raise RLTProvenanceError("RLT Stage 2 manifest fingerprint disagrees")
    source = _mapping(manifest.get("source"), "RLT Stage 2 source")
    if source.get("groot_checkpoint_fingerprint") != provenance.checkpoint_fingerprint:
        raise RLTProvenanceError(
            "RLT bundle was trained with different GR00T weights, config, processor, or statistics"
        )
    manifest_spec = _mapping(manifest.get("spec"), "RLT Stage 2 spec")
    training_round = _mapping(
        manifest.get("training_round"),
        "RLT Stage 2 training round",
    )
    saved_round_fingerprint = training_round.get("round_fingerprint")
    if (
        not isinstance(saved_round_fingerprint, str)
        or len(saved_round_fingerprint) != 64
        or canonical_sha256(
            {
                key: value
                for key, value in training_round.items()
                if key != "round_fingerprint"
            }
        )
        != saved_round_fingerprint
    ):
        raise RLTProvenanceError("RLT Stage 2 training round fingerprint disagrees")
    expected = {
        "action_normalization_id": provenance.action_normalization_id,
        "action_codec_id": provenance.action_codec_id,
    }
    for key, value in expected.items():
        if manifest_spec.get(key) != value or getattr(spec, key, None) != value:
            raise RLTProvenanceError(f"RLT Stage 2 {key} disagrees with runtime")


def validate_stage1_encoder_provenance(
    representation_contract: Mapping[str, Any],
    provenance: GR00TRLTProvenance,
) -> None:
    """Bind a frozen Stage-1 encoder to the complete executable policy."""

    contract = _mapping(representation_contract, "RLT Stage 1 representation contract")
    expected = {
        "policy_weight_fingerprint": provenance.weight_fingerprint,
        "policy_checkpoint_fingerprint": provenance.checkpoint_fingerprint,
        "policy_processor_fingerprint": provenance.processor_fingerprint,
        "action_normalization_id": provenance.action_normalization_id,
        "action_codec_id": provenance.action_codec_id,
    }
    missing = [key for key in expected if key not in contract]
    if missing:
        raise RLTProvenanceError(
            "RLT Stage 1 encoder lacks canonical processor provenance "
            f"({', '.join(missing)}); retrain Stage 1"
        )
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RLTProvenanceError(
                f"RLT Stage 1 encoder {key} disagrees with the GR00T runtime"
            )


__all__ = [
    "GR00TRLTProvenance",
    "GROOT_EMBODIMENT",
    "RLT_ACTION_GROUP_NAMES",
    "RLT_ACTION_GROUPS",
    "RLT_CAMERA_KEYS",
    "RLTProvenanceError",
    "SG2_RECORDER_ACTION_NAMES",
    "build_groot_rlt_provenance",
    "canonical_sha256",
    "rlt_action_codec_contract",
    "rlt_action_codec_id",
    "validate_stage1_encoder_provenance",
    "validate_stage2_bundle_provenance",
]
