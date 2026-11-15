#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Strict GR00T Training-Time RTC request and checkpoint contracts.

This module deliberately contains no GR00T or Torch imports.  It is shared by
the parent Cyclo runtime while the model-side, per-action-token time support is
implemented in the upstream GR00T package.  A legacy checkpoint must never be
silently treated as TT-RTC capable merely because it exposes GR00T's older
inference-time ``RTC`` option.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

from cyclo_brain.model.common import (
    GROOT_REFERENCE_ACTION_HORIZON,
    RLT_ACTION_DIM,
    RLT_ACTION_HORIZON,
)


TT_RTC_REQUEST_MODE = "tt_rtc"
TT_RTC_SCHEMA = "cyclo.training-time-rtc/v1"
TT_RTC_TRAINING_RECORD_SCHEMA = "cyclo.tt-rtc-training/v1"
TT_RTC_RLT_CONTEXT_SCHEMA = "cyclo.groot.tt-rtc-rlt-context/v1"
TT_RTC_ACTION_HORIZON = GROOT_REFERENCE_ACTION_HORIZON
TT_RTC_ACTION_DIM = RLT_ACTION_DIM
TT_RTC_MAX_DELAY_STEPS = GROOT_REFERENCE_ACTION_HORIZON - RLT_ACTION_HORIZON
TT_RTC_ACTION_HZ = 15.0
TT_RTC_RLT_CHUNK_LENGTH = RLT_ACTION_HORIZON
TT_RTC_MANIFEST_NAME = "tt_rtc_manifest.json"
TT_RTC_MODEL_ACTION_HORIZON = 40
TT_RTC_MODEL_ACTION_DIM = 132
TT_RTC_PROCESSOR_ACTION_HORIZON = 40
TT_RTC_PROCESSOR_ACTION_DIM = 132
TT_RTC_NUM_INFERENCE_TIMESTEPS = 4
TT_RTC_NUM_TIMESTEP_BUCKETS = 1000
TT_RTC_PROFILE_DELAY_RANGES = {
    (16, 19): (0, 6),
    (32, 16): (0, 8),
    (20, 16): (6, 8),
}


class TTRTCContractError(ValueError):
    """A request/checkpoint does not satisfy the fixed Cyclo TT-RTC contract."""


@dataclass(frozen=True)
class TTRTCRequest:
    """Validated TT-RTC request in the published physical-action domain."""

    delay_steps: int
    action_dim: int
    prefix_actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class TTRTCCapability:
    """Validated checkpoint capability and the source file that declared it."""

    source: Path
    payload: Mapping[str, Any]


def _as_plain_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TTRTCContractError(f"TT-RTC manifest {name} must be an object")
    return value


def _exact_number(value: object, expected: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != float(expected)
    ):
        raise TTRTCContractError(
            f"TT-RTC manifest {name} must be {expected}, got {value!r}"
        )


def _exact_integer(value: object, expected: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise TTRTCContractError(
            f"TT-RTC manifest {name} must be {expected}, got {value!r}"
        )


def parse_tt_rtc_request(
    request: object,
    *,
    action_dim: int = TT_RTC_ACTION_DIM,
    min_delay_steps: int = 0,
    max_delay_steps: int = TT_RTC_MAX_DELAY_STEPS,
) -> TTRTCRequest | None:
    """Return a validated TT-RTC request, or ``None`` for normal sync/async.

    ``rtc_prefix_action_list`` is row-major and contains the exact actions that
    the publisher has already committed for the handoff interval.  Its domain
    remains physical here; the future GR00T policy-side API is responsible for
    applying the checkpoint processor exactly once before model conditioning.
    """

    raw_mode = str(getattr(request, "action_request_mode", "") or "").strip().lower()
    if raw_mode in {"", "sync", "async"}:
        return None
    if raw_mode != TT_RTC_REQUEST_MODE:
        raise TTRTCContractError(
            "Unsupported action_request_mode; expected 'sync', 'async', or 'tt_rtc'"
        )

    if (
        type(min_delay_steps) is not int
        or type(max_delay_steps) is not int
        or min_delay_steps < 0
        or max_delay_steps < min_delay_steps
    ):
        raise TTRTCContractError("Invalid TT-RTC request delay bounds")

    delay_steps = getattr(request, "rtc_delay_steps", None)
    if (
        isinstance(delay_steps, bool)
        or not isinstance(delay_steps, int)
        or not (
            delay_steps == 0
            or min_delay_steps <= delay_steps <= max_delay_steps
        )
    ):
        raise TTRTCContractError(
            "TT-RTC rtc_delay_steps must be 0 for bootstrap or an integer in "
            f"{min_delay_steps}..{max_delay_steps}"
        )

    expected_action_dim = action_dim
    action_dim = getattr(request, "rtc_action_dim", None)
    if (
        isinstance(action_dim, bool)
        or not isinstance(action_dim, int)
        or action_dim != expected_action_dim
    ):
        raise TTRTCContractError(
            f"TT-RTC rtc_action_dim must be {expected_action_dim}"
        )

    raw_values = getattr(request, "rtc_prefix_action_list", None)
    if hasattr(raw_values, "reshape") and hasattr(raw_values, "tolist"):
        try:
            values = raw_values.reshape(-1).tolist()
        except (TypeError, ValueError) as error:
            raise TTRTCContractError(
                "TT-RTC rtc_prefix_action_list cannot be flattened"
            ) from error
    else:
        values = raw_values
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
    ):
        raise TTRTCContractError("TT-RTC rtc_prefix_action_list must be a sequence")
    expected_values = delay_steps * action_dim
    if len(values) != expected_values:
        raise TTRTCContractError(
            "TT-RTC prefix shape mismatch: expected "
            f"({delay_steps}, {action_dim})/{expected_values} values, got {len(values)}"
        )

    flat: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TTRTCContractError(
                f"TT-RTC prefix value {index} is not a finite number"
            )
        item = float(value)
        if not math.isfinite(item):
            raise TTRTCContractError(
                f"TT-RTC prefix value {index} is not a finite number"
            )
        flat.append(item)

    prefix = tuple(
        tuple(flat[start : start + action_dim])
        for start in range(0, len(flat), action_dim)
    )
    return TTRTCRequest(
        delay_steps=delay_steps,
        action_dim=action_dim,
        prefix_actions=prefix,
    )


def _training_manifest_metadata(root: Path, payload: Mapping[str, Any]) -> dict:
    """Adapt external training metadata in memory; never rewrite the checkpoint."""
    try:
        source_schema = payload.get("schema")
        if source_schema not in {TT_RTC_SCHEMA, TT_RTC_TRAINING_RECORD_SCHEMA}:
            raise TTRTCContractError(
                "TT-RTC training manifest schema must be "
                f"{TT_RTC_TRAINING_RECORD_SCHEMA!r}, got {source_schema!r}"
            )
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        processor = json.loads((root / "processor_config.json").read_text(encoding="utf-8"))["processor_kwargs"]
        rtc = dict(_as_plain_mapping(payload.get("training_time_rtc"), "training_time_rtc"))
        if config.get("training_time_rtc") is not True:
            raise TTRTCContractError("Training config must declare training_time_rtc=true")
        for field, key in (
            ("action_horizon", "tt_rtc_logical_action_horizon"),
            ("action_dimension", "tt_rtc_logical_action_dim"),
            ("max_delay_steps", "tt_rtc_max_delay_steps"),
            ("model_action_horizon", "action_horizon"),
            ("model_action_dimension", "max_action_dim"),
        ):
            _exact_integer(config.get(key), rtc.get(field), f"training_config.{key}")
        horizon = len(processor["modality_configs"]["new_embodiment"]["action"]["delta_indices"])
        _exact_integer(horizon, rtc["action_horizon"], "training_processor.action_horizon")
        for field, key in (
            ("min_delay_steps", "tt_rtc_min_delay_steps"),
            ("min_postfix_steps", "tt_rtc_min_postfix_steps"),
        ):
            if field in rtc or key in config:
                if field not in rtc or key not in config:
                    raise TTRTCContractError(
                        f"Training manifest/config must both declare {field}"
                    )
                _exact_integer(config.get(key), rtc.get(field), f"training_config.{key}")
        delay_sampling = _as_plain_mapping(
            rtc.get("delay_sampling"),
            "delay_sampling",
        )
        if delay_sampling.get("type") == "uniform_choice":
            min_delay = rtc.get("min_delay_steps", 0)
            max_delay = rtc.get("max_delay_steps")
            expected_values = list(range(min_delay, max_delay + 1))
            values = delay_sampling.get("values")
            probabilities = delay_sampling.get("probabilities")
            expected_probability = 1.0 / len(expected_values)
            if values != expected_values:
                raise TTRTCContractError(
                    "Training delay_sampling.values must cover every integer "
                    "in the configured delay range"
                )
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != len(expected_values)
                or any(
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isclose(
                        float(probability),
                        expected_probability,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for probability in probabilities
                )
            ):
                raise TTRTCContractError(
                    "Training delay_sampling.probabilities must be uniform"
                )
            rtc["delay_sampling"] = {
                "type": "uniform_integer",
                "min_inclusive": min_delay,
                "max_inclusive": max_delay,
            }
        # The existing source clock is 15 Hz; absent training metadata does
        # not declare a new rate. The loaded policy is cross-checked below.
        extra = {
            "action_hz": config.get("training_time_rtc_action_hz", TT_RTC_ACTION_HZ),
            "num_inference_timesteps": config["num_inference_timesteps"],
            "num_timestep_buckets": config["num_timestep_buckets"],
            "processor_action_horizon": processor["max_action_horizon"],
            "processor_action_dimension": processor["max_action_dim"],
        }
        for key, value in extra.items():
            if key in rtc and rtc[key] != value:
                raise TTRTCContractError(f"Training manifest disagrees with config: {key}")
            rtc[key] = value
        if rtc.get("state_timestep") != "sampled_flow_time":
            raise TTRTCContractError("Training state_timestep must be sampled_flow_time")
        for key, value in (
            ("training_time_rtc_enabled", True),
            ("training_time_rtc_max_delay_steps", rtc["max_delay_steps"]),
        ):
            if key in config and config[key] != value:
                raise TTRTCContractError(f"Conflicting training config field: {key}")
        adapted = {
            **payload,
            "schema": TT_RTC_SCHEMA,
            "training_time_rtc": rtc,
        }
        if source_schema != TT_RTC_SCHEMA:
            adapted["training_record_schema"] = source_schema
        return adapted
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        raise TTRTCContractError(f"Cannot adapt TT-RTC training metadata: {error}") from error


def _read_manifest(model_path: str | os.PathLike[str]) -> tuple[Path, Mapping[str, Any]]:
    root = Path(os.path.abspath(Path(model_path).expanduser()))
    if root.is_symlink() or not root.is_dir():
        raise TTRTCContractError(
            f"TT-RTC model path must be a non-symlink directory: {root}"
        )

    dedicated = root / TT_RTC_MANIFEST_NAME
    config = root / "config.json"
    training = root / "experiment_cfg" / "tt_rtc_training_manifest.json"
    candidates = (dedicated, config, training)
    for path in candidates:
        if path.is_symlink():
            raise TTRTCContractError(f"TT-RTC manifest must not be a symlink: {path}")
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TTRTCContractError(f"Cannot read TT-RTC manifest: {path}") from error
        if not isinstance(payload, Mapping):
            raise TTRTCContractError(f"TT-RTC manifest root must be an object: {path}")
        # A normal Hugging Face config is a candidate only when it explicitly
        # carries the new contract.  Legacy rtc_training_prefix_steps is not
        # sufficient evidence of Training-Time RTC training.
        if path == config and (
            "training_time_rtc" not in payload or payload["training_time_rtc"] is True
        ):
            continue
        if path == training:
            payload = _training_manifest_metadata(root, payload)
        return path, payload

    raise TTRTCContractError(
        f"TT-RTC capability is missing; add {TT_RTC_MANIFEST_NAME} or an explicit "
        "training_time_rtc object to config.json after TT-RTC fine-tuning"
    )


def load_tt_rtc_capability(
    model_path: str | os.PathLike[str],
    *,
    require_rlt: bool = False,
) -> TTRTCCapability:
    """Load and validate the machine-readable TT-RTC checkpoint contract."""

    source, payload = _read_manifest(model_path)
    schema = payload.get("schema")
    if schema != TT_RTC_SCHEMA:
        raise TTRTCContractError(
            f"TT-RTC manifest schema must be {TT_RTC_SCHEMA!r}, got {schema!r}"
        )

    rtc = _as_plain_mapping(payload.get("training_time_rtc"), "training_time_rtc")
    if rtc.get("trained") is not True:
        raise TTRTCContractError("TT-RTC checkpoint is not marked trained=true")
    horizon, dimension = rtc.get("action_horizon"), rtc.get("action_dimension")
    profile_delay_range = (
        TT_RTC_PROFILE_DELAY_RANGES.get((horizon, dimension))
        if type(horizon) is int and type(dimension) is int
        else None
    )
    if profile_delay_range is None:
        supported = ", ".join(
            f"{supported_horizon}x{supported_dimension}"
            for supported_horizon, supported_dimension in TT_RTC_PROFILE_DELAY_RANGES
        )
        raise TTRTCContractError(
            "Unsupported TT-RTC action_horizon/action_dimension; expected one of "
            f"{supported}"
        )
    expected_min_delay, expected_max_delay = profile_delay_range
    _exact_number(rtc.get("action_hz"), TT_RTC_ACTION_HZ, "action_hz")
    # The runtime supports 0..8 for 32x16. Older 0..6-trained checkpoints
    # retain their declared bound; runtime support is not training evidence.
    if (horizon, dimension) == (32, 16) and rtc.get("max_delay_steps") == 6:
        expected_max_delay = 6
    _exact_integer(rtc.get("max_delay_steps"), expected_max_delay, "max_delay_steps")
    _exact_integer(
        rtc.get("model_action_horizon"),
        TT_RTC_MODEL_ACTION_HORIZON,
        "model_action_horizon",
    )
    _exact_integer(
        rtc.get("model_action_dimension"),
        TT_RTC_MODEL_ACTION_DIM,
        "model_action_dimension",
    )
    _exact_integer(
        rtc.get("num_inference_timesteps"),
        TT_RTC_NUM_INFERENCE_TIMESTEPS,
        "num_inference_timesteps",
    )
    _exact_integer(
        rtc.get("num_timestep_buckets"),
        TT_RTC_NUM_TIMESTEP_BUCKETS,
        "num_timestep_buckets",
    )
    _exact_integer(
        rtc.get("processor_action_horizon"),
        TT_RTC_PROCESSOR_ACTION_HORIZON,
        "processor_action_horizon",
    )
    _exact_integer(
        rtc.get("processor_action_dimension"),
        TT_RTC_PROCESSOR_ACTION_DIM,
        "processor_action_dimension",
    )
    if rtc.get("prefix_input") != "ground_truth_clean_action":
        raise TTRTCContractError(
            "TT-RTC manifest prefix_input must be 'ground_truth_clean_action'"
        )
    if rtc.get("loss_region") != "postfix_only":
        raise TTRTCContractError("TT-RTC manifest loss_region must be 'postfix_only'")
    if rtc.get("per_action_timestep") is not True:
        raise TTRTCContractError("TT-RTC manifest per_action_timestep must be true")

    delay_sampling = _as_plain_mapping(rtc.get("delay_sampling"), "delay_sampling")
    if delay_sampling.get("type") != "uniform_integer":
        raise TTRTCContractError(
            "TT-RTC manifest delay_sampling.type must be 'uniform_integer'"
        )
    min_delay_steps = rtc.get(
        "min_delay_steps",
        delay_sampling.get("min_inclusive"),
    )
    _exact_integer(min_delay_steps, expected_min_delay, "min_delay_steps")
    _exact_integer(
        delay_sampling.get("min_inclusive"),
        expected_min_delay,
        "delay_sampling.min_inclusive",
    )
    _exact_integer(
        delay_sampling.get("max_inclusive"),
        expected_max_delay,
        "delay_sampling.max_inclusive",
    )
    min_postfix_steps = horizon - expected_max_delay
    if "min_postfix_steps" in rtc:
        _exact_integer(
            rtc.get("min_postfix_steps"),
            min_postfix_steps,
            "min_postfix_steps",
        )

    flow = _as_plain_mapping(rtc.get("flow_convention"), "flow_convention")
    _exact_number(flow.get("noise_endpoint"), 0.0, "flow_convention.noise_endpoint")
    _exact_number(flow.get("clean_endpoint"), 1.0, "flow_convention.clean_endpoint")
    if flow.get("velocity_target") != "action_minus_noise":
        raise TTRTCContractError(
            "TT-RTC manifest flow_convention.velocity_target must be 'action_minus_noise'"
        )

    if min_postfix_steps < TT_RTC_RLT_CHUNK_LENGTH:
        raise TTRTCContractError(
            "TT-RTC action horizon must retain at least "
            f"{TT_RTC_RLT_CHUNK_LENGTH} postfix actions at max delay"
        )

    if require_rlt:
        rlt = _as_plain_mapping(payload.get("rlt"), "rlt")
        _exact_integer(rlt.get("chunk_length"), TT_RTC_RLT_CHUNK_LENGTH, "rlt.chunk_length")
        _exact_integer(
            rlt.get("reference_horizon"),
            horizon,
            "rlt.reference_horizon",
        )
        expected_reference_slice = f"[d:d+{TT_RTC_RLT_CHUNK_LENGTH}]"
        if rlt.get("reference_slice") != expected_reference_slice:
            raise TTRTCContractError(
                "TT-RTC RLT manifest reference_slice must be "
                f"'{expected_reference_slice}'"
            )

    normalized_rtc = dict(rtc)
    normalized_rtc["min_delay_steps"] = expected_min_delay
    normalized_rtc["min_postfix_steps"] = min_postfix_steps
    normalized_payload = {**payload, "training_time_rtc": normalized_rtc}
    return TTRTCCapability(source=source, payload=normalized_payload)


def validate_tt_rtc_rlt_binding(
    capability: TTRTCCapability,
    spec: object,
) -> None:
    """Bind one validated TT-RTC checkpoint to one validated RLT bundle.

    The GR00T checkpoint owns the TT-RTC timing contract, while a canonical
    Stage-2 bundle owns the RLT reference/action contract and its exact GR00T
    provenance.  Older checkpoints and Stage-2 bundles predate a duplicated
    combined ``tt_rtc_manifest.json`` inside the bundle, so validate the two
    already authenticated runtime contracts directly instead of requiring an
    unauthenticated sidecar copy.
    """

    rtc = _as_plain_mapping(
        capability.payload.get("training_time_rtc"),
        "training_time_rtc",
    )
    horizon = rtc.get("action_horizon")
    dimension = rtc.get("action_dimension")
    _exact_integer(
        getattr(spec, "reference_horizon", None),
        horizon,
        "rlt_bundle.reference_horizon",
    )
    _exact_integer(
        getattr(spec, "action_dim", None),
        dimension,
        "rlt_bundle.action_dim",
    )
    _exact_integer(
        getattr(spec, "proprio_dim", None),
        dimension,
        "rlt_bundle.proprio_dim",
    )
    _exact_integer(
        getattr(spec, "chunk_length", None),
        TT_RTC_RLT_CHUNK_LENGTH,
        "rlt_bundle.chunk_length",
    )

    max_delay_steps = rtc.get("max_delay_steps")
    if (
        type(horizon) is not int
        or type(max_delay_steps) is not int
        or horizon - max_delay_steps < TT_RTC_RLT_CHUNK_LENGTH
    ):
        raise TTRTCContractError(
            "TT-RTC checkpoint cannot provide a full delayed RLT action chunk"
        )
    # Schema v1 actors, including the 8ecc baseline, were trained as direct
    # action actors and accept the complete reference slice selected by the
    # TT-RTC caller. Schema v2 actors bind one fixed delayed slice in their
    # signed training contract and must continue to match it exactly.
    if getattr(spec, "schema_version", None) != 1:
        _exact_integer(
            getattr(spec, "reference_offset_steps", None),
            max_delay_steps,
            "rlt_bundle.reference_offset_steps",
        )


def validate_tt_rtc_model_contract(
    capability: TTRTCCapability,
    model_contract: Mapping[str, Any],
) -> None:
    """Cross-check the manifest against the model and processor actually loaded."""

    if model_contract.get("enabled") is not True:
        raise TTRTCContractError(
            "Loaded GR00T config does not enable training_time_rtc_enabled"
        )
    rtc = _as_plain_mapping(
        capability.payload.get("training_time_rtc"),
        "training_time_rtc",
    )
    comparisons = (
        ("max_delay_steps", rtc["max_delay_steps"]),
        ("action_horizon", rtc["action_horizon"]),
        ("action_dimension", rtc["action_dimension"]),
        ("model_action_horizon", TT_RTC_MODEL_ACTION_HORIZON),
        ("model_action_dimension", TT_RTC_MODEL_ACTION_DIM),
        ("processor_action_horizon", TT_RTC_PROCESSOR_ACTION_HORIZON),
        ("processor_action_dimension", TT_RTC_PROCESSOR_ACTION_DIM),
        ("num_inference_timesteps", TT_RTC_NUM_INFERENCE_TIMESTEPS),
        ("num_timestep_buckets", TT_RTC_NUM_TIMESTEP_BUCKETS),
    )
    for name, expected in comparisons:
        value = model_contract.get(name)
        _exact_integer(value, expected, f"loaded_model.{name}")
        _exact_integer(rtc.get(name), value, f"manifest_vs_loaded_model.{name}")

    action_hz = model_contract.get("action_hz")
    _exact_number(action_hz, TT_RTC_ACTION_HZ, "loaded_model.action_hz")
    _exact_number(
        rtc.get("action_hz"),
        float(action_hz),
        "manifest_vs_loaded_model.action_hz",
    )


__all__ = [
    "TT_RTC_ACTION_DIM",
    "TT_RTC_ACTION_HORIZON",
    "TT_RTC_ACTION_HZ",
    "TT_RTC_MANIFEST_NAME",
    "TT_RTC_MODEL_ACTION_DIM",
    "TT_RTC_MODEL_ACTION_HORIZON",
    "TT_RTC_MAX_DELAY_STEPS",
    "TT_RTC_NUM_INFERENCE_TIMESTEPS",
    "TT_RTC_NUM_TIMESTEP_BUCKETS",
    "TT_RTC_PROCESSOR_ACTION_DIM",
    "TT_RTC_PROCESSOR_ACTION_HORIZON",
    "TT_RTC_PROFILE_DELAY_RANGES",
    "TT_RTC_REQUEST_MODE",
    "TT_RTC_RLT_CONTEXT_SCHEMA",
    "TT_RTC_RLT_CHUNK_LENGTH",
    "TT_RTC_SCHEMA",
    "TT_RTC_TRAINING_RECORD_SCHEMA",
    "TTRTCCapability",
    "TTRTCContractError",
    "TTRTCRequest",
    "load_tt_rtc_capability",
    "parse_tt_rtc_request",
    "validate_tt_rtc_model_contract",
    "validate_tt_rtc_rlt_binding",
]
