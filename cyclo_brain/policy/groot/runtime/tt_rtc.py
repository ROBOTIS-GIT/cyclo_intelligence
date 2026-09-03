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


TT_RTC_REQUEST_MODE = "tt_rtc"
TT_RTC_SCHEMA = "cyclo.training-time-rtc/v1"
TT_RTC_RLT_CONTEXT_SCHEMA = "cyclo.groot.tt-rtc-rlt-context/v1"
TT_RTC_ACTION_HORIZON = 16
TT_RTC_ACTION_DIM = 19
TT_RTC_MAX_DELAY_STEPS = 6
TT_RTC_ACTION_HZ = 15.0
TT_RTC_RLT_CHUNK_LENGTH = 10
TT_RTC_MANIFEST_NAME = "tt_rtc_manifest.json"


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


def parse_tt_rtc_request(request: object) -> TTRTCRequest | None:
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

    delay_steps = getattr(request, "rtc_delay_steps", None)
    if (
        isinstance(delay_steps, bool)
        or not isinstance(delay_steps, int)
        or not 0 <= delay_steps <= TT_RTC_MAX_DELAY_STEPS
    ):
        raise TTRTCContractError(
            f"TT-RTC rtc_delay_steps must be an integer in 0..{TT_RTC_MAX_DELAY_STEPS}"
        )

    action_dim = getattr(request, "rtc_action_dim", None)
    if (
        isinstance(action_dim, bool)
        or not isinstance(action_dim, int)
        or action_dim != TT_RTC_ACTION_DIM
    ):
        raise TTRTCContractError(
            f"TT-RTC rtc_action_dim must be {TT_RTC_ACTION_DIM}"
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


def _read_manifest(model_path: str | os.PathLike[str]) -> tuple[Path, Mapping[str, Any]]:
    root = Path(os.path.abspath(Path(model_path).expanduser()))
    if root.is_symlink() or not root.is_dir():
        raise TTRTCContractError(
            f"TT-RTC model path must be a non-symlink directory: {root}"
        )

    dedicated = root / TT_RTC_MANIFEST_NAME
    config = root / "config.json"
    candidates = (dedicated, config)
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
        if path == config and "training_time_rtc" not in payload:
            continue
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
    _exact_integer(rtc.get("action_horizon"), TT_RTC_ACTION_HORIZON, "action_horizon")
    _exact_integer(rtc.get("action_dimension"), TT_RTC_ACTION_DIM, "action_dimension")
    _exact_number(rtc.get("action_hz"), TT_RTC_ACTION_HZ, "action_hz")
    _exact_integer(rtc.get("max_delay_steps"), TT_RTC_MAX_DELAY_STEPS, "max_delay_steps")
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
    _exact_integer(delay_sampling.get("min_inclusive"), 0, "delay_sampling.min_inclusive")
    _exact_integer(
        delay_sampling.get("max_inclusive"),
        TT_RTC_MAX_DELAY_STEPS,
        "delay_sampling.max_inclusive",
    )

    flow = _as_plain_mapping(rtc.get("flow_convention"), "flow_convention")
    _exact_number(flow.get("noise_endpoint"), 0.0, "flow_convention.noise_endpoint")
    _exact_number(flow.get("clean_endpoint"), 1.0, "flow_convention.clean_endpoint")
    if flow.get("velocity_target") != "action_minus_noise":
        raise TTRTCContractError(
            "TT-RTC manifest flow_convention.velocity_target must be 'action_minus_noise'"
        )

    if TT_RTC_ACTION_HORIZON - TT_RTC_MAX_DELAY_STEPS < TT_RTC_RLT_CHUNK_LENGTH:
        raise AssertionError("Internal TT-RTC horizon contract is inconsistent")

    if require_rlt:
        rlt = _as_plain_mapping(payload.get("rlt"), "rlt")
        _exact_integer(rlt.get("chunk_length"), TT_RTC_RLT_CHUNK_LENGTH, "rlt.chunk_length")
        _exact_integer(
            rlt.get("reference_horizon"),
            TT_RTC_ACTION_HORIZON,
            "rlt.reference_horizon",
        )
        if rlt.get("reference_slice") != "[d:d+10]":
            raise TTRTCContractError(
                "TT-RTC RLT manifest reference_slice must be '[d:d+10]'"
            )

    return TTRTCCapability(source=source, payload=payload)


__all__ = [
    "TT_RTC_ACTION_DIM",
    "TT_RTC_ACTION_HORIZON",
    "TT_RTC_ACTION_HZ",
    "TT_RTC_MANIFEST_NAME",
    "TT_RTC_MAX_DELAY_STEPS",
    "TT_RTC_REQUEST_MODE",
    "TT_RTC_RLT_CONTEXT_SCHEMA",
    "TT_RTC_RLT_CHUNK_LENGTH",
    "TT_RTC_SCHEMA",
    "TTRTCCapability",
    "TTRTCContractError",
    "TTRTCRequest",
    "load_tt_rtc_capability",
    "parse_tt_rtc_request",
]
