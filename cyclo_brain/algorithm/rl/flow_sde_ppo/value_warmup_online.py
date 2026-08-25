"""Fail-closed transfer of an offline value warm-up into online Flow-SDE PPO.

The warm-up bundle is an initialization artifact, not a deployable actor.  This
module verifies that the bundle was completed for the *exact* policy and task
used by the online job, restores the value MLP and its AdamW moments, and then
returns immutable provenance suitable for online checkpoints and manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .value_warmup import VALUE_WARMUP_FORMAT, module_sha256
from .value_warmup_cli import BUNDLE_FORMAT, REQUIRED_POLICY_ARTIFACTS
from .value_warmup_eval import validate_current_value_head_state_dict


VALUE_INITIALIZATION_FORMAT = "cyclo.flow_sde_ppo.value_initialization.v1"
_EXPECTED_ARTIFACTS = {
    "model_path": "pretrained_model",
    "checkpoint_path": "training_state/value_warmup.pt",
    "progress_path": "progress.jsonl",
}


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} is missing: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(f"required policy artifact is missing: {path}") from None
    return f"sha256:{digest.hexdigest()}"


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash a module state using the same exact contract as ``module_sha256``."""

    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("value-head state_dict must map string names to tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _validate_manifest(bundle: Path, manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
    if manifest.get("format") != BUNDLE_FORMAT or manifest.get("status") != "complete":
        raise ValueError("value warm-up bundle must have complete v1 format")
    if manifest.get("artifacts") != _EXPECTED_ARTIFACTS:
        raise ValueError("value warm-up bundle artifact contract changed")
    for relative in _EXPECTED_ARTIFACTS.values():
        path = bundle / relative
        if relative == "pretrained_model":
            if not path.is_dir():
                raise FileNotFoundError(f"value warm-up model artifact is missing: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"value warm-up artifact is missing: {path}")

    config = _mapping(manifest.get("config"), name="value warm-up manifest config")
    result = _mapping(manifest.get("result"), name="value warm-up manifest result")
    steps = config.get("steps")
    completed = result.get("completed_steps")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or completed != steps
    ):
        raise ValueError("value warm-up bundle did not complete all configured optimizer steps")
    return config, steps


def _validate_policy_identity(
    *,
    bundle: Path,
    base_checkpoint: Path,
    policy: nn.Module,
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> str:
    base = _mapping(manifest.get("base"), name="value warm-up base identity")
    if checkpoint.get("base_identity") != base:
        raise ValueError("value warm-up checkpoint and manifest base identities differ")
    expected_module_sha = base.get("policy_sha256")
    if not isinstance(expected_module_sha, str) or not expected_module_sha.startswith("sha256:"):
        raise ValueError("value warm-up base policy digest is invalid")
    if (
        checkpoint.get("policy_sha256_before") != expected_module_sha
        or checkpoint.get("policy_sha256_after") != expected_module_sha
        or _mapping(manifest.get("result"), name="manifest result").get(
            "policy_sha256_before"
        )
        != expected_module_sha
        or manifest["result"].get("policy_sha256_after") != expected_module_sha
    ):
        raise ValueError("value warm-up reports a policy mutation")

    actual_module_sha = module_sha256(policy)
    if actual_module_sha != expected_module_sha:
        raise ValueError(
            "online base policy does not exactly match the value warm-up policy module"
        )

    expected_artifacts = _mapping(base.get("artifacts"), name="base policy artifacts")
    if set(expected_artifacts) != set(REQUIRED_POLICY_ARTIFACTS):
        raise ValueError("value warm-up base policy artifact set changed")
    for name in REQUIRED_POLICY_ARTIFACTS:
        expected = expected_artifacts[name]
        if not isinstance(expected, str) or not expected.startswith("sha256:"):
            raise ValueError(f"value warm-up policy artifact digest is invalid: {name}")
        if _file_sha256(base_checkpoint / name) != expected:
            raise ValueError(f"online base policy artifact does not match warm-up: {name}")
        if _file_sha256(bundle / "pretrained_model" / name) != expected:
            raise ValueError(f"bundled policy artifact does not match warm-up manifest: {name}")
    return actual_module_sha


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    completed_steps: int,
) -> tuple[Mapping[str, Tensor], Mapping[str, Any]]:
    if checkpoint.get("format") != VALUE_WARMUP_FORMAT or checkpoint.get("status") != "complete":
        raise ValueError("value warm-up checkpoint must have complete v1 format")
    if checkpoint.get("config") != manifest.get("config"):
        raise ValueError("value warm-up checkpoint and manifest configs differ")
    if checkpoint.get("dataset_contract") != manifest.get("dataset_contract"):
        raise ValueError("value warm-up checkpoint and manifest dataset contracts differ")
    if checkpoint.get("dataset_identities") != manifest.get("datasets"):
        raise ValueError("value warm-up checkpoint and manifest dataset identities differ")
    if checkpoint.get("completed_steps") != completed_steps:
        raise ValueError("value warm-up checkpoint completed step count differs")
    value_state = _mapping(checkpoint.get("value_head"), name="value-head checkpoint state")
    optimizer_state = _mapping(
        checkpoint.get("value_optimizer"), name="value optimizer checkpoint state"
    )
    return value_state, optimizer_state  # type: ignore[return-value]


def _strict_optimizer_restore(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    parameters: Sequence[nn.Parameter],
    *,
    completed_steps: int,
) -> tuple[float, float, int]:
    """Restore AdamW moments while retaining the online job's configured LR.

    Warm-up and online PPO use the same AdamW defaults.  Momentum/variance and
    step counters continue, but learning rate is an online PPO runtime control,
    so loading the bundle must not silently replace it with the warm-up LR.
    """

    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("online value optimizer must be torch.optim.AdamW")
    saved_groups = optimizer_state.get("param_groups")
    saved_states = optimizer_state.get("state")
    if not isinstance(saved_groups, list) or len(saved_groups) != 1:
        raise ValueError("value warm-up optimizer must contain exactly one parameter group")
    if not isinstance(saved_states, Mapping):
        raise TypeError("value warm-up optimizer state must be a mapping")
    saved_group = _mapping(saved_groups[0], name="value optimizer parameter group")
    parameter_ids = saved_group.get("params")
    if not isinstance(parameter_ids, list) or len(parameter_ids) != len(parameters):
        raise ValueError("value warm-up optimizer parameter count does not match value head")
    if len(set(parameter_ids)) != len(parameter_ids) or set(saved_states) != set(parameter_ids):
        raise ValueError("value warm-up optimizer state does not cover every value parameter")

    runtime_group = optimizer.param_groups[0]
    runtime_lr = float(runtime_group["lr"])
    warmup_lr = float(saved_group.get("lr", math.nan))
    if not math.isfinite(warmup_lr) or warmup_lr <= 0.0:
        raise ValueError("value warm-up optimizer learning rate is invalid")
    # Everything except parameter identifiers and learning rate must match the
    # current AdamW contract. This fails before ``load_state_dict`` on a future
    # optimizer/hyperparameter change rather than accepting a silent mismatch.
    ignored = {"params", "lr", "initial_lr"}
    for name, current in runtime_group.items():
        if name in ignored:
            continue
        if name not in saved_group or saved_group[name] != current:
            raise ValueError(f"value warm-up AdamW hyperparameter mismatch: {name}")

    steps: list[int] = []
    for parameter_id, parameter in zip(parameter_ids, parameters, strict=True):
        state = _mapping(saved_states[parameter_id], name="value optimizer parameter state")
        step = state.get("step")
        if isinstance(step, Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()):
                raise ValueError("value optimizer step must be one finite scalar")
            step_value = float(step.item())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            step_value = float(step)
        else:
            raise TypeError("value optimizer step must be numeric")
        if not step_value.is_integer() or int(step_value) != completed_steps:
            raise ValueError("value optimizer step does not match completed warm-up steps")
        steps.append(int(step_value))
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(name)
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"value optimizer {name} is incompatible with value parameter")

    optimizer.load_state_dict(dict(optimizer_state))
    for group in optimizer.param_groups:
        group["lr"] = runtime_lr
        if "initial_lr" in group:
            group["initial_lr"] = runtime_lr
    return warmup_lr, runtime_lr, min(steps)


def load_value_warmup_bundle(
    bundle_path: str | Path,
    *,
    base_checkpoint: str | Path,
    policy: nn.Module,
    value_head: nn.Module,
    value_optimizer: torch.optim.Optimizer,
    conditioning_dim: int,
    task_instruction: str,
) -> dict[str, Any]:
    """Strictly initialize online PPO's critic from one complete warm-up bundle."""

    bundle = Path(bundle_path).expanduser().resolve(strict=True)
    if not bundle.is_dir():
        raise NotADirectoryError(bundle)
    base = Path(base_checkpoint).expanduser().resolve(strict=True)
    manifest = _read_json(bundle / "run_manifest.json", name="value warm-up manifest")
    config, completed_steps = _validate_manifest(bundle, manifest)
    if config.get("task_instruction") != task_instruction:
        raise ValueError(
            "online task instruction must exactly match the value warm-up task instruction"
        )
    checkpoint_path = bundle / _EXPECTED_ARTIFACTS["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("value warm-up checkpoint must contain a mapping")
    value_state, optimizer_state = _validate_checkpoint(
        checkpoint, manifest, completed_steps=completed_steps
    )
    policy_sha = _validate_policy_identity(
        bundle=bundle,
        base_checkpoint=base,
        policy=policy,
        manifest=manifest,
        checkpoint=checkpoint,
    )
    architecture = validate_current_value_head_state_dict(
        value_state, conditioning_dim=conditioning_dim
    )
    expected_current = value_head.state_dict()
    if set(expected_current) != set(value_state):
        raise ValueError("online value-head tensor keys do not match warm-up architecture")
    for name, expected in expected_current.items():
        actual = value_state[name]
        if (
            not isinstance(actual, Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not bool(torch.isfinite(actual).all())
        ):
            raise ValueError(f"online value-head tensor contract mismatch: {name}")

    checkpoint_value_sha = _state_dict_sha256(value_state)
    value_head.load_state_dict(value_state, strict=True)
    loaded_value_sha = module_sha256(value_head)
    if loaded_value_sha != checkpoint_value_sha:
        raise RuntimeError("online value head did not exactly reload warm-up tensors")
    parameters = tuple(parameter for parameter in value_head.parameters() if parameter.requires_grad)
    warmup_lr, runtime_lr, optimizer_step = _strict_optimizer_restore(
        value_optimizer,
        optimizer_state,
        parameters,
        completed_steps=completed_steps,
    )
    return {
        "format": VALUE_INITIALIZATION_FORMAT,
        "source": "offline_value_warmup",
        "bundle_path": str(bundle),
        "bundle_format": BUNDLE_FORMAT,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": VALUE_WARMUP_FORMAT,
        "completed_steps": completed_steps,
        "task_instruction": task_instruction,
        "base_policy_sha256": policy_sha,
        "checkpoint_value_sha256": checkpoint_value_sha,
        "loaded_value_sha256": loaded_value_sha,
        "exact_value_reload": True,
        "optimizer_state_continued": True,
        "optimizer_step": optimizer_step,
        "warmup_value_learning_rate": warmup_lr,
        "online_value_learning_rate": runtime_lr,
        "value_head_architecture": architecture,
    }


__all__ = [
    "VALUE_INITIALIZATION_FORMAT",
    "load_value_warmup_bundle",
]
