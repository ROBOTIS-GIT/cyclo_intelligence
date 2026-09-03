"""Durable, fail-closed artifacts between Flow-SDE collection and PPO update.

The payload intentionally contains only tensors and Python primitives so it
can be loaded with ``torch.load(..., weights_only=True)``.  Reconstructing the
public rollout dataclasses re-runs their shape, dtype, device, and finite-value
validation before an optimizer is allowed to consume the data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .batch import FlowSDERollout
from .on_policy import FlowSDEEpisode, FlowSDETransition


ROLLOUT_BUNDLE_FORMAT = "cyclo.flow_sde_ppo.rollout.bundle.v1"
ROLLOUT_PAYLOAD_FORMAT = "cyclo.flow_sde_ppo.rollout.payload.v1"
POLICY_IDENTITY_FORMAT = "cyclo.flow_sde_ppo.policy_identity.v1"
SOURCE_POLICY_FORMAT = "cyclo.flow_sde_ppo.source_policy.v1"
ROLLOUT_CONSUMPTION_FORMAT = "cyclo.flow_sde_ppo.rollout.consumption.v1"
TRAINER_CHECKPOINT_FORMAT = "cyclo.flow_sde_ppo.training.v1"
ROLLOUT_PAYLOAD_NAME = "rollout.pt"
ROLLOUT_MANIFEST_NAME = "manifest.json"
ROLLOUT_CONSUMPTION_NAME = "consumption.json"
SOURCE_TRAINER_STATE_NAME = "source_training_state/trainer_state.pt"


@dataclass(frozen=True)
class LoadedFlowSDERolloutBundle:
    """A sealed rollout bundle that passed integrity and contract checks."""

    path: Path
    source_trainer_checkpoint: Path
    episodes: tuple[FlowSDEEpisode, ...]
    policy_identity: dict[str, Any]
    source_policy: dict[str, Any]
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    consumption_receipt: dict[str, Any] | None


def _canonical_json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON-compatible data") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return decoded


def _validate_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + 64
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return value


def validate_policy_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact actor/critic/config identity that created a rollout."""

    canonical = _canonical_json_object(identity, name="Flow-SDE policy identity")
    required = {
        "format",
        "source_update_step",
        "actor_sha256",
        "critic_sha256",
        "ppo_config",
    }
    if set(canonical) != required:
        missing = sorted(required - set(canonical))
        extra = sorted(set(canonical) - required)
        raise ValueError(
            "Flow-SDE policy identity fields changed"
            f" (missing={missing}, extra={extra})"
        )
    if canonical["format"] != POLICY_IDENTITY_FORMAT:
        raise ValueError("Flow-SDE policy identity format is unsupported")
    update_step = canonical["source_update_step"]
    if isinstance(update_step, bool) or not isinstance(update_step, int) or update_step < 0:
        raise ValueError("Flow-SDE policy identity update step must be non-negative")
    _validate_sha256(canonical["actor_sha256"], name="Flow-SDE actor identity")
    _validate_sha256(canonical["critic_sha256"], name="Flow-SDE critic identity")
    if not isinstance(canonical["ppo_config"], dict) or not canonical["ppo_config"]:
        raise ValueError("Flow-SDE policy identity PPO config must be a non-empty object")
    return canonical


def validate_source_policy(source_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate everything needed to reconstruct the frozen policy process."""

    canonical = _canonical_json_object(source_policy, name="Flow-SDE source policy")
    required = {
        "format",
        "checkpoint_path",
        "artifacts",
        "frozen_policy_sha256",
        "policy_contract",
        "critic_contract",
        "task_instruction",
        "robot_type",
    }
    if set(canonical) != required:
        raise ValueError("Flow-SDE source policy fields changed")
    if canonical["format"] != SOURCE_POLICY_FORMAT:
        raise ValueError("Flow-SDE source policy format is unsupported")
    checkpoint_path = canonical["checkpoint_path"]
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise ValueError("Flow-SDE source policy checkpoint path must be absolute")
    artifacts = canonical["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Flow-SDE source policy artifacts must be non-empty")
    for name, digest in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Flow-SDE source policy artifact names must be non-empty")
        _validate_sha256(digest, name=f"Flow-SDE source policy artifact {name}")
    _validate_sha256(
        canonical["frozen_policy_sha256"], name="Flow-SDE frozen policy identity"
    )
    for name in ("policy_contract", "critic_contract"):
        if not isinstance(canonical[name], dict) or not canonical[name]:
            raise ValueError(f"Flow-SDE source policy {name} must be non-empty")
    for name in ("task_instruction", "robot_type"):
        if not isinstance(canonical[name], str) or not canonical[name].strip():
            raise ValueError(f"Flow-SDE source policy {name} must be non-empty")
    return canonical


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _state_dict_sha256(state: Mapping[str, Any], *, name: str) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Flow-SDE {name} state must be a non-empty mapping")
    digest = hashlib.sha256()
    for tensor_name, tensor in state.items():
        if not isinstance(tensor_name, str) or not isinstance(tensor, Tensor):
            raise ValueError(f"Flow-SDE {name} state must contain named tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(tensor_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _validate_source_training_state(
    state: Mapping[str, Any],
    *,
    policy_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Flow-SDE rollout source training state must be a non-empty mapping")
    required = {
        "format",
        "config",
        "update_step",
        "actor",
        "value_head",
        "actor_optimizer",
        "value_optimizer",
        "torch_rng_state",
    }
    if not required.issubset(state):
        raise ValueError("Flow-SDE rollout source training state is incomplete")
    if state["format"] != TRAINER_CHECKPOINT_FORMAT:
        raise ValueError("Flow-SDE rollout source trainer format is unsupported")
    if state["config"] != policy_identity["ppo_config"]:
        raise ValueError("Flow-SDE rollout source trainer PPO config does not match")
    if state["update_step"] != policy_identity["source_update_step"]:
        raise ValueError("Flow-SDE rollout source trainer update step does not match")
    if _state_dict_sha256(state["actor"], name="actor") != policy_identity["actor_sha256"]:
        raise ValueError("Flow-SDE rollout source trainer actor does not match")
    if (
        _state_dict_sha256(state["value_head"], name="critic")
        != policy_identity["critic_sha256"]
    ):
        raise ValueError("Flow-SDE rollout source trainer critic does not match")
    for name in ("actor_optimizer", "value_optimizer"):
        if not isinstance(state[name], Mapping) or not state[name]:
            raise ValueError(f"Flow-SDE rollout source trainer {name} is invalid")
    rng_state = state["torch_rng_state"]
    if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8:
        raise ValueError("Flow-SDE rollout source trainer RNG state is invalid")
    return dict(state)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _episode_to_payload(episode: FlowSDEEpisode) -> dict[str, Any]:
    if not isinstance(episode, FlowSDEEpisode):
        raise TypeError("Flow-SDE rollout bundle only accepts complete episodes")
    transitions = episode.transitions
    return {
        "conditioning": torch.stack(
            [item.conditioning.detach().cpu() for item in transitions]
        ),
        "chains": torch.cat(
            [item.rollout.chains.detach().cpu() for item in transitions], dim=0
        ),
        "denoise_indices": torch.cat(
            [item.rollout.denoise_indices.detach().cpu() for item in transitions], dim=0
        ),
        "old_log_probs": torch.cat(
            [item.rollout.old_log_probs.detach().cpu() for item in transitions], dim=0
        ),
        "action_mask": torch.cat(
            [item.rollout.action_mask.detach().cpu() for item in transitions], dim=0
        ),
        "rewards": torch.tensor(
            [item.reward for item in transitions], dtype=torch.float64
        ),
        "terminated": torch.tensor(
            [item.terminated for item in transitions], dtype=torch.bool
        ),
        "truncated": torch.tensor(
            [item.truncated for item in transitions], dtype=torch.bool
        ),
        "old_values": torch.tensor(
            [item.old_value for item in transitions], dtype=torch.float64
        ),
        "bootstrap_value": torch.tensor(episode.bootstrap_value, dtype=torch.float64),
    }


def _require_tensor(payload: Mapping[str, Any], name: str) -> Tensor:
    value = payload.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"Flow-SDE rollout payload {name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"Flow-SDE rollout payload {name} must be stored on CPU")
    return value


def _episode_from_payload(payload: Mapping[str, Any]) -> FlowSDEEpisode:
    if not isinstance(payload, Mapping):
        raise ValueError("Flow-SDE rollout episode payload must be an object")
    required = {
        "conditioning",
        "chains",
        "denoise_indices",
        "old_log_probs",
        "action_mask",
        "rewards",
        "terminated",
        "truncated",
        "old_values",
        "bootstrap_value",
    }
    if set(payload) != required:
        raise ValueError("Flow-SDE rollout episode payload fields changed")

    conditioning = _require_tensor(payload, "conditioning")
    chains = _require_tensor(payload, "chains")
    denoise_indices = _require_tensor(payload, "denoise_indices")
    old_log_probs = _require_tensor(payload, "old_log_probs")
    action_mask = _require_tensor(payload, "action_mask")
    rewards = _require_tensor(payload, "rewards")
    terminated = _require_tensor(payload, "terminated")
    truncated = _require_tensor(payload, "truncated")
    old_values = _require_tensor(payload, "old_values")
    bootstrap_value = _require_tensor(payload, "bootstrap_value")

    if conditioning.ndim != 2 or conditioning.shape[0] < 1:
        raise ValueError("Flow-SDE stored conditioning must have non-empty shape (K, C)")
    transitions_count = conditioning.shape[0]
    if chains.ndim != 4 or chains.shape[0] != transitions_count:
        raise ValueError("Flow-SDE stored chains must have shape (K, N+1, H, A)")
    if denoise_indices.shape != (transitions_count,) or denoise_indices.dtype != torch.long:
        raise ValueError("Flow-SDE stored denoise indices must be int64 with shape (K,)")
    expected_action_shape = (transitions_count, chains.shape[2], chains.shape[3])
    if old_log_probs.shape != expected_action_shape or old_log_probs.dtype != torch.float32:
        raise ValueError("Flow-SDE stored old log-probabilities have an invalid contract")
    if action_mask.shape != expected_action_shape or action_mask.dtype != torch.bool:
        raise ValueError("Flow-SDE stored action mask has an invalid contract")
    for name, value, dtype in (
        ("rewards", rewards, torch.float64),
        ("old_values", old_values, torch.float64),
        ("terminated", terminated, torch.bool),
        ("truncated", truncated, torch.bool),
    ):
        if value.shape != (transitions_count,) or value.dtype != dtype:
            raise ValueError(f"Flow-SDE stored {name} has an invalid contract")
    if bootstrap_value.shape != () or bootstrap_value.dtype != torch.float64:
        raise ValueError("Flow-SDE stored bootstrap value must be a float64 scalar")

    transitions: list[FlowSDETransition] = []
    for index in range(transitions_count):
        rollout = FlowSDERollout(
            chains=chains[index : index + 1],
            denoise_indices=denoise_indices[index : index + 1],
            old_log_probs=old_log_probs[index : index + 1],
            action_mask=action_mask[index : index + 1],
        )
        transitions.append(
            FlowSDETransition(
                conditioning=conditioning[index],
                rollout=rollout,
                reward=float(rewards[index]),
                terminated=bool(terminated[index]),
                truncated=bool(truncated[index]),
                old_value=float(old_values[index]),
            )
        )
    return FlowSDEEpisode(
        tuple(transitions),
        bootstrap_value=float(bootstrap_value),
    )


def _bundle_contract(episodes: Sequence[FlowSDEEpisode]) -> dict[str, int]:
    first = episodes[0].transitions[0]
    tensor_contract = (
        first.conditioning.shape[0],
        first.rollout.chains.shape[1],
        first.rollout.chains.shape[2],
        first.rollout.chains.shape[3],
    )
    for episode in episodes:
        candidate = episode.transitions[0]
        if (
            candidate.conditioning.shape[0],
            candidate.rollout.chains.shape[1],
            candidate.rollout.chains.shape[2],
            candidate.rollout.chains.shape[3],
        ) != tensor_contract:
            raise ValueError("Flow-SDE rollout bundle mixes incompatible tensor contracts")
    return {
        "episodes": len(episodes),
        "transitions": sum(len(episode.transitions) for episode in episodes),
        "conditioning_dim": tensor_contract[0],
        "chain_length": tensor_contract[1],
        "horizon": tensor_contract[2],
        "action_dim": tensor_contract[3],
    }


def save_rollout_bundle(
    path: str | Path,
    episodes: Sequence[FlowSDEEpisode],
    *,
    policy_identity: Mapping[str, Any],
    source_policy: Mapping[str, Any],
    source_training_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Seal complete on-policy episodes in a new immutable directory."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("Flow-SDE rollout bundle target cannot be a symbolic link")
    resolved = requested.resolve()
    if resolved.exists():
        raise FileExistsError(f"Flow-SDE rollout bundle already exists: {resolved}")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise TypeError("Flow-SDE rollout episodes must be a sequence")
    episode_tuple = tuple(episodes)
    if not episode_tuple:
        raise ValueError("Flow-SDE rollout bundle requires at least one episode")
    payload_episodes = [_episode_to_payload(episode) for episode in episode_tuple]
    identity = validate_policy_identity(policy_identity)
    canonical_source_policy = validate_source_policy(source_policy)
    validated_training_state = _validate_source_training_state(
        source_training_state,
        policy_identity=identity,
    )
    canonical_metadata = _canonical_json_object(
        metadata or {}, name="Flow-SDE rollout metadata"
    )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    )
    try:
        source_checkpoint = temporary / SOURCE_TRAINER_STATE_NAME
        source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(validated_training_state, source_checkpoint)
        source_checkpoint_sha256 = _sha256_file(source_checkpoint)
        payload_path = temporary / ROLLOUT_PAYLOAD_NAME
        torch.save(
            {
                "format": ROLLOUT_PAYLOAD_FORMAT,
                "source_trainer_sha256": source_checkpoint_sha256,
                "episodes": payload_episodes,
            },
            payload_path,
        )
        manifest = {
            "format": ROLLOUT_BUNDLE_FORMAT,
            "status": "sealed",
            "created_at_unix": time.time(),
            "payload": {
                "path": ROLLOUT_PAYLOAD_NAME,
                "sha256": _sha256_file(payload_path),
            },
            "source_trainer_checkpoint": {
                "path": SOURCE_TRAINER_STATE_NAME,
                "sha256": source_checkpoint_sha256,
            },
            "contract": _bundle_contract(episode_tuple),
            "policy_identity": identity,
            "source_policy": canonical_source_policy,
            "metadata": canonical_metadata,
        }
        manifest_path = temporary / ROLLOUT_MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_file(payload_path)
        _fsync_file(source_checkpoint)
        _fsync_directory(source_checkpoint.parent)
        _fsync_directory(temporary)
        if resolved.exists():
            raise FileExistsError(f"Flow-SDE rollout bundle already exists: {resolved}")
        os.replace(temporary, resolved)
        _fsync_directory(resolved.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return resolved


def load_rollout_bundle(
    path: str | Path,
    *,
    expected_policy_identity: Mapping[str, Any] | None = None,
) -> LoadedFlowSDERolloutBundle:
    """Load a sealed bundle and reject any identity or payload mismatch."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("Flow-SDE rollout bundle cannot be a symbolic link")
    resolved = requested.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Flow-SDE rollout bundle must be a real directory: {resolved}")
    manifest_path = resolved / ROLLOUT_MANIFEST_NAME
    payload_path = resolved / ROLLOUT_PAYLOAD_NAME
    source_checkpoint = resolved / SOURCE_TRAINER_STATE_NAME
    consumption_path = resolved / ROLLOUT_CONSUMPTION_NAME
    if (
        manifest_path.is_symlink()
        or payload_path.is_symlink()
        or source_checkpoint.parent.is_symlink()
        or source_checkpoint.is_symlink()
        or consumption_path.is_symlink()
    ):
        raise ValueError("Flow-SDE rollout bundle artifacts cannot be symbolic links")
    if (
        not manifest_path.is_file()
        or not payload_path.is_file()
        or not source_checkpoint.is_file()
    ):
        raise FileNotFoundError("Flow-SDE rollout bundle is incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Flow-SDE rollout manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Flow-SDE rollout manifest must be an object")
    if manifest.get("format") != ROLLOUT_BUNDLE_FORMAT or manifest.get("status") != "sealed":
        raise ValueError("Flow-SDE rollout bundle is not sealed or uses an unknown format")
    payload_manifest = manifest.get("payload")
    if (
        not isinstance(payload_manifest, dict)
        or payload_manifest.get("path") != ROLLOUT_PAYLOAD_NAME
    ):
        raise ValueError("Flow-SDE rollout payload manifest is invalid")
    expected_payload_sha = _validate_sha256(
        payload_manifest.get("sha256"), name="Flow-SDE rollout payload"
    )
    if _sha256_file(payload_path) != expected_payload_sha:
        raise ValueError("Flow-SDE rollout payload failed its integrity check")
    checkpoint_manifest = manifest.get("source_trainer_checkpoint")
    if (
        not isinstance(checkpoint_manifest, dict)
        or checkpoint_manifest.get("path") != SOURCE_TRAINER_STATE_NAME
    ):
        raise ValueError("Flow-SDE rollout source trainer manifest is invalid")
    expected_checkpoint_sha = _validate_sha256(
        checkpoint_manifest.get("sha256"), name="Flow-SDE rollout source trainer"
    )
    if _sha256_file(source_checkpoint) != expected_checkpoint_sha:
        raise ValueError("Flow-SDE rollout source trainer failed its integrity check")

    identity = validate_policy_identity(manifest.get("policy_identity", {}))
    if expected_policy_identity is not None:
        expected = validate_policy_identity(expected_policy_identity)
        if identity != expected:
            raise ValueError("Flow-SDE rollout was generated by a different actor or critic")
    source_policy = validate_source_policy(manifest.get("source_policy", {}))
    metadata = _canonical_json_object(
        manifest.get("metadata", {}), name="Flow-SDE rollout metadata"
    )

    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != ROLLOUT_PAYLOAD_FORMAT:
        raise ValueError("Flow-SDE rollout payload format is unsupported")
    if payload.get("source_trainer_sha256") != expected_checkpoint_sha:
        raise ValueError("Flow-SDE rollout payload is bound to a different source trainer")
    encoded_episodes = payload.get("episodes")
    if not isinstance(encoded_episodes, list) or not encoded_episodes:
        raise ValueError("Flow-SDE rollout payload contains no episodes")
    episodes = tuple(_episode_from_payload(item) for item in encoded_episodes)
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or contract != _bundle_contract(episodes):
        raise ValueError("Flow-SDE rollout manifest contract does not match its tensors")
    consumption_receipt = None
    if consumption_path.exists():
        if not consumption_path.is_file():
            raise ValueError("Flow-SDE rollout consumption receipt is invalid")
        try:
            consumption_receipt = json.loads(consumption_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Flow-SDE rollout consumption receipt is not valid JSON") from error
        if (
            not isinstance(consumption_receipt, dict)
            or consumption_receipt.get("format") != ROLLOUT_CONSUMPTION_FORMAT
            or consumption_receipt.get("source_policy_identity") != identity
        ):
            raise ValueError("Flow-SDE rollout consumption receipt contract changed")
        result_identity = validate_policy_identity(
            consumption_receipt.get("result_policy_identity", {})
        )
        if result_identity["source_update_step"] != identity["source_update_step"] + 1:
            raise ValueError("Flow-SDE rollout consumption update step changed")
        _canonical_json_object(
            consumption_receipt.get("metrics", {}),
            name="Flow-SDE rollout consumption metrics",
        )
        result_checkpoint = consumption_receipt.get("trainer_checkpoint")
        if not isinstance(result_checkpoint, dict):
            raise ValueError("Flow-SDE rollout consumption checkpoint is invalid")
        checkpoint_path = result_checkpoint.get("path")
        if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
            raise ValueError("Flow-SDE rollout consumption checkpoint path is invalid")
        _validate_sha256(
            result_checkpoint.get("sha256"),
            name="Flow-SDE rollout consumption checkpoint",
        )
    return LoadedFlowSDERolloutBundle(
        path=resolved,
        source_trainer_checkpoint=source_checkpoint,
        episodes=episodes,
        policy_identity=identity,
        source_policy=source_policy,
        metadata=metadata,
        manifest=manifest,
        consumption_receipt=consumption_receipt,
    )


def mark_rollout_bundle_consumed(
    bundle: LoadedFlowSDERolloutBundle,
    *,
    result_policy_identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
    trainer_checkpoint: str | Path,
) -> Path:
    """Append one durable receipt after the updated trainer is committed."""

    if not isinstance(bundle, LoadedFlowSDERolloutBundle):
        raise TypeError("Flow-SDE consumption requires a loaded rollout bundle")
    receipt_path = bundle.path / ROLLOUT_CONSUMPTION_NAME
    if bundle.consumption_receipt is not None or receipt_path.exists():
        raise RuntimeError("Flow-SDE rollout bundle was already consumed")
    result_identity = validate_policy_identity(result_policy_identity)
    if (
        result_identity["source_update_step"]
        != bundle.policy_identity["source_update_step"] + 1
    ):
        raise ValueError("Flow-SDE rollout result must advance exactly one update step")
    canonical_metrics = _canonical_json_object(metrics, name="Flow-SDE update metrics")
    requested_checkpoint = Path(trainer_checkpoint).expanduser()
    if requested_checkpoint.is_symlink():
        raise ValueError("Flow-SDE result trainer checkpoint cannot be a symbolic link")
    resolved_checkpoint = requested_checkpoint.resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError("Flow-SDE result trainer checkpoint does not exist")
    receipt = {
        "format": ROLLOUT_CONSUMPTION_FORMAT,
        "consumed_at_unix": time.time(),
        "source_policy_identity": bundle.policy_identity,
        "result_policy_identity": result_identity,
        "metrics": canonical_metrics,
        "trainer_checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": _sha256_file(resolved_checkpoint),
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ROLLOUT_CONSUMPTION_NAME}.",
        suffix=".tmp",
        dir=bundle.path,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, receipt_path)
        except FileExistsError as error:
            raise RuntimeError("Flow-SDE rollout bundle was already consumed") from error
        temporary.unlink()
        _fsync_directory(bundle.path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt_path


__all__ = [
    "LoadedFlowSDERolloutBundle",
    "POLICY_IDENTITY_FORMAT",
    "ROLLOUT_BUNDLE_FORMAT",
    "ROLLOUT_CONSUMPTION_FORMAT",
    "ROLLOUT_CONSUMPTION_NAME",
    "ROLLOUT_MANIFEST_NAME",
    "ROLLOUT_PAYLOAD_FORMAT",
    "ROLLOUT_PAYLOAD_NAME",
    "SOURCE_TRAINER_STATE_NAME",
    "SOURCE_POLICY_FORMAT",
    "load_rollout_bundle",
    "mark_rollout_bundle_consumed",
    "save_rollout_bundle",
    "validate_policy_identity",
    "validate_source_policy",
]
