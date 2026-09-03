"""Run live on-policy Flow-SDE PPO against Cyclo's atomic SG2 simulator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cyclo_brain.model.multi_task_dit.checkpoint_validation import (
    assert_deployment_artifacts,
    validate_policy_contract,
)
from cyclo_brain.model.multi_task_dit.flow_sde_adapter import MultiTaskDiTFlowAdapter
from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead

from .config import FlowSDEPPOConfig
from .live_source import (
    AtomicOutcomeFile,
    CycloFlowSDEEpisodeSource,
    CycloLeRobotObservationSource,
    FlowSDECollectionCancelled,
    ZenohAtomicActionStepTransport,
)
from .rollout_bundle import (
    SOURCE_POLICY_FORMAT,
    load_rollout_bundle,
    mark_rollout_bundle_consumed,
    save_rollout_bundle,
)
from .runner import (
    FlowSDEPPOTrainer,
    collect_one_episode,
    update_rollout_bundle,
)
from .value_warmup import module_sha256
from .value_warmup_online import load_value_warmup_bundle


DEFAULT_TASK = "pick up the jelly bag"
ONLINE_BUNDLE_FORMAT = "cyclo.flow_sde_ppo.online.bundle.v1"
RESUME_PROVENANCE_FORMAT = "cyclo.flow_sde_ppo.resume.v1"
REQUIRED_POLICY_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


class JsonlProgress:
    """Mirror machine-readable events to stdout and one append-only file."""

    def __init__(self, path: Path, *, job_id: str) -> None:
        self.path = path
        self.job_id = job_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "job_id": self.job_id,
            "event": event,
            **payload,
        }
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)
        return record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--operation",
        choices=("combined", "collect", "update"),
        default="combined",
        help="Compatibility job, rollout-only collection, or bundle-only PPO update",
    )
    parser.add_argument(
        "--rollout-bundle",
        type=Path,
        help="One sealed rollout bundle consumed by --operation update",
    )
    parser.add_argument(
        "--control-file",
        type=Path,
        help="Atomic outcome JSON; defaults to OUTPUT/control/outcome.json",
    )
    parser.add_argument("--robot-type", default="ffw_sg2_rev1")
    parser.add_argument("--task-instruction", default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--max-chunk-decisions", type=int, default=120)
    parser.add_argument("--ack-timeout", type=float, default=5.0)
    parser.add_argument("--sensor-timeout", type=float, default=10.0)
    parser.add_argument("--actor-lr", type=float, default=3.0e-5)
    parser.add_argument("--value-lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--value-warmup-bundle",
        type=Path,
        help=(
            "Completed offline value-warmup bundle used to initialize the online "
            "critic and continue its AdamW moments"
        ),
    )
    parser.add_argument("--num-denoising-steps", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume OUTPUT/training_state/trainer_state.pt if it exists",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Explicit trainer_state.pt from a completed Flow-SDE PPO output bundle. "
            "The current base must be a verified exported, immediate, or lineage policy"
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("episodes", "ppo_epochs", "minibatch_size", "max_chunk_decisions"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("ack_timeout", "sensor_timeout", "actor_lr", "value_lr", "noise_level"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive")
    if not args.job_id.strip():
        raise ValueError("job-id must be non-empty")
    if not args.task_instruction.strip():
        raise ValueError("task-instruction must be non-empty")
    if args.operation == "update" and args.rollout_bundle is None:
        raise ValueError("rollout-bundle is required for update operation")
    if args.operation != "update" and args.rollout_bundle is not None:
        raise ValueError("rollout-bundle is only valid for update operation")
    if args.operation == "collect" and args.episodes != 1:
        raise ValueError(
            "collect operation requires exactly one episode per PPO update"
        )
    initialization_sources = (
        bool(args.resume),
        args.resume_checkpoint is not None,
        args.value_warmup_bundle is not None,
    )
    if sum(initialization_sources) > 1:
        raise ValueError(
            "resume, resume-checkpoint, and value-warmup-bundle are mutually exclusive"
        )
    if args.operation == "update" and any(initialization_sources):
        raise ValueError(
            "update operation restores initialization from its rollout bundle"
        )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the LeRobot container")
    return torch.device(requested)


def _load_assets(base_checkpoint: Path, device: torch.device):
    from lerobot.policies import get_policy_class, make_pre_post_processors

    pretrained_dir = assert_deployment_artifacts(base_checkpoint)
    config_payload = json.loads((pretrained_dir / "config.json").read_text(encoding="utf-8"))
    policy_type = config_payload.get("type")
    if policy_type != "multi_task_dit":
        raise ValueError(
            f"Flow-SDE PPO requires type='multi_task_dit', got {policy_type!r}"
        )
    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(str(pretrained_dir)).to(device).eval()
    contract = validate_policy_contract(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return pretrained_dir, policy, preprocessor, postprocessor, contract


def _reload_export(exported: Path, device: torch.device) -> dict[str, Any]:
    """Actually reload the deployable policy and both stored processors."""

    from lerobot.policies import get_policy_class, make_pre_post_processors

    pretrained_dir = assert_deployment_artifacts(exported)
    payload = json.loads((pretrained_dir / "config.json").read_text(encoding="utf-8"))
    if payload.get("type") != "multi_task_dit":
        raise RuntimeError("Exported policy changed type during Flow-SDE PPO save")
    policy_class = get_policy_class("multi_task_dit")
    reloaded = policy_class.from_pretrained(str(pretrained_dir)).to(device).eval()
    contract = validate_policy_contract(reloaded)
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        policy_cfg=reloaded.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    if not callable(loaded_preprocessor) or not callable(loaded_postprocessor):
        raise RuntimeError("Exported LeRobot processors are not callable after reload")
    del loaded_preprocessor, loaded_postprocessor, reloaded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return contract


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    """Hash one regular artifact without loading a large checkpoint in memory."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required Flow-SDE PPO artifact is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _policy_artifact_hashes(pretrained_dir: str | Path) -> dict[str, str]:
    resolved = assert_deployment_artifacts(pretrained_dir)
    return {
        name: _file_sha256(resolved / name)
        for name in REQUIRED_POLICY_ARTIFACTS
    }


def _frozen_policy_sha256(policy: torch.nn.Module) -> str:
    """Hash the policy state that PPO promises not to update."""

    if not isinstance(policy, torch.nn.Module):
        raise TypeError("Frozen policy identity requires a torch module")
    digest = hashlib.sha256()
    tensor_count = 0
    for name, tensor in policy.state_dict().items():
        if name == "noise_predictor" or name.startswith("noise_predictor."):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        tensor_count += 1
    if tensor_count < 1:
        raise ValueError("Flow-SDE PPO policy exposes no frozen state to identify")
    return f"sha256:{digest.hexdigest()}"


def _checkpoint_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.suffix:
        candidate = candidate / "trainer_state.pt"
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resolved}")
    return resolved


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_digest(value: Any, *, name: str) -> str:
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


def _validate_explicit_resume_source(
    resume_checkpoint: str | Path,
    *,
    pretrained_dir: Path,
    base_policy_artifacts: dict[str, str],
    policy: torch.nn.Module,
    task_instruction: str,
    robot_type: str,
    config: FlowSDEPPOConfig,
) -> tuple[Path, dict[str, Any], str]:
    """Validate the completed actor+critic bundle before mutating the trainer.

    Cross-job continuation is intentionally stricter than the legacy same-output
    ``--resume`` switch.  The explicit checkpoint must remain coupled to the
    exact exported, immediate, or lineage policy selected as this run's base.
    """

    checkpoint = _checkpoint_file(resume_checkpoint)
    if checkpoint.name != "trainer_state.pt" or checkpoint.parent.name != "training_state":
        raise ValueError(
            "resume-checkpoint must be a bundle training_state/trainer_state.pt"
        )
    source_root = checkpoint.parent.parent.resolve()
    manifest_path = source_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Completed Flow-SDE PPO run manifest does not exist: {manifest_path}"
        )
    try:
        manifest = _require_object(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            name="resume run manifest",
        )
    except json.JSONDecodeError as error:
        raise ValueError("Resume run manifest is not valid JSON") from error
    if manifest.get("format") != ONLINE_BUNDLE_FORMAT or manifest.get("status") != "complete":
        raise ValueError("Resume checkpoint is not from a completed online PPO bundle")
    if manifest.get("task_instruction") != task_instruction:
        raise ValueError("Resume checkpoint task instruction does not match this run")
    if manifest.get("robot_type") != robot_type:
        raise ValueError("Resume checkpoint robot type does not match this run")
    if manifest.get("ppo_config") != asdict(config):
        raise ValueError("Resume checkpoint PPO config does not match this run")

    lineage_policy_checkpoint = manifest.get("lineage_policy_checkpoint")
    if not isinstance(lineage_policy_checkpoint, str) or not Path(
        lineage_policy_checkpoint
    ).is_absolute():
        raise ValueError("Resume manifest lineage_policy_checkpoint is invalid")
    immediate_base_checkpoint = manifest.get("base_checkpoint")
    if not isinstance(immediate_base_checkpoint, str) or not Path(
        immediate_base_checkpoint
    ).is_absolute():
        raise ValueError("Resume manifest base_checkpoint is invalid")
    immediate_base_hashes = _require_object(
        manifest.get("base_policy_artifacts"),
        name="resume base policy hashes",
    )
    lineage_policy_hashes = _require_object(
        manifest.get("lineage_policy_artifacts"),
        name="resume lineage policy hashes",
    )

    artifacts = _require_object(manifest.get("artifacts"), name="resume artifacts")
    trainer_artifact = _require_object(
        artifacts.get("trainer_checkpoint"),
        name="resume trainer checkpoint artifact",
    )
    if trainer_artifact.get("path") != "training_state/trainer_state.pt":
        raise ValueError("Resume manifest trainer checkpoint path changed")
    expected_checkpoint_sha = _validate_digest(
        trainer_artifact.get("sha256"),
        name="resume trainer checkpoint hash",
    )
    if _file_sha256(checkpoint) != expected_checkpoint_sha:
        raise ValueError("Resume trainer checkpoint failed its integrity check")

    policy_artifact = _require_object(
        artifacts.get("pretrained_model"),
        name="resume pretrained policy artifact",
    )
    if policy_artifact.get("path") != "pretrained_model":
        raise ValueError("Resume manifest pretrained policy path changed")
    source_policy = (source_root / "pretrained_model").resolve()
    expected_policy_hashes = _require_object(
        policy_artifact.get("files"),
        name="resume pretrained policy hashes",
    )
    if set(expected_policy_hashes) != set(REQUIRED_POLICY_ARTIFACTS):
        raise ValueError("Resume manifest exported policy artifact set changed")
    assert_deployment_artifacts(source_policy)
    for name in REQUIRED_POLICY_ARTIFACTS:
        expected = _validate_digest(
            expected_policy_hashes[name],
            name=f"resume exported policy hash {name}",
        )
        if _file_sha256(source_policy / name) != expected:
            raise ValueError(f"Resume bundle exported policy failed integrity check: {name}")
    for contract_name, contract_hashes in (
        ("immediate", immediate_base_hashes),
        ("lineage", lineage_policy_hashes),
    ):
        if set(contract_hashes) != set(REQUIRED_POLICY_ARTIFACTS):
            raise ValueError(f"Resume manifest {contract_name} policy artifact set changed")
        for name in REQUIRED_POLICY_ARTIFACTS:
            _validate_digest(
                contract_hashes[name],
                name=f"resume {contract_name} policy hash {name}",
            )
    allowed_bases = (
        (source_policy, expected_policy_hashes, "exported"),
        (Path(immediate_base_checkpoint).resolve(), immediate_base_hashes, "immediate"),
        (Path(lineage_policy_checkpoint).resolve(), lineage_policy_hashes, "lineage"),
    )
    matched_base_kind = ""
    for candidate_path, candidate_hashes, candidate_kind in allowed_bases:
        if candidate_path != pretrained_dir.resolve():
            continue
        if set(candidate_hashes) != set(REQUIRED_POLICY_ARTIFACTS):
            raise ValueError(f"Resume manifest {candidate_kind} policy artifact set changed")
        for name in REQUIRED_POLICY_ARTIFACTS:
            expected = _validate_digest(
                candidate_hashes[name],
                name=f"resume {candidate_kind} policy hash {name}",
            )
            if base_policy_artifacts.get(name) != expected:
                raise ValueError(
                    f"Resume {candidate_kind} policy failed integrity check: {name}"
                )
        matched_base_kind = candidate_kind
        break
    if not matched_base_kind:
        raise ValueError(
            "Current base checkpoint is not the resumed run's exported, immediate, "
            "or lineage policy"
        )

    result = _require_object(manifest.get("result"), name="resume result")
    source_update_step = result.get("update_step")
    if (
        isinstance(source_update_step, bool)
        or not isinstance(source_update_step, int)
        or source_update_step < 1
    ):
        raise ValueError("Resume manifest update_step is invalid")
    expected_actor_sha = _validate_digest(
        result.get("actor_sha256"),
        name="resume actor hash",
    )
    expected_critic_sha = _validate_digest(
        result.get("critic_sha256"),
        name="resume critic hash",
    )
    expected_frozen_policy_sha = _validate_digest(
        result.get("frozen_policy_sha256"),
        name="resume frozen policy hash",
    )
    if _frozen_policy_sha256(policy) != expected_frozen_policy_sha:
        raise ValueError("Resume base has a different frozen policy body")

    provenance = {
        "format": RESUME_PROVENANCE_FORMAT,
        "mode": "explicit_checkpoint",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha,
        "source_update_step": source_update_step,
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": _file_sha256(manifest_path),
        "lineage_policy_checkpoint": lineage_policy_checkpoint,
        "lineage_policy_artifacts": dict(lineage_policy_hashes),
        "matched_base_kind": matched_base_kind,
        "source_actor_sha256": expected_actor_sha,
        "source_critic_sha256": expected_critic_sha,
        "source_frozen_policy_sha256": expected_frozen_policy_sha,
    }
    return checkpoint, provenance, expected_critic_sha


def _configure_zenoh_cache(cache_path: str | Path = "/zenoh_cache") -> str | None:
    """Use the uid-readable SDK cache mounted by docker-compose."""

    resolved = Path(cache_path)
    if not resolved.is_dir():
        return None
    os.environ.setdefault("ZENOH_ROS2_SDK_CACHE", str(resolved))
    return os.environ["ZENOH_ROS2_SDK_CACHE"]


def _configure_live_transport_environment() -> None:
    """Match the policy container's supervised inference processes."""

    os.environ.setdefault("ROS_DOMAIN_ID", "30")
    os.environ.setdefault(
        "ZENOH_CONFIG_OVERRIDE",
        "transport/shared_memory/enabled=true",
    )
    os.environ.setdefault("ZENOH_SHM_ENABLED", "true")
    os.environ.setdefault("ZENOH_TRANSPORT_SHM_ENABLED", "true")
    _configure_zenoh_cache()


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    control_file = (
        args.control_file.expanduser().resolve()
        if args.control_file is not None
        else output_dir / "control" / "outcome.json"
    )
    progress = JsonlProgress(output_dir / "progress.jsonl", job_id=args.job_id.strip())
    training_state = output_dir / "training_state" / "trainer_state.pt"
    pretrained_output = output_dir / "pretrained_model"
    rollout_bundles: list[Path] = []

    # uid=1000 cannot read /root/.cache in the policy container. The compose
    # service exposes a host-populated, read-only SDK cache at /zenoh_cache.
    _configure_live_transport_environment()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)

    progress.emit(
        "starting",
        stage="load_checkpoint",
        operation=args.operation,
        base_checkpoint=str(args.base_checkpoint),
        output_dir=str(output_dir),
        control_file=str(control_file),
        device=str(device),
        episodes=args.episodes,
        resume_checkpoint=(
            str(args.resume_checkpoint) if args.resume_checkpoint is not None else ""
        ),
        legacy_resume=bool(args.resume),
        value_warmup_bundle=(
            str(args.value_warmup_bundle) if args.value_warmup_bundle is not None else ""
        ),
        ros_domain_id=os.environ["ROS_DOMAIN_ID"],
        zenoh_sdk_cache=os.environ.get("ZENOH_ROS2_SDK_CACHE", ""),
    )
    source = None
    try:
        pretrained_dir, policy, preprocessor, postprocessor, contract = _load_assets(
            args.base_checkpoint,
            device,
        )
        adapter = MultiTaskDiTFlowAdapter(policy, freeze_observation_encoder=True)
        value_head = MultiTaskDiTValueHead(adapter.conditioning_dim).to(device)
        config = FlowSDEPPOConfig(
            num_denoising_steps=args.num_denoising_steps,
            noise_level=args.noise_level,
            actor_learning_rate=args.actor_lr,
            value_learning_rate=args.value_lr,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )
        trainer = FlowSDEPPOTrainer(adapter, value_head, config=config)
        base_policy_artifacts = _policy_artifact_hashes(pretrained_dir)
        lineage_policy_checkpoint = str(pretrained_dir)
        lineage_policy_artifacts = dict(base_policy_artifacts)
        resume_provenance: dict[str, Any] | None = None
        if args.resume:
            resume_path = _checkpoint_file(training_state)
            restored_step = trainer.load_checkpoint(
                resume_path,
                strict_config=True,
                load_optimizers=True,
            )
            resume_provenance = {
                "format": RESUME_PROVENANCE_FORMAT,
                "mode": "legacy_output",
                "checkpoint_path": str(resume_path),
                "checkpoint_sha256": _file_sha256(resume_path),
                "source_update_step": restored_step,
                "source_manifest_path": "",
                "source_manifest_sha256": "",
                "lineage_policy_checkpoint": lineage_policy_checkpoint,
                "lineage_policy_artifacts": dict(lineage_policy_artifacts),
                "matched_base_kind": "legacy_output",
                "source_actor_sha256": module_sha256(trainer._actor_module()),
                "source_critic_sha256": module_sha256(trainer.value_head),
                "source_frozen_policy_sha256": _frozen_policy_sha256(policy),
                "exact_actor_reload": True,
                "exact_critic_reload": True,
                "optimizer_state_continued": True,
            }
            progress.emit("trainer_resumed", **resume_provenance)
        elif args.resume_checkpoint is not None:
            resume_path, resume_provenance, expected_critic_sha = (
                _validate_explicit_resume_source(
                    args.resume_checkpoint,
                    pretrained_dir=pretrained_dir,
                    base_policy_artifacts=base_policy_artifacts,
                    policy=policy,
                    task_instruction=args.task_instruction,
                    robot_type=args.robot_type,
                    config=config,
                )
            )
            restored_step = trainer.load_checkpoint(
                resume_path,
                strict_config=True,
                load_optimizers=True,
            )
            if restored_step != resume_provenance["source_update_step"]:
                raise RuntimeError("Resume checkpoint update step disagrees with its manifest")
            exact_actor_reload = (
                module_sha256(trainer._actor_module())
                == resume_provenance["source_actor_sha256"]
            )
            exact_critic_reload = module_sha256(trainer.value_head) == expected_critic_sha
            if not exact_actor_reload or not exact_critic_reload:
                raise RuntimeError("Resume checkpoint did not reload actor and critic exactly")
            lineage_policy_checkpoint = resume_provenance["lineage_policy_checkpoint"]
            lineage_policy_artifacts = dict(
                resume_provenance["lineage_policy_artifacts"]
            )
            resume_provenance.update(
                {
                    "exact_actor_reload": True,
                    "exact_critic_reload": True,
                    "optimizer_state_continued": True,
                }
            )
            progress.emit("trainer_resumed", **resume_provenance)
        elif args.value_warmup_bundle is not None:
            value_initialization = load_value_warmup_bundle(
                args.value_warmup_bundle,
                base_checkpoint=pretrained_dir,
                policy=policy,
                value_head=value_head,
                value_optimizer=trainer.value_optimizer,
                conditioning_dim=adapter.conditioning_dim,
                task_instruction=args.task_instruction,
            )
            trainer.record_value_initialization_provenance(value_initialization)
            progress.emit("critic_initialized", **value_initialization)

        source_lineage = {
            "resume": resume_provenance,
            "value_initialization": trainer.value_initialization_provenance,
        }
        source_policy_contract = {
            "format": SOURCE_POLICY_FORMAT,
            "checkpoint_path": str(pretrained_dir),
            "artifacts": base_policy_artifacts,
            "frozen_policy_sha256": _frozen_policy_sha256(policy),
            "policy_contract": contract,
            "critic_contract": {
                "type": "multi_task_dit_value_head",
                "conditioning_dim": adapter.conditioning_dim,
            },
            "task_instruction": args.task_instruction,
            "robot_type": args.robot_type,
        }

        startup_manifest = {
            "format": "cyclo.flow_sde_ppo.online_startup.v1",
            "status": "ready",
            "operation": args.operation,
            "job_id": args.job_id,
            "base_checkpoint": str(pretrained_dir),
            "base_policy_artifacts": base_policy_artifacts,
            "lineage_policy_checkpoint": lineage_policy_checkpoint,
            "lineage_policy_artifacts": lineage_policy_artifacts,
            "task_instruction": args.task_instruction,
            "robot_type": args.robot_type,
            "ppo_config": asdict(config),
            "resume_checkpoint": (
                str(resume_path) if resume_provenance is not None else ""
            ),
            "source_lineage": source_lineage,
            "value_initialization_provenance": trainer.value_initialization_provenance,
        }
        _atomic_json(output_dir / "startup_manifest.json", startup_manifest)

        completed = 0
        if args.operation == "update":
            sealed = load_rollout_bundle(args.rollout_bundle)
            rollout_bundle = sealed.path
            rollout_bundles.append(rollout_bundle)
            progress.emit(
                "ready",
                stage="update",
                checkpoint=str(pretrained_dir),
                rollout_bundle=str(rollout_bundle),
                contract=contract,
                ppo_config=asdict(config),
            )
            metrics = update_rollout_bundle(
                trainer,
                sealed,
                expected_source_policy=source_policy_contract,
            )
            trainer.save_checkpoint(training_state)
            # Exercise the actual restore contract after every update. This is
            # intentionally the same trainer so validation does not double GPU
            # memory for the 225M-parameter policy.
            restored_step = trainer.load_checkpoint(training_state)
            if restored_step != metrics.update_step:
                raise RuntimeError("Trainer checkpoint reload step mismatch")
            mark_rollout_bundle_consumed(
                sealed,
                result_policy_identity=trainer.rollout_policy_identity(),
                metrics=metrics.as_dict(),
                trainer_checkpoint=training_state,
            )
            completed = len(sealed.episodes)
            progress.emit(
                "episode_updated",
                episode=completed,
                episodes=completed,
                episode_return=sum(item.episode_return for item in sealed.episodes),
                checkpoint=str(training_state),
                rollout_bundle=str(rollout_bundle),
                metrics=metrics.as_dict(),
            )
        else:
            observations = CycloLeRobotObservationSource(
                policy=policy,
                preprocessor=preprocessor,
                robot_type=args.robot_type,
                task_instruction=args.task_instruction,
                sensor_timeout=args.sensor_timeout,
            )
            transport = ZenohAtomicActionStepTransport(ack_timeout=args.ack_timeout)
            source = CycloFlowSDEEpisodeSource(
                observations=observations,
                actions=transport,
                outcomes=AtomicOutcomeFile(control_file, job_id=args.job_id),
                postprocessor=postprocessor,
                max_chunk_decisions=args.max_chunk_decisions,
                sensor_timeout=args.sensor_timeout,
            )
            progress.emit(
                "ready",
                stage="rollout",
                checkpoint=str(pretrained_dir),
                contract=contract,
                ppo_config=asdict(config),
                source_lineage=source_lineage,
                value_initialization_provenance=trainer.value_initialization_provenance,
            )

            attempted = 0
            while completed < args.episodes:
                attempted += 1
                progress.emit(
                    "episode_started",
                    episode=completed + 1,
                    episodes=args.episodes,
                    attempt=attempted,
                )
                try:
                    episode = collect_one_episode(trainer, source)
                except FlowSDECollectionCancelled as exc:
                    # Cancel discards the partial rollout and never reaches a
                    # bundle or optimizer boundary.
                    progress.emit(
                        "episode_cancelled",
                        episode=completed + 1,
                        episodes=args.episodes,
                        attempt=attempted,
                        completed_episodes=completed,
                        message=str(exc),
                    )
                    continue

                policy_identity = trainer.rollout_policy_identity()
                rollout_bundle = save_rollout_bundle(
                    output_dir
                    / "rollouts"
                    / (
                        f"update_{policy_identity['source_update_step']:06d}_"
                        f"{uuid.uuid4().hex}"
                    ),
                    [episode],
                    policy_identity=policy_identity,
                    source_policy=source_policy_contract,
                    source_training_state=trainer.training_state_dict(),
                    metadata={
                        "job_id": args.job_id,
                        "episode": completed + 1,
                        "attempt": attempted,
                        "task_instruction": args.task_instruction,
                        "robot_type": args.robot_type,
                        "episode_return": episode.episode_return,
                        "terminated": episode.transitions[-1].terminated,
                        "truncated": episode.transitions[-1].truncated,
                        **source.last_episode_diagnostics,
                    },
                )
                sealed = load_rollout_bundle(
                    rollout_bundle,
                    expected_policy_identity=policy_identity,
                )
                rollout_bundles.append(rollout_bundle)
                completed += 1

                if args.operation == "collect":
                    progress.emit(
                        "episode_collected",
                        episode=completed,
                        episodes=args.episodes,
                        episode_return=episode.episode_return,
                        rollout_bundle=str(rollout_bundle),
                        **source.last_episode_diagnostics,
                    )
                    continue

                metrics = update_rollout_bundle(
                    trainer,
                    sealed,
                    expected_source_policy=source_policy_contract,
                )
                trainer.save_checkpoint(training_state)
                restored_step = trainer.load_checkpoint(training_state)
                if restored_step != metrics.update_step:
                    raise RuntimeError("Trainer checkpoint reload step mismatch")
                mark_rollout_bundle_consumed(
                    sealed,
                    result_policy_identity=trainer.rollout_policy_identity(),
                    metrics=metrics.as_dict(),
                    trainer_checkpoint=training_state,
                )
                progress.emit(
                    "episode_updated",
                    episode=completed,
                    episodes=args.episodes,
                    episode_return=episode.episode_return,
                    terminated=episode.transitions[-1].terminated,
                    truncated=episode.transitions[-1].truncated,
                    checkpoint=str(training_state),
                    rollout_bundle=str(rollout_bundle),
                    metrics=metrics.as_dict(),
                    **source.last_episode_diagnostics,
                )

            if args.operation == "collect":
                summary = {
                    "status": "completed",
                    "operation": "collect",
                    "job_id": args.job_id,
                    "episodes": completed,
                    "base_checkpoint": str(pretrained_dir),
                    "task_instruction": args.task_instruction,
                    "robot_type": args.robot_type,
                    "ppo_config": asdict(config),
                    "rollout_bundles": [str(path) for path in rollout_bundles],
                }
                _atomic_json(output_dir / "summary.json", summary)
                progress.emit("rollout_completed", **summary)
                return 0

        exported = trainer.export_pretrained_policy(
            pretrained_output,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        updates = trainer.update_step
        value_initialization_provenance = trainer.value_initialization_provenance
        actor_sha256 = module_sha256(trainer._actor_module())
        critic_sha256 = module_sha256(trainer.value_head)
        frozen_policy_sha256 = _frozen_policy_sha256(trainer.policy)
        exported_policy_artifacts = _policy_artifact_hashes(exported)
        trainer_checkpoint_sha256 = _file_sha256(training_state)
        # The live RobotClient and original 225M-parameter policy are no longer
        # needed. Release them before the strict export reload to avoid doubling
        # GPU memory on a single 5090.
        if source is not None:
            source.close()
            source = None
        del trainer, value_head, adapter, policy, preprocessor, postprocessor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        export_contract = _reload_export(exported, device)
        summary = {
            "status": "completed",
            "operation": args.operation,
            "job_id": args.job_id,
            "episodes": completed,
            "updates": updates,
            "trainer_checkpoint": str(training_state),
            "pretrained_model": str(exported),
            "base_checkpoint": str(pretrained_dir),
            "base_policy_artifacts": base_policy_artifacts,
            "lineage_policy_checkpoint": lineage_policy_checkpoint,
            "lineage_policy_artifacts": lineage_policy_artifacts,
            "task_instruction": args.task_instruction,
            "robot_type": args.robot_type,
            "ppo_config": asdict(config),
            "resume_checkpoint": (
                resume_provenance["checkpoint_path"]
                if resume_provenance is not None
                else ""
            ),
            "export_contract": export_contract,
            "source_lineage": source_lineage,
            "value_initialization_provenance": value_initialization_provenance,
            "actor_sha256": actor_sha256,
            "critic_sha256": critic_sha256,
            "frozen_policy_sha256": frozen_policy_sha256,
            "rollout_bundles": [str(path) for path in rollout_bundles],
            "run_manifest": str(output_dir / "run_manifest.json"),
        }
        _atomic_json(output_dir / "summary.json", summary)
        run_manifest = {
            "format": ONLINE_BUNDLE_FORMAT,
            "status": "complete",
            "operation": args.operation,
            "job_id": args.job_id,
            "base_checkpoint": str(pretrained_dir),
            "base_policy_artifacts": base_policy_artifacts,
            "lineage_policy_checkpoint": lineage_policy_checkpoint,
            "lineage_policy_artifacts": lineage_policy_artifacts,
            "task_instruction": args.task_instruction,
            "robot_type": args.robot_type,
            "ppo_config": asdict(config),
            "resume_checkpoint": (
                resume_provenance["checkpoint_path"]
                if resume_provenance is not None
                else ""
            ),
            "result": {
                "episodes": completed,
                "update_step": updates,
                "actor_sha256": actor_sha256,
                "critic_sha256": critic_sha256,
                "frozen_policy_sha256": frozen_policy_sha256,
            },
            "source_lineage": source_lineage,
            "artifacts": {
                "pretrained_model": {
                    "path": "pretrained_model",
                    "files": exported_policy_artifacts,
                },
                "trainer_checkpoint": {
                    "path": "training_state/trainer_state.pt",
                    "sha256": trainer_checkpoint_sha256,
                },
                "rollout_bundles": [
                    {
                        "path": (
                            str(path.relative_to(output_dir))
                            if path.is_relative_to(output_dir)
                            else str(path)
                        ),
                        "manifest_sha256": _file_sha256(path / "manifest.json"),
                        "consumption_sha256": _file_sha256(path / "consumption.json"),
                    }
                    for path in rollout_bundles
                ],
                "startup_manifest_path": "startup_manifest.json",
                "progress_path": "progress.jsonl",
                "summary_path": "summary.json",
            },
        }
        run_manifest_path = output_dir / "run_manifest.json"
        _atomic_json(run_manifest_path, run_manifest)
        summary["run_manifest_sha256"] = _file_sha256(run_manifest_path)
        _atomic_json(output_dir / "summary.json", summary)
        progress.emit("completed", **summary)
        return 0
    except FlowSDECollectionCancelled as exc:
        # Defensive boundary for cancellation outside the episode loop (for
        # example during future source setup extensions).
        summary = {
            "status": "cancelled",
            "job_id": args.job_id,
            "message": str(exc),
            "trainer_checkpoint": str(training_state) if training_state.is_file() else "",
            "rollout_bundles": [str(path) for path in rollout_bundles],
        }
        _atomic_json(output_dir / "summary.json", summary)
        progress.emit("cancelled", message=str(exc))
        return 0
    except KeyboardInterrupt:
        summary = {
            "status": "stopped",
            "job_id": args.job_id,
            "message": "Flow-SDE PPO stopped by SIGINT/KeyboardInterrupt",
            "trainer_checkpoint": str(training_state) if training_state.is_file() else "",
            "rollout_bundles": [str(path) for path in rollout_bundles],
        }
        _atomic_json(output_dir / "summary.json", summary)
        progress.emit("stopped", message=summary["message"])
        return 0
    except Exception as exc:
        summary = {
            "status": "failed",
            "job_id": args.job_id,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "rollout_bundles": [str(path) for path in rollout_bundles],
        }
        _atomic_json(output_dir / "summary.json", summary)
        progress.emit("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if source is not None:
            source.close()


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(_parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
