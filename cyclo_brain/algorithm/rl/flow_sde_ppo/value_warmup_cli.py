"""Train a frozen-policy MultiTaskDiT value head from labelled LeRobot v3 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from cyclo_brain.model.multi_task_dit.checkpoint_validation import (
    assert_deployment_artifacts,
    validate_policy_contract,
)
from cyclo_brain.model.multi_task_dit.flow_sde_adapter import (
    CYCLO_SG2_CAMERA_KEYS,
    MultiTaskDiTFlowAdapter,
)
from cyclo_brain.model.multi_task_dit.success_dataset import discover_episode_outcomes
from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead

from .value_warmup import (
    EpisodeBalancedChunkBoundaryDataset,
    MultiTaskDiTValueWarmupRunner,
    ValueWarmupConfig,
    ValueWarmupProgress,
    module_sha256,
)


BUNDLE_FORMAT = "cyclo.flow_sde_ppo.value_warmup.bundle.v1"
DEFAULT_TASK = "pick up the jelly bag"
REQUIRED_POLICY_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        required=True,
        action="append",
        type=Path,
        help="Repeat for each immutable LeRobot v3 Data Epoch.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--value-lr", type=float, default=1.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--task-instruction", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "torchcodec", "video_reader"),
        default="pyav",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class JsonlProgress:
    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"event": event, "timestamp": time.time(), **payload}
        line = _canonical_json(record).decode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)
        return record


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _resolve_dataset_root(value: Path) -> Path:
    root = value.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot info.json is missing: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"LeRobot info.json is invalid: {info_path}") from error
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"value warm-up requires LeRobot v3.0, got {info.get('codebase_version')!r}")
    features = info.get("features")
    if not isinstance(features, Mapping) or features.get("episode_success", {}).get("dtype") != "bool":
        raise ValueError("LeRobot v3 dataset must declare boolean episode_success")
    return root


def _dataset_identity(root: Path, outcomes: Any) -> dict[str, Any]:
    metadata_files = sorted(
        path for path in (root / "meta").rglob("*") if path.is_file() and not path.is_symlink()
    )
    if not metadata_files:
        raise FileNotFoundError(f"LeRobot metadata is empty: {root / 'meta'}")
    metadata_digest = hashlib.sha256()
    for path in metadata_files:
        metadata_digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        metadata_digest.update(b"\0")
        metadata_digest.update(_sha256_file(path).encode("ascii"))
        metadata_digest.update(b"\0")
    # Bind the run to every selected local file without spending minutes
    # re-hashing large camera videos. Metadata itself is content-hashed; data
    # and videos contribute canonical relative path, byte count, and mtime.
    all_files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    file_manifest: list[tuple[str, int, int]] = []
    total_bytes = 0
    for path in all_files:
        stat = path.stat()
        total_bytes += stat.st_size
        file_manifest.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    identity = hashlib.sha256(
        _canonical_json(
            {
                "root": str(root),
                "metadata_sha256": metadata_digest.hexdigest(),
                "files": file_manifest,
            }
        )
    ).hexdigest()
    return {
        "path": str(root),
        "identity_sha256": f"sha256:{identity}",
        "metadata_sha256": f"sha256:{metadata_digest.hexdigest()}",
        "file_count": len(all_files),
        "total_bytes": total_bytes,
        "success_episodes": outcomes.success_episode_count,
        "failure_episodes": outcomes.failure_episode_count,
        "success_frames": outcomes.success_frames,
        "failure_frames": outcomes.failure_frames,
    }


def _base_identity(pretrained_dir: Path, policy: torch.nn.Module) -> dict[str, Any]:
    return {
        "path": str(pretrained_dir),
        "policy_sha256": module_sha256(policy),
        "artifacts": {
            name: _sha256_file(pretrained_dir / name) for name in REQUIRED_POLICY_ARTIFACTS
        },
    }


def _canonical_observation_batch(
    observations: Mapping[str, torch.Tensor],
    *,
    task_instruction: str,
    n_obs_steps: int,
) -> dict[str, Any]:
    state = observations.get("observation.state")
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError("value warm-up state must have shape (B, S)")
    if n_obs_steps != 1:
        raise ValueError("offline value warm-up currently requires n_obs_steps=1")
    batch_size = state.shape[0]
    result: dict[str, Any] = {"observation.state": state.unsqueeze(1)}
    for key in CYCLO_SG2_CAMERA_KEYS:
        image = observations.get(key)
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != batch_size:
            raise ValueError(f"value warm-up camera {key!r} must have shape (B, C, H, W)")
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        elif not image.is_floating_point():
            raise TypeError(f"value warm-up camera {key!r} must be uint8 or floating point")
        result[key] = image.unsqueeze(1)
    result["task"] = [task_instruction] * batch_size
    return result


def _copy_policy_unchanged(source: Path, destination: Path, identity: Mapping[str, Any]) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    expected = identity["artifacts"]
    for name in REQUIRED_POLICY_ARTIFACTS:
        if _sha256_file(destination / name) != expected[name]:
            raise RuntimeError(f"copied policy artifact changed: {name}")


def _validate_args(args: argparse.Namespace) -> ValueWarmupConfig:
    return ValueWarmupConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        value_lr=args.value_lr,
        gamma=args.gamma,
        task_instruction=args.task_instruction,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
    )


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from torchvision.transforms.v2 import Resize

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = _validate_args(args)
    device = _resolve_device(args.device)
    pretrained_dir = assert_deployment_artifacts(args.base_checkpoint)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite value warm-up bundle: {output_dir}")
    roots = tuple(_resolve_dataset_root(value) for value in args.dataset_root)
    if len(set(roots)) != len(roots):
        raise ValueError("dataset-root values must be unique")
    for input_root in (pretrained_dir, *roots):
        if input_root == output_dir or input_root in output_dir.parents:
            raise ValueError("output-dir must be outside the base checkpoint and dataset roots")
    outcomes = tuple(discover_episode_outcomes(root) for root in roots)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.incomplete"
    staging.mkdir()
    progress = JsonlProgress(staging / "progress.jsonl")
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    previous_handlers = {
        number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        progress.emit("starting", phase="load_inputs", steps=config.steps)
        dataset_identities = tuple(
            _dataset_identity(root, split) for root, split in zip(roots, outcomes, strict=True)
        )

        config_payload = json.loads((pretrained_dir / "config.json").read_text(encoding="utf-8"))
        policy_type = config_payload.get("type")
        if policy_type != "multi_task_dit":
            raise ValueError(f"value warm-up requires type='multi_task_dit', got {policy_type!r}")
        policy = get_policy_class(policy_type).from_pretrained(str(pretrained_dir)).to(device).eval()
        contract = validate_policy_contract(policy)
        preprocessor, _postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained_dir),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        base_identity = _base_identity(pretrained_dir, policy)

        datasets = tuple(
            LeRobotDataset(
                repo_id=f"local/value-warmup-{index}-{root.name}",
                root=root,
                image_transforms=Resize((256, 256), antialias=True),
                delta_timestamps=None,
                force_cache_sync=False,
                download_videos=False,
                video_backend=args.video_backend,
                return_uint8=True,
            )
            for index, root in enumerate(roots)
        )
        observation_keys = ("observation.state", *CYCLO_SG2_CAMERA_KEYS)
        dataset = EpisodeBalancedChunkBoundaryDataset(
            datasets,
            observation_keys=observation_keys,
            n_action_steps=int(policy.config.n_action_steps),
            gamma=config.gamma,
            dataset_names=tuple(str(root) for root in roots),
        )
        adapter = MultiTaskDiTFlowAdapter(policy, freeze_observation_encoder=True)
        value_head = MultiTaskDiTValueHead(adapter.conditioning_dim).to(device)

        def encode(observations: Mapping[str, torch.Tensor], task: str) -> torch.Tensor:
            raw = _canonical_observation_batch(
                observations, task_instruction=task, n_obs_steps=policy.config.n_obs_steps
            )
            processed = preprocessor(raw)
            return adapter.encode_conditioning(processed)

        checkpoint = staging / "training_state" / "value_warmup.pt"
        runner = MultiTaskDiTValueWarmupRunner(
            policy,
            value_head,
            dataset,
            encode,
            config=config,
            checkpoint_path=checkpoint,
            base_identity=base_identity,
            dataset_identities=dataset_identities,
        )

        def emit_progress(snapshot: ValueWarmupProgress) -> None:
            progress.emit("progress", **asdict(snapshot))

        result = runner.run(progress=emit_progress, should_stop=stop_requested.is_set)
        _copy_policy_unchanged(pretrained_dir, staging / "pretrained_model", base_identity)
        manifest = {
            "format": BUNDLE_FORMAT,
            "status": result.status,
            "created_at": datetime.now(UTC).isoformat(),
            "base": base_identity,
            "datasets": list(dataset_identities),
            "config": asdict(config),
            "dataset_contract": dataset.contract(),
            "result": {
                "completed_steps": result.completed_steps,
                "final_value_loss": result.final_value_loss,
                "mean_value_loss": result.mean_value_loss,
                "elapsed_seconds": result.elapsed_seconds,
                "policy_sha256_before": result.policy_sha256_before,
                "policy_sha256_after": result.policy_sha256_after,
            },
            "artifacts": {
                "model_path": "pretrained_model",
                "checkpoint_path": "training_state/value_warmup.pt",
                "progress_path": "progress.jsonl",
            },
            "policy_contract": contract,
        }
        _atomic_json(staging / "run_manifest.json", manifest)
        terminal = {
            "status": result.status,
            "model_path": str(output_dir / "pretrained_model"),
            "checkpoint_path": str(output_dir / "training_state" / "value_warmup.pt"),
            "bundle_path": str(output_dir),
        }
        progress.emit("result", **terminal)
        os.replace(staging, output_dir)
        return terminal
    except Exception as error:
        progress.emit(
            "failed",
            phase="error",
            error_type=type(error).__name__,
            message=str(error),
            incomplete_bundle_path=str(staging),
        )
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main(argv: Sequence[str] | None = None) -> None:
    run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
