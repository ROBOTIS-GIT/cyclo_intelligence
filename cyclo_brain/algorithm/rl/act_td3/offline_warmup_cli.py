"""Standalone local-data CLI for the actor-frozen ACT-TD3 critic warm-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from cyclo_brain.model.act import (
    ACTTwinChunkCritic,
    load_act_physical_action_domain,
    load_act_policy_assets,
)

from .config import ACTTD3Config
from .learner import ACTTD3Learner
from .lerobot_offline import (
    ACTTD3LeRobotCollator,
    FixedHorizonLeRobotACTTD3Dataset,
    VirtualCumulativeLeRobotACTTD3Dataset,
)
from .offline_warmup import (
    ACTTD3CriticWarmupProgress,
    ACTTD3CriticWarmupRunner,
    _atomic_torch_save,
)
from .training_identity import build_act_td3_multi_root_training_data_identity


_MAX_SEED = 2**63 - 2
_VIDEO_BACKENDS = ("pyav", "torchcodec", "video_reader")


def _integer(value: str, *, name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
    if not minimum <= result <= maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return result


def _seed(value: str) -> int:
    return _integer(value, name="seed", minimum=0, maximum=_MAX_SEED)


def _positive(value: str) -> int:
    return _integer(value, name="value", minimum=1, maximum=2**31 - 1)


def _critic_updates(value: str) -> int:
    return _integer(
        value,
        name="critic_updates",
        minimum=1,
        maximum=1_000_000,
    )


def _warmup_boundary(value: str) -> int:
    return _integer(
        value,
        name="max_critic_updates",
        minimum=1,
        maximum=1_000_000,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up ACT-TD3 critics from a finalized local LeRobot v3 dataset. "
            "The official ACT actor remains bitwise unchanged."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        action="append",
        help="Immutable LeRobot v3 root; repeat in replay order.",
    )
    parser.add_argument("--act-checkpoint", required=True, type=Path)
    parser.add_argument("--robot-config", required=True, type=Path)
    parser.add_argument("--robot-type", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", required=True, type=_seed)
    parser.add_argument(
        "--sampling-seed",
        type=_seed,
        help="Replay sampler seed; defaults to seed + 2.",
    )
    parser.add_argument("--batch-size", required=True, type=_positive)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--critic-updates",
        type=_critic_updates,
        default=ACTTD3Config().critic_warmup_updates,
        help="Total critic optimizer updates for the actor-frozen warm-up.",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help=(
            "Optional exact <ACT checkpoint>/critic directory. A completed "
            "critic-only latest.pt and manifest.json are published atomically; "
            "running, failed, and stopped jobs never replace them."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-critic-updates",
        type=_warmup_boundary,
        help="Absolute update boundary, not an additional update count.",
    )
    parser.add_argument("--checkpoint-interval", type=_positive, default=500)
    parser.add_argument("--progress-interval", type=_positive, default=10)
    parser.add_argument(
        "--video-backend",
        choices=_VIDEO_BACKENDS,
        default="pyav",
    )
    return parser


def _input_directory(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {resolved}")
    return resolved


def _input_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def _output_checkpoint(
    path: Path,
    *,
    resume: bool,
    inputs: Sequence[Path],
    allowed_output_root: Path | None = None,
) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("ACT-TD3 checkpoint path must not be a symbolic link")
    resolved = expanded.resolve(strict=False)
    if resume and not resolved.is_file():
        raise FileNotFoundError(f"ACT-TD3 resume checkpoint does not exist: {resolved}")
    if not resume and resolved.exists():
        raise FileExistsError(f"ACT-TD3 checkpoint already exists: {resolved}")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"ACT-TD3 checkpoint must be a regular file: {resolved}")
    raw_allowed_root = (
        allowed_output_root.expanduser()
        if allowed_output_root is not None
        else None
    )
    if raw_allowed_root is not None and (
        raw_allowed_root.is_symlink() or raw_allowed_root.parent.is_symlink()
    ):
        raise ValueError("ACT-TD3 checkpoint output root must not be a symbolic link")
    allowed_root = (
        raw_allowed_root.resolve(strict=False)
        if raw_allowed_root is not None
        else None
    )
    if allowed_root is not None:
        if resolved.parent != allowed_root or resolved.suffix != ".pt":
            raise ValueError(
                "ACT-TD3 policy-local runner checkpoint must be one .pt file "
                f"directly under {allowed_root}"
            )
    for input_path in inputs:
        input_root = input_path if input_path.is_dir() else input_path.parent
        if resolved == input_root or input_root in resolved.parents:
            if allowed_root is not None and resolved.parent == allowed_root:
                continue
            raise ValueError(
                "ACT-TD3 checkpoint must be outside dataset, ACT checkpoint, "
                "and robot-config inputs"
            )
    return resolved


def _dataset_root_arguments(value: Any) -> tuple[Path, ...]:
    """Normalize argparse and legacy programmatic one-root namespaces."""

    if isinstance(value, Path):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        roots = tuple(value)
        if roots and all(isinstance(root, Path) for root in roots):
            return roots
    raise TypeError("dataset_root must contain one or more paths")


def _publish_directory(path: Path, *, act_checkpoint: Path) -> Path:
    """Resolve only the dedicated policy-local critic directory."""

    expanded = path.expanduser()
    expected = act_checkpoint / "critic"
    if expanded.is_symlink():
        raise ValueError("ACT-TD3 critic publish directory must not be a symbolic link")
    resolved = expanded.resolve(strict=False)
    if resolved != expected:
        raise ValueError(
            "ACT-TD3 critic publish directory must be exactly "
            f"{expected}"
        )
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    for name in ("latest.pt", "manifest.json"):
        target = resolved / name
        if target.is_symlink():
            raise ValueError(f"ACT-TD3 critic artifact must not be a symbolic link: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"ACT-TD3 critic artifact must be a regular file: {target}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            dict(value),
            stream,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_completed_critic(
    *,
    runner: ACTTD3CriticWarmupRunner,
    publish_dir: Path,
    act_checkpoint: Path,
    dataset_roots: Sequence[Path],
    identity: Any,
) -> tuple[Path, Path]:
    """Transactionally publish a verified critic artifact and commit manifest."""

    artifact = runner.critic_artifact_state()
    publish_dir.mkdir(mode=0o775, parents=False, exist_ok=True)
    if publish_dir.is_symlink() or not publish_dir.is_dir():
        raise ValueError("ACT-TD3 critic publish directory became unsafe")

    token = uuid.uuid4().hex
    prepared_checkpoint = publish_dir / f".latest.pt.{token}.prepared"
    prepared_manifest = publish_dir / f".manifest.json.{token}.prepared"
    latest = publish_dir / "latest.pt"
    manifest_path = publish_dir / "manifest.json"
    latest_backup = publish_dir / f".latest.pt.{token}.backup"
    manifest_backup = publish_dir / f".manifest.json.{token}.backup"
    latest_committed = False
    manifest_committed = False
    try:
        _atomic_torch_save(prepared_checkpoint, artifact)
        checked = torch.load(
            prepared_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        expected_artifact_keys = {
            "format",
            "status",
            "contract",
            "actor_sha256",
            "actor_target_sha256",
            "critic",
            "critic_target",
            "critic_optimizer",
            "completed_critic_updates",
            "completed_actor_updates",
        }
        if (
            not isinstance(checked, Mapping)
            or set(checked) != expected_artifact_keys
            or checked.get("format") != runner.CRITIC_ARTIFACT_FORMAT
            or checked.get("status") != "complete"
            or checked.get("actor_sha256") != artifact["actor_sha256"]
            or checked.get("completed_critic_updates") != runner.total_critic_updates
            or checked.get("completed_actor_updates") != 0
        ):
            raise RuntimeError("published ACT-TD3 critic artifact verification failed")

        checkpoint_sha256 = _sha256_file(prepared_checkpoint)
        checkpoint_bytes = prepared_checkpoint.stat().st_size
        manifest = {
            "format": "cyclo_brain.act_td3_critic_manifest/v1",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_policy": {
                "path": str(act_checkpoint),
                "actor_sha256": artifact["actor_sha256"],
            },
            "artifact": {
                "format": runner.CRITIC_ARTIFACT_FORMAT,
                "checkpoint_path": "latest.pt",
                "sha256": checkpoint_sha256,
                "byte_count": checkpoint_bytes,
            },
            "training_data": {
                "identity": identity.identity,
                "dataset_roots": [str(root) for root in dataset_roots],
                "file_count": identity.file_count,
                "byte_count": identity.byte_count,
                "component_sha256": identity.component_sha256,
                "virtual_contract": identity.virtual_contract,
            },
            "dataset": artifact["contract"]["dataset"],
            "learner": artifact["contract"]["learner"],
            "completed_critic_updates": artifact["completed_critic_updates"],
            "completed_actor_updates": artifact["completed_actor_updates"],
            "actor_exactly_unchanged": True,
        }
        # Normalize tuples and other JSON sequence values before both writing
        # and comparing.  The real learner contract contains tuples (for
        # example observation_keys and actor_trainable_groups), while a JSON
        # round trip necessarily represents them as lists.
        canonical_manifest = json.loads(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        _write_json_durable(prepared_manifest, canonical_manifest)
        parsed_manifest = json.loads(prepared_manifest.read_text(encoding="utf-8"))
        if parsed_manifest != canonical_manifest:
            raise RuntimeError("published ACT-TD3 critic manifest verification failed")

        for target, backup in (
            (latest, latest_backup),
            (manifest_path, manifest_backup),
        ):
            if target.is_symlink():
                raise ValueError(f"ACT-TD3 critic artifact became a symbolic link: {target}")
            if target.exists():
                if not target.is_file():
                    raise ValueError(
                        f"ACT-TD3 critic artifact is not a regular file: {target}"
                    )
                os.link(target, backup)

        os.replace(prepared_checkpoint, latest)
        latest_committed = True
        os.replace(prepared_manifest, manifest_path)
        manifest_committed = True
        _fsync_directory(publish_dir)
    except Exception:
        if manifest_committed:
            if manifest_backup.exists():
                os.replace(manifest_backup, manifest_path)
            else:
                manifest_path.unlink(missing_ok=True)
        if latest_committed:
            if latest_backup.exists():
                os.replace(latest_backup, latest)
            else:
                latest.unlink(missing_ok=True)
        _fsync_directory(publish_dir)
        raise
    finally:
        for path in (
            prepared_checkpoint,
            prepared_manifest,
            latest_backup,
            manifest_backup,
        ):
            path.unlink(missing_ok=True)

    return latest, manifest_path


def _require_unchanged_training_data_identity(
    *,
    expected: Any,
    datasets: Sequence[Any],
    dataset_roots: Sequence[Path],
    act_checkpoint: Path,
    action_domains: Sequence[Any],
    robot_type: str,
    video_backend: str,
) -> Any:
    """Re-hash every identity-bound input immediately before publication."""

    observed = build_act_td3_multi_root_training_data_identity(
        datasets,
        dataset_roots,
        act_checkpoint_root=act_checkpoint,
        action_domains=action_domains,
        robot_type=robot_type,
        video_backend=video_backend,
    )
    if observed != expected:
        raise RuntimeError(
            "ACT-TD3 training data or base policy changed during critic warm-up"
        )
    return observed


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid ACT-TD3 device: {value!r}") from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("ACT-TD3 CPU device cannot have an index")
        return device
    if device.type != "cuda" or device.index is None:
        raise ValueError("ACT-TD3 device must be 'cpu' or an explicit CUDA index")
    if not torch.cuda.is_available():
        raise RuntimeError("ACT-TD3 CUDA was requested but is unavailable")
    if not 0 <= device.index < torch.cuda.device_count():
        raise ValueError(f"ACT-TD3 CUDA device index is unavailable: {device.index}")
    torch.cuda.set_device(device)
    return device


def _require_local_dataset_layout(dataset_root: Path) -> None:
    required_files = (
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "tasks.parquet",
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot dataset file is missing: {path}")
    for relative in ("meta/episodes", "data"):
        directory = dataset_root / relative
        if not directory.is_dir() or not any(directory.rglob("*.parquet")):
            raise FileNotFoundError(
                f"LeRobot dataset has no parquet files under: {directory}"
            )


def _require_referenced_dataset_files(dataset_root: Path, metadata: Any) -> None:
    episode_indices = tuple(range(int(metadata.total_episodes)))
    referenced = {
        Path(metadata.get_data_file_path(index)) for index in episode_indices
    }
    referenced.update(
        Path(metadata.get_video_file_path(index, video_key))
        for video_key in metadata.video_keys
        for index in episode_indices
    )
    for relative in referenced:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"LeRobot metadata path escapes dataset root: {relative}")
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot referenced file is missing: {path}")


def _json_line(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True),
        file=stream,
        flush=True,
    )


def _progress_line(progress: ACTTD3CriticWarmupProgress) -> None:
    _json_line({"event": "progress", **asdict(progress)})


def run_from_args(
    args: argparse.Namespace,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> ACTTD3CriticWarmupProgress:
    if (
        args.max_critic_updates is not None
        and args.max_critic_updates > args.critic_updates
    ):
        raise ValueError("max_critic_updates cannot exceed critic_updates")
    dataset_roots = tuple(
        _input_directory(value, f"dataset root {index}")
        for index, value in enumerate(_dataset_root_arguments(args.dataset_root))
    )
    if len(set(dataset_roots)) != len(dataset_roots):
        raise ValueError("dataset roots must be unique and ordered")
    act_checkpoint = _input_directory(args.act_checkpoint, "ACT checkpoint")
    robot_config = _input_file(args.robot_config, "robot config")
    if not isinstance(args.robot_type, str) or not args.robot_type.strip():
        raise ValueError("robot_type must be a non-empty string")
    device = _device(args.device)
    sampling_seed = args.sampling_seed
    if sampling_seed is None:
        sampling_seed = args.seed + 2
        if sampling_seed > _MAX_SEED:
            raise ValueError("default sampling seed exceeds the supported range")
    publish_dir = (
        _publish_directory(args.publish_dir, act_checkpoint=act_checkpoint)
        if args.publish_dir is not None
        else None
    )
    checkpoint = _output_checkpoint(
        args.checkpoint,
        resume=args.resume,
        inputs=(*dataset_roots, act_checkpoint, robot_config),
        allowed_output_root=(publish_dir / "runs" if publish_dir is not None else None),
    )

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets: list[Any] = []
    action_domains: list[Any] = []
    for root_index, dataset_root in enumerate(dataset_roots):
        _require_local_dataset_layout(dataset_root)
        # Resolve every referenced local file before LeRobotDataset
        # construction. This prevents its Hub fallback from turning a malformed
        # local run into a network-dependent one.
        local_repo_id = f"local/warmup_{root_index:04d}_{dataset_root.name}"
        metadata = LeRobotDatasetMetadata(repo_id=local_repo_id, root=dataset_root)
        _require_referenced_dataset_files(dataset_root, metadata)
        datasets.append(
            LeRobotDataset(
                repo_id=local_repo_id,
                root=dataset_root,
                image_transforms=None,
                delta_timestamps=None,
                force_cache_sync=False,
                download_videos=False,
                video_backend=args.video_backend,
                return_uint8=False,
            )
        )
        action_domains.append(
            load_act_physical_action_domain(
                dataset_root / "meta" / "info.json",
                robot_config,
                robot_type=args.robot_type,
            )
        )
    identity = build_act_td3_multi_root_training_data_identity(
        datasets,
        dataset_roots,
        act_checkpoint_root=act_checkpoint,
        action_domains=action_domains,
        robot_type=args.robot_type,
        video_backend=args.video_backend,
    )

    assets = load_act_policy_assets(act_checkpoint, device=device)
    actor = assets.policy
    actor_parameter = next(actor.parameters())

    # Checkpoint loading and processor construction may consume global RNG.
    # Reset immediately before the only newly initialized model.
    torch.manual_seed(args.seed)
    critic = ACTTwinChunkCritic(actor.config)
    critic.initialize_visual_backbones_from_actor(actor)
    critic.to(device=actor_parameter.device, dtype=actor_parameter.dtype)

    root_replays = tuple(
        FixedHorizonLeRobotACTTD3Dataset(
            dataset,
            execution_horizon=int(actor.config.n_action_steps),
            observation_keys=tuple(actor.config.input_features or {}),
        )
        for dataset in datasets
    )
    replay = VirtualCumulativeLeRobotACTTD3Dataset(root_replays)
    if replay.num_successes < 1 or replay.num_failures < 1:
        raise ValueError(
            "ACT-TD3 critic warm-up requires at least one success and one failure"
        )
    collator = ACTTD3LeRobotCollator(assets.preprocessor)
    config = ACTTD3Config(
        discount_reference_hz=float(replay.fps),
        critic_warmup_updates=args.critic_updates,
    )
    learner = ACTTD3Learner(
        actor,
        critic,
        config,
        random_seed=args.seed,
    )
    runner = ACTTD3CriticWarmupRunner(
        learner,
        replay,
        collator,
        batch_size=args.batch_size,
        sampling_seed=sampling_seed,
        training_data_identity=identity.identity,
        checkpoint_path=checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        resume=args.resume,
    )

    _json_line(
        {
            "event": "manifest",
            "algorithm": "ACT-TD3 critic warm-up (actor frozen)",
            "device": str(device),
            "seed": args.seed,
            "sampling_seed": sampling_seed,
            "batch_size": args.batch_size,
            "checkpoint": str(checkpoint),
            "publish_dir": str(publish_dir) if publish_dir is not None else None,
            "resume": args.resume,
            "max_critic_updates": args.max_critic_updates,
            "total_critic_updates": config.critic_warmup_updates,
            "dataset": {
                "roots": len(dataset_roots),
                "root_paths": [str(root) for root in dataset_roots],
                "frames": sum(len(dataset) for dataset in datasets),
                "episodes": replay.num_episodes,
                "successes": replay.num_successes,
                "failures": replay.num_failures,
                "macro_transitions": len(replay),
                "fps": replay.fps,
            },
            "actor": {
                "action_dim": learner.action_dim,
                "prediction_horizon": learner.prediction_horizon,
                "execution_horizon": learner.execution_horizon,
                "action_domain": learner.ACTION_DOMAIN,
                "target_policy_smoothing": learner.TARGET_POLICY_SMOOTHING,
                "actor_q_gradient": learner.ACTOR_Q_GRADIENT,
                "action_clamp": False,
            },
            "training_data": {
                "identity": identity.identity,
                "file_count": identity.file_count,
                "byte_count": identity.byte_count,
                "component_sha256": identity.component_sha256,
                "video_backend": args.video_backend,
            },
        }
    )
    result = runner.run(
        max_critic_updates=args.max_critic_updates,
        progress_callback=_progress_line,
        should_stop=should_stop,
    )
    published_checkpoint: Path | None = None
    published_manifest: Path | None = None
    if result.status == "complete" and publish_dir is not None:
        identity = _require_unchanged_training_data_identity(
            expected=identity,
            datasets=datasets,
            dataset_roots=dataset_roots,
            act_checkpoint=act_checkpoint,
            action_domains=action_domains,
            robot_type=args.robot_type,
            video_backend=args.video_backend,
        )
        published_checkpoint, published_manifest = _publish_completed_critic(
            runner=runner,
            publish_dir=publish_dir,
            act_checkpoint=act_checkpoint,
            dataset_roots=dataset_roots,
            identity=identity,
        )
    _json_line(
        {
            "event": "result",
            **asdict(result),
            "checkpoint_path": str(published_checkpoint or checkpoint),
            "manifest_path": (
                str(published_manifest) if published_manifest is not None else None
            ),
            "training_data_identity": identity.identity,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        stop_requested = threading.Event()
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
        signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())
        try:
            run_from_args(args, should_stop=stop_requested.is_set)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
    except KeyboardInterrupt:
        _json_line(
            {"event": "error", "error_type": "KeyboardInterrupt", "message": "interrupted"},
            stream=sys.stderr,
        )
        return 130
    except Exception as error:
        _json_line(
            {
                "event": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            stream=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_from_args"]
