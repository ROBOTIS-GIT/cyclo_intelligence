"""Standalone local-data CLI for the actor-frozen ACT-TD3 critic warm-up."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

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
)
from .offline_warmup import ACTTD3CriticWarmupProgress, ACTTD3CriticWarmupRunner
from .training_identity import build_act_td3_training_data_identity


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


def _warmup_boundary(value: str) -> int:
    return _integer(
        value,
        name="max_critic_updates",
        minimum=1,
        maximum=ACTTD3Config().critic_warmup_updates,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up ACT-TD3 critics from a finalized local LeRobot v3 dataset. "
            "The official ACT actor remains bitwise unchanged."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
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


def _output_checkpoint(path: Path, *, resume: bool, inputs: Sequence[Path]) -> Path:
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
    for input_path in inputs:
        input_root = input_path if input_path.is_dir() else input_path.parent
        if resolved == input_root or input_root in resolved.parents:
            raise ValueError(
                "ACT-TD3 checkpoint must be outside dataset, ACT checkpoint, "
                "and robot-config inputs"
            )
    return resolved


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


def run_from_args(args: argparse.Namespace) -> ACTTD3CriticWarmupProgress:
    dataset_root = _input_directory(args.dataset_root, "dataset root")
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
    checkpoint = _output_checkpoint(
        args.checkpoint,
        resume=args.resume,
        inputs=(dataset_root, act_checkpoint, robot_config),
    )
    _require_local_dataset_layout(dataset_root)

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Resolve every referenced local file before LeRobotDataset construction.
    # This prevents its Hub fallback from turning a malformed local run into a
    # network-dependent one.
    local_repo_id = f"local/{dataset_root.name}"
    metadata = LeRobotDatasetMetadata(repo_id=local_repo_id, root=dataset_root)
    _require_referenced_dataset_files(dataset_root, metadata)

    dataset = LeRobotDataset(
        repo_id=local_repo_id,
        root=dataset_root,
        image_transforms=None,
        delta_timestamps=None,
        force_cache_sync=False,
        download_videos=False,
        video_backend=args.video_backend,
        return_uint8=False,
    )
    action_domain = load_act_physical_action_domain(
        dataset_root / "meta" / "info.json",
        robot_config,
        robot_type=args.robot_type,
    )
    identity = build_act_td3_training_data_identity(
        dataset,
        dataset_root=dataset_root,
        act_checkpoint_root=act_checkpoint,
        action_domain=action_domain,
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

    replay = FixedHorizonLeRobotACTTD3Dataset(
        dataset,
        execution_horizon=int(actor.config.n_action_steps),
        observation_keys=tuple(actor.config.input_features or {}),
    )
    collator = ACTTD3LeRobotCollator(assets.preprocessor)
    config = ACTTD3Config(discount_reference_hz=float(replay.fps))
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
            "resume": args.resume,
            "max_critic_updates": args.max_critic_updates,
            "total_critic_updates": config.critic_warmup_updates,
            "dataset": {
                "frames": len(dataset),
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
    return runner.run(
        max_critic_updates=args.max_critic_updates,
        progress_callback=_progress_line,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run_from_args(args)
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
