"""Standalone cumulative-replay ACT-TD3 training and ACT export command."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from cyclo_brain.model.act import (
    ACT_TRAINABLE_GROUPS,
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
from .offline_training import (
    ACTTD3OfflineTrainingProgress,
    ACTTD3OfflineTrainingRunner,
)
from .offline_warmup_cli import (
    _MAX_SEED,
    _VIDEO_BACKENDS,
    _device,
    _input_directory,
    _input_file,
    _positive,
    _require_local_dataset_layout,
    _require_referenced_dataset_files,
    _seed,
)
from .training_identity import (
    build_act_td3_multi_root_training_data_identity,
    build_act_td3_training_data_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one cumulative-replay ACT-TD3 round with a fixed 2:1 "
            "critic-to-actor update schedule, capped at 200 episodes."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        action="append",
        help=(
            "Immutable LeRobot v3 data-epoch root; repeat in collection order "
            "to construct a virtual cumulative replay without merging files."
        ),
    )
    parser.add_argument("--act-checkpoint", required=True, type=Path)
    parser.add_argument("--robot-config", required=True, type=Path)
    parser.add_argument("--robot-type", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", required=True, type=_seed)
    parser.add_argument(
        "--sampling-seed",
        type=_seed,
        help="Replay permutation seed; defaults to seed + 2.",
    )
    parser.add_argument("--batch-size", required=True, type=_positive)
    parser.add_argument(
        "--actor-trainable-group",
        action="append",
        choices=ACT_TRAINABLE_GROUPS,
        dest="actor_trainable_groups",
        help=(
            "ACT parameter group to update; repeat for multiple groups. "
            "Defaults to every group."
        ),
    )
    parser.add_argument(
        "--critic-epochs",
        type=_positive,
        default=ACTTD3OfflineTrainingRunner.CRITIC_EPOCHS,
    )
    parser.add_argument(
        "--actor-equivalent-epochs",
        type=_positive,
        default=ACTTD3OfflineTrainingRunner.ACTOR_EQUIVALENT_EPOCHS,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Immutable round directory. Contains training_state/act_td3.pt and, "
            "after completion, pretrained_model/."
        ),
    )
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted round from output-dir.",
    )
    continuation.add_argument(
        "--parent-checkpoint",
        type=Path,
        help="Completed prior round training_state/act_td3.pt for a grown replay.",
    )
    parser.add_argument(
        "--allow-partial-round",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Every round now infers and accepts "
            "1..50 new episodes from the dataset and optional parent checkpoint."
        ),
    )
    parser.add_argument(
        "--max-round-critic-updates",
        type=_positive,
        help="Absolute in-round smoke/resume boundary; omit for all critic epochs.",
    )
    parser.add_argument("--checkpoint-interval", type=_positive, default=100)
    parser.add_argument("--progress-interval", type=_positive, default=10)
    parser.add_argument(
        "--video-backend",
        choices=_VIDEO_BACKENDS,
        default="pyav",
    )
    return parser


def _json_line(value: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=stream,
        flush=True,
    )


def _progress_line(progress: ACTTD3OfflineTrainingProgress) -> None:
    _json_line({"event": "progress", **asdict(progress)})


def _validate_schedule(critic_epochs: int, actor_equivalent_epochs: int) -> None:
    if (
        critic_epochs
        != ACTTD3OfflineTrainingRunner.POLICY_UPDATE_PERIOD
        * actor_equivalent_epochs
    ):
        raise ValueError(
            "ACT-TD3 critic_epochs must equal policy_update_period "
            "times actor_equivalent_epochs"
        )


def _schedule_manifest(runner: ACTTD3OfflineTrainingRunner) -> dict[str, int]:
    return {
        "critic_epochs": runner.critic_epochs,
        "actor_equivalent_epochs": runner.actor_equivalent_epochs,
        "policy_update_period": runner.POLICY_UPDATE_PERIOD,
        # Retain the original field for consumers while defining it as a cap.
        "round_episodes": runner.ROUND_EPISODES,
        "max_new_episodes_per_round": runner.ROUND_EPISODES,
        "max_episodes": runner.MAX_EPISODES,
    }


def _round_manifest(runner: ACTTD3OfflineTrainingRunner) -> dict[str, int]:
    return {
        "index": runner.round_index,
        "new_episodes": runner.new_episode_count,
        "batches_per_epoch": runner.batches_per_epoch,
        "critic_updates": runner.total_critic_updates,
        "actor_updates": runner.total_actor_updates,
    }


def _result_manifest(
    result: ACTTD3OfflineTrainingProgress,
    *,
    actor_trainable_groups: Sequence[str],
    runner: ACTTD3OfflineTrainingRunner,
    identity: Any,
    model_directory: Path | None,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "event": "result",
        **asdict(result),
        "actor_trainable_groups": list(actor_trainable_groups),
        "batch_size": batch_size,
        "schedule": _schedule_manifest(runner),
        "round": _round_manifest(runner),
        "training_data": _training_identity_summary(identity),
        "model_path": str(model_directory) if model_directory is not None else None,
    }


def _dataset_root_arguments(value: Any) -> tuple[Path, ...]:
    """Normalize argparse and legacy programmatic one-root namespaces."""

    if isinstance(value, Path):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        roots = tuple(value)
        if roots and all(isinstance(root, Path) for root in roots):
            return roots
    raise TypeError("dataset_root must contain one or more paths")


def _training_identity_summary(identity: Any) -> dict[str, Any]:
    roots = identity.virtual_contract.get("data_roots", [])
    return {
        "identity": identity.identity,
        "file_count": identity.file_count,
        "byte_count": identity.byte_count,
        "component_sha256": identity.component_sha256,
        "data_roots": roots,
    }


def _output_directory(
    value: Path,
    *,
    resume: bool,
    inputs: Sequence[Path],
) -> Path:
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise ValueError("ACT-TD3 output directory must not be a symbolic link")
    resolved = expanded.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    if resume:
        checkpoint = resolved / "training_state" / "act_td3.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ACT-TD3 resume checkpoint does not exist: {checkpoint}"
            )
    elif resolved.exists():
        raise FileExistsError(f"ACT-TD3 output directory already exists: {resolved}")
    for input_path in inputs:
        input_root = input_path if input_path.is_dir() else input_path.parent
        if resolved == input_root or input_root in resolved.parents:
            raise ValueError(
                "ACT-TD3 output must be outside dataset, ACT checkpoint, "
                "and robot-config inputs"
            )
    return resolved


def _atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
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
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _assert_actor_state_equal(expected: Mapping[str, Tensor], actual: Any) -> None:
    actual_state = actual.state_dict()
    if set(expected) != set(actual_state):
        raise RuntimeError("exported ACT actor state keys disagree")
    for name, expected_value in expected.items():
        actual_value = actual_state[name]
        if not isinstance(actual_value, Tensor):
            raise RuntimeError(f"exported ACT actor value is not a tensor: {name}")
        expected_cpu = expected_value.detach().cpu()
        actual_cpu = actual_value.detach().cpu()
        if expected_cpu.dtype != actual_cpu.dtype or expected_cpu.shape != actual_cpu.shape:
            raise RuntimeError(
                f"exported ACT actor tensor contract disagrees: {name}"
            )
        if not torch.equal(expected_cpu, actual_cpu):
            difference = (expected_cpu - actual_cpu).abs()
            raise RuntimeError(
                "exported ACT actor tensor disagrees: "
                f"{name}; max_abs={float(difference.max())}; "
                f"different={int(torch.ne(expected_cpu, actual_cpu).sum())}; "
                f"expected_finite={bool(torch.isfinite(expected_cpu).all())}; "
                f"actual_finite={bool(torch.isfinite(actual_cpu).all())}"
            )


def _verify_export(
    model_directory: Path,
    *,
    expected_actor_state: Mapping[str, Tensor],
    expected_action_mean: Tensor,
    expected_action_std: Tensor,
    expected_normalizer_eps: float,
    device: torch.device,
) -> None:
    verified = load_act_policy_assets(model_directory, device=device)
    _assert_actor_state_equal(expected_actor_state, verified.policy)
    if (
        not torch.equal(expected_action_mean.cpu(), verified.action_mean.cpu())
        or not torch.equal(expected_action_std.cpu(), verified.action_std.cpu())
        or float(expected_normalizer_eps) != float(verified.normalizer_eps)
    ):
        raise RuntimeError("exported ACT processor action statistics disagree")


def _export_policy_assets(
    model_directory: Path,
    *,
    learner: ACTTD3Learner,
    source_assets: Any,
) -> None:
    expected_actor_state = {
        name: value.detach().cpu().clone()
        for name, value in learner.actor.state_dict().items()
    }
    verification = {
        "expected_actor_state": expected_actor_state,
        "expected_action_mean": source_assets.action_mean,
        "expected_action_std": source_assets.action_std,
        "expected_normalizer_eps": source_assets.normalizer_eps,
        "device": learner.device,
    }
    if model_directory.exists():
        if not model_directory.is_dir():
            raise NotADirectoryError(model_directory)
        _verify_export(model_directory, **verification)
        return

    model_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{model_directory.name}.",
            suffix=".tmp",
            dir=model_directory.parent,
        )
    )
    try:
        learner.actor.eval().save_pretrained(temporary_directory)
        source_assets.preprocessor.save_pretrained(
            temporary_directory,
            config_filename="policy_preprocessor.json",
        )
        source_assets.postprocessor.save_pretrained(
            temporary_directory,
            config_filename="policy_postprocessor.json",
        )
        _assert_actor_state_equal(expected_actor_state, learner.actor)
        _verify_export(temporary_directory, **verification)
        os.replace(temporary_directory, model_directory)
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


def _publish_policy_assets_for_unchanged_training_data(
    model_directory: Path,
    *,
    learner: ACTTD3Learner,
    source_assets: Any,
    expected_identity: Any,
    dataset: Any,
    dataset_root: Path,
    act_checkpoint_root: Path,
    action_domain: Any,
    robot_type: str,
    video_backend: str,
) -> None:
    current_identity = build_act_td3_training_data_identity(
        dataset,
        dataset_root=dataset_root,
        act_checkpoint_root=act_checkpoint_root,
        action_domain=action_domain,
        robot_type=robot_type,
        video_backend=video_backend,
    )
    if current_identity != expected_identity:
        raise RuntimeError(
            "ACT-TD3 training data identity changed during training; "
            "refusing model export"
        )
    _export_policy_assets(
        model_directory,
        learner=learner,
        source_assets=source_assets,
    )


def _publish_policy_assets_for_unchanged_multi_root_training_data(
    model_directory: Path,
    *,
    learner: ACTTD3Learner,
    source_assets: Any,
    expected_identity: Any,
    datasets: Sequence[Any],
    dataset_roots: Sequence[Path],
    act_checkpoint_root: Path,
    action_domains: Sequence[Any],
    robot_type: str,
    video_backend: str,
) -> None:
    current_identity = build_act_td3_multi_root_training_data_identity(
        datasets,
        dataset_roots,
        act_checkpoint_root=act_checkpoint_root,
        action_domains=action_domains,
        robot_type=robot_type,
        video_backend=video_backend,
    )
    if current_identity != expected_identity:
        raise RuntimeError(
            "ACT-TD3 training data identity changed during training; "
            "refusing model export"
        )
    _export_policy_assets(
        model_directory,
        learner=learner,
        source_assets=source_assets,
    )


def _run_from_args_unlocked(
    args: argparse.Namespace,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> ACTTD3OfflineTrainingProgress:
    _validate_schedule(args.critic_epochs, args.actor_equivalent_epochs)
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
    parent_checkpoint = (
        _input_file(args.parent_checkpoint, "parent checkpoint")
        if args.parent_checkpoint is not None
        else None
    )
    output_dir = _output_directory(
        args.output_dir,
        resume=args.resume,
        inputs=(*dataset_roots, act_checkpoint, robot_config),
    )
    checkpoint = output_dir / "training_state" / "act_td3.pt"
    if args.resume:
        resume_from = checkpoint
    else:
        resume_from = parent_checkpoint

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets: list[Any] = []
    action_domains: list[Any] = []
    for root_index, dataset_root in enumerate(dataset_roots):
        _require_local_dataset_layout(dataset_root)
        local_repo_id = f"local/data_epoch_{root_index:04d}_{dataset_root.name}"
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
            "ACT-TD3 cumulative replay requires at least one success and one failure"
        )
    collator = ACTTD3LeRobotCollator(assets.preprocessor)
    config = ACTTD3Config(
        discount_reference_hz=float(replay.fps),
        critic_warmup_updates=0,
        actor_trainable_groups=tuple(
            args.actor_trainable_groups or ACT_TRAINABLE_GROUPS
        ),
    )
    learner = ACTTD3Learner(
        actor,
        critic,
        config,
        random_seed=args.seed,
    )
    runner = ACTTD3OfflineTrainingRunner(
        learner,
        replay,
        collator,
        batch_size=args.batch_size,
        sampling_seed=sampling_seed,
        training_data_identity=identity,
        checkpoint_path=checkpoint,
        resume_from=resume_from,
        critic_epochs=args.critic_epochs,
        actor_equivalent_epochs=args.actor_equivalent_epochs,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
    )

    _json_line(
        {
            "event": "manifest",
            "algorithm": "ACT-TD3 cumulative replay",
            "actor_trainable_groups": list(config.actor_trainable_groups),
            "schedule": _schedule_manifest(runner),
            "device": str(device),
            "seed": args.seed,
            "sampling_seed": sampling_seed,
            "batch_size": args.batch_size,
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint),
            "resume_from": str(resume_from) if resume_from is not None else None,
            "legacy_allow_partial_round": bool(args.allow_partial_round),
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
            "round": _round_manifest(runner),
            "training_data": {
                **_training_identity_summary(identity),
                "video_backend": args.video_backend,
            },
        }
    )
    result = runner.run(
        max_round_critic_updates=args.max_round_critic_updates,
        progress_callback=_progress_line,
        should_stop=should_stop,
    )

    model_directory: Path | None = None
    if result.status == "complete":
        model_directory = output_dir / "pretrained_model"
        _publish_policy_assets_for_unchanged_multi_root_training_data(
            model_directory,
            learner=learner,
            source_assets=assets,
            expected_identity=identity,
            datasets=datasets,
            dataset_roots=dataset_roots,
            act_checkpoint_root=act_checkpoint,
            action_domains=action_domains,
            robot_type=args.robot_type,
            video_backend=args.video_backend,
        )
    final = _result_manifest(
        result,
        actor_trainable_groups=config.actor_trainable_groups,
        runner=runner,
        identity=identity,
        model_directory=model_directory,
        batch_size=args.batch_size,
    )
    _atomic_json_save(output_dir / "training_manifest.json", final)
    _json_line(final)
    return result


def run_from_args(
    args: argparse.Namespace,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> ACTTD3OfflineTrainingProgress:
    """Validate, train, and export without blocking concurrent recording."""
    return _run_from_args_unlocked(args, should_stop=should_stop)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        stop_requested = threading.Event()
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(
            signal.SIGINT,
            lambda _signum, _frame: stop_requested.set(),
        )
        signal.signal(
            signal.SIGTERM,
            lambda _signum, _frame: stop_requested.set(),
        )
        try:
            run_from_args(args, should_stop=stop_requested.is_set)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
    except KeyboardInterrupt:
        _json_line(
            {
                "event": "error",
                "error_type": "KeyboardInterrupt",
                "message": "interrupted",
            },
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
