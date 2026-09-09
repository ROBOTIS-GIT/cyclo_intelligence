"""Command-line entrypoint for multi-root ACT imitation learning."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from cyclo_brain.algorithm.il.common.cli import (
    episode_csv as _success_csv,
    json_line as _json_line,
    non_negative_integer as _non_negative,
    positive_float as _positive_float,
    positive_integer as _positive,
    run_with_stop_signals,
)
from cyclo_brain.model.act.trainability import ACT_TRAINABLE_GROUPS

from .dataset import RootSelection
from .training import (
    ACT_CHUNK_SIZE,
    ACTBCTrainingConfig,
    ACTBCTrainingProgress,
    run_training,
    write_failed_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh LeRobot 0.5.2 ACT/CVAE policy on selected "
            "demonstrations from ordered immutable LeRobot v3 data epochs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        type=Path,
        help="LeRobot v3 data-epoch root; repeat in cumulative collection order.",
    )
    parser.add_argument(
        "--episodes",
        "--success-episodes",
        dest="success_episodes",
        action="append",
        required=True,
        type=_success_csv,
        help=(
            "Comma-separated root-local demonstration episode indices. Labeled "
            "roots verify that selected episodes succeeded; unlabeled roots need "
            "no outcome feature. Repeat once per --dataset-root in the same order."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=_positive)
    parser.add_argument("--batch-size", required=True, type=_positive)
    parser.add_argument("--save-freq", required=True, type=_positive)
    parser.add_argument(
        "--chunk-size",
        type=_positive,
        default=ACT_CHUNK_SIZE,
        help=(
            "ACT prediction and execution horizon. "
            f"Defaults to {ACT_CHUNK_SIZE}; the supervisor accepts 1 through 100."
        ),
    )
    parser.add_argument("--progress-interval", type=_positive, default=10)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-5)
    parser.add_argument("--num-workers", type=_non_negative, default=4)
    parser.add_argument("--seed", type=_non_negative, default=1000)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "torchcodec", "video_reader"),
        default="pyav",
    )
    parser.add_argument("--grad-clip-norm", type=_positive_float, default=10.0)
    parser.add_argument(
        "--trainable-group",
        action="append",
        choices=ACT_TRAINABLE_GROUPS,
        dest="trainable_groups",
        help="ACT network group to train; repeat as needed. Defaults to every group.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ACTBCTrainingConfig:
    roots = tuple(args.dataset_root or ())
    episode_groups = tuple(args.success_episodes or ())
    if len(roots) != len(episode_groups):
        raise ValueError(
            "--dataset-root and --success-episodes must be repeated the same number of times"
        )
    selections = tuple(
        RootSelection(root=root, success_episodes=episodes)
        for root, episodes in zip(roots, episode_groups, strict=True)
    )
    return ACTBCTrainingConfig(
        selections=selections,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        chunk_size=args.chunk_size,
        progress_interval=args.progress_interval,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        video_backend=args.video_backend,
        grad_clip_norm=args.grad_clip_norm,
        trainable_groups=tuple(args.trainable_groups or ACT_TRAINABLE_GROUPS),
    )


def _progress_line(progress: ACTBCTrainingProgress) -> None:
    _json_line(progress.to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(argv)
        config = config_from_args(args)
        result = run_with_stop_signals(
            lambda should_stop: run_training(
                config,
                should_stop=should_stop,
                progress_callback=_progress_line,
            )
        )
        _json_line(result.to_dict())
        return 0
    except KeyboardInterrupt:
        value = {
            "event": "error",
            "error_type": "KeyboardInterrupt",
            "message": "interrupted",
        }
        _json_line(value, stream=sys.stderr)
        return 130
    except Exception as error:
        if args is not None and getattr(args, "output_dir", None) is not None:
            write_failed_result(args.output_dir, error)
        _json_line(
            {
                "event": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
