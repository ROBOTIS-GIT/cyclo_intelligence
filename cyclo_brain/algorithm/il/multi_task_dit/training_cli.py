"""Command-line entrypoint for MultiTaskDiT imitation learning."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from cyclo_brain.algorithm.il.common.cli import (
    episode_csv as _episode_csv,
    json_line as _json_line,
    non_negative_integer as _non_negative,
    positive_float as _positive_float,
    positive_integer as _positive,
    run_with_stop_signals,
)
from cyclo_brain.algorithm.il.common.dataset import RootSelection

from .training import (
    DEFAULT_TASK_INSTRUCTION,
    MULTI_TASK_DIT_HORIZON,
    MultiTaskDiTILConfig,
    MultiTaskDiTILProgress,
    run_training,
    training_manifest,
    write_failed_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh LeRobot MultiTaskDiT flow-matching policy on selected "
            "demonstrations from ordered immutable LeRobot v3 roots."
        )
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        type=Path,
        help="LeRobot v3 root; repeat in cumulative collection order.",
    )
    parser.add_argument(
        "--episodes",
        "--success-episodes",
        dest="episodes",
        action="append",
        required=True,
        type=_episode_csv,
        help=(
            "Comma-separated root-local demonstration episode indices. Labeled "
            "roots verify success; unlabeled roots need no outcome feature."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=_positive)
    parser.add_argument("--batch-size", required=True, type=_positive)
    parser.add_argument("--save-freq", required=True, type=_positive)
    parser.add_argument(
        "--chunk-size",
        type=_positive,
        default=MULTI_TASK_DIT_HORIZON,
        help=(
            "Action prediction/execution horizon; current Cyclo contract requires "
            f"{MULTI_TASK_DIT_HORIZON}."
        ),
    )
    parser.add_argument(
        "--task-instruction",
        default=DEFAULT_TASK_INSTRUCTION,
        help="Language instruction used to condition every selected demonstration.",
    )
    parser.add_argument("--progress-interval", type=_positive, default=10)
    parser.add_argument("--learning-rate", type=_positive_float, default=2e-5)
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
        "--no-amp",
        action="store_true",
        help="Disable CUDA bfloat16 autocast for diagnosis.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MultiTaskDiTILConfig:
    roots = tuple(args.dataset_root or ())
    episode_groups = tuple(args.episodes or ())
    if len(roots) != len(episode_groups):
        raise ValueError(
            "--dataset-root and --episodes must be repeated the same number of times"
        )
    selections = tuple(
        RootSelection(root=root, success_episodes=episodes)
        for root, episodes in zip(roots, episode_groups, strict=True)
    )
    return MultiTaskDiTILConfig(
        selections=selections,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        chunk_size=args.chunk_size,
        task_instruction=args.task_instruction,
        progress_interval=args.progress_interval,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        video_backend=args.video_backend,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=not args.no_amp,
    )


def _progress_line(progress: MultiTaskDiTILProgress) -> None:
    _json_line(progress.to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(argv)
        config = config_from_args(args)
        _json_line(training_manifest(config))
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
