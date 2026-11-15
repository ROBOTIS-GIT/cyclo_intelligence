#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prepare immutable per-dataset replay inputs for fresh RLT Stage 2.

This process owns GR00T and the frozen Stage-1 encoder only long enough to
extract ``z_rl`` and the matching GR00T reference.  It never loads a Stage-2
bundle, Actor, Critic, optimizer, or training state.  Each completed replay is
published below ``<dataset>/rlt/replay_cache`` and can then be consumed by a
fresh Actor warm-up with no replay source bundle.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import torch

from cyclo_brain.algorithm.rl.rlt import (
    RLTStage2Spec,
    load_frozen_rl_token_encoder,
)
from cyclo_brain.model.common import RLT_ACTION_HORIZON

from .rlt_cli_common import json_line, positive_int, resolved_directory
from .rlt_provenance import (
    GR00TRLTProvenance,
    build_groot_rlt_provenance,
    validate_stage1_encoder_provenance,
)
from .rlt_stage2_dataset import (
    GR00TRLTStage2Extractor,
    RLTStage2DatasetConfig,
    RLTStage2FeatureReplay,
    materialize_rlt_stage2_replay,
    open_rlt_stage2_source,
)

from .rlt_tt_rtc_actor_replay import CACHE_SUBDIR, HOLD_CACHE_SUBDIR, TERMINAL_HOLD_POLICY


_REPLAY_RELATIVE_PATHS = {
    "legacy": Path("rlt/replay_cache"),
    "tt_rtc_actor": Path("rlt") / HOLD_CACHE_SUBDIR / "replay_cache",
    "tt_rtc_rl": Path("rlt/tt_rtc_transition_v1/replay_cache"),
}


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 <= parsed < 2**63 - 4:
        raise argparse.ArgumentTypeError("must be in 0..2**63-5")
    return parsed


def _resolved_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{name} must be a non-empty regular file: {path}")
    return path


def _spec_from_encoder(
    encoder: Any,
    *,
    action_hz: float,
    provenance: GR00TRLTProvenance,
) -> RLTStage2Spec:
    return RLTStage2Spec(
        reference_contract_fingerprint=(
            encoder.representation_contract_fingerprint
        ),
        rl_token_artifact_fingerprint=encoder.artifact_fingerprint,
        rl_token_dim=int(encoder.config.embedding_dim),
        proprio_dim=provenance.action_dim,
        reference_horizon=provenance.reference_horizon,
        chunk_length=RLT_ACTION_HORIZON,
        action_dim=provenance.action_dim,
        action_hz=float(action_hz),
        action_normalization_id=provenance.action_normalization_id,
        action_codec_id=provenance.action_codec_id,
        model_domain="normalized",
        schema_version=1,
    )


def _progress(
    *,
    dataset_path: Path,
    dataset_index: int,
    dataset_count: int,
    completed_features: int,
    total_features: int,
    started: float,
) -> None:
    fraction = (
        completed_features / total_features if total_features > 0 else 0.0
    )
    percentage = 100.0 * (dataset_index + fraction) / dataset_count
    elapsed = time.monotonic() - started
    eta = (
        elapsed * (100.0 - percentage) / percentage
        if percentage > 0.0
        else None
    )
    json_line(
        {
            "event": "stage2_replay_prepare_progress",
            "status": "running",
            "phase": "extracting_features",
            "dataset_path": str(dataset_path),
            "completed_datasets": dataset_index,
            "total_datasets": dataset_count,
            "completed_features": completed_features,
            "total_features": total_features,
            "percentage": percentage,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "message": (
                "Preparing Stage 2 replay for "
                f"{dataset_path.name} ({completed_features}/{total_features})"
            ),
        }
    )


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GR00T RLT replay preparation requires an available CUDA device")

    datasets = tuple(
        resolved_directory(path, "LeRobot dataset") for path in args.dataset_root
    )
    if not datasets:
        raise ValueError("RLT replay preparation requires at least one dataset")
    checkpoint = resolved_directory(args.groot_checkpoint, "GR00T checkpoint")
    encoder_path = _resolved_file(args.rl_token_encoder, "RL Token encoder")
    config = RLTStage2DatasetConfig()
    provenance = build_groot_rlt_provenance(checkpoint)
    started = time.monotonic()

    targets: list[Path] = []
    for dataset in datasets:
        rlt_root = dataset / "rlt"
        target = dataset / _REPLAY_RELATIVE_PATHS[args.replay_mode]
        if rlt_root.is_symlink() or target.is_symlink():
            raise ValueError(
                f"RLT replay output path is unsafe: {target}"
            )
        if target.exists() and not target.is_dir():
            raise ValueError(f"RLT replay output is not a directory: {target}")
        targets.append(target)

    json_line(
        {
            "event": "stage2_replay_prepare_manifest",
            "status": "running",
            "phase": "loading_frozen_sources",
            "job_id": args.job_id,
            "dataset_roots": [str(path) for path in datasets],
            "groot_checkpoint": str(checkpoint),
            "rl_token_encoder": str(encoder_path),
            "feature_batch_size": args.feature_batch_size,
            "replay_mode": args.replay_mode,
            "message": "Loading frozen GR00T and RL Token encoder",
        }
    )

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    encoder = load_frozen_rl_token_encoder(
        encoder_path,
        device=args.device,
        dtype=torch.float32,
    )
    validate_stage1_encoder_provenance(
        encoder.representation_contract,
        provenance,
    )
    spec = _spec_from_encoder(
        encoder,
        action_hz=config.expected_fps,
        provenance=provenance,
    )
    policy = None
    extractor = None
    prepared: list[tuple[Path | None, Path, Any]] = []
    try:
        if any(not target.exists() for target in targets):
            policy = Gr00tPolicy(
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                model_path=str(checkpoint),
                device=args.device,
            )
            if args.replay_mode != "legacy":
                from .rlt_tt_rtc_actor_replay import configure_tt_rtc_policy

                configure_tt_rtc_policy(policy, checkpoint)
            extractor = GR00TRLTStage2Extractor(policy, encoder)
        with ExitStack() as stack:
            for index, (dataset, target) in enumerate(
                zip(datasets, targets, strict=True)
            ):
                if target.exists():
                    replay = RLTStage2FeatureReplay.load(
                        target,
                        expected_spec=spec,
                    )
                    if (
                        args.replay_mode == "tt_rtc_actor"
                        and replay.metadata.get("terminal_hold_policy") != TERMINAL_HOLD_POLICY
                    ):
                        raise ValueError(f"Actor replay is missing successful terminal-hold windows: {target}")
                    recorded_roots = tuple(
                        Path(path).resolve()
                        for path in replay.metadata.get("dataset_roots", [])
                    )
                    if recorded_roots != (dataset.resolve(),):
                        raise ValueError(
                            "Existing Stage 2 replay dataset identity disagrees: "
                            f"{target}"
                        )
                    prepared.append((None, target, replay))
                    continue
                if extractor is None:
                    raise RuntimeError("GR00T replay extractor was not initialized")
                source = open_rlt_stage2_source(
                    dataset,
                    expected_fps=config.expected_fps,
                    action_dim=spec.action_dim,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary_root = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            prefix=f".replay-{args.job_id[:12]}-",
                            dir=target.parent,
                        )
                    )
                )
                temporary_replay = temporary_root / "replay_cache"
                progress = lambda completed, total, dataset=dataset, index=index: _progress(
                    dataset_path=dataset,
                    dataset_index=index,
                    dataset_count=len(datasets),
                    completed_features=completed,
                    total_features=total,
                    started=started,
                )
                if args.replay_mode == "legacy":
                    replay = materialize_rlt_stage2_replay(
                        [source],
                        extractor=extractor,
                        spec=spec,
                        output_root=temporary_replay,
                        config=config,
                        feature_batch_size=args.feature_batch_size,
                        reference_seed=args.seed,
                        require_both_outcomes=True,
                        progress_callback=progress,
                    )
                else:
                    from .rlt_tt_rtc_actor_replay import (
                        materialize_tt_rtc_actor_replay,
                    )

                    # Actor warmup needs terminal holds; Critic/RL retains the
                    # unpadded action windows used for delayed transitions.
                    actor_target = (target if args.replay_mode == "tt_rtc_actor"
                                    else dataset / "rlt" / CACHE_SUBDIR / "replay_cache")
                    if actor_target.is_dir():
                        actor_replay = RLTStage2FeatureReplay.load(
                            actor_target,
                            expected_spec=spec,
                        )
                    else:
                        actor_target.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(
                            prefix=f".actor-{args.job_id[:12]}-",
                            dir=actor_target.parent,
                        ) as actor_temporary_root:
                            actor_temporary = Path(actor_temporary_root) / "replay_cache"
                            actor_replay = materialize_tt_rtc_actor_replay(
                                [source],
                                extractor=extractor,
                                spec=spec,
                                output_root=actor_temporary,
                                config=config,
                                feature_batch_size=1,
                                reference_seed=args.seed,
                                require_both_outcomes=False,
                                terminal_hold=args.replay_mode == "tt_rtc_actor",
                                progress_callback=progress,
                            )
                            actor_temporary.rename(actor_target)
                    if args.replay_mode == "tt_rtc_actor":
                        replay = actor_replay
                        prepared.append((None, target, replay))
                        continue

                    from .rlt_tt_rtc_transition_replay import (
                        build_tt_rtc_transition_replay,
                    )

                    replay = build_tt_rtc_transition_replay(
                        actor_replay,
                        output_root=temporary_replay,
                    )
                prepared.append((temporary_replay, target, replay))

            for temporary_replay, target, _replay in prepared:
                if temporary_replay is None:
                    continue
                if target.exists() or target.is_symlink():
                    raise RuntimeError(
                        "RLT replay output appeared during preparation; refusing to overwrite "
                        f"{target}"
                    )
                temporary_replay.rename(target)
    finally:
        if extractor is not None:
            del extractor
        if policy is not None:
            del policy
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_transitions = sum(len(replay) for _temporary, _target, replay in prepared)
    weighted_reward = sum(
        len(replay) * replay.average_reward
        for _temporary, _target, replay in prepared
    )
    average_reward = weighted_reward / total_transitions
    json_line(
        {
            "event": "stage2_replay_prepare_result",
            "status": "completed",
            "phase": "complete",
            "job_id": args.job_id,
            "replay_paths": [str(target) for target in targets],
            "dataset_roots": [str(path) for path in datasets],
            "transition_count": total_transitions,
            "average_reward": average_reward,
            "spec_fingerprint": prepared[0][2].spec_fingerprint,
            "replay_mode": args.replay_mode,
            "percentage": 100.0,
            "elapsed_seconds": time.monotonic() - started,
            "message": "Stage 2 replay preparation completed",
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--groot-checkpoint", required=True)
    parser.add_argument("--rl-token-encoder", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--feature-batch-size", type=positive_int, default=2)
    parser.add_argument(
        "--replay-mode",
        choices=tuple(_REPLAY_RELATIVE_PATHS),
        default="legacy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=_seed, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        json_line(
            {
                "event": "stage2_replay_prepare_result",
                "status": "stopped",
                "job_id": args.job_id,
                "message": "Stage 2 replay preparation stopped before publication",
            },
            stream=sys.stderr,
        )
        return 130
    except Exception as error:
        json_line(
            {
                "event": "stage2_replay_prepare_error",
                "status": "failed",
                "job_id": args.job_id,
                "message": (
                    "Stage 2 replay preparation failed: "
                    f"{type(error).__name__}: {error}"
                ),
                "error": f"{type(error).__name__}: {error}",
            },
            stream=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
