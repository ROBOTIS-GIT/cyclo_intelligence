#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Train the PI-RLT Action MLP and twin-Q critics from recorded replay.

This one-shot process is launched by ``rlt_stage2_service.py``.  It first
materializes frozen GR00T/Stage-1 features, unloads GR00T, and only then places
the compact Stage-2 learner on the requested device.  ``steps`` always means
additional critic updates in this invocation; a resume run retains cumulative
actor/critic counters and optimizer/RNG state in a new immutable bundle.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import gc
import math
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

from cyclo_brain.algorithm.rl.rlt import (
    RLTStage2Config,
    RLTStage2Run,
    RLTStage2Spec,
    build_stage2_training_round,
    load_frozen_rl_token_encoder,
    validate_stage2_replay_lineage,
)
from cyclo_brain.model.common import (
    GROOT_REFERENCE_ACTION_HORIZON,
    RLT_ACTION_DIM,
    RLT_ACTION_HORIZON,
)

from .rlt_cli_common import (
    json_line,
    positive_int,
    prepare_output_directory,
    resolved_directory,
)
from .rlt_provenance import (
    GR00TRLTProvenance,
    build_groot_rlt_provenance,
    validate_stage1_encoder_provenance,
)
from .rlt_stage2_dataset import (
    GR00TRLTStage2Extractor,
    RLTStage2DatasetConfig,
    RLTStage2Source,
    materialize_rlt_stage2_replay,
    open_rlt_stage2_source,
)


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


def _sample_indices(
    sample_count: int,
    batch_size: int,
    steps: int,
    *,
    seed: int,
) -> Iterator[list[int]]:
    if sample_count < 1 or batch_size < 1 or steps < 1:
        raise ValueError("RLT Stage 2 sampler dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pending: list[int] = []
    for _ in range(steps):
        while len(pending) < batch_size:
            pending.extend(torch.randperm(sample_count, generator=generator).tolist())
        yield pending[:batch_size]
        del pending[:batch_size]


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
        proprio_dim=RLT_ACTION_DIM,
        reference_horizon=GROOT_REFERENCE_ACTION_HORIZON,
        chunk_length=RLT_ACTION_HORIZON,
        action_dim=RLT_ACTION_DIM,
        action_hz=float(action_hz),
        action_normalization_id=provenance.action_normalization_id,
        action_codec_id=provenance.action_codec_id,
        model_domain="normalized",
        schema_version=1,
    )


def _validate_encoder_checkpoint_identity(
    encoder: Any,
    provenance: GR00TRLTProvenance,
) -> None:
    validate_stage1_encoder_provenance(
        encoder.representation_contract,
        provenance,
    )


def _progress(
    *,
    phase: str,
    completed_steps: int,
    total_steps: int,
    started: float,
    actor_loss: float | None,
    critic_loss: float | None,
    average_reward: float | None,
    percentage: float | None = None,
    message: str,
) -> None:
    elapsed = time.monotonic() - started
    eta = (
        elapsed / completed_steps * (total_steps - completed_steps)
        if completed_steps > 0 and total_steps >= completed_steps
        else None
    )
    json_line(
        {
            "event": "stage2_training_progress",
            "status": "running",
            "phase": phase,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "percentage": (
                float(percentage)
                if percentage is not None
                else 100.0 * completed_steps / total_steps
            ),
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "average_reward": average_reward,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "message": message,
        }
    )


def _load_sources(
    roots: Sequence[Path],
    config: RLTStage2DatasetConfig,
) -> tuple[RLTStage2Source, ...]:
    sources = tuple(
        open_rlt_stage2_source(root, expected_fps=config.expected_fps)
        for root in roots
    )
    if not sources:
        raise ValueError("RLT Stage 2 requires at least one LeRobot dataset")
    return sources


def _bind_training_round_with_lineage(
    stage2: RLTStage2Run,
    training_round: Mapping[str, Any],
    *,
    replay_root: Path,
) -> None:
    """Bind a round only after enforcing immutable-prefix resume lineage."""

    if stage2.initialization_mode == "resume":
        if stage2.training_round is None:
            raise ValueError("RLT Stage 2 resume bundle has no parent training round")
        validate_stage2_replay_lineage(stage2.training_round, training_round)
    stage2.bind_training_round(training_round, replay_root=replay_root)


def _materialize(
    *,
    checkpoint: Path,
    encoder_path: Path,
    sources: Sequence[RLTStage2Source],
    spec: RLTStage2Spec,
    output: Path,
    dataset_config: RLTStage2DatasetConfig,
    feature_batch_size: int,
    reference_seed: int,
    device: str,
    total_steps: int,
    started: float,
):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GR00T RLT Stage 2 requires an available CUDA device")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    encoder = load_frozen_rl_token_encoder(
        encoder_path,
        device=device,
        dtype=torch.float32,
    )
    policy = None
    extractor = None
    last_reported = -1

    def extraction_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        interval = max(1, total // 100)
        if completed != total and completed - last_reported < interval:
            return
        last_reported = completed
        _progress(
            phase="preparing_replay",
            completed_steps=0,
            total_steps=total_steps,
            started=started,
            actor_loss=None,
            critic_loss=None,
            average_reward=None,
            percentage=20.0 * completed / total if total else 0.0,
            message=f"Extracting frozen GR00T features ({completed}/{total})",
        )

    try:
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(checkpoint),
            device=device,
        )
        extractor = GR00TRLTStage2Extractor(policy, encoder)
        replay = materialize_rlt_stage2_replay(
            sources,
            extractor=extractor,
            spec=spec,
            output_root=output / "replay_cache",
            config=dataset_config,
            feature_batch_size=feature_batch_size,
            reference_seed=reference_seed,
            progress_callback=extraction_progress,
        )
    finally:
        if extractor is not None:
            del extractor
        if policy is not None:
            del policy
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return replay


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_roots = tuple(
        resolved_directory(path, "LeRobot dataset") for path in args.dataset_root
    )
    dataset_config = RLTStage2DatasetConfig()
    sources = _load_sources(dataset_roots, dataset_config)
    initial_run: RLTStage2Run | None = None
    if args.initialization_mode == "new":
        if not args.groot_checkpoint or not args.rl_token_encoder or args.rlt_bundle:
            raise ValueError(
                "new RLT Stage 2 requires --groot-checkpoint and --rl-token-encoder only"
            )
        checkpoint = resolved_directory(args.groot_checkpoint, "GR00T checkpoint")
        encoder_path = _resolved_file(args.rl_token_encoder, "RL Token encoder")
        inputs = (*dataset_roots, checkpoint, encoder_path.parent.parent)
    else:
        if args.groot_checkpoint or args.rl_token_encoder or not args.rlt_bundle:
            raise ValueError("resume RLT Stage 2 requires --rlt-bundle only")
        bundle = resolved_directory(args.rlt_bundle, "RLT Stage 2 bundle")
        initial_run = RLTStage2Run.resume(
            bundle,
            device="cpu",
            expected_replay_root=bundle / "replay_cache",
        )
        checkpoint = resolved_directory(
            initial_run.source.groot_checkpoint,
            "bundled GR00T checkpoint",
        )
        encoder_path = _resolved_file(
            initial_run.encoder_artifact_path,
            "bundled RL Token encoder",
        )
        inputs = (*dataset_roots, bundle, checkpoint)

    output = prepare_output_directory(
        args.output_dir,
        inputs,
        stage_label="RLT Stage 2",
        overlap_description="an input",
    )
    provenance = build_groot_rlt_provenance(checkpoint)
    checkpoint_fingerprint = provenance.checkpoint_fingerprint
    encoder_cpu = load_frozen_rl_token_encoder(encoder_path, device="cpu")
    _validate_encoder_checkpoint_identity(encoder_cpu, provenance)
    if initial_run is None:
        spec = _spec_from_encoder(
            encoder_cpu,
            action_hz=dataset_config.expected_fps,
            provenance=provenance,
        )
    else:
        if initial_run.source.groot_checkpoint_fingerprint != checkpoint_fingerprint:
            raise ValueError("RLT resume GR00T checkpoint fingerprint disagrees")
        spec = initial_run.learner.spec
        if spec.rl_token_artifact_fingerprint != encoder_cpu.artifact_fingerprint:
            raise ValueError("RLT resume encoder fingerprint disagrees")
        if (
            spec.action_normalization_id != provenance.action_normalization_id
            or spec.action_codec_id != provenance.action_codec_id
        ):
            raise ValueError("RLT resume action processor contract disagrees")
        if not math.isclose(
            spec.action_hz,
            dataset_config.expected_fps,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("RLT resume action Hz disagrees with replay")
    representation_fingerprint = encoder_cpu.representation_contract_fingerprint
    del encoder_cpu
    if initial_run is not None:
        del initial_run

    started = time.monotonic()
    json_line(
        {
            "event": "stage2_training_manifest",
            "status": "running",
            "phase": "preparing_replay",
            "job_id": args.job_id,
            "initialization_mode": args.initialization_mode,
            "dataset_roots": [str(root) for root in dataset_roots],
            "groot_checkpoint": str(checkpoint),
            "rl_token_encoder": str(encoder_path),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "save_freq": args.save_freq,
            "dataset_contract": {
                "fps": dataset_config.expected_fps,
                "chunk_length": dataset_config.chunk_length,
                "stride": dataset_config.stride,
                "discount": dataset_config.discount,
            },
            "message": "Preparing frozen GR00T replay features",
        }
    )
    replay = _materialize(
        checkpoint=checkpoint,
        encoder_path=encoder_path,
        sources=sources,
        spec=spec,
        output=output,
        dataset_config=dataset_config,
        feature_batch_size=args.feature_batch_size,
        reference_seed=args.seed,
        device=args.device,
        total_steps=args.steps,
        started=started,
    )

    if args.initialization_mode == "new":
        stage2 = RLTStage2Run.new(
            encoder_path,
            spec=spec,
            groot_checkpoint=str(checkpoint),
            groot_checkpoint_fingerprint=checkpoint_fingerprint,
            representation_contract_fingerprint=representation_fingerprint,
            config=RLTStage2Config(),
            random_seed=args.seed,
            device=args.device,
            dtype=torch.float32,
        )
    else:
        stage2 = RLTStage2Run.resume(
            args.rlt_bundle,
            device=args.device,
            expected_groot_checkpoint_fingerprint=checkpoint_fingerprint,
            expected_replay_root=bundle / "replay_cache",
        )

    first_cumulative_step = stage2.learner.completed_critic_updates
    sampling_seed = args.seed + first_cumulative_step
    if sampling_seed >= 2**63 - 4:
        raise ValueError("RLT Stage 2 sampling seed exceeds the supported range")
    replay_root = output / "replay_cache"
    training_round = build_stage2_training_round(
        replay_root,
        expected_spec_fingerprint=replay.spec_fingerprint,
        reference_seed=args.seed,
        feature_batch_size=args.feature_batch_size,
        sampling_seed=sampling_seed,
        batch_size=args.batch_size,
        steps=args.steps,
        starting_critic_updates=first_cumulative_step,
    )
    _bind_training_round_with_lineage(
        stage2,
        training_round,
        replay_root=replay_root,
    )
    last_actor_loss: float | None = None
    last_critic_loss: float | None = None
    _progress(
        phase="training_actor_critic",
        completed_steps=0,
        total_steps=args.steps,
        started=started,
        actor_loss=None,
        critic_loss=None,
        average_reward=replay.average_reward,
        percentage=20.0,
        message="Training RLT Action MLP and twin Q critics",
    )
    with tempfile.TemporaryDirectory(prefix=f"cyclo-rlt-stage2-{args.job_id[:12]}-") as recovery:
        recovery_bundle = Path(recovery) / "latest"
        for local_step, indices in enumerate(
            _sample_indices(
                len(replay),
                args.batch_size,
                args.steps,
                seed=sampling_seed,
            ),
            start=1,
        ):
            batch = replay.batch(
                indices,
                device=stage2.learner.device,
                dtype=stage2.learner.dtype,
            )
            update = stage2.learner.update(batch)
            last_critic_loss = update.critic_loss
            if update.actor_loss is not None:
                last_actor_loss = update.actor_loss
            if local_step % args.save_freq == 0 and local_step != args.steps:
                stage2.save(recovery_bundle)
            if local_step % args.progress_interval == 0 or local_step == args.steps:
                _progress(
                    phase="training_actor_critic",
                    completed_steps=local_step,
                    total_steps=args.steps,
                    started=started,
                    actor_loss=last_actor_loss,
                    critic_loss=last_critic_loss,
                    average_reward=replay.average_reward,
                    percentage=20.0 + 80.0 * local_step / args.steps,
                    message=(
                        "Training RLT Action MLP and twin Q critics "
                        f"({update.completed_critic_updates} cumulative critic updates)"
                    ),
                )

        bundle = stage2.save(output)

    actor_path = bundle / "artifacts/rlt_actor.pt"
    bundled_encoder = bundle / "artifacts/rl_token_encoder.pt"
    training_state = bundle / "training_state/rlt_stage2.pt"
    manifest_path = bundle / "manifest.json"
    json_line(
        {
            "event": "stage2_training_result",
            "status": "completed",
            "phase": "completed",
            "job_id": args.job_id,
            "completed_steps": args.steps,
            "total_steps": args.steps,
            "completed_critic_updates": stage2.learner.completed_critic_updates,
            "completed_actor_updates": stage2.learner.completed_actor_updates,
            "percentage": 100.0,
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "average_reward": replay.average_reward,
            "actor_artifact": str(actor_path),
            "encoder_artifact": str(bundled_encoder),
            "checkpoint": str(training_state),
            "bundle_manifest": str(manifest_path),
            "output_dir": str(bundle),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialization-mode",
        choices=("new", "resume"),
        required=True,
    )
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--groot-checkpoint")
    parser.add_argument("--rl-token-encoder")
    parser.add_argument("--rlt-bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--steps", type=positive_int, required=True)
    parser.add_argument("--batch-size", type=positive_int, required=True)
    parser.add_argument("--save-freq", type=positive_int, required=True)
    parser.add_argument("--progress-interval", type=positive_int, default=10)
    parser.add_argument("--feature-batch-size", type=positive_int, default=2)
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
                "event": "result",
                "status": "stopped",
                "job_id": args.job_id,
                "message": "RLT Stage 2 stopped before bundle publication",
            }
        )
        return 130
    except Exception as error:
        json_line(
            {
                "event": "error",
                "status": "failed",
                "job_id": args.job_id,
                "message": f"RLT Stage 2 failed: {type(error).__name__}: {error}",
                "error": f"{type(error).__name__}: {error}",
            },
            stream=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
