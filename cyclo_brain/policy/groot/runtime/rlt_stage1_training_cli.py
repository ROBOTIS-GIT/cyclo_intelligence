#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""One-shot GR00T RL Token Training command used by the supervisor API.

The command intentionally has two memory phases:

1. Load a frozen GR00T policy and cache detached backbone tokens.
2. Unload GR00T, then train the small RL-token encoder/reconstruction decoder.

This prevents GR00T parameters from entering the optimizer and avoids keeping
the VLA and Stage-1 optimizer state on the GPU at the same time.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cyclo_brain.algorithm.rl.rlt import (
    RLTokenAutoencoder,
    RLTokenConfig,
    RLTokenStage1Trainer,
)

from .rlt_adapter import GR00TRLTTokenExtractor, _checkpoint_weight_fingerprint
from .rlt_stage1_dataset import RLTStage1LeRobotV21Source


_CACHE_FORMAT = "cyclo.groot.rlt.stage1_feature_cache/v1"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _json_line(payload: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=stream,
        flush=True,
    )


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _atomic_json_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _resolved_directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a real directory: {path}")
    return path


def _prepare_output(value: str | Path, inputs: Sequence[Path]) -> Path:
    output = Path(value).expanduser().absolute()
    if output.is_symlink():
        raise ValueError("RLT Stage 1 output must not be a symbolic link")
    for source in inputs:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("RLT Stage 1 output overlaps an input directory")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"RLT Stage 1 output is not empty: {output}")
    else:
        output.mkdir(parents=True)
    return output


def _progress(
    *,
    phase: str,
    completed: int,
    total: int,
    started: float,
    reconstruction_loss: float | None = None,
) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed / completed * (total - completed) if completed else None
    payload = {
        "event": "progress",
        "status": "running",
        "phase": phase,
        "percentage": 100.0 * completed / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "reconstruction_loss": reconstruction_loss,
    }
    if phase == "extracting":
        payload.update({"completed_samples": completed, "total_samples": total})
    else:
        payload.update(
            {
                "completed_steps": completed,
                "step": completed,
                "total_steps": total,
            }
        )
    _json_line(payload)


class _FeatureCacheWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.shards: list[dict[str, int | str]] = []
        self.sample_count = 0
        self.embedding_dim: int | None = None
        self.max_sequence_tokens = 0
        self.max_selected_tokens = 0

    def append(self, extracted: Mapping[str, Tensor]) -> None:
        tokens = extracted["tokens"]
        token_valid = extracted["token_valid"]
        image_token = extracted["image_token"]
        if tokens.requires_grad:
            raise RuntimeError("GR00T feature cache received attached tensors")
        if tokens.ndim != 3 or tuple(token_valid.shape) != tuple(tokens.shape[:2]):
            raise RuntimeError("GR00T feature cache tensor shapes disagree")
        if tuple(image_token.shape) != tuple(tokens.shape[:2]):
            raise RuntimeError("GR00T feature cache image mask shape disagrees")
        width = int(tokens.shape[2])
        if self.embedding_dim is None:
            self.embedding_dim = width
        elif self.embedding_dim != width:
            raise RuntimeError("GR00T embedding width changed during extraction")
        selected_counts = (token_valid & image_token).sum(dim=1)
        if not bool((selected_counts > 0).all()):
            raise RuntimeError("GR00T extraction produced a sample without image tokens")

        index = len(self.shards)
        path = self.root / f"shard_{index:06d}.pt"
        payload = {
            "format": _CACHE_FORMAT,
            "tokens": tokens.detach().to(device="cpu", dtype=torch.bfloat16),
            "token_valid": token_valid.detach().to(device="cpu", dtype=torch.bool),
            "image_token": image_token.detach().to(device="cpu", dtype=torch.bool),
        }
        _atomic_torch_save(path, payload)
        count = int(tokens.shape[0])
        self.shards.append(
            {
                "file": path.name,
                "start": self.sample_count,
                "count": count,
                "sequence_tokens": int(tokens.shape[1]),
            }
        )
        self.sample_count += count
        self.max_sequence_tokens = max(self.max_sequence_tokens, int(tokens.shape[1]))
        self.max_selected_tokens = max(
            self.max_selected_tokens,
            int(selected_counts.max().item()),
        )

    def finalize(self, extra: Mapping[str, Any]) -> Path:
        if self.sample_count < 1 or self.embedding_dim is None:
            raise RuntimeError("GR00T feature cache is empty")
        manifest = {
            "format": _CACHE_FORMAT,
            "sample_count": self.sample_count,
            "embedding_dim": self.embedding_dim,
            "max_sequence_tokens": self.max_sequence_tokens,
            "max_selected_tokens": self.max_selected_tokens,
            "shards": self.shards,
            **dict(extra),
        }
        path = self.root / "manifest.json"
        _atomic_json_save(path, manifest)
        return path


class _FeatureCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("RLT Stage 1 feature cache manifest is unreadable") from error
        if not isinstance(manifest, dict) or manifest.get("format") != _CACHE_FORMAT:
            raise ValueError("RLT Stage 1 feature cache format is invalid")
        self.manifest = manifest
        self.sample_count = int(manifest["sample_count"])
        self.embedding_dim = int(manifest["embedding_dim"])
        self.max_sequence_tokens = int(manifest["max_sequence_tokens"])
        self.max_selected_tokens = int(manifest["max_selected_tokens"])
        self.shards = tuple(manifest["shards"])
        if self.sample_count < 1 or not self.shards:
            raise ValueError("RLT Stage 1 feature cache is empty")
        self._lru: OrderedDict[int, Mapping[str, Tensor]] = OrderedDict()

    def _load_shard(self, shard_index: int) -> Mapping[str, Tensor]:
        cached = self._lru.pop(shard_index, None)
        if cached is None:
            record = self.shards[shard_index]
            path = self.root / str(record["file"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("format") != _CACHE_FORMAT:
                raise ValueError("RLT Stage 1 feature cache shard is invalid")
            cached = {
                "tokens": payload["tokens"],
                "token_valid": payload["token_valid"],
                "image_token": payload["image_token"],
            }
        self._lru[shard_index] = cached
        while len(self._lru) > 8:
            self._lru.popitem(last=False)
        return cached

    def _location(self, sample_index: int) -> tuple[int, int]:
        if not 0 <= sample_index < self.sample_count:
            raise IndexError(sample_index)
        for shard_index, record in enumerate(self.shards):
            start = int(record["start"])
            count = int(record["count"])
            if start <= sample_index < start + count:
                return shard_index, sample_index - start
        raise RuntimeError("RLT Stage 1 cache index is not covered by a shard")

    def batch(self, indices: Sequence[int], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        if not indices:
            raise ValueError("RLT Stage 1 cache batch is empty")
        rows: list[tuple[Tensor, Tensor, Tensor]] = []
        max_tokens = 0
        for sample_index in indices:
            shard_index, row_index = self._location(int(sample_index))
            shard = self._load_shard(shard_index)
            token = shard["tokens"][row_index]
            valid = shard["token_valid"][row_index]
            image = shard["image_token"][row_index]
            rows.append((token, valid, image))
            max_tokens = max(max_tokens, int(token.shape[0]))
        tokens = torch.zeros(
            (len(rows), max_tokens, self.embedding_dim), dtype=torch.bfloat16
        )
        token_valid = torch.zeros((len(rows), max_tokens), dtype=torch.bool)
        image_token = torch.zeros((len(rows), max_tokens), dtype=torch.bool)
        for row, (token, valid, image) in enumerate(rows):
            length = int(token.shape[0])
            tokens[row, :length] = token
            token_valid[row, :length] = valid
            image_token[row, :length] = image
        return (
            tokens.to(device=device),
            token_valid.to(device=device),
            image_token.to(device=device),
        )


def _sample_indices(
    sample_count: int,
    batch_size: int,
    steps: int,
    *,
    seed: int,
) -> Iterator[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pending: list[int] = []
    for _ in range(steps):
        while len(pending) < batch_size:
            pending.extend(torch.randperm(sample_count, generator=generator).tolist())
        yield pending[:batch_size]
        del pending[:batch_size]


def _extract_features(
    checkpoint: Path,
    dataset_roots: Sequence[Path],
    cache_root: Path,
    *,
    batch_size: int,
    device: str,
    weight_fingerprint: str,
) -> _FeatureCache:
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("GR00T RL Token Training requires a CUDA device")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    sources = tuple(RLTStage1LeRobotV21Source(root) for root in dataset_roots)
    total_samples = sum(len(source) for source in sources)
    writer = _FeatureCacheWriter(cache_root)
    started = time.monotonic()
    _progress(
        phase="extracting",
        completed=0,
        total=total_samples,
        started=started,
    )
    policy = None
    extractor = None
    completed = 0
    try:
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(checkpoint),
            device=device,
        )
        extractor = GR00TRLTTokenExtractor(policy)
        for source in sources:
            for observation in source.iter_batches(batch_size):
                extracted = extractor.extract(observation)
                writer.append(extracted)
                completed += int(extracted["tokens"].shape[0])
                _progress(
                    phase="extracting",
                    completed=completed,
                    total=total_samples,
                    started=started,
                )
        writer.finalize(
            {
                "dataset_roots": [str(root) for root in dataset_roots],
                "groot_checkpoint": str(checkpoint),
                "policy_weight_fingerprint": weight_fingerprint,
            }
        )
    finally:
        if extractor is not None:
            del extractor
        if policy is not None:
            del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _FeatureCache(cache_root)


def _representation_contract(
    cache: _FeatureCache,
    checkpoint: Path,
    weight_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "backend": "gr00t-n1.7",
        "embodiment": "new_embodiment",
        "policy_checkpoint": str(checkpoint),
        "policy_weight_fingerprint": weight_fingerprint,
        "embeddings": {
            "source": "Qwen3 final-layer backbone tokens before the action head",
            "width": cache.embedding_dim,
            "mask_true_means_valid": True,
            "image_token_mask_available": True,
            "token_selection": "image",
        },
    }


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = _resolved_directory(args.groot_checkpoint, "GR00T checkpoint")
    dataset_roots = tuple(
        _resolved_directory(path, "LeRobot dataset") for path in args.dataset_root
    )
    output = _prepare_output(args.output_dir, (*dataset_roots, checkpoint))
    weight_fingerprint = _checkpoint_weight_fingerprint(checkpoint)
    extraction_batch_size = min(args.batch_size, 4)
    cache = _extract_features(
        checkpoint,
        dataset_roots,
        output / "feature_cache",
        batch_size=extraction_batch_size,
        device=args.device,
        weight_fingerprint=weight_fingerprint,
    )

    model_config = RLTokenConfig(
        embedding_dim=cache.embedding_dim,
        max_tokens=cache.max_selected_tokens,
        num_heads=8,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dim=2048,
        dropout=0.0,
        token_selection="image",
        loss_reduction="paper_per_sample_sum",
    )
    contract = _representation_contract(cache, checkpoint, weight_fingerprint)
    trainer = RLTokenStage1Trainer(
        RLTokenAutoencoder(model_config),
        contract,
        device=args.device,
    )
    checkpoint_path = output / "training_state/rlt_stage1.pt"
    artifact_path = output / "artifacts/rl_token_encoder.pt"
    started = time.monotonic()
    _progress(phase="training", completed=0, total=args.steps, started=started)
    last_loss: float | None = None
    for indices in _sample_indices(
        cache.sample_count,
        args.batch_size,
        args.steps,
        seed=args.seed,
    ):
        tokens, token_valid, image_token = cache.batch(indices, trainer.device)
        metrics = trainer.train_step(tokens, token_valid, image_token)
        last_loss = metrics.reconstruction_loss
        if metrics.completed_steps % args.save_freq == 0:
            trainer.save_checkpoint(checkpoint_path)
        if (
            metrics.completed_steps % args.progress_interval == 0
            or metrics.completed_steps == args.steps
        ):
            _progress(
                phase="training",
                completed=metrics.completed_steps,
                total=args.steps,
                started=started,
                reconstruction_loss=metrics.reconstruction_loss,
            )

    trainer.save_checkpoint(checkpoint_path)
    trainer.export_encoder(artifact_path)
    artifact_payload = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    artifact_fingerprint = str(artifact_payload.get("artifact_fingerprint") or "")
    if len(artifact_fingerprint) != 64:
        raise RuntimeError("RLT Stage 1 encoder artifact fingerprint is invalid")
    run_manifest_path = checkpoint_path.with_name(f"{checkpoint_path.name}.run.json")
    _atomic_json_save(
        run_manifest_path,
        {
            "format": "cyclo.groot.rlt.stage1_run/v1",
            "status": "completed",
            "job_id": args.job_id,
            "dataset_roots": [str(path) for path in dataset_roots],
            "groot_checkpoint": str(checkpoint),
            "policy_weight_fingerprint": weight_fingerprint,
            "completed_steps": trainer.completed_steps,
            "batch_size": args.batch_size,
            "artifact": {
                "path": str(artifact_path),
                "artifact_fingerprint": artifact_fingerprint,
            },
            "checkpoint": {"path": str(checkpoint_path)},
        },
    )
    _json_line(
        {
            "event": "result",
            "status": "completed",
            "phase": "completed",
            "job_id": args.job_id,
            "completed_steps": trainer.completed_steps,
            "total_steps": args.steps,
            "percentage": 100.0,
            "reconstruction_loss": last_loss,
            "checkpoint_path": str(checkpoint_path),
            "encoder_artifact_path": str(artifact_path),
            "run_manifest_path": str(run_manifest_path),
            "output_dir": str(output),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-checkpoint", required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--steps", type=_positive_int, required=True)
    parser.add_argument("--batch-size", type=_positive_int, required=True)
    parser.add_argument("--save-freq", type=_positive_int, required=True)
    parser.add_argument("--progress-interval", type=_positive_int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        _json_line(
            {"event": "result", "status": "stopped", "job_id": args.job_id}
        )
        return 130
    except Exception as error:
        _json_line(
            {
                "event": "result",
                "status": "failed",
                "job_id": args.job_id,
                "error": f"{type(error).__name__}: {error}",
            },
            stream=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
