#!/usr/bin/env python3
"""Build a GR00T DiT TensorRT engine outside the inference START path."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from runtime.inference_engine import GR00TInference, build_trt_engine


LOGGER = logging.getLogger("groot_trt_prepare")


def _default_engine_path(model_path: str) -> str:
    return os.path.join(model_path, "dit_model_bf16.trt")


def _default_workspace_mb() -> int:
    value = os.environ.get("GROOT_TRT_WORKSPACE_MB")
    if value:
        try:
            parsed = int(value)
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid GROOT_TRT_WORKSPACE_MB=%r; using 4096",
                value,
            )
        else:
            if parsed > 0:
                return parsed
    return 4096


def _manifest_path(engine_path: str) -> str:
    return f"{engine_path}.json"


def _write_manifest(engine_path: str, payload: dict) -> None:
    manifest_path = _manifest_path(engine_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, manifest_path)


def _base_manifest(args, engine_path: str, status: str, started_at: float) -> dict:
    now = time.time()
    return {
        "status": status,
        "model_path": args.model_path,
        "engine_path": engine_path,
        "robot_type": args.robot_type,
        "precision": "bf16",
        "workspace_mb": args.workspace_mb,
        "started_at": started_at,
        "updated_at": now,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--robot-type", required=True)
    parser.add_argument("--task-instruction", default="")
    parser.add_argument("--workspace-mb", type=int, default=_default_workspace_mb())
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    args.model_path = os.path.normpath(args.model_path)
    engine_path = os.path.normpath(args.engine_path or _default_engine_path(args.model_path))
    started_at = time.time()

    if os.path.exists(engine_path) and os.path.getsize(engine_path) > 0 and not args.force:
        manifest = _base_manifest(args, engine_path, "ready", started_at)
        manifest["finished_at"] = time.time()
        manifest["message"] = "TRT engine already exists"
        _write_manifest(engine_path, manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.force and os.path.exists(engine_path):
        os.remove(engine_path)

    manifest = _base_manifest(args, engine_path, "building", started_at)
    manifest["message"] = "Building TensorRT engine"
    _write_manifest(engine_path, manifest)

    inference = GR00TInference()
    try:
        LOGGER.info("Loading GR00T policy from: %s", args.model_path)
        inference._sync_hf_token_for_gated_backbones()
        inference.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.model_path,
            device="cuda",
        )
        inference.init_policy_info()

        LOGGER.info("Building model-schema synthetic observation")
        observation = inference.build_synthetic_observation(args.task_instruction)
        if observation.get("success") is False:
            raise RuntimeError(observation.get("message") or "observation failed")

        build_trt_engine(
            inference.policy,
            observation,
            engine_path,
            workspace_mb=args.workspace_mb,
        )
        if not os.path.exists(engine_path) or os.path.getsize(engine_path) <= 0:
            raise RuntimeError(
                f"TensorRT builder completed without writing engine: {engine_path}"
            )

        manifest = _base_manifest(args, engine_path, "ready", started_at)
        manifest["finished_at"] = time.time()
        manifest["message"] = "TensorRT engine ready"
        manifest["engine_size_bytes"] = os.path.getsize(engine_path)
        _write_manifest(engine_path, manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    except Exception as exc:
        manifest = _base_manifest(args, engine_path, "failed", started_at)
        manifest["finished_at"] = time.time()
        manifest["message"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        _write_manifest(engine_path, manifest)
        LOGGER.error("TensorRT engine build failed: %s", exc, exc_info=True)
        return 1
    finally:
        try:
            inference.cleanup()
        except Exception:
            LOGGER.debug("cleanup failed", exc_info=True)


if __name__ == "__main__":
    sys.exit(main())
