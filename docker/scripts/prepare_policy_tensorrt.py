#!/usr/bin/env python3
"""Prepare any registered ACT/GR00T checkpoint using its production container.

No policy backend or robot command service is started. Engines are generated
for the selected checkpoint, local CUDA device and installed TensorRT version.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Cyclo /workspace/model/... path or its host folder")
    parser.add_argument("--robot-type", default="ffw_sg2_rev1")
    parser.add_argument("--task-instruction", default="Place the paper towel roll on the holder using the right arm.")
    parser.add_argument("--workspace-mb", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    host_root = repo / "docker/workspace/model"
    path = Path(args.model_path)
    if path.is_relative_to("/workspace/model"):
        relative = path.relative_to("/workspace/model")
    elif path.resolve().is_relative_to(host_root.resolve()):
        relative = path.resolve().relative_to(host_root.resolve())
    else:
        parser.error("Register the checkpoint under Cyclo's /workspace/model/{lerobot,groot} first")
    if not relative.parts or relative.parts[0] not in {"lerobot", "groot"}:
        parser.error("Only registered ACT/GR00T checkpoints are supported")
    backend = relative.parts[0]
    if not (host_root / relative / "config.json").is_file():
        parser.error("The selected checkpoint has no config.json")
    module = "runtime.prepare_trt_engine" if backend == "groot" else "lerobot_engine.prepare_trt_engine"
    env = os.environ.copy()
    env.setdefault("DISPLAY", "")
    cmd = ["docker", "compose", "-f", str(repo / "docker/docker-compose.yml"),
           "run", "--rm", "-T", "--no-deps", "--entrypoint", "python3",
           "-e", "PYTHONPATH=/app:/gr00t:/policy_runtime:/zenoh_sdk:/robot_client_sdk:/action_chunk_processing_sdk",
           "-e", "NO_ALBUMENTATIONS_UPDATE=1", backend, "-m", module,
           "--model-path", str(Path("/workspace/model") / relative),
           "--robot-type", args.robot_type, "--task-instruction", args.task_instruction]
    if args.workspace_mb is not None:
        cmd += ["--workspace-mb", str(args.workspace_mb)]
    if args.force:
        cmd.append("--force")
    raise SystemExit(subprocess.call(cmd, cwd=repo, env=env))


if __name__ == "__main__":
    main()
