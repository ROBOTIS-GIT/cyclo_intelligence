"""Build checkpoint-specific ACT TensorRT plans without robot connections."""
import argparse
import json
import logging
from pathlib import Path

import torch

from .loading import LoadingMixin
from .tensorrt_act import ADAPTER, DEFAULT_ENGINE, build_engine
from tensorrt_assets import validate_manifest


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--robot-type", default="")
    parser.add_argument("--task-instruction", default="")
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_path = LoadingMixin._resolve_model_dir(args.model_path)
    policy_type = json.loads((Path(model_path) / "config.json").read_text()).get("type", "act")
    if policy_type != "act":
        raise RuntimeError(f"TensorRT adapter unavailable for {policy_type!r}; do not reuse an ACT engine")
    engine_path = str(Path(args.engine_path) if args.engine_path else Path(model_path) / DEFAULT_ENGINE)
    if Path(engine_path).is_file() and not args.force:
        print(json.dumps(validate_manifest(model_path, engine_path, ADAPTER)))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("ACT TensorRT preparation requires CUDA")
    torch.set_num_threads(2)
    policy, _, _ = LoadingMixin._load_policy_assets(model_path, torch.device("cuda"))
    print(json.dumps(build_engine(policy, model_path, engine_path, args.workspace_mb)))


if __name__ == "__main__":
    main()
