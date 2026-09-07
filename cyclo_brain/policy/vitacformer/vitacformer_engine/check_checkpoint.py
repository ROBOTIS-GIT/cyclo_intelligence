"""Load weights and run one synthetic observation without robot subscriptions.

Usage inside the policy container:
    python3 -m vitacformer_engine.check_checkpoint /workspace/model/vitacformer/run
"""

import argparse
import json

import torch

from .constants import IMAGE_KEY, STATE_KEY, TACTILE_BATCH_KEY
from .model import load_vitacformer_policy
from .prediction import PredictionMixin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    policy = load_vitacformer_policy(args.model_path, device)
    batch = {
        IMAGE_KEY: torch.zeros(1, 3, 188, 336, device=device),
        STATE_KEY: policy._state_mean.expand(1, 6, -1).clone(),
        TACTILE_BATCH_KEY: torch.zeros(1, 18, 180, device=device),
    }
    runner = PredictionMixin()
    runner._policy = policy
    chunk = runner._predict_chunk(batch)
    print(json.dumps({
        "checkpoint": policy.config.checkpoint_path,
        "device": str(device),
        "shape": list(chunk.shape),
        "finite": True,
        "robot_subscriptions": False,
    }))


if __name__ == "__main__":
    main()
