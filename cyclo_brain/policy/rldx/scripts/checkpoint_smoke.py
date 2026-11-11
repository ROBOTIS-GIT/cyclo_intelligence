"""Run real RLDX weights on recorded data without ROS or robot commands."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy


def smoke(checkpoint, dataset):
    metadata = json.loads((checkpoint / "cyclo_input_metadata.json").read_text())
    policy = RLDXPolicy(model_path=str(checkpoint), device="cuda", strict=True,
                        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT)
    mc = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset, mc)
    episode = loader[0]
    observation = {
        "video": {key: np.asarray(episode[f"video.{key}"].iloc[0])[None, None]
                  for key in mc["video"].modality_keys},
        "state": {key: np.asarray(episode[f"state.{key}"].iloc[0], dtype=np.float32)[None, None]
                  for key in mc["state"].modality_keys},
        "language": {key: [[episode[f"language.{key}"].iloc[0]]]
                     for key in mc["language"].modality_keys},
    }
    report = {"checkpoint": str(checkpoint), "cameras": {k: list(v.shape) for k, v in observation["video"].items()},
              "instruction": observation["language"], "calls": []}
    for index in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions, _ = policy.get_action(observation, options={"session_ids": ["smoke"], "reset_memory": [index == 0]})
        torch.cuda.synchronize()
        action = actions["joint_position"]
        if action.shape != (1, 16, len(metadata["action_names"])) or not np.isfinite(action).all():
            raise ValueError(f"Invalid action: {action.shape}")
        report["calls"].append({"seconds": time.perf_counter() - start, "shape": list(action.shape)})
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    policy.reset()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    smoke(args.checkpoint, args.dataset)
