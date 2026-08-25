"""Run the real LeRobot MultiTaskDiT and Cyclo Flow-SDE PPO path on CUDA.

This is an integration check, not a quality benchmark.  It can use synthetic
observations or one read-only batch from a local LeRobot v3 dataset.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch

from cyclo_brain.algorithm.rl.flow_sde_ppo import (
    FlowSDEPPOConfig,
    clipped_value_loss,
    ppo_clipped_actor_loss,
    recompute_flow_sde_log_probs,
    sample_flow_sde_chunk,
)
from cyclo_brain.model.multi_task_dit import (
    CYCLO_SG2_ACTION_NAMES,
    CYCLO_SG2_CAMERA_KEYS,
    MultiTaskDiTFlowAdapter,
    MultiTaskDiTValueHead,
    canonicalize_dataset_stats,
    canonicalize_training_batch,
    with_default_task_instruction,
)
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
    make_multi_task_dit_pre_post_processors,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


BATCH_SIZE = 1
STATE_DIM = 22
ACTION_DIM = 22
HORIZON = 16
IMAGE_SHAPE = (3, 256, 256)
def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional local LeRobot v3 root; no dataset files are modified.",
    )
    parser.add_argument("--task-instruction", default="pick up the jelly bag")
    return parser.parse_args()


def _synthetic_batch(task_instruction: str) -> tuple[dict[str, object], None, dict[str, object]]:
    batch: dict[str, object] = {
        OBS_STATE: torch.randn(BATCH_SIZE, 1, STATE_DIM),
        ACTION: torch.randn(BATCH_SIZE, HORIZON, ACTION_DIM),
        "task": [task_instruction],
    }
    for key in CYCLO_SG2_CAMERA_KEYS:
        batch[key] = torch.rand(BATCH_SIZE, 1, *IMAGE_SHAPE)
    return batch, None, {"source": "synthetic", "fps": None, "episode_success": None}


def _dataset_batch(
    config: MultiTaskDiTConfig,
    dataset_root: Path,
    task_instruction: str,
) -> tuple[dict[str, object], dict, dict[str, object]]:
    from torch.utils.data import DataLoader
    from torchvision.transforms.v2 import Resize

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {dataset_root}")
    metadata = LeRobotDatasetMetadata("local/cyclo_showroom", root=dataset_root)
    if metadata.fps != 15:
        raise ValueError(f"Expected the current 15 Hz SG2 dataset, got {metadata.fps} Hz")
    action_names = tuple(metadata.features[ACTION].get("names") or ())
    state_names = tuple(metadata.features[OBS_STATE].get("names") or ())
    if action_names != CYCLO_SG2_ACTION_NAMES or state_names != CYCLO_SG2_ACTION_NAMES:
        raise ValueError("Dataset state/action order does not match the canonical 22D SG2 contract")
    if set(CYCLO_SG2_CAMERA_KEYS) - set(metadata.camera_keys):
        raise ValueError("Dataset does not contain all three canonical SG2 cameras")

    dataset = LeRobotDataset(
        "local/cyclo_showroom",
        root=dataset_root,
        episodes=[0],
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        image_transforms=Resize(IMAGE_SHAPE[-2:], antialias=True),
        video_backend="pyav",
        return_uint8=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)))
    batch = canonicalize_training_batch(
        batch,
        n_obs_steps=config.n_obs_steps,
        image_size=IMAGE_SHAPE[-2:],
        task_instruction=task_instruction,
    )
    outcome = batch.get("episode_success")
    success = bool(outcome.reshape(-1)[0]) if isinstance(outcome, torch.Tensor) else None
    return (
        batch,
        canonicalize_dataset_stats(dataset.meta.stats),
        {
            "source": str(dataset_root),
            "fps": metadata.fps,
            "episode_success": success,
            "dataset_frames_in_episode_0": len(dataset),
        },
    )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MultiTaskDiT CUDA validation requires an NVIDIA GPU")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch does not contain the detected architecture {architecture}")

    input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)
        for key in CYCLO_SG2_CAMERA_KEYS
    }
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    normalization_mapping = (
        {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
        if args.dataset_root is not None
        else {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    config = MultiTaskDiTConfig(
        input_features=input_features,
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        device="cuda",
        n_obs_steps=1,
        horizon=HORIZON,
        n_action_steps=HORIZON,
        objective="flow_matching",
        sigma_min=0.0,
        num_integration_steps=4,
        integration_method="euler",
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.0,
        image_crop_shape=(224, 224),
        image_crop_is_random=False,
        normalization_mapping=normalization_mapping,
    )
    if tuple(config.image_features) != CYCLO_SG2_CAMERA_KEYS:
        raise RuntimeError("MultiTaskDiT camera order changed while building the config")

    started = time.perf_counter()
    if args.dataset_root is None:
        raw_batch, dataset_stats, source_info = _synthetic_batch(args.task_instruction)
    else:
        raw_batch, dataset_stats, source_info = _dataset_batch(
            config,
            args.dataset_root,
            args.task_instruction,
        )
    preprocessor, _ = make_multi_task_dit_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
    )
    policy = MultiTaskDiTPolicy(config).to(device).eval()
    adapter = MultiTaskDiTFlowAdapter(policy)
    value_head = MultiTaskDiTValueHead(adapter.conditioning_dim).to(device)
    torch.cuda.reset_peak_memory_stats(device)

    batch = preprocessor(with_default_task_instruction(raw_batch))

    if batch[OBS_STATE].shape != (BATCH_SIZE, 1, STATE_DIM):
        raise RuntimeError("Preprocessed state shape does not match the Cyclo contract")
    for key in CYCLO_SG2_CAMERA_KEYS:
        if batch[key].shape != (BATCH_SIZE, 1, *IMAGE_SHAPE) or not batch[key].is_cuda:
            raise RuntimeError(f"Preprocessed camera {key!r} does not match the Cyclo contract")
    if batch[OBS_LANGUAGE_TOKENS].shape != (BATCH_SIZE, 77):
        raise RuntimeError("CLIP language token shape does not match the MultiTaskDiT contract")
    if batch[OBS_LANGUAGE_ATTENTION_MASK].shape != (BATCH_SIZE, 77):
        raise RuntimeError("CLIP language attention mask shape does not match the contract")

    policy.zero_grad(set_to_none=True)
    flow_matching_loss, _ = policy(batch)
    if flow_matching_loss.ndim != 0 or not bool(torch.isfinite(flow_matching_loss)):
        raise RuntimeError("MultiTaskDiT flow-matching loss is not finite")
    flow_matching_loss.backward()
    action_parameters = adapter.trainable_parameters()
    flow_matching_gradient_norm = _gradient_norm(action_parameters)
    if flow_matching_gradient_norm <= 0.0:
        raise RuntimeError("MultiTaskDiT action transformer received no training gradient")
    if any(parameter.grad is not None for parameter in policy.observation_encoder.parameters()):
        raise RuntimeError("Frozen MultiTaskDiT observation encoder received a gradient")

    conditioning = adapter.encode_conditioning(batch)
    expected_conditioning_dim = 3 * 768 + STATE_DIM + config.hidden_dim
    if conditioning.shape != (BATCH_SIZE, expected_conditioning_dim):
        raise RuntimeError(f"Unexpected MultiTaskDiT conditioning shape {tuple(conditioning.shape)}")

    with torch.inference_mode():
        ode_chunk = policy.objective.conditional_sample(
            policy.noise_predictor,
            BATCH_SIZE,
            conditioning,
        )
    if ode_chunk.shape != (BATCH_SIZE, HORIZON, ACTION_DIM):
        raise RuntimeError("MultiTaskDiT ODE inference returned the wrong action shape")

    flow_sde_config = FlowSDEPPOConfig(num_denoising_steps=4)
    action_mask = adapter.executed_action_mask(BATCH_SIZE, device=device)
    rollout = sample_flow_sde_chunk(
        adapter.velocity,
        conditioning,
        horizon=HORIZON,
        action_dim=ACTION_DIM,
        config=flow_sde_config,
        action_mask=action_mask,
        denoise_indices=torch.tensor([2], device=device, dtype=torch.long),
    )
    new_log_probs = recompute_flow_sde_log_probs(
        adapter.velocity,
        conditioning,
        rollout,
        config=flow_sde_config,
    )
    torch.testing.assert_close(new_log_probs, rollout.old_log_probs, rtol=1e-4, atol=1e-4)

    policy.zero_grad(set_to_none=True)
    value_head.zero_grad(set_to_none=True)
    actor_loss, actor_metrics = ppo_clipped_actor_loss(
        new_log_probs,
        rollout.old_log_probs,
        torch.ones(BATCH_SIZE, device=device),
        action_mask,
        clip_ratio_low=flow_sde_config.clip_ratio_low,
        clip_ratio_high=flow_sde_config.clip_ratio_high,
    )
    values = value_head(conditioning)
    value_loss = clipped_value_loss(
        values,
        values.detach(),
        torch.ones_like(values),
        value_clip=flow_sde_config.value_clip,
    )
    total_ppo_loss = actor_loss + flow_sde_config.value_loss_coefficient * value_loss
    total_ppo_loss.backward()
    ppo_actor_gradient_norm = _gradient_norm(action_parameters)
    ppo_value_gradient_norm = _gradient_norm(tuple(value_head.parameters()))
    if ppo_actor_gradient_norm <= 0.0 or ppo_value_gradient_norm <= 0.0:
        raise RuntimeError("Flow-SDE PPO actor or value head received no gradient")

    torch.cuda.synchronize(device)
    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(device),
        "architecture": architecture,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "diffusers": importlib.metadata.version("diffusers"),
        "input": "3 cameras + 22D state + language",
        **source_info,
        "conditioning_shape": list(conditioning.shape),
        "action_chunk_shape": list(ode_chunk.shape),
        "flow_sde_chain_shape": list(rollout.chains.shape),
        "flow_matching_loss": float(flow_matching_loss.detach()),
        "flow_matching_gradient_norm": flow_matching_gradient_norm,
        "ppo_actor_loss": float(actor_loss.detach()),
        "ppo_value_loss": float(value_loss.detach()),
        "ppo_actor_gradient_norm": ppo_actor_gradient_norm,
        "ppo_value_gradient_norm": ppo_value_gradient_norm,
        "ppo_ratio": float(actor_metrics["ratio"]),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
