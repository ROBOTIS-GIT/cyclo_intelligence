"""Success-only Flow-Matching pretraining for Cyclo's SG2 MultiTaskDiT.

This module deliberately keeps LeRobot's policy, processors, optimizer preset,
and checkpoint layout intact.  The small Cyclo boundary is necessary because
the showroom v3 dataset stores singleton observation windows without an image
history axis and stores ``task`` as an opaque integer ID.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .flow_sde_adapter import CYCLO_SG2_CAMERA_KEYS, MultiTaskDiTFlowAdapter
from .checkpoint_validation import validate_checkpoint_round_trip
from .lerobot_batch import (
    CYCLO_SG2_ACTION_NAMES,
    canonicalize_dataset_stats,
    canonicalize_training_batch,
)
from .success_dataset import discover_episode_outcomes


DEFAULT_DATASET_ROOT = Path(
    "/workspace/lerobot/Task_20260814_090416_inference_MCAP_lerobot_v30_actionfix_atomic"
)
DEFAULT_OUTPUT_PARENT = Path("/workspace/checkpoint/multi_task_dit")
DEFAULT_TASK_INSTRUCTION = "pick up the jelly bag"
IMAGE_SHAPE = (3, 256, 256)
STATE_DIM = 22
ACTION_DIM = 22
HORIZON = 16


@dataclass(frozen=True)
class TrainingSummary:
    dataset_root: str
    output_dir: str
    checkpoint_dir: str
    task_instruction: str
    success_episodes: tuple[int, ...]
    failure_episodes: tuple[int, ...]
    success_frames: int
    steps: int
    batch_size: int
    overfit_one_batch: bool
    initial_fixed_loss: float
    final_fixed_loss: float
    minimum_train_loss: float
    final_train_loss: float
    mean_step_seconds: float
    mean_data_seconds: float
    peak_cuda_memory_bytes: int
    trainable_parameters: int
    frozen_parameters: int
    checkpoint_state_tensors: int
    checkpoint_velocity_max_abs_error: float
    device: str
    amp_dtype: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--task-instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Repeat one real batch to validate loss descent before a full run.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA bfloat16 autocast (intended only for diagnosis).",
    )
    return parser.parse_args()


def _validate_cli(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    if not isinstance(args.task_instruction, str) or not args.task_instruction.strip():
        raise ValueError("task-instruction must be non-empty")


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_PARENT / f"showroom_flow_matching_pretrain_{timestamp}"


def _build_policy_config(*, device: str):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)
        for key in CYCLO_SG2_CAMERA_KEYS
    }
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    return MultiTaskDiTConfig(
        input_features=input_features,
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        device=device,
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
        do_mask_loss_for_padding=True,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        push_to_hub=False,
    )


def _validate_dataset_contract(metadata: Any) -> None:
    from lerobot.utils.constants import ACTION, OBS_STATE

    if metadata.fps != 15:
        raise ValueError(f"Expected a 15 Hz SG2 dataset, got {metadata.fps} Hz")
    camera_keys = tuple(metadata.camera_keys)
    if set(camera_keys) != set(CYCLO_SG2_CAMERA_KEYS) or len(camera_keys) != 3:
        raise ValueError(
            "Dataset cameras must be exactly left_wrist, left_head, and right_wrist"
        )
    action_names = tuple(metadata.features[ACTION].get("names") or ())
    state_names = tuple(metadata.features[OBS_STATE].get("names") or ())
    if action_names != CYCLO_SG2_ACTION_NAMES:
        raise ValueError("Dataset action order does not match the showroom recorder's 22D contract")
    if state_names != CYCLO_SG2_ACTION_NAMES:
        raise ValueError("Dataset state order does not match the showroom recorder's 22D contract")


def _processor_stats(metadata_stats: Any) -> dict[str, dict[str, Any]]:
    from lerobot.utils.constants import IMAGENET_STATS

    stats = canonicalize_dataset_stats(metadata_stats)
    for camera_key in CYCLO_SG2_CAMERA_KEYS:
        for statistic, value in IMAGENET_STATS.items():
            stats[camera_key][statistic] = torch.as_tensor(value, dtype=torch.float32).clone()
    return stats


def _make_train_config(
    *,
    dataset_root: Path,
    output_dir: Path,
    success_episodes: tuple[int, ...],
    policy_config: Any,
    steps: int,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig

    resize_config = ImageTransformsConfig(
        enable=True,
        max_num_transforms=1,
        random_order=False,
        tfs={
            "resize": ImageTransformConfig(
                weight=1.0,
                type="Resize",
                kwargs={"size": [IMAGE_SHAPE[-2], IMAGE_SHAPE[-1]], "antialias": True},
            )
        },
    )
    config = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/cyclo_showroom",
            root=str(dataset_root),
            episodes=list(success_episodes),
            image_transforms=resize_config,
            use_imagenet_stats=True,
            video_backend="pyav",
            return_uint8=True,
        ),
        policy=policy_config,
        output_dir=output_dir,
        job_name="cyclo_showroom_multi_task_dit_flow_matching",
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        persistent_workers=num_workers > 0,
        steps=steps,
        eval_freq=0,
        log_freq=1,
        save_checkpoint=True,
        save_freq=steps,
        wandb=WandBConfig(enable=False),
    )
    # Populate the exact optimizer and scheduler presets in train_config.json.
    config.validate()
    return config


def _next_batch(loader: Any, iterator: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _deterministic_loss(
    policy: torch.nn.Module,
    batch: dict[str, Any],
    *,
    seed: int,
    use_amp: bool,
) -> float:
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss, _ = policy(batch)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError("MultiTaskDiT fixed-batch loss is not finite")
    return float(loss)


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _write_training_contract(
    pretrained_dir: Path,
    *,
    task_instruction: str,
    success_episodes: tuple[int, ...],
) -> None:
    contract = {
        "format_version": 1,
        "policy": "multi_task_dit",
        "objective": "flow_matching",
        "freeze_observation_encoder": True,
        "trainable_module": "noise_predictor",
        "task_instruction": task_instruction,
        "camera_order": list(CYCLO_SG2_CAMERA_KEYS),
        "state_action_names": list(CYCLO_SG2_ACTION_NAMES),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": HORIZON,
        "control_hz": 15,
        "success_episodes": list(success_episodes),
        "padding_loss_masked": True,
    }
    (pretrained_dir / "cyclo_training_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> TrainingSummary:
    from torch.utils.data import DataLoader
    from torchvision.transforms.v2 import Resize

    from lerobot.common.train_utils import (
        get_step_checkpoint_dir,
        save_checkpoint,
        update_last_checkpoint,
    )
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
        make_multi_task_dit_pre_post_processors,
    )

    _validate_cli(args)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {dataset_root}")
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing training run: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("MultiTaskDiT pretraining requires an NVIDIA CUDA device")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    use_amp = not args.no_amp

    metadata = LeRobotDatasetMetadata("local/cyclo_showroom", root=dataset_root)
    _validate_dataset_contract(metadata)
    outcomes = discover_episode_outcomes(dataset_root)
    if not outcomes.success_episodes:
        raise ValueError("The dataset contains no successful episodes for imitation pretraining")

    policy_config = _build_policy_config(device="cuda")
    train_config = _make_train_config(
        dataset_root=dataset_root,
        output_dir=output_dir,
        success_episodes=outcomes.success_episodes,
        policy_config=policy_config,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dataset = LeRobotDataset(
        "local/cyclo_showroom",
        root=dataset_root,
        episodes=list(outcomes.success_episodes),
        delta_timestamps=resolve_delta_timestamps(policy_config, metadata),
        image_transforms=Resize(IMAGE_SHAPE[-2:], antialias=True),
        video_backend="pyav",
        return_uint8=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    iterator = iter(loader)

    stats = _processor_stats(dataset.meta.stats)
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(
        policy_config,
        dataset_stats=stats,
    )
    policy = MultiTaskDiTPolicy(policy_config).to(device).eval()
    adapter = MultiTaskDiTFlowAdapter(policy, freeze_observation_encoder=True)
    trainable_parameters = adapter.trainable_parameters()
    if not trainable_parameters:
        raise RuntimeError("MultiTaskDiT action transformer has no trainable parameters")
    optimizer = train_config.optimizer.build(trainable_parameters)
    scheduler = train_config.scheduler.build(optimizer, args.steps)

    raw_fixed_batch, iterator = _next_batch(loader, iterator)
    canonical_fixed_batch = canonicalize_training_batch(
        raw_fixed_batch,
        n_obs_steps=policy_config.n_obs_steps,
        image_size=IMAGE_SHAPE[-2:],
        task_instruction=args.task_instruction,
    )
    fixed_batch = preprocessor(canonical_fixed_batch)
    initial_fixed_loss = _deterministic_loss(
        policy,
        fixed_batch,
        seed=args.seed + 1,
        use_amp=use_amp,
    )

    step_times: list[float] = []
    data_times: list[float] = []
    train_losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        data_started = time.perf_counter()
        if args.overfit_one_batch:
            batch = fixed_batch
        else:
            raw_batch, iterator = _next_batch(loader, iterator)
            batch = preprocessor(
                canonicalize_training_batch(
                    raw_batch,
                    n_obs_steps=policy_config.n_obs_steps,
                    image_size=IMAGE_SHAPE[-2:],
                    task_instruction=args.task_instruction,
                )
            )
        data_times.append(time.perf_counter() - data_started)

        policy.observation_encoder.eval()
        policy.noise_predictor.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss, _ = policy(batch)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite MultiTaskDiT loss at step {step}")
        loss.backward()
        gradient_norm = _gradient_norm(trainable_parameters)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"Invalid action-transformer gradient at step {step}: {gradient_norm}")
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            train_config.optimizer.grad_clip_norm,
            error_if_nonfinite=True,
        )
        if any(parameter.grad is not None for parameter in policy.observation_encoder.parameters()):
            raise RuntimeError("Frozen observation encoder received a training gradient")
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        train_losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "steps": args.steps,
                        "loss": train_losses[-1],
                        "gradient_norm": gradient_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "step_seconds": step_times[-1],
                        "data_seconds": data_times[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    policy.eval()
    adapter = MultiTaskDiTFlowAdapter(policy, freeze_observation_encoder=True)
    final_fixed_loss = _deterministic_loss(
        policy,
        fixed_batch,
        seed=args.seed + 1,
        use_amp=use_amp,
    )

    checkpoint_dir = get_step_checkpoint_dir(output_dir, args.steps, args.steps)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=args.steps,
        cfg=train_config,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    update_last_checkpoint(checkpoint_dir)
    _write_training_contract(
        checkpoint_dir / "pretrained_model",
        task_instruction=args.task_instruction.strip(),
        success_episodes=outcomes.success_episodes,
    )
    checkpoint_validation = validate_checkpoint_round_trip(
        policy,
        checkpoint_dir,
        preprocessor=preprocessor,
        raw_batch=canonical_fixed_batch,
        postprocessor=postprocessor,
        normalized_action=torch.zeros(
            (args.batch_size, HORIZON, ACTION_DIM),
            dtype=torch.float32,
            device=device,
        ),
        seed=args.seed + 2,
    )
    (output_dir / "checkpoint_validation.json").write_text(
        json.dumps(checkpoint_validation.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    frozen_count = sum(
        parameter.numel() for parameter in policy.parameters() if not parameter.requires_grad
    )
    summary = TrainingSummary(
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        task_instruction=args.task_instruction.strip(),
        success_episodes=outcomes.success_episodes,
        failure_episodes=outcomes.failure_episodes,
        success_frames=outcomes.success_frames,
        steps=args.steps,
        batch_size=args.batch_size,
        overfit_one_batch=args.overfit_one_batch,
        initial_fixed_loss=initial_fixed_loss,
        final_fixed_loss=final_fixed_loss,
        minimum_train_loss=min(train_losses),
        final_train_loss=train_losses[-1],
        mean_step_seconds=sum(step_times) / len(step_times),
        mean_data_seconds=sum(data_times) / len(data_times),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
        trainable_parameters=trainable_count,
        frozen_parameters=frozen_count,
        checkpoint_state_tensors=checkpoint_validation.state_tensor_count,
        checkpoint_velocity_max_abs_error=checkpoint_validation.velocity_max_abs_error,
        device=torch.cuda.get_device_name(device),
        amp_dtype="bfloat16" if use_amp else "float32",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", **asdict(summary)}, sort_keys=True), flush=True)
    return summary


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
