"""Stoppable MultiTaskDiT flow-matching imitation-learning loop.

The pinned LeRobot implementation still owns the policy, objective,
pre/post-processors, optimizer preset, and checkpoint serialization.  Cyclo's
boundary selects immutable v3 demonstration episodes, makes the SG2 camera
shapes/order explicit, supplies the language instruction, and emits the same
JSON progress/result contract used by the supervisor's ACT trainer.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cyclo_brain.algorithm.il.act_bc.dataset import (
    LeRobotDatasetDependencies,
    RootSelection,
    VirtualACTBCDataset,
    load_virtual_act_bc_dataset,
)
from cyclo_brain.model.multi_task_dit.flow_sde_adapter import CYCLO_SG2_CAMERA_KEYS
from cyclo_brain.model.multi_task_dit.lerobot_batch import (
    CYCLO_SG2_ACTION_NAMES,
    canonicalize_dataset_stats,
    canonicalize_training_batch,
)


MULTI_TASK_DIT_HORIZON = 16
IMAGE_SIZE = (256, 256)
IMAGE_SHAPE = (3, *IMAGE_SIZE)
STATE_DIM = 22
ACTION_DIM = 22
CONTROL_HZ = 15
DEFAULT_TASK_INSTRUCTION = "pick up the jelly bag"
_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


@dataclass(frozen=True)
class OfficialTrainingDependencies:
    """Pinned LeRobot APIs isolated behind an injectable test boundary."""

    dataset: LeRobotDatasetDependencies
    policy_config_cls: type
    policy_feature_cls: type
    feature_type: Any
    normalization_mode: Any
    dataset_config_cls: type
    train_config_cls: type
    policy_cls: type
    make_pre_post_processors: Callable[..., tuple[Any, Any]]
    get_step_checkpoint_dir: Callable[[Path, int, int], Path]
    save_checkpoint: Callable[..., None]
    update_last_checkpoint: Callable[[Path], Any]
    cycle: Callable[[Any], Any]
    resize_factory: Callable[[tuple[int, int]], Any]
    imagenet_stats: Mapping[str, Any]


def load_official_training_dependencies() -> OfficialTrainingDependencies:
    """Import GPU/LeRobot dependencies only inside the training process."""

    from torchvision.transforms.v2 import Resize

    from lerobot.common.train_utils import (
        get_step_checkpoint_dir,
        save_checkpoint,
        update_last_checkpoint,
    )
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import _load_episode_with_stats
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
        make_multi_task_dit_pre_post_processors,
    )
    from lerobot.utils.constants import IMAGENET_STATS
    from lerobot.utils.utils import cycle

    return OfficialTrainingDependencies(
        dataset=LeRobotDatasetDependencies(
            metadata_cls=LeRobotDatasetMetadata,
            dataset_cls=LeRobotDataset,
            resolve_delta_timestamps=resolve_delta_timestamps,
            aggregate_stats=aggregate_stats,
            load_episode_with_stats=_load_episode_with_stats,
        ),
        policy_config_cls=MultiTaskDiTConfig,
        policy_feature_cls=PolicyFeature,
        feature_type=FeatureType,
        normalization_mode=NormalizationMode,
        dataset_config_cls=DatasetConfig,
        train_config_cls=TrainPipelineConfig,
        policy_cls=MultiTaskDiTPolicy,
        make_pre_post_processors=make_multi_task_dit_pre_post_processors,
        get_step_checkpoint_dir=get_step_checkpoint_dir,
        save_checkpoint=save_checkpoint,
        update_last_checkpoint=update_last_checkpoint,
        cycle=cycle,
        resize_factory=lambda size: Resize(size, antialias=True),
        imagenet_stats=IMAGENET_STATS,
    )


@dataclass(frozen=True)
class MultiTaskDiTILConfig:
    """Immutable configuration for one from-scratch SG2 DiT IL run."""

    selections: tuple[RootSelection, ...]
    output_dir: Path
    steps: int
    batch_size: int
    save_freq: int
    progress_interval: int = 10
    chunk_size: int = MULTI_TASK_DIT_HORIZON
    task_instruction: str = DEFAULT_TASK_INSTRUCTION
    learning_rate: float = 2e-5
    num_workers: int = 4
    seed: int = 1000
    device: str = "cuda"
    video_backend: str = "pyav"
    grad_clip_norm: float = 10.0
    use_amp: bool = True

    def __post_init__(self) -> None:
        selections = tuple(self.selections)
        if not selections:
            raise ValueError("MultiTaskDiT imitation learning requires at least one dataset root")
        roots = tuple(selection.root for selection in selections)
        if len(roots) != len(set(roots)):
            raise ValueError("MultiTaskDiT imitation-learning roots cannot contain duplicates")
        output_dir = Path(self.output_dir).expanduser().resolve()
        for root in roots:
            if (
                output_dir == root
                or output_dir.is_relative_to(root)
                or root.is_relative_to(output_dir)
            ):
                raise ValueError(
                    "MultiTaskDiT output directory must not contain or be contained "
                    "by a dataset root"
                )
        for name, value in (
            ("steps", self.steps),
            ("batch_size", self.batch_size),
            ("save_freq", self.save_freq),
            ("progress_interval", self.progress_interval),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_size != MULTI_TASK_DIT_HORIZON:
            raise ValueError(
                "Cyclo MultiTaskDiT imitation learning requires "
                f"chunk_size={MULTI_TASK_DIT_HORIZON}"
            )
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(self.grad_clip_norm)) or self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be finite and positive")
        task_instruction = str(self.task_instruction).strip()
        if not task_instruction:
            raise ValueError("task_instruction must be non-empty")
        device = str(self.device)
        if device != "cpu" and device != "cuda" and not (
            device.startswith("cuda:") and device[len("cuda:") :].isdigit()
        ):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "task_instruction", task_instruction)
        object.__setattr__(self, "device", device)


@dataclass(frozen=True)
class MultiTaskDiTILProgress:
    status: str
    step: int
    total_steps: int
    percentage: float
    loss: float | None
    l1_loss: None
    kld_loss: None
    flow_matching_loss: float | None
    elapsed_seconds: float
    eta_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"event": "progress", **self.__dict__}


@dataclass(frozen=True)
class MultiTaskDiTILResult:
    status: str
    step: int
    total_steps: int
    percentage: float
    loss: float | None
    l1_loss: None
    kld_loss: None
    flow_matching_loss: float | None
    elapsed_seconds: float
    model_path: str | None
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"event": "result", **self.__dict__}


def _atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_failed_result(output_dir: Path, error: BaseException) -> dict[str, Any]:
    value = {
        "event": "result",
        "status": "failed",
        "error_type": type(error).__name__,
        "message": str(error),
        "model_path": None,
        "checkpoint_path": None,
    }
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and output_dir.is_dir():
        _atomic_json_save(output_dir / "result.json", value)
    return value


def training_manifest(config: MultiTaskDiTILConfig) -> dict[str, Any]:
    return {
        "event": "manifest",
        "schema_version": 1,
        "method": "imitation_learning",
        "policy_type": "multi_task_dit",
        "implementation": "lerobot-0.5.2-multi-task-dit-flow-matching",
        "dataset_roots": [str(selection.root) for selection in config.selections],
        "episode_indices": [
            list(selection.success_episodes) for selection in config.selections
        ],
        "output_dir": str(config.output_dir),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "save_freq": config.save_freq,
        "progress_interval": config.progress_interval,
        "chunk_size": config.chunk_size,
        "horizon": config.chunk_size,
        "n_action_steps": config.chunk_size,
        "task_instruction": config.task_instruction,
        "learning_rate": config.learning_rate,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "device": config.device,
        "video_backend": config.video_backend,
        "objective": "flow_matching",
        "freeze_observation_encoder": True,
        "loss": "padding-masked flow-matching mean squared error",
    }


def _prepare_output(config: MultiTaskDiTILConfig) -> None:
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {config.output_dir}")
        if any(config.output_dir.iterdir()):
            raise FileExistsError(
                f"MultiTaskDiT output directory is not empty: {config.output_dir}"
            )
    else:
        config.output_dir.mkdir(parents=True)
    _atomic_json_save(config.output_dir / "manifest.json", training_manifest(config))


def _build_policy_config(
    config: MultiTaskDiTILConfig,
    dependencies: OfficialTrainingDependencies,
) -> Any:
    input_features = {
        key: dependencies.policy_feature_cls(
            type=dependencies.feature_type.VISUAL,
            shape=IMAGE_SHAPE,
        )
        for key in CYCLO_SG2_CAMERA_KEYS
    }
    input_features["observation.state"] = dependencies.policy_feature_cls(
        type=dependencies.feature_type.STATE,
        shape=(STATE_DIM,),
    )
    return dependencies.policy_config_cls(
        input_features=input_features,
        output_features={
            "action": dependencies.policy_feature_cls(
                type=dependencies.feature_type.ACTION,
                shape=(ACTION_DIM,),
            )
        },
        device=config.device,
        n_obs_steps=1,
        horizon=config.chunk_size,
        n_action_steps=config.chunk_size,
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
        optimizer_lr=config.learning_rate,
        normalization_mapping={
            "VISUAL": dependencies.normalization_mode.MEAN_STD,
            "STATE": dependencies.normalization_mode.MIN_MAX,
            "ACTION": dependencies.normalization_mode.MIN_MAX,
        },
        push_to_hub=False,
    )


def _build_train_config(
    config: MultiTaskDiTILConfig,
    policy_config: Any,
    dependencies: OfficialTrainingDependencies,
) -> Any:
    dataset_config = dependencies.dataset_config_cls(
        repo_id=f"cyclo-local/{config.selections[0].root.name}",
        root=str(config.selections[0].root),
        episodes=list(config.selections[0].success_episodes),
        video_backend=config.video_backend,
        return_uint8=True,
    )
    train_config = dependencies.train_config_cls(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=config.output_dir,
        job_name="cyclo_multi_task_dit_imitation_learning",
        seed=config.seed,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        persistent_workers=config.num_workers > 0,
        steps=config.steps,
        eval_freq=0,
        log_freq=config.progress_interval,
        save_checkpoint=True,
        save_freq=config.save_freq,
    )
    optimizer_config = policy_config.get_optimizer_preset()
    optimizer_config.lr = config.learning_rate
    optimizer_config.grad_clip_norm = config.grad_clip_norm
    train_config.optimizer = optimizer_config
    train_config.scheduler = policy_config.get_scheduler_preset()
    return train_config


def _validate_dataset_contract(metadata: Any) -> None:
    if float(getattr(metadata, "fps", float("nan"))) != CONTROL_HZ:
        raise ValueError(f"Expected a {CONTROL_HZ} Hz SG2 dataset")
    camera_keys = tuple(getattr(metadata, "camera_keys", ()))
    if set(camera_keys) != set(CYCLO_SG2_CAMERA_KEYS) or len(camera_keys) != 3:
        raise ValueError(
            "Dataset cameras must be exactly left_wrist, left_head, and right_wrist"
        )
    features = getattr(metadata, "features", {})
    action_names = tuple(features.get("action", {}).get("names") or ())
    state_names = tuple(features.get("observation.state", {}).get("names") or ())
    if action_names != CYCLO_SG2_ACTION_NAMES:
        raise ValueError("Dataset action order does not match the SG2 22D contract")
    if state_names != CYCLO_SG2_ACTION_NAMES:
        raise ValueError("Dataset state order does not match the SG2 22D contract")


def _processor_stats(
    metadata_stats: Mapping[str, Mapping[str, Any]],
    imagenet_stats: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    stats = canonicalize_dataset_stats(metadata_stats)
    for camera_key in CYCLO_SG2_CAMERA_KEYS:
        for statistic, value in imagenet_stats.items():
            stats[camera_key][statistic] = torch.as_tensor(
                value,
                dtype=torch.float32,
            ).clone()
    return stats


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    return generator


def _build_dataloader(
    config: MultiTaskDiTILConfig,
    dataset: VirtualACTBCDataset,
    generator: torch.Generator,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        drop_last=False,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
        generator=generator,
    )


def _finite_metric(value: Any, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} became non-finite")
    return result


def _make_progress(
    *,
    status: str,
    step: int,
    total_steps: int,
    started_at: float,
    clock: Callable[[], float],
    loss: float | None,
) -> MultiTaskDiTILProgress:
    elapsed = max(0.0, float(clock() - started_at))
    percentage = 100.0 * step / total_steps
    eta = None if step == 0 else max(0.0, elapsed * (total_steps - step) / step)
    if status != "running":
        eta = 0.0 if status == "complete" else None
    return MultiTaskDiTILProgress(
        status=status,
        step=step,
        total_steps=total_steps,
        percentage=percentage,
        loss=loss,
        l1_loss=None,
        kld_loss=None,
        flow_matching_loss=loss,
        elapsed_seconds=elapsed,
        eta_seconds=eta,
    )


def _emit_progress(
    config: MultiTaskDiTILConfig,
    progress: MultiTaskDiTILProgress,
    callback: Callable[[MultiTaskDiTILProgress], None] | None,
) -> None:
    _atomic_json_save(config.output_dir / "progress.json", progress.to_dict())
    if callback is not None:
        callback(progress)


def _write_training_contract(
    model_dir: Path,
    config: MultiTaskDiTILConfig,
) -> None:
    contract = {
        "format_version": 1,
        "policy": "multi_task_dit",
        "method": "imitation_learning",
        "objective": "flow_matching",
        "freeze_observation_encoder": True,
        "trainable_module": "noise_predictor",
        "task_instruction": config.task_instruction,
        "camera_order": list(CYCLO_SG2_CAMERA_KEYS),
        "state_action_names": list(CYCLO_SG2_ACTION_NAMES),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": config.chunk_size,
        "control_hz": CONTROL_HZ,
        "dataset_roots": [str(selection.root) for selection in config.selections],
        "episode_indices": [
            list(selection.success_episodes) for selection in config.selections
        ],
        "padding_loss_masked": True,
    }
    _atomic_json_save(model_dir / "cyclo_training_contract.json", contract)


def _validate_checkpoint(checkpoint_dir: Path) -> tuple[Path, Path]:
    model_dir = checkpoint_dir / "pretrained_model"
    training_state = checkpoint_dir / "training_state"
    missing = [name for name in _MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"MultiTaskDiT checkpoint {checkpoint_dir} is missing model files: "
            + ", ".join(missing)
        )
    if not training_state.is_dir() or not (training_state / "training_step.json").is_file():
        raise RuntimeError(f"MultiTaskDiT checkpoint {checkpoint_dir} has no valid training_state")
    return model_dir.resolve(), training_state.resolve()


def _save_training_checkpoint(
    *,
    dependencies: OfficialTrainingDependencies,
    config: MultiTaskDiTILConfig,
    train_config: Any,
    policy: Any,
    optimizer: Any,
    scheduler: Any,
    preprocessor: Any,
    postprocessor: Any,
    step: int,
) -> tuple[Path, Path]:
    checkpoint_dir = dependencies.get_step_checkpoint_dir(
        config.output_dir,
        config.steps,
        step,
    )
    dependencies.save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        cfg=train_config,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    dependencies.update_last_checkpoint(checkpoint_dir)
    model_dir = checkpoint_dir / "pretrained_model"
    _write_training_contract(model_dir, config)
    return _validate_checkpoint(checkpoint_dir)


def run_training(
    config: MultiTaskDiTILConfig,
    *,
    dependencies: OfficialTrainingDependencies | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_callback: Callable[[MultiTaskDiTILProgress], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> MultiTaskDiTILResult:
    """Train a fresh flow-matching DiT on the selected demonstrations."""

    dependencies = dependencies or load_official_training_dependencies()
    should_stop = should_stop or (lambda: False)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MultiTaskDiT imitation learning requires an available CUDA device")

    policy_config = _build_policy_config(config, dependencies)
    train_config = _build_train_config(config, policy_config, dependencies)
    _prepare_output(config)
    started_at = clock()
    generator = _seed_everything(config.seed)

    dataset = load_virtual_act_bc_dataset(
        config.selections,
        policy_config=policy_config,
        dependencies=dependencies.dataset,
        video_backend=config.video_backend,
        image_transforms=dependencies.resize_factory(IMAGE_SIZE),
    )
    _validate_dataset_contract(dataset.meta)
    stats = _processor_stats(dataset.meta.stats, dependencies.imagenet_stats)
    preprocessor, postprocessor = dependencies.make_pre_post_processors(
        policy_config,
        dataset_stats=stats,
    )
    policy = dependencies.policy_cls(policy_config).to(config.device)
    observation_encoder = getattr(policy, "observation_encoder", None)
    noise_predictor = getattr(policy, "noise_predictor", None)
    if not isinstance(observation_encoder, torch.nn.Module):
        raise TypeError("MultiTaskDiT policy has no observation_encoder module")
    if not isinstance(noise_predictor, torch.nn.Module):
        raise TypeError("MultiTaskDiT policy has no noise_predictor module")
    observation_encoder.requires_grad_(False)
    observation_encoder.eval()
    trainable_parameters = tuple(
        parameter for parameter in noise_predictor.parameters() if parameter.requires_grad
    )
    if not trainable_parameters:
        raise RuntimeError("MultiTaskDiT noise predictor has no trainable parameters")
    optimizer = train_config.optimizer.build(trainable_parameters)
    scheduler = (
        train_config.scheduler.build(optimizer, config.steps)
        if train_config.scheduler is not None
        else None
    )
    dataloader = _build_dataloader(config, dataset, generator)
    batches = dependencies.cycle(dataloader)

    loss_value: float | None = None
    _emit_progress(
        config,
        _make_progress(
            status="running",
            step=0,
            total_steps=config.steps,
            started_at=started_at,
            clock=clock,
            loss=loss_value,
        ),
        progress_callback,
    )
    last_checkpoint_step = 0
    last_model_path: Path | None = None
    last_training_state: Path | None = None
    step = 0
    amp_enabled = config.use_amp and config.device.startswith("cuda")

    while step < config.steps:
        if should_stop():
            break
        raw_batch = next(batches)
        if should_stop():
            break
        batch = preprocessor(
            canonicalize_training_batch(
                raw_batch,
                n_obs_steps=policy_config.n_obs_steps,
                image_size=IMAGE_SIZE,
                task_instruction=config.task_instruction,
            )
        )

        observation_encoder.eval()
        noise_predictor.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            loss, _ = policy.forward(batch)
        loss_value = _finite_metric(loss, name="loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            config.grad_clip_norm,
            error_if_nonfinite=True,
        )
        if _finite_metric(gradient_norm, name="gradient_norm") <= 0.0:
            raise RuntimeError("MultiTaskDiT noise predictor received no training gradient")
        if any(parameter.grad is not None for parameter in observation_encoder.parameters()):
            raise RuntimeError("Frozen MultiTaskDiT observation encoder received a gradient")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1

        saving_step = step % config.save_freq == 0 or step == config.steps
        if saving_step:
            last_model_path, last_training_state = _save_training_checkpoint(
                dependencies=dependencies,
                config=config,
                train_config=train_config,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                step=step,
            )
            last_checkpoint_step = step

        stopped = should_stop()
        if step % config.progress_interval == 0 or saving_step or stopped:
            _emit_progress(
                config,
                _make_progress(
                    status="stopped" if stopped else "running",
                    step=step,
                    total_steps=config.steps,
                    started_at=started_at,
                    clock=clock,
                    loss=loss_value,
                ),
                progress_callback,
            )
        if stopped:
            break

    status = "complete" if step == config.steps else "stopped"
    if step > 0 and last_checkpoint_step != step:
        last_model_path, last_training_state = _save_training_checkpoint(
            dependencies=dependencies,
            config=config,
            train_config=train_config,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            step=step,
        )
    final_progress = _make_progress(
        status=status,
        step=step,
        total_steps=config.steps,
        started_at=started_at,
        clock=clock,
        loss=loss_value,
    )
    _emit_progress(config, final_progress, progress_callback)
    result = MultiTaskDiTILResult(
        status=status,
        step=step,
        total_steps=config.steps,
        percentage=final_progress.percentage,
        loss=loss_value,
        l1_loss=None,
        kld_loss=None,
        flow_matching_loss=loss_value,
        elapsed_seconds=final_progress.elapsed_seconds,
        model_path=str(last_model_path) if status == "complete" else None,
        checkpoint_path=(
            str(last_training_state) if last_training_state is not None else None
        ),
    )
    _atomic_json_save(config.output_dir / "result.json", result.to_dict())
    return result


__all__ = [
    "ACTION_DIM",
    "CONTROL_HZ",
    "DEFAULT_TASK_INSTRUCTION",
    "IMAGE_SHAPE",
    "IMAGE_SIZE",
    "MULTI_TASK_DIT_HORIZON",
    "MultiTaskDiTILConfig",
    "MultiTaskDiTILProgress",
    "MultiTaskDiTILResult",
    "OfficialTrainingDependencies",
    "STATE_DIM",
    "load_official_training_dependencies",
    "run_training",
    "training_manifest",
    "write_failed_result",
]
