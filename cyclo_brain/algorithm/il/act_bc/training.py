"""Cooperatively stoppable ACT behavior-cloning training loop.

The loop is deliberately small.  Model construction, preprocessing,
ACT/CVAE loss computation, optimizer presets, and checkpoint serialization all
come from the pinned LeRobot 0.5.2 fork.  Cyclo owns only multi-root selection,
progress reporting, and the cooperative stop boundary.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cyclo_brain.model.act.trainability import (
    ACT_TRAINABLE_GROUPS,
    apply_act_trainable_groups,
    canonicalize_act_trainable_groups,
)

from .dataset import (
    LeRobotDatasetDependencies,
    RootSelection,
    VirtualACTBCDataset,
    load_virtual_act_bc_dataset,
)


ACT_CHUNK_SIZE = 30
_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


@dataclass(frozen=True)
class OfficialTrainingDependencies:
    """LeRobot APIs isolated behind one injectable integration boundary."""

    dataset: LeRobotDatasetDependencies
    act_config_cls: type
    dataset_config_cls: type
    train_config_cls: type
    make_policy: Callable[..., Any]
    make_pre_post_processors: Callable[..., tuple[Any, Any]]
    make_optimizer_and_scheduler: Callable[..., tuple[Any, Any]]
    get_step_checkpoint_dir: Callable[[Path, int, int], Path]
    save_checkpoint: Callable[..., None]
    update_last_checkpoint: Callable[[Path], Any]
    cycle: Callable[[Any], Any]
    apply_trainable_groups: Callable[[Any, Sequence[str]], tuple[str, ...]] = (
        apply_act_trainable_groups
    )


def load_official_training_dependencies() -> OfficialTrainingDependencies:
    """Import the pinned LeRobot fork only inside its training process."""

    from lerobot.common.train_utils import (
        get_step_checkpoint_dir,
        save_checkpoint,
        update_last_checkpoint,
    )
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import _load_episode_with_stats
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.utils import cycle

    return OfficialTrainingDependencies(
        dataset=LeRobotDatasetDependencies(
            metadata_cls=LeRobotDatasetMetadata,
            dataset_cls=LeRobotDataset,
            resolve_delta_timestamps=resolve_delta_timestamps,
            aggregate_stats=aggregate_stats,
            load_episode_with_stats=_load_episode_with_stats,
        ),
        act_config_cls=ACTConfig,
        dataset_config_cls=DatasetConfig,
        train_config_cls=TrainPipelineConfig,
        make_policy=make_policy,
        make_pre_post_processors=make_pre_post_processors,
        make_optimizer_and_scheduler=make_optimizer_and_scheduler,
        get_step_checkpoint_dir=get_step_checkpoint_dir,
        save_checkpoint=save_checkpoint,
        update_last_checkpoint=update_last_checkpoint,
        cycle=cycle,
    )


@dataclass(frozen=True)
class ACTBCTrainingConfig:
    """Immutable configuration for one from-scratch ACT imitation run."""

    selections: tuple[RootSelection, ...]
    output_dir: Path
    steps: int
    batch_size: int
    save_freq: int
    progress_interval: int = 10
    chunk_size: int = ACT_CHUNK_SIZE
    learning_rate: float = 1e-5
    num_workers: int = 4
    seed: int = 1000
    device: str = "cuda"
    video_backend: str = "pyav"
    grad_clip_norm: float = 10.0
    trainable_groups: tuple[str, ...] = field(default_factory=lambda: ACT_TRAINABLE_GROUPS)

    def __post_init__(self) -> None:
        selections = tuple(self.selections)
        if not selections:
            raise ValueError("ACT behavior cloning requires at least one dataset root")
        roots = tuple(selection.root for selection in selections)
        if len(roots) != len(set(roots)):
            raise ValueError("ACT behavior-cloning dataset roots cannot contain duplicates")
        output_dir = Path(self.output_dir).expanduser().resolve()
        for root in roots:
            if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
                raise ValueError(
                    "ACT behavior-cloning output directory must not contain or be contained by a dataset root"
                )
        for name, value in (
            ("steps", self.steps),
            ("batch_size", self.batch_size),
            ("save_freq", self.save_freq),
            ("progress_interval", self.progress_interval),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.chunk_size != ACT_CHUNK_SIZE:
            raise ValueError(
                f"Cyclo ACT imitation learning requires chunk_size={ACT_CHUNK_SIZE}"
            )
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(self.grad_clip_norm)) or self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be finite and positive")
        device = str(self.device)
        if device != "cpu" and device != "cuda" and not (
            device.startswith("cuda:") and device[len("cuda:") :].isdigit()
        ):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        groups = canonicalize_act_trainable_groups(self.trainable_groups)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "trainable_groups", groups)


@dataclass(frozen=True)
class ACTBCTrainingProgress:
    status: str
    step: int
    total_steps: int
    percentage: float
    loss: float | None
    l1_loss: float | None
    kld_loss: float | None
    elapsed_seconds: float
    eta_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"event": "progress", **self.__dict__}


@dataclass(frozen=True)
class ACTBCTrainingResult:
    status: str
    step: int
    total_steps: int
    percentage: float
    loss: float | None
    l1_loss: float | None
    kld_loss: float | None
    elapsed_seconds: float
    model_path: str | None
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"event": "result", **self.__dict__}


def _atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_failed_result(output_dir: Path, error: BaseException) -> dict[str, Any]:
    """Persist a machine-readable terminal failure when CLI setup had started."""

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


def _manifest(config: ACTBCTrainingConfig) -> dict[str, Any]:
    return {
        "event": "manifest",
        "schema_version": 1,
        "method": "imitation_learning",
        "policy_type": "act",
        "implementation": "lerobot-0.5.2-act-cvae",
        "dataset_roots": [str(selection.root) for selection in config.selections],
        "success_episode_indices": [
            list(selection.success_episodes) for selection in config.selections
        ],
        "output_dir": str(config.output_dir),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "save_freq": config.save_freq,
        "progress_interval": config.progress_interval,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.chunk_size,
        "learning_rate": config.learning_rate,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "device": config.device,
        "video_backend": config.video_backend,
        "trainable_groups": list(config.trainable_groups),
        "loss": "masked_l1 + kl_weight * kld",
        "use_vae": True,
    }


def _prepare_output(config: ACTBCTrainingConfig) -> None:
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {config.output_dir}")
        if any(config.output_dir.iterdir()):
            raise FileExistsError(
                f"ACT behavior-cloning output directory is not empty: {config.output_dir}"
            )
    else:
        config.output_dir.mkdir(parents=True)
    _atomic_json_save(config.output_dir / "manifest.json", _manifest(config))


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
    metrics: Mapping[str, float | None],
) -> ACTBCTrainingProgress:
    elapsed = max(0.0, float(clock() - started_at))
    percentage = 100.0 * step / total_steps
    eta = None if step == 0 else max(0.0, elapsed * (total_steps - step) / step)
    if status != "running":
        eta = 0.0 if status == "complete" else None
    return ACTBCTrainingProgress(
        status=status,
        step=step,
        total_steps=total_steps,
        percentage=percentage,
        loss=metrics.get("loss"),
        l1_loss=metrics.get("l1_loss"),
        kld_loss=metrics.get("kld_loss"),
        elapsed_seconds=elapsed,
        eta_seconds=eta,
    )


def _emit_progress(
    config: ACTBCTrainingConfig,
    progress: ACTBCTrainingProgress,
    callback: Callable[[ACTBCTrainingProgress], None] | None,
) -> None:
    _atomic_json_save(config.output_dir / "progress.json", progress.to_dict())
    if callback is not None:
        callback(progress)


def _validate_checkpoint(checkpoint_dir: Path) -> tuple[Path, Path]:
    model_dir = checkpoint_dir / "pretrained_model"
    training_state = checkpoint_dir / "training_state"
    missing = [name for name in _MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"ACT checkpoint {checkpoint_dir} is missing model files: " + ", ".join(missing)
        )
    if not training_state.is_dir() or not (training_state / "training_step.json").is_file():
        raise RuntimeError(f"ACT checkpoint {checkpoint_dir} has no valid training_state")
    return model_dir.resolve(), training_state.resolve()


def _save_training_checkpoint(
    *,
    dependencies: OfficialTrainingDependencies,
    config: ACTBCTrainingConfig,
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
    return _validate_checkpoint(checkpoint_dir)


def _make_policy_and_config(
    config: ACTBCTrainingConfig,
    dataset: VirtualACTBCDataset,
    dependencies: OfficialTrainingDependencies,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    policy_config = dependencies.act_config_cls(
        chunk_size=config.chunk_size,
        n_action_steps=config.chunk_size,
        optimizer_lr=config.learning_rate,
        optimizer_lr_backbone=config.learning_rate,
        device=config.device,
        use_vae=True,
    )
    dataset_config = dependencies.dataset_config_cls(
        repo_id=f"cyclo-local/{config.selections[0].root.name}",
        root=str(config.selections[0].root),
        episodes=list(config.selections[0].success_episodes),
        video_backend=config.video_backend,
    )
    train_config = dependencies.train_config_cls(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=config.output_dir,
        job_name="cyclo_act_imitation_learning",
        seed=config.seed,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        steps=config.steps,
        eval_freq=0,
        log_freq=config.progress_interval,
        save_checkpoint=True,
        save_freq=config.save_freq,
    )
    train_config.optimizer = policy_config.get_optimizer_preset()
    train_config.optimizer.grad_clip_norm = config.grad_clip_norm
    train_config.scheduler = policy_config.get_scheduler_preset()
    policy = dependencies.make_policy(cfg=policy_config, ds_meta=dataset.meta, rename_map={})
    applied_groups = dependencies.apply_trainable_groups(policy, config.trainable_groups)
    if tuple(applied_groups) != config.trainable_groups:
        raise RuntimeError("ACT trainability helper changed the canonical group selection")
    preprocessor, postprocessor = dependencies.make_pre_post_processors(
        policy_cfg=policy_config,
        dataset_stats=dataset.meta.stats,
    )
    optimizer, scheduler = dependencies.make_optimizer_and_scheduler(train_config, policy)
    return train_config, policy, preprocessor, postprocessor, optimizer, scheduler


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _build_dataloader(
    config: ACTBCTrainingConfig,
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


def run_training(
    config: ACTBCTrainingConfig,
    *,
    dependencies: OfficialTrainingDependencies | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_callback: Callable[[ACTBCTrainingProgress], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ACTBCTrainingResult:
    """Train ACT on selected successes, stopping only at optimizer boundaries."""

    dependencies = dependencies or load_official_training_dependencies()
    should_stop = should_stop or (lambda: False)
    _prepare_output(config)
    started_at = clock()
    generator = _seed_everything(config.seed)

    # ACTConfig is needed to derive the exact 30-action delta timestamp window.
    window_config = dependencies.act_config_cls(
        chunk_size=config.chunk_size,
        n_action_steps=config.chunk_size,
        device=config.device,
        use_vae=True,
    )
    dataset = load_virtual_act_bc_dataset(
        config.selections,
        policy_config=window_config,
        dependencies=dependencies.dataset,
        video_backend=config.video_backend,
    )
    train_config, policy, preprocessor, postprocessor, optimizer, scheduler = (
        _make_policy_and_config(config, dataset, dependencies)
    )
    dataloader = _build_dataloader(config, dataset, generator)
    batches = dependencies.cycle(dataloader)
    policy.train()

    metrics: dict[str, float | None] = {
        "loss": None,
        "l1_loss": None,
        "kld_loss": None,
    }
    initial = _make_progress(
        status="running",
        step=0,
        total_steps=config.steps,
        started_at=started_at,
        clock=clock,
        metrics=metrics,
    )
    _emit_progress(config, initial, progress_callback)
    last_checkpoint_step = 0
    last_model_path: Path | None = None
    last_training_state: Path | None = None
    step = 0

    while step < config.steps:
        if should_stop():
            break
        batch = next(batches)
        if should_stop():
            break
        for camera_key in dataset.meta.camera_keys:
            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, output = policy.forward(batch)
        total_loss = _finite_metric(loss, name="loss")
        loss.backward()
        trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1

        output = output or {}
        metrics = {
            "loss": total_loss,
            "l1_loss": _finite_metric(output.get("l1_loss", total_loss), name="l1_loss"),
            "kld_loss": _finite_metric(output.get("kld_loss", 0.0), name="kld_loss"),
        }
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
            progress = _make_progress(
                status="running" if not stopped else "stopped",
                step=step,
                total_steps=config.steps,
                started_at=started_at,
                clock=clock,
                metrics=metrics,
            )
            _emit_progress(config, progress, progress_callback)
        if stopped:
            break

    status = "complete" if step == config.steps else "stopped"
    if status == "stopped" and step > 0 and last_checkpoint_step != step:
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

    terminal_progress = _make_progress(
        status=status,
        step=step,
        total_steps=config.steps,
        started_at=started_at,
        clock=clock,
        metrics=metrics,
    )
    _emit_progress(config, terminal_progress, progress_callback)
    result = ACTBCTrainingResult(
        status=status,
        step=step,
        total_steps=config.steps,
        percentage=terminal_progress.percentage,
        loss=terminal_progress.loss,
        l1_loss=terminal_progress.l1_loss,
        kld_loss=terminal_progress.kld_loss,
        elapsed_seconds=terminal_progress.elapsed_seconds,
        # A stopped checkpoint is resumable/debuggable, but is never advertised
        # as a completed deployment candidate.
        model_path=str(last_model_path) if status == "complete" else None,
        checkpoint_path=str(last_training_state) if last_training_state is not None else None,
    )
    _atomic_json_save(config.output_dir / "result.json", result.to_dict())
    return result


__all__ = [
    "ACT_CHUNK_SIZE",
    "ACTBCTrainingConfig",
    "ACTBCTrainingProgress",
    "ACTBCTrainingResult",
    "OfficialTrainingDependencies",
    "load_official_training_dependencies",
    "run_training",
    "write_failed_result",
]
