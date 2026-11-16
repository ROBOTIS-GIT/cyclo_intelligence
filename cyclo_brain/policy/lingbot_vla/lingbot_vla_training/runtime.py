"""Native full-model training with explicit Cyclo input and artifact contracts.

Invoke through torchrun, including for one GPU. Resume is deliberately absent.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from .artifacts import complete_checkpoint, prune_checkpoints, sha256, verify_assets, verify_checkpoint, write_json
from .data import PreparedEpisodes, TrainingDataset
from .losses import MaskedTarget, install_boundary_losses, prepare_loss_inputs


DEFAULTS = dict(max_steps=2, micro_batch_size=1, gradient_accumulation_steps=1, save_steps=1,
                save_total_limit=1, num_workers=0, seed=42, image_augment=False,
                enable_gradient_checkpointing=True, use_compile=False, chunk_size=50, image_size=256)


def typed_native_arguments(values, argument_class):
    """Match native argparse float conversion, including YAML scientific notation."""
    result = dict(values)
    for field in fields(argument_class):
        if field.type is float and field.name in result:
            value = result[field.name]
            if isinstance(value, bool):
                raise ValueError(f"{field.name} must be a finite number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            result[field.name] = value
    return result


def read_run(path):
    path = Path(path).resolve(strict=True)
    run = json.loads(path.read_text())
    if run.get("version") != 1 or set(run) - {"version", "profile", "dataset", "assets_manifest", "recipe", "output_dir", "settings"}:
        raise ValueError("Expected a version-1 LingBot training Run")
    settings = {**DEFAULTS, **run.get("settings", {})}
    if set(settings) - set(DEFAULTS) - {"lr"}:
        raise ValueError("Unknown training setting")
    for key in ("max_steps", "micro_batch_size", "gradient_accumulation_steps", "save_steps", "save_total_limit"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("image_augment", "enable_gradient_checkpointing", "use_compile"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if settings["use_compile"]:
        raise ValueError("Compilation of boundary-masked training has not been qualified")
    profile = yaml.safe_load(Path(run["profile"]).read_text())
    manifest = json.loads(Path(run["assets_manifest"]).read_text())
    paths = verify_assets(manifest)
    recipe = yaml.safe_load(Path(run["recipe"]).read_text())
    align = recipe["train"]["align_params"]
    if (
        recipe["train"].get("optimizer") != "muon" or recipe["train"].get("data_parallel_mode") != "fsdp2"
        or align.get("depth", {}).get("model_type") != "MoRGBD"
        or align.get("depth", {}).get("use_future_depth") is not True
        or align.get("use_future_video") is not True
        or align.get("video", {}).get("use_current_patch_loss") is not True
        or align["video"].get("use_cls_loss", False)
        or align["video"].get("num_future_frames") != 1
        or any(recipe["train"].get(key, False) for key in ("freeze_vit", "freeze_vision_encoder", "train_expert_only"))
    ):
        raise ValueError("Expected full-model FSDP2/Muon native depth/future-depth/current+future-video recipe")
    for value in (align.get("depth_loss_weight"), align.get("future_depth_loss_weight"), align["video"].get("future_video_loss_weight")):
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError("All teacher losses must have finite positive weights")
    episodes = PreparedEpisodes(run["dataset"], profile)
    stats = episodes.statistics()
    settings.setdefault("lr", recipe["train"]["lr"])
    if type(settings["chunk_size"]) is not int or settings["chunk_size"] < 2:
        raise ValueError("chunk_size must be at least two")
    if type(settings["image_size"]) is not int or settings["image_size"] < 1:
        raise ValueError("image_size must be positive")
    if not isinstance(settings["lr"], (int, float)) or not 0 < settings["lr"] < float("inf"):
        raise ValueError("lr must be finite and positive")
    return run, settings, profile, manifest, paths, recipe, episodes, stats


def configure(recipe, paths, settings, output):
    from tasks.vla.train_lingbotvla import MyTrainingArguments
    recipe = deepcopy(recipe)
    train = recipe["train"]
    train.update({key: settings[key] for key in ("max_steps", "micro_batch_size", "gradient_accumulation_steps", "save_steps",
                                                 "seed", "enable_gradient_checkpointing", "use_compile", "chunk_size", "lr")})
    train.update(output_dir=str(output), global_batch_size=None, num_train_epochs=None,
                 enable_resume=False, load_checkpoint_path=None, init_device="cuda", save_hf_weights=False)
    model = recipe["model"]
    model.update(tokenizer_path=str(paths["processor"]), model_path=str(paths["pretrained"]),
                 vlm_repo_id=None, post_training=True, incremental_training=False)
    align = train["align_params"]
    align["depth"].update(moge_path=str(paths["moge"]), morgbd_path=str(paths["morgbd"]))
    align["video"].update(ckpt_path=str(paths["video"]), config_path=str(paths["video_config"]))
    allowed = {f.name for f in fields(MyTrainingArguments) if f.init}
    unknown = set(train) - allowed
    if unknown:
        raise ValueError(f"Unknown native training settings: {sorted(unknown)}")
    train.update(typed_native_arguments(train, MyTrainingArguments))
    args = MyTrainingArguments(**train)
    if args.enable_fsdp_offload or args.enable_activation_offload:
        raise ValueError("Offloaded training is not qualified by this wrapper")
    return recipe, args


def build_model(recipe, args, weights):
    from lingbotvla.models.loader import CustomizedModelingLoader
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    kwargs = {**recipe["model"], **vars(args)}
    config = LingbotVLAV2Config(**kwargs)
    # Native factory guesses Qwen generation from a directory name. Select the
    # v2 implementation explicitly so portable asset mount paths remain valid.
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    policy = CustomizedModelingLoader(LingbotVlaV2Policy).load_model(
        init_kwargs={"config": config, "torch_dtype": torch.float32, "attn_implementation": "flash_attention_2"},
        weights_path=str(weights), init_device="cuda", post_training=True,
        adanorm_time=recipe["model"].get("adanorm_time", False), incremental_training=False,
    )
    install_boundary_losses(policy.model)
    policy = build_parallelize_model(
        policy, enable_full_shard=args.enable_full_shard, enable_mixed_precision=args.enable_mixed_precision,
        enable_fp32=args.enable_fp32, enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        init_device="cuda", fsdp_kwargs={}, basic_modules=policy._no_split_modules if args.module_fsdp_enable else None,
        enable_reentrant=args.enable_reentrant, enable_forward_prefetch=args.enable_forward_prefetch,
        fsdp_llm_blocks=False, ignore_norm=False, use_depth_align=True,
        split_fused_experts_from_decoder_fsdp=args.split_fused_experts_from_decoder_fsdp,
        vlm_fsdp=args.vlm_fsdp, use_future_image=True,
    )
    return policy, config


def start_distributed():
    from lingbotvla.distributed.parallel_state import init_parallel_state
    if not all(key in os.environ for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        raise ValueError("Use python -m torch.distributed.run --standalone --nproc_per_node=N")
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise ValueError("This initial qualification supports one GPU; multi-rank masked reduction is not qualified")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    init_parallel_state(dp_size=dist.get_world_size(), dp_shard_size=dist.get_world_size(), dp_mode="fsdp2")


def teacher_inputs(batch, teachers, align):
    from lingbotvla.models.vla.vision_models.module_utils import get_depth_target, get_video_target
    batch = {name: value.cuda() for name, value in batch.items()}
    current, future = batch.pop("pil_images"), batch.pop("future_pil_images")
    fps = batch.pop("effective_fps")
    if not torch.allclose(fps, fps[:1].expand_as(fps)):
        raise ValueError("Mixed dataset FPS is unsupported")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        depth, _ = get_depth_target("MoRGBD", teachers[:2], current)
        future_depth, _ = get_depth_target("MoRGBD", teachers[:2], future)
        video = get_video_target(teachers[2], current, future, align["video"], effective_fps=float(fps[0]))
    return prepare_loss_inputs(batch, depth=depth, future_depth=future_depth, video=video)


@contextmanager
def training_probe_context(policy):
    # Eval selects upstream's atomic-accumulation MoE shortcut. Verify the
    # training forward instead, without changing router counters or model modes.
    snapshots = []
    for module in policy.modules():
        buffers = {name: (value, None if value is None else value.detach().clone())
                   for name, value in module._buffers.items() if name in module._non_persistent_buffers_set}
        snapshots.append((module, module.training, buffers))
    policy.train()
    try:
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for module, training, buffers in snapshots:
                module.training = training
                for name, (original, saved) in buffers.items():
                    if original is not None:
                        original.copy_(saved)
                    module._buffers[name] = original


def probe(policy, inputs):
    inputs = dict(inputs)
    generator = torch.Generator(device="cuda").manual_seed(1729)
    inputs["noise"] = torch.randn(inputs["actions"].shape, generator=generator, device="cuda")
    inputs["time"] = torch.full(inputs["actions"].shape[:1], 0.5, device="cuda")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1729)
        with training_probe_context(policy):
            result = policy(**inputs)
    values = {f"loss_{index}": torch.as_tensor(value).detach().float().cpu() for index, value in enumerate(result[:6])}
    for index, value in enumerate(result[7:], start=7):
        if value is not None:
            values[f"prediction_{index}"] = value.detach().float().cpu()
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite reload probe")
    return inputs, values


def save_checkpoint(policy, config, processor, inputs, *, output, step, train_config, stats, profile, keep):
    from lingbotvla.models.module_utils import save_model_weights
    from safetensors.torch import save_file
    rank = dist.get_rank()
    directory = Path(output) / "checkpoints"
    staging = directory / f".step-{step}.partial"
    destination = directory / f"step-{step}"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        staging.mkdir(exist_ok=False)
    dist.barrier()
    inputs, reference = probe(policy, inputs)
    save_model_weights(staging, policy.state_dict(), global_rank=rank, save_dtype=None, model_assets=[config])
    if rank == 0:
        processor.save_pretrained(staging / "processor")
        # Qwen's architecture config is required in addition to tokenizer assets.
        shutil.copy2(Path(config.tokenizer_path) / "config.json", staging / "processor/config.json")
        write_json(staging / "train_config.json", train_config)
        write_json(staging / "norm_stats.json", stats)
        write_json(staging / "cyclo_io_mapping.json", {
            "version": 2, "policy_id": profile["policy_id"], "action_representation": "absolute",
            "state_names": profile["state_names"], "action_names": profile["action_names"],
            "canonical_mapping": profile["canonical_mapping"], "transforms": profile["transforms"],
            "cameras": profile["cameras"], "camera_order": profile["camera_order"], "normalization": "mean_std",
        })
        packed = {f"reference.{key}": value.contiguous() for key, value in reference.items()}
        for name, value in inputs.items():
            if isinstance(value, MaskedTarget):
                packed[f"masked.{name}"] = value.values.cpu().contiguous()
                packed[f"valid.{name}"] = value.valid.cpu().contiguous()
            else:
                packed[f"input.{name}"] = value.cpu().contiguous()
        # Clone shared teacher tensors: safetensors rejects aliased storage.
        save_file({key: value.clone() for key, value in packed.items()}, staging / "probe.safetensors")
        complete_checkpoint(staging, destination, step)
        prune_checkpoints(directory, keep)
        print(json.dumps({"event": "checkpoint_complete", "optimizer_step": step, "path": str(destination)}), flush=True)
    dist.barrier()


def train(run_path):
    print(json.dumps({"event": "preflight_started"}), flush=True)
    run, settings, profile, assets, paths, recipe, episodes, stats = read_run(run_path)
    output = Path(run["output_dir"])
    start_distributed()
    try:
        rank = dist.get_rank()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        recipe, args = configure(recipe, paths, settings, output)
        from lingbotvla.utils import helper
        from lingbotvla.models import build_processor
        from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, build_video_model
        from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
        from lingbotvla.optim import build_muon_optimizer, build_lr_scheduler

        helper.set_seed(args.seed, args.enable_full_determinism)
        print(json.dumps({"event": "pretrained_load_started"}), flush=True)
        policy, config = build_model(recipe, args, paths["pretrained"])
        print(json.dumps({"event": "pretrained_loaded", "parameter_count": sum(p.numel() for p in policy.parameters()),
                          "trainable_parameter_count": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                          "gpu_allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
        processor = build_processor(str(paths["processor"]))
        teachers = (*build_depth_model(args.align_params), build_video_model(args.align_params["video"]))
        for teacher in teachers:
            teacher.eval().requires_grad_(False)
        print(json.dumps({"event": "teachers_loaded", "gpu_allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
        dataset = TrainingDataset(episodes, stats, processor, config, chunk_size=args.chunk_size,
                                  image_size=settings["image_size"], warmup=args.align_params["video"]["use_warmup_frame"],
                                  image_augment=settings["image_augment"])
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.micro_batch_size, sampler=sampler,
                            num_workers=settings["num_workers"], drop_last=False)
        optimizer = build_muon_optimizer(policy, args, lr=args.lr, weight_decay=args.weight_decay)
        optimizer.register_step_pre_hook(build_moe_load_balance_hook(
            policy, coeff=args.bias_update_speed, bias_centering=args.bias_centering, update_interval=args.bias_update_interval))
        scheduler = build_lr_scheduler(optimizer, train_steps=args.max_steps, lr=args.lr, lr_min=args.lr_min,
                                       lr_decay_style=args.lr_decay_style, lr_decay_ratio=args.lr_decay_ratio,
                                       lr_warmup_ratio=args.lr_warmup_ratio, lr_start=args.lr_start)
        provenance = {"version": 1, "profile": profile, "recipe": recipe, "settings": settings,
                      "assets": assets, "source_run_sha256": sha256(run_path), "dataset_layout": "v2.1",
                      "python": os.sys.executable, "cyclo_commit": os.environ.get("CYCLO_COMMIT", "uncommitted"),
                      "world_size": dist.get_world_size(), "gpu": torch.cuda.get_device_name(),
                      "resume_supported": False, "checkpoint_format": "native_safetensors_weights_only"}
        if rank == 0:
            write_json(output / "train_config.json", provenance)
        policy.train()
        iterator, epoch = iter(loader), 0
        for step in range(1, args.max_steps + 1):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(6, device="cuda")
            for _ in range(args.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                inputs = teacher_inputs(batch, teachers, args.align_params)
                # Native FSDP owns model precision. An outer autocast changes
                # FP32 flex-attention outputs and breaks its eager backward.
                result = policy(**inputs)
                loss = result[0] / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                metrics += torch.stack([torch.as_tensor(value, device="cuda").detach() for value in result[:6]]) / args.gradient_accumulation_steps
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(json.dumps({"event": "optimizer_step", "optimizer_step": step,
                                  **dict(zip(("loss", "action_loss", "depth_loss", "future_depth_loss", "video_loss", "moe_loss"), metrics.tolist())),
                                  "grad_norm": float(norm), "gpu_peak_bytes": torch.cuda.max_memory_allocated()}), flush=True)
                print(json.dumps({"event": "step_timing", "optimizer_step": step,
                                  "seconds": time.monotonic() - started}), flush=True)
            if step % args.save_steps == 0 or step == args.max_steps:
                save_checkpoint(policy, config, processor, inputs, output=output, step=step,
                                train_config=provenance, stats=stats, profile=profile, keep=settings["save_total_limit"])
    finally:
        dist.destroy_process_group()


def reload_checkpoint(checkpoint):
    from safetensors.torch import load_file
    checkpoint = Path(checkpoint).resolve(strict=True)
    receipt = verify_checkpoint(checkpoint)
    saved = json.loads((checkpoint / "train_config.json").read_text())
    paths = {key: Path(value["path"]) for key, value in saved["assets"]["assets"].items()}
    paths.update(pretrained=checkpoint, processor=checkpoint / "processor")
    # Teachers are not loaded: the exact detached targets accompany the probe.
    for key in ("moge", "morgbd", "video", "video_config"):
        paths[key] = Path("/unused-for-reload")
    start_distributed()
    try:
        recipe, args = configure(saved["recipe"], paths, saved["settings"], checkpoint)
        policy, _ = build_model(recipe, args, checkpoint)
        packed = load_file(checkpoint / "probe.safetensors")
        inputs = {key.removeprefix("input."): value.cuda() for key, value in packed.items() if key.startswith("input.")}
        for key, value in packed.items():
            if key.startswith("masked."):
                name = key.removeprefix("masked.")
                inputs[name] = MaskedTarget(value.cuda(), packed[f"valid.{name}"].cuda())
        _, actual = probe(policy, inputs)
        for key, value in actual.items():
            torch.testing.assert_close(value, packed[f"reference.{key}"], rtol=1e-4, atol=1e-5)
        if dist.get_rank() == 0:
            print(json.dumps({"event": "reload_verified", "checkpoint": str(checkpoint), "optimizer_step": receipt["optimizer_step"],
                              "complete_sha256": sha256(checkpoint / "complete.json"), "pid": os.getpid(),
                              "compared": sorted(actual), "resume_verified": False}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "train", "reload"))
    parser.add_argument("path", type=Path, help="Run JSON for validate/train, completed checkpoint for reload")
    args = parser.parse_args()
    if args.phase == "train":
        train(args.path)
    elif args.phase == "reload":
        reload_checkpoint(args.path)
    else:
        _, settings, _, _, _, _, episodes, _ = read_run(args.path)
        print(json.dumps({"phase": "validated", "frames": len(episodes), "episodes": len(episodes.episodes),
                          "settings": settings, "training_qualified": False}))


if __name__ == "__main__":
    main()
