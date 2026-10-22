"""Single-GPU native training and fresh-process checkpoint qualification."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np

from .channels import read_selection
from .checkpoint import file_hash, retain_checkpoints, save_checkpoint, verify_checkpoint
from .dataset import PreparedDataset
from .profile import canonical_hash, inside, load_profile, read_json


def validate_config(config):
    required = {"version", "model_id", "policy_id", "profile", "dataset", "assets",
                "output_dir", "parameters", "source"}
    if set(config) - required - {"channel_selection", "channel_selection_yaml"} or not required <= config.keys():
        raise ValueError("Run config has missing or unsupported fields")
    if config["version"] != 1 or config["model_id"] != "abot-m0" or config["policy_id"] != "abot:m0":
        raise ValueError("Expected an ABot-M0 version 1 Run")
    if "channel_selection" in config and "channel_selection_yaml" in config:
        raise ValueError("Provide channel_selection or channel_selection_yaml, not both")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only single-process, single-GPU training is supported")
    p = config["parameters"]
    required_parameters = {"steps", "batch_size", "gradient_accumulation_steps", "save_interval",
                           "save_total_limit", "learning_rate", "optimizer", "seed", "image_size",
                           "horizon", "repeated_diffusion_steps", "gradient_checkpointing", "max_grad_norm"}
    if set(p) != required_parameters:
        raise ValueError("Explicit training parameters are required; no upstream defaults are inherited")
    for key in ("steps", "batch_size", "gradient_accumulation_steps", "save_interval",
                "save_total_limit", "horizon", "repeated_diffusion_steps"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(p["seed"]) is not int or not 0 <= p["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(p["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("learning_rate", "max_grad_norm"):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if p["optimizer"] not in ("sgd", "adamw"):
        raise ValueError("optimizer must be sgd or adamw; neither is substituted automatically")
    if not isinstance(config["source"], dict) or not config["source"]:
        raise ValueError("Source/image provenance is required")
    return deepcopy(config)


def prepare(config):
    import yaml

    config = validate_config(config)
    receipt_path = Path("/opt/abot-build/receipt.json")
    if receipt_path.is_file():
        build = read_json(receipt_path)
        for key in ("cyclo_commit", "upstream_commit"):
            if config["source"].get(key) != build[key]:
                raise ValueError(f"Run source differs from built image: {key}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(config["source"].get("image_id", ""))):
            raise ValueError("Bench must provide the final image ID in source.image_id")
        config["source"]["build_receipt_sha256"] = file_hash(receipt_path)
        config["source"]["environment"] = {key: build[key] for key in (
            "python", "python_version", "torch", "cuda", "pyarrow", "flash_attention", "build_inputs_sha256",
        )}
    root = Path(config["dataset"]["path"]).resolve(strict=True)
    info = read_json(inside(root, "meta/info.json"))
    selection = config.get("channel_selection")
    if "channel_selection_yaml" in config:
        selection = read_selection(config.pop("channel_selection_yaml"))
        config["channel_selection"] = selection
    profiles = Path(__file__).resolve().parents[1] / "training_profiles"
    profile = load_profile(profiles, config["profile"]["id"], config["profile"]["sha256"],
                           info, channel_selection=selection)
    if set(config["assets"]) != set(profile["assets"]):
        raise ValueError("Run assets differ from the pinned profile")
    for key, declared in profile["assets"].items():
        supplied = config["assets"][key]
        for field in ("provider", "repo_id", "revision"):
            if supplied.get(field) != declared[field]:
                raise ValueError(f"Asset identity differs: {key}.{field}")
        directory = Path(supplied["path"]).resolve(strict=True)
        expected = supplied.get("sha256", {})
        if not expected or any(expected.get(name) != digest for name, digest in declared.get("files", {}).items()):
            raise ValueError(f"Asset inventory/hashes are required: {key}")
        entries = list(directory.rglob("*"))
        if (any(p.is_symlink() for p in entries)
                or {str(p.relative_to(directory)) for p in entries if p.is_file()} != set(expected)):
            raise ValueError(f"Asset file inventory differs: {key}")
        for name, digest in expected.items():
            if file_hash(inside(directory, name)) != digest:
                raise ValueError(f"Asset checksum mismatch: {key}/{name}")
    base = config["assets"][profile["asset_roles"]["base"]]
    processor = config["assets"][profile["asset_roles"]["processor"]]
    config_path = inside(base["path"], "config.yaml")
    weights = inside(base["path"], "checkpoints/ABot_M0_Pretrain.pt")
    pretrained_config = yaml.safe_load(config_path.read_text())
    source_data = pretrained_config.get("datasets", {}).get("vla_data", {})
    model_config = {
        "framework": deepcopy(pretrained_config["framework"]),
        "datasets": {"vla_data": {k: source_data[k] for k in ("CoT_prompt",) if k in source_data}},
    }
    p = config["parameters"]
    dataset = PreparedDataset(root, profile, horizon=p["horizon"], image_size=p["image_size"],
                              channel_selection=selection)
    model_config = dataset.configure_model(model_config)
    model_config["cyclo_qwen_config"] = read_json(inside(processor["path"], "config.json"))
    model_config["trainer"] = {"pretrained_checkpoint": None, "repeated_diffusion_steps": p["repeated_diffusion_steps"]}
    # Construction uses local configs; the upstream forward does not resolve this path.
    model_config["framework"]["qwenvl"]["base_vlm"] = str(processor["path"])
    sample = dataset[0]
    if not np.isfinite(sample["state"]).all() or not np.isfinite(sample["action"]).all():
        raise ValueError("Initial training batch contains nonfinite inputs")
    return config, dataset, model_config, weights, Path(processor["path"])


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prediction(model, sample, seed):
    import torch

    # Saving must not reset the training RNG stream or alter dropout state.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    was_training = model.training
    try:
        with torch.random.fork_rng():
            seed_all(seed)
            model.eval()
            with torch.inference_mode():
                result = model.predict_action([sample])["normalized_actions"]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train(was_training)
    if not np.isfinite(result).all():
        raise ValueError("Reload probe contains nonfinite actions")
    return result.astype(np.float32)


def train(config):
    import torch
    from .native import initialize

    config, dataset, model_config, weights, processor_path = prepare(config)
    print("CYCLO_STAGE " + json.dumps({"name": "input_validated", "dimensions": dataset.model_dimensions}), flush=True)
    if not torch.cuda.is_available():
        raise ValueError("Native ABot training requires CUDA")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Output directory must be empty; optimizer resume is unsupported")
    p = config["parameters"]
    seed_all(p["seed"])
    started = time.monotonic()
    model, report = initialize(model_config, processor_path, weights)
    print("CYCLO_STAGE " + json.dumps({"name": "pretrained_loaded", "initialization": report}), flush=True)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("This recipe does not silently freeze model parameters")
    model = model.cuda()
    model.qwen_vl_interface.model.config.use_cache = False
    if p["gradient_checkpointing"]:
        model.qwen_vl_interface.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if p["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=p["learning_rate"], momentum=0, foreach=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"],
                                     betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False)
    sampler = torch.Generator().manual_seed(p["seed"])
    order, cursor = [], 0
    latest = None
    for step in range(1, p["steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(p["gradient_accumulation_steps"]):
            batch = []
            for _ in range(p["batch_size"]):
                if cursor >= len(order):
                    order, cursor = torch.randperm(len(dataset), generator=sampler).tolist(), 0
                batch.append(dataset[order[cursor]])
                cursor += 1
            loss = model(batch)["action_loss"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite action loss")
            (loss / p["gradient_accumulation_steps"]).backward()
            losses.append(loss.detach().item())
        gradients = {"qwen_vl_interface": 0, "action_model": 0}
        missing = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                missing.append(name)
            else:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Nonfinite gradient: {name}")
                group = name.split(".")[0]
                if group in gradients:
                    gradients[group] += parameter.numel()
        if not all(gradients.values()):
            raise ValueError("Backbone and action expert must both receive gradients")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_grad_norm"],
                                             error_if_nonfinite=True, foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print("CYCLO_TRAINING_PROGRESS " + json.dumps({"version": 1, "optimizer_step": step,
              "total_steps": p["steps"], "metrics": {"loss": sum(losses) / len(losses),
              "action_loss": sum(losses) / len(losses), "learning_rate": p["learning_rate"]}}), flush=True)
        if step % p["save_interval"] == 0 or step == p["steps"]:
            expected = prediction(model, dataset[0], p["seed"])
            saved_run = deepcopy(config)
            saved_run["qualification"] = {"seed": p["seed"], "sample_index": 0,
                "normalized_actions": expected.tolist(), "rtol": 0.005, "atol": 0.005,
                "gradient_elements": gradients, "parameters_without_gradient": missing,
                "gradient_norm": float(norm), "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                "elapsed_s": time.monotonic() - started, "reload_verified": False}
            metadata = dataset.checkpoint_metadata(saved_run, model.saved_config(), report)
            latest = save_checkpoint(output, step=step, model=model,
                                     processor=model.qwen_vl_interface.processor, metadata=metadata)
            retain_checkpoints(output, save_total_limit=p["save_total_limit"])
            print("CYCLO_CHECKPOINT " + json.dumps({"path": str(latest), "optimizer_step": step}), flush=True)
    return latest


def reload_checkpoint(config, checkpoint):
    import torch
    from .native import reload_model

    config = validate_config(config)
    if checkpoint is None:
        raise ValueError("reload requires an explicit --checkpoint")
    receipt = verify_checkpoint(checkpoint)
    data = PreparedDataset.from_checkpoint(config["dataset"]["path"], checkpoint)
    saved = read_json(Path(checkpoint) / "train_config.json")
    if config["profile"] != saved["profile"] or config["model_id"] != saved["model_id"]:
        raise ValueError("Run identity differs from checkpoint")
    if config["parameters"] != saved["parameters"]:
        raise ValueError("Reload parameters differ from checkpoint")
    for key in ("repo_id", "revision"):
        if config["dataset"].get(key) != saved["dataset"].get(key):
            raise ValueError(f"Reload dataset identity differs: {key}")
    for key in ("cyclo_commit", "upstream_commit", "image_id"):
        if config["source"].get(key) != saved["source"].get(key):
            raise ValueError(f"Reload source identity differs: {key}")
    def identities(assets):
        return {key: {k: v for k, v in asset.items() if k != "path"}
                for key, asset in assets.items()}

    if identities(config["assets"]) != identities(saved["assets"]):
        raise ValueError("Reload asset identities differ from checkpoint")
    receipt_path = Path("/opt/abot-build/receipt.json")
    if receipt_path.is_file() and file_hash(receipt_path) != saved["source"].get("build_receipt_sha256"):
        raise ValueError("Reload image build receipt differs from checkpoint")
    selection = config.get("channel_selection")
    if "channel_selection_yaml" in config:
        selection = read_selection(config["channel_selection_yaml"])
    from .channels import resolve_channels
    if resolve_channels(data.info, selection) != data.channels:
        raise ValueError("Reload Run selection differs from checkpoint")
    probe = saved["qualification"]
    model_config = read_json(Path(checkpoint) / "model_config.json")
    model = reload_model(checkpoint, model_config, receipt).cuda()
    actual = prediction(model, data[probe["sample_index"]], probe["seed"])
    expected = np.asarray(probe["normalized_actions"], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=probe["rtol"], atol=probe["atol"])
    result = {"version": 1, "checkpoint": str(Path(checkpoint).resolve()),
              "checkpoint_receipt_sha256": file_hash(Path(checkpoint) / "checkpoint.json"),
              "passed": True, "max_absolute_error": float(np.max(np.abs(actual - expected))),
              "shape": list(actual.shape), "cuda_peak_bytes": torch.cuda.max_memory_allocated()}
    print("CYCLO_RELOAD_RESULT " + json.dumps(result), flush=True)
    return result


def main(config_path, phase, checkpoint=None):
    if phase != "reload" and checkpoint is not None:
        raise ValueError("--checkpoint is only supported for reload; optimizer resume is unsupported")
    config = read_json(config_path)
    if phase == "validate":
        _, data, _, _, _ = prepare(config)
        print("CYCLO_VALIDATION " + json.dumps({"passed": True, "samples": len(data),
              "dimensions": data.model_dimensions, "channels": data.channels,
              "statistics_sha256": canonical_hash(data.statistics)}), flush=True)
    elif phase == "train":
        train(config)
    elif phase == "reload":
        reload_checkpoint(config, checkpoint)
    else:
        raise ValueError("Unknown phase")
