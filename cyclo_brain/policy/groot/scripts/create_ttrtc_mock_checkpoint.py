#!/usr/bin/env python3
"""Build a full GR00T checkpoint for TT-RTC plumbing tests without training.

The action weights come from the official base model. The entire VLM comes from
the supplied vision-tuned overlay. This is explicitly marked as an untrained
TT-RTC mock and can use the normal robot publishing path for Gazebo testing.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_checkpoint(base: Path, vlm_weights: Path, training_run: Path, output: Path,
                     max_shard_size_mb: int = 1024) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base, vlm_weights, training_run, output = [
        Path(path).resolve() for path in (base, vlm_weights, training_run, output)
    ]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output}")
    if max_shard_size_mb < 1:
        raise ValueError("max_shard_size_mb must be positive")
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((training_run / "experiment_cfg/final_model_config.json").read_text())
    if (config.get("model_type") != "Gr00tN1d7" or not config.get("training_time_rtc")
            or config.get("logical_action_horizon") != 32 or config.get("logical_action_dim") != 16
            or config.get("action_horizon") != 40 or config.get("max_action_dim") != 132):
        raise ValueError("Training run must describe the GR00T-N1.7 H32/D16 TT-RTC architecture")
    processor_dir = training_run / "processor"
    processor_files = ("processor_config.json", "statistics.json", "embodiment_id.json")
    for name in processor_files:
        if not (processor_dir / name).is_file():
            raise FileNotFoundError(processor_dir / name)

    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(base / name, framework="pt", device="cpu"))
                   for name in sorted(set(index.values()))}
        overlay = stack.enter_context(safe_open(vlm_weights, framework="pt", device="cpu"))
        backbone_keys = {key for key in index if key.startswith("backbone.")}
        sources = {key: key for key in overlay.keys()}
        # Safetensors omits this alias of Qwen's tied token embedding. Both
        # checkpoint entries must contain the overlaid value when re-sharded.
        lm_head = "backbone.model.lm_head.weight"
        embedding = "backbone.model.model.language_model.embed_tokens.weight"
        if lm_head in backbone_keys and lm_head not in sources and embedding in sources:
            sources[lm_head] = embedding
        if set(sources) != backbone_keys:
            raise ValueError("VLM overlay must contain the complete backbone and no action weights")
        for key, source_key in sources.items():
            if (overlay.get_slice(source_key).get_shape()
                    != readers[index[key]].get_slice(key).get_shape()):
                raise ValueError(f"VLM tensor shape differs from the base checkpoint: {key}")

        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
        try:
            shard_paths, shard_keys, tensors = [], [], {}
            shard_bytes = total_bytes = 0
            max_bytes = max_shard_size_mb * 1024 * 1024

            def flush():
                if not tensors:
                    return
                path = staging / f"part-{len(shard_paths) + 1:05d}.safetensors"
                save_file(tensors, path, metadata={"format": "pt"})
                shard_paths.append(path)
                shard_keys.append(list(tensors))
                tensors.clear()

            for key in sorted(index):
                reader = overlay if key in sources else readers[index[key]]
                tensor = reader.get_tensor(sources.get(key, key))
                target_dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
                size = tensor.numel() * (2 if tensor.is_floating_point() else tensor.element_size())
                if tensors and shard_bytes + size > max_bytes:
                    flush()
                    shard_bytes = 0
                # Separate storage also handles the tied lm_head/embedding alias
                # without save_file rejecting shared tensors in the same shard.
                tensors[key] = tensor.to(dtype=target_dtype, device="cpu", copy=True).contiguous()
                shard_bytes += size
                total_bytes += size
            flush()
            weight_map = {}
            for position, (path, keys) in enumerate(zip(shard_paths, shard_keys), start=1):
                name = f"model-{position:05d}-of-{len(shard_paths):05d}.safetensors"
                path.rename(staging / name)
                weight_map.update({key: name for key in keys})
            write_json(staging / "model.safetensors.index.json",
                       {"metadata": {"total_size": total_bytes}, "weight_map": weight_map})
            config.update(dtype="bfloat16", load_bf16=True, mock_checkpoint=True,
                          mock_checkpoint_kind="base-plus-vision-no-ttrtc-training",
                          trained=False, complete=False, step=0)
            config.setdefault("state_history_length", 1)
            write_json(staging / "config.json", config)
            for name in processor_files:
                shutil.copy2(processor_dir / name, staging / name)

            manifest = {
                "schema": "cyclo.tt-rtc-training/v1",
                "scope": "mock-inference-only",
                "mock_checkpoint": True,
                "trained": False,
                "complete": False,
                "step": 0,
                "training_time_rtc": True,
                "logical_action_horizon": 32,
                "logical_action_dim": 16,
                "model_action_horizon": 40,
                "model_action_dim": 132,
                "prefix_choices": list(range(9)),
                "normalization": "train-only-min-max-clipped",
                "inference_contract": {
                    "prefix_coordinates": "normalized-logical", "prefix_shape": "[B,d,16]",
                    "first_prefix_length": 0, "action_fps": 15, "num_inference_timesteps": 4,
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_checkpoint": str(base),
                "base_index_sha256": sha256(base / "model.safetensors.index.json"),
                "vlm_overlay": {"path": str(vlm_weights), "sha256": sha256(vlm_weights),
                                "tensor_count": len(sources), "tied_alias_restored": lm_head},
                "processor_source": str(processor_dir),
                "processor_sha256": {name: sha256(processor_dir / name) for name in processor_files},
                "config_source": str(training_run / "experiment_cfg/final_model_config.json"),
                "action_weights_source": "official-base-checkpoint-without-ttrtc-training",
                "tensor_count": len(index), "weight_bytes": total_bytes,
                "dtype": "bfloat16",
            }
            write_json(staging / "experiment_cfg/tt_rtc_training_manifest.json", manifest)
            (staging / "MOCK_CHECKPOINT.txt").write_text(
                "UNTRAINED TT-RTC MOCK CHECKPOINT\n"
                "Full GR00T-N1.7 base action weights + vision-tuned VLM.\n"
                "Uses the training run's 15 Hz processor and TT-RTC architecture.\n"
                "No TT-RTC optimization steps have been performed on this artifact.\n"
                "Intended for load/inference/queue integration tests, including Gazebo robot mode.\n"
            )
            if output.exists():
                raise FileExistsError(f"Output appeared during generation: {output}")
            staging.chmod(0o755)
            staging.rename(output)
            return {"path": str(output), "tensor_count": len(index),
                    "weight_bytes": total_bytes, "shards": len(shard_paths),
                    "mock_checkpoint": True, "trained": False}
        except BaseException:
            shutil.rmtree(staging)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Cached original GR00T-N1.7 checkpoint")
    parser.add_argument("--vlm-weights", type=Path, required=True, help="Full vision-tuned VLM safetensors")
    parser.add_argument("--training-run", type=Path, required=True, help="Run containing processor and experiment_cfg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size-mb", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(build_checkpoint(args.base, args.vlm_weights, args.training_run,
                                      args.output, args.max_shard_size_mb), indent=2))


if __name__ == "__main__":
    main()
