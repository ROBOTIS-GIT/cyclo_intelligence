"""Bind TensorRT plans to checkpoint contents and the device that built them."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def load_tensorrt():
    """Use system TensorRT; repair the base Jetson image's empty DLA stub.

    The optional library is extracted from the working local GR00T image,
    without replacing host libraries or installing another TensorRT version.
    """
    try:
        import tensorrt as trt
    except ImportError as error:
        if "libnvdla_compiler.so" not in str(error):
            raise
        import ctypes
        path = Path(os.environ.get("CYCLO_TRT_DLA_COMPILER", "/workspace/runtime_vendor/jetson/libnvdla_compiler.so"))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Jetson TensorRT DLA dependency missing; run tools/prepare_tensorrt_runtime.sh") from error
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        import tensorrt as trt
    return trt


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(model_path):
    root = Path(model_path)
    assets = {"config.json", "processor_config.json", "statistics.json", "embodiment_id.json",
              "model.safetensors.index.json"}
    files = sorted({*root.glob("*.safetensors"),
                    *(p for p in root.glob("*.json") if p.name in assets
                      or p.name.startswith(("policy_", "preprocessor", "postprocessor")))})
    if not (root / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"Checkpoint config/weights missing: {root}")
    return {p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in files}


def runtime_identity():
    import torch
    trt = load_tensorrt()
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT requires a CUDA device")
    return {"tensorrt": trt.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability())}


def write_manifest(engine_path, payload):
    path = Path(str(engine_path) + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_manifest(model_path, engine_path, adapter):
    path = Path(str(engine_path) + ".json")
    if not path.is_file():
        raise RuntimeError(f"TensorRT manifest missing; rebuild engine: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "ready" or manifest.get("adapter") != adapter:
        raise RuntimeError("TensorRT engine is unverified or has the wrong model adapter; rebuild it")
    if manifest.get("checkpoint_identity") != checkpoint_identity(model_path):
        raise RuntimeError("TensorRT engine belongs to different/changed checkpoint contents; rebuild it")
    if manifest.get("runtime_identity") != runtime_identity():
        raise RuntimeError("TensorRT GPU/CUDA/version changed; rebuild engine on this device")
    if manifest.get("engine_sha256") != sha256_file(engine_path):
        raise RuntimeError("TensorRT engine contents changed; rebuild it")
    return manifest
