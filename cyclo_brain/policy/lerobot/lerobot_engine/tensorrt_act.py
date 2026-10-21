"""ACT neural network TensorRT adapter; processors and policy queues stay native."""
from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tensorrt_assets import checkpoint_identity, runtime_identity, sha256_file, validate_manifest, write_manifest, load_tensorrt


DEFAULT_ENGINE = "act_model_fp32.trt"
ADAPTER = "lerobot_act"


@contextmanager
def full_fp32_reference():
    """Compare FP32 TRT against FP32 PyTorch, excluding TF32 convolution rounding."""
    previous = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def input_spec(config):
    # ACT's own policy creates the ordered list of images from image_features.
    keys = [key for key in config.input_features if key not in config.image_features]
    keys += list(config.image_features)
    return [{"key": key, "name": f"input_{index}",
             "shape": [1, *config.input_features[key].shape]} for index, key in enumerate(keys)]


class ExportACT(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.spec = input_spec(config)
        self.image_keys = list(config.image_features)

    def forward(self, *inputs):
        batch = {item["key"]: tensor for item, tensor in zip(self.spec, inputs)}
        if self.image_keys:
            batch["observation.images"] = [batch[key] for key in self.image_keys]
        return self.model(batch)[0]


class TensorRTSession:
    """Single CUDA execution context; the engine prediction lock serializes use."""
    def __init__(self, path):
        trt = load_tensorrt()
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs):
        trt = self.trt
        inputs = {name: value.contiguous() for name, value in inputs.items()}
        outputs = {}
        for name in self.names:
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                value = inputs[name]
                if not value.is_cuda or value.dtype != dtype or tuple(value.shape) != shape:
                    raise ValueError(f"TensorRT input {name} needs CUDA {dtype} {shape}; got {value.dtype} {tuple(value.shape)}")
            else:
                value = torch.empty(shape, dtype=dtype, device="cuda")
                outputs[name] = value
            self.context.set_tensor_address(name, value.data_ptr())
        # Keep input tensors alive until the caller's stream completes. ACT's
        # synchronous postprocessor/CPU copy follows this inference call.
        self._inputs = inputs
        caller_stream = torch.cuda.current_stream()
        self.stream.wait_stream(caller_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        caller_stream.wait_stream(self.stream)
        for value in [*inputs.values(), *outputs.values()]:
            value.record_stream(self.stream)
        return outputs["actions"]


class TensorRTACT(nn.Module):
    def __init__(self, path, manifest, config):
        super().__init__()
        self.session = TensorRTSession(path)
        self.spec = manifest["inputs"]
        if self.spec != input_spec(config):
            raise RuntimeError("ACT TensorRT input schema differs from policy config")
        self.image_keys = list(config.image_features)

    def forward(self, batch):
        images = dict(zip(self.image_keys, batch.get("observation.images", [])))
        values = {item["name"]: images[item["key"]] if item["key"] in images else batch[item["key"]]
                  for item in self.spec}
        return self.session(values), (None, None)


def enable_tensorrt(policy, model_path, engine_path):
    manifest = validate_manifest(model_path, engine_path, ADAPTER)
    policy.model = TensorRTACT(engine_path, manifest, policy.config).eval()


def build_engine(policy, model_path, engine_path, workspace_mb=2048):
    trt = load_tensorrt()
    if workspace_mb <= 0:
        raise ValueError("workspace_mb must be positive")
    spec = input_spec(policy.config)
    wrapper = ExportACT(policy.model, policy.config).eval()
    generator = torch.Generator(device="cuda").manual_seed(726)
    samples = [tuple(torch.randn(item["shape"], generator=generator, device="cuda") for item in spec)
               for _ in range(3)]
    started = time.time()
    manifest = {"adapter": ADAPTER, "status": "building", "precision": "fp32",
                "model_path": str(model_path), "engine_path": str(engine_path),
                "inputs": spec, "workspace_mb": workspace_mb, "started_at": started,
                "runtime_identity": runtime_identity(), "checkpoint_identity": checkpoint_identity(model_path)}
    write_manifest(engine_path, manifest)
    try:
        with tempfile.TemporaryDirectory(prefix=".act_trt_export_", dir=Path(engine_path).parent) as temporary:
            onnx_path = str(Path(temporary) / "act.onnx")
            staged_path = Path(temporary) / DEFAULT_ENGINE
            previous_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                # Legacy ONNX constant folding creates CPU scalar constants.
                # Export on CPU to avoid mixed-device folding in torch 2.7;
                # build and numerical verification still use this CUDA device.
                wrapper.to("cpu")
                with torch.inference_mode():
                    torch.onnx.export(wrapper, tuple(value.cpu() for value in samples[0]), onnx_path, dynamo=False, opset_version=17,
                                      input_names=[item["name"] for item in spec], output_names=["actions"])
            finally:
                wrapper.to("cuda")
                torch.backends.mha.set_fastpath_enabled(previous_fastpath)
            logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(onnx_path):
                raise RuntimeError("ONNX import failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
            # FP32 provides a conservative first deployment baseline; disable
            # TF32 so the numerical gate compares like precision with PyTorch.
            config.clear_flag(trt.BuilderFlag.TF32)
            # Timing/compilation caches reuse tactics, never checkpoint weights.
            # Cache per network shape and device to shorten subsequent builds.
            schema = {"runtime": runtime_identity(), "shapes": [item["shape"] for item in spec],
                      "network": {key: getattr(policy.config, key, None) for key in
                                  ["chunk_size", "vision_backbone", "replace_final_stride_with_dilation",
                                   "pre_norm", "dim_model", "n_heads", "dim_feedforward", "feedforward_activation",
                                   "n_encoder_layers", "n_decoder_layers", "latent_dim"]}}
            cache_key = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:24]
            cache_root = Path(os.environ.get("CYCLO_TRT_TIMING_CACHE_DIR", "/workspace/runtime_vendor/tensorrt"))
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / f"act_{cache_key}.cache"
            cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
            if not config.set_timing_cache(cache, False):
                raise RuntimeError("TensorRT timing cache is incompatible with this device")
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("TensorRT builder did not produce an ACT engine")
            staged_path.write_bytes(bytes(plan))
            cache_temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
            cache_temporary.write_bytes(bytes(config.get_timing_cache().serialize()))
            os.replace(cache_temporary, cache_path)
            session = TensorRTSession(staged_path)
            verification = []
            with torch.inference_mode(), full_fp32_reference():
                for sample in samples:
                    reference = wrapper(*sample).float().cpu().numpy()
                    accelerated = session(dict(zip([item["name"] for item in spec], sample))).float().cpu().numpy()
                    if reference.shape != accelerated.shape or not np.isfinite(accelerated).all():
                        raise RuntimeError("ACT TensorRT output shape/nonfinite failure")
                    error = float(np.max(np.abs(reference - accelerated)))
                    if not np.allclose(reference, accelerated, rtol=1e-3, atol=1e-4):
                        raise RuntimeError(f"ACT TensorRT numerical validation failed: max_abs_error={error}")
                    verification.append({"max_abs_error": error, "shape": list(reference.shape),
                                         "reference": "pytorch_fp32_tf32_disabled"})
            # Verification succeeds before replacing an existing working plan.
            os.replace(staged_path, engine_path)
        manifest.update(status="ready", finished_at=time.time(), updated_at=time.time(),
                        verification=verification, engine_size_bytes=Path(engine_path).stat().st_size,
                        engine_sha256=sha256_file(engine_path), message="ACT TensorRT engine ready (FP32, 3 numerical checks)")
        write_manifest(engine_path, manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", finished_at=time.time(), message=str(error))
        write_manifest(engine_path, manifest)
        raise
