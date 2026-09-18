"""LeRobot bindings and tensor operators for the model-independent input DAG."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from inference_inputs import Binding, Graph, query_from_config
from inference_inputs.graph import fields
from inference_inputs.operators import default_registry
from inference_inputs.memory import Memory, Condition, copy_cpu, register_memory
from inference_inputs.resources import Budget
from .image_preprocessing import _UniqueKeyLoader, apply_rotation
from .input_config import ImageOperations, build_input_graph


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "inference_inputs"
logger = logging.getLogger("lerobot_engine")


class PipelineProcessor:
    """Keep the saved processor public reset API for model-owned step adapters."""
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, observation):
        batch = self.engine._preprocessor(observation)
        evaluation = getattr(self.engine, "_input_evaluation", None)
        return evaluation.run("after", batch) if evaluation is not None else batch

    def reset(self):
        reset = getattr(self.engine._preprocessor, "reset", None)
        if callable(reset):
            reset()
        self.engine._input_evaluation = None

    @property
    def steps(self):
        return getattr(self.engine._preprocessor, "steps", ())


class InputPipelineConfig:
    @classmethod
    def from_user_config(cls, config, checkpoint, path, *, handlers=None):
        return cls(build_input_graph(config, checkpoint, handlers), checkpoint, path)

    def __init__(self, config, checkpoint, path):
        self.config = deepcopy(config)
        self.checkpoint = checkpoint
        self.path = path
        execution = config.get("execution", {})
        fields(execution, {"request_after"}, "execution options")
        self.request_after = Condition.parse(execution.get("request_after", {}))
        memory = config.get("memory", {})
        fields(memory, {"max_bytes", "slots"}, "memory options")
        def codec(value):
            if isinstance(value, torch.Tensor):
                copy = value.detach().clone()
                return copy, copy.numel() * copy.element_size()
            if isinstance(value, (tuple, list, dict)):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                copied, size = {}, 64
                for key, item in items:
                    result, used = codec(item)
                    copied[key], size = result, size + used + len(str(key))
                return (copied if isinstance(value, dict) else tuple(copied.values())), size
            return copy_cpu(value)
        self.budget = Budget(memory.get("max_bytes", 256 * 1024 * 1024))
        self.memory = Memory(memory.get("slots", {}), max_bytes=self.budget.limit, codec=codec, budget=self.budget)
        startup = config.get("startup", {"missing": "wait"})
        fields(startup, {"missing"}, "startup options")
        if startup.get("missing") not in {"wait", "error"}:
            raise ValueError("startup missing must be wait or error")
        self.startup = startup["missing"]

    @property
    def requires_feedback(self):
        return bool(self.memory.conditions) or self.request_after.event != "prediction_success" or any(
            "cache" in node for node in self.config.get("nodes", {}).values()
        )

    def compile(self, engine, max_age_s=None):
        config = deepcopy(self.config)

        def sample(source, sampling):
            return query_from_config({"source": source, "max_age_s": max_age_s, **sampling})

        def cameras(sampling):
            keys = list(engine._cameras.values())
            queries = tuple(sample(f"camera:{name}", sampling) for name in engine._cameras)
            # Explicit temporal camera construction uses an operator on each tuple.
            count = queries[0].sample_count if queries else 1
            return Binding(queries, lambda values: {
                key: values[i * count] if count == 1 else tuple(values[i * count:(i + 1) * count])
                for i, key in enumerate(keys)
            })

        def state(sampling):
            return Binding(tuple(sample("sensor:odom" if name == "mobile" else f"joint:follower_{name}", sampling)
                                 for name in engine._state_modalities), lambda values: tuple(values))

        registry = default_registry()
        register_memory(registry, self.memory)

        def to_tensor(options, context):
            fields(options, {"dtype", "device"}, "to_tensor")
            dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
                      "int64": torch.int64, "bool": torch.bool}
            if options.get("dtype") not in dtypes or options.get("device") not in {"cpu", "worker"}:
                raise ValueError("to_tensor requires an explicit dtype and device: cpu/worker")
            dtype = dtypes[options["dtype"]]
            device = engine._device if options["device"] == "worker" else torch.device("cpu")
            def run(values):
                if len(values) != 1:
                    raise ValueError("to_tensor expects one input")
                value = values[0].value
                tensor = value.clone() if isinstance(value, torch.Tensor) else torch.from_numpy(np.array(value, copy=True))
                return values[0].derived(tensor.to(device=device, dtype=dtype))
            return run
        registry.register("to_tensor", to_tensor)

        def tensor_join(kind):
            def compile_join(options, context):
                fields(options, {"axis"}, kind)
                axis = options.get("axis")
                if type(axis) is not int:
                    raise ValueError("tensor join requires an explicit axis")
                def run(values):
                    tensors = [value.value for value in values]
                    if not tensors or any(not isinstance(value, torch.Tensor) for value in tensors):
                        raise ValueError("tensor join requires tensors; declare to_tensor for raw data")
                    return (torch.stack if kind == "tensor_stack" else torch.cat)(tensors, dim=axis)
                return run
            return compile_join
        registry.register("tensor_stack", tensor_join("tensor_stack"))
        registry.register("tensor_concat", tensor_join("tensor_concat"))

        def mapping_op(name, allowed, factory):
            def compile_op(options, context):
                fields(options, allowed, name)
                operation = factory(options)
                def run(values):
                    if len(values) != 1 or not isinstance(values[0].value, dict):
                        raise ValueError(f"{name} expects a camera mapping")
                    semantics = dict(values[0].semantics)
                    if name == "image_prepare":
                        semantics.update(axes=["B", "C", "H", "W"], normalization="divide_by_255")
                    return values[0].derived({key: operation(key, value) for key, value in values[0].value.items()},
                                             semantics=semantics)
                return run
            registry.register(name, compile_op)

        def rotations(options):
            if options != {"from": "robot_config"}:
                raise ValueError("image_rotation requires from: robot_config")
            camera_config = engine._robot._config.get("cameras", {})
            resolved = {key: camera_config.get(camera, {}).get("rotation_deg", 0)
                        for camera, key in engine._cameras.items()}
            return lambda key, image: apply_rotation(image, resolved[key])

        mapping_op("image_rotation", {"from"}, rotations)

        def prepare(options):
            transform = ImageOperations(options["images"], options["cameras"],
                                        self.checkpoint.get("input_features", {}))
            return lambda key, image: transform.apply(image, key)
        mapping_op("image_prepare", {"images", "cameras"}, prepare)

        def device(options):
            if options:
                raise ValueError("image_device uses the selected Worker device without options")
            return lambda key, image: image.contiguous().to(engine._device)
        mapping_op("image_device", set(), device)

        def state_compat(options, context):
            if options != {"layout": "robot_config", "size": "checkpoint", "chunk_mismatch": "pad_or_truncate", "step_mismatch": "error"}:
                raise ValueError("legacy_state requires explicit robot/checkpoint layout and compatibility behavior")
            return lambda values: engine._transform_state(values[0].value)
        registry.register("legacy_state", state_compat)

        def task(options, context):
            if options:
                raise ValueError("task_batch has no options")
            return lambda values: [values[0].value or ""]
        registry.register("task_batch", task)

        bindings = {"robot_cameras": cameras, "robot_state": state}
        extensions = getattr(getattr(engine, "_adapter_definition", None), "input_extensions", None)
        if extensions is not None:
            extensions(registry, bindings, engine)
        return Graph(config, registry, bindings=bindings, budget=self.budget, codec=self.memory.codec)


def load_input_pipeline(model_path, config_dir=None):
    checkpoint = json.loads((Path(model_path) / "config.json").read_text())
    policy_type = checkpoint.get("type")
    if not isinstance(policy_type, str) or not re.fullmatch(r"[a-z0-9_]+", policy_type):
        raise ValueError("checkpoint must declare a valid policy type for input configuration")
    path = (Path(config_dir) if config_dir is not None else CONFIG_DIR) / f"{policy_type}.yaml"
    config = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    from .adapters import resolve_adapter
    definition = resolve_adapter(policy_type)
    pipeline = InputPipelineConfig.from_user_config(config, checkpoint, path, handlers=definition.input_handlers)
    if definition.input_extensions is None:
        # Validate standard recipes before any large model allocation. Extension
        # operators that bind model APIs are compiled once weights are available.
        cameras = {key.removeprefix("observation.images."): key for key in checkpoint.get("input_features", {})
                   if key.startswith("observation.images.")}
        probe = SimpleNamespace(_cameras=cameras, _state_modalities=["state"], _device=torch.device("cpu"),
                                _robot=SimpleNamespace(_config={}), _transform_state=lambda values: values,
                                _adapter_definition=definition)
        pipeline.compile(probe).close()
    logger.info("Cyclo inference inputs: %s (snapshot until Clear/LOAD)", path)
    return pipeline
