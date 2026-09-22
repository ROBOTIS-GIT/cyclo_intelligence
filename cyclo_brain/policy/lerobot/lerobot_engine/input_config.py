"""User preprocessing settings translated to a private LeRobot input graph.

Handlers are trusted graph builders registered by the model adapter. They own
their option schema, execution prerequisites and transactional memory recipe.
"""

from copy import deepcopy

import numpy as np
import torch

from inference_inputs.graph import fields
from .image_preprocessing import ImagePreprocessing, IMAGE_KEY_PREFIX, _tensor


class ImageOperations:
    """Ordered spatial transforms, retaining uint8/OpenCV and float/Torch semantics."""

    def __init__(self, images, cameras, features):
        image_features = {k: v for k, v in features.items() if k.startswith(IMAGE_KEY_PREFIX)}
        fields(cameras, set(image_features), "camera overrides (checkpoint observation keys)")
        # Validate defaults even when no camera uses them.
        self._compile(images, {"shape": [3, 1, 1]})
        self.operations = {key: self._compile(cameras.get(key, images), feature)
                           for key, feature in image_features.items()}

    @staticmethod
    def _compile(operations, feature):
        if operations == "identity":
            operations = []
        if not isinstance(operations, list):
            raise ValueError("images must be identity or an ordered transform list")
        compiled = []
        tensor = False
        key = IMAGE_KEY_PREFIX + "input"
        for item in operations:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError("each image transform must contain one operation")
            kind, options = next(iter(item.items()))
            if kind not in {"resize", "center_crop", "letterbox"} or not isinstance(options, dict):
                raise ValueError(f"unsupported image transform: {kind!r}")
            options = deepcopy(options)
            backend = options.pop("backend", None)
            if backend not in {"opencv", "torch"}:
                raise ValueError("each image transform requires backend: opencv or torch")
            if "type" in options:
                raise ValueError("transform type is declared by its key, not an option")
            if tensor and backend == "opencv":
                raise ValueError("OpenCV cannot follow Torch: implicit float-to-uint8 conversion is forbidden")
            tensor |= backend == "torch"
            operation = ImagePreprocessing(
                {"backend": backend, "operations": [{"type": kind, **options}]}, {key: feature},
            )
            compiled.append((backend, operation, key))
        return tuple(compiled)

    def apply(self, image, policy_key):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]):
            raise ValueError("Expected nonempty RGB uint8 HWC camera input")
        value = image
        for backend, transform, key in self.operations[policy_key]:
            if backend == "torch" and not isinstance(value, torch.Tensor):
                value = _tensor(value)
            value = transform.apply_spatial(value, key)
        return value.contiguous() if isinstance(value, torch.Tensor) else _tensor(value)


def build_input_graph(config, checkpoint, handlers=None):
    fields(config, {"preprocessing"}, "input settings; use preprocessing, not sources/nodes/outputs")
    if "preprocessing" not in config:
        raise ValueError("input settings require preprocessing: identity or a mapping")
    preprocessing = config["preprocessing"]
    if preprocessing == "identity":
        preprocessing = {}
    fields(preprocessing, {"images", "cameras", "custom"}, "preprocessing")
    images = deepcopy(preprocessing.get("images", "identity"))
    cameras = deepcopy(preprocessing.get("cameras", {}))
    ImageOperations(images, cameras, checkpoint.get("input_features", {}))
    graph = {
        "sources": {
            "images": {"binding": "robot_cameras"},
            "joints": {"binding": "robot_state"},
            "instruction": {"source": "instruction"},
        },
        "nodes": {
            "rotated": {"op": "image_rotation", "inputs": ["images"], "options": {"from": "robot_config"}},
            "prepared": {"op": "image_prepare", "inputs": ["rotated"],
                         "options": {"images": images, "cameras": cameras}},
            "images_ready": {"op": "image_device", "inputs": ["prepared"]},
            "state": {"op": "mapped_state", "inputs": ["joints"]},
            "task": {"op": "task_batch", "inputs": ["instruction"]},
        },
        "outputs": {"before": {"*": "images_ready", "observation.state": "state", "task": "task"},
                    "after": {"*": "processed"}},
    }
    if "custom" in preprocessing:
        custom = preprocessing["custom"]
        fields(custom, {"handler", "options"}, "custom preprocessing")
        name = custom.get("handler")
        if not isinstance(name, str) or name not in (handlers or {}):
            raise ValueError(f"unregistered preprocessing handler for this model: {name!r}")
        options = custom.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("custom handler options must be a mapping")
        graph = handlers[name](deepcopy(graph), deepcopy(options))
    fields(graph, {"sources", "nodes", "outputs", "execution", "memory", "startup"}, "internal input graph")
    return graph
