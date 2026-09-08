#!/usr/bin/env python3
# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0

"""Explicit Cyclo spatial transforms, preceding the saved LeRobot processor."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

IMAGE_KEY_PREFIX = "observation.images."
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "image_preprocessing"
logger = logging.getLogger("lerobot_engine")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Non-string or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Invalid {label}; allowed keys: {sorted(allowed)}")


def _size(value, feature):
    if value == "checkpoint":
        shape = (
            feature.get("shape")
            if isinstance(feature, dict)
            else getattr(feature, "shape", None)
        )
        if shape is None or len(shape) != 3 or shape[0] != 3:
            raise ValueError("size: checkpoint requires an RGB (3, H, W) input feature")
        value = list(shape[1:])
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(v) is not int or v <= 0 for v in value)
    ):
        raise ValueError(
            "size must be [height, width] with positive integers, or checkpoint"
        )
    return tuple(value)


def _steps(operations, backend, feature):
    if not isinstance(operations, list) or not operations:
        raise ValueError(
            "operations must be a nonempty list (use type: identity for no transform)"
        )
    result = []
    for operation in operations:
        _fields(
            operation,
            {"type", "size", "interpolation", "antialias", "placement", "fill"},
            "operation",
        )
        kind = operation.get("type")
        if kind == "identity":
            if len(operation) != 1 or len(operations) != 1:
                raise ValueError(
                    "identity must be the only operation and have no options"
                )
            result.append({"type": kind})
            continue
        allowed = {"type", "size"}
        if kind in {"resize", "letterbox"}:
            allowed |= {"interpolation", "antialias"}
        elif kind != "center_crop":
            raise ValueError(f"Unknown operation type: {kind!r}")
        if kind == "letterbox":
            allowed |= {"placement", "fill"}
        _fields(operation, allowed, kind)
        step = {"type": kind, "size": _size(operation.get("size"), feature)}
        if kind in {"resize", "letterbox"}:
            mode = operation.get("interpolation")
            if mode not in {"nearest", "bilinear", "bicubic", "area"}:
                raise ValueError(
                    "resize/letterbox requires interpolation: nearest, bilinear, bicubic or area"
                )
            antialias = operation.get("antialias", False)
            if type(antialias) is not bool or (
                "antialias" in operation and backend != "torch"
            ):
                raise ValueError(
                    "antialias is a boolean option for backend: torch only"
                )
            if antialias and mode not in {"bilinear", "bicubic"}:
                raise ValueError("antialias requires bilinear or bicubic")
            step.update(interpolation=mode, antialias=antialias)
        if kind == "letterbox":
            placement = operation.get("placement", "center")
            fill = operation.get("fill", 0)
            if (
                placement not in {"center", "top_left"}
                or type(fill) is not int
                or not 0 <= fill <= 255
            ):
                raise ValueError(
                    "letterbox needs placement: center/top_left and integer fill in [0, 255]"
                )
            step.update(placement=placement, fill=fill)
        result.append(step)
    return tuple(result)


class ImagePreprocessing:
    """LOAD-time snapshot. YAML edits never change an active inference session."""

    def __init__(self, config, features):
        _fields(
            config,
            {"backend", "operations", "cameras"},
            "image preprocessing config",
        )
        self.backend = config.get("backend")
        if self.backend not in {"torch", "opencv"}:
            raise ValueError("backend must be torch or opencv")
        cameras = config.get("cameras", {})
        image_features = {
            k: v for k, v in features.items() if k.startswith(IMAGE_KEY_PREFIX)
        }
        _fields(
            cameras,
            set(image_features),
            "camera overrides (use checkpoint observation keys)",
        )
        # Validate the default even if every camera supplies an override.
        _steps(config.get("operations"), self.backend, {"shape": [3, 1, 1]})
        self.operations = {
            key: _steps(cameras.get(key, config["operations"]), self.backend, feature)
            for key, feature in image_features.items()
        }

    def apply(self, image, policy_key, rotation_deg=0):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("Expected RGB uint8 image with shape (H, W, 3)")
        image = np.ascontiguousarray(apply_rotation(image, rotation_deg))
        if not image.shape[0] or not image.shape[1]:
            raise ValueError("Empty camera frame")
        value = _tensor(image) if self.backend == "torch" else image
        for step in self.operations[policy_key]:
            kind = step["type"]
            if kind == "identity":
                continue
            h, w = value.shape[-2:] if self.backend == "torch" else value.shape[:2]
            target_h, target_w = step["size"]
            if kind == "center_crop":
                if target_h > h or target_w > w:
                    raise ValueError(
                        f"center_crop {step['size']} exceeds input {(h, w)}"
                    )
                # Match torchvision's center-crop rounding for odd differences.
                top, left = round((h - target_h) / 2), round((w - target_w) / 2)
                value = (
                    value[..., top : top + target_h, left : left + target_w]
                    if self.backend == "torch"
                    else value[top : top + target_h, left : left + target_w]
                )
                continue
            size = (target_h, target_w)
            if kind == "letterbox":
                ratio = min(target_h / h, target_w / w)
                size = (
                    max(1, min(target_h, round(h * ratio))),
                    max(1, min(target_w, round(w * ratio))),
                )
            value = self._resize(value, size, step)
            if kind == "letterbox":
                dh, dw = target_h - size[0], target_w - size[1]
                top, left = (
                    (dh // 2, dw // 2) if step["placement"] == "center" else (0, 0)
                )
                if self.backend == "torch":
                    value = F.pad(
                        value,
                        (left, dw - left, top, dh - top),
                        value=step["fill"] / 255.0,
                    )
                else:
                    canvas = np.full(
                        (target_h, target_w, 3), step["fill"], dtype=np.uint8
                    )
                    canvas[top : top + size[0], left : left + size[1]] = value
                    value = canvas
        return value.contiguous() if self.backend == "torch" else _tensor(value)

    def _resize(self, value, size, step):
        if self.backend == "torch":
            options = {}
            if step["interpolation"] in {"bilinear", "bicubic"}:
                options = {"align_corners": False, "antialias": step["antialias"]}
            return F.interpolate(
                value, size=size, mode=step["interpolation"], **options
            )
        import cv2

        modes = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        return cv2.resize(
            value, (size[1], size[0]), interpolation=modes[step["interpolation"]]
        )


def _tensor(image):
    return (
        torch.from_numpy(image.copy())
        .to(torch.float32)
        .div_(255)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


def load_image_preprocessing(model_path, config_dir=None):
    with (Path(model_path) / "config.json").open() as stream:
        checkpoint = json.load(stream)
    policy_type = checkpoint.get("type")
    if not isinstance(policy_type, str) or not re.fullmatch(r"[a-z0-9_]+", policy_type):
        raise ValueError(
            "Checkpoint must declare a valid policy type for image preprocessing"
        )
    path = (
        Path(config_dir) if config_dir is not None else CONFIG_DIR
    ) / f"{policy_type}.yaml"
    with path.open() as stream:
        config = yaml.load(stream, Loader=_UniqueKeyLoader)
    pipeline = ImagePreprocessing(config, checkpoint.get("input_features", {}))
    logger.info(
        "Cyclo image preprocessing: file=%s backend=%s cameras=%s; saved processor and model transforms remain active",
        path,
        pipeline.backend,
        json.dumps(pipeline.operations, sort_keys=True),
    )
    return pipeline


def apply_rotation(image, rotation_deg=0):
    rotation = float(rotation_deg or 0) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"unsupported camera rotation_deg={rotation_deg}")
    return np.rot90(image, k=-int(rotation // 90)) if rotation else image
