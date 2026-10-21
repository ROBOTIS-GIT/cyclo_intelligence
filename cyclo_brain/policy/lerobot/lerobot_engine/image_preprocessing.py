#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Image shape helpers for LeRobot inference."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


IMAGE_KEY_PREFIX = "observation.images."


def infer_image_resize_targets(
    input_features: Mapping[str, Any],
) -> Dict[str, Tuple[int, int]]:
    """Return per-policy-key resize targets as ``(width, height)``."""
    targets: Dict[str, Tuple[int, int]] = {}
    for key, feature in (input_features or {}).items():
        if not str(key).startswith(IMAGE_KEY_PREFIX):
            continue
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, Mapping):
            shape = feature.get("shape")
        if shape and len(shape) == 3:
            _channels, height, width = shape
            targets[str(key)] = (int(width), int(height))
    return targets


def apply_rotation(image: np.ndarray, rotation_deg: int | float | None) -> np.ndarray:
    """Apply camera rotation metadata using the same direction as OpenCV."""
    rotation = int(rotation_deg or 0) % 360
    if rotation == 0:
        return image
    if rotation == 90:
        return np.rot90(image, k=3)
    if rotation == 180:
        return np.rot90(image, k=2)
    if rotation == 270:
        return np.rot90(image, k=1)
    raise ValueError(f"unsupported camera rotation_deg={rotation_deg}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Resize to ``(width, height)`` if requested."""
    if not target_size:
        return image
    width, height = target_size
    if image.shape[1] == width and image.shape[0] == height:
        return image

    import cv2

    return cv2.resize(image, (width, height))


def prepare_policy_image(
    image: np.ndarray,
    *,
    rotation_deg: int | float | None = 0,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Rotate first, then resize to the policy feature's expected shape."""
    return resize_image(apply_rotation(image, rotation_deg), target_size)


def prepare_letterbox_tensor(image, *, rotation_deg=0, target_size):
    """RGB uint8 -> BCHW float, bilinear antialiased resize + centered zero pad.

    Padding is applied before the policy's normalization, matching the published
    ACT training contract. Round resized dimensions rather than truncating them.
    """
    import torch
    import torch.nn.functional as F

    image = apply_rotation(image, rotation_deg)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    width, height = target_size
    source_h, source_w = image.shape[:2]
    scale = min(height / source_h, width / source_w)
    resized_h, resized_w = round(source_h * scale), round(source_w * scale)
    tensor = F.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear",
                           align_corners=False, antialias=True)
    left, top = (width - resized_w) // 2, (height - resized_h) // 2
    return F.pad(tensor, (left, width - resized_w - left, top, height - resized_h - top))
