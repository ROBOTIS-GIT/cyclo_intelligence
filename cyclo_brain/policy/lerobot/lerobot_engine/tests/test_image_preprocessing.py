#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "image_preprocessing.py"
spec = importlib.util.spec_from_file_location("image_preprocessing", MODULE_PATH)
image_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_preprocessing)


class ImagePreprocessingTest(unittest.TestCase):
    def test_rotates_wrist_image_from_640x480_to_480x640(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        rotated = image_preprocessing.apply_rotation(image, 270)

        self.assertEqual(rotated.shape, (640, 480, 3))


KEY = "observation.images.head"
WRIST = "observation.images.wrist"
FEATURES = {KEY: {"shape": [3, 4, 6]}, WRIST: {"shape": [3, 4, 6]}}
IMAGE = np.random.default_rng(42).integers(0, 256, (11, 19, 3), dtype=np.uint8)


def pipeline(operations=None, backend="torch", **kwargs):
    return image_preprocessing.ImagePreprocessing(
        {
            "backend": backend,
            "operations": operations or [{"type": "identity"}],
            **kwargs,
        },
        FEATURES,
    )


def tensor(image):
    return torch.from_numpy(image.copy()).float().div(255).permute(2, 0, 1).unsqueeze(0)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_identity_only_rotates_and_converts(rotation):
    result = pipeline().apply(IMAGE, KEY, rotation)
    torch.testing.assert_close(
        result, tensor(np.rot90(IMAGE, -rotation // 90)), rtol=0, atol=0
    )
    assert result.is_contiguous()
    assert result.shape[-2:] != (4, 6)  # No hidden feature-shape resize.


@pytest.mark.parametrize("mode", ["area", "nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("backend", ["torch", "opencv"])
def test_resize_matches_explicit_library_and_numeric_order(mode, backend):
    result = pipeline(
        [{"type": "resize", "size": "checkpoint", "interpolation": mode}], backend
    ).apply(IMAGE, KEY)
    if backend == "torch":
        options = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
        expected = F.interpolate(tensor(IMAGE), (4, 6), mode=mode, **options)
    else:
        modes = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        expected = tensor(cv2.resize(IMAGE, (6, 4), interpolation=modes[mode]))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_torch_antialias_matches_training_transform():
    steps = [
        {
            "type": "resize",
            "size": [4, 6],
            "interpolation": "bilinear",
            "antialias": True,
        }
    ]
    result = pipeline(steps).apply(IMAGE, KEY)
    torch.testing.assert_close(
        result,
        F.interpolate(
            tensor(IMAGE), (4, 6), mode="bilinear", align_corners=False, antialias=True
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("backend", ["torch", "opencv"])
def test_center_crop_and_operation_order(backend):
    steps = [
        {"type": "center_crop", "size": [5, 7]},
        {"type": "resize", "size": [3, 4], "interpolation": "nearest"},
    ]
    result = pipeline(steps, backend).apply(IMAGE, KEY)
    expected = pipeline([steps[1]], backend).apply(IMAGE[3:8, 6:13], KEY)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["torch", "opencv"])
@pytest.mark.parametrize("placement", ["center", "top_left"])
def test_letterbox_preserves_aspect_and_padding_location(backend, placement):
    steps = [
        {
            "type": "letterbox",
            "size": [6, 6],
            "interpolation": "nearest",
            "placement": placement,
            "fill": 0,
        }
    ]
    result = pipeline(steps, backend).apply(np.full((2, 4, 3), 255, np.uint8), KEY)
    expected = torch.zeros(1, 3, 6, 6)
    top = 1 if placement == "center" else 0
    expected[:, :, top : top + 3, :] = 1
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_camera_override_replaces_not_appends_operations():
    transform = pipeline(
        [{"type": "resize", "size": [4, 6], "interpolation": "area"}],
        cameras={WRIST: [{"type": "identity"}]},
    )
    assert transform.apply(IMAGE, KEY).shape == (1, 3, 4, 6)
    assert transform.apply(IMAGE, WRIST).shape == (1, 3, 11, 19)


def test_center_crop_odd_offsets_match_torchvision_rounding():
    transform = pipeline([{"type": "center_crop", "size": [4, 6]}])
    # round((11-4)/2)=4, round((19-6)/2)=6, not floor(3.5)=3.
    torch.testing.assert_close(transform.apply(IMAGE, KEY), tensor(IMAGE[4:8, 6:12]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "config",
    [
        {"backend": "pillow"},
        {"unknown": True},
        {"cameras": {"typo": [{"type": "identity"}]}},
        {"operations": []},
        {"operations": [{"type": "random_crop", "size": [2, 2]}]},
        {"operations": [{"type": "resize", "size": [0, 4], "interpolation": "area"}]},
        {
            "operations": [
                {"type": "resize", "size": [True, 4], "interpolation": "area"}
            ]
        },
        {"operations": [{"type": "resize", "size": [4, 4]}]},
        {
            "operations": [
                {
                    "type": "resize",
                    "size": [4, 4],
                    "interpolation": "area",
                    "antialias": True,
                }
            ]
        },
        {"operations": [{"type": "identity", "size": [4, 4]}]},
        {
            "operations": [
                {"type": "center_crop", "size": [4, 4], "interpolation": "area"}
            ]
        },
    ],
)
def test_invalid_configuration_is_rejected(config):
    base = {"backend": "torch", "operations": [{"type": "identity"}]}
    with pytest.raises(ValueError):
        image_preprocessing.ImagePreprocessing({**base, **config}, FEATURES)


def test_rejects_opencv_antialias_and_bad_inputs():
    with pytest.raises(ValueError, match="antialias"):
        pipeline(
            [
                {
                    "type": "resize",
                    "size": [4, 4],
                    "interpolation": "bilinear",
                    "antialias": False,
                }
            ],
            "opencv",
        )
    with pytest.raises(ValueError, match="uint8"):
        pipeline().apply(IMAGE.astype(float), KEY)
    with pytest.raises(ValueError, match="exceeds"):
        pipeline([{"type": "center_crop", "size": [200, 200]}]).apply(IMAGE, KEY)
    with pytest.raises(ValueError, match="rotation"):
        pipeline().apply(IMAGE, KEY, 90.5)


def test_file_reload_snapshot_and_missing_invalid_files(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "act", "input_features": FEATURES})
    )
    path = tmp_path / "act.yaml"
    with pytest.raises(FileNotFoundError):
        image_preprocessing.load_image_preprocessing(tmp_path, tmp_path)
    path.write_text("backend: torch\noperations: [{type: identity}]\n")
    first = image_preprocessing.load_image_preprocessing(tmp_path, tmp_path)
    path.write_text(
        "backend: torch\noperations: [{type: resize, size: [4, 6], interpolation: area}]\n"
    )
    second = image_preprocessing.load_image_preprocessing(tmp_path, tmp_path)
    assert first.apply(IMAGE, KEY).shape != second.apply(IMAGE, KEY).shape
    path.write_text("backend: torch\nbackend: opencv\n")
    with pytest.raises(ValueError, match="duplicate"):
        image_preprocessing.load_image_preprocessing(tmp_path, tmp_path)


def test_all_catalog_policies_have_valid_defaults():
    root = MODULE_PATH.parents[1]
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    names = {
        name
        for model in manifest["models"]
        for name in [model["id"], *model.get("aliases", [])]
    }
    assert {p.stem for p in image_preprocessing.CONFIG_DIR.glob("*.yaml")} == names
    for name in names:
        config = yaml.safe_load(
            (image_preprocessing.CONFIG_DIR / f"{name}.yaml").read_text()
        )
        transform = image_preprocessing.ImagePreprocessing(config, FEATURES)
        assert transform.apply(IMAGE, KEY).shape[0:2] == (1, 3)


def test_vla_jepa_default_preserves_native_images_despite_checkpoint_size():
    config = yaml.safe_load((image_preprocessing.CONFIG_DIR / "vla_jepa.yaml").read_text())
    transform = image_preprocessing.ImagePreprocessing(config, FEATURES)
    torch.testing.assert_close(transform.apply(IMAGE, KEY), tensor(IMAGE), rtol=0, atol=0)
