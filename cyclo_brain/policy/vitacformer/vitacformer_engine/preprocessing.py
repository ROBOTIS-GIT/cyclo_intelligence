"""Build the exact ViTacFormer SH5 inference observation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .constants import (
    CAMERA_NAME,
    IMAGE_KEY,
    JOINT_NAMES,
    STATE_HISTORY_HZ,
    STATE_HISTORY_SIZE,
    STATE_KEY,
    TACTILE_BATCH_KEY,
    TACTILE_HISTORY_HZ,
    TACTILE_HISTORY_SIZE,
)


def _prepare_image(
    image: np.ndarray,
    rotation_deg: int | float | None,
) -> np.ndarray:
    rotation = int(rotation_deg or 0) % 360
    if rotation == 90:
        image = np.rot90(image, k=3)
    elif rotation == 180:
        image = np.rot90(image, k=2)
    elif rotation == 270:
        image = np.rot90(image, k=1)
    elif rotation != 0:
        raise ValueError(f"unsupported camera rotation_deg={rotation_deg}")

    if tuple(image.shape[:2]) != (188, 336):
        import cv2

        image = cv2.resize(image, (336, 188))
    return np.ascontiguousarray(image)


class PreprocessingMixin:
    def _build_observation(self) -> Dict[str, torch.Tensor]:
        assert self._robot is not None
        assert self._policy is not None
        assert self._device is not None

        images = self._robot.get_images(format="rgb")
        image = images.get(CAMERA_NAME)
        if image is None:
            raise RuntimeError(f"Missing ViTacFormer camera frame: {CAMERA_NAME}")
        camera_cfg = self._robot._config.get("cameras", {}).get(CAMERA_NAME, {})
        image = _prepare_image(image, camera_cfg.get("rotation_deg", 0))
        image_tensor = (
            torch.from_numpy(image.copy())
            .to(torch.float32)
            .div_(255.0)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(self._device)
        )

        state = self._robot.get_joint_position_history(
            list(JOINT_NAMES),
            history_size=STATE_HISTORY_SIZE,
            sample_hz=STATE_HISTORY_HZ,
        )
        if tuple(state.shape) != (STATE_HISTORY_SIZE, len(JOINT_NAMES)):
            raise RuntimeError(
                "ViTacFormer state history must be (6, 54), got "
                f"{tuple(state.shape)}"
            )

        sides = {}
        for side, sensor_name in self._tactile_inputs.items():
            taxels = self._robot.get_tactile_taxel_history(
                sensor_name,
                history_size=TACTILE_HISTORY_SIZE,
                sample_hz=TACTILE_HISTORY_HZ,
            )
            if tuple(taxels.shape) != (TACTILE_HISTORY_SIZE, 5, 3, 3):
                raise RuntimeError(
                    f"ViTacFormer {side} tactile history must be "
                    f"(18, 5, 3, 3), got {tuple(taxels.shape)}"
                )
            flattened = np.asarray(taxels, dtype=np.float32).reshape(18, 45)
            baseline = np.asarray(
                self._tactile_baselines[side],
                dtype=np.float32,
            ).reshape(1, 45)
            sides[side] = np.maximum(flattened - baseline, 0.0)

        raw = np.concatenate([sides["left"], sides["right"]], axis=1)
        relative = raw - raw[[0]]
        tactile = np.concatenate([raw, relative], axis=1).astype(
            np.float32,
            copy=False,
        )
        if not (
            np.isfinite(state).all()
            and np.isfinite(tactile).all()
            and torch.isfinite(image_tensor).all()
        ):
            raise RuntimeError("ViTacFormer observation contains NaN or Inf")

        return {
            IMAGE_KEY: image_tensor,
            STATE_KEY: torch.from_numpy(
                np.ascontiguousarray(state, dtype=np.float32)
            ).unsqueeze(0).to(self._device),
            TACTILE_BATCH_KEY: torch.from_numpy(
                np.ascontiguousarray(tactile)
            ).unsqueeze(0).to(self._device),
        }
