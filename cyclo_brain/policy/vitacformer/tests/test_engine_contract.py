#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest
import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "vitacformer_engine"

from vitacformer_engine import constants, io_mapping, preprocessing, prediction


class _FakeRobot:
    _config = {"cameras": {constants.CAMERA_NAME: {"rotation_deg": 0}}}

    def get_images(self, format="rgb"):
        assert format == "rgb"
        return {
            constants.CAMERA_NAME: np.full(
                (188, 336, 3),
                255,
                dtype=np.uint8,
            )
        }

    def get_joint_position_history(self, joint_names, history_size, sample_hz):
        assert tuple(joint_names) == constants.JOINT_NAMES
        assert history_size == 6
        assert sample_hz == 10.0
        return np.arange(6 * 54, dtype=np.float32).reshape(6, 54)

    def get_tactile_taxel_history(self, sensor_name, history_size, sample_hz):
        assert history_size == 18
        assert sample_hz == 30.0
        start = 3.0 if sensor_name == "left_hand_pressure" else 5.0
        frames = np.arange(18, dtype=np.float32)[:, None, None, None] + start
        return np.broadcast_to(frames, (18, 5, 3, 3)).copy()


class _Preprocessor(preprocessing.PreprocessingMixin):
    def __init__(self):
        self._robot = _FakeRobot()
        self._policy = object()
        self._device = torch.device("cpu")
        self._tactile_inputs = {
            "left": "left_hand_pressure",
            "right": "right_hand_pressure",
        }
        self._tactile_baselines = {
            "left": np.full((5, 3, 3), 2.0, dtype=np.float32),
            "right": np.full((5, 3, 3), 4.0, dtype=np.float32),
        }


def test_preprocessing_builds_strict_vitacformer_shapes_and_representation():
    batch = _Preprocessor()._build_observation()

    assert tuple(batch[constants.IMAGE_KEY].shape) == (1, 3, 188, 336)
    assert tuple(batch[constants.STATE_KEY].shape) == (1, 6, 54)
    assert tuple(batch[constants.TACTILE_BATCH_KEY].shape) == (1, 18, 180)
    assert torch.all(batch[constants.IMAGE_KEY] == 1.0)

    tactile = batch[constants.TACTILE_BATCH_KEY][0].numpy()
    # Both hands begin one count above their episode baseline.
    np.testing.assert_allclose(tactile[0, :90], 1.0)
    np.testing.assert_allclose(tactile[0, 90:], 0.0)
    # The second half is each raw channel relative to the oldest frame.
    np.testing.assert_allclose(tactile[-1, 90:], 17.0)


def test_tactile_sensor_resolution_is_fail_closed():
    resolve = io_mapping.IoMappingMixin._resolve_tactile_inputs
    assert resolve(["left_hand_pressure", "right_hand_pressure"]) == {
        "left": "left_hand_pressure",
        "right": "right_hand_pressure",
    }
    with pytest.raises(RuntimeError, match="one left tactile sensor"):
        resolve(["right_hand_pressure"])


def test_prediction_returns_only_a_finite_100_by_54_chunk():
    class _Policy(torch.nn.Module):
        def __init__(self, rows=100):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.rows = rows

        def predict_action_chunk(self, _batch):
            return torch.zeros((1, self.rows, 54), dtype=torch.float32)

    runner = prediction.PredictionMixin()
    runner._policy = _Policy()
    result = runner._predict_chunk({})
    assert result.shape == (100, 54)
    assert result.dtype == np.float64

    runner._policy = _Policy(rows=99)
    with pytest.raises(RuntimeError, match="1, 100, 54"):
        runner._predict_chunk({})
