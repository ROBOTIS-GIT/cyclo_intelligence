"""Host-only tests for strict MultiTaskDiT checkpoint validation."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cyclo_brain.model.multi_task_dit.checkpoint_validation import (
    DEFAULT_DEPLOYMENT_CONTRACT,
    assert_exact_state_dict,
    resolve_pretrained_model_dir,
    validate_checkpoint_round_trip,
    validate_policy_contract,
)
from cyclo_brain.model.multi_task_dit.flow_sde_adapter import CYCLO_SG2_CAMERA_KEYS


class _Feature:
    def __init__(self, shape):
        self.shape = shape


class _Config:
    image_features = {key: _Feature((3, 8, 8)) for key in CYCLO_SG2_CAMERA_KEYS}
    robot_state_feature = _Feature((22,))
    action_feature = _Feature((22,))
    horizon = 16
    n_obs_steps = 1
    n_action_steps = 16
    objective = "flow_matching"
    sigma_min = 0.0


class _ObservationEncoder(nn.Module):
    conditioning_dim = 5

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, batch):
        return batch["conditioning"] * self.scale


class _NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, latent, progress, conditioning):
        return latent * self.scale + progress[:, None, None] + conditioning[:, :1, None]


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.observation_encoder = _ObservationEncoder()
        self.noise_predictor = _NoisePredictor()

    def _prepare_batch(self, batch):
        return batch


def _write_artifacts(root: Path) -> Path:
    pretrained = root / "pretrained_model"
    pretrained.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (pretrained / name).touch()
    return pretrained


class CheckpointValidationTest(unittest.TestCase):
    def test_contract_requires_canonical_camera_and_22d_chunk(self):
        summary = validate_policy_contract(_Policy())
        self.assertEqual(summary["camera_keys"], list(CYCLO_SG2_CAMERA_KEYS))
        self.assertEqual(summary["state_dim"], 22)
        self.assertEqual(summary["action_dim"], 22)
        self.assertEqual(summary["horizon"], 16)

        policy = _Policy()
        policy.config.image_features = dict(reversed(policy.config.image_features.items()))
        with self.assertRaisesRegex(ValueError, "camera order"):
            validate_policy_contract(policy)

    def test_exact_state_dict_rejects_changed_value(self):
        reference = {"weight": torch.tensor([1.0, 2.0])}
        candidate = {"weight": torch.tensor([1.0, 3.0])}
        with self.assertRaisesRegex(AssertionError, "changed"):
            assert_exact_state_dict(reference, candidate)

    def test_round_trip_checks_weights_velocity_and_processors(self):
        reference = _Policy()

        def load_policy(_path, source):
            loaded = _Policy()
            loaded.load_state_dict(source.state_dict(), strict=True)
            return loaded

        preprocessor = lambda batch: {"state": batch["state"] + 1.0}  # noqa: E731
        postprocessor = lambda action: action * 2.0  # noqa: E731

        def load_processors(_path, _policy):
            return preprocessor, postprocessor

        with tempfile.TemporaryDirectory() as directory:
            pretrained = _write_artifacts(Path(directory))
            self.assertEqual(resolve_pretrained_model_dir(Path(directory)), pretrained.resolve())
            result = validate_checkpoint_round_trip(
                reference,
                Path(directory),
                contract=DEFAULT_DEPLOYMENT_CONTRACT,
                preprocessor=preprocessor,
                raw_batch={"state": torch.tensor([[1.0, 2.0]])},
                postprocessor=postprocessor,
                normalized_action=torch.tensor([[0.25, -0.5]]),
                policy_loader=load_policy,
                processor_loader=load_processors,
            )

        self.assertEqual(result.state_tensor_count, len(reference.state_dict()))
        self.assertEqual(result.velocity_max_abs_error, 0.0)
        self.assertTrue(result.preprocessor_checked)
        self.assertTrue(result.postprocessor_checked)

    def test_processor_inputs_must_be_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_artifacts(Path(directory))
            with self.assertRaisesRegex(ValueError, "preprocessor and raw_batch"):
                validate_checkpoint_round_trip(
                    _Policy(),
                    directory,
                    preprocessor=lambda value: value,
                    policy_loader=lambda _path, source: source,
                )


if __name__ == "__main__":
    unittest.main()
