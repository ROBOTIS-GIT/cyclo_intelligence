"""Contract tests against the unmodified LeRobot 0.5.2 ACTPolicy."""

import tempfile
import unittest
from pathlib import Path

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor.pipeline import ProcessorMigrationError
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from cyclo_brain.model.act import (
    compute_act_bc_loss,
    create_act_model,
    differentiable_act_action_chunk,
    load_act_model,
    load_act_policy_assets,
    predict_act_action_chunk,
)


def _tiny_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        chunk_size=3,
        n_action_steps=3,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        dropout=0.0,
        pretrained_backbone_weights=None,
        device="cpu",
    )


def _observation_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.tensor(
            [[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32
        )[:batch_size],
        OBS_ENV_STATE: torch.tensor(
            [[-0.4, 0.5], [0.6, -0.7]], dtype=torch.float32
        )[:batch_size],
    }


def _tiny_visual_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            f"{OBS_IMAGES}.left": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 32, 32)
            ),
            f"{OBS_IMAGES}.right": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 32, 32)
            ),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        chunk_size=3,
        n_action_steps=3,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        dropout=0.0,
        pretrained_backbone_weights=None,
        device="cpu",
    )


def _bc_batch() -> dict[str, torch.Tensor]:
    return {
        **_observation_batch(),
        ACTION: torch.tensor(
            [
                [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
                [[-0.1, 0.0], [0.1, 0.2], [0.3, 0.4]],
            ],
            dtype=torch.float32,
        ),
        "action_is_pad": torch.tensor(
            [[False, False, False], [False, False, True]]
        ),
    }


def _normalization_stats(
    *,
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        ACTION: {
            "mean": (
                torch.tensor([0.25, -0.5], dtype=torch.float32)
                if action_mean is None
                else action_mean
            ),
            "std": (
                torch.tensor([0.75, 1.5], dtype=torch.float32)
                if action_std is None
                else action_std
            ),
        },
        OBS_STATE: {
            "mean": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "std": torch.tensor([1.0, 1.0], dtype=torch.float32),
        },
        OBS_ENV_STATE: {
            "mean": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "std": torch.tensor([1.0, 1.0], dtype=torch.float32),
        },
    }


def _save_policy_and_processors(
    checkpoint: Path,
    config: ACTConfig,
    *,
    pre_stats: dict[str, dict[str, torch.Tensor]],
    post_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    policy = create_act_model(config).eval()
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=pre_stats)
    _, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=pre_stats if post_stats is None else post_stats,
    )
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)


class ACTModelContractTest(unittest.TestCase):
    def test_fresh_model_is_placed_on_configured_device(self):
        policy = create_act_model(_tiny_config())

        self.assertEqual(next(policy.parameters()).device, torch.device("cpu"))

    def test_official_policy_computes_bc_loss_and_gradients(self):
        policy = create_act_model(_tiny_config())
        policy.train()

        loss, metrics = compute_act_bc_loss(policy, _bc_batch())
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(set(metrics), {"l1_loss", "kld_loss"})
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in policy.parameters()
            )
        )

    def test_differentiable_chunk_matches_official_inference(self):
        torch.manual_seed(7)
        policy = create_act_model(_tiny_config()).eval()
        observation = _observation_batch()

        official = predict_act_action_chunk(policy, observation)
        differentiable = differentiable_act_action_chunk(policy, observation)

        torch.testing.assert_close(differentiable, official, rtol=0.0, atol=0.0)
        self.assertFalse(official.requires_grad)
        self.assertTrue(differentiable.requires_grad)
        differentiable.sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool((parameter.grad != 0).any())
                for parameter in policy.parameters()
            )
        )

    def test_multicamera_packing_matches_official_inference(self):
        torch.manual_seed(13)
        policy = create_act_model(_tiny_visual_config()).eval()
        observation = {
            OBS_STATE: torch.tensor([[0.1, -0.2]], dtype=torch.float32),
            f"{OBS_IMAGES}.left": torch.zeros(1, 3, 32, 32),
            f"{OBS_IMAGES}.right": torch.ones(1, 3, 32, 32),
        }

        official = predict_act_action_chunk(policy, observation)
        differentiable = differentiable_act_action_chunk(policy, observation)

        self.assertEqual(
            list(policy.config.image_features),
            [f"{OBS_IMAGES}.left", f"{OBS_IMAGES}.right"],
        )
        torch.testing.assert_close(differentiable, official, rtol=0.0, atol=0.0)

    def test_differentiable_chunk_is_deterministic_across_rng_seeds(self):
        policy = create_act_model(_tiny_config()).eval()
        observation = _observation_batch()

        torch.manual_seed(1)
        first = differentiable_act_action_chunk(policy, observation)
        torch.manual_seed(2)
        second = differentiable_act_action_chunk(policy, observation)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_differentiable_chunk_requires_explicit_eval_mode(self):
        policy = create_act_model(_tiny_config())

        with self.assertRaisesRegex(RuntimeError, r"policy\.eval\(\)"):
            differentiable_act_action_chunk(policy, _observation_batch())

    def test_bc_loss_requires_explicit_train_mode(self):
        policy = create_act_model(_tiny_config()).eval()

        with self.assertRaisesRegex(RuntimeError, r"policy\.train\(\)"):
            compute_act_bc_loss(policy, _bc_batch())

    def test_checkpoint_round_trip_preserves_official_inference(self):
        torch.manual_seed(11)
        policy = create_act_model(_tiny_config()).eval()
        observation = _observation_batch(batch_size=1)
        expected = predict_act_action_chunk(policy, observation)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            policy.save_pretrained(checkpoint)
            restored = load_act_model(checkpoint, device="cpu", strict=True)

        actual = predict_act_action_chunk(restored, observation)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertFalse(restored.training)

    def test_asset_round_trip_loads_saved_processors_and_action_stats(self):
        policy = create_act_model(_tiny_config()).eval()
        stats = _normalization_stats()
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=stats,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            policy.save_pretrained(checkpoint)
            preprocessor.save_pretrained(checkpoint)
            postprocessor.save_pretrained(checkpoint)
            assets = load_act_policy_assets(checkpoint, device="cpu")

        torch.testing.assert_close(assets.action_mean, stats[ACTION]["mean"])
        torch.testing.assert_close(assets.action_std, stats[ACTION]["std"])
        self.assertEqual(assets.action_dim, 2)
        self.assertEqual(assets.normalizer_eps, 1e-8)
        self.assertFalse(assets.policy.training)
        self.assertEqual(next(assets.policy.parameters()).device, torch.device("cpu"))

        exposed_mean = assets.action_mean
        exposed_mean.zero_()
        torch.testing.assert_close(assets.action_mean, stats[ACTION]["mean"])

    def test_asset_loader_fails_fast_on_missing_saved_processors(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            create_act_model(_tiny_config()).save_pretrained(checkpoint)

            with self.assertRaises(ProcessorMigrationError):
                load_act_policy_assets(checkpoint, device="cpu")

    def test_asset_loader_fails_fast_on_processor_stat_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            _save_policy_and_processors(
                checkpoint,
                _tiny_config(),
                pre_stats=_normalization_stats(),
                post_stats=_normalization_stats(
                    action_mean=torch.tensor([0.25, -0.25], dtype=torch.float32)
                ),
            )

            with self.assertRaisesRegex(ValueError, "mean statistics do not match"):
                load_act_policy_assets(checkpoint, device="cpu")

    def test_asset_loader_fails_fast_on_non_mean_std_action_mode(self):
        config = _tiny_config()
        config.normalization_mapping[FeatureType.ACTION] = NormalizationMode.MIN_MAX
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            _save_policy_and_processors(
                checkpoint,
                config,
                pre_stats=_normalization_stats(),
            )

            with self.assertRaisesRegex(ValueError, "ACTION normalization must be MEAN_STD"):
                load_act_policy_assets(checkpoint, device="cpu")

    def test_asset_loader_fails_fast_on_action_stat_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            _save_policy_and_processors(
                checkpoint,
                _tiny_config(),
                pre_stats=_normalization_stats(
                    action_mean=torch.tensor([0.25], dtype=torch.float32),
                    action_std=torch.tensor([0.75], dtype=torch.float32),
                ),
            )

            with self.assertRaisesRegex(ValueError, r"shape \(2,\)"):
                load_act_policy_assets(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
