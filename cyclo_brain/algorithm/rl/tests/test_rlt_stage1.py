"""CPU tests for the paper-faithful RLT Stage-1 training boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cyclo_brain.algorithm.rl.rlt import (
    RLTokenAutoencoder,
    RLTokenConfig,
    RLTokenForward,
    RLTokenStage1Config,
    RLTokenStage1Trainer,
    load_frozen_rl_token_encoder,
    rl_token_reconstruction_loss,
)


def _model_config() -> RLTokenConfig:
    return RLTokenConfig(
        embedding_dim=8,
        max_tokens=4,
        num_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dim=16,
        dropout=0.0,
        token_selection="image",
    )


def _representation_contract() -> dict[str, object]:
    return {
        "format": "groot-final-layer-tokens/v1",
        "embeddings": {"width": 8, "dtype": "float32"},
        "token_selection": "image",
    }


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    tokens = torch.randn(2, 4, 8, generator=generator)
    token_valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
        dtype=torch.bool,
    )
    image_token = torch.tensor(
        [[True, False, True, False], [False, True, True, False]],
        dtype=torch.bool,
    )
    return tokens, token_valid, image_token


class RLTokenStage1Test(unittest.TestCase):
    def test_autoencoder_selects_image_tokens_and_uses_paper_loss(self) -> None:
        model = RLTokenAutoencoder(_model_config())
        tokens, token_valid, image_token = _batch()

        output = model(tokens, token_valid, image_token)

        self.assertEqual(tuple(output.z_rl.shape), (2, 8))
        self.assertEqual(tuple(output.target.shape), (2, 2, 8))
        torch.testing.assert_close(output.target[0, 0], tokens[0, 0])
        torch.testing.assert_close(output.target[0, 1], tokens[0, 2])
        torch.testing.assert_close(output.target[1, 0], tokens[1, 1])
        torch.testing.assert_close(output.target[1, 1], tokens[1, 2])
        self.assertFalse(output.target.requires_grad)

        synthetic = RLTokenForward(
            z_rl=torch.zeros(2, 8),
            reconstruction=torch.zeros(2, 2, 8),
            target=torch.tensor(
                [
                    [[1.0] * 8, [2.0] * 8],
                    [[3.0] * 8, [9.0] * 8],
                ]
            ),
            target_valid=torch.tensor(
                [[True, True], [True, False]],
                dtype=torch.bool,
            ),
        )
        metrics = rl_token_reconstruction_loss(synthetic)
        # Equation (2): mean over samples of the sum over valid token SSE.
        self.assertEqual(metrics.valid_tokens, 3)
        self.assertAlmostEqual(float(metrics.loss), (40.0 + 72.0) / 2.0)
        self.assertAlmostEqual(float(metrics.element_mse), 112.0 / 24.0, places=6)

    def test_trainer_owns_only_encoder_decoder_and_rejects_live_vla_graph(self) -> None:
        torch.manual_seed(3)
        frozen_groot = nn.Linear(8, 8)
        frozen_groot.requires_grad_(False)
        groot_before = {
            name: value.detach().clone()
            for name, value in frozen_groot.state_dict().items()
        }
        raw, token_valid, image_token = _batch()
        with torch.no_grad():
            tokens = frozen_groot(raw)

        model = RLTokenAutoencoder(_model_config())
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        trainer = RLTokenStage1Trainer(
            model,
            _representation_contract(),
            config=RLTokenStage1Config(learning_rate=1e-2),
        )
        optimizer_parameters = trainer.optimizer.param_groups[0]["params"]
        self.assertEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in model.parameters()},
        )

        update = trainer.train_step(tokens, token_valid, image_token)

        self.assertEqual(update.completed_steps, 1)
        self.assertTrue(update.reconstruction_loss > 0.0)
        self.assertTrue(update.grad_norm > 0.0)
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in model.state_dict().items()
                if name.startswith("encoder.")
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in model.state_dict().items()
                if name.startswith("decoder.")
            )
        )
        for name, value in frozen_groot.state_dict().items():
            torch.testing.assert_close(value, groot_before[name], rtol=0.0, atol=0.0)

        live_tokens = raw.clone().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached"):
            trainer.train_step(live_tokens, token_valid, image_token)

    def test_checkpoint_resume_and_encoder_only_export_roundtrip(self) -> None:
        torch.manual_seed(7)
        tokens, token_valid, image_token = _batch()
        training_config = RLTokenStage1Config(learning_rate=2e-3)
        trainer = RLTokenStage1Trainer(
            RLTokenAutoencoder(_model_config()),
            _representation_contract(),
            config=training_config,
        )
        trainer.train_step(tokens, token_valid, image_token)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = trainer.save_checkpoint(root / "rlt_stage1.pt")
            artifact = trainer.export_encoder(root / "rl_token_encoder.pt")

            resumed = RLTokenStage1Trainer(
                RLTokenAutoencoder(_model_config()),
                _representation_contract(),
                config=training_config,
            )
            self.assertEqual(resumed.load_checkpoint(checkpoint), 1)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(
                    resumed.model.state_dict()[name],
                    value,
                    rtol=0.0,
                    atol=0.0,
                )

            trainer.model.eval()
            expected = trainer.model.encode(tokens, token_valid, image_token)
            frozen = load_frozen_rl_token_encoder(artifact)
            actual = frozen(tokens, token_valid, image_token)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            self.assertTrue(payload["encoder"])
            self.assertFalse(
                any(
                    "decoder" in name or "output_projection" in name
                    for name in payload["encoder"]
                )
            )

            first = trainer.train_step(tokens, token_valid, image_token)
            second = resumed.train_step(tokens, token_valid, image_token)
            self.assertEqual(first.completed_steps, 2)
            self.assertEqual(second.completed_steps, 2)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(
                    resumed.model.state_dict()[name],
                    value,
                    rtol=0.0,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
