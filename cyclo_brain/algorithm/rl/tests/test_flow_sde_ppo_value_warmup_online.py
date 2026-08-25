"""Strict offline-value to online-PPO transfer contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cyclo_brain.algorithm.rl.flow_sde_ppo.config import FlowSDEPPOConfig
from cyclo_brain.algorithm.rl.flow_sde_ppo.live_cli import _parse_args, _validate_args
from cyclo_brain.algorithm.rl.flow_sde_ppo.runner import FlowSDEPPOTrainer
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup import (
    VALUE_WARMUP_FORMAT,
    module_sha256,
)
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_cli import BUNDLE_FORMAT
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_online import (
    VALUE_INITIALIZATION_FORMAT,
    load_value_warmup_bundle,
)
from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead


TASK = "pick up the jelly bag"
ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def _file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation_encoder = nn.Linear(3, 3)
        self.noise_predictor = nn.Linear(3, 3)


class _Adapter:
    horizon = 2
    action_dim = 3
    conditioning_dim = 3

    def __init__(self, policy: _Policy) -> None:
        self.policy = policy

    def encode_conditioning(self, batch):
        return batch["conditioning"]

    def velocity(self, latent, progress, conditioning):
        return latent + self.policy.noise_predictor(conditioning)[:, None, :]

    def executed_action_mask(self, batch_size, *, device):
        return torch.ones(batch_size, self.horizon, self.action_dim, dtype=torch.bool, device=device)

    @staticmethod
    def executed_actions(chunk):
        return chunk

    def trainable_parameters(self):
        return tuple(self.policy.noise_predictor.parameters())


def _write_policy_artifacts(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "config.json": b'{"type":"multi_task_dit"}\n',
        "model.safetensors": b"stable-test-policy-artifact",
        "policy_preprocessor.json": b"{}\n",
        "policy_postprocessor.json": b"{}\n",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return {name: _file_sha256(root / name) for name in ARTIFACTS}


def _make_bundle(root: Path, policy: _Policy):
    base = root / "base"
    artifacts = _write_policy_artifacts(base)
    bundle = root / "bundle"
    bundled_policy = bundle / "pretrained_model"
    _write_policy_artifacts(bundled_policy)
    (bundle / "progress.jsonl").write_text("{}\n", encoding="utf-8")
    checkpoint_path = bundle / "training_state" / "value_warmup.pt"
    checkpoint_path.parent.mkdir(parents=True)

    torch.manual_seed(29)
    warmup_value = MultiTaskDiTValueHead(3)
    warmup_optimizer = torch.optim.AdamW(warmup_value.parameters(), lr=5.0e-4)
    prediction = warmup_value(torch.randn(4, 3))
    prediction.square().mean().backward()
    warmup_optimizer.step()
    value_state = {
        name: tensor.detach().clone() for name, tensor in warmup_value.state_dict().items()
    }
    policy_sha = module_sha256(policy)
    config = {
        "steps": 1,
        "batch_size": 4,
        "value_lr": 5.0e-4,
        "gamma": 0.99,
        "task_instruction": TASK,
        "seed": 17,
        "checkpoint_interval": 1,
        "progress_interval": 1,
    }
    base_identity = {
        "path": str(base),
        "policy_sha256": policy_sha,
        "artifacts": artifacts,
    }
    dataset_contract = {"n_action_steps": 16}
    datasets = [{"path": "/immutable/test-data", "identity_sha256": "sha256:data"}]
    checkpoint = {
        "format": VALUE_WARMUP_FORMAT,
        "status": "complete",
        "config": config,
        "completed_steps": 1,
        "dataset_contract": dataset_contract,
        "base_identity": base_identity,
        "dataset_identities": datasets,
        "value_head": value_state,
        "value_optimizer": warmup_optimizer.state_dict(),
        "policy_sha256_before": policy_sha,
        "policy_sha256_after": policy_sha,
    }
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "format": BUNDLE_FORMAT,
        "status": "complete",
        "base": base_identity,
        "datasets": datasets,
        "config": config,
        "dataset_contract": dataset_contract,
        "result": {
            "completed_steps": 1,
            "policy_sha256_before": policy_sha,
            "policy_sha256_after": policy_sha,
        },
        "artifacts": {
            "model_path": "pretrained_model",
            "checkpoint_path": "training_state/value_warmup.pt",
            "progress_path": "progress.jsonl",
        },
    }
    (bundle / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return base, bundle, value_state, manifest, checkpoint_path


def _make_trainer(policy: _Policy):
    adapter = _Adapter(policy)
    value_head = MultiTaskDiTValueHead(adapter.conditioning_dim)
    config = FlowSDEPPOConfig(value_learning_rate=2.0e-4)
    return FlowSDEPPOTrainer(adapter, value_head, config=config)


class FlowSDEValueWarmupOnlineTest(unittest.TestCase):
    def test_exact_value_and_adamw_state_transfer_retains_online_lr_and_provenance(self):
        torch.manual_seed(7)
        policy = _Policy()
        actor_before = {
            name: tensor.detach().clone()
            for name, tensor in policy.noise_predictor.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            base, bundle, expected_value, _manifest, _checkpoint = _make_bundle(
                Path(temporary), policy
            )
            trainer = _make_trainer(policy)
            provenance = load_value_warmup_bundle(
                bundle,
                base_checkpoint=base,
                policy=policy,
                value_head=trainer.value_head,
                value_optimizer=trainer.value_optimizer,
                conditioning_dim=3,
                task_instruction=TASK,
            )
            trainer.record_value_initialization_provenance(provenance)

            self.assertEqual(provenance["format"], VALUE_INITIALIZATION_FORMAT)
            self.assertTrue(provenance["exact_value_reload"])
            self.assertTrue(provenance["optimizer_state_continued"])
            self.assertEqual(provenance["optimizer_step"], 1)
            self.assertEqual(provenance["warmup_value_learning_rate"], 5.0e-4)
            self.assertEqual(provenance["online_value_learning_rate"], 2.0e-4)
            self.assertEqual(trainer.value_optimizer.param_groups[0]["lr"], 2.0e-4)
            self.assertTrue(trainer.value_optimizer.state)
            for name, tensor in trainer.value_head.state_dict().items():
                torch.testing.assert_close(tensor, expected_value[name], rtol=0.0, atol=0.0)
            for name, tensor in policy.noise_predictor.state_dict().items():
                torch.testing.assert_close(tensor, actor_before[name], rtol=0.0, atol=0.0)

            saved = trainer.training_state_dict()
            self.assertEqual(saved["value_initialization_provenance"], provenance)
            with tempfile.TemporaryDirectory() as checkpoint_root:
                path = trainer.save_checkpoint(checkpoint_root)
                restored = _make_trainer(_Policy())
                restored.load_checkpoint(path, strict_config=True)
                self.assertEqual(restored.value_initialization_provenance, provenance)

    def test_rejects_task_policy_artifact_architecture_and_incomplete_bundle_mismatches(self):
        cases = ("task", "policy", "artifact", "architecture", "incomplete")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(13)
                policy = _Policy()
                root = Path(temporary)
                base, bundle, _state, manifest, checkpoint_path = _make_bundle(root, policy)
                trainer = _make_trainer(policy)
                task = TASK
                if case == "task":
                    task = "different instruction"
                elif case == "policy":
                    with torch.no_grad():
                        next(policy.parameters()).add_(1.0)
                elif case == "artifact":
                    (base / "model.safetensors").write_bytes(b"changed")
                elif case == "architecture":
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    checkpoint["value_head"]["network.0.weight"] = torch.zeros(511, 3)
                    torch.save(checkpoint, checkpoint_path)
                elif case == "incomplete":
                    manifest["status"] = "stopped"
                    (bundle / "run_manifest.json").write_text(
                        json.dumps(manifest) + "\n", encoding="utf-8"
                    )

                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    load_value_warmup_bundle(
                        bundle,
                        base_checkpoint=base,
                        policy=policy,
                        value_head=trainer.value_head,
                        value_optimizer=trainer.value_optimizer,
                        conditioning_dim=3,
                        task_instruction=task,
                    )

    def test_live_cli_accepts_bundle_but_rejects_bundle_with_resume(self):
        common = [
            "--base-checkpoint",
            "/model",
            "--output-dir",
            "/output",
            "--job-id",
            "job",
            "--value-warmup-bundle",
            "/warmup",
        ]
        args = _parse_args(common)
        _validate_args(args)
        self.assertEqual(args.value_warmup_bundle, Path("/warmup"))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_args(_parse_args([*common, "--resume"]))

    def test_live_cli_accepts_explicit_resume_checkpoint_exclusively(self):
        common = [
            "--base-checkpoint",
            "/model",
            "--output-dir",
            "/output",
            "--job-id",
            "job",
        ]
        args = _parse_args([*common, "--resume-checkpoint", "/previous/trainer_state.pt"])
        _validate_args(args)
        self.assertEqual(args.resume_checkpoint, Path("/previous/trainer_state.pt"))
        for conflicting in (
            ["--resume"],
            ["--value-warmup-bundle", "/warmup"],
        ):
            with self.subTest(conflicting=conflicting), self.assertRaisesRegex(
                ValueError, "mutually exclusive"
            ):
                _validate_args(
                    _parse_args(
                        [
                            *common,
                            "--resume-checkpoint",
                            "/previous/trainer_state.pt",
                            *conflicting,
                        ]
                    )
                )


if __name__ == "__main__":
    unittest.main()
