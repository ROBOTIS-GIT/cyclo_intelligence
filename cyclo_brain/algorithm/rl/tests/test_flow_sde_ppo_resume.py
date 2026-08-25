"""Cross-job online Flow-SDE PPO bundle and resume contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from cyclo_brain.algorithm.rl.flow_sde_ppo.config import FlowSDEPPOConfig
from cyclo_brain.algorithm.rl.flow_sde_ppo.live_cli import (
    ONLINE_BUNDLE_FORMAT,
    REQUIRED_POLICY_ARTIFACTS,
    RESUME_PROVENANCE_FORMAT,
    _file_sha256,
    _frozen_policy_sha256,
    _policy_artifact_hashes,
    _validate_explicit_resume_source,
)
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup import module_sha256


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation_encoder = nn.Linear(3, 3)
        self.noise_predictor = nn.Linear(3, 3)


def _write_policy(root: Path, *, model_bytes: bytes = b"policy") -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "config.json": b'{"type":"multi_task_dit"}\n',
        "model.safetensors": model_bytes,
        "policy_preprocessor.json": b"{}\n",
        "policy_postprocessor.json": b"{}\n",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return _policy_artifact_hashes(root)


class FlowSDEPPORuntimeResumeContractTest(unittest.TestCase):
    def _bundle(self, root: Path, policy: _Policy, config: FlowSDEPPOConfig):
        exported = root / "pretrained_model"
        policy_hashes = _write_policy(exported)
        checkpoint = root / "training_state" / "trainer_state.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"full-actor-critic-optimizer-state")
        actor_sha = module_sha256(policy.noise_predictor)
        critic_sha = "sha256:" + "1" * 64
        frozen_sha = _frozen_policy_sha256(policy)
        manifest = {
            "format": ONLINE_BUNDLE_FORMAT,
            "status": "complete",
            "job_id": "previous-job",
            "base_checkpoint": str(exported.resolve()),
            "base_policy_artifacts": policy_hashes,
            "lineage_policy_checkpoint": str(exported.resolve()),
            "lineage_policy_artifacts": policy_hashes,
            "task_instruction": "pick up the jelly bag",
            "robot_type": "ffw_sg2_rev1",
            "ppo_config": asdict(config),
            "result": {
                "episodes": 1,
                "update_step": 7,
                "actor_sha256": actor_sha,
                "critic_sha256": critic_sha,
                "frozen_policy_sha256": frozen_sha,
            },
            "source_lineage": {"resume": None, "value_initialization": None},
            "artifacts": {
                "pretrained_model": {
                    "path": "pretrained_model",
                    "files": policy_hashes,
                },
                "trainer_checkpoint": {
                    "path": "training_state/trainer_state.pt",
                    "sha256": _file_sha256(checkpoint),
                },
                "startup_manifest_path": "startup_manifest.json",
                "progress_path": "progress.jsonl",
                "summary_path": "summary.json",
            },
        }
        (root / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return exported, checkpoint, policy_hashes, critic_sha

    def test_completed_bundle_validates_before_cross_job_resume(self):
        torch.manual_seed(5)
        policy = _Policy()
        config = FlowSDEPPOConfig()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exported, checkpoint, policy_hashes, critic_sha = self._bundle(
                root, policy, config
            )
            resolved, provenance, expected_critic = _validate_explicit_resume_source(
                checkpoint,
                pretrained_dir=exported,
                base_policy_artifacts=policy_hashes,
                policy=policy,
                task_instruction="pick up the jelly bag",
                robot_type="ffw_sg2_rev1",
                config=config,
            )
            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(provenance["format"], RESUME_PROVENANCE_FORMAT)
            self.assertEqual(provenance["source_update_step"], 7)
            self.assertEqual(provenance["matched_base_kind"], "exported")
            self.assertEqual(expected_critic, critic_sha)

    def test_resume_rejects_mutated_checkpoint_policy_and_config(self):
        cases = ("checkpoint", "policy", "config")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(9)
                policy = _Policy()
                config = FlowSDEPPOConfig()
                root = Path(temporary)
                exported, checkpoint, policy_hashes, _critic_sha = self._bundle(
                    root, policy, config
                )
                requested_config = config
                if case == "checkpoint":
                    checkpoint.write_bytes(b"mutated")
                elif case == "policy":
                    (exported / "model.safetensors").write_bytes(b"mutated")
                    policy_hashes = _policy_artifact_hashes(exported)
                else:
                    requested_config = FlowSDEPPOConfig(ppo_epochs=7)

                with self.assertRaises(ValueError):
                    _validate_explicit_resume_source(
                        checkpoint,
                        pretrained_dir=exported,
                        base_policy_artifacts=policy_hashes,
                        policy=policy,
                        task_instruction="pick up the jelly bag",
                        robot_type="ffw_sg2_rev1",
                        config=requested_config,
                    )

    def test_required_policy_artifact_contract_is_stable(self):
        self.assertEqual(
            REQUIRED_POLICY_ARTIFACTS,
            (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ),
        )


if __name__ == "__main__":
    unittest.main()
