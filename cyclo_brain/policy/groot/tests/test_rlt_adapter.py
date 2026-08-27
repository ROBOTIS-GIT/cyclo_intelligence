#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.rlt_adapter import (  # noqa: E402
    GR00TRLTInferenceAdapter,
    is_deployment_qualified,
    resolve_rlt_bundle,
)


class RLTBundleResolverTests(unittest.TestCase):
    def test_stage2_directory_resolves_matching_sibling_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor_root = root / "stage2"
            actor = actor_root / "artifacts/rlt_actor.pt"
            actor.parent.mkdir(parents=True)
            actor.write_bytes(b"actor")
            (actor_root / "training_state").mkdir()
            (actor_root / "training_state/rlt_stage2.pt.run.json").write_text(
                json.dumps({
                    "base_contract": {
                        "learner": {"contract": {"spec": {
                            "rl_token_artifact_fingerprint": "matching"
                        }}}
                    }
                }),
                encoding="utf-8",
            )
            encoder_root = root / "stage1"
            encoder = encoder_root / "artifacts/rl_token_encoder.pt"
            encoder.parent.mkdir(parents=True)
            encoder.write_bytes(b"encoder")
            (encoder_root / "training_state").mkdir()
            (encoder_root / "training_state/rlt_stage1.pt.run.json").write_text(
                json.dumps({"artifact": {"artifact_fingerprint": "matching"}}),
                encoding="utf-8",
            )

            bundle = resolve_rlt_bundle(actor_root)

            self.assertEqual(bundle.actor, actor)
            self.assertEqual(bundle.encoder, encoder)

    def test_ambiguous_sibling_encoders_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor = root / "stage2/artifacts/rlt_actor.pt"
            actor.parent.mkdir(parents=True)
            actor.write_bytes(b"actor")
            for name in ("stage1-a", "stage1-b"):
                encoder = root / name / "artifacts/rl_token_encoder.pt"
                encoder.parent.mkdir(parents=True)
                encoder.write_bytes(b"encoder")

            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_rlt_bundle(root / "stage2")


class _Processor:
    def __call__(self, messages):
        return {"state": messages[0]["content"].states["state"]}

    def decode_action(self, action, _tag, _states):
        return {"action": np.asarray(action, dtype=np.float32)}


class _Backbone:
    def __call__(self, inputs):
        batch = inputs["state"].shape[0]
        return {
            "backbone_features": torch.arange(
                batch * 2 * 4, dtype=torch.float32
            ).reshape(batch, 2, 4),
            "backbone_attention_mask": torch.ones(batch, 2, dtype=torch.bool),
            "image_mask": torch.tensor([[True, False]] * batch),
        }


class _ActionHead:
    def get_action(self, backbone, _action_inputs, _options=None):
        batch = backbone["backbone_features"].shape[0]
        backbone["backbone_features"].add_(1000.0)
        return {"action_pred": torch.ones(batch, 16, 19)}


class _Model:
    dtype = torch.float32

    def __init__(self):
        self.backbone = _Backbone()
        self.action_head = _ActionHead()

    def prepare_input(self, inputs):
        return inputs, inputs


class _Policy:
    strict = False
    embodiment_tag = "test"

    def __init__(self):
        self.model = _Model()
        self.processor = _Processor()
        self.modality_configs = {
            "state": SimpleNamespace(modality_keys=["state"]),
            "action": SimpleNamespace(modality_keys=["action"]),
        }

    def _unbatch_observation(self, observation):
        return [
            {"state": {"state": observation["state"]["state"][index]}}
            for index in range(observation["state"]["state"].shape[0])
        ]

    def _to_vla_step_data(self, observation):
        return SimpleNamespace(states=observation["state"])

    def collate_fn(self, processed):
        state = np.stack([item["state"] for item in processed])
        return {"inputs": {"state": torch.from_numpy(state)}}


class _Shadow:
    actor_qualification = "training_only_not_deployment_validated"
    spec = SimpleNamespace(
        reference_horizon=16,
        chunk_length=10,
        action_dim=19,
        proprio_dim=19,
    )

    def __init__(self):
        self.tokens = None

    def __call__(self, tokens, _valid, _image, proprio, reference):
        self.tokens = tokens.clone()
        self.proprio = proprio.clone()
        self.reference = reference.clone()
        batch = tokens.shape[0]
        return SimpleNamespace(action_mean=torch.full((batch, 10, 19), 0.25))


class RLTInferenceAdapterTests(unittest.TestCase):
    def test_deployment_qualification_fails_closed(self) -> None:
        self.assertTrue(is_deployment_qualified("deployment_qualified"))
        self.assertFalse(
            is_deployment_qualified("training_only_not_deployment_validated")
        )
        self.assertFalse(is_deployment_qualified("deployment_candidate"))
        self.assertFalse(is_deployment_qualified(None))

    def test_adapter_exposes_bundle_deployment_qualification(self) -> None:
        policy = _Policy()
        shadow = _Shadow()
        shadow.actor_qualification = "deployment_qualified"

        adapter = GR00TRLTInferenceAdapter(
            policy,
            shadow,
            SimpleNamespace(root=Path("."), encoder=Path("e"), actor=Path("a")),
        )

        self.assertEqual(adapter.qualification, "deployment_qualified")
        self.assertTrue(adapter.deployment_qualified)

    def test_rlt_candidate_uses_raw_tokens_and_returns_10_by_19(self) -> None:
        policy = _Policy()
        shadow = _Shadow()
        adapter = GR00TRLTInferenceAdapter(
            policy,
            shadow,
            SimpleNamespace(root=Path("."), encoder=Path("e"), actor=Path("a")),
        )
        observation = {
            "state": {"state": np.zeros((1, 1, 19), dtype=np.float32)}
        }

        action = adapter.get_action(observation)

        self.assertEqual(action["action"].shape, (1, 10, 19))
        self.assertEqual(tuple(shadow.reference.shape), (1, 16, 19))
        self.assertEqual(tuple(shadow.proprio.shape), (1, 19))
        torch.testing.assert_close(
            shadow.tokens[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
        )
        np.testing.assert_allclose(action["action"], 0.25)


if __name__ == "__main__":
    unittest.main()
