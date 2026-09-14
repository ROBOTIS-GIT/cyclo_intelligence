#!/usr/bin/env python3

from __future__ import annotations

import json
from copy import deepcopy
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from unittest import mock


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime import rlt_adapter as rlt_adapter_module  # noqa: E402
from runtime.rlt_adapter import (  # noqa: E402
    GR00TRLTInferenceAdapter,
    GR00TRLTTokenExtractor,
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
        self.training = True
        self.requires_grad = True

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, value):
        self.requires_grad = bool(value)
        return self

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


class _TTRTCPolicy(_Policy):
    def __init__(self, *, preserve_prefix=True, horizon=16, action_dim=19):
        super().__init__()
        self.preserve_prefix = preserve_prefix
        self.tt_kwargs = None
        self.horizon = horizon
        self.action_dim = action_dim

    def get_tt_rtc_model_contract(self):
        return {"action_horizon": self.horizon, "action_dimension": self.action_dim}

    def get_action_tt_rtc(self, _observation, **kwargs):
        self.tt_kwargs = kwargs
        delay_steps = kwargs["delay_steps"]
        normalized_prefix = torch.full(
            (1, delay_steps, self.action_dim),
            0.5,
            dtype=torch.float32,
        )
        reference = torch.arange(
            self.horizon * self.action_dim,
            dtype=torch.float32,
        ).reshape(1, self.horizon, self.action_dim)
        if self.preserve_prefix:
            reference[:, :delay_steps] = normalized_prefix
        return {}, {
            "tt_rtc_rlt_context": {
                "schema": "cyclo.groot.tt-rtc-rlt-context/v1",
                "delay_steps": delay_steps,
                "tokens": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
                "token_valid": torch.ones(1, 2, dtype=torch.bool),
                "image_token": torch.tensor([[True, False]]),
                "proprio": torch.zeros(1, self.action_dim),
                "reference_actions": reference,
                "normalized_committed_prefix": normalized_prefix,
                "batched_states": {
                    "state": np.zeros((1, 1, self.action_dim), dtype=np.float32)
                },
            }
        }


class _Shadow:
    actor_qualification = "training_only_not_deployment_validated"
    spec = SimpleNamespace(
        reference_horizon=16,
        chunk_length=10,
        action_dim=19,
        proprio_dim=19,
        rl_token_artifact_fingerprint='e' * 64,
    )

    def __init__(self):
        self.tokens = None

    def __call__(
        self,
        tokens,
        _valid,
        _image,
        proprio,
        reference,
        *,
        reference_offset_steps=0,
    ):
        self.tokens = tokens.clone()
        self.proprio = proprio.clone()
        self.reference = reference.clone()
        self.reference_offset_steps = reference_offset_steps
        self.reference_slice = reference[
            :, reference_offset_steps : reference_offset_steps + 10
        ].clone()
        batch = tokens.shape[0]
        return SimpleNamespace(
            action_mean=torch.full((batch, 10, self.spec.action_dim), 0.25),
            z_rl=tokens.mean(dim=1), reference_prefix=self.reference_slice,
        )


class RLTInferenceAdapterTests(unittest.TestCase):
    def test_async_training_requires_completed_bundle_before_enabling(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = GR00TRLTInferenceAdapter(_Policy(), _Shadow(), SimpleNamespace(root=Path(directory)))
            with self.assertRaisesRegex(ValueError, 'completed Stage-2 bundle'):
                adapter.set_async_training(True)
            self.assertFalse(adapter.async_training_status()['async_enabled'])
            self.assertEqual(adapter.async_training_status()['replay_source'], {
                'kind': 'selected_datasets', 'paths': [], 'transitions': 0,
            })
            self.assertIsNone(adapter._async_prepare)

    def test_async_training_stages_mlp_and_manual_apply_preserves_frozen_models(self):
        from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch
        from cyclo_brain.algorithm.rl.rlt import build_stage2_training_round, stage2_spec_fingerprint
        from runtime.rlt_stage2_dataset import RLTStage2FeatureReplay

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _run(root, action_dim=16)
            batch = _batch(run)
            metadata = json.loads((run.replay_root / 'manifest.json').read_text())['metadata']
            replay = RLTStage2FeatureReplay(run.learner.spec, {
                name: getattr(batch, name) for name in RLTStage2FeatureReplay._TENSOR_NAMES
            }, metadata=metadata)
            replay.save(run.replay_root)
            run.bind_training_round(build_stage2_training_round(
                run.replay_root, expected_spec_fingerprint=stage2_spec_fingerprint(run.learner.spec),
                reference_seed=17, feature_batch_size=2, sampling_seed=23,
                batch_size=3, steps=10, starting_critic_updates=0,
            ), replay_root=run.replay_root)
            bundle = run.save(root / 'bundle')
            shutil.copytree(run.replay_root, bundle / 'replay_cache')
            actor = deepcopy(run.learner.actor).eval().requires_grad_(False)
            encoder = object()
            shadow = SimpleNamespace(actor=actor, encoder=encoder, spec=run.learner.spec,
                                     actor_qualification='training_only_not_deployment_validated')
            policy = SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')))
            adapter = GR00TRLTInferenceAdapter(policy, shadow, SimpleNamespace(root=bundle))
            # Recording identifies weights; resume must identify the full checkpoint.
            adapter.recording_identity = {'groot_sha256': 'c' * 64, 'actor_sha256': 'b' * 64}
            adapter._groot_checkpoint_fingerprint = run.source.groot_checkpoint_fingerprint
            try:
                adapter.set_async_training(True)
                adapter._async_prepare.join(5)
                self.assertFalse(adapter._async_prepare.is_alive())
                self.assertIsNone(adapter.async_training_status()['training_error'])
                worker = adapter._async_training
                self.assertIsNotNone(worker)
                with worker.inference():
                    self.assertTrue(worker.status()['waiting_data'])
                    self.assertIsNone(worker.status()['last_update'])
                # Existing bundle replay must not be sampled implicitly.
                replay.metadata['usable_episode_count'] = 2
                adapter._async_replay.build = mock.Mock(return_value=replay)
                adapter.select_async_datasets(['/selected/lerobot'])
                with worker._condition:
                    self.assertTrue(worker._condition.wait_for(
                        lambda: worker._error is not None or (
                            worker._last_update and worker._last_update['completed_actor_updates'] >= 10
                        ), timeout=5,
                    ))
                adapter.set_async_training(False)
                with adapter.inference_slot():
                    pass
                self.assertEqual(adapter.async_training_status()['replay_source'], {
                    'kind': 'selected_datasets', 'paths': ['/selected/lerobot'],
                    'transitions': len(replay), 'episodes': 2, 'batch_size': 3,
                })
                self.assertIsNone(worker.status()['error'])
                self.assertIs(shadow.actor, actor)
                self.assertTrue(adapter.policy_updates.status()['can_apply'])
                adapter.policy_updates.request_apply()
                with adapter.inference_slot():
                    adapter.apply_pending_policy()
                self.assertIsNot(shadow.actor, actor)
                self.assertIs(shadow.encoder, encoder)
                self.assertIs(adapter.policy, policy)
                self.assertEqual(adapter.recording_identity['base_actor_sha256'], 'b' * 64)
                self.assertIn('actor_state_sha256', adapter.recording_identity)
                self.assertNotIn('actor_sha256', adapter.recording_identity)
                adapter.set_async_training(True, max_updates=2)
                self.assertIs(adapter._async_training, worker)
                with worker._condition:
                    self.assertTrue(worker._condition.wait_for(
                        lambda: worker._limit_reached and not worker._busy, timeout=5))
                status = adapter.async_training_status()
                self.assertFalse(status['async_enabled'])
                self.assertFalse(adapter._async_enabled)
                self.assertFalse(adapter._async_replay._enabled)
                self.assertEqual(status['updates_this_run'], 2)
                self.assertEqual(status['max_updates'], 2)
                self.assertEqual(status['total_critic_updates'], worker.learner.completed_critic_updates)
                self.assertFalse(worker.learner.update_pending)
            finally:
                adapter.close_async_training(dispose=True)
            self.assertFalse(worker._thread.is_alive())

    def test_load_validates_stage1_and_stage2_runtime_provenance(self) -> None:
        policy = _Policy()
        policy.model.device = torch.device("cpu")
        shadow = _Shadow()
        shadow.encoder = SimpleNamespace(representation_contract={"contract": True})
        bundle = SimpleNamespace(
            root=Path("/bundle"),
            encoder=Path("/bundle/artifacts/rl_token_encoder.pt"),
            actor=Path("/bundle/artifacts/rlt_actor.pt"),
        )
        provenance = SimpleNamespace(weight_fingerprint="a" * 64, checkpoint_fingerprint="c" * 64)

        with mock.patch(
            "cyclo_brain.algorithm.rl.rlt.load_groot_rlt_shadow_policy",
            return_value=shadow,
        ), mock.patch.object(
            rlt_adapter_module,
            "resolve_rlt_bundle",
            return_value=bundle,
        ), mock.patch.object(
            rlt_adapter_module,
            "build_groot_rlt_provenance",
            return_value=provenance,
        ), mock.patch.object(
            rlt_adapter_module,
            "validate_stage1_encoder_provenance",
        ) as validate_stage1, mock.patch.object(
            rlt_adapter_module,
            "validate_stage2_bundle_provenance",
        ) as validate_stage2, mock.patch.object(
            rlt_adapter_module, 'file_sha256', return_value='b' * 64,
        ):
            adapter = GR00TRLTInferenceAdapter.load(
                policy,
                "/bundle",
                "/model",
            )

        self.assertIs(adapter.shadow_policy, shadow)
        self.assertEqual(adapter._groot_checkpoint_fingerprint, "c" * 64)
        self.assertEqual(adapter.recording_identity['groot_sha256'], "a" * 64)
        self.assertEqual(adapter.recording_identity['actor_sha256'], 'b' * 64)
        validate_stage1.assert_called_once_with(
            shadow.encoder.representation_contract,
            provenance,
        )
        validate_stage2.assert_called_once_with(
            Path("/bundle"),
            spec=shadow.spec,
            provenance=provenance,
        )

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

    def test_tt_rtc_reference_uses_delay_shift_instead_of_first_ten(self) -> None:
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

        adapter.get_action(observation, reference_offset_steps=6)

        self.assertEqual(shadow.reference_offset_steps, 6)
        torch.testing.assert_close(
            shadow.reference_slice,
            shadow.reference[:, 6:16],
            rtol=0.0,
            atol=0.0,
        )

    def test_tt_rtc_mlp_uses_same_forward_context_and_shifted_reference(self) -> None:
        policy = _TTRTCPolicy()
        shadow = _Shadow()
        adapter = GR00TRLTInferenceAdapter(
            policy,
            shadow,
            SimpleNamespace(root=Path("."), encoder=Path("e"), actor=Path("a")),
        )
        adapter.require_tt_rtc_capability = lambda: None

        action = adapter.get_action_tt_rtc(
            {},
            committed_action_prefix=np.zeros((1, 6, 19), dtype=np.float32),
            delay_steps=6,
            action_horizon=16,
        )

        self.assertEqual(action["action"].shape, (1, 10, 19))
        self.assertTrue(policy.tt_kwargs["return_rlt_context"])
        self.assertEqual(shadow.reference_offset_steps, 6)
        torch.testing.assert_close(
            shadow.reference_slice,
            shadow.reference[:, 6:16],
            rtol=0.0,
            atol=0.0,
        )

    def test_tt_rtc_32x16_selects_ten_contiguous_actions_after_each_prefix(self):
        for delay in range(7):
            with self.subTest(delay=delay):
                policy = _TTRTCPolicy(horizon=32, action_dim=16)
                shadow = _Shadow()
                shadow.spec = SimpleNamespace(**{**vars(shadow.spec),
                    "reference_horizon": 32, "action_dim": 16, "proprio_dim": 16})
                adapter = GR00TRLTInferenceAdapter(policy, shadow,
                    SimpleNamespace(root=Path("."), encoder=Path("e"), actor=Path("a")))
                adapter.require_tt_rtc_capability = lambda: None
                action = adapter.get_action_tt_rtc({},
                    committed_action_prefix=np.zeros((1, delay, 16), dtype=np.float32),
                    delay_steps=delay, action_horizon=32, capture_context=True)
                self.assertEqual(action["action"].shape, (1, 10, 16))
                torch.testing.assert_close(shadow.reference_slice,
                                           shadow.reference[:, delay:delay+10])
                self.assertEqual(adapter.recording_context["reference_actions"].shape, (1, 32, 16))
                self.assertEqual(adapter.recording_context["mlp_reference"].shape, (1, 10, 16))

    def test_legacy_bundle_cannot_be_used_with_32x16_policy(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            GR00TRLTInferenceAdapter(_TTRTCPolicy(horizon=32, action_dim=16),
                _Shadow(), SimpleNamespace(root=Path(".")))

    def test_real_action_mlp_forward_accepts_32x16_reference(self):
        from cyclo_brain.algorithm.rl.rlt.shadow import GR00TRLTShadowPolicy
        from cyclo_brain.model.mlp import RLTGaussianChunkActor

        class _Encoder(torch.nn.Module):
            def forward(self, tokens, _valid, _image):
                return tokens.mean(dim=1)

        spec = SimpleNamespace(reference_horizon=32, chunk_length=10,
                               action_dim=16, proprio_dim=16)
        actor = RLTGaussianChunkActor(4, 16, 10, 16,
                                     fixed_standard_deviation=0.1, hidden_dims=(16,))
        shadow = GR00TRLTShadowPolicy(_Encoder(), actor, spec,
                                     actor_qualification="test_only")
        reference = torch.arange(512, dtype=torch.float32).reshape(1, 32, 16)
        for delay in range(7):
            result = shadow(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                            torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 16),
                            reference, reference_offset_steps=delay)
            self.assertEqual(result.action_mean.shape, (1, 10, 16))
            self.assertTrue(torch.isfinite(result.action_mean).all())
            torch.testing.assert_close(result.reference_prefix, reference[:, delay:delay+10])

    def test_tt_rtc_mlp_rejects_a_reference_that_does_not_preserve_prefix(self) -> None:
        policy = _TTRTCPolicy(preserve_prefix=False)
        shadow = _Shadow()
        adapter = GR00TRLTInferenceAdapter(
            policy,
            shadow,
            SimpleNamespace(root=Path("."), encoder=Path("e"), actor=Path("a")),
        )
        adapter.require_tt_rtc_capability = lambda: None

        with self.assertRaisesRegex(RuntimeError, "exact normalized committed prefix"):
            adapter.get_action_tt_rtc(
                {},
                committed_action_prefix=np.zeros((1, 6, 19), dtype=np.float32),
                delay_steps=6,
                action_horizon=16,
            )

    def test_recording_captures_actual_token_and_shifted_reference_only_on_request(self):
        policy, shadow = _TTRTCPolicy(), _Shadow()
        adapter = GR00TRLTInferenceAdapter(
            policy, shadow,
            SimpleNamespace(root=Path('.'), encoder=Path('e'), actor=Path('a')),
        )
        adapter.require_tt_rtc_capability = lambda: None
        kwargs = dict(
            committed_action_prefix=np.zeros((1, 6, 19), dtype=np.float32),
            delay_steps=6, action_horizon=16,
        )
        action = adapter.get_action_tt_rtc({}, capture_context=True, **kwargs)
        recorded = adapter.recording_context
        np.testing.assert_array_equal(recorded['z_rl'], shadow.tokens.mean(dim=1))
        np.testing.assert_array_equal(recorded['mlp_reference'], shadow.reference[:, 6:16])
        np.testing.assert_array_equal(recorded['action_mean'], action['action'])
        self.assertEqual(recorded['reference_actions'].shape, (1, 16, 19))
        self.assertTrue(all(isinstance(value, np.ndarray) for value in recorded.values()))
        adapter.get_action_tt_rtc({}, **kwargs)
        self.assertIsNone(adapter.recording_context)

    def test_standard_rlt_recording_retains_unshifted_reference(self):
        policy, shadow = _Policy(), _Shadow()
        adapter = GR00TRLTInferenceAdapter(
            policy, shadow,
            SimpleNamespace(root=Path('.'), encoder=Path('e'), actor=Path('a')),
        )
        adapter.get_action(
            {'state': {'state': np.zeros((1, 1, 19), dtype=np.float32)}},
            capture_context=True,
        )
        np.testing.assert_array_equal(
            adapter.recording_context['mlp_reference'], shadow.reference[:, :10],
        )

    def test_stage1_extractor_freezes_model_and_skips_action_head(self) -> None:
        policy = _Policy()
        extractor = GR00TRLTTokenExtractor(policy)
        observation = {
            "state": {"state": np.zeros((1, 1, 19), dtype=np.float32)}
        }

        result = extractor.extract(observation)

        self.assertFalse(policy.model.training)
        self.assertFalse(policy.model.requires_grad)
        self.assertEqual(tuple(result["tokens"].shape), (1, 2, 4))
        self.assertEqual(result["token_valid"].dtype, torch.bool)
        self.assertEqual(result["image_token"].dtype, torch.bool)
        torch.testing.assert_close(
            result["tokens"][0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
        )


if __name__ == "__main__":
    unittest.main()
