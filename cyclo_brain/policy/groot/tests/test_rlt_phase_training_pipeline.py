"""Real saved replay -> CLI -> bundle -> next phase, without GR00T installed."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch
from cyclo_brain.algorithm.rl.rlt import RLTStage2Run
from cyclo_brain.algorithm.rl.rlt.stage2 import RLTStage2Learner
from cyclo_brain.policy.groot.runtime import rlt_stage2_training_cli as cli
from cyclo_brain.policy.groot.runtime.rlt_stage2_dataset import RLTStage2FeatureReplay
from cyclo_brain.policy.groot.runtime.rlt_saved_replay import load_saved_replay


class PhasePipelineTests(unittest.TestCase):
    def test_saved_inputs_three_cli_phases_no_forward_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed_dir = root / 'stage1' / 'artifacts'
            seed_dir.mkdir(parents=True)
            fixture = _run(seed_dir, action_dim=16)
            batch = _batch(fixture)
            spec = fixture.learner.spec
            dataset = root / 'dataset'
            checkpoint = root / 'groot'
            dataset.mkdir(); checkpoint.mkdir()
            metadata = json.loads((fixture.replay_root / 'manifest.json').read_text())['metadata']
            metadata.update(dataset_roots=[str(dataset)], config={'discount': .99}, action_codec='test')
            metadata['row_provenance'] = [
                {'episode_success': True, 'action_sources': ['human'] * 10},
                {'episode_success': False, 'action_sources': ['human'] * 10},
                {'episode_success': True, 'action_sources': ['vla'] * 10},
            ]
            tensors = {k: getattr(batch, k) for k in RLTStage2FeatureReplay._TENSOR_NAMES}
            replay = RLTStage2FeatureReplay(spec, tensors, metadata=metadata)
            replay.save(dataset / 'rlt/replay_cache')
            provenance = SimpleNamespace(
                checkpoint_fingerprint='a' * 64,
                action_dim=16, reference_horizon=32,
                action_normalization_id=spec.action_normalization_id,
                action_codec_id=spec.action_codec_id,
            )
            parent = None
            for phase in ('actor_warmup', 'critic_warmup', 'rl'):
                output = root / phase
                argv = ['--training-phase', phase, '--initialization-mode', 'new' if parent is None else 'resume',
                        '--dataset-root', str(dataset), '--output-dir', str(output), '--job-id', phase,
                        '--steps', '3', '--batch-size', '2', '--save-freq', '2',
                        '--progress-interval', '1', '--device', 'cpu']
                if phase == 'actor_warmup':
                    argv += ['--actor-warmup-data-scope', 'success_only']
                if parent is None:
                    argv += ['--groot-checkpoint', str(checkpoint), '--rl-token-encoder', str(fixture.encoder_artifact_path)]
                else:
                    argv += ['--rlt-bundle', str(parent)]
                events = []
                with patch.object(cli, 'build_groot_rlt_provenance', return_value=provenance), \
                     patch.object(cli, '_validate_encoder_checkpoint_identity'), \
                     patch.object(cli, '_materialize', side_effect=AssertionError('VLA inference forbidden')), \
                     patch('cyclo_brain.policy.groot.runtime.rlt_mlp_replay.prepare_mlp_replay',
                           side_effect=lambda replay, *_args, **_kwargs: (replay, None)), \
                     patch.object(cli, 'json_line', side_effect=lambda event: events.append(event)):
                    self.assertEqual(cli.run(cli.build_parser().parse_args(argv)), 0)
                    if phase == 'rl':
                        # Inject a finite but impossible Bellman target at the
                        # learner boundary; the real CLI must stop before save.
                        failed_args = cli.build_parser().parse_args(argv)
                        failed_args.output_dir = str(root / 'diverged')
                        with patch.object(RLTStage2Learner, 'bellman_target',
                                          side_effect=lambda batch: torch.full_like(batch.reward, 100)):
                            with self.assertRaisesRegex(FloatingPointError, 'target_abs_max'):
                                cli.run(failed_args)
                        self.assertFalse((root / 'diverged/manifest.json').exists())
                        self.assertFalse((root / 'diverged/artifacts/rlt_actor.pt').exists())
                result = next(e for e in events if e.get('event') == 'stage2_training_result')
                self.assertEqual(result['training_phase'], phase)
                request = json.loads((output / 'training_request.json').read_text())
                self.assertEqual(request['actor_warmup_target'], 'executed_actions')
                if phase == 'rl':
                    self.assertEqual(request['actor_rl_failure_target'], 'frozen_warmup_actor')
                    self.assertEqual(request['actor_rl_objective'], 'q_plus_human_or_frozen_actor/v2')
                    actor_progress = next(e for e in events if e.get('actor_negative_q') is not None)
                    self.assertIsNotNone(actor_progress['actor_reference_constraint'])
                    self.assertIsNotNone(actor_progress['policy_q_abs_max'])
                    self.assertIsNotNone(actor_progress['policy_deviation_max'])
                    self.assertIsNotNone(result['target_abs_max'])
                if phase == 'actor_warmup':
                    self.assertEqual(
                        request['actor_warmup_data']['eligible_transition_count'],
                        2,
                    )
                restored = RLTStage2Run.resume(output)
                self.assertIn(phase, restored.learner.completed_phases)
                saved = RLTStage2FeatureReplay.load(output / 'replay_cache', expected_spec=spec)
                for name in tensors:
                    torch.testing.assert_close(saved.tensors[name], tensors[name], rtol=0, atol=0)
                parent = output
            self.assertEqual(restored.learner.completed_actor_warmup_updates, 3)
            self.assertEqual(restored.learner.completed_critic_warmup_updates, 3)
            self.assertEqual(restored.learner.completed_critic_updates, 6)
            self.assertEqual(restored.learner.completed_actor_updates, 4)
            with self.assertRaisesRegex(ValueError, 'retain'):
                load_saved_replay([root / 'wrong'], spec=spec, output_root=root / 'bad', replay_bundle=parent)
            (parent / 'replay_cache/replay.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'digest'):
                load_saved_replay([dataset], spec=spec, output_root=root / 'bad', replay_bundle=parent)

    def test_missing_inputs_rejected_without_feature_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _run(root)
            with self.assertRaisesRegex(ValueError, 'VLA inference is disabled'):
                load_saved_replay([root / 'video_only'], spec=fixture.learner.spec, output_root=root / 'bad')
