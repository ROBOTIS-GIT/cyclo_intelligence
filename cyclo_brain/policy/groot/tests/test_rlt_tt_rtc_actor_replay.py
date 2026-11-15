"""TT-RTC Actor replay alignment and phase-isolation regression tests."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import rlt_tt_rtc_actor_replay as rtc
from runtime.rlt_stage2_dataset import RLTStage2FeatureReplay, RLTStage2RawEpisode
from runtime.rlt_stage2_training_cli import _training_sample_pool
from runtime.rlt_saved_replay import load_saved_replay
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run


class Source:
    fps = 15.0
    def __init__(self, root, length=18, source_frame_indices=()):
        self.root = root
        self.length = length
        self.source_frame_indices = tuple(source_frame_indices)
    def content_snapshot(self):
        return dict(content_fingerprint='a'*64, file_count=1, total_byte_count=1, files=[])
    def iter_episodes(self):
        state = np.arange(self.length,dtype=np.float32)[:,None].repeat(8,axis=1)
        yield RLTStage2RawEpisode(self.root,0,self.length,True,tuple([0]*self.length),
            {'arm_left':state,'arm_right':state},
            {'arm_left':state+100,'arm_right':state+100}, self.source_frame_indices)
    def iter_observation_batches(self, episode, indices, batch_size):
        for i in indices:yield [i], {'frame':i}


class Extractor:
    action_groups=('arm_left','arm_right')
    def __init__(self):self.anchors=[]
    def normalize_actions(self, actions, states):
        state=np.concatenate(list(states.values()),axis=-1)
        self.anchors.append(state.copy())
        return np.concatenate(list(actions.values()),axis=-1)-state


class TTActorReplayTests(unittest.TestCase):
    def test_window_boundaries_and_short_episodes(self):
        self.assertEqual(rtc.actor_window_starts(15),())
        self.assertEqual(rtc.actor_window_starts(16),(0,))
        self.assertEqual(rtc.actor_window_starts(18),(0,1,2))
        with self.assertRaises(ValueError):rtc.actor_window_starts(18,prefix=5)

    def test_alignment_cache_mask_and_no_fabricated_td_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=_run(root,action_dim=16);source=Source(root/'dataset');ext=Extractor()
            labels={(0,i):('vla' if i==7 else 'human') for i in range(18)}
            calls=[]
            def features(extractor,observation,prefix,spec):
                t=observation['frame'];calls.append(t)
                np.testing.assert_array_equal(prefix[:,0],np.arange(t,t+6)+100)
                return dict(z_rl=torch.full((8,),float(t)),proprio=torch.full((16,),float(t)),
                            reference_actions=torch.full((10,16),123.0))
            out=source.root/'rlt'/rtc.CACHE_SUBDIR/'replay_cache'
            with patch('runtime.rlt_data_validation._recorded_sources',return_value=(set(),labels)), patch.object(rtc,'extract_tt_rtc',side_effect=features):
                replay=rtc.materialize_tt_rtc_actor_replay((source,),extractor=ext,spec=run.learner.spec,output_root=out)
                again=rtc.materialize_tt_rtc_actor_replay((source,),extractor=ext,spec=run.learner.spec,output_root=root/'again')
            self.assertEqual(calls,[0,1,2])
            self.assertEqual(again.metadata['feature_cache_hits'],3)
            self.assertEqual(len(replay),3)
            for k in ('reward','bootstrap_discount','next_z_rl'):
                self.assertNotIn(k,replay.tensors)
            torch.testing.assert_close(replay.tensors['executed_actions'][0,:,0],torch.arange(106,116).float())
            self.assertEqual(replay.metadata['row_provenance'][0]['action_start_frame_index'],6)
            self.assertEqual(replay.tensors['imitation_mask'][0].tolist(),[1,0,1,1,1,1,1,1,1,1])
            for t,anchor in enumerate(ext.anchors[:3]):np.testing.assert_array_equal(anchor,np.full((10,16),t))
            batch=replay.batch([0],device='cpu');batch.validate(run.learner.spec,actor_only=True)
            self.assertIsNone(batch.next_z_rl)
            with self.assertRaisesRegex(ValueError,'cannot train Critic/RL'):batch.validate(run.learner.spec)
            with self.assertRaisesRegex(ValueError,'cannot train Critic/RL'):
                _training_sample_pool(replay,training_phase='rl',actor_warmup_data_scope='all')
            pool,_=_training_sample_pool(replay,training_phase='actor_warmup',actor_warmup_data_scope='success_only')
            self.assertEqual(pool,(0,1,2))
            with patch('runtime.rlt_data_validation.open_rlt_stage2_source',return_value=source):
                loaded=load_saved_replay([source.root],spec=run.learner.spec,output_root=root/'loaded')
            self.assertTrue(loaded.actor_only)
            self.assertEqual(loaded.metadata['prefix_length'],6)

    def test_prepare_cli_publishes_terminal_hold_replay_and_reuses_original_features(self):
        from runtime import rlt_stage2_replay_cli as cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _run(root, action_dim=16).learner.spec
            source = Source(root / 'dataset')
            extractor = Extractor()
            labels = {(0, i): 'human' for i in range(source.length)}
            features = dict(z_rl=torch.ones(8), proprio=torch.ones(16),
                            reference_actions=torch.zeros(10, 16))
            legacy = source.root / 'rlt' / rtc.CACHE_SUBDIR / 'replay_cache'
            checkpoint = root / 'model'
            checkpoint.mkdir()
            encoder = root / 'encoder.pt'
            encoder.write_bytes(b'encoder')
            args = cli.build_parser().parse_args([
                '--dataset-root', str(source.root), '--groot-checkpoint', str(checkpoint),
                '--rl-token-encoder', str(encoder), '--job-id', 'test-actor',
                '--replay-mode', 'tt_rtc_actor', '--device', 'cpu',
            ])
            modules = {
                'gr00t.data.embodiment_tags': SimpleNamespace(
                    EmbodimentTag=SimpleNamespace(NEW_EMBODIMENT='test')),
                'gr00t.policy.gr00t_policy': SimpleNamespace(Gr00tPolicy=lambda **kwargs: object()),
            }
            with patch('runtime.rlt_data_validation._recorded_sources', return_value=(set(), labels)), \
                    patch.object(rtc, 'extract_tt_rtc', return_value=features) as extract:
                rtc.materialize_tt_rtc_actor_replay(
                    [source], extractor=extractor, spec=spec, output_root=legacy)
                self.assertEqual(extract.call_count, 3)
                with patch.dict(sys.modules, modules), \
                        patch.object(cli, 'build_groot_rlt_provenance'), \
                        patch.object(cli, 'load_frozen_rl_token_encoder', return_value=SimpleNamespace(representation_contract={})), \
                        patch.object(cli, 'validate_stage1_encoder_provenance'), \
                        patch.object(cli, '_spec_from_encoder', return_value=spec), \
                        patch.object(cli, 'GR00TRLTStage2Extractor', return_value=extractor), \
                        patch.object(cli, 'open_rlt_stage2_source', return_value=source), \
                        patch.object(rtc, 'configure_tt_rtc_policy'), \
                        patch.object(cli.torch.cuda, 'is_available', return_value=False), \
                        patch.object(cli, 'json_line') as event:
                    self.assertEqual(cli.run(args), 0)
                    target = source.root / 'rlt' / rtc.HOLD_CACHE_SUBDIR / 'replay_cache'
                    self.assertEqual(event.call_args.args[0]['replay_paths'], [str(target)])
                    replay = RLTStage2FeatureReplay.load(target)
                    self.assertEqual(len(replay), source.length)
                    self.assertEqual(replay.metadata['terminal_hold_policy'], rtc.TERMINAL_HOLD_POLICY)
                    self.assertEqual(replay.metadata['feature_cache_hits'], 3)
                    self.assertEqual(replay.metadata['synthetic_hold_count'], 15)
                    self.assertEqual(extract.call_count, 18)
                    self.assertEqual(len(RLTStage2FeatureReplay.load(legacy)), 3)
                    with patch('runtime.rlt_data_validation.open_rlt_stage2_source', return_value=source):
                        loaded = load_saved_replay(
                            [source.root], spec=spec, output_root=root / 'loaded_hold',
                            training_phase='actor_warmup')
                    self.assertEqual(len(loaded), 18)
                    self.assertEqual(loaded.metadata['terminal_hold_policy'], rtc.TERMINAL_HOLD_POLICY)
                    self.assertEqual(loaded.metadata['synthetic_hold_count'], 15)
                    # Repeating preparation must reuse the published result.
                    self.assertEqual(cli.run(args), 0)
                    self.assertEqual(extract.call_count, 18)

    def test_downsampled_frames_keep_original_action_source_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=_run(root,action_dim=16)
            source_indices=tuple(range(0,36,2))
            source=Source(root/'dataset',source_frame_indices=source_indices)
            labels={(0,i):('vla' if i==14 else 'human') for i in source_indices}
            features=dict(z_rl=torch.ones(8),proprio=torch.ones(16),
                          reference_actions=torch.zeros(10,16))
            with patch('runtime.rlt_data_validation._recorded_sources',return_value=(set(),labels)), \
                    patch.object(rtc,'extract_tt_rtc',return_value=features):
                replay=rtc.materialize_tt_rtc_actor_replay(
                    (source,),extractor=Extractor(),spec=run.learner.spec,
                    output_root=root/'actor_replay')
            row=replay.metadata['row_provenance'][0]
            self.assertEqual(row['source_observation_frame_index'],0)
            self.assertEqual(row['source_prefix_frame_indices'],[0,2,4,6,8,10])
            self.assertEqual(row['source_action_frame_indices'],[12,14,16,18,20,22,24,26,28,30])
            self.assertEqual(replay.tensors['imitation_mask'][0].tolist(),[1,0,1,1,1,1,1,1,1,1])

    def test_warmup_uses_only_human_elements_and_needs_no_next_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=_run(Path(tmp),action_dim=16);spec=run.learner.spec
            data=dict(z_rl=torch.ones(1,8),proprio=torch.ones(1,16),reference_actions=torch.zeros(1,10,16),
                executed_actions=torch.full((1,10,16),100.0),committed_prefix=torch.zeros(1,6,16),
                imitation_mask=torch.zeros(1,10))
            data['executed_actions'][:,0]=1;data['imitation_mask'][:,0]=1
            replay=RLTStage2FeatureReplay(spec,data,metadata={'replay_role':rtc.ROLE,'prefix_length':6})
            for p in run.learner.actor.parameters():p.data.zero_()
            run.learner.begin_phase('actor_warmup')
            update=run.learner.update_phase(replay.batch([0],device='cpu'))
            self.assertAlmostEqual(update.actor_loss,1.0,places=6)

    def test_extractor_uses_tt_rtc_context_and_slices_after_prefix(self):
        from runtime.tt_rtc import TT_RTC_RLT_CONTEXT_SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            spec=_run(Path(tmp),action_dim=16).learner.spec
            prefix=np.arange(96,dtype=np.float32).reshape(6,16)
            reference=torch.arange(32*16).reshape(1,32,16).float()
            reference[:,:6]=torch.from_numpy(prefix)
            ctx=dict(schema=TT_RTC_RLT_CONTEXT_SCHEMA,delay_steps=6,tokens=torch.ones(1,4,8),
                token_valid=torch.ones(1,4,dtype=torch.bool),image_token=torch.ones(1,4,dtype=torch.bool),
                proprio=torch.ones(1,16),reference_actions=reference,
                normalized_committed_prefix=torch.from_numpy(prefix[None]),batched_states={'arm_left':0,'arm_right':0})
            captured={}
            def runner(obs,**kwargs):captured.update(kwargs);return {},{'tt_rtc_rlt_context':ctx}
            policy=SimpleNamespace(get_action_tt_rtc=runner,modality_configs={'state':SimpleNamespace(modality_keys=['arm_left','arm_right'])})
            extractor=SimpleNamespace(policy=policy,encoder=lambda *_:torch.ones(1,8))
            result=rtc.extract_tt_rtc(extractor,{},prefix,spec)
            self.assertEqual(captured['delay_steps'],6)
            torch.testing.assert_close(result['reference_actions'],reference[0,6:16])
            ctx['reference_actions']=torch.zeros_like(reference)
            with self.assertRaisesRegex(RuntimeError,'exact normalized'):
                rtc.extract_tt_rtc(extractor,{},prefix,spec)

    def test_human_subset_validation_and_checkpoint_resume(self):
        from cyclo_brain.algorithm.rl.rlt import RLTStage2Run, build_stage2_training_round
        from runtime.rlt_data_validation import prepare_references
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=_run(root,action_dim=16);source=Source(root/'dataset')
            snapshot = json.loads((root/'replay_cache/manifest.json').read_text())['metadata']['dataset_snapshots'][0]
            source.content_snapshot = lambda: snapshot
            labels={(0,i):('human' if i==6 else 'vla') for i in range(18)}
            features=dict(z_rl=torch.ones(8),proprio=torch.ones(16),reference_actions=torch.zeros(10,16))
            with patch('runtime.rlt_data_validation._recorded_sources',return_value=(set(),labels)),patch.object(rtc,'extract_tt_rtc',return_value=features):
                replay=rtc.materialize_tt_rtc_actor_replay((source,),extractor=Extractor(),spec=run.learner.spec,output_root=root/'actor_replay')
                prepared,report=prepare_references(replay)
            self.assertEqual(report['status'],'ready')
            pool,record=_training_sample_pool(prepared,training_phase='actor_warmup',actor_warmup_data_scope='all')
            self.assertEqual(pool,(0,))
            self.assertEqual(record['target_source'],'human')
            replay_root = root/'bundle/replay_cache'
            prepared.save(replay_root)
            training_round = build_stage2_training_round(replay_root,
                expected_spec_fingerprint=prepared.spec_fingerprint,reference_seed=0,
                feature_batch_size=1,sampling_seed=23,batch_size=1,steps=1,starting_critic_updates=0)
            run.bind_training_round(training_round,replay_root=replay_root)
            run.learner.begin_phase('actor_warmup')
            run.learner.configure_actor_warmup_data(record)
            run.learner.update_phase(prepared.batch(pool,device='cpu'))
            run.save(root/'bundle')
            resumed=RLTStage2Run.resume(root/'bundle',device='cpu')
            self.assertEqual(resumed.learner.actor_warmup_data,record)
            self.assertEqual(resumed.learner.completed_actor_warmup_updates,1)

    def test_unknown_source_blocks_preparation_before_gr00t(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=_run(root,action_dim=16)
            with patch('runtime.rlt_data_validation._recorded_sources',return_value=(set(),{})),patch.object(rtc,'extract_tt_rtc') as extract:
                with self.assertRaisesRegex(ValueError,'action_source'):
                    rtc.materialize_tt_rtc_actor_replay((Source(root/'data'),),extractor=Extractor(),spec=run.learner.spec,output_root=root/'out')
                extract.assert_not_called()

if __name__=='__main__':unittest.main()

