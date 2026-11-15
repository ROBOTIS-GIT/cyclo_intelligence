"""TT-RTC decision inputs, terminal returns, and offline replay reuse."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cyclo_brain.algorithm.rl.rlt import RLTStage2Run, build_stage2_training_round
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run
from cyclo_brain.policy.groot.tests.test_rlt_tt_rtc_actor_replay import Source, Extractor
from runtime import rlt_tt_rtc_actor_replay as rtc
from runtime import rlt_tt_rtc_transition_replay as td
from runtime.rlt_stage2_dataset import RLTStage2FeatureReplay
from runtime.rlt_stage2_training_cli import _training_sample_pool
from runtime.rlt_saved_replay import load_saved_replay, has_fresh_tt_replay


def _actor_replay(root, *, length=40, source_name='human'):
    run = _run(root/'encoder', action_dim=16)
    source = Source(root/'dataset', length=length)
    source.root.mkdir(parents=True)
    labels = {(0, i): source_name for i in range(length)}
    features = lambda _e, observation, _p, _s: dict(
        z_rl=torch.full((8,), float(observation['frame'])),
        proprio=torch.full((16,), float(observation['frame'])),
        reference_actions=torch.full((10, 16), float(observation['frame'])),
    )
    snapshot = json.loads((root/'encoder/replay_cache/manifest.json').read_text())['metadata']['dataset_snapshots'][0]
    source.content_snapshot = lambda: snapshot
    with patch('runtime.rlt_data_validation._recorded_sources', return_value=(set(), labels)), \
            patch.object(rtc, 'extract_tt_rtc', side_effect=features):
        actor = rtc.materialize_tt_rtc_actor_replay(
            (source,), extractor=Extractor(), spec=run.learner.spec,
            output_root=source.root/'rlt/tt_rtc_actor_v1/replay_cache')
    return run, source, actor


class TTTransitionReplayTests(unittest.TestCase):
    def test_decision_interval_next_inputs_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,actor=_actor_replay(root)
            replay=td.build_tt_rtc_transition_replay(actor, output_root=root/'td')
            self.assertEqual(replay.metadata['replay_role'],td.ROLE)
            self.assertEqual(replay.metadata['config']['decision_interval_steps'],10)
            torch.testing.assert_close(replay.tensors['next_z_rl'][0],torch.full((8,),10.0))
            torch.testing.assert_close(replay.tensors['next_proprio'][0],torch.full((16,),10.0))
            torch.testing.assert_close(replay.tensors['next_reference_actions'][0],torch.full((10,16),10.0))
            torch.testing.assert_close(replay.tensors['next_committed_prefix'][0,:,0],torch.arange(110,116).float())
            row=replay.metadata['row_provenance'][0]
            self.assertEqual(row['next_request_after_action_steps'],4)
            self.assertEqual(row['next_observation_frame_index'],10)
            self.assertEqual(row['target_kind'],'decision_td')
            self.assertAlmostEqual(float(replay.tensors['bootstrap_discount'][0]),0.99**10,places=6)

    def test_terminal_observed_return_keeps_complete_current_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,actor=_actor_replay(root)
            replay=td.build_tt_rtc_transition_replay(actor,output_root=root/'td')
            self.assertEqual(len(replay),25)
            self.assertEqual(replay.metadata['terminal_return_count'],10)
            for i in range(15):
                self.assertEqual(replay.metadata['row_provenance'][i]['target_kind'],'decision_td')
            for i in range(15,25):
                self.assertEqual(replay.metadata['row_provenance'][i]['target_kind'],'observed_terminal_return')
                self.assertEqual(replay.metadata['row_provenance'][i]['target_end_frame_index'],40)
                self.assertAlmostEqual(float(replay.tensors['reward'][i]),0.99**(39-i),places=6)
                self.assertEqual(float(replay.tensors['bootstrap_discount'][i]),0.0)
            metadata={**actor.metadata,'row_provenance':[
                {**row,'episode_success':False} for row in actor.metadata['row_provenance']]}
            failed=RLTStage2FeatureReplay(actor.spec,{k:v.clone() for k,v in actor.tensors.items()},metadata=metadata)
            failure=td.build_tt_rtc_transition_replay(failed,output_root=root/'failure_td')
            self.assertEqual(float(failure.tensors['reward'].sum()),0.0)

    def test_mlp_only_recordings_train_critic_but_are_not_human_warmup_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,actor=_actor_replay(root,source_name='mlp')
            replay=td.build_tt_rtc_transition_replay(actor,output_root=root/'td')
            with self.assertRaisesRegex(ValueError,'human action targets'):
                _training_sample_pool(replay,training_phase='actor_warmup',actor_warmup_data_scope='all')
            for phase in ('critic_warmup','rl'):
                indices,_=_training_sample_pool(replay,training_phase=phase,actor_warmup_data_scope='all')
                self.assertEqual(indices,tuple(range(len(replay))))

    def test_three_phases_and_bundle_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,_,actor=_actor_replay(root)
            replay_root=root/'bundle/replay_cache'
            replay=td.build_tt_rtc_transition_replay(actor,output_root=replay_root)
            round_=build_stage2_training_round(replay_root,expected_spec_fingerprint=replay.spec_fingerprint,
                reference_seed=0,feature_batch_size=1,sampling_seed=23,batch_size=1,steps=4,starting_critic_updates=0)
            run.bind_training_round(round_,replay_root=replay_root)
            for phase in ('actor_warmup','critic_warmup','rl'):
                run.learner.begin_phase(phase)
                if phase=='actor_warmup':
                    _,record=_training_sample_pool(replay,training_phase=phase,actor_warmup_data_scope='all')
                    run.learner.configure_actor_warmup_data(record)
                steps=2 if phase=='rl' else 1
                for step in range(steps):run.learner.update_phase(replay.batch([step],device='cpu'))
                run.learner.complete_phase()
            run.save(root/'bundle')
            resumed=RLTStage2Run.resume(root/'bundle',device='cpu')
            self.assertEqual(resumed.learner.completed_critic_updates,3)
            self.assertEqual(resumed.learner.completed_actor_warmup_updates,1)
            self.assertEqual(resumed.learner.actor_warmup_data['target_source'],'human')

    def test_prepare_reuses_saved_actor_replay_without_importing_gr00t(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,source,actor=_actor_replay(root)
            checkpoint=root/'groot';checkpoint.mkdir()
            self.assertTrue(has_fresh_tt_replay([source.root],'actor_warmup'))
            original=(source.root/'rlt/tt_rtc_actor_v1/replay_cache/replay.pt').read_bytes()
            with patch('runtime.rlt_data_validation.open_rlt_stage2_source',return_value=source), \
                    patch.dict(sys.modules,{'gr00t':None}):
                loaded=load_saved_replay([source.root],spec=run.learner.spec,
                    output_root=root/'loaded',training_phase='actor_warmup')
            self.assertTrue(loaded.actor_only)
            transition_root=source.root/'rlt/tt_rtc_transition_v1/replay_cache'
            td.build_tt_rtc_transition_replay(actor,output_root=transition_root)
            self.assertTrue(has_fresh_tt_replay([source.root],'rl'))
            with patch('runtime.rlt_data_validation.open_rlt_stage2_source',return_value=source), \
                    patch.dict(sys.modules,{'gr00t':None}):
                loaded_td=load_saved_replay([source.root],spec=run.learner.spec,
                    output_root=root/'loaded_td',training_phase='rl')
            self.assertEqual(loaded_td.metadata['replay_role'],td.ROLE)
            self.assertEqual(original,(source.root/'rlt/tt_rtc_actor_v1/replay_cache/replay.pt').read_bytes())
            rows=[dict(row) for row in actor.metadata['row_provenance']]
            rows[0]['action_start_frame_index']+=1
            invalid=RLTStage2FeatureReplay(actor.spec,{k:v.clone() for k,v in actor.tensors.items()},
                metadata={**actor.metadata,'row_provenance':rows})
            with self.assertRaises(ValueError):td.build_tt_rtc_transition_replay(invalid,output_root=root/'bad')


if __name__=='__main__':unittest.main()
