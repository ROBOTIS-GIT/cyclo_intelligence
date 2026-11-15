"""Independent MLP joint normalization and terminal-hold regression tests."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cyclo_brain.algorithm.rl.rlt.normalization import (
    active_joint_indices, apply_fixed_actions, denormalize, fit_normalization,
    normalize, reference_to_mlp, validate_normalization,
)
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _batch, _run
from cyclo_brain.policy.groot.tests.test_rlt_tt_rtc_actor_replay import Source, Extractor
from runtime import rlt_tt_rtc_actor_replay as rtc
from runtime.rlt_adapter import GR00TRLTInferenceAdapter
from runtime.rlt_mlp_replay import prepare_mlp_replay, prepare_mlp_transition_replay
from runtime.rlt_provenance import rlt_action_group_names
from runtime.rlt_tt_rtc_transition_replay import build_tt_rtc_transition_replay


def _names(dimension=16):
    return [name for names in rlt_action_group_names(dimension).values() for name in names]


def _normalization(dimension=16):
    states=np.stack([np.full(dimension,-1,dtype=np.float32),np.full(dimension,1,dtype=np.float32)])
    actions=np.stack([np.full(dimension,-2,dtype=np.float32),np.full(dimension,2,dtype=np.float32)])
    return fit_normalization(states,actions,joint_names=_names(dimension),
        vla_lower=[-4]*dimension,vla_upper=[4]*dimension,source_fingerprint='a'*64)


def _write_checkpoint(root, dimension=16):
    groups=rlt_action_group_names(dimension)
    configs=[{'rep':'ABSOLUTE','type':'NON_EEF'} for _ in groups]
    processor={'processor_kwargs':{'use_mean_std':False,'use_percentiles':False,
        'modality_configs':{'new_embodiment':{'action':{
            'modality_keys':list(groups),'action_configs':configs}}}}}
    statistics={'new_embodiment':{'action':{
        key:{'min':[-4.0]*len(names),'max':[4.0]*len(names)}
        for key,names in groups.items()}}}
    root.mkdir()
    (root/'processor_config.json').write_text(json.dumps(processor))
    (root/'statistics.json').write_text(json.dumps(statistics))


class MLPNormalizationTests(unittest.TestCase):
    def test_fitted_bounds_margin_floor_and_round_trip_are_unclipped(self):
        d=16;states=np.zeros((3,d),dtype=np.float32);actions=np.zeros((3,d),dtype=np.float32)
        config=fit_normalization(states,actions,joint_names=_names(d),vla_lower=[-1]*d,
            vla_upper=[1]*d,source_fingerprint='b'*64)
        arm_span=np.deg2rad(1.0)*1.1
        self.assertAlmostEqual(config['state']['upper'][0]-config['state']['lower'][0],arm_span)
        self.assertAlmostEqual(config['action']['upper'][7]-config['action']['lower'][7],.022)
        values=np.full((2,d),3.0,dtype=np.float32)
        np.testing.assert_allclose(denormalize(normalize(values,config),config),values,rtol=1e-5,atol=1e-5)
        self.assertGreater(float(normalize(values,config).max()),1.0)

    def test_validation_fingerprint_canonical_order_and_fixed_policy(self):
        config=_normalization()
        self.assertEqual(validate_normalization(config,16),config)
        self.assertEqual(active_joint_indices(config),list(range(7,15)))
        damaged=dict(config);damaged['fingerprint']='0'*64
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            validate_normalization(damaged,16)
        damaged=json.loads(json.dumps(config));damaged['joint_names'][0],damaged['joint_names'][1]=damaged['joint_names'][1],damaged['joint_names'][0]
        with self.assertRaises(ValueError):validate_normalization(damaged,16)

    def test_reference_decode_clips_only_frozen_vla_and_fixed_outputs(self):
        config=_normalization();reference=np.full((1,10,16),2.0,dtype=np.float32)
        converted=reference_to_mlp(reference,config)
        expected=normalize(np.full((1,10,16),4.0,dtype=np.float32),config)
        np.testing.assert_allclose(converted,expected)
        physical=np.zeros((1,10,16),dtype=np.float32)
        held=apply_fixed_actions(physical,config,np.full((1,7),.25,dtype=np.float32))
        np.testing.assert_array_equal(held[0,:,0:7],np.full((10,7),.25,dtype=np.float32))
        np.testing.assert_array_equal(held[0,:,15],np.full(10,.9,dtype=np.float32))

    def test_terminal_hold_replay_is_rebuilt_from_raw_success_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=_run(root/'run',action_dim=16);source=Source(root/'dataset',length=3)
            source.root.mkdir();snapshot=json.loads((root/'run/replay_cache/manifest.json').read_text())['metadata']['dataset_snapshots'][0]
            source.content_snapshot=lambda:snapshot
            labels={(0,i):'human' for i in range(3)}
            features=dict(z_rl=torch.ones(8),proprio=torch.ones(16),reference_actions=torch.zeros(10,16))
            with patch('runtime.rlt_data_validation._recorded_sources',return_value=(set(),labels)), \
                    patch.object(rtc,'extract_tt_rtc',return_value=features):
                replay=rtc.materialize_tt_rtc_actor_replay((source,),extractor=Extractor(),
                    spec=run.learner.spec,output_root=root/'actor',terminal_hold=True)
            checkpoint=root/'groot';_write_checkpoint(checkpoint)
            with patch('runtime.rlt_data_validation.open_rlt_stage2_source',return_value=source), \
                    patch('runtime.rlt_mlp_replay.open_rlt_stage2_source',return_value=source):
                prepared,config=prepare_mlp_replay(replay,checkpoint,output_root=root/'mlp')
            self.assertEqual(len(prepared),3)
            self.assertEqual(prepared.metadata['mlp_normalization_id'],config['fingerprint'])
            self.assertEqual(prepared.metadata['terminal_hold_policy'],rtc.TERMINAL_HOLD_POLICY)
            raw_last=np.full((10,16),102,dtype=np.float32)
            np.testing.assert_allclose(prepared.tensors['executed_actions'][-1].numpy(),normalize(raw_last,config))
            with patch('runtime.rlt_data_validation.open_rlt_stage2_source',return_value=source), \
                    patch('runtime.rlt_mlp_replay.open_rlt_stage2_source',return_value=source):
                copied,same=prepare_mlp_replay(prepared,checkpoint,output_root=root/'copy',expected_normalization=config)
            self.assertEqual(copied.metadata['mlp_normalization_id'],same['fingerprint'])

    def test_actor_warmup_ignores_held_and_fixed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=_run(Path(tmp),action_dim=16);config=_normalization()
            run.learner.config=replace(run.learner.config,mlp_normalization=config)
            run.learner.actor.mlp_normalization=config
            batch=_batch(run)
            batch=replace(batch,executed_actions=torch.ones_like(batch.executed_actions),
                mlp_normalization_id=config['fingerprint'])
            batch.executed_actions[...,0:7]=1000
            batch.executed_actions[...,15]=1000
            for parameter in run.learner.actor.parameters():parameter.data.zero_()
            run.learner.begin_phase('actor_warmup')
            update=run.learner.update_phase(batch)
            self.assertAlmostEqual(update.actor_loss,1.0,places=6)
            run.learner.complete_phase()
            prefix=torch.randn(batch.batch_size,6,16)
            batch=replace(batch,committed_prefix=prefix,next_committed_prefix=prefix.clone())
            projected=run.learner._apply_mlp_action_policy(
                torch.zeros_like(batch.executed_actions), prefix
            )
            torch.testing.assert_close(
                projected[...,0:7],
                prefix[:,-1:,0:7].expand(-1,projected.shape[1],-1),
            )
            expected_gripper=normalize(
                np.full((1,1,16),.9,dtype=np.float32),config
            )[0,0,15]
            torch.testing.assert_close(
                projected[...,15],
                torch.full_like(projected[...,15],float(expected_gripper)),
            )
            run.learner.begin_phase('critic_warmup')
            critic=run.learner.update_phase(batch)
            self.assertTrue(np.isfinite(critic.critic_loss))
            run.learner.complete_phase()
            run.learner.begin_phase('rl')
            batch=replace(
                batch,
                episode_success=torch.tensor([True,False,True]),
            )
            for _ in range(2):
                update=run.learner.update_phase(batch)
            self.assertTrue(np.isfinite(update.critic_loss))
            self.assertIsNotNone(update.actor_loss)

    def test_real_transition_replay_migrates_to_actor_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=_run(root/'run',action_dim=16)
            source=Source(root/'dataset',length=26);source.root.mkdir()
            labels={(0,i):'human' for i in range(source.length)}
            features=dict(z_rl=torch.ones(8),proprio=torch.ones(16),
                          reference_actions=torch.zeros(10,16))
            with patch('runtime.rlt_data_validation._recorded_sources',return_value=(set(),labels)), \
                    patch.object(rtc,'extract_tt_rtc',return_value=features):
                actor=rtc.materialize_tt_rtc_actor_replay(
                    (source,),extractor=Extractor(),spec=run.learner.spec,
                    output_root=root/'actor',terminal_hold=False)
            transition=build_tt_rtc_transition_replay(
                actor,output_root=root/'transition')
            config=_normalization()
            with patch('runtime.rlt_data_validation.open_rlt_stage2_source',return_value=source), \
                    patch('runtime.rlt_mlp_replay.open_rlt_stage2_source',return_value=source):
                prepared,same=prepare_mlp_transition_replay(
                    transition,output_root=root/'mlp_transition',
                    expected_normalization=config)
            self.assertEqual(same,config)
            self.assertEqual(prepared.metadata['mlp_normalization_id'],config['fingerprint'])
            self.assertEqual(
                prepared.metadata['mlp_coordinate_contract'],
                'state_action_reference_and_prefix/v1',
            )
            raw_state=np.zeros(16,dtype=np.float32)
            np.testing.assert_allclose(
                prepared.tensors['proprio'][0].numpy(),
                normalize(raw_state,config,'state'))
            raw_actions=np.arange(106,116,dtype=np.float32)[:,None].repeat(16,axis=1)
            np.testing.assert_allclose(
                prepared.tensors['executed_actions'][0].numpy(),
                normalize(raw_actions,config))
            nonterminal=prepared.tensors['bootstrap_discount'][:,0] > 0
            self.assertTrue(bool(nonterminal.any()))
            self.assertTrue(bool((~nonterminal).any()))
            torch.testing.assert_close(
                prepared.tensors['next_proprio'][~nonterminal],
                torch.zeros_like(prepared.tensors['next_proprio'][~nonterminal]),
            )
            batch=prepared.batch([0],device='cpu')
            batch.validate(run.learner.spec)
            self.assertEqual(batch.committed_prefix.shape,(1,6,16))

    def test_runtime_uses_raw_state_mlp_codec_and_prefix_hold(self):
        config=_normalization();a=object.__new__(GR00TRLTInferenceAdapter)
        a.shadow_policy=SimpleNamespace(actor=SimpleNamespace(mlp_normalization=config))
        a.spec=SimpleNamespace(action_dim=16)
        a.policy=SimpleNamespace(modality_configs={
            'state':SimpleNamespace(modality_keys=['arm_left','arm_right']),
            'action':SimpleNamespace(modality_keys=['arm_left','arm_right'])})
        states={k:np.full((1,1,8),.4,dtype=np.float32) for k in ['arm_left','arm_right']};prefix=np.full((1,6,16),.3,dtype=np.float32)
        x,ref=a._mlp_inputs(torch.zeros(1,16),torch.zeros(1,32,16),states,prefix)
        np.testing.assert_allclose(x.numpy(),normalize(np.full((1,16),.4,dtype=np.float32),config,'state'))
        np.testing.assert_allclose(ref.numpy(),reference_to_mlp(np.zeros((1,32,16),dtype=np.float32),config))
        decoded=a._decode_candidate(torch.zeros(1,10,16),states)
        np.testing.assert_array_equal(decoded['arm_left'][...,0:7],np.full((1,10,7),.3,dtype=np.float32))
        np.testing.assert_array_equal(decoded['arm_right'][...,7],np.full((1,10),.9,dtype=np.float32))


if __name__=='__main__':unittest.main()
