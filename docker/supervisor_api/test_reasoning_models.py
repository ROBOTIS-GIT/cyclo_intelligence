import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from reasoning_models import ReasoningModels, catalog_digest, load_options, matches_loaded
from reasoning import ReasoningService, StartRequest, MissionDecisionRequest
import reasoning


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = [{'mission_id': 'm2', 'revision': 1}]
        self.registry = ReasoningModels(lambda _: self.catalog, self.root)
        adapter = self.root/'annotation/warehouse/custom/adapter'
        adapter.mkdir(parents=True)
        (adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','base_model_name_or_path':'Qwen/Qwen3.5-4B'}))
        (adapter/'adapter_model.safetensors').write_bytes(b'test-weights')
        self.model = {'schema':'qwen_mission_annotation_v3','id':'trained','map_name':'warehouse',
                      'model_name':'My trained model','base_model':'Qwen/Qwen3.5-4B','revision':'rev',
                      'adapter_path':str(adapter),'sample_count':3}
        self.meta = adapter.parent/'model.json'
        self.meta.write_text(json.dumps(self.model))

    def test_only_completed_matching_map_models_are_offered(self):
        self.assertEqual(self.registry.list('warehouse')[0]['id'], 'trained')
        self.assertIsNone(self.registry.list('warehouse')[0]['catalog_compatible'])
        self.assertFalse([m for m in self.registry.list('other') if m['source'] != 'base'])
        with self.assertRaises(HTTPException): self.registry.resolve('trained','other')
        Path(self.model['adapter_path'],'adapter_model.safetensors').unlink()
        self.assertFalse([m for m in self.registry.list('warehouse') if m['source'] != 'base'])
        with self.assertRaises(HTTPException): self.registry.resolve('trained','warehouse')

    def test_rejects_incompatible_metadata_base_model_and_escaped_files(self):
        for field,value in [('schema','old_description_contract'),('base_model','Qwen/Qwen3.5-9B'),('adapter_path','/tmp')]:
            self.meta.write_text(json.dumps({**self.model,field:value}))
            self.assertFalse([m for m in self.registry.list('warehouse') if m['source'] != 'base'])
        self.meta.write_text(json.dumps(self.model))
        weights=Path(self.model['adapter_path'],'adapter_model.safetensors')
        weights.unlink(); weights.symlink_to('/etc/hosts')
        self.assertFalse([m for m in self.registry.list('warehouse') if m['source'] != 'base'])

    def test_base_checkpoints_are_selectable_without_an_adapter(self):
        models = [m for m in self.registry.list('warehouse') if m['source'] == 'base']
        self.assertEqual({m['base_model'] for m in models},
                         {'Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B'})
        model = self.registry.resolve('base:Qwen3.5-2B', 'warehouse')
        self.assertEqual(model['revision'], '15852e8c16360a2fea060d615a32b45270f8a8fc')
        self.assertEqual(load_options(model)['adapter_path'], '')
        self.assertTrue(matches_loaded({'model': model['base_model'],
                                        'revision': model['revision'], 'adapter': None}, model))
        self.assertEqual(model['map_name'], 'warehouse')
        small = self.registry.resolve('base:Qwen3.5-0.8B', 'warehouse')
        self.assertEqual(small['revision'], '2fc06364715b967f1860aea9cf38778875588b17')
        self.assertEqual(small['adapter_path'], '')

    def test_reports_catalog_compatibility_against_live_design(self):
        self.meta.write_text(json.dumps({**self.model,
            'catalog_snapshot_id':catalog_digest(self.catalog),
            'mission_ids':['m2'],'mission_revisions':{'m2':1}}))
        current=self.registry.list('warehouse')[0]
        self.assertTrue(current['catalog_compatible'])
        self.catalog.append({'mission_id':'observe','revision':1})
        stale=self.registry.list('warehouse')[0]
        self.assertFalse(stale['catalog_compatible'])
        self.assertEqual(stale['current_catalog_snapshot_id'],catalog_digest(self.catalog))

    def test_compatibility_uses_actual_training_snapshots_and_supervision_counts(self):
        current_id = catalog_digest(self.catalog)
        model = {**self.model, 'catalog_snapshot_id': current_id,
                 'training_catalog_snapshot_ids': ['older-recorded-catalog', current_id],
                 'reasoning_tasks': ['mission_selection', 'mission_supervision'],
                 'task_sample_counts': {'mission_selection': 20, 'mission_supervision': 0},
                 'supervision_decision_counts': {'pause': 0, 'continue': 0}}
        self.meta.write_text(json.dumps(model))
        saved = self.registry.list('warehouse')[0]
        self.assertFalse(saved['catalog_compatible'])
        self.assertFalse(saved['supervision_trained'])
        model.update(training_catalog_snapshot_ids=[current_id],
                     task_sample_counts={'mission_supervision': 2},
                     supervision_decision_counts={'pause': 1, 'continue': 1})
        self.meta.write_text(json.dumps(model))
        saved = self.registry.list('warehouse')[0]
        self.assertTrue(saved['catalog_compatible'])
        self.assertTrue(saved['supervision_trained'])
        model.update(task_sample_counts={'mission_supervision': 1},
                     supervision_decision_counts={'pause': 0, 'continue': 0, 'cancel': 1})
        self.meta.write_text(json.dumps(model))
        self.assertTrue(self.registry.list('warehouse')[0]['supervision_trained'])

    def test_final_reasoning_models_are_separate_and_new_auto_label_models_are_hidden(self):
        final_adapter=self.root/'reasoning/warehouse/final/adapter'
        final_adapter.mkdir(parents=True)
        for name in ('adapter_config.json','adapter_model.safetensors'):
            (final_adapter/name).write_bytes((Path(self.model['adapter_path'])/name).read_bytes())
        final={**self.model,'id':'final','model_name':'Final reasoning','purpose':'reasoning',
               'adapter_path':str(final_adapter)}
        (final_adapter.parent/'model.json').write_text(json.dumps(final))
        self.meta.write_text(json.dumps({**self.model,'purpose':'auto_labeling'}))
        models=self.registry.list('warehouse')
        self.assertEqual([model['id'] for model in models if model['source'] != 'base'],['final'])
        self.assertEqual(models[0]['source'],'fine_tuning')

    def test_legacy_reports_must_match_missions_and_output_contract(self):
        adapter=self.root/'old-model';adapter.mkdir()
        for name in ('adapter_config.json','adapter_model.safetensors'):
            (adapter/name).write_bytes((Path(self.model['adapter_path'])/name).read_bytes())
        report={'model':'Qwen/Qwen3.5-4B','revision':'rev','samples':1,
                'targets':[{'decision':'start','mission_id':'m2','reason':'Visible'}]}
        path=adapter/'training_report.json';path.write_text(json.dumps(report))
        self.assertEqual(self.registry.resolve('legacy:old-model','warehouse')['source'],'legacy')
        report['targets'][0]['mission_id']='foreign';path.write_text(json.dumps(report))
        self.assertEqual(len([m for m in self.registry.list('warehouse') if m['source'] != 'base']),1)
        report['targets']=[{'description':'old format'}];path.write_text(json.dumps(report))
        self.assertEqual(len([m for m in self.registry.list('warehouse') if m['source'] != 'base']),1)


class SelectedModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.model={'id':'trained','map_name':'warehouse','model_name':'Custom',
                    'base_model':'Qwen/Qwen3.5-4B','revision':'rev','adapter_path':'/models/custom'}
        self.registry=Mock();self.registry.resolve.return_value=self.model
        self.store=Mock();self.store.catalog_provider.return_value=[{'mission_id':'m2','map_name':'warehouse','mission_name':'Mission2'}]
        self.service=ReasoningService(Mock(),Mock(),self.store,self.registry)
        self.health={'ready':True,'model':self.model['base_model'],'revision':'rev','adapter':'/models/custom'}

    def start_without_thread(self):
        with patch.object(reasoning.threading,'Thread'):
            state=self.service.start(StartRequest(model_id='trained',map_name='warehouse'))
        return state

    def test_switches_from_preloaded_base_to_selected_adapter_and_verifies(self):
        state=self.start_without_thread()
        self.assertEqual(state['selected_model_id'],'trained')
        with patch.object(reasoning,'qwen_request',side_effect=[{'ready':True,'model':'old'}, {}, self.health]) as call:
            self.service.prepare()
        self.assertEqual(call.call_args_list[1].args,('/load',{'model_id':self.model['base_model'],'revision':'rev','adapter_path':'/models/custom'}))
        self.assertEqual(self.service.status()['phase'],'ready')
        self.assertEqual(self.service.status()['model_name'],'Custom')

    def test_load_mismatch_or_failure_never_falls_back_to_old_model(self):
        for response in ([{'ready':True}, {}, {'ready':True,'model':'old'}], [{'ready':False}, HTTPException(503,'Failed')]):
            self.start_without_thread()
            with patch.object(reasoning,'qwen_request',side_effect=response):self.service.prepare()
            self.assertEqual(self.service.status()['phase'],'error')
            self.assertIsNone(self.service.status()['model'])

    def test_different_model_during_loading_requires_stop(self):
        self.start_without_thread()
        for phase in ('starting','loading'):
            self.service.update(phase=phase)
            self.registry.resolve.return_value={**self.model,'id':'other'}
            with self.assertRaises(HTTPException) as exc:
                self.service.start(StartRequest(model_id='other',map_name='warehouse'))
            self.assertEqual(exc.exception.status_code,409)
        self.service.prepare_lock.release()

    def test_stale_session_and_other_map_rejected_before_camera(self):
        state=self.start_without_thread();self.service.prepare_lock.release();self.service.update(phase='ready')
        for req in (MissionDecisionRequest(map_name='warehouse',localized=True,expected_session_id='old'),
                    MissionDecisionRequest(map_name='other',localized=True,expected_session_id=state['session_id'])):
            with patch.object(reasoning,'capture_head') as capture:
                with self.assertRaises(HTTPException) as exc:self.service.decide(req)
                self.assertEqual(exc.exception.status_code,412);capture.assert_not_called()

    def test_only_selected_models_decision_is_executable(self):
        state=self.start_without_thread();self.service.prepare_lock.release()
        self.service.update(phase='ready',model=self.health['model'],model_revision='rev',adapter='/models/custom')
        req=MissionDecisionRequest(map_name='warehouse',localized=True,expected_session_id=state['session_id'])
        generated={**self.health,'text':json.dumps({'decision':'start','mission_id':'m2','reason':'Visible'}),'finish_reason':'stop'}
        with patch.object(reasoning,'capture_head',return_value={'frame_stamp':'1','image_base64':'a'}):
            with patch.object(reasoning,'qwen_request',return_value=generated):
                out=self.service.decide(req)
                self.assertTrue(out['action_allowed']);self.assertEqual(out['selected_model_id'],'trained')
            with patch.object(reasoning,'qwen_request',return_value={**generated,'adapter':'wrong'}):
                out=self.service.decide(req)
                self.assertFalse(out['action_allowed']);self.assertIn('differs',out['error'])
