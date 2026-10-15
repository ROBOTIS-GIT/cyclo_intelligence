"""Cancellation is task-scoped from capture through export and Qwen training."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from supervisor_api import collection as c
from supervisor_api import mission_decisions as contract
from supervisor_api import qwen_annotation_workflow as q
from supervisor_api.reasoning_contract import MissionDecisionRequest, validate_mission_decision
from supervisor_api.test_continuous_reasoning_data import CATALOG, FRAME, supervision

CANCEL = {'decision': 'cancel', 'mission_id': 'clean',
          'reason': 'The family has resumed eating, so clearing the table is no longer appropriate.'}


@pytest.mark.parametrize('status', ['running', 'paused'])
def test_live_and_training_cancel_have_the_same_active_task_and_prompt(status):
    context = supervision(status)
    request = MissionDecisionRequest(localized=True, **context)
    live = validate_mission_decision({'finish_reason': 'stop', 'text': json.dumps(CANCEL)}, CATALOG, request)
    sample = contract.sample_for(context, CATALOG, CANCEL, 'images/frame.jpg')
    assert live['action_allowed']
    assert live['selected_mission']['mission_id'] == 'clean'
    assert live['decision'] == json.loads(sample['messages'][-1]['content'][0]['text'])
    assert sample['messages'][0]['content'][0]['text'] == contract.SUPERVISION_SYSTEM
    assert 'cancel' in sample['messages'][0]['content'][0]['text']
    assert [m['mission_id'] for m in json.loads(sample['messages'][1]['content'][1]['text'])['missions']] == ['clean']


@pytest.mark.parametrize('context', [
    {}, {'active_mission': {'mission_id': 'clean', 'status': 'running'}},
    {'reasoning_task': 'mission_supervision'},
    {'reasoning_task': 'patrol_supervision', 'active_mission': None, 'observation': {'phase': 'returning'}},
    supervision('returning'), supervision('completed'),
    {**supervision(), 'observation': {'phase': 'returning'}},
    {**supervision(), 'active_mission': {'mission_id': 'clean', 'status': 'paused', 'action': 'return-posture'}},
])
def test_cancel_cannot_be_reinterpreted_as_selection_patrol_or_return(context):
    with pytest.raises(ValueError):
        contract.validate_target(CANCEL, context, CATALOG)


@pytest.mark.parametrize('mid', [None, '', 'serve'])
def test_cancel_must_name_the_active_mission(mid):
    with pytest.raises(ValueError):
        contract.validate_target({**CANCEL, 'mission_id': mid}, supervision(), CATALOG)


def test_legacy_return_labels_stay_readable_without_adding_cancel():
    context = supervision('returning')
    before = deepcopy(context)
    for action in ('continue', 'pause'):
        sample = contract.sample_for(context, CATALOG, {**CANCEL, 'decision': action}, 'frame.jpg')
        assert 'Do not cancel' in sample['messages'][0]['content'][0]['text']
        user = json.loads(sample['messages'][1]['content'][1]['text'])
        assert user['context']['active_mission']['status'] == 'returning'
    assert context == before
    assert contract.decisions_for(context) == ('continue', 'pause')


def test_cancel_round_trip_annotation_presets_export_and_balanced_training(tmp_path):
    store = c.CollectionStore(tmp_path / 'data', lambda _: CATALOG)
    ep = store.start(c.EpisodeRequest(map_name='Home', episode_type='mission', recording_mission_name='CleanTable'))
    for status in ('running', 'paused'):
        frame = store.capture(ep['id'], FRAME, supervision(status))
        store.annotate(ep['id'], c.AnnotationRequest(frame_id=frame['frame_id'], **CANCEL,
                       annotator='human', review_status='approved'))
    store.end(ep['id'])
    saved = store.get(ep['id'])
    assert saved['annotations'][0]['target'] == CANCEL
    presets = store.reason_presets('Home')
    assert presets[0]['decision'] == 'cancel' and presets[0]['mission_id'] == 'clean'
    output, _ = store._training_files(saved)
    samples = output['train.jsonl'] + output['val.jsonl'] + output['test.jsonl']
    assert json.loads(samples[0]['messages'][-1]['content'][0]['text']) == CANCEL
    workflow = q.QwenAnnotationWorkflow(store, Mock(), tmp_path / 'models', recover=False)
    with patch.object(q.threading, 'Thread'):
        model = workflow.train_reasoning(q.ReasoningTrainRequest(map_name='Home', episode_ids=[ep['id']],
                    model_name='Cancel supervisor', balance_decisions=True))
    assert model['input_policy'] == 'continuous_autonomy_v4'
    assert model['supervision_decision_counts'] == {'continue': 0, 'pause': 0, 'cancel': 2}
    training = c.read_json(Path(workflow.jobs['reasoning_training']['request_path']))
    assert training['sampling']['supervision_retained'] == 2
    assert json.loads(training['frames'][0]['sample']['messages'][-1]['content'][0]['text']) == CANCEL


def test_cancel_auto_label_is_draft_and_cannot_label_return_frames(tmp_path):
    store = c.CollectionStore(tmp_path / 'data', lambda _: CATALOG)
    ep = store.start(c.EpisodeRequest(map_name='Home', episode_type='mission', recording_mission_name='CleanTable'))
    frame = store.capture(ep['id'], FRAME, supervision())
    returning = store.capture(ep['id'], FRAME, supervision('returning'))
    store.end(ep['id'])
    workflow = q.QwenAnnotationWorkflow(store, Mock(), tmp_path / 'models', recover=False)
    model = {'schema': q.SCHEMA, 'id': 'b'*32, 'map_name': 'Home', 'model_name': 'Supervisor',
             'catalog_snapshot_id': c.digest(CATALOG)}
    c.write_json(workflow.root / 'Home' / 'Supervisor' / 'model.json', model)
    with patch.object(q.threading, 'Thread'):
        job = workflow.batch(q.BatchRequest(model_id=model['id'], episode_ids=[ep['id']]))
    request = c.read_json(Path(workflow.jobs['batch']['request_path']))
    rows = {row['frame_id']: row for row in request['frames']}
    assert workflow._prediction(rows[frame['frame_id']], {'text': json.dumps(CANCEL)}, job['id']) == 'created'
    label = store.get(ep['id'])['annotations'][0]
    assert label['review_status'] == 'draft' and label['target'] == CANCEL
    assert returning['frame_id'] not in rows  # Post-mission return cannot be supervised as an active task.
    assert set(rows) == {frame['frame_id']}
    with pytest.raises(HTTPException):
        store.annotate(ep['id'], c.AnnotationRequest(frame_id=returning['frame_id'], **CANCEL,
                       annotator='human', review_status='approved'))
