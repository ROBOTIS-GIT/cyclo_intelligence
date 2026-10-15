"""Saved spot deletion must not leave dangling observation references."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from supervisor_api import autonomy_config as config
from supervisor_api.autonomy import create_router
from supervisor_api.autonomy_config import AutonomyRepository, ObservationPlan, ObservationPoint


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config.spots, 'SPOTS_ROOT', tmp_path)
    monkeypatch.setattr(config.missions, 'collection_catalog', lambda _: [])
    for sid, label in [('observation', 'Observation'), ('accidental', 'Observation 5'), ('table', 'Table')]:
        config.spots.create_spot(config.spots.SpotCreateRequest(
            map_name='Home', id=sid, label=label, pose=config.spots.SpotPose(x=0, y=0)))
    repo = AutonomyRepository()
    repo.save('Home', ObservationPlan(observations=[ObservationPoint(spot_id='observation')]))
    return repo


def test_delete_unused_spot_keeps_plan_and_other_spots(store):
    before = store.read('Home')
    store.delete_spot('Home', 'accidental', before.revision)
    assert store.read('Home') == before
    assert [s.id for s in config.spots._list_spots('Home')] == ['observation', 'table']
    # Reload from disk as a new API request would.
    assert [s['id'] for s in AutonomyRepository().view('Home')['spots']] == ['observation', 'table']


def test_delete_referenced_spot_removes_plan_entry_and_increments_revision(store):
    store.delete_spot('Home', 'observation', 1)
    plan = store.read('Home')
    assert plan.revision == 2
    assert plan.observations == []
    store.resolve('Home', plan)
    assert [s.id for s in config.spots._list_spots('Home')] == ['accidental', 'table']


def test_failed_delete_restores_plan(store, monkeypatch):
    before = store.read('Home')
    def fail(*_, **__): raise HTTPException(503, 'Disk full')
    monkeypatch.setattr(config.spots, 'delete_spot', fail)
    with pytest.raises(HTTPException): store.delete_spot('Home', 'observation', 1)
    assert store.read('Home') == before
    assert len(config.spots._list_spots('Home')) == 3


def test_stale_delete_does_not_remove_new_plan_reference(store):
    plan = store.read('Home')
    plan.observations.append(ObservationPoint(spot_id='accidental'))
    store.save('Home', plan)
    with pytest.raises(HTTPException) as error: store.delete_spot('Home', 'accidental', 1)
    assert error.value.status_code == 409
    assert len(config.spots._list_spots('Home')) == 3
    assert len(store.read('Home').observations) == 2


def test_unknown_id_and_wrong_map_do_not_modify_saved_spots(store):
    for map_name, spot_id, revision in [('Home', 'missing', 1), ('Other', 'accidental', 0)]:
        with pytest.raises(HTTPException) as error: store.delete_spot(map_name, spot_id, revision)
        assert error.value.status_code == 404
    assert len(config.spots._list_spots('Home')) == 3


def test_api_blocks_delete_during_execution_and_returns_updated_plan(store):
    controller = SimpleNamespace(lock=threading.RLock(), owned=True, repository=store)
    app = FastAPI(); app.include_router(create_router(controller))
    client = TestClient(app)
    url = '/autonomy/maps/Home/spots/observation?plan_revision=1'
    assert client.delete(url).status_code == 409
    assert len(store.read('Home').observations) == 1
    controller.owned = False
    response = client.delete(url)
    assert response.status_code == 200
    assert response.json()['plan']['observations'] == []
    assert all(s['id'] != 'observation' for s in response.json()['spots'])


def test_record_creates_and_edits_same_spot_then_operate_resolves_it(store):
    controller = SimpleNamespace(lock=threading.RLock(), owned=False, repository=store)
    app = FastAPI(); app.include_router(create_router(controller))
    client = TestClient(app)
    body = {'label': 'Laundry', 'pose': {'x': 1, 'y': 2, 'yaw': 0.5}, 'plan_revision': 1}
    response = client.post('/autonomy/maps/Home/spots', json=body)
    assert response.status_code == 200, response.text
    spot = response.json()
    assert spot['metadata']['role'] == 'observation'
    assert store.read('Home').revision == 2
    plan = store.read('Home')
    plan.observations.append(ObservationPoint(spot_id=spot['id']))
    store.save('Home', plan)
    body.update(label='Laundry view', pose={'x': 3, 'y': 4, 'yaw': 1.5}, plan_revision=3)
    url = '/autonomy/maps/Home/spots/' + spot['id']
    response = client.patch(url, json=body)
    assert response.status_code == 200, response.text
    assert response.json()['id'] == spot['id']
    resolved, _ = store.resolve('Home', store.read('Home'))
    assert resolved[-1]['label'] == 'Laundry view'
    assert resolved[-1]['pose']['x'] == 3
    assert client.patch(url, json=body).status_code == 409
    assert store.read('Home').revision == 4
    assert client.delete(url + '?plan_revision=4').status_code == 200
    assert [p.spot_id for p in store.read('Home').observations] == ['observation']


def test_record_rejects_running_autonomy_and_invalid_observation_edits(store):
    controller = SimpleNamespace(lock=threading.RLock(), owned=True, repository=store)
    app = FastAPI(); app.include_router(create_router(controller))
    client = TestClient(app)
    body = {'label': 'New', 'pose': {'x': 1, 'y': 2, 'yaw': 0}, 'plan_revision': 1}
    base = '/autonomy/maps/Home/spots'
    assert client.post(base, json=body).status_code == 409
    assert client.patch(base + '/observation', json=body).status_code == 409
    controller.owned = False
    assert client.patch(base + '/table', json=body).status_code == 422
    assert client.patch(base.replace('Home', 'Other') + '/observation', json={**body, 'plan_revision': 0}).status_code == 404
    assert client.post(base, json={**body, 'label': ' '}).status_code == 422
    assert client.post(base, json={**body, 'pose': {**body['pose'], 'frame_id': 'odom'}}).status_code == 422
    # A legacy observation referenced by the plan is editable without migrating its ID.
    assert client.patch(base + '/observation', json=body).status_code == 200
    assert store.read('Home').observations[0].spot_id == 'observation'


def test_episode_keeps_observation_snapshot_after_spot_is_edited_or_deleted(store, tmp_path):
    from supervisor_api.collection import CollectionStore, EpisodeRequest
    from supervisor_api.autonomy_routes import ObservationSpotWrite
    dataset = CollectionStore(tmp_path / 'episodes', lambda _: [])
    episode = dataset.start(EpisodeRequest(map_name='Home', robot_type='ffw_sg2',
        environment='simulation', observation_spot_id='observation'))
    assert episode['observation_spot']['label'] == 'Observation'
    store.write_observation_spot('Home', ObservationSpotWrite(label='Renamed',
        pose=config.spots.SpotPose(x=3, y=4), plan_revision=1), 'observation')
    store.delete_spot('Home', 'observation', 2)
    saved = dataset.get(episode['id'])
    assert saved['observation_spot_id'] == 'observation'
    assert saved['observation_spot']['label'] == 'Observation'
    assert saved['observation_spot']['pose']['x'] == 0
    with pytest.raises(HTTPException):
        dataset.start(EpisodeRequest(map_name='Home', robot_type='ffw_sg2', observation_spot_id='missing'))


def test_removing_legacy_spot_from_plan_keeps_it_available_in_record(store):
    # Model a plan written before observation roles existed.
    config.spots.update_spot('observation', config.spots.SpotUpdateRequest(map_name='Home', metadata={}))
    plan = store.read('Home'); plan.observations = []
    store.save('Home', plan)
    saved = next(s for s in store.view('Home')['spots'] if s['id'] == 'observation')
    assert saved['metadata']['role'] == 'observation'
