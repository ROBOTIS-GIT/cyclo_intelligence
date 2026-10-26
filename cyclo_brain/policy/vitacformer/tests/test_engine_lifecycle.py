from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engine_process.protocol import EngineCommandRequest
from engine_process.worker import EngineWorker
from vitacformer_engine import ViTacFormerEngine
from vitacformer_engine.constants import ACTION_KEYS


def test_engine_works_with_upstream_worker_load_infer_unload(monkeypatch):
    engine = ViTacFormerEngine()
    policy = SimpleNamespace(reset=lambda: None)
    robot = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(engine, '_load_policy_assets', lambda *_: policy)

    def init_robot(_robot_type):
        engine._robot = robot
        engine._action_keys = list(ACTION_KEYS)

    monkeypatch.setattr(engine, '_init_robot', init_robot)
    monkeypatch.setattr(engine, '_build_observation', lambda: {})
    monkeypatch.setattr(engine, '_predict_chunk', lambda _: np.arange(5400).reshape(100, 54))
    worker = EngineWorker(engine)
    loaded = worker.handle(EngineCommandRequest(
        command=0, seq_id=1, model_path='/model', robot_type='ffw_sh5_rev1',
    ))
    assert loaded.success and loaded.action_keys == list(ACTION_KEYS)
    assert not worker.handle(EngineCommandRequest(command=0, model_path='/model')).success
    result = worker.handle(EngineCommandRequest(command=1, seq_id=2))
    assert result.success and result.seq_id == 2
    assert (result.chunk_size, result.action_dim) == (20, 54)
    np.testing.assert_array_equal(result.action_list, np.arange(1080))
    assert worker.handle(EngineCommandRequest(command=2, seq_id=3)).success
    assert not engine.is_ready
    assert not worker.handle(EngineCommandRequest(command=1, seq_id=4)).success


def test_failed_load_releases_partial_resources(monkeypatch):
    engine = ViTacFormerEngine()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(engine, '_load_policy_assets', lambda *_: SimpleNamespace(reset=lambda: None))
    closed = []

    def init_robot(_robot_type):
        engine._robot = SimpleNamespace(close=lambda: closed.append(True))
        raise RuntimeError('Missing tactile input')

    monkeypatch.setattr(engine, '_init_robot', init_robot)
    result = engine.load_policy(SimpleNamespace(model_path='/model', robot_type='ffw_sh5_rev1'))
    assert not result['success']
    assert closed == [True]
    assert engine._policy is None and engine._robot is None


@pytest.mark.parametrize('robot_type,acceleration', [('ffw_sg2_rev1', ''), ('ffw_sh5_rev1', 'tensorrt_dit')])
def test_unsupported_load_contract_fails_before_weights(robot_type, acceleration):
    result = ViTacFormerEngine().load_policy(SimpleNamespace(
        model_path='/missing', robot_type=robot_type, acceleration_mode=acceleration,
    ))
    assert not result['success'] and 'supports only' in result['message']
