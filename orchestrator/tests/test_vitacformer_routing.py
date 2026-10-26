"""Routing contract without importing ROS or starting an orchestrator node."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def route():
    source = Path(__file__).parents[1] / 'orchestrator/orchestrator_node.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'OrchestratorNode')
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == '_determine_service_prefix')
    scope = {'Path': Path, 'json': json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    node = SimpleNamespace(LEROBOT_POLICIES={'act', 'smolvla'}, get_logger=lambda: Mock())
    return lambda **kwargs: scope['_determine_service_prefix'](node, SimpleNamespace(**kwargs))


@pytest.mark.parametrize('selection', ['', 'checkpoints', 'checkpoints/1000', 'checkpoints/1000/model.pt'])
def test_vitacformer_training_metadata_routes_legacy_service_to_new_backend(tmp_path, route, selection):
    (tmp_path / 'train_config.json').write_text(json.dumps({'architecture': 'ViTacFormer'}))
    weights = tmp_path / 'checkpoints/1000/model.pt'
    weights.parent.mkdir(parents=True)
    weights.touch()
    assert route(policy_path=str(tmp_path / selection), service_type='lerobot') == '/vitacformer'


def test_regular_models_keep_explicit_service_and_config_fallback(tmp_path, route):
    (tmp_path / 'config.json').write_text(json.dumps({'type': 'act'}))
    assert route(policy_path=str(tmp_path), service_type='groot') == '/groot'
    assert route(policy_path=str(tmp_path), service_type='') == '/lerobot'
    (tmp_path / 'config.json').write_text('invalid json')
    assert route(policy_path=str(tmp_path), service_type='vitacformer') == '/vitacformer'
    assert route(policy_path='', service_type='') == '/groot'
