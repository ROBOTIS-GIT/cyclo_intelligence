"""The shared catalog loader must stay process-local and ROS-independent."""

from pathlib import Path

import pytest

from shared import policy_catalog


POLICY_ROOT = Path(__file__).resolve().parents[2] / 'cyclo_brain' / 'policy'


@pytest.fixture(autouse=True)
def clear_catalog_cache():
    policy_catalog.load_policy_catalog.cache_clear()
    yield
    policy_catalog.load_policy_catalog.cache_clear()


def test_root_candidates_preserve_priority_and_source_discovery(monkeypatch):
    monkeypatch.setenv('CYCLO_POLICY_ROOT', ' /custom/policy ')
    candidates = policy_catalog._policy_root_candidates()

    assert candidates[:2] == [Path('/custom/policy'), Path('/opt/cyclo/policy')]
    assert POLICY_ROOT in candidates
    assert candidates[-1] == Path('/root/ros2_ws/src/cyclo_intelligence/cyclo_brain/policy')
    assert len(candidates) == len(set(candidates))


def test_root_candidates_deduplicate_configured_baked_path(monkeypatch):
    monkeypatch.setenv('CYCLO_POLICY_ROOT', '/opt/cyclo/policy')
    candidates = policy_catalog._policy_root_candidates()
    assert candidates[0] == Path('/opt/cyclo/policy')
    assert candidates.count(Path('/opt/cyclo/policy')) == 1


def test_loader_uses_one_cached_catalog_for_all_callers(monkeypatch):
    monkeypatch.setattr(policy_catalog, '_policy_root_candidates', lambda: [POLICY_ROOT])
    first = policy_catalog.load_policy_catalog()
    second = policy_catalog.load_policy_catalog()

    assert second is first
    assert policy_catalog.load_policy_catalog.cache_info().maxsize == 1
    assert policy_catalog.load_policy_catalog.cache_info().hits == 1
    module, catalog = first
    assert Path(module.__file__) == POLICY_ROOT / 'common' / 'catalog' / 'catalog.py'
    assert module.resolve_policy(catalog, 'abot_m0')[1]['policy_id'] == 'abot:m0'


@pytest.mark.parametrize('policy_id,expected', [
    ('lerobot:act', 'chunk'), ('lerobot:diffusion', 'step'), ('act', 'chunk'),
])
def test_execution_mode_uses_manifest_policy_and_alias(monkeypatch, policy_id, expected):
    monkeypatch.setattr(policy_catalog, '_policy_root_candidates', lambda: [POLICY_ROOT])
    assert policy_catalog.execution_mode(policy_id) == expected


def test_missing_catalog_and_empty_policy_keep_existing_behavior(monkeypatch):
    monkeypatch.setattr(policy_catalog, '_policy_root_candidates', lambda: [])
    assert policy_catalog.execution_mode('') is None
    assert policy_catalog.execution_mode(None) is None
    assert policy_catalog.load_policy_catalog.cache_info().misses == 0
    with pytest.raises(RuntimeError, match='Cyclo policy catalog is unavailable'):
        policy_catalog.load_policy_catalog()


def test_unknown_policy_still_raises_catalog_error(monkeypatch):
    monkeypatch.setattr(policy_catalog, '_policy_root_candidates', lambda: [POLICY_ROOT])
    module, _ = policy_catalog.load_policy_catalog()
    with pytest.raises(module.CatalogError, match='unknown policy id'):
        policy_catalog.execution_mode('missing:model')
