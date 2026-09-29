"""Shared, process-local policy metadata without importing model Workers."""

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path


def _policy_root_candidates():
    candidates = []
    configured = os.environ.get('CYCLO_POLICY_ROOT', '').strip()
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path('/opt/cyclo/policy'))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / 'cyclo_brain' / 'policy')
    candidates.append(Path('/root/ros2_ws/src/cyclo_intelligence/cyclo_brain/policy'))
    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=1)
def load_policy_catalog():
    for root in _policy_root_candidates():
        module_path = root / 'common' / 'catalog' / 'catalog.py'
        try:
            available = module_path.is_file()
        except OSError:
            available = False
        if not available:
            continue
        spec = importlib.util.spec_from_file_location('cyclo_policy_catalog_orchestrator', module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, module.load_catalog(root)
    raise RuntimeError('Cyclo policy catalog is unavailable')


def execution_mode(policy_id):
    if not policy_id:
        return None
    module, catalog = load_policy_catalog()
    _, model = module.resolve_policy(catalog, policy_id)
    return model.get('execution_mode')
