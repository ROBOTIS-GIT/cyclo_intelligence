# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Seongwoo Kim

"""Keep the behavior-tree engine and the orchestrator control plane apart.

``orchestrator.bt`` runs as its own process (``bt_node``) and talks to the
rest of the system only through ``interfaces`` services. This test turns
that convention into a build check so the two halves stay independently
extractable: the engine may not import the control plane, and the control
plane may not import the engine.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / 'orchestrator'
BT_ROOT = PACKAGE_ROOT / 'bt'
ENGINE_PREFIX = 'orchestrator.bt'


def _module_name(path: Path) -> str:
    """Dotted module name of ``path`` relative to the package root."""
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix('')
    parts = list(relative.parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def _resolve_relative(path: Path, level: int, module: str) -> str:
    package_parts = _module_name(path).split('.')
    if not path.name == '__init__.py':
        package_parts = package_parts[:-1]
    base = package_parts[:len(package_parts) - (level - 1)]
    return '.'.join(part for part in base + [module] if part)


def _imported_modules(path: Path):
    """Yield every absolute module name ``path`` imports.

    Relative imports are resolved against the file's package and
    ``importlib.import_module('<literal>')`` calls are included, so the
    boundary cannot be crossed through either form.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                yield node.module or ''
            else:
                yield _resolve_relative(path, node.level, node.module or '')
        elif isinstance(node, ast.Call):
            func = node.func
            is_import_module = (
                (isinstance(func, ast.Attribute) and func.attr == 'import_module')
                or (isinstance(func, ast.Name) and func.id == 'import_module')
            )
            if is_import_module and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    yield node.args[0].value


def _python_files(root: Path):
    return sorted(
        path for path in root.rglob('*.py')
        if '__pycache__' not in path.parts
    )


def _is_engine_module(name: str) -> bool:
    return name == ENGINE_PREFIX or name.startswith(ENGINE_PREFIX + '.')


def _is_control_plane_module(name: str) -> bool:
    return (
        (name == 'orchestrator' or name.startswith('orchestrator.'))
        and not _is_engine_module(name)
    )


def test_layout_is_where_the_boundary_expects_it():
    # An empty parametrization would silently skip the guard after a move.
    assert (BT_ROOT / 'bt_node.py').is_file()
    assert (PACKAGE_ROOT / 'orchestrator_node.py').is_file()
    assert _module_name(BT_ROOT / 'bt_node.py') == 'orchestrator.bt.bt_node'
    assert _resolve_relative(BT_ROOT / 'actions' / 'wait.py', 2, 'bt_core') == 'orchestrator.bt.bt_core'
    assert _resolve_relative(BT_ROOT / 'actions' / 'wait.py', 3, 'internal.x') == 'orchestrator.internal.x'


@pytest.mark.parametrize('path', _python_files(BT_ROOT), ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_engine_does_not_import_control_plane(path):
    offenders = sorted(
        name for name in _imported_modules(path) if _is_control_plane_module(name)
    )
    assert not offenders, f'{path.relative_to(PACKAGE_ROOT)} imports {offenders}'


@pytest.mark.parametrize(
    'path',
    [p for p in _python_files(PACKAGE_ROOT) if BT_ROOT not in p.parents],
    ids=lambda p: str(p.relative_to(PACKAGE_ROOT)),
)
def test_control_plane_does_not_import_engine(path):
    offenders = sorted(
        name for name in _imported_modules(path) if _is_engine_module(name)
    )
    assert not offenders, f'{path.relative_to(PACKAGE_ROOT)} imports {offenders}'
