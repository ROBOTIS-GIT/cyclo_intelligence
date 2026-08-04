#!/usr/bin/env python3
#
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

"""Single source of truth for BT node discovery.

Each Action / Control class can be discovered from the actions/controls
folders without explicit registration. Catalog entries are synthesized from
class names and constructor signatures.
"""

import importlib
import inspect
import pkgutil
from typing import Type
from typing import get_args
from typing import get_origin

from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import BTNode
from orchestrator.bt.controls.base_control import BaseControl


# Bumped whenever the catalog schema changes shape (new required field, etc).
# The UI compares this against its cached copy to decide whether to invalidate
# localStorage. Bumping is cheap — just remember to do it.
SCHEMA_VERSION = '1.0'

# Internal ctor kwargs that should never appear as XML ports. The loader
# supplies these from runtime config (node handle, joint topology, etc).
_INTERNAL_KWARGS = frozenset({
    'node',
    'name',
    'groups',
    'positions',
    'topic_config',
    'service_name',
    'head_joint_names',
    'left_joint_names',
    'right_joint_names',
    'lift_joint_name',
    'position_threshold',
})

_ALLOWED_PORT_TYPES = frozenset({'bool', 'number', 'string'})


def _import_package_modules(package_name: str):
    """Import every direct module under a package so subclasses register."""
    package = importlib.import_module(package_name)
    module_names = set()
    for module_info in sorted(
        pkgutil.iter_modules(package.__path__),
        key=lambda info: info.name,
    ):
        if module_info.name.startswith('_'):
            continue
        module_name = f'{package_name}.{module_info.name}'
        importlib.import_module(module_name)
        module_names.add(module_name)
    return module_names


def _import_node_modules():
    """Load action/control modules from disk before scanning subclasses."""
    importlib.invalidate_caches()
    return {
        'action': _import_package_modules('orchestrator.bt.actions'),
        'control': _import_package_modules('orchestrator.bt.controls'),
    }


def _collect_subclasses(
    base: Type[BTNode],
    allowed_modules: set,
) -> dict[str, Type[BTNode]]:
    """Walk the subclass tree below `base` and return tag → class."""
    out: dict[str, Type[BTNode]] = {}
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        # Defer recursion into grandchildren so abstract/helper classes do not
        # block their concrete subclasses from being seen.
        stack.extend(cls.__subclasses__())
        if cls.__module__ not in allowed_modules:
            continue
        out[_tag_for_class(cls)] = cls
    return out


def _category_for_class(cls: Type[BTNode]) -> str:
    if issubclass(cls, BaseAction):
        return 'action'
    if issubclass(cls, BaseControl):
        return 'control'
    return ''


def _stringify_default(value) -> str:
    if value is inspect.Parameter.empty or value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _annotation_to_port_type(annotation, default) -> str:
    if isinstance(default, bool):
        return 'bool'
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        return 'number'

    if isinstance(annotation, str):
        annotation = annotation.lower()
        if 'bool' in annotation:
            return 'bool'
        if 'int' in annotation or 'float' in annotation:
            return 'number'
        return 'string'

    origin = get_origin(annotation)
    args = get_args(annotation)
    candidates = set(args if origin is not None else [annotation])
    candidates.discard(type(None))

    if bool in candidates:
        return 'bool'
    if int in candidates or float in candidates:
        return 'number'
    return 'string'


def _port_from_param(param) -> dict:
    return {
        'name': param.name,
        'type': _annotation_to_port_type(param.annotation, param.default),
        'default': _stringify_default(param.default),
    }


def _ctor_port_defs(cls: Type[BTNode]) -> list[dict]:
    sig = inspect.signature(cls.__init__)
    ports = []
    for param in sig.parameters.values():
        if (
            param.name == 'self'
            or param.name in _INTERNAL_KWARGS
            or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):
            continue
        ports.append(_port_from_param(param))
    return ports


def _tag_for_class(cls: Type[BTNode]) -> str:
    return cls.__name__


def _catalog_entry_for_class(cls: Type[BTNode], category: str) -> dict:
    """Return a complete catalog entry for registry consumers."""
    return {
        'tag': _tag_for_class(cls),
        'category': category,
        'ports': _ctor_port_defs(cls),
    }


def build_registry() -> dict[str, Type[BTNode]]:
    """Build the tag → class registry from in-memory subclasses.

    Importing the actions / controls packages here guarantees every
    concrete class shows up in __subclasses__() even if a caller forgot
    to import them earlier in their process.
    """
    imported_modules = _import_node_modules()

    registry: dict[str, Type[BTNode]] = {}
    registry.update(_collect_subclasses(
        BaseAction,
        imported_modules['action'],
    ))
    registry.update(_collect_subclasses(
        BaseControl,
        imported_modules['control'],
    ))
    return registry


def _ctor_kwargs(cls: Type[BTNode]) -> list[str]:
    """Return ctor kwarg names declared on `cls`, minus internals."""
    sig = inspect.signature(cls.__init__)
    return [
        name for name in sig.parameters
        if name != 'self' and name not in _INTERNAL_KWARGS
    ]


def validate_registry(registry: dict[str, Type[BTNode]]) -> list[str]:
    """Return a list of human-readable problems found in the registry.

    Caller (bt_node.py) decides what to do with the list — typically
    log each entry as an error. An empty list means generated catalog
    entries line up with the ctor signatures.
    """
    problems: list[str] = []
    for tag, cls in registry.items():
        entry = _catalog_entry_for_class(cls, _category_for_class(cls))
        category = entry.get('category')
        if category not in ('action', 'control'):
            problems.append(
                f"[{tag}] category must be 'action' or 'control', "
                f'got {category!r}'
            )

        ports = entry.get('ports', [])
        if not isinstance(ports, list):
            problems.append(f"[{tag}] ports must be a list")
            continue

        declared = {p.get('name') for p in ports if isinstance(p, dict)}
        ctor_names = set(_ctor_kwargs(cls))

        # Every declared port must map to a ctor kwarg — otherwise the
        # XML attribute the UI emits would be silently dropped.
        for name in declared:
            if name not in ctor_names:
                problems.append(
                    f"[{tag}] port '{name}' is not a constructor kwarg "
                    f"(known kwargs: {sorted(ctor_names)})"
                )

        for port in ports:
            if not isinstance(port, dict):
                problems.append(f"[{tag}] non-dict entry in ports")
                continue
            ptype = port.get('type')
            if ptype not in _ALLOWED_PORT_TYPES:
                problems.append(
                    f"[{tag}.{port.get('name')}] type must be one of "
                    f"{sorted(_ALLOWED_PORT_TYPES)}, got {ptype!r}"
                )
            if 'default' not in port:
                problems.append(
                    f"[{tag}.{port.get('name')}] port missing 'default'"
                )

    return problems


def catalog_payload(
    registry: dict[str, Type[BTNode]] | None = None,
) -> list[dict]:
    """Return node catalog entries ready to be JSON-serialized for the UI."""
    registry = registry or build_registry()
    return [
        _catalog_entry_for_class(cls, _category_for_class(cls))
        for tag, cls in sorted(registry.items())
    ]
