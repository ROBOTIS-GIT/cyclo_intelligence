"""Model registration changes no shared runtime, UI or robot transport code."""

import subprocess
import sys

import pytest

from inference_context import LatestValues
from inference_context.contract import ExecutionContract
from lerobot_engine.adapters import AdapterDefinition, AdapterRegistry, resolve_adapter


def test_register_model_with_input_extensions_and_execution():
    registry = AdapterRegistry()
    sentinel = object()

    def extensions(operators, sources, engine):
        operators.register("custom", lambda options, context: lambda values: values[0])

    definition = AdapterDefinition(
        contract=ExecutionContract("step"),
        step_factory=lambda *args: sentinel,
        input_extensions=extensions,
    )
    registry.register("new_policy", definition)
    resolved = registry.resolve("new_policy")
    assert resolved is definition
    assert resolved.create_execution_adapter(None, None, None, None) is sentinel
    from inference_inputs import Graph
    from inference_inputs.operators import default_registry
    operators = default_registry()
    resolved.input_extensions(operators, {}, None)
    batch = Graph({"sources": {"x": {"source": "joint:new"}},
                   "nodes": {"y": {"op": "custom", "inputs": ["x"]}},
                   "outputs": {"before": {"custom_state": "y"}}}, operators).assemble(LatestValues({"joint:new": [1., 2.]}))
    assert batch == {"custom_state": [1., 2.]}
    assert registry.resolve("act").contract.mode == "chunk"


def test_duplicate_registration_and_incomplete_contract_fail_early():
    registry = AdapterRegistry()
    registry.register("custom", AdapterDefinition())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("custom", AdapterDefinition())
    with pytest.raises(ValueError, match="together"):
        AdapterDefinition(contract=ExecutionContract("step"))
    with pytest.raises(TypeError, match="callable"):
        AdapterDefinition(input_extensions="not a function")


def test_registered_models_keep_their_distinct_contracts():
    assert resolve_adapter("lingbot_va").contract.initial_action_timeout_s == 60.
    assert resolve_adapter("multi_task_dit").contract.is_step
    assert resolve_adapter("act").create_execution_adapter(None, None, None, None) is None


def test_plain_models_do_not_need_custom_input_extensions():
    assert AdapterDefinition().input_extensions is None


def test_adapter_inspection_does_not_import_model_frameworks():
    code = (
        "import sys; from lerobot_engine.adapters import resolve_adapter; "
        "assert resolve_adapter('lingbot_va').contract.is_step; "
        "assert 'torch' not in sys.modules; assert 'lerobot' not in sys.modules"
    )
    import os
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=15)
