"""Resolved model integration hooks, independent of UI and robot transport."""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
from typing import Callable

from inference_context.contract import ExecutionContract


@dataclass(frozen=True)
class AdapterDefinition:
    contract: ExecutionContract = ExecutionContract()
    step_factory: Callable | None = None
    layout_validator: Callable | None = None
    checkpoint_validator: Callable | None = None
    requested_config_validator: Callable | None = None
    batch_validator: Callable | None = None
    policy_loader: Callable | None = None
    predictor_factory: Callable | None = None
    input_extensions: Callable | None = None
    input_handlers: Mapping[str, Callable] = field(default_factory=dict)
    history_max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.input_handlers, Mapping) or any(
            not isinstance(name, str) or not name.isidentifier() or not callable(builder)
            for name, builder in self.input_handlers.items()
        ):
            raise ValueError("input_handlers must map registered names to graph builders")
        object.__setattr__(self, "input_handlers", MappingProxyType(dict(self.input_handlers)))
        if type(self.history_max_bytes) is not int or self.history_max_bytes <= 0:
            raise ValueError("history memory budget must be a positive integer")
        if not isinstance(self.contract, ExecutionContract):
            raise TypeError("adapter requires an ExecutionContract")
        if self.contract.is_step != (self.step_factory is not None):
            raise ValueError("step contract and step factory must be declared together")
        for callback in (self.step_factory, self.layout_validator,
                         self.checkpoint_validator, self.requested_config_validator, self.batch_validator,
                         self.policy_loader, self.predictor_factory, self.input_extensions):
            if callback is not None and not callable(callback):
                raise TypeError("adapter hooks must be callable")

    def create_execution_adapter(self, policy, preprocessor, postprocessor, to_numpy):
        if self.step_factory is None:
            return None
        return self.step_factory(policy, preprocessor, postprocessor, to_numpy)
