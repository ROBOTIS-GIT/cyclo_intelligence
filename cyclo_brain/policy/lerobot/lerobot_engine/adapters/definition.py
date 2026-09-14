"""Resolved model integration hooks, independent of UI and robot transport."""

from dataclasses import dataclass
from typing import Callable

from inference_context.contract import ExecutionContract

from ..input_plan import latest_input_plan


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
    input_plan_factory: Callable = latest_input_plan
    history_max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if type(self.history_max_bytes) is not int or self.history_max_bytes <= 0:
            raise ValueError("history memory budget must be a positive integer")
        if not isinstance(self.contract, ExecutionContract):
            raise TypeError("adapter requires an ExecutionContract")
        if self.contract.is_step != (self.step_factory is not None):
            raise ValueError("step contract and step factory must be declared together")
        if not callable(self.input_plan_factory):
            raise TypeError("input_plan_factory must be callable")
        for callback in (self.step_factory, self.layout_validator,
                         self.checkpoint_validator, self.requested_config_validator, self.batch_validator,
                         self.policy_loader, self.predictor_factory):
            if callback is not None and not callable(callback):
                raise TypeError("adapter hooks must be callable")

    def create_execution_adapter(self, policy, preprocessor, postprocessor, to_numpy):
        if self.step_factory is None:
            return None
        return self.step_factory(policy, preprocessor, postprocessor, to_numpy)
