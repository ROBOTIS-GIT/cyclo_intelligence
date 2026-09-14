"""Multi-Task DiT owns observation/action queues behind select_action/reset."""

from inference_context.contract import ExecutionContract
from .validation import validate_step_layout
from .definition import AdapterDefinition
from .public_step import PublicStepAdapter


ADAPTER = AdapterDefinition(
    contract=ExecutionContract("step"),
    step_factory=PublicStepAdapter,
    layout_validator=validate_step_layout,
)
