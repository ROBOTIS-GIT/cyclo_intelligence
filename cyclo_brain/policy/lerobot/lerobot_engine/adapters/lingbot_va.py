"""LingBot's private action cache must agree with the commands actually sent."""

from functools import partial

from inference_context.contract import ExecutionContract
from .validation import validate_step_layout
from .definition import AdapterDefinition
from .public_step import PublicStepAdapter


ADAPTER = AdapterDefinition(
    # Public reset clears CPU text encoding; measured cold inference is ~25 s.
    contract=ExecutionContract("step", initial_action_timeout_s=60.0),
    step_factory=partial(PublicStepAdapter, require_exact_publication=True),
    layout_validator=validate_step_layout,
)
