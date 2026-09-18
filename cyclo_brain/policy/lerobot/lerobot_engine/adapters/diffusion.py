"""Use Diffusion's public online API, including its observation/action queues."""

from inference_context.contract import ExecutionContract
from .definition import AdapterDefinition
from .public_step import PublicStepAdapter
from .validation import validate_equal_camera_sizes, validate_step_layout


def create_step(policy, preprocessor, postprocessor, to_numpy):
    steps = getattr(preprocessor, "steps", ())
    if steps:
        from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

        if any(isinstance(step, RelativeActionsProcessorStep) and step.enabled for step in steps):
            raise ValueError(
                "Diffusion online execution does not support relative-action processors: "
                "cached actions would be reanchored to a different state each step. "
                "A reviewed chunk-anchored adapter is required; do not disable a training transform."
            )

    def validate_batch(batch):
        validate_equal_camera_sizes(policy.config, batch, policy.config.image_features)

    return PublicStepAdapter(
        policy, preprocessor, postprocessor, to_numpy, batch_validator=validate_batch,
    )


ADAPTER = AdapterDefinition(
    contract=ExecutionContract("step"),
    step_factory=create_step,
    layout_validator=validate_step_layout,
    batch_validator=validate_equal_camera_sizes,
)
