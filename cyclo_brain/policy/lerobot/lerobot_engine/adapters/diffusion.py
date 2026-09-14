"""Diffusion stacks processed camera tensors before the image encoder."""

from .definition import AdapterDefinition
from .validation import validate_equal_camera_sizes


ADAPTER = AdapterDefinition(batch_validator=validate_equal_camera_sizes)
