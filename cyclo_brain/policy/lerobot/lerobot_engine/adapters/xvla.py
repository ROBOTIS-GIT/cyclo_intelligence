"""XVLA can resize with padding internally; otherwise stacking needs equal sizes."""

from .definition import AdapterDefinition
from .validation import validate_equal_camera_sizes


def validate_batch(config, batch, camera_keys):
    if getattr(config, "resize_imgs_with_padding", None) is None:
        validate_equal_camera_sizes(config, batch, camera_keys)


ADAPTER = AdapterDefinition(batch_validator=validate_batch)
