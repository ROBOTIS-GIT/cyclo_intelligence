"""Constraints of the pinned WALL-X implementation; never truncate robot joints."""

from cyclo_lerobot_io.mapping import feature_dim

from .definition import AdapterDefinition
from .validation import validate_single_observation


def validate_checkpoint(config, model_path):
    validate_single_observation(config, model_path)
    for features, key, maximum in (
        ("input_features", "observation.state", "max_state_dim"),
        ("output_features", "action", "max_action_dim"),
    ):
        dim = feature_dim(config.get(features, {}), key)
        # The pinned WALL-X core pads to a literal 20, independently of config.
        if config.get(maximum, 20) != 20 or dim > 20:
            raise ValueError(
                f"WALL-X requires {maximum}=20 and {key} dimension <= 20; got {dim}"
            )


ADAPTER = AdapterDefinition(
    checkpoint_validator=validate_checkpoint,
)
