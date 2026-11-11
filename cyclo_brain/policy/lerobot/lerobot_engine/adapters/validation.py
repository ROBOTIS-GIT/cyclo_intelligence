"""Common checkpoint and camera constraints for model adapters."""

def validate_single_observation(config, model_path):
    if config.get("n_obs_steps", 1) != 1:
        raise ValueError(f"{config.get('type')}: Cyclo currently supplies one observation, not history")


def validate_equal_camera_sizes(config, batch, camera_keys):
    shapes = {key: tuple(batch[key].shape[-2:]) for key in camera_keys}
    if len(set(shapes.values())) > 1:
        raise ValueError(
            f"{config.type} cameras must have equal sizes before stacking: {shapes}. "
            f"Check {config.type}.yaml against the training preprocessing."
        )
