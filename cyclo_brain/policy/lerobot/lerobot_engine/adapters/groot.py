"""LeRobot GR00T checkpoint contract, separate from the native GR00T Worker."""

import json
from pathlib import Path

from .definition import AdapterDefinition
from .validation import validate_single_observation


def validate_requested_config(config):
    if config.get("type") != "groot":
        raise ValueError(
            "GR00T N1.7 (LeRobot) requires a LeRobot checkpoint with type=groot; "
            "raw NVIDIA checkpoints belong to the independent GR00T Worker."
        )


def validate_checkpoint(config, model_path):
    validate_single_observation(config, model_path)
    _validate_groot_checkpoint(config, Path(model_path))


def _validate_groot_checkpoint(config, root):
    if not config.get("embodiment_tag") or not config.get("base_model_path"):
        raise ValueError("LeRobot GR00T requires saved embodiment_tag and base_model_path")
    base = Path(config["base_model_path"])
    if base.is_absolute() and not base.is_dir():
        raise ValueError(f"LeRobot GR00T base_model_path is missing in this Worker: {base}")

    pipelines = {}
    for name in ("policy_preprocessor", "policy_postprocessor"):
        path = root / f"{name}.json"
        if not path.is_file():
            raise ValueError(f"LeRobot GR00T requires its saved processor: {path.name}")
        with path.open() as stream:
            pipeline = json.load(stream)
        steps = pipeline.get("steps") if isinstance(pipeline, dict) else None
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"LeRobot GR00T has an invalid saved processor: {path.name}")
        pipelines[name] = steps
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"Invalid processor step in {path.name}")
            if step.get("state_file") and not (root / step["state_file"]).is_file():
                raise ValueError(f"Missing GR00T processor state file: {step['state_file']}")

    def step_config(name, registry):
        for step in pipelines[name]:
            if step.get("registry_name") == registry:
                value = step.get("config", {})
                if isinstance(value, dict):
                    value = dict(value)
                    # LeRobot saves these statistics separately from get_config().
                    if step.get("state_file"):
                        from safetensors.numpy import load_file

                        stats = {}
                        for key, tensor in load_file(root / step["state_file"]).items():
                            feature, statistic = key.rsplit(".", 1)
                            stats.setdefault(feature, {})[statistic] = tensor
                        value["stats"] = stats
                    return value
        return None

    pack = step_config("policy_preprocessor", "groot_n1_7_pack_inputs_v1")
    encode = step_config("policy_preprocessor", "groot_n1_7_vlm_encode_v1")
    if pack is None or encode is None:
        raise ValueError("LeRobot GR00T requires the saved N1.7 pack and vision processor steps")
    for key in ("state_horizon", "video_horizon"):
        if pack.get(key) not in (None, 1):
            raise ValueError(f"LeRobot GR00T {key}={pack[key]} requires unsupported observation history")
    if pack.get("embodiment_tag", "new_embodiment") != config["embodiment_tag"]:
        raise ValueError("GR00T processor embodiment_tag does not match the checkpoint")
    mapping = pack.get("embodiment_mapping", {})
    if mapping and config["embodiment_tag"] not in mapping:
        raise ValueError("GR00T processor has no mapping for the checkpoint embodiment_tag")
    stats = pack.get("stats") or {}
    raw_stats = pack.get("raw_stats") or {}
    if pack.get("normalize_min_max", True) and not (
        stats.get("observation.state") or raw_stats.get("state")
    ):
        raise ValueError("GR00T processor is missing state normalization statistics")
    decode = step_config("policy_postprocessor", "groot_n1_7_action_decode_v1")
    unpack = step_config("policy_postprocessor", "groot_action_unpack_unnormalize_v2")
    if decode is not None:
        if not (decode.get("raw_stats") or {}).get("action") or not decode.get("modality_config"):
            raise ValueError("GR00T action decoder is missing statistics or modality_config")
    elif unpack is not None:
        if unpack.get("normalize_min_max", True) and not (unpack.get("stats") or {}).get("action"):
            raise ValueError("GR00T action decoder is missing normalization statistics")
    else:
        raise ValueError("LeRobot GR00T requires its saved N1.7 action decoder")



ADAPTER = AdapterDefinition(
    checkpoint_validator=validate_checkpoint,
    requested_config_validator=validate_requested_config,
)
