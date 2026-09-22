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



def validate_relative_channels(mapping, decode_step):
    """Match the native decoder's named group references at the external boundary."""
    from lerobot.policies.groot.utils import config_value, stat_dim_from_entry

    if decode_step.pack_step is None:
        raise ValueError("GR00T relative action decoder has no state pack step")
    pack = decode_step.pack_step.get_config()
    decode = decode_step.get_config()
    if decode.get("action_decode_transform"):
        raise ValueError("GR00T relative action transform requires a separate channel contract")

    def split_channels(options, modality, names):
        config = (options.get("modality_config") or {}).get(modality, {})
        stats = (options.get("raw_stats") or {}).get(modality, {})
        keys = config.get("modality_keys", [])
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
            raise ValueError(f"GR00T {modality} group order is missing")
        if len(set(keys)) != len(keys):
            raise ValueError(f"GR00T {modality} group order is ambiguous")
        groups, offset = {}, 0
        for key in keys:
            width = stat_dim_from_entry(stats.get(key, {}))
            if width <= 0 or offset + width > len(names):
                raise ValueError(f"GR00T {modality} group {key} does not match external channels")
            groups[key] = names[offset:offset + width]
            offset += width
        if offset != len(names):
            raise ValueError(f"GR00T {modality} groups do not cover external channels")
        return config, groups

    _, states = split_channels(pack, "state", mapping.state_names)
    config, actions = split_channels(decode, "action", mapping.action_names)
    configs = config.get("action_configs", [])
    if not isinstance(configs, list) or len(configs) != len(actions):
        raise ValueError("GR00T relative action group configuration is incomplete")
    for (key, names), spec in zip(actions.items(), configs, strict=True):
        if not isinstance(spec, dict):
            raise ValueError(f"GR00T action group configuration is invalid: {key}")
        if config_value(spec.get("rep")) != "relative":
            continue
        if config_value(spec.get("type")) != "non_eef":
            raise ValueError("GR00T relative EEF channels are not a joint/velocity mapping")
        state_key = spec.get("state_key") or key
        if states.get(state_key) != names:
            raise ValueError(f"GR00T relative state/action channel mismatch: {key} -> {state_key}")


ADAPTER = AdapterDefinition(
    checkpoint_validator=validate_checkpoint,
    requested_config_validator=validate_requested_config,
)
