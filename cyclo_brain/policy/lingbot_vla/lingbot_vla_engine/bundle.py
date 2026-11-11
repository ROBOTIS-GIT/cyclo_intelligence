"""Validate portable checkpoint assets without importing model dependencies."""

import ast
import json
from pathlib import Path

import numpy as np
import yaml

from .mapping import channel_names


def config_pairs(values, label):
    result = {}
    for value in values:
        item = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(item, dict) or len(item) != 1 or result.keys() & item.keys():
            raise ValueError(f"Invalid or duplicate {label} entry: {value}")
        result.update(item)
    return result


def validate_assets(training, robot, stats, metadata):
    required = {"robot_type", "state_key", "state_names", "action_key", "action_names", "cameras"}
    if set(metadata) != required or not all(isinstance(metadata[k], str) and metadata[k]
                                           for k in ("robot_type", "state_key", "action_key")):
        raise ValueError("Invalid cyclo_input_metadata.json fields")
    if not metadata["state_key"].startswith("observation.state") or not metadata["action_key"].startswith("action"):
        raise ValueError("Expected raw observation.state and action feature keys")
    states = channel_names(metadata["state_names"], "state_names")
    actions = channel_names(metadata["action_names"], "action_names")
    cameras = channel_names(metadata["cameras"], "cameras")
    if any(not key.startswith("observation.images.") for key in cameras):
        raise ValueError("Camera keys must start with observation.images.")
    data = training["data"]
    dimensions = config_pairs(data["joints"], "joints")
    norms = config_pairs(data["norm_type"], "norm_type")
    if any(type(dim) is not int or dim < 0 for dim in dimensions.values()):
        raise ValueError("Canonical dimensions must be nonnegative integers")
    model = dict(training["model"], **training["train"])
    horizon = model.get("chunk_size", 50)
    if type(horizon) is not int or horizon < 1:
        raise ValueError("chunk_size must be a positive integer")
    for key in ("max_state_dim", "max_action_dim"):
        if sum(dimensions.values()) > model.get(key, 55):
            raise ValueError(f"Canonical features exceed {key}")
    if set(robot) != {"norm_stats", "states", "actions", "images"}:
        raise ValueError("robot_config.yaml requires states, actions, images, and norm_stats")
    norm_fields = {"identity": (), "meanstd": ("mean", "std"), "std": ("std",),
                   "bounds_98": ("q02", "q98"), "bounds_99": ("q01", "q99"),
                   "bounds_98_woclip": ("q02", "q98"), "bounds_99_woclip": ("q01", "q99"),
                   "minmax": ("min", "max"), "minmax_woclip": ("min", "max")}
    mapped = {}
    for category, prefix, names, source in (
        ("states", "observation.state.", states, metadata["state_key"]),
        ("actions", "action.", actions, metadata["action_key"]),
    ):
        covered = []
        mapped[category] = {}
        for entry in robot[category]:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"{category}: explicit origin_keys mappings are required")
            key, mapping = next(iter(entry.items()))
            if not key.startswith(prefix) or key in mapped[category]:
                raise ValueError(f"Invalid or duplicate canonical feature: {key}")
            joint = key[len(prefix):]
            if joint not in dimensions or dimensions[joint] <= 0:
                raise ValueError(f"Undeclared canonical feature: {key}")
            allowed = {"origin_keys"} if category == "states" else {"origin_keys", "subtract_state", "relative_type"}
            if set(mapping) - allowed or "origin_keys" not in mapping:
                raise ValueError(f"{key}: geometric conversion/implicit mappings are not supported")
            if category == "actions":
                if type(mapping.get("subtract_state")) is not bool:
                    raise ValueError(f"{key}: subtract_state must be explicit")
                if joint.startswith("end."):
                    raise ValueError("End-effector actions require a separate robot execution adapter")
                if mapping["subtract_state"] and mapping.get("relative_type", "quaternion_local") is not None:
                    raise ValueError(f"{key}: joint-relative actions require explicit relative_type: null in the training mapping")
                if ("velocity" in joint or joint == "effector.position") and mapping["subtract_state"]:
                    raise ValueError(f"{key}: upstream forbids relative velocity/gripper actions")
            indices = []
            for part in mapping["origin_keys"]:
                if set(part) != {source}:
                    raise ValueError(f"{key}: expected source {source}")
                selection = part[source]
                if set(selection) != {"start", "end"}:
                    raise ValueError(f"{key}: only explicit start/end slices are supported")
                start, end = selection["start"], selection["end"]
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(names):
                    raise ValueError(f"{key}: invalid source slice")
                indices.extend(range(start, end))
            if not indices or len(indices) > dimensions[joint]:
                raise ValueError(f"{key}: mapping exceeds canonical capacity")
            covered.extend(indices)
            mapped[category][joint] = (indices, mapping)
            norm = norms.get(joint)
            if norm not in norm_fields:
                raise ValueError(f"{key}: unsupported normalization {norm}")
            feature_stats = stats.get("norm_stats", {}).get(key)
            # Even identity needs dimension statistics for upstream's unpadding masks.
            if not isinstance(feature_stats, dict) or "mean" not in feature_stats:
                raise ValueError(f"Missing statistics for {key}")
            for field in set(norm_fields[norm]) | {"mean"}:
                value = np.asarray(feature_stats.get(field), dtype=np.float64)
                if value.ndim not in (1, 2) or value.shape[-1] != len(indices) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid statistics: {key}.{field}")
                if value.ndim == 2 and (category == "states" or value.shape[0] < horizon):
                    raise ValueError(f"Invalid normalization horizon: {key}.{field}")
                if field == "std" and (value < 0).any():
                    raise ValueError(f"Negative standard deviation: {key}")
        if sorted(covered) != list(range(len(names))):
            raise ValueError(f"{category}: every source channel must be mapped exactly once")
    for joint, (indices, mapping) in mapped["actions"].items():
        if mapping["subtract_state"]:
            state_indices = mapped["states"].get(joint, ([],))[0]
            if [actions[i] for i in indices] != [states[i] for i in state_indices]:
                raise ValueError(f"Relative action/state channels do not match: {joint}")
    image_sources, image_targets = [], []
    for entry in robot["images"]:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Images require explicit origin_keys mappings")
        target, mapping = next(iter(entry.items()))
        if set(mapping) != {"origin_keys"} or not isinstance(mapping["origin_keys"], str):
            raise ValueError(f"Invalid image mapping: {target}")
        image_targets.append(target)
        image_sources.append(mapping["origin_keys"])
    declared = [f"observation.images.{name}" for name in data["cameras"]]
    if (len(set(declared)) != len(declared) or len(set(image_targets)) != len(image_targets)
            or not set(image_targets).issubset(declared)
            or len(set(image_sources)) != len(image_sources) or set(image_sources) != set(cameras)):
        raise ValueError("Camera mapping must match training cameras and exported source keys")
    return horizon


class CheckpointBundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.weights = self.root / "checkpoints/export/hf_ckpt"
        if not list(self.weights.glob("*.safetensors")):
            raise ValueError("Select an exported LingBot-VLA bundle with checkpoints/export/hf_ckpt weights")
        self.training = yaml.safe_load((self.root / "lingbotvla_cli.yaml").read_text())
        self.robot_config = self.root / "robot_config.yaml"
        self.norm_stats = self.root / "norm_stats.json"
        self.metadata = json.loads((self.root / "cyclo_input_metadata.json").read_text())
        self.horizon = validate_assets(self.training, yaml.safe_load(self.robot_config.read_text()),
                                       json.loads(self.norm_stats.read_text()), self.metadata)
        if any(not isinstance(value, str) for key in ("joints", "norm_type")
               for value in self.training["data"][key]):
            raise ValueError("Use export_checkpoint.py to serialize native joints/norm_type configuration")
        tokenizer = self.training["model"].get("tokenizer_path", "")
        if not tokenizer or (tokenizer.startswith(("/", ".")) and not Path(tokenizer).is_dir()):
            raise ValueError("tokenizer_path is not portable; export with --base-model or mount its assets")
