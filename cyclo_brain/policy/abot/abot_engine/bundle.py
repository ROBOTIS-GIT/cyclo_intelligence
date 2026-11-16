"""Validate explicit training-time IO metadata before allocating a model."""

import json
from pathlib import Path

import numpy as np
import yaml


def names(value, label, *, allow_empty=False):
    if (not isinstance(value, list) or (not value and not allow_empty)
            or not all(isinstance(n, str) and n for n in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} requires unique ordered names")
    return list(value)


def indices(value, count, dimension, label):
    if (not isinstance(value, list) or len(value) != count
            or any(type(i) is not int or i < 0 or i >= dimension for i in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Invalid {label} model indices")
    return value


class Normalization:
    """Explicit per-channel transforms matching upstream Normalizer semantics."""

    def __init__(self, modes, statistics, count):
        if (not isinstance(modes, list) or len(modes) != count
                or any(m not in ("identity", "min_max", "q99", "mean_std") for m in modes)):
            raise ValueError("Declare one supported normalization mode per channel")
        self.groups = []
        for mode in ("min_max", "q99", "mean_std"):
            columns = np.array([i for i, m in enumerate(modes) if m == mode], dtype=int)
            if not len(columns):
                continue
            keys = {"min_max": ("min", "max"), "q99": ("q01", "q99"),
                    "mean_std": ("mean", "std")}[mode]
            values = []
            for key in keys:
                value = np.asarray(statistics[key], dtype=np.float32)
                if value.shape != (count,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key} normalization statistics")
                values.append(value[columns])
            low, high = values
            scale = high if mode == "mean_std" else high - low
            if (scale < 0).any():
                raise ValueError("Negative normalization range or standard deviation")
            self.groups.append((mode, columns, low, scale))

    def apply(self, value, *, inverse=False):
        result = np.asarray(value, dtype=np.float32).copy()
        for mode, columns, low, scale in self.groups:
            x = result[..., columns]
            if inverse:
                x = x * scale + low if mode == "mean_std" else (x + 1) * .5 * scale + low
            else:
                valid = scale != 0
                transformed = (x - low) / np.where(valid, scale, 1)
                if mode != "mean_std":
                    transformed = 2 * transformed - 1
                x = np.where(valid, transformed, 0 if mode == "min_max" else x)
                if mode == "q99":
                    x = np.clip(x, -1, 1)
            result[..., columns] = x
        if not np.isfinite(result).all():
            raise ValueError("Non-finite transformed state/action")
        return result


class Bundle:
    def __init__(self, path, robot_type):
        self.path = Path(path).resolve()
        self.metadata = meta = json.loads((self.path / "cyclo_input_metadata.json").read_text())
        required = {"policy_id", "robot_type", "checkpoint", "cameras", "state_names", "action_names",
                    "state_indices", "action_indices", "state_normalization", "action_normalization",
                    "statistics_key", "action_mode", "observation_offsets", "include_state"}
        if set(meta) != required:
            raise ValueError(f"Invalid ABot metadata fields: {sorted(set(meta) ^ required)}")
        if meta["policy_id"] != "abot:m0" or meta["robot_type"] != robot_type:
            raise ValueError("Checkpoint policy_id/robot_type differs from the requested robot")
        if meta["action_mode"] != "absolute" or meta["observation_offsets"] != [0]:
            raise ValueError("ABot currently requires absolute actions and current observations")
        self.cameras = names(meta["cameras"], "cameras")
        self.state_names = names(meta["state_names"], "state", allow_empty=True)
        self.action_names = names(meta["action_names"], "action")
        if type(meta["include_state"]) is not bool or meta["include_state"] != bool(self.state_names):
            raise ValueError("include_state must match the explicit state channels")
        relative = Path(meta["checkpoint"])
        self.checkpoint = (self.path / relative).resolve()
        if (relative.is_absolute() or not self.checkpoint.is_relative_to(self.path)
                or self.checkpoint.suffix != ".pt" or not self.checkpoint.is_file()):
            raise ValueError("checkpoint must name an existing .pt file inside the bundle")
        self.config = yaml.safe_load((self.path / "config.yaml").read_text())
        framework = self.config["framework"]
        if framework.get("name") != "ABot_M0":
            raise ValueError("Expected framework.name=ABot_M0")
        action = framework["action_model"]
        self.action_dim = action["action_dim"]
        self.state_dim = action["state_dim"]
        self.horizon = action["future_action_window_size"] + 1
        for label, value in (("action_dim", self.action_dim), ("horizon", self.horizon)):
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid {label}")
        if (type(self.state_dim) is not int or self.state_dim < 0
                or action.get("past_action_window_size", 0) != 0
                or action["action_horizon"] != self.horizon):
            raise ValueError("Unsupported state dimension or action history/horizon")
        data = self.config["datasets"]["vla_data"]
        include = data.get("include_state", False)
        if include not in (True, False, "True", "False"):
            raise ValueError("Invalid training include_state")
        if (include in (True, "True")) != meta["include_state"]:
            raise ValueError("Training include_state differs from exported metadata")
        if data.get("action_mode", "abs") != "abs":
            raise ValueError("Training action_mode must be abs; relative/delta actions need another adapter")
        self.state_indices = indices(meta["state_indices"], len(self.state_names), self.state_dim, "state")
        self.action_indices = indices(meta["action_indices"], len(self.action_names), self.action_dim, "action")
        stats = json.loads((self.path / "dataset_statistics.json").read_text())
        if meta["statistics_key"] not in stats:
            raise ValueError("statistics_key is absent from dataset_statistics.json")
        selected = stats[meta["statistics_key"]]
        self.state_norm = Normalization(meta["state_normalization"], selected.get("state", {}), len(self.state_names))
        self.action_norm = Normalization(meta["action_normalization"], selected["action"], len(self.action_names))

    def state(self, value):
        if not self.metadata["include_state"]:
            return None
        if np.shape(value) != (len(self.state_names),) or not np.isfinite(value).all():
            raise ValueError("Invalid state shape/values")
        padded = np.zeros((1, self.state_dim), np.float32)
        padded[0, self.state_indices] = self.state_norm.apply(value)
        return padded

    def actions(self, prediction):
        value = np.asarray(prediction["normalized_actions"])
        if value.shape != (1, self.horizon, self.action_dim) or not np.isfinite(value).all():
            raise ValueError(f"Invalid ABot action chunk shape/values: {value.shape}")
        return self.action_norm.apply(value[0][:, self.action_indices], inverse=True)
