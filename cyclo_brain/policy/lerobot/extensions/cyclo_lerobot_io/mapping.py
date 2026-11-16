"""Portable channel order at the saved processor's external I/O boundary.

This module intentionally has no Torch, dataset, or Cyclo runtime dependencies.
Names describe physical channels, not padded internal model dimensions.
"""

import json
import logging
from pathlib import Path

FILENAME = "cyclo_io_mapping.json"


def feature_dim(features, key):
    feature = features.get(key)
    shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
    if not shape or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError(f"Expected a positive vector feature: {key}")
    return shape[0]


def validate_mapping(mapping, *, state_dim=None, action_dim=None):
    if not isinstance(mapping, dict) or type(mapping.get("version")) is not int or mapping["version"] != 1:
        raise ValueError(f"{FILENAME}: version must be 1")
    for key, dim in (("state_names", state_dim), ("action_names", action_dim)):
        names = mapping.get(key)
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name or name != name.strip() for name in names)
        ):
            raise ValueError(f"{FILENAME}: {key} must be a nonempty list of channel names")
        if len(set(names)) != len(names):
            raise ValueError(f"{FILENAME}: duplicate {key}")
        if dim is not None and len(names) != dim:
            raise ValueError(f"{FILENAME}: {key} has {len(names)} channels, checkpoint requires {dim}")
    if "dataset" in mapping and not isinstance(mapping["dataset"], dict):
        raise ValueError(f"{FILENAME}: dataset must be an object")
    return mapping


def read_mapping(directory):
    path = Path(directory) / FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    try:
        with path.open() as stream:
            return validate_mapping(json.load(stream))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid channel metadata {path}: {exc}") from exc


def mapping_from_features(features, *, repo_id=None, revision=None):
    names = {}
    missing = []
    for feature, key in (("observation.state", "state_names"), ("action", "action_names")):
        metadata = features.get(feature, {})
        value = metadata.get("names")
        if value is None:
            missing.append(feature)
            continue
        # Validate available names even when the other vector has no metadata.
        probe = {"version": 1, "state_names": value, "action_names": value}
        validate_mapping(probe, state_dim=feature_dim(features, feature))
        names[key] = value
    if missing:
        logging.warning("No %s channel names; %s will not be exported", ", ".join(missing), FILENAME)
        return None
    mapping = {"version": 1, **names, "dataset": {"repo_id": repo_id, "revision": revision}}
    return validate_mapping(
        mapping,
        state_dim=feature_dim(features, "observation.state"),
        action_dim=feature_dim(features, "action"),
    )
