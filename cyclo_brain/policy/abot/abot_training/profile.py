"""Pinned profile validation, independent of Bench and upstream registries."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .channels import resolve_channels, resolve_dimensions


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def inside(root, relative):
    root = Path(root).resolve(strict=True)
    path = Path(relative)
    resolved = (root / path).resolve(strict=True)
    if path.is_absolute() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Expected a file inside {root}: {relative}")
    return resolved


def _names(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(x, str) or not x or x.strip() != x for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} must contain unique ordered channel names")
    return value


def validate_profile(profile, info, *, channel_selection=None):
    if (profile.get("version") != 2 or profile.get("model_id") != "abot-m0"
            or profile.get("policy_id") != "abot:m0"
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", str(profile.get("profile_id", "")))):
        raise ValueError("Expected an ABot-M0 version 2 profile")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Provide a prepared LeRobot v2.1 dataset; conversion belongs to Bench")
    if not profile.get("robot_type") or profile["robot_type"] != info.get("robot_type"):
        raise ValueError("Profile robot_type differs from dataset metadata")
    if profile.get("action_representation") != "absolute":
        raise ValueError("Only absolute targets without state subtraction are supported")
    if profile.get("padding_mask") != "exclude_padded_channels":
        raise ValueError("Padded action channels must be excluded from the loss")
    if profile.get("normalization") not in ("min_max", "mean_std", "quantile"):
        raise ValueError("Unsupported normalization")
    modalities = profile.get("modalities", {})
    if set(modalities) != {"state", "action", "video"}:
        raise ValueError("Explicit state/action/video modalities are required")
    for kind, feature_key in (("state", "observation.state"), ("action", "action")):
        names = _names(profile.get(f"{kind}_names"), kind)
        feature = info["features"][feature_key]
        if feature.get("names") != names or feature.get("shape") != [len(names)]:
            raise ValueError(f"{kind} names/order/dimensions differ from dataset metadata")
        groups = modalities[kind]
        if (not isinstance(groups, dict) or not groups
                or any(not isinstance(v, list) or not v for v in groups.values())
                or [n for group in groups.values() for n in group] != names):
            raise ValueError(f"{kind} modalities must cover every dataset channel in order")
    channels = resolve_channels(info, channel_selection)
    resolve_dimensions(profile.get("padding"), channels["selected"])
    cameras = profile.get("cameras")
    if (not isinstance(cameras, dict) or not cameras
            or any(not isinstance(k, str) or not k or not isinstance(v, str)
                   for k, v in cameras.items())
            or len(set(cameras.values())) != len(cameras)
            or any(info["features"].get(v, {}).get("dtype") != "video" for v in cameras.values())
            or modalities["video"] != cameras):
        raise ValueError("Camera roles must match distinct dataset video features")
    # Object key order is not covered by canonical JSON hashes. Require an array.
    order = profile.get("camera_order")
    if (not isinstance(order, list) or len(order) != len(cameras)
            or any(not isinstance(x, str) for x in order) or set(order) != set(cameras)):
        raise ValueError("camera_order must explicitly list every camera role once")
    assets = profile.get("assets")
    if (not isinstance(assets, dict) or not assets
            or any(not isinstance(a, dict) or not isinstance(a.get("repo_id"), str) or not a["repo_id"].strip()
                   or a.get("provider") not in ("huggingface", "modelscope")
                   or not re.fullmatch(r"[0-9a-f]{40}", str(a.get("revision", "")))
                   for a in assets.values())):
        raise ValueError("Assets require provider, repository IDs and immutable revisions")
    roles = profile.get("asset_roles", {})
    if (not isinstance(roles, dict) or set(roles) != {"base", "processor"}
            or any(not isinstance(v, str) or v not in assets for v in roles.values())):
        raise ValueError("ABot requires pinned base and processor assets")
    if profile.get("use_vggt") is not False:
        raise ValueError("This training recipe requires use_vggt=false")
    if profile.get("head_initialization") != "reinitialize_mismatched":
        raise ValueError("Declare head_initialization=reinitialize_mismatched")
    if profile.get("vocabulary_initialization") != "restore_checkpoint":
        raise ValueError("Declare vocabulary_initialization=restore_checkpoint")
    return deepcopy(profile)


def load_profile(directory, profile_id, expected_sha256, info, *, channel_selection=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", profile_id):
        raise ValueError("Invalid profile_id")
    profile = read_json(inside(directory, f"{profile_id}.json"))
    if profile.get("profile_id") != profile_id or canonical_hash(profile) != expected_sha256:
        raise ValueError("Pinned profile identity/hash mismatch")
    return validate_profile(profile, info, channel_selection=channel_selection)
