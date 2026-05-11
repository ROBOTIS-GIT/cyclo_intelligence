#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

"""Phase 4 — VLA-semantic robot config schema parsing.

Single source of truth for reading the per-robot yaml under
``shared/robot_configs/<robot>_config.yaml``. The yaml is structured
around what a VLA dataset / training pipeline cares about:

  observation:
    images:
      <cam_name>:
        topic: <ros topic>
        rotation_deg: <0/90/180/270>      # optional
    state:
      <group_name>:
        topic: <ros topic>
        msg_type: <ros msg type string>
        joint_names: [<name>, ...]
  action:
    <modality>:
      topic: <ros topic>                  # both inference command + record target
      msg_type: <ros msg type string>
      joint_names: [<name>, ...]
  recording:
    extra_topics: [<topic>, ...]          # /tf, camera_info, ...
  urdf_path: <path>
  robot_name: <human readable>            # optional

Kept self-contained (no `shared` package imports) so policy containers
can drop this file onto sys.path next to the yaml — the
``shared/robot_configs/`` directory is bind-mounted into each container
as ``/orchestrator_config/``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def find_robot_config_path(
    robot_type: str,
    explicit_path: Optional[str] = None,
) -> Path:
    """Locate ``<robot_type>_config.yaml`` across the three deployment shapes.

    Resolution order:
      1. ``explicit_path`` if supplied (caller passed a fully-qualified path).
      2. ``ORCHESTRATOR_CONFIG_PATH`` env var (a directory).
      3. ``ROBOT_CLIENT_CONFIG_DIR`` env var (a directory).
      4. ``/orchestrator_config/`` (container bind mount of
         ``shared/robot_configs/``).
      5. Source-tree fallback: walk up from this file until we find
         ``shared/robot_configs/`` (host-side dev / unit tests).
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    candidates: List[Path] = []
    for env_var in ("ORCHESTRATOR_CONFIG_PATH", "ROBOT_CLIENT_CONFIG_DIR"):
        env_dir = os.environ.get(env_var)
        if env_dir:
            candidates.append(Path(env_dir) / f"{robot_type}_config.yaml")

    candidates.append(Path("/orchestrator_config") / f"{robot_type}_config.yaml")

    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "shared" / "robot_configs" / f"{robot_type}_config.yaml"
        if cand.exists():
            candidates.append(cand)
            break

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Robot config for '{robot_type}' not found. Searched: "
        f"{[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_robot_section(
    robot_type: str,
    explicit_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read yaml and drill into ``orchestrator.ros__parameters.<robot_type>``."""
    path = find_robot_config_path(robot_type, explicit_path=explicit_path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    try:
        section = raw["orchestrator"]["ros__parameters"][robot_type]
    except KeyError as e:
        raise KeyError(
            f"orchestrator.ros__parameters.{robot_type} missing in {path}: {e}"
        ) from e
    # Stash the config file's directory so accessors can resolve any
    # relative paths (e.g. urdf_path) anchored at the yaml location.
    section["__config_dir__"] = str(path.parent)
    return section


# ---------------------------------------------------------------------------
# Field accessors — all return shallow copies of the yaml structure so
# callers can mutate without poisoning the cached dict.
# ---------------------------------------------------------------------------


def get_image_topics(section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return ``{cam_name: {topic, rotation_deg?, msg_type}}``.

    Default ``msg_type`` is ``sensor_msgs/msg/CompressedImage`` — the only
    image type the conversion + web video pipelines support today. yaml
    can override per-camera if a non-compressed source ever lands.
    """
    images = (section.get("observation") or {}).get("images") or {}
    result: Dict[str, Dict[str, Any]] = {}
    for name, cfg in images.items():
        if not isinstance(cfg, dict):
            continue
        entry: Dict[str, Any] = {
            "topic": cfg["topic"],
            "msg_type": cfg.get("msg_type", "sensor_msgs/msg/CompressedImage"),
        }
        rot = cfg.get("rotation_deg")
        if rot:
            entry["rotation_deg"] = int(rot)
        result[name] = entry
    return result


def get_state_groups(section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return ``{group_name: {topic, msg_type, joint_names}}``.

    ``joint_names`` is the layout the driver publishes. For non-JointState
    sources (Odometry, Twist) it's the synthetic axis labels (e.g.
    ``[linear_x, linear_y, angular_z]``).
    """
    groups = (section.get("observation") or {}).get("state") or {}
    result: Dict[str, Dict[str, Any]] = {}
    for name, cfg in groups.items():
        if not isinstance(cfg, dict):
            continue
        result[name] = {
            "topic": cfg["topic"],
            "msg_type": cfg.get("msg_type", "sensor_msgs/msg/JointState"),
            "joint_names": list(cfg.get("joint_names") or []),
        }
    return result


def get_action_groups(section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return ``{modality: {topic, msg_type, joint_names}}``.

    Each entry's ``topic`` is BOTH the inference command target (Process B
    publishes here) AND the rosbag record target. ``joint_names`` is the
    dimension layout for that slice of the action vector.
    """
    actions = section.get("action") or {}
    result: Dict[str, Dict[str, Any]] = {}
    for modality, cfg in actions.items():
        if not isinstance(cfg, dict):
            continue
        result[modality] = {
            "topic": cfg["topic"],
            "msg_type": cfg.get("msg_type", "trajectory_msgs/msg/JointTrajectory"),
            "joint_names": list(cfg.get("joint_names") or []),
        }
    return result


def get_action_joint_names(section: Dict[str, Any]) -> Dict[str, List[str]]:
    """Convenience: ``{modality: joint_names}`` straight from action groups."""
    return {m: cfg["joint_names"] for m, cfg in get_action_groups(section).items()}


def get_recording_extra_topics(section: Dict[str, Any]) -> List[str]:
    extras = (section.get("recording") or {}).get("extra_topics") or []
    return [t for t in extras if t]


def get_recording_topics(section: Dict[str, Any]) -> List[str]:
    """Full topic list for rosbag recording: images + state + action + extras.

    Order is deterministic (yaml insertion order, then extras appended) so
    the recorder's topic inventory is reproducible across runs.
    """
    topics: List[str] = []
    for cfg in get_image_topics(section).values():
        topics.append(cfg["topic"])
    for cfg in get_state_groups(section).values():
        topics.append(cfg["topic"])
    for cfg in get_action_groups(section).values():
        topics.append(cfg["topic"])
    topics.extend(get_recording_extra_topics(section))
    return topics


def get_urdf_path(section: Dict[str, Any]) -> str:
    """Return the URDF path, resolving relative entries against the yaml dir.

    yaml may set ``urdf_path`` as either an absolute path or a path
    relative to the config file's directory (e.g. ``urdf/<robot>.urdf``).
    Relative entries are anchored at ``__config_dir__`` (stashed by
    :func:`load_robot_section`); if that anchor is missing the raw
    string is returned unchanged.
    """
    raw = section.get("urdf_path") or ""
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    config_dir = section.get("__config_dir__")
    if not config_dir:
        return str(p)
    return str((Path(config_dir) / p).resolve())


def get_robot_name(section: Dict[str, Any]) -> str:
    return str(section.get("robot_name") or "")
