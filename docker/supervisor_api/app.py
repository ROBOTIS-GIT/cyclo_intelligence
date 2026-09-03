#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
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

"""supervisor_api — PLAN §4.7 + §4.8 control plane.

Thin FastAPI layer sitting between the UI (via nginx /api/) and:
  (a) the s6-rc service manager inside this container, for the ROS2
      longruns (orchestrator / cyclo_data / web_video_server), and
  (b) the host Docker daemon, for policy containers that ship
      out-of-image (lerobot — and groot once D10-groot lands).

Run as:
    uvicorn supervisor_api.app:app \
        --host "${CYCLO_SUPERVISOR_API_HOST:-127.0.0.1}" \
        --port "${CYCLO_SUPERVISOR_API_PORT:-7100}"

nginx proxies /api/ → 127.0.0.1:7100 (Step 6-E).

Environment overrides:
    CYCLO_SUPERVISOR_API_HOST         bind host (default 127.0.0.1)
    CYCLO_SUPERVISOR_API_PORT         bind port (default 7100)
    CYCLO_SUPERVISOR_API_REPO_MOUNT   in-container path of the repo bind-mount
                                      (default /root/ros2_ws/src/cyclo_intelligence)
    CYCLO_SUPERVISOR_API_COMPOSE_FILE absolute path to docker-compose.yml inside
                                      this container (default <repo-mount>/docker/docker-compose.yml)
    CYCLO_SUPERVISOR_API_CONTAINER_NAME
                                      Docker container name to inspect for
                                      host-side bind mount paths
                                      (default cyclo_intelligence fallback)
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt, ValidationError

# Tests load this file directly under the synthetic module name
# ``supervisor_api_app``. Pin the package parent and load the navigation
# router from the sibling file so route registration does not depend on
# pytest's import path or module cache state.
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

_NAVIGATION_PATH = Path(__file__).resolve().with_name("navigation.py")
_NAVIGATION_SPEC = importlib.util.spec_from_file_location(
    "supervisor_api.navigation",
    _NAVIGATION_PATH,
)
if _NAVIGATION_SPEC is None or _NAVIGATION_SPEC.loader is None:
    raise ImportError(f"Cannot load navigation router from {_NAVIGATION_PATH}")
_navigation_module = importlib.util.module_from_spec(_NAVIGATION_SPEC)
sys.modules[_NAVIGATION_SPEC.name] = _navigation_module
_NAVIGATION_SPEC.loader.exec_module(_navigation_module)
navigation_router = _navigation_module.router


logger = logging.getLogger("supervisor_api")


def _include_router_with_eager_routes(fastapi_app, router) -> None:
    """Register a router and keep concrete route paths visible.

    FastAPI 0.139 stores included routers as lazy ``_IncludedRouter`` entries.
    Runtime dispatch can still resolve them, but tests and simple health checks
    that inspect ``app.routes`` do not see the concrete ``route.path`` values.
    Keep the normal include call for older FastAPI releases, then expand the
    router routes only when the concrete paths are absent.
    """
    fastapi_app.include_router(router)
    expected_paths = {
        route.path for route in router.routes if hasattr(route, "path")
    }
    registered_paths = {
        route.path for route in fastapi_app.routes if hasattr(route, "path")
    }
    if expected_paths.issubset(registered_paths):
        return

    fastapi_app.router.routes = [
        route for route in fastapi_app.router.routes
        if getattr(route, "original_router", None) is not router
    ]
    fastapi_app.router.routes.extend(router.routes)


# -- s6-rc runner --------------------------------------------------------------


# Names the UI may start/stop. Kept explicit so a stray POST can't
# poke at s6-agent or the log pipelines.
_USER_SERVICES: tuple[str, ...] = (
    "orchestrator",
    "cyclo_data",
    "bt_node",
    "web_video_server",
)

_BT_ROBOT_TYPE_FILE = "/run/cyclo_intelligence/bt_node_robot_type"
_BT_SUPPORTED_ROBOT_TYPE = "ffw_sg2_rev1"
_ROBOT_TYPE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class _S6Result:
    rc: int
    stdout: str
    stderr: str


async def _run(
    *cmd: str,
    timeout: float = 10.0,
    env: Optional[Dict[str, str]] = None,
) -> _S6Result:
    """Run a subprocess, return stdout/stderr/rc."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, f"{cmd[0]} timed out after {timeout}s")
    return _S6Result(
        rc=proc.returncode or 0,
        stdout=stdout.decode(errors="replace").strip(),
        stderr=stderr.decode(errors="replace").strip(),
    )


def _require_known_service(name: str) -> None:
    if name not in _USER_SERVICES:
        raise HTTPException(
            404,
            f"Unknown service '{name}'. Known: {', '.join(_USER_SERVICES)}",
        )


def _validate_robot_type(robot_type: str) -> str:
    normalized = robot_type.strip()
    if not normalized:
        raise HTTPException(400, "robot_type is required")
    if not _ROBOT_TYPE_RE.fullmatch(normalized):
        raise HTTPException(400, "robot_type contains unsupported characters")
    return normalized


def _validate_bt_robot_type(robot_type: str = "") -> str:
    normalized = robot_type.strip() or _BT_SUPPORTED_ROBOT_TYPE
    normalized = _validate_robot_type(normalized)
    if normalized != _BT_SUPPORTED_ROBOT_TYPE:
        raise HTTPException(
            400,
            "bt_node currently supports only "
            f"{_BT_SUPPORTED_ROBOT_TYPE}",
        )
    return normalized


def _write_bt_robot_type(robot_type: str) -> None:
    os.makedirs(os.path.dirname(_BT_ROBOT_TYPE_FILE), exist_ok=True)
    with open(_BT_ROBOT_TYPE_FILE, "w", encoding="utf-8") as f:
        f.write(robot_type + "\n")


# -- API models ----------------------------------------------------------------


class ServiceStatus(BaseModel):
    name: str
    state: Literal["up", "down", "unknown"]
    pid: Optional[int] = None
    uptime_s: Optional[int] = None
    raw: str


class ServiceList(BaseModel):
    services: List[ServiceStatus]


class ActionResult(BaseModel):
    ok: bool
    message: str


class ServiceActionRequest(BaseModel):
    robot_type: str = ""


class HealthResponse(BaseModel):
    ok: bool
    container: str
    s6_ready: bool


class WorkspaceMountResponse(BaseModel):
    container_root: str
    host_root: Optional[str] = None
    host_available: bool
    message: str = ""


class BackendStatus(BaseModel):
    name: str
    image: str
    image_pulled: bool
    image_status: Literal["current", "stale", "missing"]
    container_state: Literal["running", "exited", "not_created", "unknown"]
    container_id: Optional[str] = None
    raw_state: Optional[str] = None
    services: List[ServiceStatus] = Field(default_factory=list)


class TrtBuildRequest(BaseModel):
    model_path: str
    engine_path: str = ""
    robot_type: str
    task_instruction: str = ""
    workspace_mb: Optional[int] = None
    force: bool = False


class TrtEngineStatus(BaseModel):
    model_path: str
    engine_path: str
    status: Literal["missing", "building", "ready", "failed", "unknown"]
    message: str = ""
    engine_size_bytes: Optional[int] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list)


_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS: tuple[str, ...] = (
    "visual_backbone",
    "cvae_encoder",
    "transformer_encoder",
    "action_decoder",
)
_OFFLINE_RL_CRITIC_SOURCES: tuple[str, ...] = (
    "resume_checkpoint",
    "parent_checkpoint",
    "policy_warmup",
    "random",
)
_OFFLINE_RL_ACTOR_OBJECTIVES: tuple[str, ...] = ("td3", "td3_bc")


class OfflineRLStartRequest(BaseModel):
    # ``dataset_path`` remains for older UI/API clients.  New collection-round
    # clients send the complete, chronologically ordered replay as
    # ``dataset_paths`` so LeRobot roots stay immutable on disk.
    dataset_path: str = ""
    dataset_paths: List[str] = Field(default_factory=list)
    act_checkpoint: str
    parent_checkpoint: str = ""
    # The optimizer family and its actor-loss contract are deliberately
    # separate.  This keeps TD3 as one algorithm in the UI/API while allowing
    # callers to choose pure Q maximization or the success-masked BC variant.
    algorithm: str = "td3"
    actor_objective: Literal["td3", "td3_bc"] = "td3_bc"
    robot_type: str
    batch_size: StrictInt = Field(default=4, ge=1, le=64)
    critic_epochs: int = Field(default=10, ge=1, le=1000)
    actor_equivalent_epochs: int = Field(default=5, ge=1, le=500)
    actor_trainable_groups: List[str] = Field(
        default_factory=lambda: list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)
    )


class OfflineRLStopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class OfflineRLCancelRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class OfflineRLLossPoint(BaseModel):
    """One finite ACT-TD3 loss sample keyed by completed critic updates."""

    step: StrictInt = Field(ge=0)
    critic_loss: Optional[float] = None
    actor_loss: Optional[float] = None


class OfflineRLRLMetricPoint(BaseModel):
    """One finite replay-round mean keyed by the public RL epoch."""

    rl_epoch: StrictInt = Field(ge=1)
    actor_loss_mean: Optional[float] = None
    critic_loss_mean: Optional[float] = None
    replay_average_reward: Optional[float] = None


class OfflineRLStatus(BaseModel):
    status: Literal[
        "idle",
        "running",
        "completed",
        "failed",
        "stopped",
        "cancelled",
    ]
    algorithm: Literal["td3"] = "td3"
    actor_objective: Literal["td3", "td3_bc"] = "td3_bc"
    percentage: float = 0.0
    episode_count: int = 0
    round_index: int = 0
    round_episode_count: int = 0
    batch_size: int = Field(default=4, ge=1, le=64)
    critic_epochs: int = 10
    actor_equivalent_epochs: int = 5
    actor_trainable_groups: List[str] = Field(
        default_factory=lambda: list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)
    )
    success_count: int = 0
    failure_count: int = 0
    completed_epochs: int = 0
    total_epochs: int = 10
    completed_critic_updates: int = 0
    total_critic_updates: int = 0
    completed_actor_updates: int = 0
    total_actor_updates: int = 0
    critic_loss: Optional[float] = None
    actor_loss: Optional[float] = None
    loss_history: List[OfflineRLLossPoint] = Field(default_factory=list)
    rl_metric_history: List[OfflineRLRLMetricPoint] = Field(default_factory=list)
    eta_seconds: Optional[float] = None
    model_path: str = ""
    checkpoint_path: str = ""
    critic_source: Literal[
        "",
        "resume_checkpoint",
        "parent_checkpoint",
        "policy_warmup",
        "random",
    ] = ""
    critic_checkpoint: str = ""
    message: str = ""
    job_id: str = ""
    dataset_path: str = ""
    dataset_paths: List[str] = Field(default_factory=list)
    act_checkpoint: str = ""
    parent_checkpoint: str = ""
    output_dir: str = ""
    returncode: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list)


class ACTTD3CriticWarmupStartRequest(BaseModel):
    """Warm up the critics attached to one selected ACT policy."""

    dataset_path: str = ""
    dataset_paths: List[str] = Field(default_factory=list)
    act_checkpoint: str
    robot_type: str
    batch_size: StrictInt = Field(default=4, ge=1, le=64)
    critic_updates: StrictInt = Field(default=5000, ge=1, le=1_000_000)


class ACTTD3CriticWarmupStopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class ACTTD3CriticWarmupStatus(BaseModel):
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    percentage: float = 0.0
    completed_critic_updates: int = 0
    total_critic_updates: int = 5000
    durable_checkpoint_updates: int = 0
    critic_loss: Optional[float] = None
    target_mean: Optional[float] = None
    eta_seconds: Optional[float] = None
    actor_exactly_unchanged: Optional[bool] = None
    episode_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    batch_size: int = Field(default=4, ge=1, le=64)
    checkpoint_path: str = ""
    manifest_path: str = ""
    message: str = ""
    job_id: str = ""
    dataset_path: str = ""
    dataset_paths: List[str] = Field(default_factory=list)
    act_checkpoint: str = ""
    returncode: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list)


class ImitationLearningStartRequest(BaseModel):
    """Start one behavior-cloning job over immutable LeRobot roots."""

    dataset_path: str = ""
    dataset_paths: List[str] = Field(default_factory=list)
    policy_type: Literal["act", "multi_task_dit"] = "act"
    task_instruction: str = Field(default="", max_length=1000)
    steps: StrictInt = Field(default=80_000, ge=1, le=1_000_000)
    batch_size: StrictInt = Field(default=8, ge=1, le=64)
    save_freq: StrictInt = Field(default=10_000, ge=1, le=1_000_000)
    chunk_size: StrictInt = Field(default=30, ge=1, le=100)
    trainable_groups: Optional[List[str]] = None


class ImitationLearningStopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class ImitationLearningStatus(BaseModel):
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    percentage: float = 0.0
    episode_count: int = 0
    excluded_episode_count: int = 0
    completed_steps: int = 0
    total_steps: int = 80_000
    batch_size: int = Field(default=8, ge=1, le=64)
    save_freq: int = 10_000
    chunk_size: int = 30
    loss: Optional[float] = None
    l1_loss: Optional[float] = None
    kld_loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    model_path: str = ""
    checkpoint_path: str = ""
    message: str = ""
    job_id: str = ""
    dataset_path: str = ""
    dataset_paths: List[str] = Field(default_factory=list)
    policy_type: Literal["act", "multi_task_dit"] = "act"
    task_instruction: str = ""
    trainable_groups: List[str] = Field(
        default_factory=lambda: list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)
    )
    output_dir: str = ""
    returncode: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list)


class OfflineRLDatasetEpisodeMedia(BaseModel):
    camera_key: str
    relative_path: str
    from_s: Optional[float] = None
    to_s: Optional[float] = None


class OfflineRLDatasetEpisodeData(BaseModel):
    """Replay-compatible state/action samples for one LeRobot episode."""

    joint_timestamps: List[float] = Field(default_factory=list)
    joint_names: List[str] = Field(default_factory=list)
    joint_positions: List[float] = Field(default_factory=list)
    action_timestamps: List[float] = Field(default_factory=list)
    action_names: List[str] = Field(default_factory=list)
    action_values: List[float] = Field(default_factory=list)
    duration: float = Field(default=0.0, ge=0.0)


class OfflineRLDatasetEpisode(BaseModel):
    index: int
    frames: int
    outcome: Literal["success", "failure", "unlabeled"]
    tasks: List[str] = Field(default_factory=list)
    media: List[OfflineRLDatasetEpisodeMedia] = Field(default_factory=list)


class OfflineRLDataEpochOutcomeCounts(BaseModel):
    total: int = Field(ge=0)
    success: int = Field(ge=0)
    failure: int = Field(ge=0)
    unlabeled: int = Field(ge=0)


class OfflineRLDataEpochProvenance(BaseModel):
    schema_version: Literal[1] = 1
    data_epoch: int = Field(ge=0)
    epoch_name: str
    output_root: str
    source_mcap: str
    behavior_policy_path: str = ""
    boundary_reason: str
    created_at: str
    fps: int = Field(ge=1, le=120)
    formats: List[Literal["v2.1", "v3.0"]] = Field(min_length=1)
    outcome_counts: OfflineRLDataEpochOutcomeCounts
    expected_outputs: Dict[str, str] = Field(default_factory=dict)


class OfflineRLDataEpochReserveRequest(BaseModel):
    destination_root: str
    source_mcap: str
    behavior_policy_path: str = ""
    boundary_reason: str = Field(
        default="manual_conversion",
        min_length=1,
        max_length=128,
    )
    fps: int = Field(default=15, ge=1, le=120)
    formats: List[Literal["v2.1", "v3.0"]] = Field(
        default_factory=lambda: ["v2.1", "v3.0"],
        min_length=1,
    )


class OfflineRLDatasetSummary(BaseModel):
    dataset_path: str
    name: str
    version: str
    fps: float
    total_episodes: int
    total_frames: int
    camera_count: int
    success_count: int
    failure_count: int
    unlabeled_count: int
    success_rate: Optional[float] = None
    episodes: List[OfflineRLDatasetEpisode] = Field(default_factory=list)
    data_epoch_provenance: Optional[OfflineRLDataEpochProvenance] = None


class OfflineRLDatasetInventory(BaseModel):
    root_path: str
    datasets: List[OfflineRLDatasetSummary] = Field(default_factory=list)


class OfflineRLDatasetDeleteRequest(BaseModel):
    dataset_path: str
    episode_indices: List[int] = Field(min_length=1)


class OfflineRLDatasetDeleteResult(BaseModel):
    ok: bool
    message: str
    dataset: Optional[OfflineRLDatasetSummary] = None
    dataset_deleted: bool = False


# -- Backend (policy container) wiring -----------------------------------------


# Compose file + repo-mount paths inside this container — the cyclo_intelligence
# service bind-mounts the repo root at /root/ros2_ws/src/cyclo_intelligence by
# default (live edits during dev). Override both with env vars when the mount
# point differs (e.g. running supervisor_api on the host for debugging).
_CYCLO_REPO_MOUNT = os.environ.get(
    "CYCLO_SUPERVISOR_API_REPO_MOUNT",
    "/root/ros2_ws/src/cyclo_intelligence",
)
_COMPOSE_FILE_IN_CONTAINER = os.environ.get(
    "CYCLO_SUPERVISOR_API_COMPOSE_FILE",
    f"{_CYCLO_REPO_MOUNT}/docker/docker-compose.yml",
)
_COMPOSE_OVERRIDE_IN_CONTAINER = os.path.join(
    os.path.dirname(_COMPOSE_FILE_IN_CONTAINER),
    "docker-compose.override.yml",
)


def _detect_arch() -> str:
    machine = os.uname().machine
    return "arm64" if machine in ("aarch64", "arm64") else "amd64"


_BACKEND_ARCH = os.environ.get("ARCH", _detect_arch())


# Image versions are hardcoded per backend below since each service has
# its own release cadence. ARCH still falls back to a uname-based sniff
# because compose only interpolates env vars on the host invocation, so
# inside the container the env var isn't set.
_BACKENDS: Dict[str, Dict[str, str]] = {
    "lerobot": {
        "service": "lerobot",
        "container": "lerobot_server",
        "image": f"robotis/lerobot-zenoh:1.3.2-{_BACKEND_ARCH}",
        "services": ["main-runtime", "engine-process"],
    },
    "groot": {
        "service": "groot",
        "container": "groot_server",
        "image": f"robotis/groot-zenoh:1.3.4-{_BACKEND_ARCH}",
        "services": ["main-runtime", "engine-process"],
    },
}

_REQUIRED_BACKEND_MOUNTS: Dict[str, tuple[str, ...]] = {
    "lerobot": (
        "/workspace",
        "/robot_client_sdk",
        "/action_chunk_processing_sdk",
        "/policy_runtime",
        "/app/lerobot_engine",
        "/orchestrator_config",
    ),
    "groot": (
        "/workspace",
        "/robot_client_sdk",
        "/action_chunk_processing_sdk",
        "/policy_runtime",
        "/app/groot_engine",
        "/app/runtime",
        "/orchestrator_config",
    ),
}

_GROOT_MODEL_ROOT = "/workspace/model/groot"


@dataclass
class _TrtBuildJob:
    model_path: str
    engine_path: str
    log_path: str
    started_at: float
    status: str = "building"
    message: str = "Building TensorRT engine"
    process: Optional[subprocess.Popen] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None


_TRT_BUILD_JOBS: Dict[str, _TrtBuildJob] = {}
_TRT_BUILD_LOCK = threading.Lock()


_OFFLINE_RL_DATASET_ROOT = Path("/workspace/lerobot")
_OFFLINE_RL_ROSBAG_ROOT = Path("/workspace/rosbag2")
_OFFLINE_RL_MODEL_ROOT = Path("/workspace/model/lerobot")
_OFFLINE_RL_OUTPUT_ROOT = _OFFLINE_RL_MODEL_ROOT / "offline_rl"
_OFFLINE_RL_LOG_ROOT = Path("/tmp/cyclo_offline_rl")
_OFFLINE_RL_CACHE_ROOT = "/tmp/cyclo_offline_rl_cache"
_OFFLINE_RL_DATASET_LOCK_PATH = Path("/workspace/.cyclo_dataset.lock")
_OFFLINE_RL_MAX_EPISODES = 200
_OFFLINE_RL_MAX_NEW_EPISODES = 50
_OFFLINE_RL_LOSS_HISTORY_POINTS = 500
_OFFLINE_RL_METRIC_HISTORY_POINTS = 200
_OFFLINE_RL_EPISODE_DATA_MAX_FRAMES = 20_000
_OFFLINE_RL_EPISODE_DATA_MAX_FIELDS = 256
_OFFLINE_RL_EPISODE_DATA_MAX_VALUES = 2_000_000
_OFFLINE_RL_LOG_LINES = 100
_OFFLINE_RL_DATA_EPOCH_FILE = "cyclo_data_epoch.json"
_OFFLINE_RL_DATA_EPOCH_PATTERN = re.compile(r"^data_epoch_(\d{4,})$")

_IMITATION_LEARNING_OUTPUT_ROOT = _OFFLINE_RL_MODEL_ROOT / "imitation_learning"
_IMITATION_LEARNING_LOG_ROOT = Path("/tmp/cyclo_imitation_learning")
_IMITATION_LEARNING_CACHE_ROOT = "/tmp/cyclo_imitation_learning_cache"
_IMITATION_LEARNING_DEFAULT_TASK_INSTRUCTION = "pick up the jelly bag"
_IMITATION_LEARNING_POLICY_LABELS = {
    "act": "ACT",
    "multi_task_dit": "MultiTaskDiT",
}
_IMITATION_LEARNING_POLICY_CHUNK_SIZES = {
    "act": 30,
    "multi_task_dit": 16,
}
_IMITATION_LEARNING_POLICY_MODULES = {
    "act": "cyclo_brain.algorithm.il.act_bc.training_cli",
    "multi_task_dit": "cyclo_brain.algorithm.il.multi_task_dit.training_cli",
}
_IMITATION_LEARNING_POLICY_OUTPUT_PREFIXES = {
    "act": "act_bc",
    "multi_task_dit": "multi_task_dit_bc",
}


@dataclass
class _OfflineRLJob:
    job_id: str
    dataset_path: str
    act_checkpoint: str
    parent_checkpoint: str
    output_dir: str
    episode_count: int
    log_path: str
    algorithm: str = "td3"
    actor_objective: str = "td3_bc"
    dataset_paths: List[str] = field(default_factory=list)
    round_index: int = 1
    round_episode_count: int = 0
    batch_size: int = 4
    critic_epochs: int = 10
    actor_equivalent_epochs: int = 5
    actor_trainable_groups: List[str] = field(
        default_factory=lambda: list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)
    )
    status: str = "running"
    percentage: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    completed_epochs: int = 0
    total_epochs: int = 10
    completed_critic_updates: int = 0
    total_critic_updates: int = 0
    completed_actor_updates: int = 0
    total_actor_updates: int = 0
    critic_loss: Optional[float] = None
    actor_loss: Optional[float] = None
    loss_history: List[OfflineRLLossPoint] = field(default_factory=list)
    rl_metric_history: List[OfflineRLRLMetricPoint] = field(default_factory=list)
    eta_seconds: Optional[float] = None
    model_path: str = ""
    checkpoint_path: str = ""
    critic_source: str = ""
    critic_checkpoint: str = ""
    message: str = "Starting ACT-TD3 offline training"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.log_tail is None:
            self.log_tail = []
        if not self.dataset_paths and self.dataset_path:
            self.dataset_paths = [self.dataset_path]


_OFFLINE_RL_JOB: Optional[_OfflineRLJob] = None
_OFFLINE_RL_LOCK = threading.Lock()
_OFFLINE_RL_DATASET_EDIT_LOCK = threading.Lock()
_OFFLINE_RL_DATASET_EDIT_ACTIVE = False


@dataclass
class _ACTTD3CriticWarmupJob:
    job_id: str
    dataset_path: str
    dataset_paths: List[str]
    act_checkpoint: str
    checkpoint_path: str
    manifest_path: str
    run_checkpoint_path: str
    episode_count: int
    success_count: int
    failure_count: int
    batch_size: int
    log_path: str
    status: str = "running"
    percentage: float = 0.0
    completed_critic_updates: int = 0
    total_critic_updates: int = 5000
    durable_checkpoint_updates: int = 0
    critic_loss: Optional[float] = None
    target_mean: Optional[float] = None
    eta_seconds: Optional[float] = None
    actor_exactly_unchanged: Optional[bool] = None
    message: str = "Starting ACT-TD3 critic warm-up"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    result_complete: bool = False
    artifact_reported: bool = False
    contract_mismatch: bool = False
    returncode: Optional[int] = None
    log_tail: List[str] = field(default_factory=list)


_ACT_TD3_CRITIC_WARMUP_JOB: Optional[_ACTTD3CriticWarmupJob] = None
_ACT_TD3_CRITIC_WARMUP_LOCK = threading.Lock()


@dataclass
class _ImitationLearningJob:
    job_id: str
    dataset_path: str
    dataset_paths: List[str]
    success_episodes: List[List[int]]
    output_dir: str
    episode_count: int
    excluded_episode_count: int
    log_path: str
    total_steps: int = 80_000
    batch_size: int = 8
    save_freq: int = 10_000
    chunk_size: int = 30
    policy_type: str = "act"
    task_instruction: str = ""
    trainable_groups: List[str] = field(
        default_factory=lambda: list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)
    )
    status: str = "running"
    percentage: float = 0.0
    completed_steps: int = 0
    loss: Optional[float] = None
    l1_loss: Optional[float] = None
    kld_loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    model_path: str = ""
    checkpoint_path: str = ""
    message: str = "Starting ACT imitation learning"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.policy_type == "multi_task_dit":
            self.trainable_groups = []
            normalized_instruction = self.task_instruction.strip()
            self.task_instruction = (
                normalized_instruction
                or _IMITATION_LEARNING_DEFAULT_TASK_INSTRUCTION
            )
        else:
            self.task_instruction = ""
        if (
            self.policy_type == "multi_task_dit"
            and self.message == "Starting ACT imitation learning"
        ):
            self.message = "Starting MultiTaskDiT imitation learning"


_IMITATION_LEARNING_JOB: Optional[_ImitationLearningJob] = None
_IMITATION_LEARNING_LOCK = threading.Lock()


def _docker_client() -> docker.DockerClient:
    return docker.from_env()


def _require_known_backend(name: str) -> Dict[str, str]:
    if name not in _BACKENDS:
        known = ", ".join(_BACKENDS) or "(none)"
        raise HTTPException(
            404, f"Unknown backend '{name}'. Known: {known}"
        )
    return _BACKENDS[name]


_HOST_PROJECT_DIR_CACHE: Optional[str] = None
_HOST_WORKSPACE_DIR_CACHE: Optional[str] = None
_HOST_HUGGINGFACE_DIR_CACHE: Optional[str] = None


def _mount_source_for_destination(mounts, destination: str) -> Optional[str]:
    for mount in mounts:
        if mount.get("Destination") == destination:
            return mount.get("Source")
    return None


def _normalized_host_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    project_dir = None
    try:
        project_dir = _host_project_dir()
    except Exception as e:  # pragma: no cover - defensive around Docker SDK
        logger.debug("could not resolve host project dir for path normalization: %s", e)
    if project_dir:
        host_repo = os.path.dirname(project_dir)
        if path == host_repo or path.startswith(host_repo + os.sep):
            translated = os.path.join(
                _CYCLO_REPO_MOUNT,
                os.path.relpath(path, host_repo),
            )
            return os.path.realpath(translated)
    return os.path.realpath(path)


def _self_container_candidates() -> List[str]:
    candidates = [
        os.environ.get("CYCLO_SUPERVISOR_API_CONTAINER_NAME"),
        os.environ.get("HOSTNAME"),
        "cyclo_intelligence",
    ]
    seen: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def _host_project_dir() -> Optional[str]:
    """Resolve the host-side path to cyclo_intelligence/docker/ by
    inspecting our own container's mounts.

    compose CLI invoked from inside a container still talks to the host
    docker daemon, so any relative path in docker-compose.yml
    (./workspace, ../cyclo_brain/sdk/...) must resolve to the host
    filesystem — not the bind-mount path inside us. We pass this dir
    via --project-directory so compose's relative-path resolution
    points at the host tree even though we're calling from inside.
    """
    global _HOST_PROJECT_DIR_CACHE
    if _HOST_PROJECT_DIR_CACHE is not None:
        return _HOST_PROJECT_DIR_CACHE
    try:
        client = _docker_client()
    except DockerException as e:
        logger.warning("docker init failed during self-inspect: %s", e)
        return None
    for own_id in _self_container_candidates():
        try:
            ctr = client.containers.get(own_id)
        except NotFound:
            continue
        except DockerException as e:
            logger.warning("self-inspect failed for %s: %s", own_id, e)
            continue
        host_repo = _mount_source_for_destination(
            ctr.attrs.get("Mounts", []),
            _CYCLO_REPO_MOUNT,
        )
        if host_repo:
            _HOST_PROJECT_DIR_CACHE = os.path.join(host_repo, "docker")
            return _HOST_PROJECT_DIR_CACHE
    logger.warning(
        "no mount found for %s — compose CLI relative paths will resolve "
        "against the in-container path, which the host docker daemon "
        "cannot satisfy",
        _CYCLO_REPO_MOUNT,
    )
    return None


def _host_workspace_dir() -> Optional[str]:
    """Resolve the host-side directory mounted at /workspace."""
    global _HOST_WORKSPACE_DIR_CACHE
    if _HOST_WORKSPACE_DIR_CACHE is not None:
        return _HOST_WORKSPACE_DIR_CACHE

    try:
        client = _docker_client()
    except DockerException as e:
        logger.warning("docker init failed during workspace self-inspect: %s", e)
    else:
        for own_id in _self_container_candidates():
            try:
                ctr = client.containers.get(own_id)
            except NotFound:
                continue
            except DockerException as e:
                logger.warning("self-inspect for workspace mount failed: %s", e)
                continue
            host_workspace = _mount_source_for_destination(
                ctr.attrs.get("Mounts", []),
                "/workspace",
            )
            if host_workspace:
                _HOST_WORKSPACE_DIR_CACHE = host_workspace
                return _HOST_WORKSPACE_DIR_CACHE

    env_path = os.environ.get("CYCLO_WORKSPACE_DIR")
    if env_path:
        logger.warning(
            "using legacy CYCLO_WORKSPACE_DIR fallback for /workspace: %s",
            env_path,
        )
        _HOST_WORKSPACE_DIR_CACHE = env_path
        return _HOST_WORKSPACE_DIR_CACHE
    return None


def _host_huggingface_dir() -> Optional[str]:
    """Resolve the host-side directory mounted at /root/.cache/huggingface."""
    global _HOST_HUGGINGFACE_DIR_CACHE
    if _HOST_HUGGINGFACE_DIR_CACHE is not None:
        return _HOST_HUGGINGFACE_DIR_CACHE

    env_path = os.environ.get("CYCLO_HUGGINGFACE_DIR")
    if env_path:
        _HOST_HUGGINGFACE_DIR_CACHE = env_path
        return _HOST_HUGGINGFACE_DIR_CACHE

    try:
        client = _docker_client()
    except DockerException as e:
        logger.warning("docker init failed during huggingface self-inspect: %s", e)
        return None

    for own_id in _self_container_candidates():
        try:
            ctr = client.containers.get(own_id)
        except NotFound:
            continue
        except DockerException as e:
            logger.warning("self-inspect for huggingface mount failed: %s", e)
            continue
        host_huggingface = _mount_source_for_destination(
            ctr.attrs.get("Mounts", []),
            "/root/.cache/huggingface",
        )
        if host_huggingface:
            _HOST_HUGGINGFACE_DIR_CACHE = host_huggingface
            return _HOST_HUGGINGFACE_DIR_CACHE
    return None


def _compose_env() -> Dict[str, str]:
    """Build env for host docker compose calls made from this container."""
    env = os.environ.copy()
    workspace_dir = _host_workspace_dir()
    huggingface_dir = _host_huggingface_dir()
    if workspace_dir:
        env["CYCLO_WORKSPACE_DIR"] = workspace_dir
    if huggingface_dir:
        env["CYCLO_HUGGINGFACE_DIR"] = huggingface_dir
    env.setdefault("ARCH", _BACKEND_ARCH)
    return env


def _compose_base_cmd() -> List[str]:
    cmd = ["docker", "compose"]
    project_dir = _host_project_dir()
    if project_dir:
        cmd += ["--project-directory", project_dir]
    cmd += ["-f", _COMPOSE_FILE_IN_CONTAINER]
    if os.path.exists(_COMPOSE_OVERRIDE_IN_CONTAINER):
        cmd += ["-f", _COMPOSE_OVERRIDE_IN_CONTAINER]
    return cmd


# -- ACT-TD3 offline training -------------------------------------------------


def _offline_rl_input_path(
    raw_path: str,
    *,
    root: Path,
    label: str,
    expect_directory: bool,
) -> Path:
    """Resolve one UI path without permitting workspace or symlink escape."""
    value = (raw_path or "").strip()
    candidate = Path(value)
    if not value or not candidate.is_absolute():
        raise HTTPException(400, f"{label} must be an absolute path")

    try:
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(503, f"{label} root is unavailable: {root}") from exc
    if root.is_symlink():
        raise HTTPException(500, f"{label} root must not be a symbolic link")

    lexical = Path(os.path.abspath(value))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, f"{label} must be under {root}") from exc

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HTTPException(400, f"{label} must not traverse symbolic links")

    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"{label} does not exist: {lexical}") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"{label} escapes {root}") from exc

    if expect_directory and not resolved.is_dir():
        raise HTTPException(400, f"{label} must be a directory")
    if not expect_directory and not resolved.is_file():
        raise HTTPException(400, f"{label} must be a file")
    return resolved


def _offline_rl_json_file(path: Path, *, label: str) -> dict:
    try:
        if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("unsafe metadata file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, f"Invalid {label}: expected a JSON object")
    return value


class _OfflineRLDatasetOperationLock:
    """Use the same advisory lock as cyclo_data without importing ROS packages."""

    def __init__(self) -> None:
        self._descriptor: Optional[int] = None

    def __enter__(self) -> "_OfflineRLDatasetOperationLock":
        path = _OFFLINE_RL_DATASET_LOCK_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            try:
                os.chmod(path, 0o666)
            except PermissionError:
                pass
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise HTTPException(
                409,
                "Dataset is busy with conversion, editing, or training",
            ) from exc
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _offline_rl_data_epoch_outcomes(source_mcap: Path) -> OfflineRLDataEpochOutcomeCounts:
    """Count immutable source episode labels before verified conversion removes MCAP."""
    metadata_paths: List[Path] = []
    root_metadata = source_mcap / "episode_info.json"
    if root_metadata.is_file() and not root_metadata.is_symlink():
        metadata_paths.append(root_metadata)
    else:
        try:
            children = sorted(source_mcap.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise HTTPException(400, f"Cannot inspect source_mcap: {source_mcap}") from exc
        for child in children:
            if child.is_symlink() or not child.is_dir():
                continue
            metadata = child / "episode_info.json"
            if metadata.is_file() and not metadata.is_symlink():
                metadata_paths.append(metadata)

    success = 0
    failure = 0
    unlabeled = 0
    for metadata_path in metadata_paths:
        metadata = _offline_rl_json_file(
            metadata_path,
            label="MCAP episode provenance",
        )
        outcome = metadata.get("episode_success")
        if outcome is True:
            success += 1
        elif outcome is False:
            failure += 1
        else:
            unlabeled += 1
    return OfflineRLDataEpochOutcomeCounts(
        total=len(metadata_paths),
        success=success,
        failure=failure,
        unlabeled=unlabeled,
    )


def _offline_rl_write_data_epoch_sidecar(
    epoch_root: Path,
    provenance: OfflineRLDataEpochProvenance,
) -> None:
    """Publish provenance atomically outside the standard LeRobot metadata tree."""
    destination = epoch_root / _OFFLINE_RL_DATA_EPOCH_FILE
    temporary = epoch_root / f".{_OFFLINE_RL_DATA_EPOCH_FILE}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(
        provenance.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    ) + "\n"
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        epoch_stat = epoch_root.stat(follow_symlinks=False)
        os.chown(
            destination,
            epoch_stat.st_uid,
            epoch_stat.st_gid,
            follow_symlinks=False,
        )
        os.chmod(destination, 0o664)
        directory_descriptor = os.open(epoch_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _offline_rl_reserve_data_epoch(
    request: OfflineRLDataEpochReserveRequest,
) -> OfflineRLDataEpochProvenance:
    """Atomically reserve the next monotonic Data Epoch under one UI root."""
    normalized_formats = list(dict.fromkeys(request.formats))
    if not normalized_formats:
        raise HTTPException(400, "At least one LeRobot format is required")

    with _OfflineRLDatasetOperationLock():
        destination_root = _offline_rl_input_path(
            request.destination_root,
            root=_OFFLINE_RL_DATASET_ROOT,
            label="destination_root",
            expect_directory=True,
        )
        source_mcap = _offline_rl_input_path(
            request.source_mcap,
            root=_OFFLINE_RL_ROSBAG_ROOT,
            label="source_mcap",
            expect_directory=True,
        )
        outcomes = _offline_rl_data_epoch_outcomes(source_mcap)

        highest_epoch = -1
        try:
            destination_children = list(destination_root.iterdir())
        except OSError as exc:
            raise HTTPException(
                400,
                f"Cannot inspect destination_root: {destination_root}",
            ) from exc
        for child in destination_children:
            match = _OFFLINE_RL_DATA_EPOCH_PATTERN.fullmatch(child.name)
            if match and child.is_dir() and not child.is_symlink():
                highest_epoch = max(highest_epoch, int(match.group(1)))

        data_epoch = highest_epoch + 1
        epoch_root: Optional[Path] = None
        destination_stat = destination_root.stat(follow_symlinks=False)
        while data_epoch < 1_000_000_000:
            candidate = destination_root / f"data_epoch_{data_epoch:04d}"
            try:
                candidate.mkdir(mode=0o775)
                os.chown(
                    candidate,
                    destination_stat.st_uid,
                    destination_stat.st_gid,
                    follow_symlinks=False,
                )
                os.chmod(candidate, 0o775)
                epoch_root = candidate
                break
            except FileExistsError:
                data_epoch += 1
            except OSError as exc:
                try:
                    candidate.rmdir()
                except OSError:
                    pass
                raise HTTPException(
                    500,
                    f"Cannot reserve Data Epoch under {destination_root}",
                ) from exc
        if epoch_root is None:
            raise HTTPException(409, "Data Epoch sequence is exhausted")

        source_name = source_mcap.name
        expected_outputs: Dict[str, str] = {}
        if "v2.1" in normalized_formats:
            expected_outputs["v21"] = str(epoch_root / f"{source_name}_lerobot_v21")
        if "v3.0" in normalized_formats:
            expected_outputs["v30"] = str(epoch_root / f"{source_name}_lerobot_v30")
        provenance = OfflineRLDataEpochProvenance(
            data_epoch=data_epoch,
            epoch_name=epoch_root.name,
            output_root=str(epoch_root),
            source_mcap=str(source_mcap),
            behavior_policy_path=request.behavior_policy_path.strip(),
            boundary_reason=request.boundary_reason.strip(),
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            fps=request.fps,
            formats=normalized_formats,
            outcome_counts=outcomes,
            expected_outputs=expected_outputs,
        )
        try:
            _offline_rl_write_data_epoch_sidecar(epoch_root, provenance)
        except Exception:
            try:
                epoch_root.rmdir()
            except OSError:
                pass
            raise
        return provenance


def _offline_rl_dataset_data_epoch_provenance(
    dataset: Path,
) -> Optional[OfflineRLDataEpochProvenance]:
    """Read a valid Cyclo epoch sidecar from the dataset's parent directory."""
    epoch_root = dataset.parent
    match = _OFFLINE_RL_DATA_EPOCH_PATTERN.fullmatch(epoch_root.name)
    sidecar = epoch_root / _OFFLINE_RL_DATA_EPOCH_FILE
    if not match or not sidecar.is_file() or sidecar.is_symlink():
        return None
    try:
        raw = _offline_rl_json_file(sidecar, label="Cyclo Data Epoch provenance")
        provenance = OfflineRLDataEpochProvenance(**raw)
        if provenance.data_epoch != int(match.group(1)):
            raise ValueError("data_epoch does not match its directory")
        if provenance.epoch_name != epoch_root.name:
            raise ValueError("epoch_name does not match its directory")
        if Path(provenance.output_root) != epoch_root:
            raise ValueError("output_root does not match its directory")
        if str(dataset) not in provenance.expected_outputs.values():
            raise ValueError("dataset is not an expected Data Epoch output")
        return provenance
    except (HTTPException, OSError, ValueError, ValidationError) as exc:
        logger.warning("ignoring invalid Data Epoch provenance %s: %s", sidecar, exc)
        return None


def _offline_rl_dataset_metadata(raw_path: str) -> tuple[Path, dict]:
    dataset = _offline_rl_input_path(
        raw_path,
        root=_OFFLINE_RL_DATASET_ROOT,
        label="dataset_path",
        expect_directory=True,
    )
    try:
        info_path = _offline_rl_input_path(
            str(dataset / "meta" / "info.json"),
            root=dataset,
            label="LeRobot meta/info.json",
            expect_directory=False,
        )
    except HTTPException as exc:
        raise HTTPException(400, "dataset_path is missing a safe meta/info.json") from exc
    return dataset, _offline_rl_json_file(
        info_path,
        label="LeRobot dataset metadata",
    )


def _offline_rl_scalar(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def _offline_rl_v3_episode_rows(
    dataset: Path,
    total_episodes: int,
) -> List[OfflineRLDatasetEpisode]:
    """Read only the small v3 episode-metadata shards, never frame data."""
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for dataset inspection") from exc

    metadata_root = dataset / "meta" / "episodes"
    if metadata_root.is_symlink() or not metadata_root.is_dir():
        raise HTTPException(400, "LeRobot v3 episode metadata is missing or unsafe")
    files = sorted(metadata_root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise HTTPException(400, "LeRobot v3 episode metadata has no parquet shards")

    rows: List[OfflineRLDatasetEpisode] = []
    seen: set[int] = set()
    success_column = "stats/episode_success/mean"
    for path in files:
        if path.is_symlink():
            raise HTTPException(400, "LeRobot episode metadata must not use symlinks")
        try:
            path.resolve(strict=True).relative_to(dataset)
            schema_names = set(parquet.ParquetFile(path).schema_arrow.names)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"Unsafe LeRobot metadata shard: {path}") from exc
        required = {"episode_index", "length"}
        if not required.issubset(schema_names):
            raise HTTPException(400, f"Invalid LeRobot metadata shard: {path}")
        columns = ["episode_index", "length"]
        if "tasks" in schema_names:
            columns.append("tasks")
        if success_column in schema_names:
            columns.append(success_column)
        try:
            records = parquet.read_table(path, columns=columns).to_pylist()
        except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
            raise HTTPException(400, f"Could not read LeRobot metadata: {path}") from exc
        for record in records:
            try:
                index = int(record["episode_index"])
                frames = int(record["length"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, f"Invalid episode row in {path}") from exc
            if index in seen or index < 0 or frames < 1:
                raise HTTPException(400, f"Invalid or duplicate LeRobot episode {index}")
            seen.add(index)
            raw_success = _offline_rl_scalar(record.get(success_column))
            if raw_success is None:
                outcome: Literal["success", "failure", "unlabeled"] = "unlabeled"
            else:
                try:
                    success_value = float(raw_success)
                    if not math.isfinite(success_value):
                        raise ValueError("episode_success must be finite")
                    outcome = "success" if success_value >= 0.5 else "failure"
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        400,
                        f"Invalid episode_success label for episode {index}",
                    ) from exc
            raw_tasks = record.get("tasks")
            tasks = (
                [str(task) for task in raw_tasks if str(task)]
                if isinstance(raw_tasks, (list, tuple))
                else ([str(raw_tasks)] if raw_tasks else [])
            )
            rows.append(
                OfflineRLDatasetEpisode(
                    index=index,
                    frames=frames,
                    outcome=outcome,
                    tasks=tasks,
                )
            )

    rows.sort(key=lambda episode: episode.index)
    expected = list(range(total_episodes))
    actual = [episode.index for episode in rows]
    if actual != expected:
        raise HTTPException(
            400,
            "LeRobot episode indices are incomplete or non-contiguous",
        )
    return rows


def _offline_rl_v21_jsonl(path: Path, *, dataset: Path, label: str) -> List[dict]:
    """Read bounded v2.1 metadata JSONL without following workspace escapes."""
    try:
        safe_path = _offline_rl_input_path(
            str(path),
            root=dataset,
            label=label,
            expect_directory=False,
        )
        if safe_path.stat().st_size > 128 * 1024 * 1024:
            raise HTTPException(400, f"{label} is too large")
        lines = safe_path.read_text(encoding="utf-8").splitlines()
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(400, f"Could not read {label}: {path}") from exc

    rows: List[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                400,
                f"Invalid {label} JSON at line {line_number}",
            ) from exc
        if not isinstance(value, dict):
            raise HTTPException(
                400,
                f"Invalid {label} row at line {line_number}",
            )
        rows.append(value)
    return rows


def _offline_rl_v21_episode_rows(
    dataset: Path,
    total_episodes: int,
) -> List[OfflineRLDatasetEpisode]:
    """Read v2.1 episode cards from its compact JSONL metadata files."""
    metadata_rows = _offline_rl_v21_jsonl(
        dataset / "meta" / "episodes.jsonl",
        dataset=dataset,
        label="LeRobot v2.1 episode metadata",
    )

    outcome_by_index: Dict[int, Literal["success", "failure", "unlabeled"]] = {}
    stats_path = dataset / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file() and not stats_path.is_symlink():
        stats_rows = _offline_rl_v21_jsonl(
            stats_path,
            dataset=dataset,
            label="LeRobot v2.1 episode statistics",
        )
        for stats_row in stats_rows:
            try:
                index = int(stats_row["episode_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(
                    400,
                    "Invalid LeRobot v2.1 episode statistics index",
                ) from exc
            success_stats = stats_row.get("stats", {}).get("episode_success")
            raw_success = (
                _offline_rl_scalar(success_stats.get("mean"))
                if isinstance(success_stats, dict)
                else None
            )
            if raw_success is None:
                outcome = "unlabeled"
            else:
                try:
                    success_value = float(raw_success)
                    if not math.isfinite(success_value):
                        raise ValueError("episode_success must be finite")
                    outcome = "success" if success_value >= 0.5 else "failure"
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        400,
                        f"Invalid episode_success label for episode {index}",
                    ) from exc
            if index in outcome_by_index:
                raise HTTPException(
                    400,
                    f"Duplicate LeRobot v2.1 episode statistics index {index}",
                )
            outcome_by_index[index] = outcome

    rows: List[OfflineRLDatasetEpisode] = []
    seen: set[int] = set()
    for metadata in metadata_rows:
        try:
            index = int(metadata["episode_index"])
            frames = int(metadata["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, "Invalid LeRobot v2.1 episode row") from exc
        if index in seen or index < 0 or frames < 1:
            raise HTTPException(
                400,
                f"Invalid or duplicate LeRobot v2.1 episode {index}",
            )
        seen.add(index)
        raw_tasks = metadata.get("tasks")
        tasks = (
            [str(task) for task in raw_tasks if str(task)]
            if isinstance(raw_tasks, (list, tuple))
            else ([str(raw_tasks)] if raw_tasks else [])
        )
        rows.append(
            OfflineRLDatasetEpisode(
                index=index,
                frames=frames,
                outcome=outcome_by_index.get(index, "unlabeled"),
                tasks=tasks,
            )
        )

    rows.sort(key=lambda episode: episode.index)
    expected = list(range(total_episodes))
    actual = [episode.index for episode in rows]
    if actual != expected:
        raise HTTPException(
            400,
            "LeRobot v2.1 episode indices are incomplete or non-contiguous",
        )
    return rows


def _offline_rl_video_keys(info: dict) -> List[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    return sorted(
        str(key)
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _offline_rl_safe_media_relative_path(
    dataset: Path,
    candidate: Path,
    *,
    label: str,
) -> Optional[str]:
    """Return one verified dataset-relative media path, or None if absent."""
    try:
        resolved = _offline_rl_input_path(
            str(candidate),
            root=dataset,
            label=label,
            expect_directory=False,
        )
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise
    return resolved.relative_to(dataset).as_posix()


def _offline_rl_v3_video_path(
    dataset: Path,
    info: dict,
    *,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    template = info.get("video_path") or (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    if not isinstance(template, str) or not template.strip():
        raise HTTPException(400, "Invalid LeRobot v3 video_path")
    try:
        relative = Path(template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        ))
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, "Invalid LeRobot v3 video_path template") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(400, "Unsafe LeRobot v3 video_path")
    return dataset / relative


def _offline_rl_v21_episode_media(
    dataset: Path,
    info: dict,
    episodes: List[OfflineRLDatasetEpisode],
) -> Dict[int, List[OfflineRLDatasetEpisodeMedia]]:
    media_by_episode: Dict[int, List[OfflineRLDatasetEpisodeMedia]] = {}
    for episode in episodes:
        media: List[OfflineRLDatasetEpisodeMedia] = []
        for video_key in _offline_rl_video_keys(info):
            path = _offline_rl_v21_episode_path(
                dataset,
                info,
                template_name="video_path",
                episode_index=episode.index,
                video_key=video_key,
            )
            relative_path = _offline_rl_safe_media_relative_path(
                dataset,
                path,
                label=f"LeRobot v2.1 episode {episode.index} video",
            )
            if relative_path is not None:
                media.append(OfflineRLDatasetEpisodeMedia(
                    camera_key=video_key,
                    relative_path=relative_path,
                ))
        media_by_episode[episode.index] = media
    return media_by_episode


def _offline_rl_v3_episode_media(
    dataset: Path,
    info: dict,
    episodes: List[OfflineRLDatasetEpisode],
) -> Dict[int, List[OfflineRLDatasetEpisodeMedia]]:
    video_keys = _offline_rl_video_keys(info)
    if not video_keys:
        return {}

    # The normal v3 row reader has already validated this folder. Keeping this
    # guard tolerant also lets callers inspect metadata-only test fixtures.
    metadata_root = dataset / "meta" / "episodes"
    if not metadata_root.exists():
        return {}
    safe_metadata_root = _offline_rl_input_path(
        str(metadata_root),
        root=dataset,
        label="LeRobot v3 episode metadata",
        expect_directory=True,
    )
    files = sorted(safe_metadata_root.glob("chunk-*/file-*.parquet"))
    if not files:
        return {}
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for media inspection") from exc

    episode_indices = {episode.index for episode in episodes}
    media_by_episode: Dict[int, List[OfflineRLDatasetEpisodeMedia]] = {
        index: [] for index in episode_indices
    }
    seen: set[int] = set()
    for path in files:
        safe_path = _offline_rl_input_path(
            str(path),
            root=dataset,
            label="LeRobot v3 episode metadata shard",
            expect_directory=False,
        )
        try:
            schema_names = set(parquet.ParquetFile(safe_path).schema_arrow.names)
        except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
            raise HTTPException(400, f"Could not inspect LeRobot metadata: {path}") from exc
        columns = ["episode_index"]
        key_columns: Dict[str, tuple[str, str, Optional[str], Optional[str]]] = {}
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            chunk_column = f"{prefix}/chunk_index"
            file_column = f"{prefix}/file_index"
            if chunk_column not in schema_names or file_column not in schema_names:
                continue
            from_column = f"{prefix}/from_timestamp"
            to_column = f"{prefix}/to_timestamp"
            has_from = from_column in schema_names
            has_to = to_column in schema_names
            if has_from != has_to:
                raise HTTPException(
                    400,
                    f"Incomplete LeRobot v3 video timestamps for {video_key}",
                )
            key_columns[video_key] = (
                chunk_column,
                file_column,
                from_column if has_from else None,
                to_column if has_to else None,
            )
            columns.extend([chunk_column, file_column])
            if has_from:
                columns.extend([from_column, to_column])
        try:
            records = parquet.read_table(safe_path, columns=columns).to_pylist()
        except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
            raise HTTPException(400, f"Could not read LeRobot metadata: {path}") from exc

        for record in records:
            try:
                episode_index = int(record["episode_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, f"Invalid episode row in {path}") from exc
            if episode_index not in episode_indices or episode_index in seen:
                raise HTTPException(
                    400,
                    f"Invalid or duplicate LeRobot episode {episode_index}",
                )
            seen.add(episode_index)
            for video_key, (
                chunk_column,
                file_column,
                from_column,
                to_column,
            ) in key_columns.items():
                try:
                    raw_chunk = record[chunk_column]
                    raw_file = record[file_column]
                    if isinstance(raw_chunk, bool) or isinstance(raw_file, bool):
                        raise ValueError("video indices must be integers")
                    chunk_index = int(raw_chunk)
                    file_index = int(raw_file)
                    if chunk_index < 0 or file_index < 0:
                        raise ValueError("video indices must be non-negative")
                    from_s: Optional[float] = None
                    to_s: Optional[float] = None
                    if from_column is not None and to_column is not None:
                        from_s = float(record[from_column])
                        to_s = float(record[to_column])
                        if (
                            not math.isfinite(from_s)
                            or not math.isfinite(to_s)
                            or from_s < 0
                            or to_s <= from_s
                        ):
                            raise ValueError("video timestamps are invalid")
                except (KeyError, TypeError, ValueError) as exc:
                    raise HTTPException(
                        400,
                        f"Invalid LeRobot v3 video metadata for episode {episode_index}",
                    ) from exc
                video_path = _offline_rl_v3_video_path(
                    dataset,
                    info,
                    video_key=video_key,
                    chunk_index=chunk_index,
                    file_index=file_index,
                )
                relative_path = _offline_rl_safe_media_relative_path(
                    dataset,
                    video_path,
                    label=f"LeRobot v3 episode {episode_index} video",
                )
                if relative_path is not None:
                    media_by_episode[episode_index].append(
                        OfflineRLDatasetEpisodeMedia(
                            camera_key=video_key,
                            relative_path=relative_path,
                            from_s=from_s,
                            to_s=to_s,
                        )
                    )
    return media_by_episode


def _offline_rl_attach_episode_media(
    dataset: Path,
    info: dict,
    version: str,
    episodes: List[OfflineRLDatasetEpisode],
) -> List[OfflineRLDatasetEpisode]:
    media_by_episode = (
        _offline_rl_v3_episode_media(dataset, info, episodes)
        if version == "v3.0"
        else _offline_rl_v21_episode_media(dataset, info, episodes)
    )
    return [
        episode.model_copy(update={"media": media_by_episode.get(episode.index, [])})
        for episode in episodes
    ]


def _offline_rl_required_episode_index(info: dict, episode_index: int) -> int:
    total_episodes = info.get("total_episodes")
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, int)
        or total_episodes < 1
    ):
        raise HTTPException(400, "LeRobot total_episodes must be a positive integer")
    if episode_index < 0 or episode_index >= total_episodes:
        raise HTTPException(404, f"LeRobot episode {episode_index} does not exist")
    return total_episodes


def _offline_rl_episode_feature_names(info: dict, feature_key: str) -> List[str]:
    features = info.get("features")
    feature = features.get(feature_key) if isinstance(features, dict) else None
    if not isinstance(feature, dict):
        raise HTTPException(400, f"LeRobot dataset is missing {feature_key}")
    if feature.get("dtype") not in {"float16", "float32", "float64"}:
        raise HTTPException(400, f"LeRobot {feature_key} must be floating point")

    shape = feature.get("shape")
    names = feature.get("names")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 1
        or not isinstance(names, list)
        or len(names) != shape[0]
    ):
        raise HTTPException(400, f"LeRobot {feature_key} shape/names are invalid")
    if any(not isinstance(name, str) for name in names):
        raise HTTPException(400, f"LeRobot {feature_key} names must be strings")
    normalized = [name.strip() for name in names]
    if any(not name for name in normalized) or len(set(normalized)) != len(normalized):
        raise HTTPException(400, f"LeRobot {feature_key} names must be unique")
    if len(normalized) > _OFFLINE_RL_EPISODE_DATA_MAX_FIELDS:
        raise HTTPException(413, f"LeRobot {feature_key} has too many fields")
    return normalized


def _offline_rl_v3_data_path(
    dataset: Path,
    info: dict,
    *,
    chunk_index: int,
    file_index: int,
) -> Path:
    template = info.get("data_path") or (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    if not isinstance(template, str) or not template.strip():
        raise HTTPException(400, "Invalid LeRobot v3 data_path")
    try:
        relative = Path(template.format(
            chunk_index=chunk_index,
            file_index=file_index,
        ))
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, "Invalid LeRobot v3 data_path template") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(400, "Unsafe LeRobot v3 data_path")
    return dataset / relative


def _offline_rl_metadata_integer(record: dict, name: str, *, episode_index: int) -> int:
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (
            isinstance(value, float)
            and (not math.isfinite(value) or not value.is_integer())
        )
    ):
        raise HTTPException(
            400,
            f"Invalid {name} for LeRobot episode {episode_index}",
        )
    parsed = int(value)
    if parsed < 0:
        raise HTTPException(
            400,
            f"Invalid {name} for LeRobot episode {episode_index}",
        )
    return parsed


def _offline_rl_v3_episode_data_location(
    dataset: Path,
    info: dict,
    episode_index: int,
    parquet,
) -> tuple[Path, int, int, int, int]:
    """Resolve one v3 frame range using only compact episode metadata."""
    metadata_root = _offline_rl_input_path(
        str(dataset / "meta" / "episodes"),
        root=dataset,
        label="LeRobot v3 episode metadata",
        expect_directory=True,
    )
    files = sorted(metadata_root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise HTTPException(400, "LeRobot v3 episode metadata has no parquet shards")
    if len(files) > 10_000:
        raise HTTPException(400, "LeRobot v3 episode metadata has too many shards")

    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    locations: List[dict] = []
    seen: set[int] = set()
    for path in files:
        safe_path = _offline_rl_input_path(
            str(path),
            root=dataset,
            label="LeRobot v3 episode metadata shard",
            expect_directory=False,
        )
        try:
            schema_names = set(parquet.ParquetFile(safe_path).schema_arrow.names)
        except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
            raise HTTPException(400, f"Could not inspect LeRobot metadata: {path}") from exc
        if not set(columns).issubset(schema_names):
            raise HTTPException(400, f"Invalid LeRobot metadata shard: {path}")
        try:
            records = parquet.read_table(safe_path, columns=columns).to_pylist()
        except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
            raise HTTPException(400, f"Could not read LeRobot metadata: {path}") from exc
        for record in records:
            index = _offline_rl_metadata_integer(
                record,
                "episode_index",
                episode_index=episode_index,
            )
            if index in seen:
                raise HTTPException(400, f"Duplicate LeRobot episode {index}")
            seen.add(index)
            length = _offline_rl_metadata_integer(
                record,
                "length",
                episode_index=index,
            )
            chunk_index = _offline_rl_metadata_integer(
                record,
                "data/chunk_index",
                episode_index=index,
            )
            file_index = _offline_rl_metadata_integer(
                record,
                "data/file_index",
                episode_index=index,
            )
            from_index = _offline_rl_metadata_integer(
                record,
                "dataset_from_index",
                episode_index=index,
            )
            to_index = _offline_rl_metadata_integer(
                record,
                "dataset_to_index",
                episode_index=index,
            )
            if length < 1 or to_index <= from_index or to_index - from_index != length:
                raise HTTPException(400, f"Invalid frame range for LeRobot episode {index}")
            locations.append({
                "episode_index": index,
                "length": length,
                "chunk_index": chunk_index,
                "file_index": file_index,
                "from_index": from_index,
                "to_index": to_index,
            })

    selected = next(
        (location for location in locations if location["episode_index"] == episode_index),
        None,
    )
    if selected is None:
        raise HTTPException(404, f"LeRobot episode {episode_index} does not exist")

    shard_locations = sorted(
        (
            location for location in locations
            if location["chunk_index"] == selected["chunk_index"]
            and location["file_index"] == selected["file_index"]
        ),
        key=lambda location: location["from_index"],
    )
    shard_start = shard_locations[0]["from_index"]
    expected_from = shard_start
    for location in shard_locations:
        if location["from_index"] != expected_from:
            raise HTTPException(400, "LeRobot v3 data shard ranges are not contiguous")
        expected_from = location["to_index"]
    shard_length = expected_from - shard_start

    local_from = selected["from_index"] - shard_start
    local_to = selected["to_index"] - shard_start
    data_path = _offline_rl_v3_data_path(
        dataset,
        info,
        chunk_index=selected["chunk_index"],
        file_index=selected["file_index"],
    )
    safe_data_path = _offline_rl_input_path(
        str(data_path),
        root=dataset,
        label=f"LeRobot v3 episode {episode_index} data",
        expect_directory=False,
    )
    return safe_data_path, local_from, local_to, selected["length"], shard_length


def _offline_rl_parquet_slice_records(
    parquet,
    path: Path,
    *,
    columns: List[str],
    start: int,
    stop: int,
    expected_row_count: int,
) -> List[dict]:
    """Read one bounded row slice without materializing later shard rows."""
    try:
        parquet_file = parquet.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        row_count = int(parquet_file.metadata.num_rows)
    except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
        raise HTTPException(400, f"Could not inspect LeRobot episode data: {path}") from exc
    if not set(columns).issubset(schema_names):
        raise HTTPException(400, f"LeRobot episode parquet is missing required columns: {path}")
    if row_count != expected_row_count:
        raise HTTPException(400, f"LeRobot data shard row count is invalid: {path}")
    if start < 0 or stop <= start or stop > row_count:
        raise HTTPException(400, f"LeRobot episode frame slice is invalid: {path}")

    rows: List[dict] = []
    cursor = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=2048, columns=columns):
            batch_end = cursor + batch.num_rows
            overlap_start = max(start, cursor)
            overlap_end = min(stop, batch_end)
            if overlap_start < overlap_end:
                rows.extend(batch.slice(
                    overlap_start - cursor,
                    overlap_end - overlap_start,
                ).to_pylist())
            cursor = batch_end
            if cursor >= stop:
                break
    except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
        raise HTTPException(400, f"Could not read LeRobot episode data: {path}") from exc
    if len(rows) != stop - start:
        raise HTTPException(400, f"LeRobot episode frame slice is incomplete: {path}")
    return rows


def _offline_rl_v21_episode_length(dataset: Path, episode_index: int) -> int:
    rows = _offline_rl_v21_jsonl(
        dataset / "meta" / "episodes.jsonl",
        dataset=dataset,
        label="LeRobot v2.1 episode metadata",
    )
    matches = []
    for row in rows:
        try:
            index = int(row.get("episode_index"))
        except (TypeError, ValueError):
            continue
        if index == episode_index:
            matches.append(row)
    if len(matches) != 1:
        raise HTTPException(400, f"Invalid LeRobot v2.1 episode {episode_index} metadata")
    return _offline_rl_metadata_integer(
        matches[0],
        "length",
        episode_index=episode_index,
    )


def _offline_rl_validate_episode_data_size(info: dict, episode_index: int, frames: int) -> None:
    joint_names = _offline_rl_episode_feature_names(info, "observation.state")
    action_names = _offline_rl_episode_feature_names(info, "action")
    if frames < 1:
        raise HTTPException(400, f"LeRobot episode {episode_index} frame count is invalid")
    if frames > _OFFLINE_RL_EPISODE_DATA_MAX_FRAMES:
        raise HTTPException(413, f"LeRobot episode {episode_index} is too long to preview")
    if frames * (len(joint_names) + len(action_names)) > (
        _OFFLINE_RL_EPISODE_DATA_MAX_VALUES
    ):
        raise HTTPException(413, f"LeRobot episode {episode_index} data is too large")


def _offline_rl_episode_data_from_records(
    info: dict,
    episode_index: int,
    records: List[dict],
    *,
    expected_frames: int,
) -> OfflineRLDatasetEpisodeData:
    joint_names = _offline_rl_episode_feature_names(info, "observation.state")
    action_names = _offline_rl_episode_feature_names(info, "action")
    _offline_rl_validate_episode_data_size(info, episode_index, expected_frames)
    if len(records) != expected_frames:
        raise HTTPException(400, f"LeRobot episode {episode_index} frame count is invalid")

    timestamps: List[float] = []
    joint_positions: List[float] = []
    action_values: List[float] = []
    previous_timestamp: Optional[float] = None
    for record in records:
        row_episode = _offline_rl_metadata_integer(
            record,
            "episode_index",
            episode_index=episode_index,
        )
        if row_episode != episode_index:
            raise HTTPException(400, f"LeRobot episode {episode_index} frame slice leaked")
        try:
            timestamp = float(_offline_rl_scalar(record.get("timestamp")))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"Invalid timestamp in episode {episode_index}") from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise HTTPException(400, f"Invalid timestamp in episode {episode_index}")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise HTTPException(400, f"Episode {episode_index} timestamps are not monotonic")
        timestamps.append(timestamp)
        previous_timestamp = timestamp

        for key, names, destination in (
            ("observation.state", joint_names, joint_positions),
            ("action", action_names, action_values),
        ):
            values = record.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != len(names):
                raise HTTPException(400, f"Invalid {key} shape in episode {episode_index}")
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        400,
                        f"Invalid {key} value in episode {episode_index}",
                    ) from exc
                if not math.isfinite(number):
                    raise HTTPException(400, f"Invalid {key} value in episode {episode_index}")
                destination.append(number)

    origin = timestamps[0]
    relative_timestamps = [timestamp - origin for timestamp in timestamps]
    duration = relative_timestamps[-1] if len(relative_timestamps) > 1 else 0.0
    return OfflineRLDatasetEpisodeData(
        joint_timestamps=relative_timestamps,
        joint_names=joint_names,
        joint_positions=joint_positions,
        action_timestamps=list(relative_timestamps),
        action_names=action_names,
        action_values=action_values,
        duration=duration,
    )


def _offline_rl_dataset_episode_data(
    raw_path: str,
    episode_index: int,
) -> OfflineRLDatasetEpisodeData:
    dataset, info = _offline_rl_dataset_metadata(raw_path)
    _offline_rl_required_episode_index(info, episode_index)
    version = str(info.get("codebase_version") or "")
    if version not in {"v2.1", "v3.0"}:
        raise HTTPException(400, f"Unsupported LeRobot version: {version or 'unknown'}")

    # Validate the feature contract before touching potentially large frame data.
    _offline_rl_episode_feature_names(info, "observation.state")
    _offline_rl_episode_feature_names(info, "action")
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for episode inspection") from exc

    columns = ["episode_index", "timestamp", "observation.state", "action"]
    if version == "v3.0":
        data_path, local_from, local_to, expected_frames, shard_frames = (
            _offline_rl_v3_episode_data_location(dataset, info, episode_index, parquet)
        )
        _offline_rl_validate_episode_data_size(info, episode_index, expected_frames)
        records = _offline_rl_parquet_slice_records(
            parquet,
            data_path,
            columns=columns,
            start=local_from,
            stop=local_to,
            expected_row_count=shard_frames,
        )
    else:
        expected_frames = _offline_rl_v21_episode_length(dataset, episode_index)
        _offline_rl_validate_episode_data_size(info, episode_index, expected_frames)
        data_path = _offline_rl_v21_episode_path(
            dataset,
            info,
            template_name="data_path",
            episode_index=episode_index,
        )
        safe_data_path = _offline_rl_input_path(
            str(data_path),
            root=dataset,
            label=f"LeRobot v2.1 episode {episode_index} data",
            expect_directory=False,
        )
        records = _offline_rl_parquet_slice_records(
            parquet,
            safe_data_path,
            columns=columns,
            start=0,
            stop=expected_frames,
            expected_row_count=expected_frames,
        )

    return _offline_rl_episode_data_from_records(
        info,
        episode_index,
        records,
        expected_frames=expected_frames,
    )


def _offline_rl_dataset_summary(raw_path: str) -> OfflineRLDatasetSummary:
    dataset, info = _offline_rl_dataset_metadata(raw_path)
    data_epoch_provenance = _offline_rl_dataset_data_epoch_provenance(dataset)
    version = str(info.get("codebase_version") or "")
    if version not in {"v2.1", "v3.0"}:
        raise HTTPException(
            400,
            "Episode summaries require a LeRobot v2.1 or v3.0 dataset; "
            f"found {version or 'unknown'}",
        )
    total_episodes = info.get("total_episodes")
    total_frames = info.get("total_frames")
    fps = info.get("fps")
    for name, value in (
        ("total_episodes", total_episodes),
        ("total_frames", total_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise HTTPException(400, f"LeRobot {name} must be a positive integer")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or fps <= 0
    ):
        raise HTTPException(400, "LeRobot fps must be finite and positive")

    episodes = (
        _offline_rl_v3_episode_rows(dataset, total_episodes)
        if version == "v3.0"
        else _offline_rl_v21_episode_rows(dataset, total_episodes)
    )
    episodes = _offline_rl_attach_episode_media(dataset, info, version, episodes)
    success_count = sum(item.outcome == "success" for item in episodes)
    failure_count = sum(item.outcome == "failure" for item in episodes)
    unlabeled_count = len(episodes) - success_count - failure_count
    labeled_count = success_count + failure_count
    features = info.get("features")
    camera_count = 0
    if isinstance(features, dict):
        camera_count = sum(
            isinstance(feature, dict) and feature.get("dtype") == "video"
            for feature in features.values()
        )
    success_rate = (
        round((success_count / labeled_count) * 100.0, 2)
        if labeled_count
        else None
    )
    return OfflineRLDatasetSummary(
        dataset_path=str(dataset),
        name=dataset.name,
        version=version,
        fps=float(fps),
        total_episodes=total_episodes,
        total_frames=total_frames,
        camera_count=camera_count,
        success_count=success_count,
        failure_count=failure_count,
        unlabeled_count=unlabeled_count,
        success_rate=success_rate,
        episodes=episodes,
        data_epoch_provenance=data_epoch_provenance,
    )


def _offline_rl_dataset_inventory(raw_root: str = "") -> OfflineRLDatasetInventory:
    """Discover safe local LeRobot datasets recursively, newest first."""
    root_value = (raw_root or str(_OFFLINE_RL_DATASET_ROOT)).strip()
    root = _offline_rl_input_path(
        root_value,
        root=_OFFLINE_RL_DATASET_ROOT,
        label="root_path",
        expect_directory=True,
    )
    candidates: List[tuple[float, OfflineRLDatasetSummary]] = []
    visited = 0
    for current_raw, directory_names, _file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        visited += 1
        if visited > 10000:
            raise HTTPException(400, "LeRobot dataset inventory is too large")
        current = Path(current_raw)
        directory_names[:] = [
            name for name in directory_names
            if not name.startswith(".") and not (current / name).is_symlink()
        ]
        info_path = current / "meta" / "info.json"
        if not info_path.is_file() or info_path.is_symlink():
            continue
        # A dataset is a terminal inventory item. Avoid descending through its
        # data/video trees, which can contain tens of thousands of files.
        directory_names.clear()
        try:
            summary = _offline_rl_dataset_summary(str(current))
            modified_at = info_path.stat().st_mtime
        except (HTTPException, OSError) as exc:
            logger.warning("skipping invalid LeRobot dataset %s: %s", current, exc)
            continue
        candidates.append((modified_at, summary))
        if len(candidates) > 500:
            raise HTTPException(400, "LeRobot dataset inventory exceeds 500 datasets")

    candidates.sort(key=lambda item: (-item[0], item[1].dataset_path))
    return OfflineRLDatasetInventory(
        root_path=str(root),
        datasets=[summary for _modified_at, summary in candidates],
    )


_LEROBOT_DELETE_EPISODES_SCRIPT = r"""
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

source = Path(sys.argv[1])
output = Path(sys.argv[2])
indices = json.loads(sys.argv[3])
expected = int(sys.argv[4])
repo_id = f"cyclo/{source.name}"
dataset = LeRobotDataset(repo_id=repo_id, root=source)
edited = delete_episodes(
    dataset,
    episode_indices=indices,
    output_dir=output,
    repo_id=repo_id,
)
if edited.meta.total_episodes != expected:
    raise RuntimeError(
        f"episode count mismatch: {edited.meta.total_episodes} != {expected}"
    )
print(json.dumps({"total_episodes": edited.meta.total_episodes}))
"""


def _offline_rl_run_lerobot_episode_delete(
    dataset: Path,
    output: Path,
    episode_indices: List[int],
    expected_episodes: int,
) -> None:
    spec = _require_known_backend("lerobot")
    container = _assert_backend_container_running("lerobot", spec)
    try:
        result = container.exec_run(
            [
                "/lerobot/.venv/bin/python",
                "-c",
                _LEROBOT_DELETE_EPISODES_SCRIPT,
                str(dataset),
                str(output),
                json.dumps(episode_indices),
                str(expected_episodes),
            ],
            user="1000:1000",
            environment={
                "HOME": "/tmp",
                "HF_HOME": "/tmp/cyclo_lerobot_dataset_edit/huggingface",
                "HF_DATASETS_CACHE": "/tmp/cyclo_lerobot_dataset_edit/datasets",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "XDG_CACHE_HOME": "/tmp/cyclo_lerobot_dataset_edit/cache",
            },
        )
    except DockerException as exc:
        raise HTTPException(503, f"LeRobot dataset editor failed to start: {exc}") from exc
    exit_code = getattr(result, "exit_code", None)
    output_bytes = getattr(result, "output", b"")
    if exit_code is None and isinstance(result, tuple) and len(result) == 2:
        exit_code, output_bytes = result
    if exit_code != 0:
        if isinstance(output_bytes, bytes):
            detail = output_bytes.decode(errors="replace")
        else:
            detail = str(output_bytes or "")
        detail = detail.strip()[-2000:]
        raise HTTPException(
            500,
            "LeRobot could not rebuild the dataset"
            + (f": {detail}" if detail else ""),
        )


def _offline_rl_v21_episode_path(
    dataset: Path,
    info: dict,
    *,
    template_name: str,
    episode_index: int,
    video_key: str = "",
) -> Path:
    """Resolve one canonical v2.1 episode artifact below ``dataset``."""
    defaults = {
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
    }
    template = info.get(template_name) or defaults.get(template_name)
    chunks_size = info.get("chunks_size", 1000)
    if (
        not isinstance(template, str)
        or not template.strip()
        or isinstance(chunks_size, bool)
        or not isinstance(chunks_size, int)
        or chunks_size < 1
    ):
        raise HTTPException(400, f"Invalid LeRobot v2.1 {template_name}")
    try:
        relative = Path(template.format(
            episode_chunk=episode_index // chunks_size,
            chunk_index=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=video_key,
        ))
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            400,
            f"Invalid LeRobot v2.1 {template_name} template",
        ) from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(400, f"Unsafe LeRobot v2.1 {template_name}")
    return dataset / relative


def _offline_rl_replace_parquet_column(table, name: str, values):
    """Replace a required Arrow column while preserving its declared type."""
    try:
        import pyarrow as arrow  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for dataset editing") from exc
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        raise HTTPException(400, f"LeRobot episode parquet is missing {name}")
    field = table.schema.field(field_index)
    try:
        return table.set_column(field_index, field, arrow.array(values, type=field.type))
    except Exception as exc:  # noqa: BLE001 - Arrow schema validation boundary
        raise HTTPException(400, f"Invalid LeRobot {name} column") from exc


def _offline_rl_v21_stat_value(stats: dict, key: str, value) -> None:
    """Update one scalar legacy statistic, retaining its list representation."""
    current = stats.get(key)
    if not isinstance(current, dict):
        return
    for statistic in ("min", "max", "mean"):
        if statistic in current:
            current[statistic] = [value]


def _offline_rl_rebuild_v21_dataset(
    source: Path,
    destination: Path,
    info: dict,
    episode_mapping: Dict[int, int],
) -> None:
    """Transactionally stage a compacted LeRobot v2.1 dataset.

    Legacy v2.1 stores one parquet and one video per episode, so deletion can
    preserve encoded media and rewrite only the two identity columns.  The
    caller swaps this staged directory into place only after full validation.
    """
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for dataset editing") from exc

    total_episodes = int(info.get("total_episodes", 0))
    splits = info.get("splits")
    if not isinstance(splits, dict) or splits != {"train": f"0:{total_episodes}"}:
        raise HTTPException(
            400,
            "LeRobot v2.1 episode deletion requires one contiguous train split",
        )
    metadata_rows = _offline_rl_v21_jsonl(
        source / "meta" / "episodes.jsonl",
        dataset=source,
        label="LeRobot v2.1 episode metadata",
    )
    metadata_by_index = {int(row["episode_index"]): row for row in metadata_rows}
    stats_path = source / "meta" / "episodes_stats.jsonl"
    stats_rows = (
        _offline_rl_v21_jsonl(
            stats_path,
            dataset=source,
            label="LeRobot v2.1 episode statistics",
        )
        if stats_path.is_file() and not stats_path.is_symlink()
        else []
    )
    stats_by_index = {int(row["episode_index"]): row for row in stats_rows}
    if len(metadata_by_index) != total_episodes or (
        stats_rows and len(stats_by_index) != total_episodes
    ):
        raise HTTPException(400, "Incomplete LeRobot v2.1 episode metadata")

    features = info.get("features")
    video_keys = sorted(
        key for key, value in (features.items() if isinstance(features, dict) else [])
        if isinstance(value, dict) and value.get("dtype") == "video"
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "meta").mkdir(parents=True, exist_ok=True)
    _offline_rl_safe_copy_file(
        source / "meta" / "tasks.jsonl",
        destination / "meta" / "tasks.jsonl",
        source,
    )

    output_metadata: List[dict] = []
    output_stats: List[dict] = []
    global_frame_index = 0
    for old_index, new_index in sorted(episode_mapping.items(), key=lambda item: item[1]):
        metadata = json.loads(json.dumps(metadata_by_index[old_index]))
        try:
            expected_frames = int(metadata["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"Invalid LeRobot v2.1 episode {old_index}") from exc
        source_data = _offline_rl_v21_episode_path(
            source,
            info,
            template_name="data_path",
            episode_index=old_index,
        )
        if source_data.is_symlink() or not source_data.is_file():
            raise HTTPException(400, f"Missing LeRobot v2.1 episode data {old_index}")
        try:
            source_data.resolve(strict=True).relative_to(source)
            table = parquet.read_table(source_data)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"Unsafe LeRobot v2.1 episode data {old_index}") from exc
        except Exception as exc:  # noqa: BLE001 - Parquet validation boundary
            raise HTTPException(400, f"Invalid LeRobot v2.1 episode data {old_index}") from exc
        if table.num_rows != expected_frames:
            raise HTTPException(400, f"LeRobot v2.1 episode {old_index} length mismatch")
        if table.schema.get_field_index("episode_index") < 0:
            raise HTTPException(400, f"Episode {old_index} is missing episode_index")
        old_episode_values = [int(value) for value in table.column("episode_index").to_pylist()]
        if set(old_episode_values) != {old_index}:
            raise HTTPException(400, f"Invalid episode_index coverage for episode {old_index}")
        table = _offline_rl_replace_parquet_column(
            table,
            "episode_index",
            [new_index] * expected_frames,
        )
        table = _offline_rl_replace_parquet_column(
            table,
            "index",
            range(global_frame_index, global_frame_index + expected_frames),
        )
        destination_data = _offline_rl_v21_episode_path(
            destination,
            info,
            template_name="data_path",
            episode_index=new_index,
        )
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, destination_data)

        for video_key in video_keys:
            source_video = _offline_rl_v21_episode_path(
                source,
                info,
                template_name="video_path",
                episode_index=old_index,
                video_key=video_key,
            )
            destination_video = _offline_rl_v21_episode_path(
                destination,
                info,
                template_name="video_path",
                episode_index=new_index,
                video_key=video_key,
            )
            if source_video.is_symlink() or not source_video.is_file():
                raise HTTPException(
                    400,
                    f"Missing LeRobot v2.1 video {video_key} episode {old_index}",
                )
            _offline_rl_safe_copy_file(source_video, destination_video, source)

        metadata["episode_index"] = new_index
        output_metadata.append(metadata)
        if stats_rows:
            stats_entry = json.loads(json.dumps(stats_by_index[old_index]))
            stats_entry["episode_index"] = new_index
            stats = stats_entry.get("stats")
            if isinstance(stats, dict):
                _offline_rl_v21_stat_value(stats, "episode_index", new_index)
                index_stats = stats.get("index")
                if isinstance(index_stats, dict):
                    _offline_rl_v21_stat_value(stats, "index", global_frame_index)
                    if "max" in index_stats:
                        index_stats["max"] = [global_frame_index + expected_frames - 1]
                    if "mean" in index_stats:
                        index_stats["mean"] = [
                            global_frame_index + (expected_frames - 1) / 2.0
                        ]
            output_stats.append(stats_entry)
        global_frame_index += expected_frames

    output_info = json.loads(json.dumps(info))
    output_info["total_episodes"] = len(episode_mapping)
    output_info["total_frames"] = global_frame_index
    output_info["total_chunks"] = max(
        1,
        (len(episode_mapping) + int(info.get("chunks_size", 1000)) - 1)
        // int(info.get("chunks_size", 1000)),
    )
    output_info["total_videos"] = len(video_keys) * len(episode_mapping)
    output_info["splits"] = {"train": f"0:{len(episode_mapping)}"}
    (destination / "meta" / "info.json").write_text(
        json.dumps(output_info, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_metadata),
        encoding="utf-8",
    )
    if stats_rows:
        (destination / "meta" / "episodes_stats.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_stats),
            encoding="utf-8",
        )

    # Validate every staged frame identity and media artifact before the caller
    # is allowed to replace the original directory.
    expected_global_index = 0
    for episode in range(len(episode_mapping)):
        path = _offline_rl_v21_episode_path(
            destination,
            output_info,
            template_name="data_path",
            episode_index=episode,
        )
        table = parquet.read_table(path, columns=["episode_index", "index"])
        frame_count = table.num_rows
        if set(int(value) for value in table.column("episode_index").to_pylist()) != {episode}:
            raise HTTPException(500, "Edited v2.1 dataset failed episode validation")
        indices = [int(value) for value in table.column("index").to_pylist()]
        if indices != list(range(expected_global_index, expected_global_index + frame_count)):
            raise HTTPException(500, "Edited v2.1 dataset failed frame-index validation")
        for video_key in video_keys:
            video = _offline_rl_v21_episode_path(
                destination,
                output_info,
                template_name="video_path",
                episode_index=episode,
                video_key=video_key,
            )
            if not video.is_file() or video.stat().st_size < 1:
                raise HTTPException(500, "Edited v2.1 dataset failed video validation")
        expected_global_index += frame_count
    if expected_global_index != global_frame_index:
        raise HTTPException(500, "Edited v2.1 dataset failed frame-count validation")


def _offline_rl_safe_copy_file(source: Path, destination: Path, root: Path) -> None:
    if source.is_symlink() or not source.is_file():
        return
    try:
        source.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Unsafe Cyclo dataset metadata: {source}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _offline_rl_preserve_annotations(
    source: Path,
    destination: Path,
    info: dict,
    episode_mapping: Dict[int, int],
) -> None:
    template = info.get("annotation_path")
    if not isinstance(template, str) or not template.strip():
        return
    chunks_size = info.get("chunks_size", 1000)
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size < 1:
        raise HTTPException(400, "Invalid LeRobot chunks_size")
    for old_index, new_index in episode_mapping.items():
        try:
            old_relative = Path(template.format(
                episode_chunk=old_index // chunks_size,
                chunk_index=old_index // chunks_size,
                episode_index=old_index,
            ))
            new_relative = Path(template.format(
                episode_chunk=new_index // chunks_size,
                chunk_index=new_index // chunks_size,
                episode_index=new_index,
            ))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, "Invalid LeRobot annotation_path template") from exc
        if old_relative.is_absolute() or ".." in old_relative.parts:
            raise HTTPException(400, "Unsafe LeRobot annotation_path")
        _offline_rl_safe_copy_file(
            source / old_relative,
            destination / new_relative,
            source,
        )


def _offline_rl_preserve_frame_reuse(
    source: Path,
    destination: Path,
    episode_mapping: Dict[int, int],
) -> None:
    source_file = source / "meta" / "frame_reuse.parquet"
    if not source_file.exists():
        return
    if source_file.is_symlink():
        raise HTTPException(400, "Unsafe frame_reuse metadata")
    try:
        import pyarrow as arrow  # type: ignore[import-not-found]
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for metadata preservation") from exc
    try:
        table = parquet.read_table(source_file)
        if "episode_index" not in table.schema.names:
            raise ValueError("episode_index column is missing")
        values = table.column("episode_index").to_pylist()
        positions = [i for i, value in enumerate(values) if int(value) in episode_mapping]
        filtered = table.take(arrow.array(positions, type=arrow.int64()))
        field_index = filtered.schema.get_field_index("episode_index")
        field = filtered.schema.field(field_index)
        mapped = [episode_mapping[int(values[i])] for i in positions]
        filtered = filtered.set_column(
            field_index,
            field,
            arrow.array(mapped, type=field.type),
        )
        output_file = destination / "meta" / "frame_reuse.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(filtered, output_file)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
        raise HTTPException(400, "Could not preserve frame_reuse metadata") from exc


def _offline_rl_preserve_episode_extras(
    source: Path,
    destination: Path,
    episode_mapping: Dict[int, int],
) -> None:
    """Carry Cyclo-only episode columns through LeRobot's standard rebuild."""
    try:
        import pyarrow as arrow  # type: ignore[import-not-found]
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - production images pin pyarrow
        raise HTTPException(503, "PyArrow is unavailable for metadata preservation") from exc
    source_files = sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    output_files = sorted((destination / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not source_files or not output_files:
        return
    try:
        source_tables = [parquet.read_table(path) for path in source_files]
        source_table = arrow.concat_tables(source_tables)
        source_rows = {
            int(row["episode_index"]): row for row in source_table.to_pylist()
        }
        inverse_mapping = {new: old for old, new in episode_mapping.items()}
        standard_names = set(parquet.ParquetFile(output_files[0]).schema_arrow.names)
        extra_fields = [
            field for field in source_table.schema
            if field.name not in standard_names
        ]
        if not extra_fields:
            return
        for output_file in output_files:
            table = parquet.read_table(output_file)
            new_indices = [int(value) for value in table.column("episode_index").to_pylist()]
            for field in extra_fields:
                values = [
                    source_rows[inverse_mapping[index]].get(field.name)
                    for index in new_indices
                ]
                table = table.append_column(
                    field,
                    arrow.array(values, type=field.type),
                )
            temporary = output_file.with_name(f".{output_file.name}.{uuid.uuid4().hex}.tmp")
            try:
                parquet.write_table(table, temporary)
                os.replace(temporary, output_file)
            finally:
                temporary.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - PyArrow validation boundary
        raise HTTPException(400, "Could not preserve Cyclo episode metadata") from exc


def _offline_rl_preserve_info_extras(
    destination: Path,
    source_info: dict,
) -> None:
    info_path = destination / "meta" / "info.json"
    output_info = _offline_rl_json_file(info_path, label="rebuilt LeRobot metadata")
    standard_keys = {
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "total_chunks",
        "chunks_size",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
        "fps",
        "splits",
        "data_path",
        "video_path",
        "features",
    }
    for key, value in source_info.items():
        if key not in standard_keys:
            output_info[key] = value
    temporary = info_path.with_name(f".{info_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(output_info, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, info_path)
    finally:
        temporary.unlink(missing_ok=True)


def _offline_rl_preserve_cyclo_metadata(
    source: Path,
    destination: Path,
    info: dict,
    episode_mapping: Dict[int, int],
) -> None:
    for relative in (Path("README.md"), Path("info.json"), Path("meta/subtasks.parquet")):
        _offline_rl_safe_copy_file(source / relative, destination / relative, source)
    _offline_rl_preserve_info_extras(destination, info)
    _offline_rl_preserve_annotations(source, destination, info, episode_mapping)
    _offline_rl_preserve_frame_reuse(source, destination, episode_mapping)
    _offline_rl_preserve_episode_extras(source, destination, episode_mapping)


def _offline_rl_delete_dataset_episodes(
    raw_path: str,
    requested_indices: List[int],
) -> Optional[OfflineRLDatasetSummary]:
    global _OFFLINE_RL_DATASET_EDIT_ACTIVE

    indices = sorted(set(requested_indices))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in requested_indices):
        raise HTTPException(400, "episode_indices must contain non-negative integers")
    if len(indices) != len(requested_indices):
        raise HTTPException(400, "episode_indices must not contain duplicates")

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "Another LeRobot dataset edit is already running")
        _OFFLINE_RL_DATASET_EDIT_ACTIVE = True

    temporary: Optional[Path] = None
    backup: Optional[Path] = None
    try:
        with _OFFLINE_RL_LOCK:
            if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
                raise HTTPException(409, "Cannot edit a dataset while offline RL is running")
        with _ACT_TD3_CRITIC_WARMUP_LOCK:
            if (
                _ACT_TD3_CRITIC_WARMUP_JOB is not None
                and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
            ):
                raise HTTPException(
                    409,
                    "Cannot edit a dataset while ACT-TD3 critic warm-up is running",
                )
        with _OfflineRLDatasetOperationLock():
            dataset, info = _offline_rl_dataset_metadata(raw_path)
            version = info.get("codebase_version")
            if version not in {"v2.1", "v3.0"}:
                raise HTTPException(
                    400,
                    "Episode deletion requires a LeRobot v2.1 or v3.0 dataset",
                )
            before = _offline_rl_dataset_summary(raw_path)
            invalid = [index for index in indices if index >= before.total_episodes]
            if invalid:
                raise HTTPException(400, f"Invalid episode indices: {invalid}")

            # LeRobot does not define a useful zero-episode dataset contract.
            # Deleting the final episode therefore removes the selected dataset
            # as one atomic directory rename, rather than leaving invalid metadata.
            if len(indices) == before.total_episodes:
                token = uuid.uuid4().hex
                trash = dataset.parent / f".{dataset.name}.delete-all-{token}.trash"
                if trash.exists():
                    raise HTTPException(409, "Dataset deletion staging path already exists")
                os.replace(dataset, trash)
                try:
                    shutil.rmtree(trash)
                except OSError as exc:
                    # The public dataset path has already disappeared atomically.
                    # Keep any hidden remainder out of inventory and report it for
                    # administrator cleanup instead of exposing a partial dataset.
                    logger.warning(
                        "dataset was removed but its hidden deletion staging path "
                        "could not be fully purged (%s): %s",
                        trash,
                        exc,
                    )
                return None

            kept = [
                index for index in range(before.total_episodes)
                if index not in set(indices)
            ]
            episode_mapping = {
                old_index: new_index for new_index, old_index in enumerate(kept)
            }
            token = uuid.uuid4().hex
            temporary = dataset.parent / f".{dataset.name}.delete-{token}.tmp"
            backup = dataset.parent / f".{dataset.name}.delete-{token}.backup"
            if temporary.exists() or backup.exists():
                raise HTTPException(409, "Dataset edit staging path already exists")

            if version == "v3.0":
                _offline_rl_run_lerobot_episode_delete(
                    dataset,
                    temporary,
                    indices,
                    len(kept),
                )
            else:
                _offline_rl_rebuild_v21_dataset(
                    dataset,
                    temporary,
                    info,
                    episode_mapping,
                )
            _offline_rl_preserve_cyclo_metadata(
                dataset,
                temporary,
                info,
                episode_mapping,
            )
            staged = _offline_rl_dataset_summary(str(temporary))
            if staged.total_episodes != len(kept):
                raise HTTPException(500, "Edited dataset failed episode-count validation")

            os.replace(dataset, backup)
            try:
                os.replace(temporary, dataset)
            except Exception:
                os.replace(backup, dataset)
                raise
            temporary = None
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                logger.warning("could not remove dataset edit backup %s: %s", backup, exc)
            backup = None
            return _offline_rl_dataset_summary(str(dataset))
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists():
            logger.error("dataset edit rollback backup retained at %s", backup)
        with _OFFLINE_RL_DATASET_EDIT_LOCK:
            _OFFLINE_RL_DATASET_EDIT_ACTIVE = False


def _lerobot_v3_dataset(
    raw_path: str,
    *,
    require_success_labels: bool,
) -> tuple[Path, int, bool]:
    """Validate one immutable LeRobot v3 root.

    Episode outcome labels are an RL data contract, not a behavior-cloning
    requirement.  The caller therefore chooses whether the feature is
    mandatory while all path, version, and episode-count checks stay shared.
    """

    dataset = _offline_rl_input_path(
        raw_path,
        root=_OFFLINE_RL_DATASET_ROOT,
        label="dataset_path",
        expect_directory=True,
    )
    try:
        info_path = _offline_rl_input_path(
            str(dataset / "meta" / "info.json"),
            root=dataset,
            label="LeRobot meta/info.json",
            expect_directory=False,
        )
    except HTTPException as exc:
        raise HTTPException(400, "dataset_path is missing a safe meta/info.json") from exc
    info = _offline_rl_json_file(info_path, label="LeRobot dataset metadata")
    if info.get("codebase_version") != "v3.0":
        raise HTTPException(400, "Offline RL requires a LeRobot v3.0 dataset")
    episodes = info.get("total_episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise HTTPException(400, "LeRobot total_episodes must be a positive integer")
    if episodes > _OFFLINE_RL_MAX_EPISODES:
        raise HTTPException(
            400,
            f"Offline RL supports at most {_OFFLINE_RL_MAX_EPISODES} episodes",
        )
    features = info.get("features")
    has_success_labels = (
        isinstance(features, dict) and "episode_success" in features
    )
    if require_success_labels and not has_success_labels:
        raise HTTPException(400, "LeRobot dataset is missing episode_success labels")
    return dataset, episodes, has_success_labels


def _offline_rl_dataset(raw_path: str) -> tuple[Path, int]:
    """Validate the stricter labeled-dataset contract required by ACT-TD3."""

    dataset, episodes, _has_success_labels = _lerobot_v3_dataset(
        raw_path,
        require_success_labels=True,
    )
    return dataset, episodes


def _offline_rl_requested_dataset_paths(
    request: (
        OfflineRLStartRequest
        | ACTTD3CriticWarmupStartRequest
        | ImitationLearningStartRequest
    ),
) -> List[str]:
    """Normalize the new ordered-list request without breaking scalar clients."""
    legacy = request.dataset_path.strip()
    requested = [value.strip() for value in request.dataset_paths]
    if any(not value for value in requested):
        raise HTTPException(400, "dataset_paths must not contain empty paths")
    if requested:
        if legacy and legacy != requested[0]:
            raise HTTPException(
                400,
                "dataset_path must match the first ordered dataset_paths entry",
            )
        return requested
    if legacy:
        return [legacy]
    raise HTTPException(400, "dataset_path or dataset_paths is required")


def _offline_rl_datasets(
    raw_paths: List[str],
) -> tuple[List[Path], int, int, int]:
    """Validate immutable v3 roots and aggregate the virtual replay contract."""
    datasets: List[Path] = []
    seen: set[Path] = set()
    episode_count = 0
    success_count = 0
    failure_count = 0
    unlabeled_count = 0
    for raw_path in raw_paths:
        dataset, episodes = _offline_rl_dataset(raw_path)
        if dataset in seen:
            raise HTTPException(400, "dataset_paths must not contain duplicates")
        seen.add(dataset)
        summary = _offline_rl_dataset_summary(str(dataset))
        if summary.version != "v3.0" or summary.total_episodes != episodes:
            raise HTTPException(400, "LeRobot dataset summary disagrees with metadata")
        datasets.append(dataset)
        episode_count += episodes
        success_count += summary.success_count
        failure_count += summary.failure_count
        unlabeled_count += summary.unlabeled_count
    if episode_count > _OFFLINE_RL_MAX_EPISODES:
        raise HTTPException(
            400,
            f"Offline RL supports at most {_OFFLINE_RL_MAX_EPISODES} episodes "
            "across dataset_paths",
        )
    if unlabeled_count:
        raise HTTPException(
            400,
            "ACT-TD3 dataset_paths contain unlabeled episodes",
        )
    if success_count < 1 or failure_count < 1:
        raise HTTPException(
            400,
            "ACT-TD3 dataset_paths require at least one success and one failure in aggregate",
        )
    return datasets, episode_count, success_count, failure_count


def _offline_rl_act_checkpoint(raw_path: str) -> Path:
    checkpoint = _offline_rl_input_path(
        raw_path,
        root=_OFFLINE_RL_MODEL_ROOT,
        label="act_checkpoint",
        expect_directory=True,
    )
    nested = checkpoint / "pretrained_model"
    if not (checkpoint / "config.json").is_file() and nested.is_dir():
        checkpoint = _offline_rl_input_path(
            str(nested),
            root=_OFFLINE_RL_MODEL_ROOT,
            label="act_checkpoint",
            expect_directory=True,
        )
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise HTTPException(
            400,
            "act_checkpoint is incomplete; missing " + ", ".join(missing),
        )
    for name in required:
        _offline_rl_input_path(
            str(checkpoint / name),
            root=checkpoint,
            label=f"act_checkpoint/{name}",
            expect_directory=False,
        )
    config = _offline_rl_json_file(
        checkpoint / "config.json",
        label="ACT policy config",
    )
    if config.get("type") != "act":
        raise HTTPException(400, "act_checkpoint config type must be 'act'")
    return checkpoint


def _offline_rl_schedule(
    critic_epochs: int,
    actor_equivalent_epochs: int,
) -> tuple[int, int]:
    for label, value in (
        ("critic_epochs", critic_epochs),
        ("actor_equivalent_epochs", actor_equivalent_epochs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise HTTPException(400, f"{label} must be a positive integer")
    if (
        critic_epochs < actor_equivalent_epochs
        or critic_epochs % actor_equivalent_epochs != 0
    ):
        raise HTTPException(
            400,
            "TD3 requires critic_epochs to be an exact integer multiple of "
            "actor_equivalent_epochs; 1:1 is supported",
        )
    return critic_epochs, actor_equivalent_epochs


def _act_trainable_groups(
    groups: List[str],
    *,
    field_name: str,
) -> List[str]:
    """Validate one ACT trainability contract and return canonical order."""
    if not isinstance(groups, list) or not groups:
        raise HTTPException(
            400,
            f"{field_name} must select at least one group; "
            "all-frozen actor training is not supported",
        )

    normalized: List[str] = []
    for value in groups:
        if not isinstance(value, str):
            raise HTTPException(400, f"{field_name} must contain strings")
        group = value.strip()
        if not group:
            raise HTTPException(400, f"{field_name} must not contain empty names")
        normalized.append(group)

    if len(set(normalized)) != len(normalized):
        raise HTTPException(400, f"{field_name} must not contain duplicates")

    unknown = sorted(set(normalized) - set(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS))
    if unknown:
        raise HTTPException(
            400,
            f"Unknown {field_name}: " + ", ".join(unknown),
        )

    ordered = [
        group for group in _OFFLINE_RL_ACTOR_TRAINABLE_GROUPS
        if group in normalized
    ]
    if ordered == ["cvae_encoder"]:
        raise HTTPException(
            400,
            f"CVAE-only {field_name} is not supported because it does not "
            "update the deployed action path",
        )
    return ordered


def _offline_rl_actor_trainable_groups(groups: List[str]) -> List[str]:
    """Validate one TD3 actor trainability contract and return canonical order."""
    return _act_trainable_groups(
        groups,
        field_name="actor_trainable_groups",
    )


def _offline_rl_objective_trainable_groups(
    actor_objective: str,
    groups: List[str],
) -> List[str]:
    """Return the effective ACT actor mask for one objective.

    Pure TD3 has no CVAE/behavior-cloning term, so its posterior encoder is
    outside the deployed deterministic action path and must stay frozen.
    """

    normalized = _offline_rl_actor_trainable_groups(groups)
    if actor_objective == "td3":
        normalized = [group for group in normalized if group != "cvae_encoder"]
        if not normalized:
            raise HTTPException(
                400,
                "Pure TD3 requires at least one trainable deterministic ACT block",
            )
    return normalized


def _offline_rl_algorithm_contract(
    request: OfflineRLStartRequest,
) -> tuple[str, str]:
    """Normalize the TD3 family and actor loss without changing intent.

    For backward compatibility, the short-lived API that encoded the actor
    objective in ``algorithm`` is accepted when that mapping is unambiguous.
    New clients send both fields.  An omitted ``actor_objective`` together
    with an explicitly supplied legacy ``algorithm='td3'`` therefore means
    pure TD3; when both fields are omitted the documented TD3+BC default is
    retained.
    """

    algorithm = request.algorithm.strip().lower()
    actor_objective = request.actor_objective.strip().lower()
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(request, "__fields_set__", set())
    objective_is_explicit = "actor_objective" in fields_set
    algorithm_is_explicit = "algorithm" in fields_set

    if algorithm == "td3_bc":
        if objective_is_explicit and actor_objective != "td3_bc":
            raise HTTPException(
                400,
                "Legacy algorithm='td3_bc' conflicts with actor_objective",
            )
        return "td3", "td3_bc"
    if algorithm != "td3":
        raise HTTPException(
            400,
            f"Offline RL algorithm '{request.algorithm}' is not implemented; "
            "available algorithm is TD3",
        )
    if not objective_is_explicit and algorithm_is_explicit:
        # Older clients used algorithm='td3' to request the pure actor loss.
        actor_objective = "td3"
    if actor_objective not in _OFFLINE_RL_ACTOR_OBJECTIVES:
        raise HTTPException(400, "Unknown TD3 actor loss option")
    return "td3", actor_objective


def _imitation_learning_trainable_groups(
    groups: Optional[List[str]],
) -> List[str]:
    """Resolve the default ACT-BC mask and reject unsafe empty selections."""
    return _act_trainable_groups(
        list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS) if groups is None else groups,
        field_name="trainable_groups",
    )


def _offline_rl_parent_checkpoint(
    raw_path: str,
    episode_count: int,
    actor_trainable_groups: Optional[List[str]] = None,
    batch_size: int = 4,
    actor_objective: Optional[str] = None,
) -> tuple[Path | None, int, int]:
    value = (raw_path or "").strip()
    if not value:
        # The initial replay may contain both the IL seed data and the first
        # policy-collected Data Epoch.  It is bounded by the cumulative replay
        # limit, not by the per-round append limit.  Once a completed TD3
        # parent exists, only the newly appended 1..50 episodes are accepted
        # below.
        if not 1 <= episode_count <= _OFFLINE_RL_MAX_EPISODES:
            raise HTTPException(
                400,
                "The first ACT-TD3 round may contain 1..200 episodes",
            )
        return None, 0, 0

    parent = _offline_rl_input_path(
        value,
        root=_OFFLINE_RL_MODEL_ROOT,
        label="parent_checkpoint",
        expect_directory=False,
    )
    if parent.name != "act_td3.pt" or parent.parent.name != "training_state":
        raise HTTPException(
            400,
            "parent_checkpoint must be a training_state/act_td3.pt file",
        )
    try:
        manifest_path = _offline_rl_input_path(
            str(parent.parent.parent / "training_manifest.json"),
            root=parent.parent.parent,
            label="parent training_manifest.json",
            expect_directory=False,
        )
    except HTTPException as exc:
        raise HTTPException(
            400,
            "parent_checkpoint is missing a safe training_manifest.json",
        ) from exc
    manifest = _offline_rl_json_file(
        manifest_path,
        label="parent training manifest",
    )
    previous_episodes = manifest.get("episode_count")
    if (
        manifest.get("event") != "result"
        or manifest.get("status") != "complete"
        or isinstance(previous_episodes, bool)
        or not isinstance(previous_episodes, int)
        or not 1 <= previous_episodes < _OFFLINE_RL_MAX_EPISODES
    ):
        raise HTTPException(400, "parent_checkpoint is not a completed ACT-TD3 round")
    added = episode_count - previous_episodes
    if not 1 <= added <= _OFFLINE_RL_MAX_NEW_EPISODES:
        raise HTTPException(400, "Each ACT-TD3 round must add 1..50 episodes")

    parent_schedule = manifest.get("schedule")
    if parent_schedule is None:
        pass
    elif isinstance(parent_schedule, dict):
        try:
            _offline_rl_schedule(
                parent_schedule.get("critic_epochs"),
                parent_schedule.get("actor_equivalent_epochs"),
            )
        except HTTPException as exc:
            raise HTTPException(400, "parent training schedule is invalid") from exc
    else:
        raise HTTPException(400, "parent training schedule is invalid")

    parent_objective = manifest.get("actor_objective")
    if actor_objective is None:
        # Internal compatibility path for callers that only validate the
        # historical schedule/data contract. The public start route always
        # passes an explicit objective.
        pass
    elif parent_objective is None:
        # The legacy hybrid cloned failed behavior too, so it is not exactly
        # either selectable objective. A new explicit lineage is required.
        raise HTTPException(
            400,
            "Legacy parent checkpoints without actor_objective cannot resume; "
            "start a fresh TD3 or TD3+BC lineage",
        )
    elif parent_objective not in _OFFLINE_RL_ACTOR_OBJECTIVES:
        raise HTTPException(400, "parent actor_objective is invalid")
    elif parent_objective != actor_objective:
        raise HTTPException(
            400,
            "parent actor_objective does not match the requested loss option",
        )
    if actor_objective is not None:
        parent_algorithm = manifest.get("algorithm")
        legacy_algorithm = {
            "td3": "ACT-TD3 cumulative replay",
            "td3_bc": "ACT-TD3+BC cumulative replay",
        }[actor_objective]
        if parent_algorithm not in (None, "td3", legacy_algorithm):
            raise HTTPException(
                400,
                "parent algorithm is not a compatible TD3 artifact",
            )

    parent_trainable_groups = manifest.get("actor_trainable_groups")
    if parent_trainable_groups is not None:
        try:
            normalized_parent_groups = _offline_rl_actor_trainable_groups(
                parent_trainable_groups
            )
            normalized_requested_groups = (
                _offline_rl_actor_trainable_groups(actor_trainable_groups)
                if actor_trainable_groups is not None
                else None
            )
        except HTTPException as exc:
            raise HTTPException(
                400,
                "parent actor trainability contract is invalid",
            ) from exc
        if (
            normalized_requested_groups is not None
            and normalized_parent_groups != normalized_requested_groups
        ):
            raise HTTPException(
                400,
                "parent actor_trainable_groups do not match the requested contract",
            )
    elif actor_trainable_groups is not None:
        try:
            normalized_requested_groups = _offline_rl_actor_trainable_groups(
                actor_trainable_groups
            )
        except HTTPException as exc:
            raise HTTPException(
                400,
                "requested actor trainability contract is invalid",
            ) from exc
        if normalized_requested_groups != list(_OFFLINE_RL_ACTOR_TRAINABLE_GROUPS):
            raise HTTPException(
                400,
                "Legacy parent checkpoints without actor_trainable_groups can "
                "resume only with all ACT actor trainable groups",
            )
    parent_batch_size = manifest.get("batch_size", 4)
    if (
        isinstance(parent_batch_size, bool)
        or not isinstance(parent_batch_size, int)
        or not 1 <= parent_batch_size <= 64
    ):
        raise HTTPException(400, "parent training batch_size is invalid")
    if parent_batch_size != batch_size:
        raise HTTPException(
            400,
            "parent training batch_size does not match the requested batch_size",
        )
    recorded_checkpoint = manifest.get("checkpoint_path")
    if recorded_checkpoint and os.path.normpath(str(recorded_checkpoint)) != str(parent):
        raise HTTPException(400, "parent training manifest checkpoint does not match")
    parent_round_index = manifest.get("round_index", 1)
    if (
        isinstance(parent_round_index, bool)
        or not isinstance(parent_round_index, int)
        or parent_round_index < 1
    ):
        raise HTTPException(400, "parent training round index is invalid")
    return parent, previous_episodes, parent_round_index


def _offline_rl_robot_config(robot_type: str) -> str:
    robot = _validate_robot_type(robot_type)
    source_root = Path(_CYCLO_REPO_MOUNT) / "shared" / "shared" / "robot_configs"
    source = source_root / f"{robot}_config.yaml"
    try:
        source.resolve(strict=True).relative_to(source_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Unknown robot_type for offline RL: {robot}") from exc
    if source.is_symlink() or not source.is_file():
        raise HTTPException(400, f"Unknown robot_type for offline RL: {robot}")
    return f"/orchestrator_config/{robot}_config.yaml"


def _offline_rl_output_path(
    job_id: str,
    episode_count: int,
    actor_objective: str = "td3_bc",
) -> Path:
    if actor_objective not in _OFFLINE_RL_ACTOR_OBJECTIVES:
        raise HTTPException(400, "Unknown ACT-TD3 actor objective")
    model_root = _OFFLINE_RL_MODEL_ROOT.resolve(strict=True)
    if _OFFLINE_RL_OUTPUT_ROOT.exists():
        if _OFFLINE_RL_OUTPUT_ROOT.is_symlink():
            raise HTTPException(500, "Offline RL output root must not be a symbolic link")
        try:
            _OFFLINE_RL_OUTPUT_ROOT.resolve(strict=True).relative_to(model_root)
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "Offline RL output root escapes the model root") from exc
    output = _OFFLINE_RL_OUTPUT_ROOT / (
        f"{actor_objective}_episodes_{episode_count:04d}_{job_id[:12]}"
    )
    if output.exists():
        raise HTTPException(409, f"Offline RL output already exists: {output}")
    return output


def _offline_rl_cancel_output(job: _OfflineRLJob) -> None:
    """Remove only this incomplete job's generated output directory.

    Cancellation is deliberately narrower than policy or dataset deletion.  A
    stopped round contains a resumable actor/critic/optimizer checkpoint, but
    none of it is deployable until the round completes.  The caller may choose
    to discard that one versioned directory while retaining the immutable
    replay, base ACT policy, completed parent round, and policy-local warm-up
    critic needed to retry the same round.
    """

    if job.actor_objective not in _OFFLINE_RL_ACTOR_OBJECTIVES:
        raise HTTPException(409, "Offline RL job has an invalid actor objective")

    expected_name = (
        f"{job.actor_objective}_episodes_{job.episode_count:04d}_{job.job_id[:12]}"
    )
    output_root = _OFFLINE_RL_OUTPUT_ROOT
    expected = output_root / expected_name
    configured = Path(job.output_dir)
    if os.path.normpath(str(configured)) != os.path.normpath(str(expected)):
        raise HTTPException(
            409,
            "Offline RL cancellation output does not match the current job",
        )

    if _OFFLINE_RL_MODEL_ROOT.is_symlink():
        raise HTTPException(500, "Offline RL model root must not be a symbolic link")
    try:
        model_root = _OFFLINE_RL_MODEL_ROOT.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(500, "Offline RL model root is unavailable") from exc
    if not model_root.is_dir() or model_root.is_symlink():
        raise HTTPException(500, "Offline RL model root must be a real directory")
    if output_root.is_symlink():
        raise HTTPException(409, "Offline RL output root must not be a symbolic link")
    if output_root.exists() and not output_root.is_dir():
        raise HTTPException(409, "Offline RL output root must be a directory")
    try:
        resolved_output_root = output_root.resolve(strict=output_root.exists())
        resolved_output_root.relative_to(model_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "Offline RL output root escapes the model root") from exc

    # ``Path.exists`` is false for a dangling symlink, so check the link before
    # accepting an absent output as an already-clean cancellation.
    if configured.is_symlink():
        raise HTTPException(
            409,
            "Offline RL cancellation output must not be a symbolic link",
        )
    if not configured.exists():
        return
    if not configured.is_dir():
        raise HTTPException(409, "Offline RL cancellation output must be a directory")
    try:
        resolved_output = configured.resolve(strict=True)
        if resolved_output.parent != resolved_output_root:
            raise ValueError("output is not a direct child of the output root")
    except (OSError, ValueError) as exc:
        raise HTTPException(
            409,
            "Offline RL cancellation output escapes the output root",
        ) from exc

    trash = resolved_output_root / (
        f".{expected_name}.cancel-{uuid.uuid4().hex}.trash"
    )
    if trash.exists() or trash.is_symlink():
        raise HTTPException(409, "Offline RL cancellation staging path already exists")
    try:
        os.replace(resolved_output, trash)
    except OSError as exc:
        raise HTTPException(500, f"Could not stage Offline RL cancellation: {exc}") from exc
    try:
        shutil.rmtree(trash)
    except OSError as exc:
        # The public output path disappeared atomically.  Keep any hidden
        # remainder outside model discovery and report it for admin cleanup.
        logger.warning(
            "Offline RL output was cancelled but hidden staging path %s could "
            "not be fully purged: %s",
            trash,
            exc,
        )


def _offline_rl_mark_cancelled(job: _OfflineRLJob) -> None:
    """Clear incomplete training telemetry while retaining the retry contract."""

    job.status = "cancelled"
    job.percentage = 0.0
    job.completed_epochs = 0
    job.completed_critic_updates = 0
    job.total_critic_updates = 0
    job.completed_actor_updates = 0
    job.total_actor_updates = 0
    job.critic_loss = None
    job.actor_loss = None
    job.loss_history.clear()
    job.rl_metric_history.clear()
    job.eta_seconds = None
    job.model_path = ""
    job.checkpoint_path = ""
    job.critic_source = ""
    job.critic_checkpoint = ""
    job.message = "ACT-TD3 training cancelled; incomplete output was discarded"
    job.process = None
    job.stop_requested = False
    job.stop_confirmed = False
    job.returncode = None


def _offline_rl_command(
    *,
    job: _OfflineRLJob,
    robot_type: str,
    robot_config: str,
) -> List[str]:
    command = _compose_base_cmd() + [
        "run",
        "--rm",
        "--no-deps",
        "--pull",
        "never",
        "--name",
        _offline_rl_container_name(job),
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp",
        # The LeRobot image is built for its normal root-run inference service
        # and therefore exports HF_HOME/HF_LEROBOT_HOME/TORCH_HOME below
        # /root/.cache.  Offline training deliberately runs as uid 1000 so it
        # must not inherit those root-only paths.  The job is offline and all
        # policy/dataset assets are local, so per-container writable caches are
        # sufficient and disappear together with the --rm training container.
        "--env",
        f"XDG_CACHE_HOME={_OFFLINE_RL_CACHE_ROOT}",
        "--env",
        f"HF_HOME={_OFFLINE_RL_CACHE_ROOT}/huggingface",
        "--env",
        f"HF_LEROBOT_HOME={_OFFLINE_RL_CACHE_ROOT}/huggingface/lerobot",
        "--env",
        f"TORCH_HOME={_OFFLINE_RL_CACHE_ROOT}/torch",
        "--env",
        f"TRITON_CACHE_DIR={_OFFLINE_RL_CACHE_ROOT}/triton",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--entrypoint",
        "/lerobot/.venv/bin/python",
        "lerobot",
        "-m",
        "cyclo_brain.algorithm.rl.act_td3.offline_training_cli",
    ]
    for dataset_path in job.dataset_paths:
        command.extend(["--dataset-root", dataset_path])
    command.extend([
        "--act-checkpoint",
        job.act_checkpoint,
        "--robot-config",
        robot_config,
        "--robot-type",
        robot_type,
        "--device",
        "cuda:0",
        "--seed",
        "17",
        "--sampling-seed",
        "19",
        "--batch-size",
        str(job.batch_size),
        "--actor-objective",
        job.actor_objective,
        "--critic-epochs",
        str(job.critic_epochs),
        "--actor-equivalent-epochs",
        str(job.actor_equivalent_epochs),
        "--output-dir",
        job.output_dir,
        "--checkpoint-interval",
        "100",
        "--progress-interval",
        "1",
    ])
    for group in job.actor_trainable_groups:
        command.extend(["--actor-trainable-group", group])
    if job.parent_checkpoint:
        command.extend(["--parent-checkpoint", job.parent_checkpoint])
    return command


def _offline_rl_container_name(job: _OfflineRLJob) -> str:
    return f"cyclo_offline_rl_{job.job_id[:12]}"


def _offline_rl_interrupt_job(job: _OfflineRLJob) -> bool:
    """Send SIGINT only to this job's container or compose subprocess."""
    docker_error: Optional[Exception] = None
    try:
        container = _docker_client().containers.get(_offline_rl_container_name(job))
        container.kill(signal="SIGINT")
        return True
    except NotFound:
        # ``docker compose run`` may still be creating the named container, or
        # ``--rm`` may already have removed it. Fall back to its exact wrapper
        # subprocess; Compose forwards SIGINT to the attached container.
        pass
    except DockerException as exc:
        docker_error = exc
        logger.warning(
            "Could not signal Offline RL container %s: %s",
            _offline_rl_container_name(job),
            exc,
        )

    process = job.process
    if process is None:
        if docker_error is not None:
            raise RuntimeError(str(docker_error)) from docker_error
        return False
    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is not None:
        return False
    try:
        process.send_signal(signal.SIGINT)
    except (OSError, ProcessLookupError) as exc:
        if callable(poll) and poll() is not None:
            return False
        if docker_error is not None:
            raise RuntimeError(f"{docker_error}; {exc}") from exc
        raise
    return True


def _offline_rl_append_log(job: _OfflineRLJob, line: str) -> None:
    if not line:
        return
    job.log_tail.append(line)
    del job.log_tail[:-_OFFLINE_RL_LOG_LINES]


def _offline_rl_finite_float(value) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _offline_rl_update_number(job: _OfflineRLJob, payload: dict, name: str) -> None:
    value = payload.get(name)
    current = getattr(job, name)
    if value is None:
        return
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(job, name, value)
        return
    finite_value = _offline_rl_finite_float(value)
    if finite_value is not None:
        setattr(job, name, finite_value)


def _offline_rl_update_loss_history(job: _OfflineRLJob, payload: dict) -> None:
    """Append or update one finite, monotonically ordered loss point."""

    step = payload.get("completed_critic_updates")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return

    def finite_loss(name: str) -> Optional[float]:
        return _offline_rl_finite_float(payload.get(name))

    critic_loss = finite_loss("critic_loss")
    actor_loss = finite_loss("actor_loss")
    if critic_loss is None and actor_loss is None:
        return

    existing_index: Optional[int] = None
    for index in range(len(job.loss_history) - 1, -1, -1):
        existing_step = job.loss_history[index].step
        if existing_step == step:
            existing_index = index
            break
        if existing_step < step:
            break

    if existing_index is not None:
        existing = job.loss_history[existing_index]
        job.loss_history[existing_index] = OfflineRLLossPoint(
            step=step,
            critic_loss=(
                critic_loss if critic_loss is not None else existing.critic_loss
            ),
            actor_loss=actor_loss if actor_loss is not None else existing.actor_loss,
        )
        return

    if job.loss_history and step < job.loss_history[-1].step:
        # Late output from an older update must not reorder the graph.
        return
    job.loss_history.append(OfflineRLLossPoint(
        step=step,
        critic_loss=critic_loss,
        actor_loss=actor_loss,
    ))
    del job.loss_history[:-_OFFLINE_RL_LOSS_HISTORY_POINTS]


def _offline_rl_update_metric_history(job: _OfflineRLJob, payload: dict) -> None:
    """Consume the authoritative finite, ordered replay-round telemetry."""

    if "rl_metric_history" not in payload:
        return
    raw_history = payload.get("rl_metric_history")
    if not isinstance(raw_history, list):
        raise ValueError("ACT-TD3 RL metric history must be a list")
    if len(raw_history) > _OFFLINE_RL_METRIC_HISTORY_POINTS:
        raise ValueError("ACT-TD3 RL metric history exceeds its bounded contract")

    points: list[OfflineRLRLMetricPoint] = []
    previous_epoch = 0
    expected_fields = {
        "rl_epoch",
        "actor_loss_mean",
        "critic_loss_mean",
        "replay_average_reward",
    }
    for raw_point in raw_history:
        if not isinstance(raw_point, dict) or set(raw_point) != expected_fields:
            raise ValueError("ACT-TD3 RL metric history point fields disagree")
        rl_epoch = raw_point.get("rl_epoch")
        if (
            isinstance(rl_epoch, bool)
            or not isinstance(rl_epoch, int)
            or rl_epoch < 1
            or rl_epoch <= previous_epoch
        ):
            raise ValueError(
                "ACT-TD3 RL metric history must be ordered and deduplicated"
            )

        values: dict[str, Optional[float]] = {}
        for name in (
            "actor_loss_mean",
            "critic_loss_mean",
            "replay_average_reward",
        ):
            raw_value = raw_point.get(name)
            if raw_value is None:
                values[name] = None
                continue
            finite_value = _offline_rl_finite_float(raw_value)
            if finite_value is None:
                raise ValueError("ACT-TD3 RL metric history contains a non-finite value")
            values[name] = finite_value
        reward = values["replay_average_reward"]
        if reward is not None and not 0.0 <= reward <= 1.0:
            raise ValueError("ACT-TD3 replay average reward is outside [0, 1]")
        points.append(
            OfflineRLRLMetricPoint(
                rl_epoch=rl_epoch,
                actor_loss_mean=values["actor_loss_mean"],
                critic_loss_mean=values["critic_loss_mean"],
                replay_average_reward=reward,
            )
        )
        previous_epoch = rl_epoch

    if (
        job.rl_metric_history
        and points
        and points[-1].rl_epoch < job.rl_metric_history[-1].rl_epoch
    ):
        # Ignore late progress output from an older RL epoch.
        return
    job.rl_metric_history = points


def _offline_rl_update_critic_source(job: _OfflineRLJob, payload: dict) -> None:
    """Consume one complete, self-consistent critic-initialization contract."""

    has_source = "critic_source" in payload
    has_checkpoint = "critic_checkpoint" in payload
    if not has_source and not has_checkpoint:
        return
    if has_source != has_checkpoint:
        raise ValueError("ACT-TD3 critic telemetry fields are incomplete")

    source = payload.get("critic_source")
    checkpoint = payload.get("critic_checkpoint")
    if source not in _OFFLINE_RL_CRITIC_SOURCES:
        raise ValueError("ACT-TD3 critic telemetry source is invalid")
    if source == "random":
        if checkpoint not in (None, ""):
            raise ValueError("Random ACT-TD3 critic must not name a checkpoint")
        job.critic_source = source
        job.critic_checkpoint = ""
        return
    if not isinstance(checkpoint, str) or not checkpoint or not os.path.isabs(checkpoint):
        raise ValueError("ACT-TD3 critic checkpoint must be an absolute path")

    normalized = os.path.normpath(checkpoint)
    if source == "parent_checkpoint":
        if not job.parent_checkpoint:
            raise ValueError("ACT-TD3 parent critic source has no configured parent")
        expected = os.path.normpath(job.parent_checkpoint)
    elif source == "policy_warmup":
        if not job.act_checkpoint:
            raise ValueError("ACT-TD3 warm critic source has no configured policy")
        expected = os.path.normpath(
            str(Path(job.act_checkpoint) / "critic" / "latest.pt")
        )
    else:
        if not job.checkpoint_path:
            raise ValueError("ACT-TD3 resume critic source has no round checkpoint")
        expected = os.path.normpath(job.checkpoint_path)
    if normalized != expected:
        raise ValueError("ACT-TD3 critic checkpoint disagrees with its source")
    job.critic_source = source
    job.critic_checkpoint = normalized


def _offline_rl_consume_event(job: _OfflineRLJob, payload: dict) -> bool:
    """Apply one trusted CLI JSON event; return True for a complete result."""
    event = payload.get("event")
    if event == "manifest":
        actor_objective = payload.get("actor_objective")
        if actor_objective != job.actor_objective:
            raise ValueError("ACT-TD3 actor objective telemetry disagrees")
        if payload.get("algorithm") != job.algorithm:
            raise ValueError("ACT-TD3 algorithm telemetry disagrees")
        dataset = payload.get("dataset") or {}
        round_info = payload.get("round") or {}
        schedule = payload.get("schedule") or {}
        if isinstance(dataset, dict):
            for source, target in (
                ("episodes", "episode_count"),
                ("successes", "success_count"),
                ("failures", "failure_count"),
            ):
                value = dataset.get(source)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    setattr(job, target, value)
        if isinstance(round_info, dict):
            value = round_info.get("index")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                job.round_index = value
            value = round_info.get("new_episodes")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                job.round_episode_count = value
            for source, target in (
                ("critic_updates", "total_critic_updates"),
                ("actor_updates", "total_actor_updates"),
            ):
                value = round_info.get(source)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    setattr(job, target, value)
        if isinstance(schedule, dict):
            for source, target in (
                ("critic_epochs", "critic_epochs"),
                ("actor_equivalent_epochs", "actor_equivalent_epochs"),
            ):
                value = schedule.get(source)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                    setattr(job, target, value)
        checkpoint = payload.get("checkpoint")
        if isinstance(checkpoint, str):
            job.checkpoint_path = checkpoint
        _offline_rl_update_critic_source(job, payload)
        job.message = "ACT-TD3 training is running"
        return False

    if event == "progress":
        for name in (
            "episode_count",
            "completed_epochs",
            "total_epochs",
            "completed_critic_updates",
            "total_critic_updates",
            "completed_actor_updates",
            "total_actor_updates",
            "critic_loss",
            "actor_loss",
            "eta_seconds",
        ):
            _offline_rl_update_number(job, payload, name)
        _offline_rl_update_loss_history(job, payload)
        _offline_rl_update_metric_history(job, payload)
        percentage = payload.get("percentage")
        finite_percentage = _offline_rl_finite_float(percentage)
        if finite_percentage is not None:
            job.percentage = max(0.0, min(100.0, finite_percentage))
        checkpoint = payload.get("checkpoint_path")
        if isinstance(checkpoint, str):
            job.checkpoint_path = checkpoint
        job.message = (
            "Exporting the trained ACT model"
            if payload.get("status") == "complete"
            else "ACT-TD3 training is running"
        )
        return False

    if event == "result":
        actor_objective = payload.get("actor_objective")
        if actor_objective != job.actor_objective:
            raise ValueError("ACT-TD3 actor objective result disagrees")
        if payload.get("algorithm") != job.algorithm:
            raise ValueError("ACT-TD3 algorithm result disagrees")
        _offline_rl_consume_event(job, {**payload, "event": "progress"})
        _offline_rl_update_critic_source(job, payload)
        if payload.get("status") == "stopped":
            job.stop_confirmed = True
        model_path = payload.get("model_path")
        if isinstance(model_path, str):
            job.model_path = model_path
        return payload.get("status") == "complete" and bool(job.model_path)

    if event == "error":
        error_type = payload.get("error_type") or "OfflineRLError"
        message = payload.get("message") or "ACT-TD3 training failed"
        job.message = f"{error_type}: {message}"
    return False


def _offline_rl_verified_model(job: _OfflineRLJob) -> bool:
    if job.critic_source not in _OFFLINE_RL_CRITIC_SOURCES:
        return False
    if (job.critic_source == "random") != (job.critic_checkpoint == ""):
        return False
    expected = Path(job.output_dir) / "pretrained_model"
    if os.path.normpath(job.model_path) != str(expected):
        return False
    try:
        model = _offline_rl_input_path(
            job.model_path,
            root=_OFFLINE_RL_MODEL_ROOT,
            label="model_path",
            expect_directory=True,
        )
    except HTTPException:
        return False
    for name in ("config.json", "model.safetensors"):
        try:
            _offline_rl_input_path(
                str(model / name),
                root=model,
                label=f"exported model {name}",
                expect_directory=False,
            )
        except HTTPException:
            return False
    return True


def _monitor_offline_rl_job(job: _OfflineRLJob) -> None:
    result_complete = False
    try:
        _OFFLINE_RL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "a", encoding="utf-8") as log:
            stdout = job.process.stdout if job.process is not None else None
            if stdout is not None:
                for raw_line in stdout:
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode(errors="replace")
                    line = raw_line.rstrip("\r\n")
                    log.write(line + "\n")
                    log.flush()
                    with _OFFLINE_RL_LOCK:
                        _offline_rl_append_log(job, line)
                        try:
                            payload = json.loads(line)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if isinstance(payload, dict):
                            result_complete = (
                                _offline_rl_consume_event(job, payload)
                                or result_complete
                            )
            returncode = job.process.wait() if job.process is not None else -1
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        logger.error("Offline RL monitor failed: %s", exc, exc_info=True)
        returncode = -1
        with _OFFLINE_RL_LOCK:
            job.message = f"Offline RL monitor failed: {exc}"

    with _OFFLINE_RL_LOCK:
        job.returncode = returncode
        if returncode == 0 and result_complete and _offline_rl_verified_model(job):
            job.status = "completed"
            job.percentage = 100.0
            job.message = "ACT-TD3 training completed"
        elif returncode == 0 and job.stop_requested and job.stop_confirmed:
            job.status = "stopped"
            job.model_path = ""
            job.eta_seconds = None
            job.message = "ACT-TD3 training stopped"
        else:
            job.status = "failed"
            if not job.message or job.message in (
                "Starting ACT-TD3 offline training",
                "ACT-TD3 training is running",
                "Exporting the trained ACT model",
            ):
                job.message = (
                    f"ACT-TD3 training exited with code {returncode}"
                    if returncode != 0
                    else "ACT-TD3 result or exported model verification is missing"
                )


def _offline_rl_status(job: Optional[_OfflineRLJob]) -> OfflineRLStatus:
    if job is None:
        return OfflineRLStatus(status="idle", message="No offline RL job has been started")
    return OfflineRLStatus(
        status=job.status,
        algorithm=job.algorithm,
        actor_objective=job.actor_objective,
        percentage=job.percentage,
        episode_count=job.episode_count,
        round_index=job.round_index,
        round_episode_count=job.round_episode_count,
        batch_size=job.batch_size,
        critic_epochs=job.critic_epochs,
        actor_equivalent_epochs=job.actor_equivalent_epochs,
        actor_trainable_groups=list(job.actor_trainable_groups),
        success_count=job.success_count,
        failure_count=job.failure_count,
        completed_epochs=job.completed_epochs,
        total_epochs=job.total_epochs,
        completed_critic_updates=job.completed_critic_updates,
        total_critic_updates=job.total_critic_updates,
        completed_actor_updates=job.completed_actor_updates,
        total_actor_updates=job.total_actor_updates,
        critic_loss=job.critic_loss,
        actor_loss=job.actor_loss,
        loss_history=list(job.loss_history),
        rl_metric_history=list(job.rl_metric_history),
        eta_seconds=job.eta_seconds,
        model_path=job.model_path if job.status == "completed" else "",
        checkpoint_path=job.checkpoint_path,
        critic_source=job.critic_source,
        critic_checkpoint=job.critic_checkpoint,
        message=job.message,
        job_id=job.job_id,
        dataset_path=job.dataset_path,
        dataset_paths=list(job.dataset_paths),
        act_checkpoint=job.act_checkpoint,
        parent_checkpoint=job.parent_checkpoint,
        output_dir=job.output_dir,
        returncode=job.returncode,
        log_tail=list(job.log_tail),
    )


# -- ACT-TD3 critic warm-up ---------------------------------------------------


def _act_td3_critic_warmup_paths(
    act_checkpoint: Path,
    job_id: str,
) -> tuple[Path, Path, Path]:
    """Return safe policy-local run, latest, and manifest artifact paths."""

    critic_dir = act_checkpoint / "critic"
    runs_dir = critic_dir / "runs"
    for directory, label in (
        (critic_dir, "critic directory"),
        (runs_dir, "critic runs directory"),
    ):
        if directory.is_symlink():
            raise HTTPException(400, f"ACT-TD3 {label} must not be a symbolic link")
        if directory.exists() and not directory.is_dir():
            raise HTTPException(400, f"ACT-TD3 {label} must be a directory")
    run_checkpoint = runs_dir / f"{job_id}.pt"
    latest = critic_dir / "latest.pt"
    manifest = critic_dir / "manifest.json"
    for path in (run_checkpoint, latest, manifest):
        if path.is_symlink():
            raise HTTPException(400, f"ACT-TD3 critic artifact must not be a symlink: {path}")
        if path.exists() and not path.is_file():
            raise HTTPException(
                400,
                f"ACT-TD3 critic artifact must be a regular file: {path}",
            )
    if run_checkpoint.exists():
        raise HTTPException(409, "ACT-TD3 critic warm-up run checkpoint already exists")
    return run_checkpoint, latest, manifest


def _act_td3_critic_warmup_container_name(job: _ACTTD3CriticWarmupJob) -> str:
    return f"cyclo_act_td3_critic_warmup_{job.job_id[:12]}"


def _act_td3_critic_warmup_command(
    *,
    job: _ACTTD3CriticWarmupJob,
    robot_type: str,
    robot_config: str,
) -> List[str]:
    command = _compose_base_cmd() + [
        "run",
        "--rm",
        "--no-deps",
        "--pull",
        "never",
        "--name",
        _act_td3_critic_warmup_container_name(job),
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp",
        "--env",
        f"XDG_CACHE_HOME={_OFFLINE_RL_CACHE_ROOT}",
        "--env",
        f"HF_HOME={_OFFLINE_RL_CACHE_ROOT}/huggingface",
        "--env",
        f"HF_LEROBOT_HOME={_OFFLINE_RL_CACHE_ROOT}/huggingface/lerobot",
        "--env",
        f"TORCH_HOME={_OFFLINE_RL_CACHE_ROOT}/torch",
        "--env",
        f"TRITON_CACHE_DIR={_OFFLINE_RL_CACHE_ROOT}/triton",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--entrypoint",
        "/lerobot/.venv/bin/python",
        "lerobot",
        "-m",
        "cyclo_brain.algorithm.rl.act_td3.offline_warmup_cli",
    ]
    for dataset_path in job.dataset_paths:
        command.extend(["--dataset-root", dataset_path])
    command.extend([
        "--act-checkpoint",
        job.act_checkpoint,
        "--robot-config",
        robot_config,
        "--robot-type",
        robot_type,
        "--device",
        "cuda:0",
        "--seed",
        "17",
        "--sampling-seed",
        "19",
        "--batch-size",
        str(job.batch_size),
        "--critic-updates",
        str(job.total_critic_updates),
        "--checkpoint",
        job.run_checkpoint_path,
        "--publish-dir",
        str(Path(job.act_checkpoint) / "critic"),
        "--checkpoint-interval",
        "500",
        "--progress-interval",
        "1",
    ])
    return command


def _act_td3_critic_warmup_interrupt_job(job: _ACTTD3CriticWarmupJob) -> bool:
    docker_error: Optional[Exception] = None
    try:
        container = _docker_client().containers.get(
            _act_td3_critic_warmup_container_name(job)
        )
        container.kill(signal="SIGINT")
        return True
    except NotFound:
        pass
    except DockerException as exc:
        docker_error = exc
        logger.warning(
            "Could not signal ACT-TD3 critic warm-up container %s: %s",
            _act_td3_critic_warmup_container_name(job),
            exc,
        )

    process = job.process
    if process is None:
        if docker_error is not None:
            raise RuntimeError(str(docker_error)) from docker_error
        return False
    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is not None:
        return False
    try:
        process.send_signal(signal.SIGINT)
    except (OSError, ProcessLookupError) as exc:
        if callable(poll) and poll() is not None:
            return False
        if docker_error is not None:
            raise RuntimeError(f"{docker_error}; {exc}") from exc
        raise
    return True


def _act_td3_critic_warmup_append_log(
    job: _ACTTD3CriticWarmupJob,
    line: str,
) -> None:
    if not line:
        return
    job.log_tail.append(line)
    del job.log_tail[:-_OFFLINE_RL_LOG_LINES]


def _act_td3_critic_warmup_update_number(
    job: _ACTTD3CriticWarmupJob,
    payload: dict,
    name: str,
) -> None:
    value = payload.get(name)
    if value is None:
        return
    current = getattr(job, name)
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(job, name, value)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            setattr(job, name, numeric)


def _act_td3_critic_warmup_consume_event(
    job: _ACTTD3CriticWarmupJob,
    payload: dict,
) -> bool:
    event = payload.get("event")
    reported_total = payload.get("total_critic_updates")
    total_matches = (
        isinstance(reported_total, int)
        and not isinstance(reported_total, bool)
        and reported_total == job.total_critic_updates
    )
    if event in {"manifest", "progress", "result"} and not total_matches:
        job.contract_mismatch = True
        job.message = (
            "ACT-TD3 critic warm-up reported a total update count that "
            "disagrees with the requested contract"
        )
    if event == "manifest":
        if total_matches:
            job.message = "ACT-TD3 critic warm-up is running"
        return False
    if event in {"progress", "result"}:
        for name in (
            "completed_critic_updates",
            "durable_checkpoint_updates",
            "critic_loss",
            "target_mean",
            "eta_seconds",
        ):
            _act_td3_critic_warmup_update_number(job, payload, name)
        percentage = payload.get("percentage")
        if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
            job.percentage = max(0.0, min(100.0, float(percentage)))
        unchanged = payload.get("actor_exactly_unchanged")
        if isinstance(unchanged, bool):
            job.actor_exactly_unchanged = unchanged
        if event == "result":
            checkpoint = payload.get("checkpoint_path")
            manifest = payload.get("manifest_path")
            complete = payload.get("status") == "complete"
            if complete and isinstance(checkpoint, str) and isinstance(manifest, str):
                job.checkpoint_path = checkpoint
                job.manifest_path = manifest
                job.artifact_reported = True
            if payload.get("status") == "stopped":
                job.stop_confirmed = True
                job.message = "ACT-TD3 critic warm-up stopped"
            job.result_complete = complete
            if complete and total_matches:
                job.message = "Verifying the completed critic artifact"
            return complete
        if total_matches:
            job.message = "ACT-TD3 critic warm-up is running"
        return False
    if event == "error":
        error_type = payload.get("error_type") or "ACTTD3CriticWarmupError"
        message = payload.get("message") or "ACT-TD3 critic warm-up failed"
        job.message = f"{error_type}: {message}"
    return False


def _sha256_regular_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _act_td3_critic_warmup_verified_artifact(
    job: _ACTTD3CriticWarmupJob,
) -> bool:
    expected_checkpoint = Path(job.act_checkpoint) / "critic" / "latest.pt"
    expected_manifest = Path(job.act_checkpoint) / "critic" / "manifest.json"
    if (
        os.path.normpath(job.checkpoint_path) != str(expected_checkpoint)
        or os.path.normpath(job.manifest_path) != str(expected_manifest)
    ):
        return False
    try:
        checkpoint = _offline_rl_input_path(
            str(expected_checkpoint),
            root=Path(job.act_checkpoint),
            label="ACT-TD3 critic latest.pt",
            expect_directory=False,
        )
        manifest_path = _offline_rl_input_path(
            str(expected_manifest),
            root=Path(job.act_checkpoint),
            label="ACT-TD3 critic manifest.json",
            expect_directory=False,
        )
        manifest = _offline_rl_json_file(
            manifest_path,
            label="ACT-TD3 critic manifest",
        )
        checkpoint_stat = checkpoint.stat(follow_symlinks=False)
        artifact = manifest.get("artifact")
        base_policy = manifest.get("base_policy")
        training_data = manifest.get("training_data")
        if (
            manifest.get("format")
            != "cyclo_brain.act_td3_critic_manifest/v1"
            or manifest.get("status") != "complete"
            or manifest.get("actor_exactly_unchanged") is not True
            or manifest.get("completed_critic_updates") != job.total_critic_updates
            or manifest.get("completed_actor_updates") != 0
            or not isinstance(artifact, dict)
            or artifact.get("format") != "cyclo_brain.act_td3_critic/v1"
            or artifact.get("checkpoint_path") != "latest.pt"
            or artifact.get("byte_count") != checkpoint_stat.st_size
            or checkpoint_stat.st_size < 1
            or not isinstance(base_policy, dict)
            or os.path.normpath(str(base_policy.get("path", "")))
            != job.act_checkpoint
            or not isinstance(base_policy.get("actor_sha256"), str)
            or not isinstance(training_data, dict)
            or training_data.get("dataset_roots") != job.dataset_paths
        ):
            return False
        expected_sha256 = artifact.get("sha256")
        return (
            isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
            and _sha256_regular_file(checkpoint) == expected_sha256
        )
    except (HTTPException, OSError, ValueError, TypeError):
        return False


def _monitor_act_td3_critic_warmup_job(job: _ACTTD3CriticWarmupJob) -> None:
    try:
        _OFFLINE_RL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "a", encoding="utf-8") as log:
            stdout = job.process.stdout if job.process is not None else None
            if stdout is not None:
                for raw_line in stdout:
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode(errors="replace")
                    line = raw_line.rstrip("\r\n")
                    log.write(line + "\n")
                    log.flush()
                    with _ACT_TD3_CRITIC_WARMUP_LOCK:
                        _act_td3_critic_warmup_append_log(job, line)
                        try:
                            payload = json.loads(line)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if isinstance(payload, dict):
                            _act_td3_critic_warmup_consume_event(job, payload)
            returncode = job.process.wait() if job.process is not None else -1
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        logger.error("ACT-TD3 critic warm-up monitor failed: %s", exc, exc_info=True)
        returncode = -1
        with _ACT_TD3_CRITIC_WARMUP_LOCK:
            job.message = f"ACT-TD3 critic warm-up monitor failed: {exc}"

    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        job.returncode = returncode
        completion_contract = (
            job.result_complete
            and job.artifact_reported
            and not job.contract_mismatch
            and job.completed_critic_updates == job.total_critic_updates
            and job.durable_checkpoint_updates == job.total_critic_updates
            and job.percentage == 100.0
            and job.actor_exactly_unchanged is True
        )
        if (
            returncode == 0
            and completion_contract
            and _act_td3_critic_warmup_verified_artifact(job)
        ):
            job.status = "completed"
            job.message = "ACT-TD3 critic warm-up completed"
        elif returncode == 0 and job.stop_requested and job.stop_confirmed:
            job.status = "stopped"
            job.eta_seconds = None
            job.message = "ACT-TD3 critic warm-up stopped"
        else:
            job.status = "failed"
            if not job.message or job.message in {
                "Starting ACT-TD3 critic warm-up",
                "ACT-TD3 critic warm-up is running",
                "Verifying the completed critic artifact",
            }:
                job.message = (
                    f"ACT-TD3 critic warm-up exited with code {returncode}"
                    if returncode != 0
                    else "ACT-TD3 critic warm-up completion contract is invalid"
                )


def _act_td3_critic_warmup_status(
    job: Optional[_ACTTD3CriticWarmupJob],
) -> ACTTD3CriticWarmupStatus:
    if job is None:
        return ACTTD3CriticWarmupStatus(
            status="idle",
            message="No ACT-TD3 critic warm-up job has been started",
        )
    return ACTTD3CriticWarmupStatus(
        status=job.status,
        percentage=job.percentage,
        completed_critic_updates=job.completed_critic_updates,
        total_critic_updates=job.total_critic_updates,
        durable_checkpoint_updates=job.durable_checkpoint_updates,
        critic_loss=job.critic_loss,
        target_mean=job.target_mean,
        eta_seconds=job.eta_seconds,
        actor_exactly_unchanged=job.actor_exactly_unchanged,
        episode_count=job.episode_count,
        success_count=job.success_count,
        failure_count=job.failure_count,
        batch_size=job.batch_size,
        checkpoint_path=job.checkpoint_path if job.status == "completed" else "",
        manifest_path=job.manifest_path if job.status == "completed" else "",
        message=job.message,
        job_id=job.job_id,
        dataset_path=job.dataset_path,
        dataset_paths=list(job.dataset_paths),
        act_checkpoint=job.act_checkpoint,
        returncode=job.returncode,
        log_tail=list(job.log_tail),
    )


# -- Imitation learning -------------------------------------------------------


def _imitation_learning_policy_label(policy_type: str) -> str:
    try:
        return _IMITATION_LEARNING_POLICY_LABELS[policy_type]
    except KeyError as exc:  # pragma: no cover - request model prevents this
        raise ValueError(
            f"Unsupported imitation-learning policy_type: {policy_type}"
        ) from exc


def _imitation_learning_running_message(job: _ImitationLearningJob) -> str:
    policy_label = _imitation_learning_policy_label(job.policy_type)
    return f"{policy_label} imitation learning is running"


def _imitation_learning_starting_message(policy_type: str) -> str:
    policy_label = _imitation_learning_policy_label(policy_type)
    return f"Starting {policy_label} imitation learning"


def _imitation_learning_datasets(
    raw_paths: List[str],
    *,
    policy_type: str = "act",
) -> tuple[List[Path], List[List[int]], int, int]:
    """Select behavior-cloning episodes from immutable LeRobot v3 roots.

    Converted RL data keeps its success-only filtering so failure behavior is
    not cloned.  A conventional unlabeled demonstration dataset has no outcome
    feature, so every episode is selected.
    """
    policy_label = _imitation_learning_policy_label(policy_type)
    datasets: List[Path] = []
    success_episodes: List[List[int]] = []
    seen: set[Path] = set()
    selected_count = 0
    excluded_count = 0

    for raw_path in raw_paths:
        dataset, episode_count, has_success_labels = _lerobot_v3_dataset(
            raw_path,
            require_success_labels=False,
        )
        if dataset in seen:
            raise HTTPException(400, "dataset_paths must not contain duplicates")
        seen.add(dataset)
        summary = _offline_rl_dataset_summary(str(dataset))
        if summary.version != "v3.0" or summary.total_episodes != episode_count:
            raise HTTPException(400, "LeRobot dataset summary disagrees with metadata")
        selected = [
            episode.index
            for episode in summary.episodes
            if not has_success_labels or episode.outcome == "success"
        ]
        if has_success_labels and not selected:
            raise HTTPException(
                400,
                f"{policy_label} imitation learning requires a successful "
                f"episode in {dataset}",
            )
        datasets.append(dataset)
        success_episodes.append(selected)
        selected_count += len(selected)
        excluded_count += episode_count - len(selected)

    if selected_count > _OFFLINE_RL_MAX_EPISODES:
        raise HTTPException(
            400,
            f"{policy_label} imitation learning supports at most "
            f"{_OFFLINE_RL_MAX_EPISODES} selected episodes",
        )
    return datasets, success_episodes, selected_count, excluded_count


def _imitation_learning_output_path(
    job_id: str,
    steps: int,
    *,
    policy_type: str = "act",
) -> Path:
    model_root = _OFFLINE_RL_MODEL_ROOT.resolve(strict=True)
    if _IMITATION_LEARNING_OUTPUT_ROOT.exists():
        if _IMITATION_LEARNING_OUTPUT_ROOT.is_symlink():
            raise HTTPException(
                500,
                "Imitation learning output root must not be a symbolic link",
            )
        try:
            _IMITATION_LEARNING_OUTPUT_ROOT.resolve(strict=True).relative_to(model_root)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                500,
                "Imitation learning output root escapes the model root",
            ) from exc
    try:
        prefix = _IMITATION_LEARNING_POLICY_OUTPUT_PREFIXES[policy_type]
    except KeyError as exc:  # pragma: no cover - request model prevents this
        raise ValueError(
            f"Unsupported imitation-learning policy_type: {policy_type}"
        ) from exc
    output = _IMITATION_LEARNING_OUTPUT_ROOT / (
        f"{prefix}_steps_{steps:06d}_{job_id[:12]}"
    )
    if output.exists():
        raise HTTPException(409, f"Imitation learning output already exists: {output}")
    return output


def _imitation_learning_container_name(job: _ImitationLearningJob) -> str:
    return f"cyclo_imitation_learning_{job.job_id[:12]}"


def _imitation_learning_command(job: _ImitationLearningJob) -> List[str]:
    try:
        training_module = _IMITATION_LEARNING_POLICY_MODULES[job.policy_type]
    except KeyError as exc:  # pragma: no cover - job creation prevents this
        raise ValueError(
            f"Unsupported imitation-learning policy_type: {job.policy_type}"
        ) from exc
    pretrained_cache_environment: List[str] = []
    if job.policy_type == "multi_task_dit":
        pretrained_cache_environment = [
            "--env",
            "HF_HUB_CACHE=/huggingface_hub",
            "--env",
            "HUGGINGFACE_HUB_CACHE=/huggingface_hub",
            "--env",
            "TRANSFORMERS_CACHE=/huggingface_hub",
        ]
    command = _compose_base_cmd() + [
        "run",
        "--rm",
        "--no-deps",
        "--pull",
        "never",
        "--name",
        _imitation_learning_container_name(job),
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp",
        "--env",
        f"XDG_CACHE_HOME={_IMITATION_LEARNING_CACHE_ROOT}",
        "--env",
        f"HF_HOME={_IMITATION_LEARNING_CACHE_ROOT}/huggingface",
        "--env",
        f"HF_LEROBOT_HOME={_IMITATION_LEARNING_CACHE_ROOT}/huggingface/lerobot",
    ] + pretrained_cache_environment + [
        "--env",
        f"TORCH_HOME={_IMITATION_LEARNING_CACHE_ROOT}/torch",
        "--env",
        f"TRITON_CACHE_DIR={_IMITATION_LEARNING_CACHE_ROOT}/triton",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--entrypoint",
        "/lerobot/.venv/bin/python",
        "lerobot",
        "-m",
        training_module,
    ]
    for dataset_path, episodes in zip(
        job.dataset_paths,
        job.success_episodes,
        strict=True,
    ):
        command.extend(["--dataset-root", dataset_path])
        command.extend([
            "--episodes",
            ",".join(str(index) for index in episodes),
        ])
    if job.policy_type == "multi_task_dit":
        command.extend(["--task-instruction", job.task_instruction])
    else:
        for group in job.trainable_groups:
            command.extend(["--trainable-group", group])
    command.extend([
        "--output-dir",
        job.output_dir,
        "--steps",
        str(job.total_steps),
        "--batch-size",
        str(job.batch_size),
        "--save-freq",
        str(job.save_freq),
        "--chunk-size",
        str(job.chunk_size),
        "--progress-interval",
        "10",
        "--device",
        "cuda",
        "--video-backend",
        "pyav",
    ])
    return command


def _imitation_learning_interrupt_job(job: _ImitationLearningJob) -> bool:
    docker_error: Optional[Exception] = None
    try:
        container = _docker_client().containers.get(
            _imitation_learning_container_name(job)
        )
        container.kill(signal="SIGINT")
        return True
    except NotFound:
        pass
    except DockerException as exc:
        docker_error = exc
        logger.warning(
            "Could not signal imitation-learning container %s: %s",
            _imitation_learning_container_name(job),
            exc,
        )

    process = job.process
    if process is None:
        if docker_error is not None:
            raise RuntimeError(str(docker_error)) from docker_error
        return False
    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is not None:
        return False
    try:
        process.send_signal(signal.SIGINT)
    except (OSError, ProcessLookupError) as exc:
        if callable(poll) and poll() is not None:
            return False
        if docker_error is not None:
            raise RuntimeError(f"{docker_error}; {exc}") from exc
        raise
    return True


def _imitation_learning_append_log(job: _ImitationLearningJob, line: str) -> None:
    if not line:
        return
    job.log_tail.append(line)
    del job.log_tail[:-_OFFLINE_RL_LOG_LINES]


def _imitation_learning_update_number(
    job: _ImitationLearningJob,
    payload: dict,
    name: str,
) -> None:
    value = payload.get(name)
    if value is None:
        return
    current = getattr(job, name)
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(job, name, value)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        setattr(job, name, float(value))


def _imitation_learning_consume_event(
    job: _ImitationLearningJob,
    payload: dict,
) -> bool:
    event = payload.get("event")
    if event == "manifest":
        job.message = _imitation_learning_running_message(job)
        return False
    if event == "progress":
        completed_steps = payload.get("completed_steps", payload.get("step"))
        if (
            isinstance(completed_steps, int)
            and not isinstance(completed_steps, bool)
            and completed_steps >= 0
        ):
            job.completed_steps = completed_steps
        for name in (
            "total_steps",
            "loss",
            "l1_loss",
            "kld_loss",
            "eta_seconds",
        ):
            _imitation_learning_update_number(job, payload, name)
        percentage = payload.get("percentage")
        if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
            job.percentage = max(0.0, min(100.0, float(percentage)))
        checkpoint = payload.get("checkpoint_path")
        if isinstance(checkpoint, str):
            job.checkpoint_path = checkpoint
        job.message = _imitation_learning_running_message(job)
        return False
    if event == "result":
        _imitation_learning_consume_event(job, {**payload, "event": "progress"})
        if payload.get("status") == "stopped":
            job.stop_confirmed = True
        model_path = payload.get("model_path")
        checkpoint_path = payload.get("checkpoint_path")
        if isinstance(model_path, str):
            job.model_path = model_path
        if isinstance(checkpoint_path, str):
            job.checkpoint_path = checkpoint_path
        return payload.get("status") == "complete" and bool(job.model_path)
    if event == "error":
        error_type = payload.get("error_type") or "ImitationLearningError"
        message = payload.get("message") or (
            f"{_imitation_learning_policy_label(job.policy_type)} "
            "imitation learning failed"
        )
        job.message = f"{error_type}: {message}"
    return False


def _imitation_learning_verified_model(job: _ImitationLearningJob) -> bool:
    digits = max(6, len(str(job.total_steps)))
    expected = (
        Path(job.output_dir)
        / "checkpoints"
        / f"{job.total_steps:0{digits}d}"
        / "pretrained_model"
    )
    if os.path.normpath(job.model_path) != str(expected):
        return False
    try:
        model = _offline_rl_input_path(
            job.model_path,
            root=_OFFLINE_RL_MODEL_ROOT,
            label="imitation model_path",
            expect_directory=True,
        )
    except HTTPException:
        return False
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        try:
            _offline_rl_input_path(
                str(model / name),
                root=model,
                label=f"imitation model {name}",
                expect_directory=False,
            )
        except HTTPException:
            return False
    try:
        config = _offline_rl_json_file(
            model / "config.json",
            label="imitation policy config",
        )
    except HTTPException:
        return False
    if config.get("type") != job.policy_type:
        return False
    return True


def _monitor_imitation_learning_job(job: _ImitationLearningJob) -> None:
    result_complete = False
    try:
        _IMITATION_LEARNING_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "a", encoding="utf-8") as log:
            stdout = job.process.stdout if job.process is not None else None
            if stdout is not None:
                for raw_line in stdout:
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode(errors="replace")
                    line = raw_line.rstrip("\r\n")
                    log.write(line + "\n")
                    log.flush()
                    with _IMITATION_LEARNING_LOCK:
                        _imitation_learning_append_log(job, line)
                        try:
                            payload = json.loads(line)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if isinstance(payload, dict):
                            result_complete = (
                                _imitation_learning_consume_event(job, payload)
                                or result_complete
                            )
            returncode = job.process.wait() if job.process is not None else -1
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        logger.error("Imitation-learning monitor failed: %s", exc, exc_info=True)
        returncode = -1
        with _IMITATION_LEARNING_LOCK:
            job.message = f"Imitation-learning monitor failed: {exc}"

    with _IMITATION_LEARNING_LOCK:
        job.returncode = returncode
        policy_label = _imitation_learning_policy_label(job.policy_type)
        completed_with_verified_model = (
            returncode == 0
            and result_complete
            and _imitation_learning_verified_model(job)
        )
        if completed_with_verified_model:
            job.status = "completed"
            job.percentage = 100.0
            job.completed_steps = job.total_steps
            job.eta_seconds = 0.0
            job.message = f"{policy_label} imitation learning completed"
        elif returncode == 0 and job.stop_requested and job.stop_confirmed:
            job.status = "stopped"
            job.model_path = ""
            job.eta_seconds = None
            job.message = f"{policy_label} imitation learning stopped"
        else:
            job.status = "failed"
            if not job.message or job.message in (
                _imitation_learning_starting_message(job.policy_type),
                _imitation_learning_running_message(job),
            ):
                job.message = (
                    f"{policy_label} imitation learning exited with code {returncode}"
                    if returncode != 0
                    else (
                        f"{policy_label} imitation-learning result or model "
                        "verification is missing"
                    )
                )


def _imitation_learning_status(
    job: Optional[_ImitationLearningJob],
) -> ImitationLearningStatus:
    if job is None:
        return ImitationLearningStatus(
            status="idle",
            message="No imitation-learning job has been started",
        )
    return ImitationLearningStatus(
        status=job.status,
        percentage=job.percentage,
        episode_count=job.episode_count,
        excluded_episode_count=job.excluded_episode_count,
        completed_steps=job.completed_steps,
        total_steps=job.total_steps,
        batch_size=job.batch_size,
        save_freq=job.save_freq,
        chunk_size=job.chunk_size,
        loss=job.loss,
        l1_loss=job.l1_loss,
        kld_loss=job.kld_loss,
        eta_seconds=job.eta_seconds,
        model_path=job.model_path if job.status == "completed" else "",
        checkpoint_path=job.checkpoint_path,
        message=job.message,
        job_id=job.job_id,
        dataset_path=job.dataset_path,
        dataset_paths=list(job.dataset_paths),
        policy_type=job.policy_type,
        task_instruction=job.task_instruction,
        trainable_groups=list(job.trainable_groups),
        output_dir=job.output_dir,
        returncode=job.returncode,
        log_tail=list(job.log_tail),
    )


def _backend_image_candidates(spec: Dict[str, str]) -> List[str]:
    candidates = [spec["image"]]
    alt = spec.get("image_alt")
    if alt and alt not in candidates:
        candidates.append(alt)
    return candidates


def _local_backend_image(client: docker.DockerClient, spec: Dict[str, str]) -> Optional[str]:
    for image in _backend_image_candidates(spec):
        try:
            client.images.get(image)
            return image
        except ImageNotFound:
            continue
    return None


def _container_raw_state(container) -> str:
    try:
        container.reload()
    except DockerException:
        pass
    return container.attrs.get("State", {}).get("Status", "unknown")


def _resolve_groot_trt_paths(
    model_path: str,
    engine_path: str = "",
) -> tuple[str, str]:
    model = os.path.normpath((model_path or "").strip())
    if not model or not os.path.isabs(model):
        raise HTTPException(400, "model_path must be an absolute path")
    root = os.path.normpath(_GROOT_MODEL_ROOT)
    if model != root and not model.startswith(root + os.sep):
        raise HTTPException(
            400,
            f"model_path must be under {_GROOT_MODEL_ROOT}",
        )

    engine = (engine_path or "").strip()
    if engine:
        if not os.path.isabs(engine):
            engine = os.path.join(model, engine)
        engine = os.path.normpath(engine)
    else:
        engine = os.path.join(model, "dit_model_bf16.trt")

    if engine != model and not engine.startswith(model + os.sep):
        raise HTTPException(400, "engine_path must be inside model_path")
    return model, engine


def _trt_manifest_path(engine_path: str) -> str:
    return f"{engine_path}.json"


def _trt_log_path(engine_path: str) -> str:
    return f"{engine_path}.build.log"


def _read_json_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("could not read json file %s: %s", path, e)
        return {}


def _write_json_file(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _tail_log(path: str, max_bytes: int = 12000, max_lines: int = 40) -> List[str]:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line][-max_lines:]


def _trt_returncode_from_log(lines: List[str]) -> Optional[int]:
    marker = "=== TensorRT build exited rc="
    for line in reversed(lines):
        if marker not in line:
            continue
        suffix = line.split(marker, 1)[1].split(None, 1)[0]
        try:
            return int(suffix)
        except ValueError:
            return None
    return None


def _trt_failure_message(returncode: Optional[int]) -> str:
    if returncode == 137:
        return (
            "TensorRT build was killed (rc=137), likely due to out-of-memory"
        )
    if returncode is not None:
        return f"TensorRT build failed (rc={returncode})"
    return "TensorRT build failed"


def _active_trt_job(engine_path: str) -> Optional[_TrtBuildJob]:
    with _TRT_BUILD_LOCK:
        job = _TRT_BUILD_JOBS.get(engine_path)
        if job and job.status == "building":
            return job
    return None


def _trt_status(model_path: str, engine_path: str) -> TrtEngineStatus:
    log_path = _trt_log_path(engine_path)
    log_tail = _tail_log(log_path)
    job = _active_trt_job(engine_path)
    if job is not None:
        return TrtEngineStatus(
            model_path=model_path,
            engine_path=engine_path,
            status="building",
            message=job.message,
            started_at=job.started_at,
            updated_at=time.time(),
            returncode=job.returncode,
            log_tail=log_tail,
        )

    manifest = _read_json_file(_trt_manifest_path(engine_path))
    engine_ready = os.path.exists(engine_path) and os.path.getsize(engine_path) > 0
    if engine_ready:
        return TrtEngineStatus(
            model_path=model_path,
            engine_path=engine_path,
            status="ready",
            message=manifest.get("message", "TensorRT engine ready"),
            engine_size_bytes=os.path.getsize(engine_path),
            started_at=manifest.get("started_at"),
            updated_at=manifest.get("updated_at"),
            finished_at=manifest.get("finished_at"),
            returncode=manifest.get("returncode"),
            log_tail=log_tail,
        )

    manifest_status = str(manifest.get("status", "") or "")
    if manifest_status == "building":
        returncode = _trt_returncode_from_log(log_tail)
        return TrtEngineStatus(
            model_path=model_path,
            engine_path=engine_path,
            status="failed",
            message=(
                _trt_failure_message(returncode)
                if returncode is not None
                else "Previous TensorRT build did not finish"
            ),
            started_at=manifest.get("started_at"),
            updated_at=manifest.get("updated_at"),
            finished_at=manifest.get("finished_at"),
            returncode=returncode,
            log_tail=log_tail,
        )
    if manifest_status == "failed":
        return TrtEngineStatus(
            model_path=model_path,
            engine_path=engine_path,
            status="failed",
            message=manifest.get("message", "TensorRT build failed"),
            started_at=manifest.get("started_at"),
            updated_at=manifest.get("updated_at"),
            finished_at=manifest.get("finished_at"),
            returncode=manifest.get("returncode"),
            log_tail=log_tail,
        )

    if not os.path.isdir(model_path):
        return TrtEngineStatus(
            model_path=model_path,
            engine_path=engine_path,
            status="unknown",
            message="Model path does not exist",
            log_tail=log_tail,
        )

    return TrtEngineStatus(
        model_path=model_path,
        engine_path=engine_path,
        status="missing",
        message="TensorRT engine is missing",
        log_tail=log_tail,
    )


def _assert_backend_container_running(name: str, spec: Dict[str, str]):
    try:
        ctr = _docker_client().containers.get(spec["container"])
    except NotFound:
        raise HTTPException(409, f"{spec['container']} is not created")
    except DockerException as e:
        raise HTTPException(500, f"docker inspect failed: {e}")
    state = _container_raw_state(ctr)
    if state != "running":
        raise HTTPException(409, f"{spec['container']} is not running ({state})")
    return ctr


def _monitor_trt_build_job(job: _TrtBuildJob, cmd: List[str]) -> None:
    try:
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        with open(job.log_path, "ab") as log:
            log.write(
                (
                    f"\n=== TensorRT build started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    f"model_path={job.model_path}\n"
                    f"engine_path={job.engine_path}\n"
                ).encode()
            )
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            with _TRT_BUILD_LOCK:
                job.process = process
            rc = process.wait()
            log.write(
                (
                    f"\n=== TensorRT build exited rc={rc} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                ).encode()
            )
    except Exception as e:
        rc = -1
        message = f"TensorRT build launch failed: {e}"
        logger.error(message, exc_info=True)
    else:
        engine_ready = (
            os.path.exists(job.engine_path)
            and os.path.getsize(job.engine_path) > 0
        )
        message = (
            "TensorRT engine ready"
            if rc == 0 and engine_ready
            else _trt_failure_message(rc)
        )

    with _TRT_BUILD_LOCK:
        engine_ready = (
            os.path.exists(job.engine_path)
            and os.path.getsize(job.engine_path) > 0
        )
        job.returncode = rc
        job.finished_at = time.time()
        job.status = "ready" if rc == 0 and engine_ready else "failed"
        job.message = message

    manifest = {
        "status": job.status,
        "model_path": job.model_path,
        "engine_path": job.engine_path,
        "message": job.message,
        "started_at": job.started_at,
        "updated_at": time.time(),
        "finished_at": job.finished_at,
        "returncode": job.returncode,
    }
    if engine_ready:
        manifest["engine_size_bytes"] = os.path.getsize(job.engine_path)
    try:
        _write_json_file(_trt_manifest_path(job.engine_path), manifest)
    except OSError as e:
        logger.warning("could not write TensorRT manifest: %s", e)


def _start_trt_build_job(
    model_path: str,
    engine_path: str,
    robot_type: str,
    task_instruction: str,
    workspace_mb: Optional[int],
    force: bool,
) -> _TrtBuildJob:
    log_path = _trt_log_path(engine_path)
    cmd = [
        "docker",
        "exec",
        _BACKENDS["groot"]["container"],
        "python3",
        "-m",
        "runtime.prepare_trt_engine",
        "--model-path",
        model_path,
        "--engine-path",
        engine_path,
        "--robot-type",
        robot_type,
        "--task-instruction",
        task_instruction,
    ]
    if workspace_mb:
        cmd.extend(["--workspace-mb", str(workspace_mb)])
    if force:
        cmd.append("--force")

    job = _TrtBuildJob(
        model_path=model_path,
        engine_path=engine_path,
        log_path=log_path,
        started_at=time.time(),
    )
    with _TRT_BUILD_LOCK:
        active = _TRT_BUILD_JOBS.get(engine_path)
        if active and active.status == "building":
            return active
        _TRT_BUILD_JOBS[engine_path] = job

    thread = threading.Thread(
        target=_monitor_trt_build_job,
        args=(job, cmd),
        daemon=True,
        name="groot-trt-build",
    )
    thread.start()
    return job


def _backend_container_image_mismatch(
    client: docker.DockerClient,
    container,
    spec: Dict[str, str],
) -> bool:
    """Return True when an existing backend container uses an older image ID."""
    container_image_id = container.attrs.get("Image")
    if not container_image_id:
        return False

    found_local_image = False
    for image in _backend_image_candidates(spec):
        try:
            expected_image = client.images.get(image)
        except ImageNotFound:
            continue
        found_local_image = True
        expected_image_id = getattr(expected_image, "id", None)
        if expected_image_id and expected_image_id == container_image_id:
            return False

    return found_local_image


def _missing_required_mounts(name: str, container) -> List[str]:
    required_mounts = _REQUIRED_BACKEND_MOUNTS.get(name, ())
    if not required_mounts:
        return []
    mounted_destinations = {
        mount.get("Destination")
        for mount in container.attrs.get("Mounts", [])
    }
    return [
        destination for destination in required_mounts
        if destination not in mounted_destinations
    ]


def _backend_container_workspace_mount_mismatch(
    container,
    expected_workspace_dir: Optional[str],
) -> bool:
    if not expected_workspace_dir:
        return False
    workspace_source = _mount_source_for_destination(
        container.attrs.get("Mounts", []),
        "/workspace",
    )
    if not workspace_source:
        return False
    return (
        _normalized_host_path(workspace_source)
        != _normalized_host_path(expected_workspace_dir)
    )


def _backend_container_stale_reason(
    name: str,
    client: docker.DockerClient,
    container,
    spec: Dict[str, str],
    expected_workspace_dir: Optional[str],
) -> Optional[str]:
    missing_mounts = _missing_required_mounts(name, container)
    if missing_mounts:
        return "missing_required_mounts=" + ",".join(missing_mounts)
    if _backend_container_workspace_mount_mismatch(
        container,
        expected_workspace_dir,
    ):
        return "workspace_mount_mismatch"
    if _backend_container_image_mismatch(client, container, spec):
        return "image_mismatch"
    return None


def _backend_raw_state_for_stale_reason(reason: str) -> str:
    if reason == "image_mismatch":
        return "stale_image"
    return reason


def _backend_service_statuses(
    container,
    raw_state: str,
    service_names: List[str],
) -> List[ServiceStatus]:
    """Inspect the two s6-managed policy runtime processes."""
    if raw_state != "running":
        return []

    services = " ".join(service_names)
    script = f"""
S6_SVSTAT=$(ls /package/admin/s6-*/command/s6-svstat 2>/dev/null | head -1)
[ -z "$S6_SVSTAT" ] && S6_SVSTAT=$(command -v s6-svstat 2>/dev/null)
if [ -z "$S6_SVSTAT" ]; then
  for svc in {services}; do
    printf '%s\ts6-svstat not found\n' "$svc"
  done
  exit 0
fi
for svc in {services}; do
  svdir="/run/service/$svc"
  if [ -d "$svdir" ]; then
    raw=$("$S6_SVSTAT" "$svdir" 2>&1)
    printf '%s\t%s\n' "$svc" "$raw"
  else
    printf '%s\tnot registered\n' "$svc"
  fi
done
"""
    try:
        result = container.exec_run(["sh", "-lc", script])
    except DockerException as e:
        return [
            ServiceStatus(
                name=name,
                state="unknown",
                raw=f"inspect failed: {e}",
            )
            for name in service_names
        ]

    output = result.output.decode(errors="replace") if result.output else ""
    statuses: List[ServiceStatus] = []
    seen = set()
    for line in output.splitlines():
        if "\t" not in line:
            continue
        name, raw = line.split("\t", 1)
        if name not in service_names:
            continue
        parsed = _parse_svstat(raw)
        statuses.append(ServiceStatus(name=name, raw=raw, **parsed))
        seen.add(name)

    for name in service_names:
        if name not in seen:
            statuses.append(
                ServiceStatus(name=name, state="unknown", raw="not reported")
            )
    return statuses


# -- parsing -------------------------------------------------------------------


def _parse_svstat(raw: str) -> dict:
    """Best-effort parse of s6-svstat output.

    Example: 'up (pid 1234) 37 seconds' or 'down 3 seconds, normally up'.
    We only need state + pid + uptime; everything else we return verbatim.
    """
    tokens = raw.split()
    state: Literal["up", "down", "unknown"] = "unknown"
    if tokens:
        if tokens[0] == "up":
            state = "up"
        elif tokens[0] == "down":
            state = "down"

    pid: Optional[int] = None
    uptime_s: Optional[int] = None
    if "(pid" in raw:
        try:
            pid_part = raw.split("(pid", 1)[1].split(")", 1)[0].strip().split()[0]
            pid = int(pid_part)
        except (ValueError, IndexError):
            pass
    if "seconds" in raw:
        # token before "seconds" is the uptime
        try:
            idx = tokens.index("seconds")
            uptime_s = int(tokens[idx - 1])
        except (ValueError, IndexError):
            pass

    return {"state": state, "pid": pid, "uptime_s": uptime_s}


# -- FastAPI app ---------------------------------------------------------------


app = FastAPI(
    title="cyclo_intelligence supervisor_api",
    description=__doc__,
    version="1.2.2",
)

_include_router_with_eager_routes(app, navigation_router)


def _flow_sde_ppo_training_conflict() -> Optional[str]:
    """Keep all GPU-training supervisors mutually exclusive."""

    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            return "Stop the ACT-TD3 job before starting Flow-SDE PPO"
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        if (
            _ACT_TD3_CRITIC_WARMUP_JOB is not None
            and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
        ):
            return "Stop ACT-TD3 critic warm-up before starting Flow-SDE PPO"
    with _IMITATION_LEARNING_LOCK:
        if (
            _IMITATION_LEARNING_JOB is not None
            and _IMITATION_LEARNING_JOB.status == "running"
        ):
            return "Stop imitation learning before starting Flow-SDE PPO"
    rlt_stage1 = globals().get("_RLT_STAGE1_SUPERVISOR")
    if rlt_stage1 is not None and rlt_stage1.is_running():
        return "Stop GR00T RLT Stage 1 before starting Flow-SDE PPO"
    rlt_stage2 = globals().get("_RLT_STAGE2_SUPERVISOR")
    if rlt_stage2 is not None and rlt_stage2.is_running():
        return "Stop GR00T RLT Stage 2 before starting Flow-SDE PPO"
    return None


def _flow_sde_ppo_interrupt_container(container_name: str) -> bool:
    """Signal only the exact Compose run container owned by this job."""

    try:
        container = _docker_client().containers.get(container_name)
        container.kill(signal="SIGINT")
        return True
    except NotFound:
        return False
    except DockerException as exc:
        raise RuntimeError(
            f"Could not signal Flow-SDE PPO container {container_name}: {exc}"
        ) from exc


# Flow-SDE PPO is intentionally a sibling of the ACT/TD3 offline API.  Its
# subprocess owns a live simulator action-step session and consumes explicit
# Success/Fail episode outcomes, so sharing the offline request model would be
# unsafe and misleading.
from supervisor_api.flow_sde_ppo_service import (  # noqa: E402
    create_flow_sde_ppo_router,
)


flow_sde_ppo_router, _FLOW_SDE_PPO_SUPERVISOR = create_flow_sde_ppo_router(
    compose_command=_compose_base_cmd,
    compose_environment=_compose_env,
    conflict_message=_flow_sde_ppo_training_conflict,
    interrupt_container=_flow_sde_ppo_interrupt_container,
)
_include_router_with_eager_routes(app, flow_sde_ppo_router)


def _rlt_stage1_training_conflict() -> Optional[str]:
    """Keep frozen-GR00T representation training GPU-exclusive."""

    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            return "Stop the ACT-TD3 job before starting GR00T RLT Stage 1"
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        if (
            _ACT_TD3_CRITIC_WARMUP_JOB is not None
            and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
        ):
            return "Stop ACT-TD3 critic warm-up before starting GR00T RLT Stage 1"
    with _IMITATION_LEARNING_LOCK:
        if (
            _IMITATION_LEARNING_JOB is not None
            and _IMITATION_LEARNING_JOB.status == "running"
        ):
            return "Stop imitation learning before starting GR00T RLT Stage 1"
    if _FLOW_SDE_PPO_SUPERVISOR.is_running():
        return "Stop Flow-SDE PPO before starting GR00T RLT Stage 1"
    rlt_stage2 = globals().get("_RLT_STAGE2_SUPERVISOR")
    if rlt_stage2 is not None and rlt_stage2.is_running():
        return "Stop GR00T RLT Stage 2 before starting GR00T RLT Stage 1"
    return None


def _rlt_stage1_interrupt_container(container_name: str) -> bool:
    """Signal only the one-shot GR00T container owned by this Stage 1 job."""

    try:
        container = _docker_client().containers.get(container_name)
        container.kill(signal="SIGINT")
        return True
    except NotFound:
        return False
    except DockerException as exc:
        raise RuntimeError(
            f"Could not signal RLT Stage 1 container {container_name}: {exc}"
        ) from exc


from supervisor_api.rlt_stage1_service import (  # noqa: E402
    create_rlt_stage1_router,
)


rlt_stage1_router, _RLT_STAGE1_SUPERVISOR = create_rlt_stage1_router(
    compose_command=_compose_base_cmd,
    compose_environment=_compose_env,
    conflict_message=_rlt_stage1_training_conflict,
    interrupt_container=_rlt_stage1_interrupt_container,
)
_include_router_with_eager_routes(app, rlt_stage1_router)


def _rlt_stage2_training_conflict() -> Optional[str]:
    """Keep RLT actor-critic training GPU-exclusive."""

    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            return "Stop the ACT-TD3 job before starting GR00T RLT Stage 2"
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        if (
            _ACT_TD3_CRITIC_WARMUP_JOB is not None
            and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
        ):
            return "Stop ACT-TD3 critic warm-up before starting GR00T RLT Stage 2"
    with _IMITATION_LEARNING_LOCK:
        if (
            _IMITATION_LEARNING_JOB is not None
            and _IMITATION_LEARNING_JOB.status == "running"
        ):
            return "Stop imitation learning before starting GR00T RLT Stage 2"
    if _FLOW_SDE_PPO_SUPERVISOR.is_running():
        return "Stop Flow-SDE PPO before starting GR00T RLT Stage 2"
    if _RLT_STAGE1_SUPERVISOR.is_running():
        return "Stop GR00T RLT Stage 1 before starting GR00T RLT Stage 2"
    return None


def _rlt_stage2_interrupt_container(container_name: str) -> bool:
    """Signal only the one-shot GR00T container owned by this Stage 2 job."""

    try:
        container = _docker_client().containers.get(container_name)
        container.kill(signal="SIGINT")
        return True
    except NotFound:
        return False
    except DockerException as exc:
        raise RuntimeError(
            f"Could not signal RLT Stage 2 container {container_name}: {exc}"
        ) from exc


from supervisor_api.rlt_stage2_service import (  # noqa: E402
    create_rlt_stage2_router,
)


rlt_stage2_router, _RLT_STAGE2_SUPERVISOR = create_rlt_stage2_router(
    compose_command=_compose_base_cmd,
    compose_environment=_compose_env,
    conflict_message=_rlt_stage2_training_conflict,
    interrupt_container=_rlt_stage2_interrupt_container,
)
_include_router_with_eager_routes(app, rlt_stage2_router)


def _reject_running_flow_sde_ppo() -> None:
    if _FLOW_SDE_PPO_SUPERVISOR.is_running():
        raise HTTPException(
            409,
            "Stop Flow-SDE PPO before starting another GPU training job",
        )
    if _RLT_STAGE1_SUPERVISOR.is_running():
        raise HTTPException(
            409,
            "Stop GR00T RLT Stage 1 before starting another GPU training job",
        )
    if _RLT_STAGE2_SUPERVISOR.is_running():
        raise HTTPException(
            409,
            "Stop GR00T RLT Stage 2 before starting another GPU training job",
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    container = os.environ.get("HOSTNAME", "unknown")
    s6_ready = os.path.isdir("/run/service")
    return HealthResponse(ok=True, container=container, s6_ready=s6_ready)


@app.get("/workspace", response_model=WorkspaceMountResponse)
async def workspace_mount() -> WorkspaceMountResponse:
    host_root = await asyncio.to_thread(_host_workspace_dir)
    if host_root:
        return WorkspaceMountResponse(
            container_root="/workspace",
            host_root=host_root,
            host_available=True,
            message="/workspace host mount resolved",
        )

    return WorkspaceMountResponse(
        container_root="/workspace",
        host_root=None,
        host_available=False,
        message="Host mount for /workspace could not be resolved",
    )


@app.get("/offline-rl/dataset", response_model=OfflineRLDatasetSummary)
async def offline_rl_dataset(dataset_path: str) -> OfflineRLDatasetSummary:
    """Return episode cards and success summary for a local v2.1/v3 dataset."""
    return await asyncio.to_thread(_offline_rl_dataset_summary, dataset_path)


@app.get(
    "/offline-rl/dataset/episode-data",
    response_model=OfflineRLDatasetEpisodeData,
)
async def offline_rl_dataset_episode_data(
    dataset_path: str,
    episode_index: int,
) -> OfflineRLDatasetEpisodeData:
    """Return lazy, replay-compatible joint/action data for one episode."""
    return await asyncio.to_thread(
        _offline_rl_dataset_episode_data,
        dataset_path,
        episode_index,
    )


@app.post(
    "/offline-rl/data-epochs/reserve",
    response_model=OfflineRLDataEpochProvenance,
)
async def offline_rl_reserve_data_epoch(
    request: OfflineRLDataEpochReserveRequest,
) -> OfflineRLDataEpochProvenance:
    """Reserve one immutable conversion output root and write its provenance."""
    return await asyncio.to_thread(_offline_rl_reserve_data_epoch, request)


@app.get("/offline-rl/datasets", response_model=OfflineRLDatasetInventory)
async def offline_rl_datasets(root_path: str = "") -> OfflineRLDatasetInventory:
    """Discover converted v2.1/v3 datasets below one safe LeRobot folder."""
    return await asyncio.to_thread(_offline_rl_dataset_inventory, root_path)


@app.post(
    "/offline-rl/dataset/delete-episodes",
    response_model=OfflineRLDatasetDeleteResult,
)
async def offline_rl_dataset_delete_episodes(
    request: OfflineRLDatasetDeleteRequest,
) -> OfflineRLDatasetDeleteResult:
    """Remove selected episodes, or the dataset when every episode is selected."""
    dataset = await asyncio.to_thread(
        _offline_rl_delete_dataset_episodes,
        request.dataset_path,
        request.episode_indices,
    )
    deleted = ", ".join(str(index) for index in sorted(request.episode_indices))
    if dataset is None:
        return OfflineRLDatasetDeleteResult(
            ok=True,
            message=f"Deleted episode(s) {deleted}; the empty dataset was removed",
            dataset=None,
            dataset_deleted=True,
        )
    return OfflineRLDatasetDeleteResult(
        ok=True,
        message=f"Deleted episode(s) {deleted}; remaining episodes were reindexed",
        dataset=dataset,
    )


@app.post("/offline-rl/start", response_model=OfflineRLStatus)
async def offline_rl_start(request: OfflineRLStartRequest) -> OfflineRLStatus:
    """Start one immutable ACT-TD3 cumulative-replay round."""
    global _OFFLINE_RL_JOB

    algorithm, actor_objective = _offline_rl_algorithm_contract(request)
    critic_epochs, actor_equivalent_epochs = _offline_rl_schedule(
        request.critic_epochs,
        request.actor_equivalent_epochs,
    )
    actor_trainable_groups = _offline_rl_objective_trainable_groups(
        actor_objective,
        request.actor_trainable_groups,
    )

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "A LeRobot dataset edit is already running")
    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            raise HTTPException(409, "An offline RL job is already running")
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        if (
            _ACT_TD3_CRITIC_WARMUP_JOB is not None
            and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
        ):
            raise HTTPException(409, "An ACT-TD3 critic warm-up job is already running")
    with _IMITATION_LEARNING_LOCK:
        if (
            _IMITATION_LEARNING_JOB is not None
            and _IMITATION_LEARNING_JOB.status == "running"
        ):
            raise HTTPException(409, "An imitation-learning job is already running")
    _reject_running_flow_sde_ppo()

    requested_dataset_paths = _offline_rl_requested_dataset_paths(request)
    datasets, episode_count, success_count, failure_count = _offline_rl_datasets(
        requested_dataset_paths
    )
    act_checkpoint = _offline_rl_act_checkpoint(request.act_checkpoint)
    parent_checkpoint, previous_episode_count, parent_round_index = (
        _offline_rl_parent_checkpoint(
            request.parent_checkpoint,
            episode_count,
            actor_trainable_groups,
            request.batch_size,
            actor_objective,
        )
    )
    robot_type = _validate_robot_type(request.robot_type)
    robot_config = _offline_rl_robot_config(robot_type)
    job_id = uuid.uuid4().hex
    output_dir = _offline_rl_output_path(job_id, episode_count, actor_objective)
    log_path = _OFFLINE_RL_LOG_ROOT / f"{job_id}.log"
    job = _OfflineRLJob(
        job_id=job_id,
        # The scalar path is retained for older status consumers.  The ordered
        # list is the authoritative immutable Data Epoch replay lineage.
        dataset_path=str(datasets[0]),
        dataset_paths=[str(dataset) for dataset in datasets],
        act_checkpoint=str(act_checkpoint),
        parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else "",
        output_dir=str(output_dir),
        episode_count=episode_count,
        log_path=str(log_path),
        algorithm=algorithm,
        actor_objective=actor_objective,
        round_index=parent_round_index + 1,
        round_episode_count=episode_count - previous_episode_count,
        batch_size=request.batch_size,
        critic_epochs=critic_epochs,
        actor_equivalent_epochs=actor_equivalent_epochs,
        actor_trainable_groups=actor_trainable_groups,
        success_count=success_count,
        failure_count=failure_count,
        total_epochs=critic_epochs,
        checkpoint_path=str(output_dir / "training_state" / "act_td3.pt"),
    )
    command = _offline_rl_command(
        job=job,
        robot_type=robot_type,
        robot_config=robot_config,
    )
    try:
        environment = _compose_env()
    except Exception as exc:  # noqa: BLE001 - Docker mount discovery boundary
        raise HTTPException(503, f"Could not resolve Docker workspace: {exc}") from exc

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "A LeRobot dataset edit is already running")
        with _OFFLINE_RL_LOCK:
            if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
                raise HTTPException(409, "An offline RL job is already running")
            with _ACT_TD3_CRITIC_WARMUP_LOCK:
                if (
                    _ACT_TD3_CRITIC_WARMUP_JOB is not None
                    and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
                ):
                    raise HTTPException(
                        409,
                        "An ACT-TD3 critic warm-up job is already running",
                    )
                with _IMITATION_LEARNING_LOCK:
                    if (
                        _IMITATION_LEARNING_JOB is not None
                        and _IMITATION_LEARNING_JOB.status == "running"
                    ):
                        raise HTTPException(
                            409,
                            "An imitation-learning job is already running",
                        )
                    _reject_running_flow_sde_ppo()
                    try:
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=environment,
                        )
                    except OSError as exc:
                        raise HTTPException(503, f"Could not launch offline RL: {exc}") from exc
                    job.process = process
                    _OFFLINE_RL_JOB = job

    thread = threading.Thread(
        target=_monitor_offline_rl_job,
        args=(job,),
        daemon=True,
        name=f"offline-rl-{job_id[:12]}",
    )
    thread.start()
    with _OFFLINE_RL_LOCK:
        return _offline_rl_status(job)


@app.get("/offline-rl/status", response_model=OfflineRLStatus)
async def offline_rl_status() -> OfflineRLStatus:
    with _OFFLINE_RL_LOCK:
        return _offline_rl_status(_OFFLINE_RL_JOB)


@app.post("/offline-rl/stop", response_model=OfflineRLStatus)
async def offline_rl_stop(request: OfflineRLStopRequest) -> OfflineRLStatus:
    """Interrupt the currently running job only when its full id matches."""
    with _OFFLINE_RL_LOCK:
        job = _OFFLINE_RL_JOB
        requested_job_id = request.job_id.strip()
        if job is None or requested_job_id != job.job_id:
            raise HTTPException(409, "Offline RL job_id is stale or no longer current")
        if job.status == "stopped":
            return _offline_rl_status(job)
        if job.status != "running":
            raise HTTPException(409, f"Offline RL job is already {job.status}")
        if job.stop_requested:
            return _offline_rl_status(job)
        job.stop_requested = True
        job.message = "Stopping ACT-TD3 training"

    try:
        interrupted = await asyncio.to_thread(_offline_rl_interrupt_job, job)
    except Exception as exc:  # noqa: BLE001 - Docker/subprocess control boundary
        with _OFFLINE_RL_LOCK:
            if _OFFLINE_RL_JOB is job and job.status != "running":
                return _offline_rl_status(job)
            if _OFFLINE_RL_JOB is job and job.status == "running":
                job.stop_requested = False
                job.message = "ACT-TD3 training is running"
        raise HTTPException(503, f"Could not stop offline RL job: {exc}") from exc

    if not interrupted:
        with _OFFLINE_RL_LOCK:
            if _OFFLINE_RL_JOB is job and job.status != "running":
                return _offline_rl_status(job)
            if _OFFLINE_RL_JOB is job:
                job.stop_requested = False
                job.message = "ACT-TD3 training is running"
        raise HTTPException(409, "Offline RL job exited before it could be stopped")

    with _OFFLINE_RL_LOCK:
        return _offline_rl_status(job)


@app.post("/offline-rl/cancel", response_model=OfflineRLStatus)
async def offline_rl_cancel(request: OfflineRLCancelRequest) -> OfflineRLStatus:
    """Discard only the current stopped/failed round's incomplete artifacts."""

    with _OFFLINE_RL_LOCK:
        job = _OFFLINE_RL_JOB
        requested_job_id = request.job_id.strip()
        if job is None or requested_job_id != job.job_id:
            raise HTTPException(409, "Offline RL job_id is stale or no longer current")
        if job.status not in {"stopped", "failed"}:
            raise HTTPException(
                409,
                f"Offline RL job must be stopped or failed before cancellation; "
                f"current status is {job.status}",
            )
        _offline_rl_cancel_output(job)
        _offline_rl_mark_cancelled(job)
        return _offline_rl_status(job)


@app.post(
    "/offline-rl/critic-warmup/start",
    response_model=ACTTD3CriticWarmupStatus,
)
async def act_td3_critic_warmup_start(
    request: ACTTD3CriticWarmupStartRequest,
) -> ACTTD3CriticWarmupStatus:
    """Start actor-frozen critic warm-up for one selected ACT policy."""

    global _ACT_TD3_CRITIC_WARMUP_JOB

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "A LeRobot dataset edit is already running")
    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            raise HTTPException(409, "An offline RL job is already running")
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        if (
            _ACT_TD3_CRITIC_WARMUP_JOB is not None
            and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
        ):
            raise HTTPException(409, "An ACT-TD3 critic warm-up job is already running")
    with _IMITATION_LEARNING_LOCK:
        if (
            _IMITATION_LEARNING_JOB is not None
            and _IMITATION_LEARNING_JOB.status == "running"
        ):
            raise HTTPException(409, "An imitation-learning job is already running")
    _reject_running_flow_sde_ppo()

    requested_dataset_paths = _offline_rl_requested_dataset_paths(request)
    datasets, episode_count, success_count, failure_count = _offline_rl_datasets(
        requested_dataset_paths
    )
    act_checkpoint = _offline_rl_act_checkpoint(request.act_checkpoint)
    robot_type = _validate_robot_type(request.robot_type)
    robot_config = _offline_rl_robot_config(robot_type)
    job_id = uuid.uuid4().hex
    run_checkpoint, latest, manifest = _act_td3_critic_warmup_paths(
        act_checkpoint,
        job_id,
    )
    job = _ACTTD3CriticWarmupJob(
        job_id=job_id,
        dataset_path=str(datasets[0]),
        dataset_paths=[str(dataset) for dataset in datasets],
        act_checkpoint=str(act_checkpoint),
        checkpoint_path=str(latest),
        manifest_path=str(manifest),
        run_checkpoint_path=str(run_checkpoint),
        episode_count=episode_count,
        success_count=success_count,
        failure_count=failure_count,
        batch_size=request.batch_size,
        total_critic_updates=request.critic_updates,
        log_path=str(_OFFLINE_RL_LOG_ROOT / f"critic_warmup_{job_id}.log"),
    )
    command = _act_td3_critic_warmup_command(
        job=job,
        robot_type=robot_type,
        robot_config=robot_config,
    )
    try:
        environment = _compose_env()
    except Exception as exc:  # noqa: BLE001 - Docker mount discovery boundary
        raise HTTPException(503, f"Could not resolve Docker workspace: {exc}") from exc

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "A LeRobot dataset edit is already running")
        with _OFFLINE_RL_LOCK:
            if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
                raise HTTPException(409, "An offline RL job is already running")
            with _ACT_TD3_CRITIC_WARMUP_LOCK:
                if (
                    _ACT_TD3_CRITIC_WARMUP_JOB is not None
                    and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
                ):
                    raise HTTPException(
                        409,
                        "An ACT-TD3 critic warm-up job is already running",
                    )
                with _IMITATION_LEARNING_LOCK:
                    if (
                        _IMITATION_LEARNING_JOB is not None
                        and _IMITATION_LEARNING_JOB.status == "running"
                    ):
                        raise HTTPException(
                            409,
                            "An imitation-learning job is already running",
                        )
                    _reject_running_flow_sde_ppo()
                    try:
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=environment,
                        )
                    except OSError as exc:
                        raise HTTPException(
                            503,
                            f"Could not launch ACT-TD3 critic warm-up: {exc}",
                        ) from exc
                    job.process = process
                    _ACT_TD3_CRITIC_WARMUP_JOB = job

    thread = threading.Thread(
        target=_monitor_act_td3_critic_warmup_job,
        args=(job,),
        daemon=True,
        name=f"act-td3-critic-warmup-{job_id[:12]}",
    )
    thread.start()
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        return _act_td3_critic_warmup_status(job)


@app.get(
    "/offline-rl/critic-warmup/status",
    response_model=ACTTD3CriticWarmupStatus,
)
async def act_td3_critic_warmup_status() -> ACTTD3CriticWarmupStatus:
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        return _act_td3_critic_warmup_status(_ACT_TD3_CRITIC_WARMUP_JOB)


@app.post(
    "/offline-rl/critic-warmup/stop",
    response_model=ACTTD3CriticWarmupStatus,
)
async def act_td3_critic_warmup_stop(
    request: ACTTD3CriticWarmupStopRequest,
) -> ACTTD3CriticWarmupStatus:
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        job = _ACT_TD3_CRITIC_WARMUP_JOB
        requested_job_id = request.job_id.strip()
        if job is None or requested_job_id != job.job_id:
            raise HTTPException(
                409,
                "ACT-TD3 critic warm-up job_id is stale or no longer current",
            )
        if job.status == "stopped":
            return _act_td3_critic_warmup_status(job)
        if job.status != "running":
            raise HTTPException(
                409,
                f"ACT-TD3 critic warm-up job is already {job.status}",
            )
        if job.stop_requested:
            return _act_td3_critic_warmup_status(job)
        job.stop_requested = True
        job.message = "Stopping ACT-TD3 critic warm-up"

    try:
        interrupted = await asyncio.to_thread(
            _act_td3_critic_warmup_interrupt_job,
            job,
        )
    except Exception as exc:  # noqa: BLE001 - Docker/subprocess control boundary
        with _ACT_TD3_CRITIC_WARMUP_LOCK:
            if _ACT_TD3_CRITIC_WARMUP_JOB is job and job.status != "running":
                return _act_td3_critic_warmup_status(job)
            if _ACT_TD3_CRITIC_WARMUP_JOB is job:
                job.stop_requested = False
                job.message = "ACT-TD3 critic warm-up is running"
        raise HTTPException(
            503,
            f"Could not stop ACT-TD3 critic warm-up: {exc}",
        ) from exc

    if not interrupted:
        with _ACT_TD3_CRITIC_WARMUP_LOCK:
            if _ACT_TD3_CRITIC_WARMUP_JOB is job and job.status != "running":
                return _act_td3_critic_warmup_status(job)
            if _ACT_TD3_CRITIC_WARMUP_JOB is job:
                job.stop_requested = False
                job.message = "ACT-TD3 critic warm-up is running"
        raise HTTPException(
            409,
            "ACT-TD3 critic warm-up exited before it could be stopped",
        )

    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        return _act_td3_critic_warmup_status(job)


@app.post("/imitation-learning/start", response_model=ImitationLearningStatus)
async def imitation_learning_start(
    request: ImitationLearningStartRequest,
) -> ImitationLearningStatus:
    """Train an official LeRobot policy with behavior cloning on selected demos."""
    global _IMITATION_LEARNING_JOB

    expected_chunk_size = _IMITATION_LEARNING_POLICY_CHUNK_SIZES[request.policy_type]
    if request.policy_type == "multi_task_dit" and request.chunk_size != expected_chunk_size:
        policy_label = _imitation_learning_policy_label(request.policy_type)
        raise HTTPException(
            400,
            f"{policy_label} imitation learning requires "
            f"chunk_size={expected_chunk_size}",
        )
    if request.policy_type == "act":
        trainable_groups = _imitation_learning_trainable_groups(
            request.trainable_groups
        )
    else:
        if request.trainable_groups:
            raise HTTPException(
                400,
                "trainable_groups is available only for ACT imitation learning",
            )
        trainable_groups = []

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "A LeRobot dataset edit is already running")
    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            raise HTTPException(409, "An offline RL job is already running")
    with _ACT_TD3_CRITIC_WARMUP_LOCK:
        if (
            _ACT_TD3_CRITIC_WARMUP_JOB is not None
            and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
        ):
            raise HTTPException(409, "An ACT-TD3 critic warm-up job is already running")
    with _IMITATION_LEARNING_LOCK:
        if (
            _IMITATION_LEARNING_JOB is not None
            and _IMITATION_LEARNING_JOB.status == "running"
        ):
            raise HTTPException(409, "An imitation-learning job is already running")
    _reject_running_flow_sde_ppo()

    requested_paths = _offline_rl_requested_dataset_paths(request)
    datasets, success_episodes, episode_count, excluded_count = (
        _imitation_learning_datasets(
            requested_paths,
            policy_type=request.policy_type,
        )
    )
    job_id = uuid.uuid4().hex
    output_dir = _imitation_learning_output_path(
        job_id,
        request.steps,
        policy_type=request.policy_type,
    )
    log_path = _IMITATION_LEARNING_LOG_ROOT / f"{job_id}.log"
    job = _ImitationLearningJob(
        job_id=job_id,
        dataset_path=str(datasets[0]),
        dataset_paths=[str(dataset) for dataset in datasets],
        success_episodes=success_episodes,
        output_dir=str(output_dir),
        episode_count=episode_count,
        excluded_episode_count=excluded_count,
        log_path=str(log_path),
        total_steps=request.steps,
        batch_size=request.batch_size,
        save_freq=request.save_freq,
        chunk_size=request.chunk_size,
        policy_type=request.policy_type,
        task_instruction=request.task_instruction,
        trainable_groups=trainable_groups,
        message=_imitation_learning_starting_message(request.policy_type),
    )
    command = _imitation_learning_command(job)
    try:
        environment = _compose_env()
    except Exception as exc:  # noqa: BLE001 - Docker mount discovery boundary
        raise HTTPException(503, f"Could not resolve Docker workspace: {exc}") from exc

    with _OFFLINE_RL_DATASET_EDIT_LOCK:
        if _OFFLINE_RL_DATASET_EDIT_ACTIVE:
            raise HTTPException(409, "A LeRobot dataset edit is already running")
        with _OFFLINE_RL_LOCK:
            if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
                raise HTTPException(409, "An offline RL job is already running")
            with _ACT_TD3_CRITIC_WARMUP_LOCK:
                if (
                    _ACT_TD3_CRITIC_WARMUP_JOB is not None
                    and _ACT_TD3_CRITIC_WARMUP_JOB.status == "running"
                ):
                    raise HTTPException(
                        409,
                        "An ACT-TD3 critic warm-up job is already running",
                    )
                with _IMITATION_LEARNING_LOCK:
                    if (
                        _IMITATION_LEARNING_JOB is not None
                        and _IMITATION_LEARNING_JOB.status == "running"
                    ):
                        raise HTTPException(
                            409,
                            "An imitation-learning job is already running",
                        )
                    _reject_running_flow_sde_ppo()
                    try:
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=environment,
                        )
                    except OSError as exc:
                        raise HTTPException(
                            503,
                            f"Could not launch imitation learning: {exc}",
                        ) from exc
                    job.process = process
                    _IMITATION_LEARNING_JOB = job

    thread = threading.Thread(
        target=_monitor_imitation_learning_job,
        args=(job,),
        daemon=True,
        name=f"imitation-learning-{job_id[:12]}",
    )
    thread.start()
    with _IMITATION_LEARNING_LOCK:
        return _imitation_learning_status(job)


@app.get("/imitation-learning/status", response_model=ImitationLearningStatus)
async def imitation_learning_status() -> ImitationLearningStatus:
    with _IMITATION_LEARNING_LOCK:
        return _imitation_learning_status(_IMITATION_LEARNING_JOB)


@app.post("/imitation-learning/stop", response_model=ImitationLearningStatus)
async def imitation_learning_stop(
    request: ImitationLearningStopRequest,
) -> ImitationLearningStatus:
    """Interrupt only the exact active imitation-learning job."""
    with _IMITATION_LEARNING_LOCK:
        job = _IMITATION_LEARNING_JOB
        requested_job_id = request.job_id.strip()
        if job is None or requested_job_id != job.job_id:
            raise HTTPException(
                409,
                "Imitation-learning job_id is stale or no longer current",
            )
        if job.status == "stopped":
            return _imitation_learning_status(job)
        if job.status != "running":
            raise HTTPException(
                409,
                f"Imitation-learning job is already {job.status}",
            )
        if job.stop_requested:
            return _imitation_learning_status(job)
        job.stop_requested = True
        policy_label = _imitation_learning_policy_label(job.policy_type)
        job.message = f"Stopping {policy_label} imitation learning"

    try:
        interrupted = await asyncio.to_thread(
            _imitation_learning_interrupt_job,
            job,
        )
    except Exception as exc:  # noqa: BLE001 - Docker/subprocess control boundary
        with _IMITATION_LEARNING_LOCK:
            if _IMITATION_LEARNING_JOB is job and job.status != "running":
                return _imitation_learning_status(job)
            if _IMITATION_LEARNING_JOB is job:
                job.stop_requested = False
                job.message = _imitation_learning_running_message(job)
        raise HTTPException(
            503,
            f"Could not stop imitation-learning job: {exc}",
        ) from exc

    if not interrupted:
        with _IMITATION_LEARNING_LOCK:
            if _IMITATION_LEARNING_JOB is job and job.status != "running":
                return _imitation_learning_status(job)
            if _IMITATION_LEARNING_JOB is job:
                job.stop_requested = False
                job.message = _imitation_learning_running_message(job)
        raise HTTPException(
            409,
            "Imitation-learning job exited before it could be stopped",
        )

    with _IMITATION_LEARNING_LOCK:
        return _imitation_learning_status(job)


@app.get("/services", response_model=ServiceList)
async def list_services() -> ServiceList:
    items: List[ServiceStatus] = []
    for name in _USER_SERVICES:
        svdir = f"/run/service/{name}"
        if not os.path.isdir(svdir):
            items.append(
                ServiceStatus(name=name, state="unknown", raw="not registered")
            )
            continue
        result = await _run("s6-svstat", svdir)
        parsed = _parse_svstat(result.stdout)
        items.append(
            ServiceStatus(
                name=name,
                state=parsed["state"],
                pid=parsed["pid"],
                uptime_s=parsed["uptime_s"],
                raw=result.stdout,
            )
        )
    return ServiceList(services=items)


@app.get("/services/{name}/status", response_model=ServiceStatus)
async def service_status(name: str) -> ServiceStatus:
    _require_known_service(name)
    svdir = f"/run/service/{name}"
    if not os.path.isdir(svdir):
        return ServiceStatus(name=name, state="unknown", raw="not registered")
    result = await _run("s6-svstat", svdir)
    parsed = _parse_svstat(result.stdout)
    return ServiceStatus(name=name, raw=result.stdout, **parsed)


@app.post("/services/{name}/start", response_model=ActionResult)
async def service_start(
    name: str,
    request: Optional[ServiceActionRequest] = None,
) -> ActionResult:
    _require_known_service(name)
    if name == "bt_node":
        robot_type = _validate_bt_robot_type(
            request.robot_type if request else ""
        )
        _write_bt_robot_type(robot_type)
    # s6-rc -u change <name> brings the service up (idempotent).
    result = await _run("s6-rc", "-u", "change", name)
    ok = result.rc == 0
    msg = result.stderr or result.stdout or f"rc={result.rc}"
    return ActionResult(ok=ok, message=msg)


@app.post("/services/{name}/stop", response_model=ActionResult)
async def service_stop(name: str) -> ActionResult:
    _require_known_service(name)
    result = await _run("s6-rc", "-d", "change", name)
    ok = result.rc == 0
    msg = result.stderr or result.stdout or f"rc={result.rc}"
    return ActionResult(ok=ok, message=msg)


# -- Backend container endpoints — PLAN §4.8 -----------------------------------
# Hybrid wiring (matches PLAN §4.8 example):
#   - pull   → docker-py client.api.pull(stream=True), SSE per layer
#   - start  → restart an existing running container, start an existing
#              stopped container, or 'docker compose up -d --no-build
#              <service>' when the container does not exist. No build is
#              attempted from the UI path; missing images are reported so the
#              user can pull/install first.
#   - stop   → docker-py container.stop(), keeping the container for reuse.
#   - restart → hard reset an existing backend, or create/start it when absent.
#   - status → docker-py images.get + containers.get


@app.get("/backends/groot/trt/status", response_model=TrtEngineStatus)
async def groot_trt_status(
    model_path: str,
    engine_path: str = "",
) -> TrtEngineStatus:
    model, engine = _resolve_groot_trt_paths(model_path, engine_path)
    return _trt_status(model, engine)


@app.post("/backends/groot/trt/build", response_model=TrtEngineStatus)
async def groot_trt_build(request: TrtBuildRequest) -> TrtEngineStatus:
    model, engine = _resolve_groot_trt_paths(
        request.model_path,
        request.engine_path,
    )
    robot_type = request.robot_type.strip()
    if not robot_type:
        raise HTTPException(400, "robot_type is required")
    if request.workspace_mb is not None and request.workspace_mb <= 0:
        raise HTTPException(400, "workspace_mb must be positive")
    if not os.path.isdir(model):
        raise HTTPException(404, f"model_path does not exist: {model}")

    spec = _require_known_backend("groot")
    await asyncio.to_thread(_assert_backend_container_running, "groot", spec)

    current = _trt_status(model, engine)
    if current.status == "ready" and not request.force:
        return current

    await asyncio.to_thread(
        _start_trt_build_job,
        model,
        engine,
        robot_type,
        request.task_instruction,
        request.workspace_mb,
        request.force,
    )
    return _trt_status(model, engine)


@app.post("/backends/{name}/pull")
async def backend_pull(name: str) -> StreamingResponse:
    spec = _require_known_backend(name)
    image = spec["image"]

    def generate():
        try:
            client = _docker_client()
        except DockerException as e:
            payload = json.dumps({"message": f"docker init failed: {e}"})
            yield f"event: error\ndata: {payload}\n\n"
            return

        try:
            for chunk in client.api.pull(image, stream=True, decode=True):
                yield f"data: {json.dumps(chunk)}\n\n"
        except DockerException as e:
            payload = json.dumps({"image": image, "message": str(e)})
            yield f"event: error\ndata: {payload}\n\n"
            return

        # Verify the image is actually present after the pull stream ends —
        # the daemon sometimes ends the stream on a 'manifest unknown' error
        # without raising on the iterator side.
        try:
            client.images.get(image)
            done = json.dumps({"image": image, "ok": True})
            yield f"event: done\ndata: {done}\n\n"
        except ImageNotFound:
            payload = json.dumps(
                {"image": image, "message": "pull stream ended but image missing"}
            )
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/backends/{name}/start", response_model=ActionResult)
async def backend_start(name: str) -> ActionResult:
    spec = _require_known_backend(name)
    return await _ensure_backend_running(name, spec)


@app.post("/backends/{name}/restart", response_model=ActionResult)
async def backend_restart(name: str) -> ActionResult:
    spec = _require_known_backend(name)
    return await _ensure_backend_running(name, spec)


@app.post("/backends/{name}/recreate", response_model=ActionResult)
async def backend_recreate(name: str) -> ActionResult:
    spec = _require_known_backend(name)

    def _remove_existing() -> tuple[str, str]:
        try:
            client = _docker_client()
        except DockerException as e:
            raise HTTPException(500, f"docker init failed: {e}")
        local_image = _local_backend_image(client, spec)
        if not local_image:
            images = ", ".join(_backend_image_candidates(spec))
            raise HTTPException(
                409,
                f"No local image for {name}. Expected one of: {images}. "
                f"Connect internet and call /backends/{name}/pull first.",
            )
        try:
            ctr = client.containers.get(spec["container"])
        except NotFound:
            removed = "not_created"
        except DockerException as e:
            raise HTTPException(500, f"inspect failed: {e}")
        else:
            try:
                ctr.remove(force=True)
                removed = "removed"
            except DockerException as e:
                raise HTTPException(500, f"remove failed: {e}")
        return local_image, removed

    local_image, removed = await asyncio.to_thread(_remove_existing)
    cmd = _compose_base_cmd() + ["create", "--no-build", spec["service"]]
    result = await _run(*cmd, timeout=60.0, env=_compose_env())
    ok = result.rc == 0
    msg = result.stderr or result.stdout or f"rc={result.rc}"
    if ok:
        msg = (
            f"{spec['container']} recreated from {local_image} "
            f"({removed}). {msg}"
        )
    return ActionResult(ok=ok, message=msg)


@app.post("/backends/{name}/stop", response_model=ActionResult)
async def backend_stop(name: str) -> ActionResult:
    spec = _require_known_backend(name)
    container_name = spec["container"]

    def _stop_existing() -> tuple[bool, str]:
        try:
            client = _docker_client()
        except DockerException as e:
            return False, f"docker init failed: {e}"
        try:
            ctr = client.containers.get(container_name)
        except NotFound:
            return True, f"{container_name} was not created"
        except DockerException as e:
            return False, f"inspect failed: {e}"
        try:
            state = _container_raw_state(ctr)
            if state == "paused":
                ctr.unpause()
                state = "running"
            if state != "running":
                return True, f"{container_name} already stopped ({state})"
            ctr.stop(timeout=10)
            return True, f"{container_name} stopped"
        except DockerException as e:
            return False, f"stop failed: {e}"

    ok, msg = await asyncio.to_thread(_stop_existing)
    return ActionResult(ok=ok, message=msg)


async def _ensure_backend_running(name: str, spec: Dict[str, str]) -> ActionResult:
    """Start policy backend without building; reset if it is already running."""

    container_name = spec["container"]

    def _start_or_restart_existing() -> tuple[Optional[bool], str]:
        try:
            client = _docker_client()
        except DockerException as e:
            return False, f"docker init failed: {e}"
        try:
            ctr = client.containers.get(container_name)
        except NotFound:
            return None, "not_created"
        except DockerException as e:
            return False, f"inspect failed: {e}"

        try:
            stale_reason = _backend_container_stale_reason(
                name,
                client,
                ctr,
                spec,
                _host_workspace_dir(),
            )
            if stale_reason:
                ctr.remove(force=True)
                return None, stale_reason

            state = _container_raw_state(ctr)
            if state == "paused":
                ctr.unpause()
                state = "running"
            if state == "running":
                ctr.restart(timeout=10)
                return True, f"{container_name} restarted"
            ctr.start()
            return True, f"{container_name} started from {state}"
        except DockerException as e:
            return False, f"start/restart failed: {e}"

    handled, msg = await asyncio.to_thread(_start_or_restart_existing)
    if handled is not None:
        return ActionResult(ok=handled, message=msg)
    compose_reason = msg

    # Container is absent. Pre-flight the image so compose up never starts an
    # implicit pull/build path from a simple ON click.
    def _find_local_image() -> Optional[str]:
        try:
            return _local_backend_image(_docker_client(), spec)
        except DockerException:
            return None

    local_image = await asyncio.to_thread(_find_local_image)
    if not local_image:
        images = ", ".join(_backend_image_candidates(spec))
        raise HTTPException(
            409,
            f"No local image for {name}. Expected one of: {images}. "
            f"Connect internet and call /backends/{name}/pull first.",
        )

    cmd = _compose_base_cmd() + ["up", "-d", "--no-build", spec["service"]]
    result = await _run(*cmd, timeout=60.0, env=_compose_env())
    ok = result.rc == 0
    msg = result.stderr or result.stdout or f"rc={result.rc}"
    if ok:
        reason = ""
        if compose_reason != "not_created":
            reason = f" after recreating stale container ({compose_reason})"
        msg = (
            f"{spec['container']} created/started{reason} "
            f"using local image {local_image}. {msg}"
        )
    return ActionResult(ok=ok, message=msg)


@app.get("/backends/{name}/status", response_model=BackendStatus)
async def backend_status(name: str) -> BackendStatus:
    spec = _require_known_backend(name)

    def _inspect():
        client = _docker_client()
        pulled = _local_backend_image(client, spec) is not None
        image_status: Literal["current", "stale", "missing"] = (
            "current" if pulled else "missing"
        )
        try:
            ctr = client.containers.get(spec["container"])
        except NotFound:
            return pulled, image_status, "not_created", None, None, []
        except DockerException as e:
            raise HTTPException(500, f"docker inspect failed: {e}")
        stale_reason = _backend_container_stale_reason(
            name,
            client,
            ctr,
            spec,
            _host_workspace_dir(),
        )
        if stale_reason:
            return (
                pulled,
                "stale",
                "exited",
                ctr.id,
                _backend_raw_state_for_stale_reason(stale_reason),
                [],
            )
        raw = _container_raw_state(ctr)
        if raw == "running":
            mapped = "running"
        elif raw in ("exited", "dead", "created", "paused"):
            mapped = "exited"
        else:
            mapped = "unknown"
        service_names = spec.get("services", ["main-runtime", "engine-process"])
        services = _backend_service_statuses(ctr, raw, service_names)
        return pulled, image_status, mapped, ctr.id, raw, services

    pulled, image_status, container_state, container_id, raw, services = await asyncio.to_thread(_inspect)
    return BackendStatus(
        name=name,
        image=spec["image"],
        image_pulled=pulled,
        image_status=image_status,
        container_state=container_state,
        container_id=container_id,
        raw_state=raw,
        services=services,
    )
