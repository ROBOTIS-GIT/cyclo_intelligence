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
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


class OfflineRLStartRequest(BaseModel):
    dataset_path: str
    act_checkpoint: str
    parent_checkpoint: str = ""
    algorithm: str = "td3"
    robot_type: str
    critic_epochs: int = Field(default=10, ge=1, le=1000)
    actor_equivalent_epochs: int = Field(default=5, ge=1, le=500)


class OfflineRLStatus(BaseModel):
    status: Literal["idle", "running", "completed", "failed"]
    percentage: float = 0.0
    episode_count: int = 0
    round_index: int = 0
    round_episode_count: int = 0
    critic_epochs: int = 10
    actor_equivalent_epochs: int = 5
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
    eta_seconds: Optional[float] = None
    model_path: str = ""
    checkpoint_path: str = ""
    message: str = ""
    job_id: str = ""
    dataset_path: str = ""
    act_checkpoint: str = ""
    parent_checkpoint: str = ""
    output_dir: str = ""
    returncode: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list)


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
_OFFLINE_RL_MODEL_ROOT = Path("/workspace/model/lerobot")
_OFFLINE_RL_OUTPUT_ROOT = _OFFLINE_RL_MODEL_ROOT / "offline_rl"
_OFFLINE_RL_LOG_ROOT = Path("/tmp/cyclo_offline_rl")
_OFFLINE_RL_MAX_EPISODES = 200
_OFFLINE_RL_MAX_NEW_EPISODES = 50
_OFFLINE_RL_LOG_LINES = 100


@dataclass
class _OfflineRLJob:
    job_id: str
    dataset_path: str
    act_checkpoint: str
    parent_checkpoint: str
    output_dir: str
    episode_count: int
    log_path: str
    round_index: int = 1
    round_episode_count: int = 0
    critic_epochs: int = 10
    actor_equivalent_epochs: int = 5
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
    eta_seconds: Optional[float] = None
    model_path: str = ""
    checkpoint_path: str = ""
    message: str = "Starting ACT-TD3 offline training"
    process: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None
    log_tail: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.log_tail is None:
            self.log_tail = []


_OFFLINE_RL_JOB: Optional[_OfflineRLJob] = None
_OFFLINE_RL_LOCK = threading.Lock()


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


def _offline_rl_dataset(raw_path: str) -> tuple[Path, int]:
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
    if not isinstance(features, dict) or "episode_success" not in features:
        raise HTTPException(400, "LeRobot dataset is missing episode_success labels")
    return dataset, episodes


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
    if critic_epochs != 2 * actor_equivalent_epochs:
        raise HTTPException(
            400,
            "TD3 requires critic_epochs = 2 * actor_equivalent_epochs "
            "because policy_update_period is fixed at 2",
        )
    return critic_epochs, actor_equivalent_epochs


def _offline_rl_parent_checkpoint(
    raw_path: str,
    episode_count: int,
) -> tuple[Path | None, int, int]:
    value = (raw_path or "").strip()
    if not value:
        if episode_count > _OFFLINE_RL_MAX_NEW_EPISODES:
            raise HTTPException(
                400,
                "The first ACT-TD3 round may contain 1..50 episodes",
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


def _offline_rl_output_path(job_id: str, episode_count: int) -> Path:
    model_root = _OFFLINE_RL_MODEL_ROOT.resolve(strict=True)
    if _OFFLINE_RL_OUTPUT_ROOT.exists():
        if _OFFLINE_RL_OUTPUT_ROOT.is_symlink():
            raise HTTPException(500, "Offline RL output root must not be a symbolic link")
        try:
            _OFFLINE_RL_OUTPUT_ROOT.resolve(strict=True).relative_to(model_root)
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "Offline RL output root escapes the model root") from exc
    output = _OFFLINE_RL_OUTPUT_ROOT / (
        f"td3_episodes_{episode_count:04d}_{job_id[:12]}"
    )
    if output.exists():
        raise HTTPException(409, f"Offline RL output already exists: {output}")
    return output


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
        f"cyclo_offline_rl_{job.job_id[:12]}",
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp",
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
        "--dataset-root",
        job.dataset_path,
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
        "4",
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
    ]
    if job.parent_checkpoint:
        command.extend(["--parent-checkpoint", job.parent_checkpoint])
    return command


def _offline_rl_append_log(job: _OfflineRLJob, line: str) -> None:
    if not line:
        return
    job.log_tail.append(line)
    del job.log_tail[:-_OFFLINE_RL_LOG_LINES]


def _offline_rl_update_number(job: _OfflineRLJob, payload: dict, name: str) -> None:
    value = payload.get(name)
    current = getattr(job, name)
    if value is None:
        return
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(job, name, value)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        setattr(job, name, float(value))


def _offline_rl_consume_event(job: _OfflineRLJob, payload: dict) -> bool:
    """Apply one trusted CLI JSON event; return True for a complete result."""
    event = payload.get("event")
    if event == "manifest":
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
        percentage = payload.get("percentage")
        if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
            job.percentage = max(0.0, min(100.0, float(percentage)))
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
        _offline_rl_consume_event(job, {**payload, "event": "progress"})
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
        percentage=job.percentage,
        episode_count=job.episode_count,
        round_index=job.round_index,
        round_episode_count=job.round_episode_count,
        critic_epochs=job.critic_epochs,
        actor_equivalent_epochs=job.actor_equivalent_epochs,
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
        eta_seconds=job.eta_seconds,
        model_path=job.model_path if job.status == "completed" else "",
        checkpoint_path=job.checkpoint_path,
        message=job.message,
        job_id=job.job_id,
        dataset_path=job.dataset_path,
        act_checkpoint=job.act_checkpoint,
        parent_checkpoint=job.parent_checkpoint,
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


@app.post("/offline-rl/start", response_model=OfflineRLStatus)
async def offline_rl_start(request: OfflineRLStartRequest) -> OfflineRLStatus:
    """Start one immutable ACT-TD3 cumulative-replay round."""
    global _OFFLINE_RL_JOB

    algorithm = request.algorithm.strip().lower()
    if algorithm != "td3":
        raise HTTPException(
            400,
            f"Offline RL algorithm '{request.algorithm}' is not implemented; "
            "only TD3 is currently available",
        )
    critic_epochs, actor_equivalent_epochs = _offline_rl_schedule(
        request.critic_epochs,
        request.actor_equivalent_epochs,
    )

    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            raise HTTPException(409, "An offline RL job is already running")

    dataset, episode_count = _offline_rl_dataset(request.dataset_path)
    act_checkpoint = _offline_rl_act_checkpoint(request.act_checkpoint)
    parent_checkpoint, previous_episode_count, parent_round_index = (
        _offline_rl_parent_checkpoint(
            request.parent_checkpoint,
            episode_count,
        )
    )
    robot_type = _validate_robot_type(request.robot_type)
    robot_config = _offline_rl_robot_config(robot_type)
    job_id = uuid.uuid4().hex
    output_dir = _offline_rl_output_path(job_id, episode_count)
    log_path = _OFFLINE_RL_LOG_ROOT / f"{job_id}.log"
    job = _OfflineRLJob(
        job_id=job_id,
        dataset_path=str(dataset),
        act_checkpoint=str(act_checkpoint),
        parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else "",
        output_dir=str(output_dir),
        episode_count=episode_count,
        log_path=str(log_path),
        round_index=parent_round_index + 1,
        round_episode_count=episode_count - previous_episode_count,
        critic_epochs=critic_epochs,
        actor_equivalent_epochs=actor_equivalent_epochs,
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

    with _OFFLINE_RL_LOCK:
        if _OFFLINE_RL_JOB is not None and _OFFLINE_RL_JOB.status == "running":
            raise HTTPException(409, "An offline RL job is already running")
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
