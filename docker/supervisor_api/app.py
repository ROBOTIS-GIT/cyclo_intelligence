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
    uvicorn supervisor_api.app:app --host 127.0.0.1 --port 8100

nginx proxies /api/ → 127.0.0.1:8100 (Step 6-E).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


logger = logging.getLogger("supervisor_api")


# -- s6-rc runner --------------------------------------------------------------


# Names the UI may start/stop. Kept explicit so a stray POST can't
# poke at s6-agent or the log pipelines.
_USER_SERVICES: tuple[str, ...] = (
    "orchestrator",
    "cyclo_data",
    "web_video_server",
)


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


class HealthResponse(BaseModel):
    ok: bool
    container: str
    s6_ready: bool


class BackendStatus(BaseModel):
    name: str
    image: str
    image_pulled: bool
    container_state: Literal["running", "exited", "not_created", "unknown"]
    container_id: Optional[str] = None
    raw_state: Optional[str] = None


# -- Backend (policy container) wiring -----------------------------------------


# Compose file path inside this container — the cyclo_intelligence service
# bind-mounts the repo root at /root/ros2_ws/src/cyclo_intelligence (live
# edits during dev).
_COMPOSE_FILE_IN_CONTAINER = (
    "/root/ros2_ws/src/cyclo_intelligence/docker/docker-compose.yml"
)
_CYCLO_REPO_MOUNT = "/root/ros2_ws/src/cyclo_intelligence"


def _detect_arch() -> str:
    machine = os.uname().machine
    return "arm64" if machine in ("aarch64", "arm64") else "amd64"


_BACKEND_VERSION = os.environ.get("VERSION", "1.0.0")
_BACKEND_ARCH = os.environ.get("ARCH", _detect_arch())


# Image tags follow docker-compose.yml's `${VERSION:-1.0.0}-${ARCH:-arm64}`
# pattern. Inside the container neither var is set (compose only
# interpolates them on the host invocation), so we fall back to the same
# defaults plus a uname-based arch sniff.
_BACKENDS: Dict[str, Dict[str, str]] = {
    "lerobot": {
        "service": "lerobot",
        "container": "lerobot_server",
        "image": f"robotis/lerobot-zenoh:{_BACKEND_VERSION}-{_BACKEND_ARCH}",
    },
    "groot": {
        "service": "groot",
        "container": "groot_server",
        "image": f"robotis/groot-zenoh:{_BACKEND_VERSION}-{_BACKEND_ARCH}",
    },
}


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
    own_id = os.environ.get("HOSTNAME")
    if not own_id:
        return None
    try:
        ctr = _docker_client().containers.get(own_id)
    except DockerException as e:
        logger.warning("self-inspect failed: %s", e)
        return None
    for mount in ctr.attrs.get("Mounts", []):
        if mount.get("Destination") == _CYCLO_REPO_MOUNT:
            host_repo = mount.get("Source")
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


def _compose_base_cmd() -> List[str]:
    cmd = ["docker", "compose"]
    project_dir = _host_project_dir()
    if project_dir:
        cmd += ["--project-directory", project_dir]
    cmd += ["-f", _COMPOSE_FILE_IN_CONTAINER]
    return cmd


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
            pid_part = raw.split("(pid", 1)[1].split(")", 1)[0].strip()
            pid = int(pid_part)
        except ValueError:
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
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    container = os.environ.get("HOSTNAME", "unknown")
    s6_ready = os.path.isdir("/run/service")
    return HealthResponse(ok=True, container=container, s6_ready=s6_ready)


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
async def service_start(name: str) -> ActionResult:
    _require_known_service(name)
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
#   - start  → 'docker compose up -d --no-build <service>' (re-uses YAML).
#              No body — robot_type is now bound at InferenceCommand.LOAD
#              by the orchestrator, not at compose time, so policy
#              containers boot idle and configure themselves once Process
#              A receives LOAD with a robot_type.
#   - stop   → docker-py container.stop()/remove()
#   - status → docker-py images.get + containers.get


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

    # Pre-flight: image must already be local. compose up would otherwise
    # silently auto-pull a multi-GB image with no progress visible to the
    # UI — pull/ exists for exactly that reason.
    def _check_image() -> bool:
        try:
            _docker_client().images.get(spec["image"])
            return True
        except ImageNotFound:
            return False
        except DockerException:
            return False

    if not await asyncio.to_thread(_check_image):
        raise HTTPException(
            409,
            f"Image {spec['image']} not pulled. Call /backends/{name}/pull first.",
        )

    cmd = _compose_base_cmd() + ["up", "-d", "--no-build", spec["service"]]
    result = await _run(*cmd, timeout=60.0)
    ok = result.rc == 0
    msg = result.stderr or result.stdout or f"rc={result.rc}"
    return ActionResult(ok=ok, message=msg)


@app.post("/backends/{name}/stop", response_model=ActionResult)
async def backend_stop(name: str) -> ActionResult:
    spec = _require_known_backend(name)
    container_name = spec["container"]

    def _stop_and_remove() -> tuple[bool, str]:
        try:
            client = _docker_client()
        except DockerException as e:
            return False, f"docker init failed: {e}"
        try:
            ctr = client.containers.get(container_name)
        except NotFound:
            return True, f"{container_name} was not running"
        except DockerException as e:
            return False, f"inspect failed: {e}"
        try:
            ctr.stop(timeout=10)
            ctr.remove(force=True)
            return True, f"{container_name} stopped and removed"
        except DockerException as e:
            return False, f"stop/remove failed: {e}"

    ok, msg = await asyncio.to_thread(_stop_and_remove)
    return ActionResult(ok=ok, message=msg)


@app.get("/backends/{name}/status", response_model=BackendStatus)
async def backend_status(name: str) -> BackendStatus:
    spec = _require_known_backend(name)

    def _inspect():
        client = _docker_client()
        try:
            client.images.get(spec["image"])
            pulled = True
        except ImageNotFound:
            pulled = False
        except DockerException:
            pulled = False
        try:
            ctr = client.containers.get(spec["container"])
        except NotFound:
            return pulled, "not_created", None, None
        except DockerException as e:
            raise HTTPException(500, f"docker inspect failed: {e}")
        raw = ctr.attrs.get("State", {}).get("Status", "unknown")
        if raw == "running":
            mapped = "running"
        elif raw in ("exited", "dead", "created", "paused"):
            mapped = "exited"
        else:
            mapped = "unknown"
        return pulled, mapped, ctr.id, raw

    pulled, container_state, container_id, raw = await asyncio.to_thread(_inspect)
    return BackendStatus(
        name=name,
        image=spec["image"],
        image_pulled=pulled,
        container_state=container_state,
        container_id=container_id,
        raw_state=raw,
    )
