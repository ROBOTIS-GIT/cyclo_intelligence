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
#
# Author: Howon Kim

"""Self-contained Navigation control plane for cyclo_intelligence.

Most ROS topics stay on rosbridge. The two large Navigation grids use a
CRC32-filtered WebSocket so unchanged maps are not sent to browser clients.
"""

from __future__ import annotations

import base64
import asyncio
import io
import json
import os
from pathlib import PurePosixPath
import re
import shlex
import tarfile
import time
from typing import Literal, Optional

import docker
from docker.errors import DockerException, NotFound
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from supervisor_api.navigation_grid_cache import (
    GRID_CACHES,
    GRID_TOPICS,
    ensure_ros_grid_subscriber_started,
)


router = APIRouter(prefix="/navigation", tags=["navigation"])

AI_WORKER_CONTAINER = os.environ.get(
    "CYCLO_NAVIGATION_CONTAINER", "ai_worker"
)
NAVIGATION_SERVICE = "ai_worker_navigation"
MAP_SAVE_SERVICE = "ai_worker_map_save"
MAPS_DIR = PurePosixPath(
    "/root/ros2_ws/src/ai_worker/ffw_navigation/maps"
)
_SAFE_MAP_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
@router.websocket("/topics/ws")
async def navigation_grid_websocket(websocket: WebSocket, topic: str):
    """Send the latest grid initially, then only when its data CRC changes."""
    if topic not in GRID_TOPICS:
        await websocket.close(code=1008, reason="Unsupported grid topic")
        return

    ensure_ros_grid_subscriber_started()
    await websocket.accept()
    cache = GRID_CACHES[topic]
    listener_id = id(websocket)
    changed = asyncio.Event()
    cache.add_listener(listener_id, asyncio.get_running_loop(), changed)
    last_marker = None
    try:
        while True:
            changed.clear()
            last_marker, payload = cache.serialized_if_changed(last_marker)
            if payload is not None:
                await websocket.send_text(payload)
            change_task = asyncio.create_task(changed.wait())
            receive_task = asyncio.create_task(websocket.receive())
            done, pending = await asyncio.wait(
                {change_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if receive_task in done:
                event = receive_task.result()
                if event["type"] == "websocket.disconnect":
                    return
    except WebSocketDisconnect:
        return
    finally:
        cache.remove_listener(listener_id)


class NavigationStatus(BaseModel):
    is_up: bool
    pid: Optional[int] = None
    uptime_seconds: Optional[int] = None
    raw: str = ""


class NavigationStartRequest(BaseModel):
    mode: Literal["map", "nav"]
    map_name: str = Field(default="map", min_length=1, max_length=128)


class MapSaveRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)


class ActionResult(BaseModel):
    ok: bool
    message: str


class NavigateGoalRequest(BaseModel):
    pose: dict
    behavior_tree: str = ""


class PgmSaveRequest(BaseModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    maxval: int = Field(gt=0, le=255)
    pixels_base64: str


def _docker_client():
    try:
        return docker.from_env()
    except DockerException as exc:
        raise HTTPException(503, f"Docker is unavailable: {exc}") from exc


def _ai_worker():
    try:
        return _docker_client().containers.get(AI_WORKER_CONTAINER)
    except NotFound as exc:
        raise HTTPException(
            404, f"Container '{AI_WORKER_CONTAINER}' was not found"
        ) from exc
    except DockerException as exc:
        raise HTTPException(503, f"Failed to inspect ai_worker: {exc}") from exc


def _exec(command, *, environment=None, timeout=None):
    try:
        result = _ai_worker().exec_run(
            command,
            environment=environment,
            demux=False,
        )
    except DockerException as exc:
        raise HTTPException(503, f"ai_worker command failed: {exc}") from exc
    output = (result.output or b"").decode("utf-8", errors="replace").strip()
    return result.exit_code, output


def _validate_map_name(value: str) -> str:
    name = value.strip()
    if not _SAFE_MAP_NAME.fullmatch(name):
        raise HTTPException(
            400,
            "Map name may contain only letters, numbers, '.', '_' and '-'",
        )
    return name


def _read_container_file(path: PurePosixPath) -> bytes:
    try:
        stream, _ = _ai_worker().get_archive(str(path))
        archive = b"".join(stream)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            member = next((item for item in tar.getmembers() if item.isfile()), None)
            if member is None:
                raise FileNotFoundError(str(path))
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(str(path))
            return extracted.read()
    except NotFound as exc:
        raise FileNotFoundError(str(path)) from exc
    except DockerException as exc:
        raise HTTPException(503, f"Failed to read {path}: {exc}") from exc


def _write_container_file(path: PurePosixPath, content: bytes) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(path.name)
        info.mode = 0o644
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    try:
        if not _ai_worker().put_archive(str(path.parent), buf.getvalue()):
            raise HTTPException(500, f"Failed to write {path}")
    except DockerException as exc:
        raise HTTPException(503, f"Failed to write {path}: {exc}") from exc


def _write_runtime_file(path: str, content: str) -> None:
    code, output = _exec(["mkdir", "-p", str(PurePosixPath(path).parent)])
    if code != 0:
        raise HTTPException(503, output or f"Failed to create parent for {path}")
    _write_container_file(PurePosixPath(path), content.encode("utf-8"))


def _s6_command(service: str, action: Literal["up", "down", "restart"]):
    control = (
        'S6_RC=$(ls /package/admin/s6-*/command/s6-rc 2>/dev/null | head -1); '
        '[ -z "$S6_RC" ] && S6_RC=$(command -v s6-rc 2>/dev/null); '
        '[ -n "$S6_RC" ] || { echo "s6-rc not found"; exit 127; }; '
    )
    if action == "up":
        body = f'"$S6_RC" -u change {service}'
    elif action == "down":
        body = f'"$S6_RC" -d change {service}'
    else:
        body = (
            f'"$S6_RC" -d change {service}; '
            f'sleep 1; "$S6_RC" -u change {service}'
        )
    code, output = _exec(["sh", "-lc", control + body])
    if code != 0:
        raise HTTPException(
            503, output or f"Failed to {action} {service}"
        )
    return output or f"{service} {action} complete"


def _service_status(service: str) -> NavigationStatus:
    script = (
        'S6_SVSTAT=$(ls /package/admin/s6-*/command/s6-svstat '
        '2>/dev/null | head -1); '
        '[ -z "$S6_SVSTAT" ] && '
        'S6_SVSTAT=$(command -v s6-svstat 2>/dev/null); '
        '[ -n "$S6_SVSTAT" ] || { echo "s6-svstat not found"; exit 127; }; '
        f'"$S6_SVSTAT" /run/service/{service}'
    )
    code, raw = _exec(["sh", "-lc", script])
    if code != 0:
        return NavigationStatus(is_up=False, raw=raw)
    is_up = raw.startswith("up")
    pid_match = re.search(r"\(pid\s+(\d+)\)", raw)
    uptime_match = re.search(r"(\d+)\s+seconds", raw)
    return NavigationStatus(
        is_up=is_up,
        pid=int(pid_match.group(1)) if pid_match else None,
        uptime_seconds=int(uptime_match.group(1)) if uptime_match else None,
        raw=raw,
    )


def _normalize_path(path: PurePosixPath) -> PurePosixPath:
    parts = []
    for part in path.parts:
        if part in {"", "/", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath("/" + "/".join(parts))


def _resolve_pgm_path(value: str) -> PurePosixPath:
    raw = value.strip()
    if not raw:
        raise HTTPException(400, "PGM path must not be empty")
    candidate = PurePosixPath(raw)
    resolved = _normalize_path(
        candidate if candidate.is_absolute() else MAPS_DIR / candidate
    )
    if not str(resolved).startswith(str(MAPS_DIR) + "/"):
        raise HTTPException(400, "PGM path escapes the maps directory")
    if resolved.suffix.lower() != ".pgm":
        raise HTTPException(400, "Only .pgm files are supported")
    return resolved


def _relative_map_path(path: PurePosixPath) -> str:
    return str(path).removeprefix(str(MAPS_DIR) + "/")


def _skip_pgm_space(data: bytes, index: int) -> int:
    while index < len(data):
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in (10, 13):
                index += 1
        elif chr(data[index]).isspace():
            index += 1
        else:
            break
    return index


def _pgm_token(data: bytes, index: int):
    index = _skip_pgm_space(data, index)
    start = index
    while (
        index < len(data)
        and not chr(data[index]).isspace()
        and data[index] != ord("#")
    ):
        index += 1
    if start == index:
        raise ValueError("Unexpected end of PGM header")
    return data[start:index].decode("ascii"), index


def _parse_pgm(data: bytes):
    magic, index = _pgm_token(data, 0)
    width_token, index = _pgm_token(data, index)
    height_token, index = _pgm_token(data, index)
    maxval_token, index = _pgm_token(data, index)
    width, height, maxval = (
        int(width_token), int(height_token), int(maxval_token)
    )
    if magic not in {"P2", "P5"} or min(width, height, maxval) <= 0 or maxval > 255:
        raise ValueError("Only valid 8-bit P2/P5 PGM files are supported")
    count = width * height
    if magic == "P5":
        if index < len(data) and chr(data[index]).isspace():
            index += 1
        pixels = list(data[index:index + count])
    else:
        pixels = []
        for _ in range(count):
            token, index = _pgm_token(data, index)
            pixels.append(int(token))
    if len(pixels) != count or any(value < 0 or value > maxval for value in pixels):
        raise ValueError("PGM pixel data is invalid")
    return width, height, maxval, pixels


def _encode_pgm(width: int, height: int, maxval: int, pixels: list[int]) -> bytes:
    header = f"P5\n# fixed by cyclo_intelligence\n{width} {height}\n{maxval}\n"
    return header.encode("ascii") + bytes(pixels)


def _ros_shell_prefix() -> str:
    return (
        'for setup_file in /opt/ros/*/setup.bash; do '
        '[ -f "$setup_file" ] && source "$setup_file" && break; done; '
        '[ -f /root/ros2_ws/install/setup.bash ] && '
        'source /root/ros2_ws/install/setup.bash; '
    )


def _ros_exec_environment() -> dict[str, str]:
    """Keep one-shot ROS CLI processes on the server's ROS graph."""
    return {
        "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "30"),
        "RMW_IMPLEMENTATION": os.environ.get(
            "RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"
        ),
    }


@router.get("/status", response_model=NavigationStatus)
def navigation_status():
    return _service_status(NAVIGATION_SERVICE)


@router.post("/start", response_model=ActionResult)
def navigation_start(request: NavigationStartRequest):
    map_name = _validate_map_name(request.map_name)
    _write_runtime_file("/run/navigation_type", request.mode)
    _write_runtime_file(
        f"/run/launch_args/{NAVIGATION_SERVICE}",
        f"map_name:={map_name}",
    )
    message = _s6_command(NAVIGATION_SERVICE, "restart")
    return ActionResult(ok=True, message=message)


@router.post("/stop", response_model=ActionResult)
def navigation_stop():
    return ActionResult(
        ok=True,
        message=_s6_command(NAVIGATION_SERVICE, "down"),
    )


@router.post("/save-map", response_model=ActionResult)
def save_map(request: MapSaveRequest):
    map_name = _validate_map_name(request.map_name)
    _write_runtime_file(
        f"/run/launch_args/{MAP_SAVE_SERVICE}",
        f"map_name:={map_name}",
    )
    return ActionResult(
        ok=True,
        message=_s6_command(MAP_SAVE_SERVICE, "restart"),
    )


@router.post("/goal", response_model=ActionResult)
def send_goal(request: NavigateGoalRequest):
    pose = json.loads(json.dumps(request.pose))
    header = pose.setdefault("header", {})
    now_ns = time.time_ns()
    header["stamp"] = {
        "sec": now_ns // 1_000_000_000,
        "nanosec": now_ns % 1_000_000_000,
    }
    payload = json.dumps({
        "pose": pose,
        "behavior_tree": request.behavior_tree,
    })
    command = (
        _ros_shell_prefix()
        + "timeout 8s ros2 action send_goal "
        + "/navigate_to_pose nav2_msgs/action/NavigateToPose "
        + shlex.quote(payload)
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    accepted = code == 0 or (code == 124 and "Goal accepted" in output)
    if not accepted:
        raise HTTPException(503, output or "NavigateToPose goal failed")
    return ActionResult(ok=True, message=output or "Goal accepted")


@router.post("/cancel", response_model=ActionResult)
def cancel_goal():
    payload = json.dumps({
        "goal_info": {
            "goal_id": {"uuid": [0] * 16},
            "stamp": {"sec": 0, "nanosec": 0},
        }
    })
    command = (
        _ros_shell_prefix()
        + "timeout 8s ros2 service call "
        + "/navigate_to_pose/_action/cancel_goal action_msgs/srv/CancelGoal "
        + shlex.quote(payload)
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or "NavigateToPose cancel failed")
    return ActionResult(ok=True, message=output or "Goals cancelled")


@router.get("/logs")
def navigation_logs(
    tail: int = Query(default=300, ge=1, le=3000),
    cursor: Optional[int] = Query(default=None, ge=0),
):
    path = PurePosixPath(f"/var/log/{NAVIGATION_SERVICE}/current")
    try:
        data = _read_container_file(path)
    except FileNotFoundError:
        return {"logs": "", "cursor": 0, "log_path": str(path)}
    if cursor is None:
        lines = data.splitlines(keepends=True)[-tail:]
        content = b"".join(lines)
    else:
        if cursor > len(data):
            cursor = 0
        content = data[cursor:]
    return {
        "logs": content.decode("utf-8", errors="replace"),
        "cursor": len(data),
        "log_path": str(path),
    }


@router.delete("/logs", response_model=ActionResult)
def clear_navigation_logs():
    path = f"/var/log/{NAVIGATION_SERVICE}/current"
    code, output = _exec(["sh", "-lc", f"truncate -s 0 {shlex.quote(path)}"])
    if code != 0:
        raise HTTPException(503, output or "Failed to clear logs")
    return ActionResult(ok=True, message="Logs cleared")


@router.get("/maps/pgm-files")
def list_pgm_files():
    code, output = _exec([
        "find", str(MAPS_DIR), "-maxdepth", "4", "-type", "f", "-name", "*.pgm"
    ])
    if code != 0:
        raise HTTPException(503, output or "Failed to list PGM files")
    files = []
    for line in output.splitlines():
        path = _normalize_path(PurePosixPath(line.strip()))
        if str(path).startswith(str(MAPS_DIR) + "/") and path.suffix.lower() == ".pgm":
            files.append({"path": _relative_map_path(path), "name": path.name})
    files.sort(key=lambda item: item["path"])
    return {"files": files}


@router.get("/maps/pgm")
def get_pgm(path: str):
    resolved = _resolve_pgm_path(path)
    try:
        width, height, maxval, pixels = _parse_pgm(
            _read_container_file(resolved)
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, f"PGM file not found: {path}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "path": _relative_map_path(resolved),
        "width": width,
        "height": height,
        "maxval": maxval,
        "pixels_base64": base64.b64encode(bytes(pixels)).decode("ascii"),
    }


@router.post("/maps/pgm/save")
def save_pgm(request: PgmSaveRequest):
    resolved = _resolve_pgm_path(request.path)
    try:
        pixels = list(base64.b64decode(request.pixels_base64, validate=True))
    except Exception as exc:
        raise HTTPException(400, f"Invalid PGM pixel payload: {exc}") from exc
    if len(pixels) != request.width * request.height:
        raise HTTPException(400, "PGM payload size does not match dimensions")
    _write_container_file(
        resolved,
        _encode_pgm(request.width, request.height, request.maxval, pixels),
    )
    return {
        "path": _relative_map_path(resolved),
        "width": request.width,
        "height": request.height,
        "saved": True,
    }
