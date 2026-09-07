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
# Author: Howon Kim, Seongwoo Kim

"""Self-contained Navigation control plane for cyclo_intelligence.

Most ROS topics stay on rosbridge. The two large Navigation grids use a
cached WebSocket; the global costmap is sent once in full and then as dirty
rectangles, with automatic full resynchronization for lagging clients.
"""

from __future__ import annotations

import base64
import asyncio
import io
import json
import math
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
from supervisor_api.navigation_missions import remove_map_missions


router = APIRouter(prefix="/navigation", tags=["navigation"])

AI_WORKER_CONTAINER = os.environ.get(
    "CYCLO_NAVIGATION_CONTAINER", "ai_worker"
)
NAVIGATION_SERVICE = "ai_worker_navigation"
MAPS_DIR = PurePosixPath(
    "/root/ros2_ws/src/ai_worker/ffw_navigation/maps"
)
LOCALIZATION_PGID_FILE = "/run/cyclo_navigation_localization.pgid"
LOCALIZATION_LOG_FILE = "/var/log/cyclo_navigation_localization.log"
NAVIGATION_PARAMS_FILE = (
    "/root/ros2_ws/src/ai_worker/ffw_navigation/config/navigation.yaml"
)
MAP_SAVE_CLI_TIMEOUT_SECONDS = 20
SAVE_MAP_WAIT_SECONDS = 12.0
INITIAL_POSE_NOMOTION_BURST_COUNT = 8
INITIAL_POSE_NOMOTION_BURST_INTERVAL_SECONDS = 0.25
DESIGN_LOCALIZATION_AMCL_PARAMETERS = {
    "laser_likelihood_max_dist": 2.0,
    "max_beams": 80,
    "resample_interval": 1,
}
AMCL_PARAMETER_SET_TIMEOUT_SECONDS = 8.0
AMCL_PARAMETER_SET_RETRY_INTERVAL_SECONDS = 0.4
NAVIGATE_GOAL_ACCEPT_TIMEOUT_SECONDS = 8
NAVIGATE_GOAL_RESULT_TIMEOUT_SECONDS = 120
_SAFE_MAP_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_MAP_ANNOTATION_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NAVIGATE_TERMINAL_STATUS = re.compile(
    r"Goal finished with status:\s*([A-Z_]+)",
    re.IGNORECASE,
)


@router.websocket("/topics/ws")
async def navigation_grid_websocket(websocket: WebSocket, topic: str):
    """Send an initial grid, then changed costmap regions or a full resync."""
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
    mode: Optional[str] = None


class NavigationStartRequest(BaseModel):
    mode: Literal["map", "nav", "localize"]
    map_name: str = Field(default="map", min_length=1, max_length=128)


class MapSaveRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)


class ActionResult(BaseModel):
    ok: bool
    message: str


class NavigateGoalRequest(BaseModel):
    pose: dict
    behavior_tree: str = ""


class NavigateThroughPosesRequest(BaseModel):
    poses: list[dict] = Field(min_length=2, max_length=3)
    behavior_tree: str = ""


class NavigateGoalResult(ActionResult):
    status: Literal[
        "SUCCEEDED",
        "ABORTED",
        "CANCELED",
        "TIMEOUT",
        "REJECTED",
        "UNKNOWN",
    ]


class InitialPoseRequest(BaseModel):
    x: float
    y: float
    yaw: float
    frame_id: str = Field(default="map", min_length=1, max_length=64)
    map_name: Optional[str] = Field(default=None, min_length=1, max_length=128)


class PgmSaveRequest(BaseModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    maxval: int = Field(gt=0, le=255)
    pixels_base64: str


class MapAnnotationPose(BaseModel):
    frame_id: str = Field(default="map", min_length=1, max_length=64)
    x: float
    y: float
    yaw: float = 0.0


class MapAnnotationSeedCell(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class MapAnnotationBounds(BaseModel):
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class MapAnnotationRegion(BaseModel):
    seed_cell: MapAnnotationSeedCell
    bounds: Optional[MapAnnotationBounds] = None
    cells: Optional[list[MapAnnotationSeedCell]] = None
    cell_count: Optional[int] = Field(default=None, ge=0)
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)


class MapAnnotation(BaseModel):
    id: str = Field(min_length=1, max_length=96)
    label: str = Field(min_length=1, max_length=80)
    color: str = Field(min_length=4, max_length=9)
    pose: MapAnnotationPose
    region: Optional[MapAnnotationRegion] = None


class MapAnnotationsSaveRequest(BaseModel):
    path: str
    annotations: list[MapAnnotation] = Field(default_factory=list)


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


def _map_artifact_paths(map_name: str) -> tuple[PurePosixPath, PurePosixPath]:
    return (
        MAPS_DIR / f"{map_name}.yaml",
        MAPS_DIR / f"{map_name}.pgm",
    )


def _container_file_signature(path: PurePosixPath) -> Optional[str]:
    code, output = _exec(["stat", "-c", "%s:%Y:%y", str(path)])
    return output if code == 0 else None


def _map_artifact_signatures(map_name: str) -> dict[str, Optional[str]]:
    yaml_path, pgm_path = _map_artifact_paths(map_name)
    return {
        "yaml": _container_file_signature(yaml_path),
        "pgm": _container_file_signature(pgm_path),
    }


def _saved_map_ready(
    previous: dict[str, Optional[str]],
    current: dict[str, Optional[str]],
) -> bool:
    if not current["yaml"] or not current["pgm"]:
        return False
    return (
        current["yaml"] != previous.get("yaml")
        or current["pgm"] != previous.get("pgm")
    )


def _wait_for_saved_map(
    map_name: str,
    previous: dict[str, Optional[str]],
) -> dict[str, Optional[str]]:
    deadline = time.monotonic() + SAVE_MAP_WAIT_SECONDS
    while time.monotonic() <= deadline:
        current = _map_artifact_signatures(map_name)
        if _saved_map_ready(previous, current):
            return current
        time.sleep(0.25)
    yaml_path, pgm_path = _map_artifact_paths(map_name)
    raise HTTPException(
        503,
        (
            "Map save did not create or update expected files: "
            f"{yaml_path.name}, {pgm_path.name}"
        ),
    )


def _s6_command(
    service: str,
    action: Literal["up", "down", "restart"],
    *,
    retries: int = 0,
    retry_delay: float = 0.2,
):
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
    for attempt in range(retries + 1):
        code, output = _exec(["sh", "-lc", control + body])
        if code == 0:
            return output or f"{service} {action} complete"
        if "unable to take locks" in output and attempt < retries:
            time.sleep(retry_delay)
            continue
        raise HTTPException(
            503, output or f"Failed to {action} {service}"
        )
    return f"{service} {action} complete"


def _sync_s6_service_down_best_effort(service: str) -> str:
    try:
        return _s6_command(service, "down", retries=15, retry_delay=0.2)
    except HTTPException as exc:
        detail = str(exc.detail)
        if "unable to take locks" not in detail:
            raise
        return f"{service} down sync deferred: {detail}"


def _request_s6_service_down(service: str) -> str:
    """Ask s6-supervise to stop a service without waiting for full teardown."""
    script = (
        'S6_SVC=$(ls /package/admin/s6-*/command/s6-svc 2>/dev/null | head -1); '
        '[ -z "$S6_SVC" ] && S6_SVC=$(command -v s6-svc 2>/dev/null); '
        '[ -n "$S6_SVC" ] || { echo "s6-svc not found"; exit 127; }; '
        f'"$S6_SVC" -d /run/service/{service}'
    )
    code, output = _exec(["sh", "-lc", script])
    if code != 0:
        raise HTTPException(
            503,
            output or f"Failed to request {service} down",
        )
    return output or f"{service} down requested"


def _clear_navigation_runtime_files() -> None:
    code, output = _exec(
        [
            "rm",
            "-f",
            "/run/navigation_type",
            f"/run/launch_args/{NAVIGATION_SERVICE}",
            LOCALIZATION_PGID_FILE,
        ]
    )
    if code != 0:
        raise HTTPException(
            503,
            output or "Failed to clear navigation runtime files",
        )


def _stop_localization_processes() -> str:
    script = f"""
set +e
PGID_FILE={shlex.quote(LOCALIZATION_PGID_FILE)}
if [ -f "${{PGID_FILE}}" ]; then
  PGID=$(cat "${{PGID_FILE}}" 2>/dev/null | tr -d " " || true)
  case "${{PGID}}" in
    ""|*[!0-9]*|1)
      echo "[localization stop] Ignoring invalid process group '${{PGID}}'"
      ;;
    *)
      echo "[localization stop] Terminating process group ${{PGID}}"
      kill -TERM -"${{PGID}}" 2>/dev/null || true
      sleep 1
      kill -KILL -"${{PGID}}" 2>/dev/null || true
      ;;
  esac
  rm -f "${{PGID_FILE}}"
fi
exit 0
"""
    code, output = _exec(["sh", "-lc", script])
    if code != 0:
        raise HTTPException(
            503,
            output or "Failed to stop localization processes",
        )
    return output


def _force_stop_navigation_processes() -> str:
    script = f"""
set +e
SERVICE={shlex.quote(NAVIGATION_SERVICE)}
PGID_FILE=/run/${{SERVICE}}.pgid
if [ -f "${{PGID_FILE}}" ]; then
  PGID=$(cat "${{PGID_FILE}}" 2>/dev/null | tr -d " " || true)
  case "${{PGID}}" in
    ""|*[!0-9]*|1)
      echo "[navigation stop] Ignoring invalid process group '${{PGID}}'"
      ;;
    *)
      echo "[navigation stop] Terminating process group ${{PGID}}"
      kill -TERM -"${{PGID}}" 2>/dev/null || true
      sleep 1
      kill -KILL -"${{PGID}}" 2>/dev/null || true
      ;;
  esac
  rm -f "${{PGID_FILE}}"
fi
PIDS=$(ps -eo pid=,comm=,args= | awk -v self="$$" '
  $1 == self {{ next }}
  $2 ~ /^(awk|sh|bash|ps)$/ {{ next }}
  /ros2 launch ffw_navigation/ ||
  /ros2 launch nav2_bringup/ ||
  /\\/slam_toolbox\\// ||
  /\\/nav2_/ ||
  /\\/opennav_docking/ {{
    print $1
  }}
')
if [ -n "${{PIDS}}" ]; then
  echo "[navigation stop] Terminating leftover navigation PIDs: ${{PIDS}}"
  for PID in ${{PIDS}}; do
    kill -TERM "${{PID}}" 2>/dev/null || true
  done
fi
sleep 1
if [ -n "${{PIDS}}" ]; then
  for PID in ${{PIDS}}; do
    kill -KILL "${{PID}}" 2>/dev/null || true
  done
fi
exit 0
"""
    code, output = _exec(["sh", "-lc", script])
    if code != 0:
        raise HTTPException(
            503,
            output or "Failed to force stop navigation processes",
        )
    localization_output = _stop_localization_processes()
    return "\n".join(
        part for part in [output, localization_output] if part
    )


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
        mode=_runtime_mode() if is_up else None,
    )


def _runtime_mode() -> Optional[str]:
    code, output = _exec(["cat", "/run/navigation_type"])
    if code != 0:
        return None
    mode = output.strip().lower()
    return mode or None


def _localization_status() -> NavigationStatus:
    script = f"""
PGID_FILE={shlex.quote(LOCALIZATION_PGID_FILE)}
if [ ! -f "${{PGID_FILE}}" ]; then
  echo "down"
  exit 3
fi
PGID=$(cat "${{PGID_FILE}}" 2>/dev/null | tr -d " " || true)
case "${{PGID}}" in
  ""|*[!0-9]*|1)
    echo "down invalid pgid"
    exit 3
    ;;
esac
if kill -0 -"${{PGID}}" 2>/dev/null; then
  ELAPSED=$(ps -o etimes= -p "${{PGID}}" 2>/dev/null | tr -d " " || true)
  echo "up (pid ${{PGID}} pgid ${{PGID}}) ${{ELAPSED:-0}} seconds"
  exit 0
fi
echo "down"
exit 3
"""
    code, raw = _exec(["sh", "-lc", script])
    if code != 0:
        return NavigationStatus(is_up=False, raw=raw)
    pid_match = re.search(r"\(pid\s+(\d+)", raw)
    uptime_match = re.search(r"(\d+)\s+seconds", raw)
    return NavigationStatus(
        is_up=True,
        pid=int(pid_match.group(1)) if pid_match else None,
        uptime_seconds=int(uptime_match.group(1)) if uptime_match else None,
        raw=raw,
        mode="localize",
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


def _resolve_map_annotations_path(pgm_path: PurePosixPath) -> PurePosixPath:
    return pgm_path.with_suffix(".annotations.json")


def _relative_map_path(path: PurePosixPath) -> str:
    return str(path).removeprefix(str(MAPS_DIR) + "/")


def _annotation_payload(annotation: MapAnnotation) -> dict[str, object]:
    annotation_id = annotation.id.strip()
    if not annotation_id:
        raise HTTPException(400, "Annotation id must not be empty")
    label = annotation.label.strip() or "Area"
    color = annotation.color.strip()
    if not _SAFE_MAP_ANNOTATION_COLOR.fullmatch(color):
        raise HTTPException(400, "Annotation color must be a #RRGGBB value")
    payload = {
        "id": annotation_id,
        "label": label[:80],
        "color": color,
        "pose": {
            "frame_id": annotation.pose.frame_id.strip() or "map",
            "x": float(annotation.pose.x),
            "y": float(annotation.pose.y),
            "yaw": float(annotation.pose.yaw),
        },
    }
    if annotation.region is not None:
        bounds = annotation.region.bounds
        payload["region"] = {
            "seed_cell": {
                "x": int(annotation.region.seed_cell.x),
                "y": int(annotation.region.seed_cell.y),
            },
            "bounds": {
                "x_min": int(bounds.x_min),
                "y_min": int(bounds.y_min),
                "x_max": int(bounds.x_max),
                "y_max": int(bounds.y_max),
            } if bounds is not None else None,
            "cells": [
                {"x": int(cell.x), "y": int(cell.y)}
                for cell in (annotation.region.cells or [])
            ] or None,
            "cell_count": annotation.region.cell_count,
            "width": annotation.region.width,
            "height": annotation.region.height,
        }
    return payload


def _annotation_from_raw(raw: object) -> Optional[dict[str, object]]:
    if not isinstance(raw, dict):
        return None
    pose = raw.get("pose")
    if not isinstance(pose, dict):
        return None
    region = raw.get("region")
    seed_cell = region.get("seed_cell") if isinstance(region, dict) else None
    bounds = region.get("bounds") if isinstance(region, dict) else None
    cells = region.get("cells") if isinstance(region, dict) else None
    try:
        annotation = MapAnnotation(
            id=str(raw.get("id") or ""),
            label=str(raw.get("label") or "Area"),
            color=str(raw.get("color") or "#C96442"),
            pose=MapAnnotationPose(
                frame_id=str(pose.get("frame_id") or "map"),
                x=float(pose.get("x")),
                y=float(pose.get("y")),
                yaw=float(pose.get("yaw") or 0.0),
            ),
            region=MapAnnotationRegion(
                seed_cell=MapAnnotationSeedCell(
                    x=int(seed_cell.get("x")),
                    y=int(seed_cell.get("y")),
                ),
                bounds=MapAnnotationBounds(
                    x_min=int(bounds.get("x_min")),
                    y_min=int(bounds.get("y_min")),
                    x_max=int(bounds.get("x_max")),
                    y_max=int(bounds.get("y_max")),
                ) if isinstance(bounds, dict) else None,
                cells=[
                    MapAnnotationSeedCell(
                        x=int(cell.get("x")),
                        y=int(cell.get("y")),
                    )
                    for cell in cells
                    if isinstance(cell, dict)
                ] if isinstance(cells, list) else None,
                cell_count=(
                    int(region.get("cell_count"))
                    if region.get("cell_count") is not None
                    else None
                ),
                width=(
                    int(region.get("width"))
                    if region.get("width") is not None
                    else None
                ),
                height=(
                    int(region.get("height"))
                    if region.get("height") is not None
                    else None
                ),
            ) if isinstance(region, dict) and isinstance(seed_cell, dict) else None,
        )
    except Exception:
        return None
    return _annotation_payload(annotation)


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


def _orientation_from_yaw(yaw: float) -> dict[str, float]:
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(yaw / 2.0),
        "w": math.cos(yaw / 2.0),
    }


def _parse_map_yaml_metadata(data: bytes) -> dict[str, object]:
    text = data.decode("utf-8", errors="replace")
    resolution_match = re.search(
        r"(?m)^\s*resolution\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:#.*)?$",
        text,
    )
    origin_match = re.search(r"(?m)^\s*origin\s*:\s*\[([^\]]+)\]", text)
    metadata: dict[str, object] = {}
    if resolution_match:
        metadata["resolution"] = float(resolution_match.group(1))
    if origin_match:
        values = [
            float(value)
            for value in re.findall(
                r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                origin_match.group(1),
            )
        ]
        if len(values) >= 3:
            metadata["origin"] = {
                "position": {"x": values[0], "y": values[1], "z": 0.0},
                "orientation": _orientation_from_yaw(values[2]),
            }
    return metadata


def _map_yaml_metadata_for_pgm(path: PurePosixPath) -> dict[str, object]:
    yaml_path = path.with_suffix(".yaml")
    try:
        return _parse_map_yaml_metadata(_read_container_file(yaml_path))
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise HTTPException(422, f"Invalid map YAML metadata: {exc}") from exc


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
            "RMW_IMPLEMENTATION", "rmw_zenoh_cpp"
        ),
    }


def _save_map_with_cli(map_name: str) -> str:
    output_prefix = MAPS_DIR / map_name
    command = (
        _ros_shell_prefix()
        + f"timeout {MAP_SAVE_CLI_TIMEOUT_SECONDS}s "
        + "ros2 run nav2_map_server map_saver_cli "
        + f"-f {shlex.quote(str(output_prefix))} "
        + "--ros-args "
        + "-p map_subscribe_transient_local:=true "
        + "-p save_map_timeout:=10.0"
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or "Map saver failed")
    return output or "map_saver_cli complete"


def _start_localization_process(map_name: str) -> str:
    map_path = MAPS_DIR / f"{map_name}.yaml"
    ros_command = (
        _ros_shell_prefix()
        + "exec ros2 launch nav2_bringup localization_launch.py "
        + f"use_sim_time:=false "
        + f"map:={shlex.quote(str(map_path))} "
        + f"params_file:={shlex.quote(NAVIGATION_PARAMS_FILE)}"
    )
    script = f"""
set -e
MAP_PATH={shlex.quote(str(map_path))}
PARAMS_FILE={shlex.quote(NAVIGATION_PARAMS_FILE)}
LOG_FILE={shlex.quote(LOCALIZATION_LOG_FILE)}
PGID_FILE={shlex.quote(LOCALIZATION_PGID_FILE)}
[ -f "${{MAP_PATH}}" ] || {{ echo "Map file not found: ${{MAP_PATH}}"; exit 2; }}
[ -f "${{PARAMS_FILE}}" ] || {{ echo "Params file not found: ${{PARAMS_FILE}}"; exit 2; }}
mkdir -p "$(dirname "${{LOG_FILE}}")"
rm -f "${{PGID_FILE}}"
: > "${{LOG_FILE}}"
nohup setsid bash --noprofile --norc -lc {shlex.quote(ros_command)} \
  >> "${{LOG_FILE}}" 2>&1 < /dev/null &
PID=$!
echo "${{PID}}" > "${{PGID_FILE}}"
sleep 0.8
if ! kill -0 -- -"${{PID}}" 2>/dev/null; then
  cat "${{LOG_FILE}}" 2>/dev/null || true
  rm -f "${{PGID_FILE}}"
  exit 1
fi
echo "localization launched pid=${{PID}} map=${{MAP_PATH}}"
"""
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", script],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or "Localization launch failed")
    return output or "Localization launched"


def _start_localization_mode(map_name: str) -> str:
    _request_s6_service_down(NAVIGATION_SERVICE)
    _force_stop_navigation_processes()
    _s6_command(NAVIGATION_SERVICE, "down", retries=15, retry_delay=0.2)
    _clear_navigation_runtime_files()
    _write_runtime_file("/run/navigation_type", "localize")
    _write_runtime_file(
        f"/run/launch_args/{NAVIGATION_SERVICE}",
        f"map_name:={map_name}",
    )
    return _start_localization_process(map_name)


def _initialpose_subscription_count() -> int:
    command = (
        _ros_shell_prefix()
        + "timeout 5s ros2 topic info /initialpose"
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        return 0
    match = re.search(r"Subscription count:\s*(\d+)", output)
    return int(match.group(1)) if match else 0


def _finite_pose_value(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise HTTPException(400, f"{name} must be finite")
    return number


def _safe_frame_id(value: str) -> str:
    frame_id = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_/.-]+", frame_id):
        raise HTTPException(
            400,
            "Frame id may contain only letters, numbers, '/', '.', '_' and '-'",
        )
    return frame_id


def _strip_ansi(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value)


def _initial_pose_success_message(output: str) -> str:
    cleaned = _strip_ansi(output or "")
    for line in reversed(cleaned.splitlines()):
        text = line.strip()
        if text.startswith("Published initial pose to /initialpose"):
            return text
    return "Published initial pose"


def _publish_initial_pose(request: InitialPoseRequest) -> str:
    x = _finite_pose_value(request.x, "x")
    y = _finite_pose_value(request.y, "y")
    yaw = _finite_pose_value(request.yaw, "yaw")
    frame_id = _safe_frame_id(request.frame_id)
    payload = {
        "frame_id": frame_id,
        "x": x,
        "y": y,
        "yaw": yaw,
        "orientation": _orientation_from_yaw(yaw),
        "nomotion_update_count": INITIAL_POSE_NOMOTION_BURST_COUNT,
        "nomotion_update_interval": INITIAL_POSE_NOMOTION_BURST_INTERVAL_SECONDS,
        "covariance": [
            0.25, 0, 0, 0, 0, 0,
            0, 0.25, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0.0685,
        ],
    }
    script = f"""
import json
import time

import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty

payload = json.loads({json.dumps(json.dumps(payload))})

rclpy.init()
node = rclpy.create_node("cyclo_initial_pose_publisher")
publisher = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
nomotion_client = node.create_client(Empty, "/request_nomotion_update")

deadline = time.monotonic() + 8.0
while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

subscriber_count = publisher.get_subscription_count()
if subscriber_count == 0:
    node.destroy_node()
    rclpy.shutdown()
    print("No AMCL subscriber found on /initialpose")
    raise SystemExit(2)

message = PoseWithCovarianceStamped()
message.header.frame_id = payload["frame_id"]
message.pose.pose.position.x = payload["x"]
message.pose.pose.position.y = payload["y"]
message.pose.pose.position.z = 0.0
message.pose.pose.orientation.x = payload["orientation"]["x"]
message.pose.pose.orientation.y = payload["orientation"]["y"]
message.pose.pose.orientation.z = payload["orientation"]["z"]
message.pose.pose.orientation.w = payload["orientation"]["w"]
message.pose.covariance = payload["covariance"]

for _ in range(5):
    message.header.stamp = (
        node.get_clock().now() - Duration(seconds=0.2)
    ).to_msg()
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.1)

nomotion_update = "unavailable"
nomotion_attempts = max(1, int(payload.get("nomotion_update_count", 1)))
nomotion_interval = max(0.0, float(payload.get("nomotion_update_interval", 0.0)))
if nomotion_client.wait_for_service(timeout_sec=2.0):
    successes = 0
    failures = 0
    for attempt in range(nomotion_attempts):
        future = nomotion_client.call_async(Empty.Request())
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if future.done() and future.exception() is None:
            successes += 1
        else:
            failures += 1
            if future.done():
                print(f"No-motion AMCL update failed: {{future.exception()}}")
            else:
                print("No-motion AMCL update timed out")
        if attempt + 1 < nomotion_attempts and nomotion_interval > 0:
            time.sleep(nomotion_interval)
    if successes == nomotion_attempts:
        nomotion_update = f"requested:{{successes}}"
    elif successes > 0:
        nomotion_update = f"partial:{{successes}}/{{nomotion_attempts}}"
    elif failures:
        nomotion_update = "failed"
else:
    print("No AMCL /request_nomotion_update service found")

node.destroy_node()
rclpy.shutdown()
print(
    f"Published initial pose to /initialpose "
    f"({{payload['x']:.3f}}, {{payload['y']:.3f}}, yaw={{payload['yaw']:.3f}}, "
    f"subscribers={{subscriber_count}}, nomotion_update={{nomotion_update}})"
)
"""
    timeout_seconds = max(
        12,
        math.ceil(
            6
            + INITIAL_POSE_NOMOTION_BURST_COUNT
            * (1.0 + INITIAL_POSE_NOMOTION_BURST_INTERVAL_SECONDS)
        ),
    )
    command = (
        _ros_shell_prefix()
        + f"timeout {timeout_seconds}s python3 -c "
        + shlex.quote(script)
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or "Initial pose publish failed")
    return _initial_pose_success_message(output)


def _request_nomotion_update() -> str:
    command = (
        _ros_shell_prefix()
        + "timeout 8s ros2 service call "
        + "/request_nomotion_update std_srvs/srv/Empty "
        + shlex.quote("{}")
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or "No-motion AMCL update failed")
    return output or "No-motion AMCL update requested"


def _request_global_localization() -> str:
    command = (
        _ros_shell_prefix()
        + "timeout 8s ros2 service call "
        + "/reinitialize_global_localization std_srvs/srv/Empty "
        + shlex.quote("{}")
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or "Global AMCL localization failed")
    return output or "Global AMCL localization requested"


def _format_ros_param_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _set_amcl_parameters(parameters: dict[str, object]) -> str:
    applied: list[str] = []
    for name, value in parameters.items():
        command = (
            _ros_shell_prefix()
            + "timeout 8s ros2 param set /amcl "
            + f"{shlex.quote(name)} "
            + shlex.quote(_format_ros_param_value(value))
        )
        deadline = time.monotonic() + AMCL_PARAMETER_SET_TIMEOUT_SECONDS
        last_output = ""
        while True:
            code, output = _exec(
                ["bash", "--noprofile", "--norc", "-c", command],
                environment=_ros_exec_environment(),
            )
            if code == 0:
                applied.append(f"{name}={_format_ros_param_value(value)}")
                break
            last_output = output
            if time.monotonic() >= deadline:
                raise HTTPException(
                    503,
                    last_output or f"Failed to set AMCL parameter: {name}",
                )
            time.sleep(AMCL_PARAMETER_SET_RETRY_INTERVAL_SECONDS)
    return "Applied AMCL parameters: " + ", ".join(applied)


@router.get("/status", response_model=NavigationStatus)
def navigation_status():
    service_status = _service_status(NAVIGATION_SERVICE)
    if service_status.is_up:
        return service_status
    localization_status = _localization_status()
    if localization_status.is_up:
        return localization_status
    return service_status


@router.post("/start", response_model=ActionResult)
def navigation_start(request: NavigationStartRequest):
    map_name = _validate_map_name(request.map_name)
    if request.mode == "localize":
        message = _start_localization_mode(map_name)
        return ActionResult(ok=True, message=message)
    _request_s6_service_down(NAVIGATION_SERVICE)
    _force_stop_navigation_processes()
    _s6_command(NAVIGATION_SERVICE, "down", retries=15, retry_delay=0.2)
    _clear_navigation_runtime_files()
    GRID_CACHES["/global_costmap/costmap"].clear()
    if request.mode == "map":
        GRID_CACHES["/map"].clear()
    # Subscribe before Nav2 activates so the initial transient-local full
    # costmap is available before subsequent dirty-rectangle updates arrive.
    ensure_ros_grid_subscriber_started()
    _write_runtime_file("/run/navigation_type", request.mode)
    _write_runtime_file(
        f"/run/launch_args/{NAVIGATION_SERVICE}",
        f"map_name:={map_name}",
    )
    message = _s6_command(NAVIGATION_SERVICE, "up")
    return ActionResult(ok=True, message=message)


@router.post("/stop", response_model=ActionResult)
def navigation_stop():
    message = _request_s6_service_down(NAVIGATION_SERVICE)
    force_message = _force_stop_navigation_processes()
    sync_message = _sync_s6_service_down_best_effort(NAVIGATION_SERVICE)
    _clear_navigation_runtime_files()
    GRID_CACHES["/global_costmap/costmap"].clear()
    if force_message:
        message = f"{message}\n{force_message}"
    if sync_message:
        message = f"{message}\n{sync_message}"
    return ActionResult(
        ok=True,
        message=message,
    )


@router.post("/save-map", response_model=ActionResult)
def save_map(request: MapSaveRequest):
    map_name = _validate_map_name(request.map_name)
    previous = _map_artifact_signatures(map_name)
    _save_map_with_cli(map_name)
    _wait_for_saved_map(map_name, previous)
    return ActionResult(
        ok=True,
        message=f"Saved map '{map_name}' as {map_name}.yaml and {map_name}.pgm",
    )


def _stamped_pose(pose_value: dict) -> dict:
    pose = json.loads(json.dumps(pose_value))
    header = pose.setdefault("header", {})
    now_ns = time.time_ns()
    header["stamp"] = {
        "sec": now_ns // 1_000_000_000,
        "nanosec": now_ns % 1_000_000_000,
    }
    return pose


def _navigate_goal_payload(request: NavigateGoalRequest) -> str:
    return json.dumps({
        "pose": _stamped_pose(request.pose),
        "behavior_tree": request.behavior_tree,
    })


def _navigate_through_poses_payload(
    request: NavigateThroughPosesRequest,
) -> str:
    return json.dumps({
        "poses": [_stamped_pose(pose) for pose in request.poses],
        "behavior_tree": request.behavior_tree,
    })


def _navigate_goal_command(payload: str, timeout_seconds: int) -> list[str]:
    command = (
        _ros_shell_prefix()
        + f"timeout {timeout_seconds}s ros2 action send_goal "
        + "/navigate_to_pose nav2_msgs/action/NavigateToPose "
        + shlex.quote(payload)
    )
    return ["bash", "--noprofile", "--norc", "-c", command]


def _navigate_through_poses_command(
    payload: str,
    timeout_seconds: int,
) -> list[str]:
    command = (
        _ros_shell_prefix()
        + f"timeout {timeout_seconds}s ros2 action send_goal "
        + "/navigate_through_poses nav2_msgs/action/NavigateThroughPoses "
        + shlex.quote(payload)
    )
    return ["bash", "--noprofile", "--norc", "-c", command]


def _navigate_terminal_status(output: str) -> str:
    match = _NAVIGATE_TERMINAL_STATUS.search(output)
    if match:
        status = match.group(1).upper()
        if status in {"SUCCEEDED", "ABORTED", "CANCELED"}:
            return status
    if "Goal was rejected" in output:
        return "REJECTED"
    return "UNKNOWN"


def _cancel_all_action_goals(
    action_name: str,
    action_label: str,
) -> str:
    payload = json.dumps({
        "goal_info": {
            "goal_id": {"uuid": [0] * 16},
            "stamp": {"sec": 0, "nanosec": 0},
        }
    })
    command = (
        _ros_shell_prefix()
        + "timeout 8s ros2 service call "
        + f"{action_name}/_action/cancel_goal action_msgs/srv/CancelGoal "
        + shlex.quote(payload)
    )
    code, output = _exec(
        ["bash", "--noprofile", "--norc", "-c", command],
        environment=_ros_exec_environment(),
    )
    if code != 0:
        raise HTTPException(503, output or f"{action_label} cancel failed")
    return output or "Goals cancelled"


def _cancel_all_navigate_goals() -> str:
    return _cancel_all_action_goals(
        "/navigate_to_pose",
        "NavigateToPose",
    )


def _cancel_all_navigate_through_poses_goals() -> str:
    return _cancel_all_action_goals(
        "/navigate_through_poses",
        "NavigateThroughPoses",
    )


def _cancel_all_navigation_goals() -> str:
    messages = []
    errors = []
    for cancel in (
        _cancel_all_navigate_goals,
        _cancel_all_navigate_through_poses_goals,
    ):
        try:
            messages.append(cancel())
        except HTTPException as exc:
            errors.append(str(exc.detail))
    if errors:
        raise HTTPException(503, "; ".join(errors))
    return "\n".join(messages) or "Goals cancelled"


@router.post("/goal", response_model=ActionResult)
def send_goal(request: NavigateGoalRequest):
    payload = _navigate_goal_payload(request)
    code, output = _exec(
        _navigate_goal_command(payload, NAVIGATE_GOAL_ACCEPT_TIMEOUT_SECONDS),
        environment=_ros_exec_environment(),
    )
    accepted = code == 0 or (code == 124 and "Goal accepted" in output)
    if not accepted:
        raise HTTPException(503, output or "NavigateToPose goal failed")
    return ActionResult(ok=True, message=output or "Goal accepted")


@router.post("/goal/wait", response_model=NavigateGoalResult)
def send_goal_and_wait(request: NavigateGoalRequest):
    payload = _navigate_goal_payload(request)
    code, output = _exec(
        _navigate_goal_command(payload, NAVIGATE_GOAL_RESULT_TIMEOUT_SECONDS),
        environment=_ros_exec_environment(),
    )
    if code == 124:
        if "Goal accepted" in output:
            _cancel_all_navigate_goals()
            return NavigateGoalResult(
                ok=False,
                status="TIMEOUT",
                message=output or "NavigateToPose goal timed out",
            )
        raise HTTPException(503, output or "NavigateToPose action server timed out")

    status = _navigate_terminal_status(output)
    if status == "REJECTED":
        return NavigateGoalResult(ok=False, status=status, message=output)
    if code != 0:
        raise HTTPException(503, output or "NavigateToPose goal failed")
    return NavigateGoalResult(
        ok=status == "SUCCEEDED",
        status=status,
        message=output or f"NavigateToPose finished with status {status}",
    )


@router.post("/goals/wait", response_model=NavigateGoalResult)
def send_goals_and_wait(request: NavigateThroughPosesRequest):
    payload = _navigate_through_poses_payload(request)
    timeout_seconds = NAVIGATE_GOAL_RESULT_TIMEOUT_SECONDS * len(request.poses)
    code, output = _exec(
        _navigate_through_poses_command(payload, timeout_seconds),
        environment=_ros_exec_environment(),
    )
    if code == 124:
        if "Goal accepted" in output:
            _cancel_all_navigate_through_poses_goals()
            return NavigateGoalResult(
                ok=False,
                status="TIMEOUT",
                message=output or "NavigateThroughPoses goal timed out",
            )
        raise HTTPException(
            503,
            output or "NavigateThroughPoses action server timed out",
        )

    status = _navigate_terminal_status(output)
    if status == "REJECTED":
        return NavigateGoalResult(ok=False, status=status, message=output)
    if code != 0:
        raise HTTPException(
            503,
            output or "NavigateThroughPoses goal failed",
        )
    return NavigateGoalResult(
        ok=status == "SUCCEEDED",
        status=status,
        message=(
            output
            or f"NavigateThroughPoses finished with status {status}"
        ),
    )


@router.post("/cancel", response_model=ActionResult)
def cancel_goal():
    return ActionResult(ok=True, message=_cancel_all_navigation_goals())


@router.post("/initial-pose", response_model=ActionResult)
def send_initial_pose(request: InitialPoseRequest):
    if request.map_name and _initialpose_subscription_count() == 0:
        _start_localization_mode(_validate_map_name(request.map_name))
    return ActionResult(ok=True, message=_publish_initial_pose(request))


@router.post("/nomotion-update", response_model=ActionResult)
def request_nomotion_update():
    return ActionResult(ok=True, message=_request_nomotion_update())


@router.post("/global-localization", response_model=ActionResult)
def request_global_localization():
    return ActionResult(ok=True, message=_request_global_localization())


@router.post("/amcl/design-localization-params", response_model=ActionResult)
def set_design_localization_amcl_parameters():
    return ActionResult(
        ok=True,
        message=_set_amcl_parameters(DESIGN_LOCALIZATION_AMCL_PARAMETERS),
    )


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
        **_map_yaml_metadata_for_pgm(resolved),
        "pixels_base64": base64.b64encode(bytes(pixels)).decode("ascii"),
    }


@router.get("/maps/annotations")
def get_map_annotations(path: str):
    resolved = _resolve_pgm_path(path)
    annotations_path = _resolve_map_annotations_path(resolved)
    try:
        raw = json.loads(
            _read_container_file(annotations_path).decode("utf-8")
        )
    except FileNotFoundError:
        raw = {"annotations": []}
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"Invalid map annotations JSON: {exc}") from exc
    raw_annotations = raw.get("annotations", []) if isinstance(raw, dict) else raw
    annotations = [
        item
        for item in (_annotation_from_raw(raw_item) for raw_item in raw_annotations)
        if item is not None
    ]
    return {
        "path": _relative_map_path(resolved),
        "annotations_path": _relative_map_path(annotations_path),
        "annotations": annotations,
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


@router.delete("/maps/pgm")
def delete_pgm(path: str):
    """Delete a saved map: the .pgm plus its .yaml / annotations sidecars in
    ai_worker, and every mission stored for the map on this side."""
    resolved = _resolve_pgm_path(path)
    code, _ = _exec(["test", "-f", str(resolved)])
    if code != 0:
        raise HTTPException(404, f"PGM file not found: {path}")
    yaml_path = resolved.with_suffix(".yaml")
    annotations_path = _resolve_map_annotations_path(resolved)
    code, output = _exec(
        ["rm", "-f", str(resolved), str(yaml_path), str(annotations_path)]
    )
    if code != 0:
        raise HTTPException(503, output or f"Failed to delete {path}")
    return {
        "path": _relative_map_path(resolved),
        "removed_missions": remove_map_missions(resolved.stem),
        "deleted": True,
    }


@router.post("/maps/annotations/save")
def save_map_annotations(request: MapAnnotationsSaveRequest):
    resolved = _resolve_pgm_path(request.path)
    annotations_path = _resolve_map_annotations_path(resolved)
    annotations = [_annotation_payload(annotation) for annotation in request.annotations]
    _write_container_file(
        annotations_path,
        json.dumps(
            {
                "path": _relative_map_path(resolved),
                "annotations": annotations,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
    )
    return {
        "path": _relative_map_path(resolved),
        "annotations_path": _relative_map_path(annotations_path),
        "annotations": annotations,
        "saved": True,
    }
