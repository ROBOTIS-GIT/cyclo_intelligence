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
# Author: Seongwoo Kim

"""Mission manifest and BT XML persistence for Mission Canvas."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import re
import shutil
import stat
import threading
from typing import Any
from uuid import uuid4
import xml.etree.ElementTree as ET

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


router = APIRouter(prefix="/navigation/missions", tags=["navigation-missions"])

NAVIGATION_DATA_ROOT = Path(
    os.environ.get("CYCLO_NAVIGATION_DATA_DIR", "/workspace/navigation")
)
MISSION_SCHEMA_VERSION = 2
DEFAULT_MISSION_NAME = "default"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_RELATIVE_FILE = re.compile(r"^[A-Za-z0-9_./-]+$")
_RESERVED_STORAGE_NAMES = {".revisions", ".staging", ".trash"}
_MISSION_MUTATION_LOCK = threading.RLock()
_MANIFEST_WRITE_PATH: ContextVar[Path | None] = ContextVar(
    "mission_manifest_write_path",
    default=None,
)


def _serialized_mission_mutation(function):
    """Serialize mission filesystem mutations performed by FastAPI threads."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _MISSION_MUTATION_LOCK:
            return function(*args, **kwargs)

    return wrapped


def _require_manifest_revision(
    manifest: "MissionLoadResponse",
    expected_revision: int | None,
) -> None:
    if expected_revision is None:
        if manifest.exists or manifest.revision > 0:
            raise HTTPException(
                409,
                "expected_revision is required when modifying a reserved mission; "
                "reload the mission and retry",
            )
        return
    if expected_revision != manifest.revision:
        raise HTTPException(
            409,
            "Mission changed in another session "
            f"(expected revision {expected_revision}, current {manifest.revision})",
        )


class SpotPose(BaseModel):
    frame_id: str = "map"
    x: float
    y: float
    yaw: float = 0.0


class MissionWaypoint(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    pose: SpotPose
    local_bt: str = Field(default="", max_length=256)
    local_bt_files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionManifest(BaseModel):
    schema_version: int = MISSION_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    map_name: str = Field(min_length=1, max_length=128)
    mission_name: str = Field(default=DEFAULT_MISSION_NAME, min_length=1, max_length=128)
    global_bt: str = Field(default="global.xml", max_length=256)
    waypoints: list[MissionWaypoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionLoadResponse(MissionManifest):
    exists: bool = False


class MissionListResponse(BaseModel):
    map_name: str
    missions: list[str] = Field(default_factory=list)


class MissionSaveRequest(BaseModel):
    expected_revision: int | None = Field(default=None, ge=0)
    global_bt: str = Field(default="global.xml", max_length=256)
    waypoints: list[MissionWaypoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionBtFileRequest(BaseModel):
    path: str = Field(min_length=1, max_length=256)
    content: str = ""
    waypoint_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_revision: int | None = Field(default=None, ge=0)


class MissionBtDefaultRequest(BaseModel):
    waypoint_id: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=256)
    expected_revision: int | None = Field(default=None, ge=0)


class MissionBtFileResponse(BaseModel):
    path: str
    content: str = ""
    exists: bool = False
    revision: int = Field(default=0, ge=0)


def _validate_safe_name(value: str, *, label: str) -> str:
    name = value.strip()
    if not name:
        raise HTTPException(400, f"{label} must not be empty")
    if name in {".", ".."}:
        raise HTTPException(400, f"{label} must name one directory")
    if not _SAFE_NAME.fullmatch(name):
        raise HTTPException(
            400,
            f"{label} may contain only letters, numbers, '.', '_' and '-'",
        )
    return name


def _validate_map_name(value: str) -> str:
    name = _validate_safe_name(value, label="map_name")
    if name.casefold() in _RESERVED_STORAGE_NAMES:
        raise HTTPException(400, "map_name is reserved")
    return name


def _validate_mission_name(value: str) -> str:
    name = _validate_safe_name(value, label="mission_name")
    if name.casefold() in _RESERVED_STORAGE_NAMES:
        raise HTTPException(400, "mission_name is reserved")
    return name


def _validate_spot_id(value: str) -> str:
    return _validate_safe_name(value, label="waypoint_id")


def _validate_relative_file(value: str, *, default: str) -> str:
    raw = (value or default).strip() or default
    if len(raw) > 256:
        raise HTTPException(400, "BT file path must be 256 characters or fewer")
    candidate = Path(raw)
    if raw.startswith("/") or ".." in candidate.parts:
        raise HTTPException(400, "BT file path must stay inside the mission")
    if not _SAFE_RELATIVE_FILE.fullmatch(raw):
        raise HTTPException(400, "BT file path contains unsupported characters")
    normalized = candidate.as_posix()
    if normalized in {"", "."} or Path(normalized).suffix.lower() != ".xml":
        raise HTTPException(400, "BT file path must name an XML file")
    return normalized


def _mission_root(map_name: str) -> Path:
    missions_root = NAVIGATION_DATA_ROOT / "missions"
    candidate = missions_root / _validate_map_name(map_name)
    if candidate.is_symlink():
        raise HTTPException(400, "Map mission directory must not be a symlink")
    try:
        candidate.resolve(strict=False).relative_to(
            missions_root.resolve(strict=False)
        )
    except ValueError as exc:
        raise HTTPException(400, "Map mission directory escapes navigation storage") from exc
    return candidate


def remove_map_missions(map_name: str) -> int:
    """Delete every mission stored for a map (used when the map itself is
    deleted); returns how many missions were removed."""
    with _MISSION_MUTATION_LOCK:
        try:
            mission_root = _mission_root(map_name)
        except HTTPException:
            return 0
        if not mission_root.is_dir():
            return 0
        removed = sum(
            1
            for child in mission_root.iterdir()
            if child.is_dir() and child.name not in _RESERVED_STORAGE_NAMES
        )
        if removed == 0 and (mission_root / "mission.json").is_file():
            # Pre-mission-name legacy layout: one mission at the root.
            removed = 1
        shutil.rmtree(mission_root, ignore_errors=True)
        return removed


def _migrate_legacy_mission(map_name: str) -> None:
    """Move the pre-mission-name artifact layout into the default mission."""
    mission_root = _mission_root(map_name)
    legacy_manifest = mission_root / "mission.json"
    default_dir = mission_root / DEFAULT_MISSION_NAME
    if not legacy_manifest.exists() or default_dir.exists():
        return

    default_dir.mkdir(parents=True)
    for name in ("mission.json", "global.xml", "locals"):
        source = mission_root / name
        if source.exists():
            os.replace(source, default_dir / name)


def _mission_dir(map_name: str, mission_name: str) -> Path:
    _migrate_legacy_mission(map_name)
    mission_root = _mission_root(map_name)
    candidate = mission_root / _validate_mission_name(mission_name)
    if candidate.is_symlink():
        raise HTTPException(400, "Mission directory must not be a symlink")
    try:
        candidate.resolve(strict=False).relative_to(
            mission_root.resolve(strict=False)
        )
    except ValueError as exc:
        raise HTTPException(400, "Mission directory escapes map storage") from exc
    return candidate


def _mission_revision_marker_path(map_name: str, mission_name: str) -> Path:
    mission_root = _mission_root(map_name)
    revision_dir = mission_root / ".revisions"
    marker_path = revision_dir / f"{_validate_mission_name(mission_name)}.revision"
    if revision_dir.is_symlink() or marker_path.is_symlink():
        raise HTTPException(400, "Mission revision path must not contain symlinks")
    try:
        marker_path.resolve(strict=False).relative_to(
            mission_root.resolve(strict=False)
        )
    except ValueError as exc:
        raise HTTPException(400, "Mission revision path escapes map storage") from exc
    return marker_path


def _mission_staging_root(map_name: str) -> Path:
    mission_root = _mission_root(map_name)
    staging_root = mission_root / ".staging"
    if staging_root.is_symlink():
        raise HTTPException(400, "Mission staging directory must not be a symlink")
    try:
        staging_root.resolve(strict=False).relative_to(
            mission_root.resolve(strict=False)
        )
    except ValueError as exc:
        raise HTTPException(400, "Mission staging directory escapes map storage") from exc
    return staging_root


def _read_mission_revision_marker(map_name: str, mission_name: str) -> int:
    try:
        raw = _mission_revision_marker_path(map_name, mission_name).read_text(
            encoding="utf-8"
        )
        value = int(raw.strip())
    except FileNotFoundError:
        return 0
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(500, f"Failed to read mission revision: {exc}") from exc
    return max(0, value)


def _write_mission_revision_marker(
    map_name: str,
    mission_name: str,
    revision: int,
) -> None:
    try:
        next_revision = max(
            _read_mission_revision_marker(map_name, mission_name),
            max(0, int(revision)),
        )
        _write_text_atomic(
            _mission_revision_marker_path(map_name, mission_name),
            f"{next_revision}\n",
        )
    except OSError as exc:
        raise HTTPException(500, f"Failed to write mission revision: {exc}") from exc


def _manifest_path(map_name: str, mission_name: str) -> Path:
    return _mission_dir(map_name, mission_name) / "mission.json"


def _empty_manifest(map_name: str, mission_name: str) -> MissionLoadResponse:
    normalized = _validate_map_name(map_name)
    normalized_mission = _validate_mission_name(mission_name)
    return MissionLoadResponse(
        exists=False,
        schema_version=MISSION_SCHEMA_VERSION,
        revision=_read_mission_revision_marker(normalized, normalized_mission),
        map_name=normalized,
        mission_name=normalized_mission,
        global_bt="global.xml",
        waypoints=[],
        metadata={},
    )


def _serialize_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_fields_set(model: BaseModel) -> set[str]:
    fields = getattr(model, "model_fields_set", None)
    if fields is None:
        fields = getattr(model, "__fields_set__", set())
    return set(fields)


def _normalize_waypoint(waypoint: MissionWaypoint) -> MissionWaypoint:
    waypoint_id = _validate_spot_id(waypoint.id)
    local_bt = _validate_relative_file(
        waypoint.local_bt,
        default=f"locals/{waypoint_id}.xml",
    )
    local_bt_files: list[str] = []
    for candidate in [local_bt, *waypoint.local_bt_files]:
        normalized_path = _validate_relative_file(candidate, default=local_bt)
        if normalized_path not in local_bt_files:
            local_bt_files.append(normalized_path)
    return MissionWaypoint(
        id=waypoint_id,
        label=waypoint.label.strip(),
        pose=waypoint.pose,
        local_bt=local_bt,
        local_bt_files=local_bt_files,
        metadata=waypoint.metadata,
    )


def _normalize_manifest(
    map_name: str,
    mission_name: str,
    *,
    global_bt: str,
    waypoints: list[MissionWaypoint],
    metadata: dict[str, Any],
    exists: bool,
    revision: int = 0,
) -> MissionLoadResponse:
    normalized_map = _validate_map_name(map_name)
    normalized_mission = _validate_mission_name(mission_name)
    normalized_global_bt = _validate_relative_file(global_bt, default="global.xml")
    normalized_waypoints = [_normalize_waypoint(waypoint) for waypoint in waypoints]
    owners: dict[str, tuple[str, str]] = {}
    waypoint_ids: set[str] = set()
    for waypoint in normalized_waypoints:
        if waypoint.id in waypoint_ids:
            raise HTTPException(400, f"Duplicate waypoint id: {waypoint.id}")
        waypoint_ids.add(waypoint.id)
        for path in waypoint.local_bt_files:
            path_key = path.casefold()
            if path_key == normalized_global_bt.casefold():
                raise HTTPException(
                    400,
                    f"Waypoint {waypoint.id} BT file collides with global_bt: {path}",
                )
            previous = owners.get(path_key)
            if previous is not None and (
                previous[0] != waypoint.id or previous[1] != path
            ):
                raise HTTPException(
                    400,
                    f"BT file path collides with {previous[1]} "
                    f"(owners: {previous[0]}, {waypoint.id})",
                )
            owners[path_key] = (waypoint.id, path)
    return MissionLoadResponse(
        exists=exists,
        schema_version=MISSION_SCHEMA_VERSION,
        revision=max(0, int(revision)),
        map_name=normalized_map,
        mission_name=normalized_mission,
        global_bt=normalized_global_bt,
        waypoints=normalized_waypoints,
        metadata=metadata,
    )


def _read_manifest(map_name: str, mission_name: str = DEFAULT_MISSION_NAME) -> MissionLoadResponse:
    normalized = _validate_map_name(map_name)
    normalized_mission = _validate_mission_name(mission_name)
    path = _manifest_path(normalized, normalized_mission)
    descriptor: int | None = None
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return _empty_manifest(normalized, normalized_mission)
    except OSError as exc:
        raise HTTPException(
            500, f"Failed to inspect mission for {normalized}: {exc}"
        ) from exc

    if stat.S_ISLNK(path_stat.st_mode):
        raise HTTPException(400, "Mission manifest must not be a symlink")
    if not stat.S_ISREG(path_stat.st_mode):
        raise HTTPException(400, "Mission manifest must be a regular file")

    try:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise HTTPException(400, "Mission manifest must be a regular file")
        if (
            opened_stat.st_dev != path_stat.st_dev
            or opened_stat.st_ino != path_stat.st_ino
        ):
            raise HTTPException(409, "Mission manifest changed while being opened")
        with os.fdopen(descriptor, "r", encoding="utf-8") as f:
            descriptor = None
            raw = json.load(f)
    except HTTPException:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            500, f"Failed to read mission for {normalized}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    if not isinstance(raw, dict):
        raise HTTPException(500, f"Mission manifest for {normalized_mission} is invalid")
    waypoints = []
    raw_waypoints = raw.get("waypoints") or []
    if not isinstance(raw_waypoints, list):
        raise HTTPException(500, f"Mission waypoints for {normalized_mission} are invalid")
    for value in raw_waypoints:
        if not isinstance(value, dict):
            raise HTTPException(500, f"Mission waypoint for {normalized_mission} is invalid")
        try:
            waypoints.append(MissionWaypoint(**value))
        except Exception as exc:
            raise HTTPException(
                500, f"Mission waypoint for {normalized_mission} is invalid: {exc}"
            ) from exc
    raw_revision = raw.get("revision", 0)
    if not isinstance(raw_revision, int) or isinstance(raw_revision, bool):
        raw_revision = 0
    return _normalize_manifest(
        normalized,
        normalized_mission,
        global_bt=str(raw.get("global_bt") or "global.xml"),
        waypoints=waypoints,
        metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        exists=True,
        revision=max(
            0,
            raw_revision,
            _read_mission_revision_marker(normalized, normalized_mission),
        ),
    )


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_manifest(manifest: MissionLoadResponse) -> None:
    path = _MANIFEST_WRITE_PATH.get()
    if path is None:
        path = _manifest_path(manifest.map_name, manifest.mission_name)
    payload = _serialize_model(manifest)
    payload.pop("exists", None)
    payload["schema_version"] = MISSION_SCHEMA_VERSION
    try:
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        _write_text_atomic(path, content)
    except OSError as exc:
        raise HTTPException(
            500, f"Failed to write mission for {manifest.map_name}: {exc}"
        ) from exc


def _resolve_bt_path(map_name: str, mission_name: str, relative_path: str) -> Path:
    mission_dir = _mission_dir(map_name, mission_name)
    safe_path = _validate_relative_file(relative_path, default="global.xml")
    if mission_dir.is_symlink():
        raise HTTPException(400, "Mission directory must not be a symlink")
    cursor = mission_dir
    for component in Path(safe_path).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise HTTPException(400, "BT file path must not contain symlinks")
    missions_root = (NAVIGATION_DATA_ROOT / "missions").resolve(strict=False)
    resolved_mission_dir = mission_dir.resolve(strict=False)
    try:
        resolved_mission_dir.relative_to(missions_root)
    except ValueError as exc:
        raise HTTPException(400, "Mission directory escapes navigation storage") from exc

    path = mission_dir / safe_path
    # resolve(strict=False) follows every existing component while still
    # supporting a not-yet-created filename. This prevents a locals/ symlink
    # from redirecting BT reads or writes outside the selected mission.
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_mission_dir)
    except ValueError as exc:
        raise HTTPException(400, "BT file path escapes the mission") from exc
    return resolved_path


def _global_bt_with_default(
    content: str,
    *,
    waypoint_id: str,
    local_bt: str,
) -> str | None:
    """Return global XML with every matching MissionStep default updated.

    A waypoint can occur more than once when a route closes its loop. Preserve
    the authored formatting and attributes by changing only local_bt inside
    matching start tags, while ElementTree validates the input and result.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise HTTPException(409, f"Global BT XML is invalid: {exc}") from exc

    matching_steps = [
        element
        for element in root.iter()
        if str(element.tag).rsplit("}", 1)[-1] == "MissionStep"
        and element.attrib.get("waypoint_id") == waypoint_id
    ]
    if not matching_steps:
        return None

    tag_pattern = re.compile(
        r"<(?:[A-Za-z_][A-Za-z0-9_.-]*:)?MissionStep\b[^>]*>",
        re.DOTALL,
    )
    waypoint_pattern = re.compile(
        r"\bwaypoint_id\s*=\s*(['\"])(.*?)\1",
        re.DOTALL,
    )
    local_bt_pattern = re.compile(
        r"\blocal_bt\s*=\s*(['\"])(.*?)\1",
        re.DOTALL,
    )
    replacements = 0

    def replace_tag(match: re.Match[str]) -> str:
        nonlocal replacements
        tag = match.group(0)
        waypoint_match = waypoint_pattern.search(tag)
        if waypoint_match is None or waypoint_match.group(2) != waypoint_id:
            return tag
        replacements += 1
        if local_bt_pattern.search(tag):
            return local_bt_pattern.sub(
                lambda value: f'local_bt={value.group(1)}{local_bt}{value.group(1)}',
                tag,
                count=1,
            )
        insertion = f' local_bt="{local_bt}"'
        return tag[:-2] + insertion + "/>" if tag.endswith("/>") else tag[:-1] + insertion + ">"

    updated = tag_pattern.sub(replace_tag, content)
    if replacements != len(matching_steps):
        raise HTTPException(409, "Global BT MissionStep format is unsupported")
    try:
        ET.fromstring(updated)
    except ET.ParseError as exc:
        raise HTTPException(409, f"Failed to update global BT XML: {exc}") from exc
    return updated


@router.get("", response_model=MissionListResponse)
@_serialized_mission_mutation
def list_missions(map_name: str = Query(min_length=1, max_length=128)):
    normalized = _validate_map_name(map_name)
    _migrate_legacy_mission(normalized)
    mission_root = _mission_root(normalized)
    missions = sorted(
        path.name for path in mission_root.iterdir()
        if (
            path.is_dir()
            and _SAFE_NAME.fullmatch(path.name)
            and path.name.casefold() not in _RESERVED_STORAGE_NAMES
            and (path / "mission.json").is_file()
        )
    ) if mission_root.exists() else []
    return MissionListResponse(map_name=normalized, missions=missions)


@router.get("/{map_name}", response_model=MissionLoadResponse)
@_serialized_mission_mutation
def load_mission(map_name: str, mission_name: str = DEFAULT_MISSION_NAME):
    return _read_manifest(map_name, mission_name)


def _prune_orphan_local_bt_files(manifest: MissionLoadResponse) -> None:
    """Delete local XML files no waypoint owns anymore.

    Clients only delete stale paths they touched in-session, so files left by
    waypoint renames/deletions in earlier sessions would otherwise accumulate
    forever. The manifest being saved is the authoritative reference list.
    """
    mission_dir = _mission_dir(manifest.map_name, manifest.mission_name)
    locals_dir = mission_dir / "locals"
    if not locals_dir.is_dir() or locals_dir.is_symlink():
        return
    try:
        locals_dir.resolve(strict=False).relative_to(mission_dir.resolve(strict=False))
    except ValueError:
        # Never follow a runtime-created locals symlink while pruning.
        return
    referenced = {
        path
        for waypoint in manifest.waypoints
        for path in waypoint.local_bt_files
    }
    referenced.add(manifest.global_bt)
    for path in locals_dir.rglob("*.xml"):
        relative_path = path.relative_to(mission_dir).as_posix()
        if relative_path not in referenced:
            try:
                path.unlink()
            except OSError:
                pass
    # A layout migration can leave the old waypoint directory empty after its
    # XML files move under locals/<waypoint-id>/. Remove empty descendants so
    # the on-disk tree mirrors the manifest instead of accumulating dead dirs.
    directories = sorted(
        (
            path
            for path in locals_dir.rglob("*")
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


@router.post("/{map_name}", response_model=MissionLoadResponse)
@_serialized_mission_mutation
def save_mission(
    map_name: str,
    request: MissionSaveRequest,
    mission_name: str = DEFAULT_MISSION_NAME,
):
    # A cached v1 UI does not know about local_bt_files. Preserve the existing
    # waypoint library when that field is omitted, while an explicit list from
    # a v2 client remains authoritative (including deliberate removals).
    existing_manifest = _read_manifest(map_name, mission_name)
    _require_manifest_revision(existing_manifest, request.expected_revision)
    existing_by_id = {
        waypoint.id: waypoint
        for waypoint in existing_manifest.waypoints
    }
    compatible_waypoints: list[MissionWaypoint] = []
    for waypoint in request.waypoints:
        existing = existing_by_id.get(waypoint.id)
        if existing is not None and "local_bt_files" not in _model_fields_set(waypoint):
            payload = _serialize_model(waypoint)
            payload["local_bt_files"] = existing.local_bt_files
            compatible_waypoints.append(MissionWaypoint(**payload))
        else:
            compatible_waypoints.append(waypoint)
    manifest = _normalize_manifest(
        map_name,
        mission_name,
        global_bt=request.global_bt,
        waypoints=compatible_waypoints,
        metadata=request.metadata,
        exists=True,
        revision=existing_manifest.revision + 1,
    )
    _write_manifest(manifest)
    _prune_orphan_local_bt_files(manifest)
    return manifest


class MissionDeleteResponse(BaseModel):
    map_name: str
    mission_name: str
    deleted: bool = True


@router.delete("/{map_name}", response_model=MissionDeleteResponse)
@_serialized_mission_mutation
def delete_mission(
    map_name: str,
    mission_name: str = Query(min_length=1, max_length=128),
    expected_revision: int = Query(ge=0),
):
    manifest = _read_manifest(map_name, mission_name)
    if not manifest.exists:
        raise HTTPException(
            404, f"Mission {mission_name} not found for {map_name}"
        )
    _require_manifest_revision(manifest, expected_revision)
    mission_dir = _mission_dir(map_name, mission_name)
    mission_root = _mission_root(map_name)
    trash_root = mission_root / ".trash"
    if trash_root.is_symlink():
        raise HTTPException(400, "Mission trash directory must not be a symlink")
    trash_dir = trash_root / f"{manifest.mission_name}.{uuid4().hex}"
    # Commit the high-water mark first. Existing manifests now read the max of
    # their stored revision and this marker, so a crash on either side of the
    # atomic move cannot reopen the old generation.
    _write_mission_revision_marker(
        manifest.map_name,
        manifest.mission_name,
        manifest.revision + 1,
    )
    try:
        trash_root.mkdir(parents=True, exist_ok=True)
        # Moving a directory on the same filesystem is atomic.
        os.replace(mission_dir, trash_dir)
    except OSError as exc:
        raise HTTPException(
            500, f"Failed to delete mission {mission_name}: {exc}"
        ) from exc
    try:
        shutil.rmtree(trash_dir)
    except OSError:
        # The mission identity is already atomically removed and protected by
        # its tombstone. A later maintenance pass can remove hidden trash.
        pass
    return MissionDeleteResponse(
        map_name=_validate_map_name(map_name),
        mission_name=_validate_mission_name(mission_name),
    )


class MissionRenameRequest(BaseModel):
    mission_name: str = Field(min_length=1, max_length=128)
    new_name: str = Field(min_length=1, max_length=128)
    expected_revision: int | None = Field(default=None, ge=0)


@router.post("/{map_name}/rename", response_model=MissionLoadResponse)
@_serialized_mission_mutation
def rename_mission(map_name: str, request: MissionRenameRequest):
    source_dir = _mission_dir(map_name, request.mission_name)
    target_dir = _mission_dir(map_name, request.new_name)
    source_manifest = _read_manifest(map_name, request.mission_name)
    if not source_manifest.exists:
        raise HTTPException(
            404, f"Mission {request.mission_name} not found for {map_name}"
        )
    _require_manifest_revision(source_manifest, request.expected_revision)
    if target_dir.exists():
        raise HTTPException(
            409, f"Mission {request.new_name} already exists for {map_name}"
        )
    target_generation = max(
        source_manifest.revision,
        _read_mission_revision_marker(map_name, request.new_name),
    ) + 1
    _write_mission_revision_marker(
        source_manifest.map_name,
        request.new_name,
        target_generation,
    )
    # Moving the directory must also retire the old identity. A stale tab that
    # still knows the source revision can no longer recreate it as revision 0.
    _write_mission_revision_marker(
        source_manifest.map_name,
        source_manifest.mission_name,
        max(
            source_manifest.revision,
            _read_mission_revision_marker(map_name, request.mission_name),
        ) + 1,
    )
    try:
        os.replace(source_dir, target_dir)
    except OSError as exc:
        raise HTTPException(
            500, f"Failed to rename mission {request.mission_name}: {exc}"
        ) from exc
    # Rewrite the manifest so its stored mission_name matches the new name.
    try:
        moved = _read_manifest(map_name, request.new_name)
        manifest = _normalize_manifest(
            moved.map_name,
            moved.mission_name,
            global_bt=moved.global_bt,
            waypoints=moved.waypoints,
            metadata=moved.metadata,
            exists=True,
            revision=target_generation,
        )
        _write_manifest(manifest)
    except HTTPException:
        try:
            os.replace(target_dir, source_dir)
        except OSError:
            pass
        raise
    return manifest


class MissionDuplicateRequest(BaseModel):
    mission_name: str = Field(min_length=1, max_length=128)
    new_name: str = Field(min_length=1, max_length=128)
    expected_revision: int | None = Field(default=None, ge=0)


def _reject_tree_symlinks(root: Path) -> None:
    def raise_walk_error(error: OSError) -> None:
        raise error

    for current_root, directory_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current = Path(current_root)
        for name in [*directory_names, *file_names]:
            if (current / name).is_symlink():
                raise HTTPException(
                    400, "Mission with symlinked content cannot be duplicated"
                )


@router.post("/{map_name}/duplicate", response_model=MissionLoadResponse)
@_serialized_mission_mutation
def duplicate_mission(map_name: str, request: MissionDuplicateRequest):
    source_dir = _mission_dir(map_name, request.mission_name)
    target_dir = _mission_dir(map_name, request.new_name)
    source_manifest = _read_manifest(map_name, request.mission_name)
    if not source_manifest.exists:
        raise HTTPException(
            404, f"Mission {request.mission_name} not found for {map_name}"
        )
    _require_manifest_revision(source_manifest, request.expected_revision)
    if target_dir.exists():
        raise HTTPException(
            409, f"Mission {request.new_name} already exists for {map_name}"
        )
    target_generation = max(
        source_manifest.revision,
        _read_mission_revision_marker(map_name, request.new_name),
    ) + 1
    manifest = _normalize_manifest(
        source_manifest.map_name,
        request.new_name,
        global_bt=source_manifest.global_bt,
        waypoints=source_manifest.waypoints,
        metadata=source_manifest.metadata,
        exists=True,
        revision=target_generation,
    )
    _write_mission_revision_marker(
        source_manifest.map_name,
        request.new_name,
        target_generation,
    )

    staging_root = _mission_staging_root(map_name)
    staging_dir = staging_root / (
        f"{manifest.mission_name}.{uuid4().hex}.duplicate"
    )
    try:
        staging_root.mkdir(parents=True, exist_ok=True)
        if not staging_root.is_dir() or staging_root.is_symlink():
            raise HTTPException(400, "Mission staging path must be a directory")

        _reject_tree_symlinks(source_dir)
        # Preserve, rather than follow, any symlink introduced after the first
        # scan. The second scan then rejects it before the tree can be published.
        shutil.copytree(source_dir, staging_dir, symlinks=True)
        _reject_tree_symlinks(staging_dir)

        override = _MANIFEST_WRITE_PATH.set(staging_dir / "mission.json")
        try:
            _write_manifest(manifest)
        finally:
            _MANIFEST_WRITE_PATH.reset(override)

        # The complete tree and its revised manifest become visible together.
        os.replace(staging_dir, target_dir)
    except HTTPException:
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass
        raise HTTPException(
            500, f"Failed to duplicate mission {request.mission_name}: {exc}"
        ) from exc
    return manifest


@router.get("/{map_name}/bt", response_model=MissionBtFileResponse)
@_serialized_mission_mutation
def load_bt_file(
    map_name: str,
    path: str = Query(min_length=1, max_length=256),
    mission_name: str = DEFAULT_MISSION_NAME,
):
    bt_path = _resolve_bt_path(map_name, mission_name, path)
    safe_path = bt_path.relative_to(_mission_dir(map_name, mission_name)).as_posix()
    revision = _read_manifest(map_name, mission_name).revision
    try:
        content = bt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return MissionBtFileResponse(
            path=safe_path, content="", exists=False, revision=revision
        )
    except OSError as exc:
        raise HTTPException(500, f"Failed to read BT XML: {exc}") from exc
    return MissionBtFileResponse(
        path=safe_path, content=content, exists=True, revision=revision
    )


@router.put("/{map_name}/bt", response_model=MissionBtFileResponse)
@_serialized_mission_mutation
def save_bt_file(
    map_name: str,
    request: MissionBtFileRequest,
    mission_name: str = DEFAULT_MISSION_NAME,
):
    bt_path = _resolve_bt_path(map_name, mission_name, request.path)
    safe_path = bt_path.relative_to(
        _mission_dir(map_name, mission_name)
    ).as_posix()
    updated_manifest: MissionLoadResponse | None = None
    current_manifest = _read_manifest(map_name, mission_name)
    _require_manifest_revision(current_manifest, request.expected_revision)
    current_owners = [
        waypoint.id
        for waypoint in current_manifest.waypoints
        if any(
            owned_path.casefold() == safe_path.casefold()
            for owned_path in waypoint.local_bt_files
        )
    ]
    if request.waypoint_id is None and current_owners:
        raise HTTPException(
            409,
            f"BT file {safe_path} is owned by waypoint {current_owners[0]}; "
            "waypoint_id and expected_revision are required to overwrite it",
        )
    membership_changed = False
    manifest_waypoints = current_manifest.waypoints
    if request.waypoint_id is not None:
        waypoint_id = _validate_spot_id(request.waypoint_id)
        if not current_manifest.exists:
            raise HTTPException(
                404,
                f"Mission {current_manifest.mission_name} not found for "
                f"{current_manifest.map_name}",
            )

        found = False
        updated_waypoints: list[MissionWaypoint] = []
        for waypoint in current_manifest.waypoints:
            if waypoint.id != waypoint_id:
                updated_waypoints.append(waypoint)
                continue
            found = True
            already_owned = any(
                owned_path.casefold() == safe_path.casefold()
                for owned_path in waypoint.local_bt_files
            )
            if safe_path.casefold() == current_manifest.global_bt.casefold():
                raise HTTPException(400, "A waypoint BT cannot replace global_bt")
            if not already_owned and not safe_path.startswith("locals/"):
                raise HTTPException(
                    400,
                    "New waypoint BT files must be stored under locals/",
                )
            if already_owned:
                updated_waypoints.append(waypoint)
            else:
                membership_changed = True
                payload = _serialize_model(waypoint)
                payload["local_bt_files"] = [*waypoint.local_bt_files, safe_path]
                updated_waypoints.append(MissionWaypoint(**payload))
        if not found:
            raise HTTPException(
                404,
                f"Waypoint {waypoint_id} not found in mission "
                f"{current_manifest.mission_name}",
            )

        # Normalize the complete manifest before touching the XML. This makes
        # path aliases/case collisions with another waypoint fail without
        # overwriting that waypoint's file.
        manifest_waypoints = updated_waypoints

    previous_content: str | None = None
    file_existed = False
    try:
        previous_content = bt_path.read_text(encoding="utf-8")
        file_existed = True
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise HTTPException(500, f"Failed to read existing BT XML: {exc}") from exc
    content_changed = not file_existed or previous_content != request.content

    # A BT content write is part of the mission snapshot just like changing
    # membership or the default pointer. Bump the same revision so a second
    # browser that loaded the old XML cannot silently overwrite this save.
    if current_manifest.exists and (content_changed or membership_changed):
        updated_manifest = _normalize_manifest(
            current_manifest.map_name,
            current_manifest.mission_name,
            global_bt=current_manifest.global_bt,
            waypoints=manifest_waypoints,
            metadata=current_manifest.metadata,
            exists=True,
            revision=current_manifest.revision + 1,
        )
    # The first file upload also reserves a not-yet-created mission name. Bump
    # even when an orphan file already has identical content: otherwise two
    # revision-0 creators could both proceed and assemble one mixed mission.
    pending_revision = (
        current_manifest.revision + 1
        if not current_manifest.exists
        else current_manifest.revision
    )

    if updated_manifest is not None:
        # Burn the next generation before publishing XML or membership. If a
        # later write fails, _read_manifest() still exposes this high-water
        # mark and forces every caller that saw the old content to reload.
        _write_mission_revision_marker(
            current_manifest.map_name,
            current_manifest.mission_name,
            updated_manifest.revision,
        )
    elif not current_manifest.exists:
        # Reserve a draft generation before publishing any XML. If the process
        # is interrupted during the following atomic file write, the generation
        # remains burned and a stale creator cannot mix files into this draft.
        _write_mission_revision_marker(
            current_manifest.map_name,
            current_manifest.mission_name,
            pending_revision,
        )

    if content_changed:
        try:
            _write_text_atomic(bt_path, request.content)
        except OSError as exc:
            raise HTTPException(500, f"Failed to write BT XML: {exc}") from exc
    if updated_manifest is not None:
        try:
            _write_manifest(updated_manifest)
        except HTTPException:
            # Keep the file content and manifest revision atomic from the API
            # caller's point of view. The mission lock prevents another writer
            # from observing this rollback window in-process.
            if content_changed:
                try:
                    if file_existed and previous_content is not None:
                        _write_text_atomic(bt_path, previous_content)
                    else:
                        bt_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    return MissionBtFileResponse(
        path=safe_path,
        content=request.content,
        exists=True,
        revision=(
            updated_manifest.revision
            if updated_manifest is not None
            else pending_revision
        ),
    )


@router.put("/{map_name}/bt/default", response_model=MissionLoadResponse)
@_serialized_mission_mutation
def set_default_bt_file(
    map_name: str,
    request: MissionBtDefaultRequest,
    mission_name: str = DEFAULT_MISSION_NAME,
):
    waypoint_id = _validate_spot_id(request.waypoint_id)
    safe_path = _validate_relative_file(request.path, default="global.xml")
    current_manifest = _read_manifest(map_name, mission_name)
    _require_manifest_revision(current_manifest, request.expected_revision)
    if not current_manifest.exists:
        raise HTTPException(
            404,
            f"Mission {current_manifest.mission_name} not found for "
            f"{current_manifest.map_name}",
        )

    found = False
    default_changed = False
    updated_waypoints: list[MissionWaypoint] = []
    for waypoint in current_manifest.waypoints:
        if waypoint.id != waypoint_id:
            updated_waypoints.append(waypoint)
            continue
        found = True
        if safe_path not in waypoint.local_bt_files:
            raise HTTPException(
                400,
                f"BT file {safe_path} is not owned by waypoint {waypoint_id}",
            )
        if safe_path.casefold() == current_manifest.global_bt.casefold():
            raise HTTPException(400, "A waypoint BT cannot replace global_bt")
        if waypoint.local_bt == safe_path:
            updated_waypoints.append(waypoint)
            continue
        default_changed = True
        payload = _serialize_model(waypoint)
        payload["local_bt"] = safe_path
        updated_waypoints.append(MissionWaypoint(**payload))
    if not found:
        raise HTTPException(
            404,
            f"Waypoint {waypoint_id} not found in mission "
            f"{current_manifest.mission_name}",
        )
    if not default_changed:
        return current_manifest

    updated_manifest = _normalize_manifest(
        current_manifest.map_name,
        current_manifest.mission_name,
        global_bt=current_manifest.global_bt,
        waypoints=updated_waypoints,
        metadata=current_manifest.metadata,
        exists=True,
        revision=current_manifest.revision + 1,
    )
    global_bt_path = _resolve_bt_path(
        current_manifest.map_name,
        current_manifest.mission_name,
        current_manifest.global_bt,
    )
    original_global_bt: str | None = None
    updated_global_bt: str | None = None
    try:
        original_global_bt = global_bt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise HTTPException(500, f"Failed to read global BT XML: {exc}") from exc
    if original_global_bt is not None:
        updated_global_bt = _global_bt_with_default(
            original_global_bt,
            waypoint_id=waypoint_id,
            local_bt=safe_path,
        )

    global_bt_changed = (
        updated_global_bt is not None
        and updated_global_bt != original_global_bt
    )
    # The default pointer and its serialized global XML form one semantic
    # mutation. Burn its generation before changing either file; rollback of a
    # later write restores content only, never the concurrency high-water mark.
    _write_mission_revision_marker(
        current_manifest.map_name,
        current_manifest.mission_name,
        updated_manifest.revision,
    )
    if global_bt_changed:
        try:
            _write_text_atomic(global_bt_path, updated_global_bt)
        except OSError as exc:
            raise HTTPException(500, f"Failed to write global BT XML: {exc}") from exc
    try:
        _write_manifest(updated_manifest)
    except HTTPException:
        if global_bt_changed and original_global_bt is not None:
            try:
                _write_text_atomic(global_bt_path, original_global_bt)
            except OSError:
                pass
        raise
    return updated_manifest


@router.delete("/{map_name}/bt", response_model=MissionBtFileResponse)
@_serialized_mission_mutation
def delete_bt_file(
    map_name: str,
    path: str = Query(min_length=1, max_length=256),
    mission_name: str = DEFAULT_MISSION_NAME,
    expected_revision: int = Query(ge=0),
):
    bt_path = _resolve_bt_path(map_name, mission_name, path)
    safe_path = bt_path.relative_to(_mission_dir(map_name, mission_name)).as_posix()
    current_manifest = _read_manifest(map_name, mission_name)
    # Validate before checking ownership or existence. Returning the latest
    # revision for a stale missing-file request would let a caller adopt a
    # concurrent edit without ever loading its contents.
    _require_manifest_revision(current_manifest, expected_revision)
    if safe_path.casefold() == current_manifest.global_bt.casefold():
        raise HTTPException(409, "Cannot delete the mission global_bt file")
    owner = next((
        waypoint.id
        for waypoint in current_manifest.waypoints
        if any(
            owned_path.casefold() == safe_path.casefold()
            for owned_path in waypoint.local_bt_files
        )
    ), None)
    if owner is not None:
        raise HTTPException(
            409,
            f"Cannot delete BT file {safe_path}; waypoint {owner} still owns it",
        )
    try:
        previous_content = bt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return MissionBtFileResponse(
            path=safe_path,
            content="",
            exists=False,
            revision=current_manifest.revision,
        )
    except OSError as exc:
        raise HTTPException(500, f"Failed to read BT XML for deletion: {exc}") from exc

    next_revision = current_manifest.revision + 1
    updated_manifest: MissionLoadResponse | None = None
    if current_manifest.exists:
        updated_manifest = _normalize_manifest(
            current_manifest.map_name,
            current_manifest.mission_name,
            global_bt=current_manifest.global_bt,
            waypoints=current_manifest.waypoints,
            metadata=current_manifest.metadata,
            exists=True,
            revision=next_revision,
        )
    # A successful marker write is the deletion's concurrency commit point.
    # Keep it even when unlink or manifest persistence fails so stale sessions
    # cannot overwrite the uncertain result without reloading.
    _write_mission_revision_marker(
        current_manifest.map_name,
        current_manifest.mission_name,
        next_revision,
    )
    try:
        bt_path.unlink()
    except OSError as exc:
        raise HTTPException(500, f"Failed to delete BT XML: {exc}") from exc

    if updated_manifest is not None:
        try:
            _write_manifest(updated_manifest)
        except HTTPException:
            try:
                _write_text_atomic(bt_path, previous_content)
            except OSError:
                pass
            raise
    return MissionBtFileResponse(
        path=safe_path,
        content="",
        exists=False,
        revision=next_revision,
    )
