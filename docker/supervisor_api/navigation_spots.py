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


"""Map Spot persistence for the Mission Canvas navigation workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


router = APIRouter(prefix="/navigation/spots", tags=["navigation-spots"])

SPOTS_ROOT = Path(
    os.environ.get("CYCLO_NAVIGATION_DATA_DIR", "/workspace/navigation")
)
SPOTS_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class SpotPose(BaseModel):
    frame_id: str = "map"
    x: float
    y: float
    yaw: float = 0.0


class MapSpot(BaseModel):
    id: str
    map_name: str
    label: str
    pose: SpotPose
    linked_bt_tree: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpotListResponse(BaseModel):
    map_name: str
    spots: list[MapSpot]


class SpotCreateRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    pose: SpotPose
    id: str = Field(default="", max_length=128)
    linked_bt_tree: str = Field(default="", max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpotUpdateRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)
    label: Optional[str] = Field(default=None, min_length=1, max_length=128)
    pose: Optional[SpotPose] = None
    linked_bt_tree: Optional[str] = Field(default=None, max_length=256)
    metadata: Optional[dict[str, Any]] = None


class ActionResult(BaseModel):
    ok: bool
    message: str


def _validate_safe_name(value: str, *, label: str) -> str:
    name = value.strip()
    if not name:
        raise HTTPException(400, f"{label} must not be empty")
    if not _SAFE_NAME.fullmatch(name):
        raise HTTPException(
            400,
            f"{label} may contain only letters, numbers, '.', '_' and '-'",
        )
    return name


def _validate_map_name(value: str) -> str:
    return _validate_safe_name(value, label="map_name")


def _validate_spot_id(value: str) -> str:
    return _validate_safe_name(value, label="spot_id")


def _map_dir(map_name: str) -> Path:
    return SPOTS_ROOT / "maps" / _validate_map_name(map_name)


def _spots_path(map_name: str) -> Path:
    return _map_dir(map_name) / "spots.json"


def _empty_store(map_name: str) -> dict[str, Any]:
    return {
        "schema_version": SPOTS_SCHEMA_VERSION,
        "map_name": map_name,
        "spots": [],
    }


def _read_store(map_name: str) -> dict[str, Any]:
    normalized = _validate_map_name(map_name)
    path = _spots_path(normalized)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return _empty_store(normalized)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            500, f"Failed to read spots for {normalized}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        return _empty_store(normalized)
    spots = data.get("spots")
    if not isinstance(spots, list):
        spots = []
    return {
        "schema_version": int(data.get("schema_version") or 0)
        or SPOTS_SCHEMA_VERSION,
        "map_name": normalized,
        "spots": spots,
    }


def _write_store(map_name: str, store: dict[str, Any]) -> None:
    normalized = _validate_map_name(map_name)
    path = _spots_path(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SPOTS_SCHEMA_VERSION,
        "map_name": normalized,
        "spots": store.get("spots", []),
    }
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except OSError as exc:
        raise HTTPException(
            500, f"Failed to write spots for {normalized}: {exc}"
        ) from exc


def _spot_from_raw(map_name: str, value: Any) -> MapSpot | None:
    if not isinstance(value, dict):
        return None
    try:
        payload = dict(value)
        payload["map_name"] = _validate_map_name(
            str(payload.get("map_name") or map_name)
        )
        payload["id"] = _validate_spot_id(str(payload.get("id") or ""))
        return MapSpot(**payload)
    except Exception:
        return None


def _list_spots(map_name: str) -> list[MapSpot]:
    store = _read_store(map_name)
    spots = []
    for raw in store.get("spots", []):
        spot = _spot_from_raw(store["map_name"], raw)
        if spot is not None:
            spots.append(spot)
    return spots


def _serialize_spot(spot: MapSpot) -> dict[str, Any]:
    if hasattr(spot, "model_dump"):
        return spot.model_dump()
    return spot.dict()


def _new_spot_id(label: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("_.-")
    if not base:
        base = "spot"
    return f"{base[:64]}_{uuid.uuid4().hex[:8]}"


@router.get("", response_model=SpotListResponse)
def list_spots(map_name: str = Query(min_length=1, max_length=128)):
    normalized = _validate_map_name(map_name)
    return SpotListResponse(
        map_name=normalized,
        spots=_list_spots(normalized),
    )


@router.post("", response_model=MapSpot)
def create_spot(request: SpotCreateRequest):
    map_name = _validate_map_name(request.map_name)
    store = _read_store(map_name)
    spots = _list_spots(map_name)
    spot_id = (
        _validate_spot_id(request.id)
        if request.id.strip()
        else _new_spot_id(request.label)
    )
    if any(spot.id == spot_id for spot in spots):
        raise HTTPException(409, f"Spot already exists: {spot_id}")

    spot = MapSpot(
        id=spot_id,
        map_name=map_name,
        label=request.label.strip(),
        pose=request.pose,
        linked_bt_tree=request.linked_bt_tree.strip(),
        metadata=request.metadata,
    )
    store["spots"] = [_serialize_spot(item) for item in spots]
    store["spots"].append(_serialize_spot(spot))
    _write_store(map_name, store)
    return spot


@router.patch("/{spot_id}", response_model=MapSpot)
def update_spot(spot_id: str, request: SpotUpdateRequest):
    normalized_id = _validate_spot_id(spot_id)
    map_name = _validate_map_name(request.map_name)
    spots = _list_spots(map_name)
    for index, spot in enumerate(spots):
        if spot.id != normalized_id:
            continue
        payload = _serialize_spot(spot)
        if request.label is not None:
            payload["label"] = request.label.strip()
        if request.pose is not None:
            payload["pose"] = _serialize_spot(request.pose)
        if request.linked_bt_tree is not None:
            payload["linked_bt_tree"] = request.linked_bt_tree.strip()
        if request.metadata is not None:
            payload["metadata"] = request.metadata
        updated = MapSpot(**payload)
        spots[index] = updated
        _write_store(
            map_name,
            {"spots": [_serialize_spot(item) for item in spots]},
        )
        return updated
    raise HTTPException(404, f"Spot not found: {normalized_id}")


@router.delete("/{spot_id}", response_model=ActionResult)
def delete_spot(
    spot_id: str,
    map_name: str = Query(min_length=1, max_length=128),
):
    normalized_id = _validate_spot_id(spot_id)
    normalized_map = _validate_map_name(map_name)
    spots = _list_spots(normalized_map)
    next_spots = [spot for spot in spots if spot.id != normalized_id]
    if len(next_spots) == len(spots):
        raise HTTPException(404, f"Spot not found: {normalized_id}")
    _write_store(
        normalized_map,
        {"spots": [_serialize_spot(item) for item in next_spots]},
    )
    return ActionResult(ok=True, message=f"Spot deleted: {normalized_id}")
