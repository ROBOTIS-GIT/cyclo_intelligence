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
# Author: Seongwoo Kim

"""Saved behavior trees for the Action Canvas.

Trees authored in the UI are user data, so they live under the workspace
volume (``CYCLO_BT_TREES_DIR``, default ``/workspace/bt/trees``) rather
than inside a code package. On first listing the directory is seeded from
two read-only sources, never overwriting an existing file:

- the example trees installed with the ``orchestrator`` package
  (``CYCLO_BT_EXAMPLE_TREES_DIR``, default ``$COLCON_WS/install/...``);
- the directory releases before 1.4.0 saved into,
  ``orchestrator/orchestrator/bt/trees`` of the bind-mounted checkout
  (``CYCLO_BT_LEGACY_TREES_DIR``, default ``$COLCON_WS/src/...``), so trees
  saved with an earlier release keep showing up after the upgrade.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from supervisor_api import bt_support

router = APIRouter(prefix="/bt", tags=["behavior-trees"])

TREES_DIR_ENV = "CYCLO_BT_TREES_DIR"
EXAMPLE_TREES_DIR_ENV = "CYCLO_BT_EXAMPLE_TREES_DIR"
LEGACY_TREES_DIR_ENV = "CYCLO_BT_LEGACY_TREES_DIR"
DEFAULT_TREES_DIR = "/workspace/bt/trees"
DEFAULT_COLCON_WS = "/root/ros2_ws"

_TREE_NAME_RE = re.compile(r"^[\w\-]+\.xml$")


def trees_dir() -> Path:
    return Path(os.environ.get(TREES_DIR_ENV, "").strip() or DEFAULT_TREES_DIR)


def _colcon_ws() -> Path:
    return Path(os.environ.get("COLCON_WS", "").strip() or DEFAULT_COLCON_WS)


def example_trees_dir() -> Optional[Path]:
    configured = os.environ.get(EXAMPLE_TREES_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return _colcon_ws() / "install" / "orchestrator" / "share" / "orchestrator" / "bt" / "trees"


def legacy_trees_dir() -> Optional[Path]:
    configured = os.environ.get(LEGACY_TREES_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return (
        _colcon_ws() / "src" / "cyclo_intelligence" / "orchestrator"
        / "orchestrator" / "bt" / "trees"
    )


def seed_sources() -> List[Path]:
    """Read-only directories whose trees are copied into the user dir once."""
    sources = []
    for candidate in (example_trees_dir(), legacy_trees_dir()):
        if candidate is not None and candidate.is_dir() and candidate not in sources:
            sources.append(candidate)
    return sources


def normalize_tree_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not candidate:
        raise HTTPException(400, "filename is required")
    if not candidate.lower().endswith(".xml"):
        candidate += ".xml"
    if not _TREE_NAME_RE.match(candidate):
        raise HTTPException(400, f"Invalid filename: {candidate!r}")
    return candidate


def ensure_trees_dir() -> Path:
    directory = trees_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for source in seed_sources():
        if source.resolve() == directory.resolve():
            continue
        for tree in sorted(source.glob("*.xml")):
            target = directory / tree.name
            if tree.is_file() and not target.exists():
                shutil.copyfile(tree, target)
    return directory


class BtSupportResponse(BaseModel):
    supported_robot_types: List[str]


class BtTreeSummary(BaseModel):
    name: str
    path: str
    modified_at: float


class BtTreeListResponse(BaseModel):
    directory: str
    trees: List[BtTreeSummary]


class BtTreeResponse(BaseModel):
    name: str
    path: str
    content: str


class BtTreeSaveRequest(BaseModel):
    filename: str
    content: str = ""
    overwrite: bool = False


class BtTreeSaveResponse(BaseModel):
    ok: bool = True
    message: str
    name: str
    path: str


@router.get("/support", response_model=BtSupportResponse)
async def bt_support_info() -> BtSupportResponse:
    try:
        robot_types = bt_support.bt_supported_robot_types()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return BtSupportResponse(supported_robot_types=robot_types)


@router.get("/trees", response_model=BtTreeListResponse)
async def list_trees() -> BtTreeListResponse:
    directory = ensure_trees_dir()
    trees = []
    for path in sorted(directory.glob("*.xml")):
        if not path.is_file():
            continue
        trees.append(BtTreeSummary(
            name=path.name,
            path=str(path),
            modified_at=path.stat().st_mtime,
        ))
    return BtTreeListResponse(directory=str(directory), trees=trees)


@router.get("/trees/{name}", response_model=BtTreeResponse)
async def read_tree(name: str) -> BtTreeResponse:
    filename = normalize_tree_name(name)
    path = ensure_trees_dir() / filename
    if not path.is_file():
        raise HTTPException(404, f"Tree not found: {filename}")
    return BtTreeResponse(
        name=filename,
        path=str(path),
        content=path.read_text(encoding="utf-8"),
    )


@router.post("/trees", response_model=BtTreeSaveResponse)
async def save_tree(request: BtTreeSaveRequest) -> BtTreeSaveResponse:
    filename = normalize_tree_name(request.filename)
    path = ensure_trees_dir() / filename
    if path.exists() and not request.overwrite:
        raise HTTPException(
            409,
            {
                "code": "file_exists",
                "message": f"Tree already exists: {filename}",
                "filename": filename,
                "path": str(path),
            },
        )
    path.write_text(request.content, encoding="utf-8")
    return BtTreeSaveResponse(message=f"Saved: {filename}", name=filename, path=str(path))
