"""Same-origin proxy for the AI Worker Intervention Controller.

The controller remains loopback-only on the robot.  Cyclo's browser talks to
this router through nginx, so the controller token and robot address never
need to be exposed to JavaScript.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/intervention", tags=["intervention"])

_CONTROLLER_BASE = os.environ.get(
    "AI_WORKER_INTERVENTION_URL",
    "http://127.0.0.1:8779/api/intervention",
).rstrip("/")
_CONTROLLER_TOKEN = os.environ.get("INTERVENTION_CONTROL_TOKEN", "").strip()
_ACTIONS = {
    "standby",
    "standby_stop",
    "start",
    "recording",
    "end",
    "complete",
    "abort",
    "reset",
    "reset_fault",
}


class InterventionActionRequest(BaseModel):
    model: Literal["a3", "mini"] = "a3"
    outcome: str = ""


class PoseMoveRequest(BaseModel):
    model: Literal["f2", "sg2"]
    left_arm: list[float]
    right_arm: list[float]
    head: list[float]
    lift: list[float]
    duration: float = 8.0


def _decode_payload(raw: bytes) -> dict:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"detail": raw.decode("utf-8", errors="replace")}
    return decoded if isinstance(decoded, dict) else {"detail": decoded}


def _controller_request(
    path: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    headers = {"Accept": "application/json", "X-Control-Client": "cyclo-playground"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if _CONTROLLER_TOKEN:
        headers["X-Control-Token"] = _CONTROLLER_TOKEN
    request = urllib.request.Request(
        f"{_CONTROLLER_BASE}/{path.lstrip('/')}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=75) as response:
            return _decode_payload(response.read())
    except urllib.error.HTTPError as exc:
        detail = _decode_payload(exc.read()).get("detail")
        raise HTTPException(
            status_code=exc.code,
            detail=detail or f"AI Worker intervention request failed ({exc.code})",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI Worker Intervention Controller unavailable: {exc}",
        ) from exc


@router.get("/state")
def intervention_state() -> dict:
    return _controller_request("state")


@router.post("/pose")
def move_to_operator_pose(request: PoseMoveRequest) -> dict:
    return _controller_request(
        "pose",
        method="POST",
        payload=request.model_dump(),
    )


@router.post("/{action}")
def intervention_action(action: str, request: InterventionActionRequest) -> dict:
    if action not in _ACTIONS:
        raise HTTPException(status_code=404, detail="Unsupported intervention action")
    return _controller_request(
        action,
        method="POST",
        payload=request.model_dump(),
    )
