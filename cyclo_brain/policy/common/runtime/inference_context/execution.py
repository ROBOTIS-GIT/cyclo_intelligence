"""Explicit execution facts. Publication is never physical execution feedback."""

from dataclasses import dataclass, fields
import base64
import binascii
import json
import math
import zlib


CONTEXT_COMPRESSION_THRESHOLD_BYTES = 64 * 1024
MAX_CONTEXT_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_CONTEXT_BYTES = 8 * 1024 * 1024


def _json_record(value):
    # These frozen records contain only validated scalars/tuples. JSON traverses
    # them directly; asdict would deepcopy every scalar in every action vector.
    if not isinstance(value, (ExecutionContext, ActionRecord, PlanningRecord, ResetRecord)):
        raise TypeError(f"not an execution record: {type(value).__name__}")
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _index(value, name, optional=False):
    if optional and value is None:
        return
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _vector(value):
    if not isinstance(value, (list, tuple)) or not value or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in value
    ):
        raise ValueError("action values must be non-empty finite numbers")
    return tuple(value)


@dataclass(frozen=True)
class ActionRecord:
    prediction_id: str
    status: str
    space: str
    values: tuple[float, ...]
    waypoint_index: int | None = None
    command_id: int | None = None
    event_id: int | None = None
    source_position: float | None = None
    blend_weight: float = 1.0
    anchor_command_id: int | None = None
    anchor_values: tuple[float, ...] | None = None
    planned_values: tuple[float, ...] | None = None
    recorded_s: float | None = None
    reason: str = ""

    def __post_init__(self):
        if not isinstance(self.prediction_id, str) or not 1 <= len(self.prediction_id) <= 128:
            raise ValueError("action record requires a bounded prediction ID")
        if self.status not in {"planned", "published", "discarded", "failed"}:
            raise ValueError("invalid action status (published does not mean executed)")
        if self.space not in {"model", "waypoint", "command"}:
            raise ValueError("action coordinate space must be explicit")
        if self.waypoint_index is not None and (
            type(self.waypoint_index) is not int or self.waypoint_index < 0
        ):
            raise ValueError("invalid waypoint index")
        object.__setattr__(self, "values", _vector(self.values))
        for name in ("command_id", "event_id", "anchor_command_id"):
            _index(getattr(self, name), name, optional=True)
        for name in ("source_position", "recorded_s"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid {name}")
        if type(self.blend_weight) not in (int, float) or not 0 < self.blend_weight <= 1:
            raise ValueError("invalid blend weight")
        for name in ("anchor_values", "planned_values"):
            value = getattr(self, name)
            if value is not None:
                value = _vector(value)
                if len(value) != len(self.values):
                    raise ValueError(f"{name} dimension mismatch")
                object.__setattr__(self, name, value)
        if self.blend_weight < 1 and (self.anchor_command_id is None or self.anchor_values is None):
            raise ValueError("blended action requires its anchor")
        if not isinstance(self.reason, str) or len(self.reason) > 1024:
            raise ValueError("action reason must be a bounded string")


@dataclass(frozen=True)
class PlanningRecord:
    prediction_id: int
    source_count: int
    source_start: int
    command_count: int
    event_id: int
    recorded_s: float

    def __post_init__(self):
        for name in ("prediction_id", "source_count", "source_start", "command_count", "event_id"):
            _index(getattr(self, name), name)
        if self.source_start > self.source_count:
            raise ValueError("alignment prefix exceeds source chunk")
        if type(self.recorded_s) not in (int, float) or not math.isfinite(self.recorded_s) or self.recorded_s < 0:
            raise ValueError("invalid planning time")


@dataclass(frozen=True)
class ResetRecord:
    event_id: int
    reason: str
    recorded_s: float

    def __post_init__(self):
        _index(self.event_id, "event_id")
        if not isinstance(self.reason, str) or len(self.reason) > 1024:
            raise ValueError("reset reason must be a bounded string")
        if type(self.recorded_s) not in (int, float) or not math.isfinite(self.recorded_s) or self.recorded_s < 0:
            raise ValueError("invalid reset time")


@dataclass(frozen=True)
class ExecutionContext:
    session_id: str
    generation: int
    revision: int
    phase: str
    actions: tuple[ActionRecord, ...] = ()
    planning: tuple[PlanningRecord, ...] = ()
    resets: tuple[ResetRecord, ...] = ()
    after_event_id: int = 0
    latest_event_id: int = 0

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not 1 <= len(self.session_id) <= 128:
            raise ValueError("execution context requires a bounded session ID")
        if any(type(v) is not int or v < 0 for v in (self.generation, self.revision)):
            raise ValueError("generation and revision must be non-negative integers")
        if self.phase not in {"ready", "running", "syncing", "paused", "stopped", "error"}:
            raise ValueError("invalid execution phase")
        if not isinstance(self.actions, (tuple, list)) or any(
            not isinstance(a, ActionRecord) for a in self.actions
        ):
            raise ValueError("actions must be ActionRecord instances")
        object.__setattr__(self, "actions", tuple(self.actions))
        for name, kind in (("planning", PlanningRecord), ("resets", ResetRecord)):
            items = getattr(self, name)
            if not isinstance(items, (tuple, list)) or any(not isinstance(item, kind) for item in items):
                raise ValueError(f"invalid {name} records")
            object.__setattr__(self, name, tuple(items))
        _index(self.after_event_id, "after_event_id")
        _index(self.latest_event_id, "latest_event_id")
        ids = [a.event_id for a in self.actions if a.event_id is not None]
        ids += [item.event_id for item in (*self.planning, *self.resets)]
        if len(ids) != self.latest_event_id - self.after_event_id or any(
            event_id != self.after_event_id + offset
            for offset, event_id in enumerate(sorted(ids), start=1)
        ):
            raise ValueError("execution feedback must contain a contiguous event interval")
        if self.after_event_id > self.latest_event_id:
            raise ValueError("execution event cursor moved backwards")

    def to_json(self) -> str:
        result = json.dumps(self, default=_json_record, allow_nan=False, sort_keys=True, separators=(",", ":"))
        encoded = result.encode("utf-8")
        if len(encoded) > MAX_EXPANDED_CONTEXT_BYTES:
            raise ValueError("expanded execution context exceeds 8 MiB")
        if len(encoded) > CONTEXT_COMPRESSION_THRESHOLD_BYTES:
            # ZOH publications repeat vectors, but every receipt/time must survive.
            # Lossless wire compression avoids dropping facts during slow inference.
            compressed = json.dumps({
                "encoding": "zlib+base64",
                "payload": base64.b64encode(zlib.compress(encoded)).decode("ascii"),
            }, separators=(",", ":"))
            # Keep the single service request bounded even when compression plus
            # base64 would expand incompressible data. The decoded budget is the
            # same for either representation; no receipt may be dropped to fit.
            if len(compressed) < len(encoded):
                result = compressed
        return result

    @classmethod
    def from_json(cls, raw: str):
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise ValueError("execution context must be JSON within 8 MiB")
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("execution context must be an object")
            if data.get("encoding") == "zlib+base64":
                if set(data) != {"encoding", "payload"} or not isinstance(data["payload"], str):
                    raise ValueError("invalid compressed execution context")
                compressed = base64.b64decode(data["payload"], validate=True)
                decoder = zlib.decompressobj()
                decoded = decoder.decompress(compressed, MAX_EXPANDED_CONTEXT_BYTES + 1)
                if (len(decoded) > MAX_EXPANDED_CONTEXT_BYTES or not decoder.eof
                        or decoder.unused_data or decoder.unconsumed_tail):
                    raise ValueError("invalid or oversized expanded execution context")
                data = json.loads(decoded)
                if not isinstance(data, dict):
                    raise ValueError("expanded execution context must be an object")
            actions = data.pop("actions", [])
            if not isinstance(actions, list):
                raise ValueError("execution actions must be a list")
            planning = data.pop("planning", [])
            resets = data.pop("resets", [])
            if not isinstance(planning, list) or not isinstance(resets, list):
                raise ValueError("execution feedback records must be lists")
            return cls(**data, actions=tuple(ActionRecord(**a) for a in actions),
                       planning=tuple(PlanningRecord(**p) for p in planning),
                       resets=tuple(ResetRecord(**r) for r in resets))
        except (TypeError, json.JSONDecodeError, RecursionError, binascii.Error,
                zlib.error, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid execution context: {exc}") from exc
