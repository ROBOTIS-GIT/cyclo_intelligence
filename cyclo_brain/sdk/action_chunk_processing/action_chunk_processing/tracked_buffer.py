"""Opt-in command planning and publication records; no robot or model imports.

The caller serializes take/publish/finish with safety transitions. Dequeuing is
not publication, and publication is never proof of physical robot execution.
"""

from collections import deque
from dataclasses import dataclass, replace
from itertools import islice
import threading
import time

import numpy as np

from .action_chunk_processor import ActionChunkProcessor


@dataclass(frozen=True)
class PlannedCommand:
    command_id: int
    prediction_id: int
    values: tuple[float, ...]
    dtype: str
    source_position: float
    blend_weight: float
    anchor_command_id: int | None
    anchor_values: tuple[float, ...] | None


@dataclass(frozen=True)
class AlignmentDecision:
    prediction_id: int
    source_count: int
    source_start: int
    command_count: int


@dataclass(frozen=True)
class PublicationEvent:
    event_id: int
    command: PlannedCommand
    status: str
    recorded_s: float
    emitted_values: tuple[float, ...] | None = None
    reason: str = ""


@dataclass(frozen=True)
class PlanningEvent:
    event_id: int
    decision: AlignmentDecision
    recorded_s: float


@dataclass(frozen=True)
class ResetEvent:
    event_id: int
    reason: str
    recorded_s: float


@dataclass(frozen=True)
class ExecutionSnapshot:
    revision: int
    latest_event_id: int
    pending: tuple[PlannedCommand, ...]
    in_flight: PlannedCommand | None
    events: tuple[PublicationEvent | PlanningEvent | ResetEvent, ...]
    error: str


class TrackedActionBuffer:
    """Legacy numerical processing with explicit, bounded execution bookkeeping.

    A scheduler must opt in and
    report publisher results, including any deadband-adjusted command values.
    Events are bounded; consumers that lag beyond retention must reset context
    instead of mistaking incomplete history for a complete execution record.
    """

    def __init__(self, *, max_commands=4096, max_events=4096, max_dimensions=256, **processing):
        for value in (max_commands, max_events, max_dimensions):
            if type(value) is not int or value <= 0:
                raise ValueError("buffer limits must be positive integers")
        self._pipeline = ActionChunkProcessor(**processing)
        self._max_commands = max_commands
        self._max_dimensions = max_dimensions
        self._pending = deque()
        self._events = deque(maxlen=max_events)
        self._in_flight = None
        self._last_taken = None
        self._last_published = None
        self._last_prediction_id = -1
        self._next_command_id = 0
        self._event_id = 0
        self._revision = 0
        self._failure = ""
        self._lock = threading.Lock()

    @property
    def buffer_size(self):
        with self._lock:
            return len(self._pending)

    @property
    def output_hz(self):
        return self._pipeline.output_hz

    def enqueue(self, prediction_id: int, actions: np.ndarray,
                scheduled_start_delay_s=None, align=True) -> AlignmentDecision:
        values = np.asarray(actions)
        if (values.ndim != 2 or values.shape[0] > self._max_commands
                or not 0 < values.shape[1] <= self._max_dimensions
                or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
            raise ValueError("actions must be finite (T,D) within buffer limits")
        if type(prediction_id) is not int or prediction_id < 0:
            raise ValueError("prediction_id must be a non-negative integer")
        with self._lock:
            if self._failure:
                raise RuntimeError(f"publication failed; clear required: {self._failure}")
            if prediction_id <= self._last_prediction_id:
                raise ValueError("prediction IDs must increase; duplicate or stale response")
            anchor = self._pending[-1] if self._pending else self._last_taken
            if anchor is not None and len(anchor.values) != values.shape[1]:
                raise ValueError("action dimensions changed; clear the previous plan first")
            anchor_values = None if anchor is None else np.asarray(anchor.values, dtype=anchor.dtype)
            # Bound resampling before allocating a potentially amplified output.
            count = len(values)
            pipeline = self._pipeline
            if pipeline._postprocess and count:
                count = pipeline._target_chunk_size or (
                    max(1, int(round((count - 1) / pipeline._inference_hz * pipeline._control_hz)))
                    if count > 1 else 1
                )
            if len(self._pending) + count > self._max_commands:
                raise ValueError("pending command budget exceeded")
            result = pipeline.prepare_chunk(values, anchor_values, scheduled_start_delay_s, align)
            positions = pipeline.source_positions(len(values) - result.source_start) + result.source_start
            weights = pipeline.blend_weights(len(result.actions), anchor is not None)
            if not np.isfinite(result.actions).all():
                raise ValueError("processed actions must be finite")
            commands = []
            for index, (action, position, weight) in enumerate(zip(result.actions, positions, weights)):
                blending = weight != 1.
                commands.append(PlannedCommand(
                    self._next_command_id + index, prediction_id,
                    tuple(float(v) for v in action), str(action.dtype), float(position), float(weight),
                    anchor.command_id if blending else None,
                    anchor.values if blending else None,
                ))
            self._pending.extend(commands)
            self._next_command_id += len(commands)
            self._last_prediction_id = prediction_id
            self._revision += 1
            decision = AlignmentDecision(prediction_id, len(values), result.source_start, len(commands))
            self._event_id += 1
            self._events.append(PlanningEvent(self._event_id, decision, time.monotonic()))
            return decision

    def take(self, *, repeat_last=False) -> PlannedCommand | None:
        with self._lock:
            if self._failure:
                raise RuntimeError(f"publication failed; clear required: {self._failure}")
            if self._in_flight is not None:
                raise RuntimeError("finish the outstanding publication before taking another action")
            if not self._pending:
                if not repeat_last or self._last_published is None:
                    return None
                # ZOH repeats retain source provenance but have their own receipt.
                command = replace(self._last_published, command_id=self._next_command_id)
                self._next_command_id += 1
            else:
                command = self._pending.popleft()
            self._in_flight = command
            self._last_taken = command
            self._revision += 1
            return command

    def finish(self, command_id: int, *, status: str, emitted_values=None, reason="") -> None:
        """Finish an attempted publication, not a physical execution ACK.

        Preview-only output is discarded here, not published to a robot. If a
        multi-topic publish partially fails, mark failed and preserve the error;
        never claim that the entire vector was sent successfully.
        """
        if status not in {"published", "failed", "discarded"}:
            raise ValueError("status must be published, failed or discarded")
        if not isinstance(reason, str) or len(reason) > 1024:
            raise ValueError("reason must be a bounded string")
        if status == "published":
            if emitted_values is None:
                raise ValueError("publication requires actual emitted command values")
            emitted = np.asarray(emitted_values)
            if emitted.ndim != 1 or emitted.dtype.kind not in "iuf" or not np.isfinite(emitted).all():
                raise ValueError("emitted command must be a finite vector")
            emitted_values = tuple(float(v) for v in emitted)
        elif emitted_values is not None:
            raise ValueError("failed/discarded commands must not claim full publication")
        with self._lock:
            command = self._in_flight
            if command is None or command.command_id != command_id:
                raise ValueError("unknown or already completed command")
            if emitted_values is not None and len(emitted_values) != len(command.values):
                raise ValueError("emitted command dimension mismatch")
            self._record(command, status, emitted_values, reason)
            if status == "published":
                self._last_published = command
            if status == "failed":
                self._failure = reason or "publisher failure"
            self._in_flight = None
            self._revision += 1

    def clear(self, reason="reset") -> None:
        """Discard the remaining plan; do not erase or falsify publication history."""
        if not isinstance(reason, str) or len(reason) > 1024:
            raise ValueError("reason must be a bounded string")
        with self._lock:
            if self._in_flight is not None:
                raise RuntimeError("resolve outstanding publication before clearing the plan")
            for command in self._pending:
                self._record(command, "discarded", None, reason)
            self._pending.clear()
            self._last_taken = None
            self._last_published = None
            self._failure = ""
            self._revision += 1
            self._event_id += 1
            self._events.append(ResetEvent(self._event_id, reason, time.monotonic()))

    def snapshot(self, *, after_event_id=0, pending_limit=None) -> ExecutionSnapshot:
        """Return the remaining plan and terminal events atomically."""
        if pending_limit is not None and (type(pending_limit) is not int or not 0 <= pending_limit <= 4096):
            raise ValueError("pending limit must be 0..4096 or None")
        with self._lock:
            if type(after_event_id) is not int or not 0 <= after_event_id <= self._event_id:
                raise ValueError("invalid event cursor")
            if self._events and after_event_id < self._events[0].event_id - 1:
                raise RuntimeError("execution history gap; reset consumer context")
            if self._events and after_event_id < self._events[0].event_id:
                events = tuple(self._events)
            else:
                # ACK cursors usually leave only a small suffix. Do not scan
                # thousands of already acknowledged receipts under the tick lock.
                newest = []
                for event in reversed(self._events):
                    if event.event_id <= after_event_id:
                        break
                    newest.append(event)
                events = tuple(reversed(newest))
            return ExecutionSnapshot(
                self._revision, self._event_id, tuple(islice(self._pending, pending_limit)), self._in_flight,
                events,
                self._failure,
            )

    def _record(self, command, status, emitted_values, reason):
        self._event_id += 1
        self._events.append(PublicationEvent(
            self._event_id, command, status, time.monotonic(), emitted_values, reason,
        ))
