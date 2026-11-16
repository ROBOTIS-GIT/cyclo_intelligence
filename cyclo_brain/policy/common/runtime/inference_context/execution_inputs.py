"""Opt-in execution input retention, owned by the serialized Worker session."""

from collections import deque
from itertools import islice
import math

from .execution import ActionRecord, ExecutionContext
from .inputs import ExecutionQuery


class ExecutionInputUnavailable(RuntimeError):
    """A new execution context is needed; waiting for robot topics cannot fix it."""


class ExecutionInputs:
    def __init__(self, spec):
        self._queries = frozenset(q for field in spec.fields for q in field.queries
                                  if isinstance(q, ExecutionQuery))
        self.sources = frozenset(q.source for q in self._queries)
        capacities = {source: max(q.count for q in self._queries if q.source == source)
                      for source in self.sources}
        self._buffers = {source: deque(maxlen=capacities[source]) for source in capacities
                         if source in {"execution:published:command", "execution:events"}}
        self._pending_limit = capacities.get("execution:pending:command", 0)
        self._pending = ()
        self._context = None

    @property
    def retained_record_count(self):
        return sum(len(records) for records in self._buffers.values()) + len(self._pending)

    def update(self, context):
        if not isinstance(context, ExecutionContext):
            raise ValueError("execution inputs require a validated context")
        previous = self._context
        if previous is None:
            if context.phase != "ready" or context.latest_event_id or context.actions:
                raise ValueError("execution inputs must start with an empty ready LOAD context")
        else:
            if context.session_id != previous.session_id:
                raise ValueError("execution input session changed without LOAD")
            if (context.generation, context.revision) < (previous.generation, previous.revision):
                raise ValueError("stale execution input context")
            if (context.generation, context.revision) == (previous.generation, previous.revision):
                if context != previous:
                    raise ValueError("execution input revision reused with different facts")
                return
            if context.after_event_id > previous.latest_event_id or context.latest_event_id < previous.latest_event_id:
                raise ValueError("execution input cursor gap or backwards movement")
        reset = previous is None or context.generation != previous.generation or (
            context.phase != previous.phase and context.phase in {"paused", "stopped", "error"}
        )
        cursor = 0 if previous is None else previous.latest_event_id
        if reset:
            # A reset context may carry unacknowledged publications from the old
            # generation. Preserve its reset boundary, never replay old commands.
            cursor = max(cursor, max((r.event_id for r in context.resets), default=context.latest_event_id))
            for records in self._buffers.values():
                records.clear()
            marker = next((r for r in context.resets if r.event_id == cursor), None)
            if marker is not None and "execution:events" in self._buffers:
                self._buffers["execution:events"].append(marker)
        if self._buffers:
            events = [a for a in context.actions if a.event_id is not None and a.event_id > cursor]
            events.extend(r for r in (*context.planning, *context.resets) if r.event_id > cursor)
            for event in sorted(events, key=lambda value: value.event_id):
                if "execution:events" in self._buffers:
                    self._buffers["execution:events"].append(event)
                if ("execution:published:command" in self._buffers and isinstance(event, ActionRecord)
                        and event.status == "published" and event.space == "command"):
                    self._buffers["execution:published:command"].append(event)
        if self._pending_limit:
            self._pending = tuple(islice(
                (a for a in context.actions if a.status == "planned" and a.space == "command"),
                self._pending_limit,
            ))
        self._context = context

    def resolve(self, query, anchor_s):
        if query not in self._queries:
            raise ValueError("execution query was not declared at LOAD")
        if self._context is None:
            raise ExecutionInputUnavailable("execution inputs have no LOAD context")
        if not math.isfinite(anchor_s):
            raise ValueError("execution input anchor must be finite")
        if query.source == "execution:context":
            return (self._context,)
        records = (self._pending[:query.count] if query.source == "execution:pending:command"
                   else tuple(self._buffers[query.source])[-query.count:])
        if query.max_age_s is not None:
            if any(r.recorded_s is None or r.recorded_s > anchor_s for r in records):
                raise ExecutionInputUnavailable("execution timestamps must use the local monotonic clock")
            records = tuple(r for r in records if anchor_s - r.recorded_s <= query.max_age_s)
        if len(records) < query.min_count:
            raise ExecutionInputUnavailable(f"{query.source}: need {query.min_count} records, have {len(records)}")
        return (records,)

    def clear(self):
        for records in self._buffers.values():
            records.clear()
        self._pending = ()
        self._context = None
