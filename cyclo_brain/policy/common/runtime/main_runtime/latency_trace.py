"""Optional, bounded JSONL tracing with all disk work on a writer thread.

Payloads should contain immutable JSON-compatible scalar values. ``emit`` only
copies the top-level payload; callers must not subsequently mutate nested data.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
import queue
import sys
import threading
import time
import uuid


class AsyncLatencyTrace:
    """Append versioned events without doing disk I/O in the control thread.

    With no path and no INFERENCE_TRACE_PATH this is a disabled no-op. Queue overflow
    drops events instead of delaying control. Counters, stderr warnings, and the
    closing record make that loss visible. Call close outside control locks.
    """

    def __init__(self, path=None, *, queue_size: int = 2048):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        selected = path if path is not None else os.environ.get("INFERENCE_TRACE_PATH")
        self.path = Path(selected).expanduser() if selected else None
        self.trace_id = uuid.uuid4().hex
        self._queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._sequence = 0
        self._accepted = 0
        self._written = 0
        self._dropped = 0
        self._serialization_errors = 0
        self._writer_error = None
        self._thread = None
        if self.path is not None:
            self._thread = threading.Thread(
                target=self._run, name="inference-latency-trace", daemon=True
            )
            self._thread.start()

    @property
    def enabled(self):
        return self.path is not None

    @property
    def stats(self):
        with self._lock:
            return {
                "accepted_events": self._accepted,
                "written_events": self._written,
                "dropped_events": self._dropped,
                "serialization_errors": self._serialization_errors,
                "writer_error": self._writer_error,
            }

    def _record(self, event, payload, sequence):
        return {
            **payload,
            "schema_version": 1,
            "trace_id": self.trace_id,
            "sequence": sequence,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
        }

    def emit(self, event: str, **payload) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._closed.is_set() or self._writer_error is not None:
                return False
            self._sequence += 1
            record = self._record(event, payload, self._sequence)
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1
                return False
            self._accepted += 1
            return True

    @staticmethod
    def _warn(message):
        print(f"Inference trace: {message}", file=sys.stderr, flush=True)

    @staticmethod
    def _write(stream, record):
        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _run(self):
        warned_drops = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                self._write(stream, self._record("trace_started", {}, 0))
                while not self._closed.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        record = None
                    if record is not None:
                        try:
                            self._write(stream, record)
                        except (TypeError, ValueError) as exc:
                            with self._lock:
                                self._serialization_errors += 1
                            self._warn(f"event could not be serialized: {exc}")
                        else:
                            with self._lock:
                                self._written += 1
                        finally:
                            self._queue.task_done()
                    if not warned_drops and self.stats["dropped_events"]:
                        warned_drops = True
                        self._warn("queue overflow; events were dropped, trace is incomplete")
                stats = self.stats
                self._write(stream, self._record("trace_closed", stats, self._sequence + 1))
                stream.flush()
                if stats["dropped_events"]:
                    self._warn(f"dropped {stats['dropped_events']} events in total")
        except Exception as exc:
            with self._lock:
                self._writer_error = f"{type(exc).__name__}: {exc}"
            self._warn(f"writer failed; trace is incomplete: {exc}")
        finally:
            # Release references after an I/O failure; queue capacity is always bounded.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def close(self, timeout: float = 5.0) -> bool:
        """Drain/flush accepted events; return false for loss, I/O error or timeout.

        A timeout leaves the daemon writer draining; a later close may wait again.
        The trace lacks trace_closed until the writer has actually flushed it.
        """
        with self._lock:
            self._closed.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            self._warn("close timed out; writer has not finished flushing")
            return False
        stats = self.stats
        return not (
            stats["dropped_events"]
            or stats["serialization_errors"]
            or stats["writer_error"]
        )


class BufferedActionTrace:
    """Attribute an append-only action queue to its source requests.

    The controller must serialize calls and call reset whenever it clears the
    action queue. Markers disappear as actions are popped; no history is kept.
    """

    def __init__(self, trace):
        self.trace = trace
        self._markers = deque()
        self._popped = 0
        self._current = None
        self._published = None
        self._has_published = False
        self._underflow = None

    def reset(self, now):
        self._end_underflow(now)
        self._markers.clear()
        self._popped = 0
        self._current = None
        self._published = None
        self._has_published = False

    def installed(self, request_id, generation, queued_steps, produced_steps, **payload):
        if produced_steps <= 0:
            return
        self._markers.append((self._popped + queued_steps, generation, request_id))
        payload.setdefault("installed_monotonic_s", time.monotonic())
        self.trace.emit("chunk_installed", request_id=request_id, generation=generation,
                        queued_control_steps=queued_steps, produced_control_steps=produced_steps,
                        **payload)

    def popped(self, has_action, now, generation):
        if not has_action:
            if self._has_published and self._underflow is None:
                self._underflow = (now, generation)
                self.trace.emit("queue_underflow", generation=generation, phase="start")
            return
        self._end_underflow(now)
        while self._markers and self._markers[0][0] <= self._popped:
            _, marker_generation, request_id = self._markers.popleft()
            self._current = (marker_generation, request_id)
        self._popped += 1

    def published(self):
        self._has_published = True
        if self._current is not None and self._current != self._published:
            generation, request_id = self._current
            self.trace.emit("command_published", generation=generation,
                            request_id=request_id, boundary=True)
            self._published = self._current

    def _end_underflow(self, now):
        if self._underflow is not None:
            started, generation = self._underflow
            self.trace.emit("queue_underflow", generation=generation, phase="end",
                            duration_s=max(0.0, now - started))
            self._underflow = None
