"""Opt-in, bounded numeric sample storage driven by input requirements.

Clock values must be from the same monotonic clock domain. Source timestamps
are metadata, not substitutes for local reception times.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading

import numpy as np

from .inputs import InputSpec, SampleQuery


@dataclass(frozen=True)
class Sample:
    sequence: int
    received_s: float
    value: np.ndarray


class HistoryStore:
    def __init__(self, spec: InputSpec, *, max_bytes: int = 256 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._retention = {}
        for field in spec.fields:
            for q in field.queries:
                if q.offsets_s != (0.0,) and q.max_age_s is None:
                    raise ValueError("history queries require an explicit maximum sample age")
                span = -min(q.offsets_s) + (q.max_age_s or 0.0)
                self._retention[q.source] = max(span, self._retention.get(q.source, 0.0))
        self._samples = {source: deque() for source in self._retention}
        self._last_seen = {}
        self._max_bytes = max_bytes
        self._bytes = 0
        self._failure = ""
        self._lock = threading.Lock()

    @property
    def sources(self) -> frozenset[str]:
        return frozenset(self._samples)

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes

    def append(self, source: str, sequence: int, received_s: float, value: np.ndarray) -> None:
        if source not in self._samples:
            return
        if type(sequence) is not int or sequence < 0 or not math.isfinite(received_s):
            raise ValueError("invalid sample identity or reception time")
        data = np.asarray(value)
        if data.dtype.kind not in "biuf" or data.size == 0:
            raise ValueError("history stores non-empty numeric arrays")
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            samples = self._samples[source]
            previous = self._last_seen.get(source)
            if previous is not None and sequence <= previous[0]:
                raise ValueError("sample sequences must increase; do not resample a cached frame")
            if previous is not None and received_s < previous[1]:
                raise ValueError("sample reception time moved backwards")
            # Discard only samples outside the declared window, across all sources.
            for key, items in self._samples.items():
                if self._retention[key] == 0:
                    continue
                cutoff = received_s - self._retention[key]
                while items and items[0].received_s < cutoff:
                    self._bytes -= items.popleft().value.nbytes
            if self._retention[source] == 0:
                while samples:
                    self._bytes -= samples.popleft().value.nbytes
            if self._bytes + data.nbytes > self._max_bytes or len(samples) >= 4096:
                self._failure = "observation history memory budget exceeded; reset required"
                raise MemoryError(self._failure)
            data = data.copy()
            data.setflags(write=False)
            samples.append(Sample(sequence, received_s, data))
            self._last_seen[source] = (sequence, received_s)
            self._bytes += data.nbytes

    def resolve(self, query: SampleQuery, anchor_s: float, *, after_s=None) -> tuple[np.ndarray, ...]:
        selected = self._select(query, anchor_s, after_s=after_s)
        # Stored arrays are immutable and retained by these references. Neither
        # reception nor a reset needs to wait for the caller's image copies.
        return tuple(s.value.copy() for s in selected)

    def check(self, query: SampleQuery, anchor_s: float, *, after_s=None) -> None:
        """Readiness only, without copying retained images or state arrays."""
        self._select(query, anchor_s, after_s=after_s)

    def _select(self, query, anchor_s, *, after_s):
        if not math.isfinite(anchor_s):
            raise ValueError("anchor must be finite")
        if after_s is not None and (type(after_s) not in (int, float)
                                   or not math.isfinite(after_s) or after_s < 0):
            raise ValueError("history barrier must be a finite monotonic timestamp")
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            if query.source not in self._samples:
                raise ValueError(f"Undeclared history source: {query.source}")
            needed = -min(query.offsets_s) + (query.max_age_s or 0.0)
            if needed > self._retention[query.source]:
                raise ValueError("query exceeds declared history; reconfigure and warm up first")
            selected = []
            for offset in query.offsets_s:
                target = anchor_s + offset
                sample = next((s for s in reversed(self._samples[query.source]) if s.received_s <= target), None)
                if sample is None or (
                    query.max_age_s is not None and target - sample.received_s > query.max_age_s
                ):
                    raise ValueError(f"Missing or stale input: {query.source} at offset {offset}")
                if selected and selected[-1].sequence == sample.sequence:
                    raise ValueError(f"Insufficient distinct samples: {query.source}")
                if offset == 0 and after_s is not None and sample.received_s <= after_s:
                    raise ValueError(f"{query.source}: observation predates publication barrier {after_s:g}")
                selected.append(sample)
            return tuple(selected)

    def reset(self) -> None:
        with self._lock:
            for samples in self._samples.values():
                samples.clear()
            self._bytes = 0
            self._failure = ""
            self._last_seen.clear()
