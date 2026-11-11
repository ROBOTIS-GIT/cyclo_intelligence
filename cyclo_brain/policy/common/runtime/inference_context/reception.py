"""Bounded reception sink/provider for RobotClient's opt-in capture API."""

import threading

from .history import HistoryStore
from .inputs import InputSpec, SampleQuery


class ReceptionHistory:
    """Retain only real callback samples; never seed history from latest getters.

    Camera sources are RGB, joint sources follow RobotClient group ordering and
    sensor sources select numeric fields (e.g. sensor:odom.linear_velocity).
    Instructions and execution facts belong to different providers.
    """

    def __init__(self, spec: InputSpec, *, max_bytes: int = 256 * 1024 * 1024, budget=None):
        self._store = HistoryStore(spec, max_bytes=max_bytes, budget=budget)
        self._failure = ""
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def sources(self) -> frozenset[str]:
        return self._store.sources

    @property
    def bytes_used(self) -> int:
        return self._store.bytes_used

    def record(self, source, sequence, received_s, value) -> None:
        with self._lock:
            if self._failure:
                return
            try:
                self._store.append(source, sequence, received_s, value)
            except Exception as exc:
                # Subscriber callbacks must survive failures; inference must not
                # silently keep using the last successfully captured observation.
                self._failure = f"Observation capture failed: {exc}; reset required"

    def resolve(self, query: SampleQuery, anchor_s: float, *, after_s=None, with_metadata=False):
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            generation = self._generation
        method = self._store.resolve_samples if with_metadata else self._store.resolve
        values = method(query, anchor_s, after_s=after_s)
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            if generation != self._generation:
                raise ValueError("observation history reset during snapshot; retry")
        return values

    def resolve_samples(self, query, anchor_s, *, after_s=None):
        return self.resolve(query, anchor_s, after_s=after_s, with_metadata=True)

    def reset(self) -> None:
        """Clear samples; use RobotClient.reset_observation_capture when attached.

        That callback barrier prevents an already-running reception callback
        from refilling the new session with the previous session's sample.
        """
        with self._lock:
            self._store.reset()
            self._failure = ""
            self._generation += 1

    def check(self, query, anchor_s, *, after_s=None):
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            self._store.check(query, anchor_s, after_s=after_s)
