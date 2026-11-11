"""Declarative input assembly. No robot, model, Torch or transport dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class InputSample:
    value: Any
    source: str
    sequence: Any = None
    received_s: float | None = None


@dataclass(frozen=True)
class SampleQuery:
    source: str
    offsets_s: tuple[float, ...] = (0.0,)
    max_age_s: float | None = None

    def __post_init__(self):
        if not self.source or not self.offsets_s:
            raise ValueError("sample query needs a source and at least one offset")
        if any(not math.isfinite(v) or v > 0 for v in self.offsets_s):
            raise ValueError("sample offsets must be finite and non-positive")
        if tuple(sorted(set(self.offsets_s))) != self.offsets_s:
            raise ValueError("sample offsets must be unique and chronological")
        if self.max_age_s is not None and (
            not math.isfinite(self.max_age_s) or self.max_age_s <= 0
        ):
            raise ValueError("max_age_s must be finite and positive")

    @property
    def sample_count(self):
        return len(self.offsets_s)


@dataclass(frozen=True)
class ExecutionQuery:
    """A bounded collection of execution facts, not fabricated sensor samples.

    Published records are command-space transport receipts, not measured robot
    motion. The transform receives the collection as one input value; bootstrap
    behavior must be explicit via min_count=0 and the model's transform.
    """

    source: str
    count: int = 1
    min_count: int = 1
    max_age_s: float | None = None

    def __post_init__(self):
        if self.source not in {"execution:published:command", "execution:pending:command",
                               "execution:events", "execution:context"}:
            raise ValueError("execution source must explicitly name its facts and coordinate space")
        if (type(self.count) is not int or not 1 <= self.count <= 4096
                or type(self.min_count) is not int or not 0 <= self.min_count <= self.count):
            raise ValueError("execution count must be 1..4096 and min_count within count")
        if self.max_age_s is not None and (
            self.source not in {"execution:published:command", "execution:events"}
            or type(self.max_age_s) not in (int, float)
            or not math.isfinite(self.max_age_s) or self.max_age_s <= 0
        ):
            raise ValueError("only terminal execution facts support a positive finite maximum age")
        if self.source == "execution:context" and (self.count != 1 or self.min_count != 1):
            raise ValueError("execution context is one required snapshot")

    @property
    def sample_count(self):
        return 1


@dataclass(frozen=True)
class InputField:
    key: str
    queries: tuple[SampleQuery | ExecutionQuery, ...]
    transform: str = "identity"
    shape: tuple[int | None, ...] | None = None

    def __post_init__(self):
        if not self.key or not self.queries or not self.transform:
            raise ValueError("input field needs a key, queries and transform")
        if any(not isinstance(q, (SampleQuery, ExecutionQuery)) for q in self.queries):
            raise TypeError("input queries must be SampleQuery or ExecutionQuery")
        if self.shape is not None and any(
            v is not None and (type(v) is not int or v <= 0) for v in self.shape
        ):
            raise ValueError("shape dimensions must be positive integers or None")


@dataclass(frozen=True)
class InputSpec:
    fields: tuple[InputField, ...]

    def __post_init__(self):
        keys = [field.key for field in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError("input keys must be unique")

    @property
    def sources(self) -> frozenset[str]:
        return frozenset(q.source for f in self.fields for q in f.queries)


class InputProvider(Protocol):
    def resolve(self, query: SampleQuery | ExecutionQuery, anchor_s: float) -> tuple[Any, ...]: ...


class ResolvedInputs:
    """Resolve a request once before transforms allocate tensors or mutate state."""

    def __init__(self, spec, provider, anchor_s):
        self._anchor_s = anchor_s
        self._samples = {}
        self._records = {}
        for field in spec.fields:
            for query in field.queries:
                if query not in self._samples:
                    resolve_samples = getattr(provider, "resolve_samples", None)
                    if callable(resolve_samples):
                        records = resolve_samples(query, anchor_s)
                        self._records[query] = records
                        samples = tuple(record.value for record in records)
                    else:
                        samples = provider.resolve(query, anchor_s)
                    if len(samples) != query.sample_count:
                        raise ValueError(f"{query.source}: provider returned the wrong sample count")
                    self._samples[query] = samples

    def resolve(self, query, anchor_s):
        if anchor_s != self._anchor_s or query not in self._samples:
            raise ValueError("input snapshot query/anchor differs from its resolved request")
        return self._samples[query]

    def resolve_samples(self, query, anchor_s):
        values = self.resolve(query, anchor_s)
        return self._records.get(query, tuple(InputSample(v, query.source) for v in values))


class RoutedInputs:
    """Bind requested sources to observation, execution or instruction providers.

    Resolution uses one caller-supplied clock anchor. There is no source-prefix
    guessing or fallback to a different provider when history is unavailable.
    """

    def __init__(self, spec: InputSpec, providers: Mapping[str, InputProvider]):
        if set(providers) != spec.sources:
            raise ValueError("provider bindings must match the declared input sources exactly")
        self._providers = dict(providers)

    def resolve(self, query: SampleQuery, anchor_s: float) -> tuple[Any, ...]:
        if query.source not in self._providers:
            raise ValueError(f"Undeclared input source: {query.source}")
        return self._providers[query.source].resolve(query, anchor_s)

    def resolve_samples(self, query, anchor_s):
        provider = self._providers[query.source]
        method = getattr(provider, "resolve_samples", None)
        if callable(method):
            return method(query, anchor_s)
        return tuple(InputSample(v, query.source) for v in provider.resolve(query, anchor_s))


class LatestValues:
    """Adapter for legacy snapshots. Never pretend a latest value is history."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = values

    def resolve(self, query: SampleQuery, anchor_s: float) -> tuple[Any, ...]:
        if query.offsets_s != (0.0,) or query.max_age_s is not None:
            raise ValueError(f"{query.source}: timestamped samples are required")
        if query.source not in self._values or self._values[query.source] is None:
            raise ValueError(f"Missing input source: {query.source}")
        return (self._values[query.source],)


class ReceivedValues:
    """Latest snapshot with callback times in the caller's monotonic clock.

    An after_s barrier means received after publication, not physical execution
    or camera exposure after a command. Remote clocks must not be used here.
    This provider retains no history and performs no timestamp alignment.
    """

    def __init__(self, values, received_s, *, after_s=None):
        if after_s is not None and (
            type(after_s) not in (int, float) or not math.isfinite(after_s) or after_s < 0
        ):
            raise ValueError("observation barrier must be a finite monotonic timestamp")
        self._values = values
        self._received_s = received_s
        self._after_s = after_s

    def resolve(self, query: SampleQuery, anchor_s: float) -> tuple[Any, ...]:
        if query.offsets_s != (0.0,):
            raise ValueError(f"{query.source}: observation history is required")
        if not math.isfinite(anchor_s):
            raise ValueError("anchor must be finite")
        stamp = self._received_s.get(query.source)
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or stamp < 0 or stamp > anchor_s):
            raise ValueError(f"{query.source}: missing or invalid reception timestamp")
        if self._after_s is not None and stamp <= self._after_s:
            raise ValueError(f"{query.source}: observation predates publication barrier {self._after_s:g}")
        if query.max_age_s is not None and anchor_s - stamp > query.max_age_s:
            raise ValueError(f"{query.source}: stale observation age={anchor_s - stamp:.3f}s")
        if query.source not in self._values or self._values[query.source] is None:
            raise ValueError(f"Missing input source: {query.source}")
        return (self._values[query.source],)

    def resolve_samples(self, query, anchor_s):
        values = self.resolve(query, anchor_s)
        stamp = self._received_s[query.source]
        return tuple(InputSample(value, query.source, stamp, stamp) for value in values)


def _identity(values: tuple[Any, ...]) -> Any:
    if len(values) != 1:
        raise ValueError("identity needs exactly one sample; declare a temporal/concat transform")
    return values[0]


class InputAssembler:
    def __init__(
        self,
        spec: InputSpec,
        transforms: Mapping[str, Callable[[tuple[Any, ...]], Any]] | None = None,
    ):
        self.spec = spec
        self._transforms = {"identity": _identity, **(transforms or {})}
        for field in spec.fields:
            if field.transform not in self._transforms:
                raise ValueError(f"Unknown input transform: {field.transform}")

    def assemble(self, provider: InputProvider, *, anchor_s: float = 0.0) -> dict[str, Any]:
        if not math.isfinite(anchor_s):
            raise ValueError("anchor must be finite")
        if not isinstance(provider, ResolvedInputs):
            provider = ResolvedInputs(self.spec, provider, anchor_s)
        result = {}
        for field in self.spec.fields:
            values = []
            for query in field.queries:
                samples = provider.resolve(query, anchor_s)
                if len(samples) != query.sample_count:
                    raise ValueError(f"{query.source}: provider returned the wrong sample count")
                values.extend(samples)
            value = self._transforms[field.transform](tuple(values))
            if field.shape is not None:
                actual = tuple(getattr(value, "shape", ()))
                if len(actual) != len(field.shape) or any(
                    expected is not None and expected != got
                    for expected, got in zip(field.shape, actual)
                ):
                    raise ValueError(f"{field.key}: expected shape {field.shape}, got {actual}")
            result[field.key] = value
        return result
