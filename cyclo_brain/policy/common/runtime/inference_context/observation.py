"""Session-scoped robot reception selected by a model-independent InputSpec."""

from dataclasses import replace

from .inputs import ExecutionQuery, InputSpec, LatestValues, RoutedInputs, SampleQuery
from .execution_inputs import ExecutionInputs
from .reception import ReceptionHistory


class _HistoryWindow:
    def __init__(self, history, after_s):
        self.history, self.after_s = history, after_s

    def resolve(self, query, anchor_s):
        return self.history.resolve(query, anchor_s, after_s=self.after_s)


class ObservationSession:
    """Attach callback history only for explicitly temporal robot sources.

    Current-only sources use the existing snapshot provider. No cached frame is
    replayed into history, and no model work runs in a reception callback.
    """

    def __init__(self, robot, spec, *, max_history_bytes=256 * 1024 * 1024):
        self.robot = robot
        self.spec = spec
        self.execution_queries = tuple(q for f in spec.fields for q in f.queries if isinstance(q, ExecutionQuery))
        self.execution = ExecutionInputs(spec) if self.execution_queries else None
        execution_sources = frozenset(q.source for q in self.execution_queries)
        if any(isinstance(q, SampleQuery) and q.source in execution_sources for f in spec.fields for q in f.queries):
            raise ValueError("execution sources require ExecutionQuery, not sensor sampling offsets")
        self.history_sources = frozenset(
            query.source for field in spec.fields for query in field.queries
            if isinstance(query, SampleQuery) and query.offsets_s != (0.0,)
        )
        self.warmup_timeout_s = (1. + max(-q.offsets_s[0] for f in spec.fields for q in f.queries
                                        if q.source in self.history_sources)) if self.history_sources else None
        if self.warmup_timeout_s is not None and self.warmup_timeout_s > 120:
            raise ValueError("observation history exceeds the 120 second warmup budget")
        self._warming_up = self.warmup_timeout_s is not None
        self.live_sources = spec.sources - self.history_sources - execution_sources - {"instruction"}
        self.live_ages = {}
        for field in spec.fields:
            for query in field.queries:
                if query.source in self.live_sources:
                    previous = self.live_ages.get(query.source)
                    age = query.max_age_s
                    self.live_ages[query.source] = age if previous is None else (
                        previous if age is None else min(previous, age)
                    )
        self.history = None
        self._closed = False
        required = {"camera_names": set(), "joint_groups": set(), "sensor_names": set()}
        groups = {"camera": "camera_names", "joint": "joint_groups", "sensor": "sensor_names"}
        for source in spec.sources - execution_sources - {"instruction"}:
            kind, separator, name = source.partition(":")
            if not separator or not name or kind not in groups:
                raise ValueError(f"No observation provider for source: {source}")
            required[groups[kind]].add(name.split(".", 1)[0] if kind == "sensor" else name)
        self.required_observations = {key: sorted(names) for key, names in required.items()}
        if self.history_sources:
            if "instruction" in self.history_sources:
                raise ValueError("instruction history requires an explicit language provider")
            fields = tuple(
                replace(field, queries=tuple(q for q in field.queries if q.source in self.history_sources))
                for field in spec.fields if any(q.source in self.history_sources for q in field.queries)
            )
            history = ReceptionHistory(InputSpec(fields), max_bytes=max_history_bytes)
            attach = getattr(robot, "attach_observation_capture", None)
            if not callable(attach):
                raise ValueError("temporal input requires RobotClient observation capture")
            attach(history)
            self.history = history

    def bind(self, live_provider, instruction, *, after_s=None):
        if self._closed:
            raise RuntimeError("observation session is closed")
        language = LatestValues({"instruction": instruction or ""})
        history = _HistoryWindow(self.history, after_s)
        return RoutedInputs(self.spec, {
            source: (history if source in self.history_sources else
                     self.execution if self.execution is not None and source in self.execution.sources else
                     language if source == "instruction" else live_provider)
            for source in self.spec.sources
        })

    def reset(self):
        if self._closed:
            raise RuntimeError("observation session is closed")
        if self.history is not None:
            self.robot.reset_observation_capture(self.history)
            self._warming_up = True

    @property
    def read_timeout_s(self):
        return self.warmup_timeout_s if self._warming_up else 1.

    def mark_ready(self):
        self._warming_up = False

    @property
    def pending_command_count(self):
        if any(q.source == "execution:context" for q in self.execution_queries):
            return None
        return max((q.count for q in self.execution_queries
                    if q.source == "execution:pending:command"), default=0)

    @property
    def requires_execution_context(self):
        return self.history is not None or self.execution is not None

    def update_execution_context(self, context):
        if self._closed:
            raise RuntimeError("observation session is closed")
        if self.execution is not None:
            self.execution.update(context)

    def check_execution(self, anchor_s):
        if self.execution is not None:
            for query in self.execution_queries:
                self.execution.resolve(query, anchor_s)

    def check_history(self, anchor_s, *, after_s=None):
        if self._closed:
            raise RuntimeError("observation session is closed")
        if self.history is not None:
            for field in self.spec.fields:
                for query in field.queries:
                    if query.source in self.history_sources:
                        self.history.check(query, anchor_s, after_s=after_s)

    def close(self):
        if not self._closed:
            if self.history is not None:
                self.robot.detach_observation_capture(self.history)
                self.history.reset()
            if self.execution is not None:
                self.execution.clear()
            self._closed = True
