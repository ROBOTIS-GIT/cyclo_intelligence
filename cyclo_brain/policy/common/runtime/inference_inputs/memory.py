"""Transactional feature memory, committed by publication facts, not predictions alone."""

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np

from .graph import Data, fields
from .resources import Budget, retained_size


CONDITIONS = {"prediction_success", "plan_accepted", "first_publication", "published_count", "plan_terminal"}


@dataclass(frozen=True)
class Condition:
    event: str = "prediction_success"
    count: int = 1

    def __post_init__(self):
        if self.event not in CONDITIONS or type(self.count) is not int or not 1 <= self.count <= 4096:
            raise ValueError("invalid memory commit condition")
        if self.event != "published_count" and self.count != 1:
            raise ValueError("count applies only to published_count")

    @classmethod
    def parse(cls, config):
        fields(config, {"event", "count"}, "commit condition")
        return cls(**config)


def copy_cpu(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("memory supports numeric arrays only")
        result = value.copy()
        result.setflags(write=False)
        return result, result.nbytes
    if isinstance(value, (str, bool, int, float)) or value is None:
        return deepcopy(value), len(str(value).encode()) + 32
    if isinstance(value, (tuple, list, dict)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        copied, size = {}, 64
        for key, item in items:
            result, used = copy_cpu(item)
            copied[key], size = result, size + used + len(str(key))
        return (copied if isinstance(value, dict) else tuple(copied.values())), size
    raise TypeError("register a storage codec for this tensor/device type")


@dataclass
class _Proposal:
    values: dict = field(default_factory=dict)
    succeeded: bool = False
    plan: object = None
    published: set = field(default_factory=set)
    terminal: set = field(default_factory=set)
    failed: bool = False


class Memory:
    def __init__(self, slots=None, *, max_bytes=256 * 1024 * 1024, codec=copy_cpu, budget=None):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("memory budget must be positive")
        self.conditions = {name: Condition.parse(config) for name, config in (slots or {}).items()}
        self.max_bytes, self.codec = max_bytes, codec
        self.budget = budget or Budget(max_bytes)
        self._committed = {}
        self._committed_prediction = {}
        self._pending = {}
        self._context = None
        self._last_prediction = -1
        self._active = None
        self._failed = False

    @property
    def bytes_used(self):
        return sum(v[1] for v in self._committed.values()) + sum(v[1] for p in self._pending.values() for v in p.values.values())

    def reset(self):
        for slot in self._committed:
            self.budget.resize((id(self), "committed", slot), 0)
        for prediction_id, proposal in self._pending.items():
            for slot in proposal.values:
                self.budget.resize((id(self), prediction_id, slot), 0)
        self._committed.clear()
        self._committed_prediction.clear()
        self._pending.clear()
        self._active = None
        self._last_prediction = -1
        self._failed = False

    def begin(self, prediction_id):
        if self._failed:
            raise RuntimeError("input memory failed; reset generation before retry")
        if type(prediction_id) is not int or prediction_id <= self._last_prediction:
            raise ValueError("memory prediction IDs must increase")
        if len(self._pending) >= 64:
            raise MemoryError("unconfirmed prediction budget exceeded")
        self._active = prediction_id
        self._last_prediction = prediction_id
        self._pending[prediction_id] = _Proposal()

    def read(self, slot, *, initial=None):
        if slot not in self.conditions:
            raise ValueError(f"undeclared memory slot: {slot}")
        stored = self._committed.get(slot)
        if stored is None:
            if initial is None:
                raise ValueError(f"memory slot {slot} has no committed value or explicit initial value")
            return initial
        envelope, _ = stored
        value, _ = self.codec(envelope.value)
        # State can change even when the input frame has not changed.
        identity = (("memory", slot, self._committed_prediction[slot]),)
        return Data(value, identity, envelope.received_s, envelope.computed_s, envelope.semantics)

    def propose(self, slot, value):
        if slot not in self.conditions or self._active not in self._pending:
            raise ValueError("memory write requires a declared slot and active prediction")
        if not isinstance(value, Data):
            raise TypeError("memory proposals require data provenance")
        proposal = self._pending[self._active]
        previous = proposal.values.get(slot)
        size = retained_size(value.value)
        if self.bytes_used + size - (previous[1] if previous else 0) > self.max_bytes:
            self.fail()
            raise MemoryError("input memory budget exceeded; reset required")
        token = (id(self), self._active, slot)
        try:
            self.budget.resize(token, size)
            copied, actual = self.codec(value.value)
            if actual != size:
                raise ValueError("storage codec byte count differs from retained size")
        except Exception:
            self.budget.resize(token, previous[1] if previous else 0)
            self.fail()
            raise
        proposal.values[slot] = (Data(copied, value.sample_ids, value.received_s, value.computed_s, value.semantics), size)

    def success(self):
        proposal = self._pending[self._active]
        proposal.succeeded = True
        self._commit(self._active)
        self._active = None

    def fail(self):
        if self._active is not None:
            self._discard(self._active)
        self._active = None
        self._failed = True

    def update(self, context):
        if context.feedback_schema != 2:
            raise ValueError("transactional memory requires feedback schema 2")
        previous = self._context
        if previous is not None:
            if context.session_id != previous.session_id or context.generation < previous.generation:
                raise ValueError("memory session/generation mismatch")
            if context.revision < previous.revision or context.latest_event_id < previous.latest_event_id:
                raise ValueError("memory feedback moved backwards")
            if context.after_event_id > previous.latest_event_id:
                raise ValueError("memory feedback event gap; reset required")
        reset = previous is None or context.generation != previous.generation or (
            context.phase in {"paused", "stopped", "error"} and context.phase != previous.phase)
        if reset:
            self.reset()
        cursor = previous.latest_event_id if previous else 0
        events = [*context.planning, *(a for a in context.actions if a.event_id is not None)]
        for event in sorted(events, key=lambda e: e.event_id):
            if event.event_id <= cursor:
                continue
            prediction_id = int(event.prediction_id)
            proposal = self._pending.get(prediction_id)
            if proposal is None:
                continue
            if hasattr(event, "command_count"):
                proposal.plan = event
                if event.command_count == 0:
                    proposal.failed = True
            elif proposal.plan is not None:
                start = proposal.plan.command_start_id
                if start <= event.command_id < start + proposal.plan.command_count:
                    proposal.terminal.add(event.command_id)
                    if event.status == "published":
                        proposal.published.add(event.command_id)
                    else:
                        proposal.failed = True
            self._commit(prediction_id)
        self._context = context

    def _commit(self, prediction_id):
        proposal = self._pending[prediction_id]
        plan = proposal.plan
        for slot in list(proposal.values):
            condition = self.conditions[slot]
            ready = condition.event == "prediction_success" and proposal.succeeded
            if plan is not None and not proposal.failed:
                if condition.event == "plan_accepted":
                    ready = True
                elif condition.event == "plan_terminal":
                    ready = len(proposal.published) == plan.command_count
                elif condition.event in {"first_publication", "published_count"}:
                    needed = condition.count if condition.event == "published_count" else 1
                    ready = all(plan.command_start_id + i in proposal.published for i in range(needed))
            if ready:
                value = proposal.values.pop(slot)
                if prediction_id >= self._committed_prediction.get(slot, -1):
                    self.budget.transfer((id(self), prediction_id, slot), (id(self), "committed", slot))
                    self._committed[slot] = value
                    self._committed_prediction[slot] = prediction_id
                else:
                    self.budget.resize((id(self), prediction_id, slot), 0)
        terminal = plan is not None and len(proposal.terminal) == plan.command_count
        if proposal.failed or terminal or (proposal.succeeded and not proposal.values):
            self._discard(prediction_id)

    def _discard(self, prediction_id):
        proposal = self._pending.pop(prediction_id, None)
        if proposal is not None:
            for slot in proposal.values:
                self.budget.resize((id(self), prediction_id, slot), 0)


def register_memory(registry, memory):
    def read(options, context):
        fields(options, {"slot", "initial"}, "memory_read")
        slot = options.get("slot")
        if slot not in memory.conditions:
            raise ValueError("memory_read requires a declared slot")
        initial = options.get("initial")
        def run(values):
            if initial == "input":
                if len(values) != 1:
                    raise ValueError("initial: input requires one explicit bootstrap input")
                fallback = values[0]
            elif initial is None:
                fallback = None
            else:
                fallback = Data(deepcopy(initial))
            return memory.read(slot, initial=fallback)
        return run

    def write(options, context):
        fields(options, {"slot"}, "memory_write")
        slot = options.get("slot")
        if slot not in memory.conditions:
            raise ValueError("memory_write requires a declared slot")
        def run(values):
            if len(values) != 1:
                raise ValueError("memory_write requires one value")
            memory.propose(slot, values[0])
            return values[0]
        return run

    registry.register("memory_read", read)
    registry.register("memory_write", write)
