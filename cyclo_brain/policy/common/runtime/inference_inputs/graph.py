"""LOAD-time DAG compilation, using the existing reception and execution providers.

Operators must not mutate borrowed inputs. A node result is evaluated once per
request. Model-specific tensors and subscriptions live outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import math
import json
import time
from types import MappingProxyType
from typing import Any, Callable

from inference_context.inputs import InputField, InputSpec, SampleQuery, ExecutionQuery, ResolvedInputs


def fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"invalid {label}; allowed fields: {sorted(allowed)}")


def query_from_config(config):
    fields(config, {"source", "offsets_s", "frame_offsets", "fps", "max_age_s", "count", "min_count"}, "source")
    source = config.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")
    if source.startswith("execution:"):
        if set(config) & {"offsets_s", "frame_offsets", "fps"}:
            raise ValueError("execution facts are not sensor frames")
        return ExecutionQuery(source, config.get("count", 1), config.get("min_count", 1), config.get("max_age_s"))
    if set(config) & {"count", "min_count"}:
        raise ValueError("sensor sampling uses explicit offsets, not execution counts")
    offsets = config.get("offsets_s", [0.])
    if "frame_offsets" in config:
        fps = config.get("fps")
        if "offsets_s" in config or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("frame offsets require explicit positive FPS and no offsets_s")
        frames = config["frame_offsets"]
        if not isinstance(frames, list) or any(type(v) is not int for v in frames):
            raise ValueError("frame offsets must be integers")
        offsets = [v / fps for v in frames]
    elif "fps" in config:
        raise ValueError("FPS requires frame_offsets")
    if not isinstance(offsets, list) or any(type(v) not in (int, float) for v in offsets):
        raise ValueError("offsets must be a list of numbers")
    if any(v != 0 for v in offsets) and config.get("max_age_s") is None:
        raise ValueError("temporal samples require explicit max_age_s tolerance")
    return SampleQuery(source, tuple(offsets), config.get("max_age_s"))


@dataclass(frozen=True)
class Data:
    value: Any
    sample_ids: tuple = ()
    received_s: tuple[float, ...] = ()
    computed_s: float | None = None
    semantics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self):
        try:
            hash(self.sample_ids)
        except TypeError as exc:
            raise ValueError("sample identities must be immutable") from exc
        if any(type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0 for stamp in self.received_s):
            raise ValueError("data reception timestamps must be finite monotonic values")

    def derived(self, value, *, semantics=None):
        return Data(value, self.sample_ids, self.received_s, time.monotonic(),
                    self.semantics if semantics is None else tuple(semantics.items()))


@dataclass(frozen=True)
class Binding:
    queries: tuple
    collect: Callable = lambda values: values[0] if len(values) == 1 else tuple(values)
    semantics: tuple = ()
    provider: Any = None


@dataclass(frozen=True)
class Operator:
    """A trusted compiler returns a request-local callable; no imports from YAML."""
    compile: Callable
    cacheable: bool = False
    mutates_inputs: bool = False


class Registry:
    def __init__(self):
        self._operators = {}

    def register(self, name, compiler, *, cacheable=False, mutates_inputs=False):
        if not isinstance(name, str) or not name or name in self._operators or not callable(compiler):
            raise ValueError(f"invalid or duplicate operator: {name!r}")
        self._operators[name] = Operator(compiler, cacheable, mutates_inputs)

    def compile(self, name, options, context):
        if name not in self._operators:
            raise ValueError(f"unregistered input operator: {name}")
        definition = self._operators[name]
        result = definition.compile(deepcopy(options), context)
        if not callable(result):
            raise TypeError(f"operator {name} compiler must return a callable")
        if not definition.mutates_inputs:
            return result
        def owned(values):
            return result(tuple(Data(deepcopy(v.value), v.sample_ids, v.received_s, v.computed_s, v.semantics) for v in values))
        return owned

    def cacheable(self, name):
        return self._operators[name].cacheable


@dataclass(frozen=True)
class _Node:
    name: str
    inputs: tuple[str, ...]
    operation: Callable
    stage: str
    contract: dict = field(default_factory=dict)
    cache_dependencies: tuple[str, ...] = ()


def _validate_data(value, contract, label):
    fields(contract, {"shape", "axes", "channels", "space", "units", "normalization", "dtype"}, "data contract")
    shape = contract.get("shape")
    if shape is not None:
        if not isinstance(shape, list) or any(v is not None and (type(v) is not int or v <= 0) for v in shape):
            raise ValueError(f"{label}: invalid shape declaration")
        actual = tuple(getattr(value.value, "shape", ()))
        if len(actual) != len(shape) or any(want is not None and want != got for want, got in zip(shape, actual)):
            raise ValueError(f"{label}: expected shape {shape}, got {actual}")
    if "dtype" in contract and str(getattr(value.value, "dtype", "")).removeprefix("torch.") != contract["dtype"]:
        raise ValueError(f"{label}: dtype mismatch")
    semantics = dict(value.semantics)
    for key in set(contract) - {"shape", "dtype"}:
        if semantics.get(key) != contract[key]:
            raise ValueError(f"{label}: missing or mismatched {key}; implicit conversion is forbidden")


class Graph:
    """One immutable compiled plan, with separate per-request evaluation objects."""

    def __init__(self, config, registry, *, bindings=None, compile_context=None, budget=None, codec=None):
        fields(config, {"sources", "nodes", "outputs", "execution", "memory", "startup"}, "input graph")
        self.config = deepcopy(config)
        from .resources import Budget
        from .memory import copy_cpu
        self.budget = budget or Budget()
        self.codec = codec or copy_cpu
        self._cache = {}
        bindings = bindings or {}
        sources = config.get("sources")
        nodes = config.get("nodes")
        outputs = config.get("outputs")
        if not isinstance(sources, dict) or not isinstance(nodes, dict) or not isinstance(outputs, dict):
            raise ValueError("sources, nodes and outputs must be mappings")
        if len(sources) + len(nodes) > 256:
            raise ValueError("input graph exceeds 256 entries")
        reserved = {"processed", "model_action", "postprocessed_action"}
        if set(sources) & set(nodes) or reserved & (set(sources) | set(nodes)):
            raise ValueError("graph names must be unique; processor and prediction outputs are reserved")
        if any(not isinstance(k, str) or not k for k in (*sources, *nodes)):
            raise ValueError("graph names must be nonempty strings")
        self.bindings = {}
        for name, source in sources.items():
            if not isinstance(source, dict):
                raise ValueError(f"{name}: source must be a mapping")
            if isinstance(source, dict) and "binding" in source:
                fields(source, {"binding", "sampling", "semantics"}, "bound source")
                if source["binding"] not in bindings:
                    raise ValueError(f"unregistered source binding: {source['binding']}")
                binding = bindings[source["binding"]](deepcopy(source.get("sampling", {})))
            else:
                binding = Binding((query_from_config({k: v for k, v in source.items() if k != "semantics"}),))
            if not isinstance(binding, Binding):
                raise ValueError(f"{name}: invalid binding")
            semantics = source.get("semantics", {})
            fields(semantics, {"axes", "channels", "units", "space", "normalization"}, "source semantics")
            if semantics:
                binding = Binding(binding.queries, binding.collect, tuple(deepcopy(semantics).items()), binding.provider)
            self.bindings[name] = binding
        self.spec = InputSpec(tuple(InputField(k, v.queries) for k, v in self.bindings.items() if v.queries))
        self.providers = {}
        for binding in self.bindings.values():
            if binding.provider is not None:
                for query in binding.queries:
                    previous = self.providers.get(query.source)
                    if previous is not None and previous is not binding.provider:
                        raise ValueError("one source must have exactly one provider owner")
                    self.providers[query.source] = binding.provider
        self.nodes = []
        visiting, compiled = set(), set(self.bindings) | reserved
        stages = {k: "before" for k in self.bindings}
        stages["processed"] = "after"
        stages.update(model_action="result", postprocessed_action="result")
        stage_index = {"before": 0, "after": 1, "result": 2}

        def visit(name):
            if name in compiled:
                return
            if name not in nodes or name in visiting:
                raise ValueError(f"unknown or cyclic node reference: {name}")
            visiting.add(name)
            node = nodes[name]
            fields(node, {"op", "inputs", "options", "stage", "contract", "cache"}, f"node {name}")
            inputs = node.get("inputs", [])
            stage = node.get("stage", "before")
            if stage not in stage_index or not isinstance(inputs, list) or any(not isinstance(v, str) for v in inputs):
                raise ValueError(f"{name}: invalid inputs or stage")
            cache_dependencies = ()
            if "cache" in node:
                if stage == "result":
                    raise ValueError("prediction result nodes cannot use cross-request caches")
                fields(node["cache"], {"dependencies"}, "node cache")
                dependencies = node["cache"].get("dependencies")
                if not isinstance(dependencies, list) or not dependencies or any(not isinstance(v, str) for v in dependencies):
                    raise ValueError("cache requires explicit dependency names")
                if not set(inputs) <= set(dependencies):
                    raise ValueError("cache must include every operator input dependency")
                # A language change must invalidate even an apparently visual encoder.
                language = {k for k, b in self.bindings.items() if any(q.source == "instruction" for q in b.queries)}
                if not language <= set(dependencies):
                    raise ValueError("cache must include instruction dependencies")
                cache_dependencies = tuple(dependencies)
            for dependency in dict.fromkeys([*inputs, *cache_dependencies]):
                visit(dependency)
                if stage_index[stage] < stage_index[stages[dependency]]:
                    raise ValueError("input nodes cannot depend on a later stage")
            contract = deepcopy(node.get("contract", {}))
            fields(contract, {"shape", "axes", "channels", "space", "units", "normalization", "dtype"}, "contract")
            operation = registry.compile(node.get("op"), node.get("options", {}), compile_context)
            if cache_dependencies and not registry.cacheable(node["op"]):
                raise ValueError("operator is not reviewed for cross-request caching")
            self.nodes.append(_Node(name, tuple(inputs), operation, stage, contract, cache_dependencies))
            stages[name] = stage
            visiting.remove(name)
            compiled.add(name)

        for name in nodes:
            visit(name)
        if not outputs or set(outputs) - set(stage_index):
            raise ValueError("outputs must name before, after or result stages")
        self.outputs = deepcopy(outputs)
        for stage, mapping in outputs.items():
            if not isinstance(mapping, dict) or any(not isinstance(k, str) or not k for k in mapping):
                raise ValueError("output stage must map model keys to node names")
            for name in mapping.values():
                if not isinstance(name, str) or name not in compiled or stage_index[stage] < stage_index[stages[name]]:
                    raise ValueError(f"invalid {stage} output: {name}")
        self.bindings = MappingProxyType(self.bindings)
        self.nodes = tuple(self.nodes)

    @property
    def has_cache(self):
        return any(node.cache_dependencies for node in self.nodes)

    def reset(self):
        for name in self._cache:
            self.budget.resize((id(self), name), 0)
        self._cache.clear()

    close = reset

    def begin(self, provider, *, anchor_s=0.):
        if not math.isfinite(anchor_s):
            raise ValueError("input anchor must be finite")
        resolved = provider if isinstance(provider, ResolvedInputs) else ResolvedInputs(self.spec, provider, anchor_s)
        values = {}
        for name, binding in self.bindings.items():
            samples = tuple(v for q in binding.queries for v in resolved.resolve_samples(q, anchor_s))
            envelopes = [v.value if isinstance(v.value, Data) else Data(
                v.value, ((v.source, v.sequence),) if v.sequence is not None else (),
                (v.received_s,) if v.received_s is not None else (),
            ) for v in samples]
            ids = tuple(i for v in envelopes for i in v.sample_ids)
            stamps = tuple(s for v in envelopes for s in v.received_s)
            values[name] = Data(binding.collect(tuple(v.value for v in envelopes)), ids, stamps,
                                semantics=binding.semantics or (envelopes[0].semantics if len(envelopes) == 1 else ()))
        return Evaluation(self, values)

    def assemble(self, provider, *, anchor_s=0.):
        return self.begin(provider, anchor_s=anchor_s).run("before")


class Evaluation:
    def __init__(self, graph, values):
        self.graph = graph
        self.values = values
        self._stages = set()

    def run(self, stage, processed=None):
        if stage not in {"before", "after", "result"} or stage in self._stages:
            raise ValueError("graph stage may run only once per request")
        if stage == "after":
            if "before" not in self._stages:
                raise ValueError("run before processor stage first")
            self.values["processed"] = Data(processed)
        if stage == "result":
            if "after" not in self._stages or not isinstance(processed, dict) or set(processed) != {"model_action", "postprocessed_action"}:
                raise ValueError("result stage requires public model and postprocessor outputs")
            ids = tuple(dict.fromkeys(i for value in self.values.values() for i in value.sample_ids))
            stamps = tuple(dict.fromkeys(s for value in self.values.values() for s in value.received_s))
            self.values.update({key: Data(value, ids, stamps, time.monotonic(), (("space", key),))
                                for key, value in processed.items()})
        for node in self.graph.nodes:
            if node.stage != stage:
                continue
            inputs = tuple(self.values[k] for k in node.inputs)
            cache_key = None
            if node.cache_dependencies:
                cache_key = tuple(self._cache_identity(self.values[key]) for key in node.cache_dependencies)
            cached = self.graph._cache.get(node.name) if cache_key is not None else None
            if cached is not None and cached[0] == cache_key:
                stored = cached[1]
                copied, _ = self.graph.codec(stored.value)
                value = Data(copied, stored.sample_ids, stored.received_s, stored.computed_s, stored.semantics)
            else:
                value = node.operation(inputs)
            if not isinstance(value, Data):
                ids = tuple(dict.fromkeys(i for v in inputs for i in v.sample_ids))
                stamps = tuple(s for v in inputs for s in v.received_s)
                value = Data(value, ids, stamps, time.monotonic())
            _validate_data(value, node.contract, node.name)
            if cache_key is not None and (cached is None or cached[0] != cache_key):
                from .resources import retained_size
                token = (id(self.graph), node.name)
                old_size = retained_size(cached[1].value) if cached else 0
                self.graph.budget.resize(token, retained_size(value.value))
                try:
                    copied, _ = self.graph.codec(value.value)
                except Exception:
                    self.graph.budget.resize(token, old_size)
                    raise
                self.graph._cache[node.name] = (cache_key, Data(copied, value.sample_ids, value.received_s,
                                                               value.computed_s, value.semantics))
            self.values[node.name] = value
        self._stages.add(stage)
        result = {}
        for key, node in self.graph.outputs.get(stage, {}).items():
            value = self.values[node].value
            if key == "*":
                if not isinstance(value, dict) or set(result) & set(value):
                    raise ValueError("expanded graph output must be a mapping with unique keys")
                result.update(value)
            elif key in result:
                raise ValueError(f"duplicate graph output: {key}")
            else:
                result[key] = value
        return result if stage in self.graph.outputs else processed

    @staticmethod
    def _cache_identity(value):
        if value.sample_ids:
            return (value.sample_ids, value.received_s, repr(value.semantics))
        if value.value is None or isinstance(value.value, (str, bool, int, float)):
            return json.dumps(value.value, allow_nan=False)
        raise ValueError("cached numeric inputs require real sample identity; latest getter calls are not samples")
