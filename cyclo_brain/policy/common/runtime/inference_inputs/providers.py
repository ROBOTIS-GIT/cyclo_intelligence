"""Lifecycle wrapper for registered non-robot sources, with reception-clock checks."""

import math

from inference_context.inputs import InputSample, SampleQuery


class ExternalInputs:
    def __init__(self, providers, spec, budget=None):
        self.providers = dict(providers)
        self.owners = list({id(provider): provider for provider in providers.values()}.values())
        self._started = []
        for provider in self.owners:
            for name in ("start", "resolve_samples", "reset", "close"):
                if not callable(getattr(provider, name, None)):
                    raise ValueError(f"registered input provider must implement {name}")
        try:
            for provider in self.owners:
                queries = tuple(dict.fromkeys(q for f in spec.fields for q in f.queries
                                if self.providers.get(q.source) is provider))
                if any(not isinstance(q, SampleQuery) for q in queries):
                    raise ValueError("execution facts are owned by Runtime, not external providers")
                self._started.append(provider)
                provider.start(queries, budget)
        except Exception:
            self.close()
            raise

    def resolve_samples(self, query, anchor_s, *, after_s=None):
        samples = tuple(self.providers[query.source].resolve_samples(query, anchor_s))
        if len(samples) != query.sample_count:
            raise ValueError("external provider sample count mismatch")
        identities = set()
        for sample, offset in zip(samples, query.offsets_s):
            if not isinstance(sample, InputSample) or sample.source != query.source or sample.sequence is None:
                raise ValueError("external provider must return identified InputSample values")
            stamp = sample.received_s
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= stamp <= anchor_s + offset:
                raise ValueError("external provider must use original local monotonic reception time")
            if query.max_age_s is not None and anchor_s + offset - stamp > query.max_age_s:
                raise ValueError("external provider sample is stale")
            if offset == 0 and after_s is not None and stamp <= after_s:
                raise ValueError("external observation predates command publication")
            if sample.sequence in identities:
                raise ValueError("external provider repeated a cached sample as history")
            identities.add(sample.sequence)
        return samples

    def resolve(self, query, anchor_s, *, after_s=None):
        return tuple(s.value for s in self.resolve_samples(query, anchor_s, after_s=after_s))

    def reset(self):
        for provider in self.owners:
            provider.reset()

    def close(self):
        errors = []
        for provider in reversed(self._started):
            try:
                provider.close()
            except Exception as exc:
                errors.append(exc)
        self._started.clear()
        if errors:
            raise RuntimeError(f"input provider close failed: {errors[0]}") from errors[0]
