"""Shared public select_action execution, with model-owned queues."""

import numpy as np

class PublicStepAdapter:
    """Require a publisher receipt before advancing a model's implicit step.

    The Worker serializes context updates and predictions. Model-space action
    caches stay inside the policy. For policies that condition on those caches,
    require_exact_publication rejects command-space modifications rather than
    treating their original predictions as the commands actually sent.
    """

    def __init__(self, policy, preprocessor, postprocessor, to_numpy,
                 *, require_exact_publication=False):
        if not callable(getattr(policy, "select_action", None)) or not callable(getattr(policy, "reset", None)):
            raise ValueError("step policy must implement public select_action and reset")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.to_numpy = to_numpy
        self.require_exact_publication = require_exact_publication
        self._context = None
        self._pending_id = None
        self._expected = None
        self._published = False
        self._failed = False
        self._last_prediction = -1
        self._observation_after_s = None

    @property
    def observation_after_s(self):
        return self._observation_after_s

    def update_execution_context(self, context):
        previous = self._context
        if previous is None:
            if (context.phase != "ready" or context.latest_event_id or context.actions
                    or context.planning or context.resets):
                raise ValueError("step adapter requires an empty ready LOAD context")
        elif context.session_id != previous.session_id or context.generation < previous.generation:
            raise ValueError("step context session/generation mismatch")
        reset = previous is None or context.generation != previous.generation
        if previous and context.phase != previous.phase and context.phase in {"paused", "stopped", "error"}:
            reset = True
        if reset:
            # Never reset weights; also clear stateful processor steps via public APIs.
            self._failed = True
            self.policy.reset()
            for processor in (self.preprocessor, self.postprocessor):
                callback = getattr(processor, "reset", None)
                if callable(callback):
                    callback()
            self._pending_id = None
            self._expected = None
            self._published = False
            self._observation_after_s = max(
                (event.recorded_s for event in context.resets), default=None,
            )
            self._failed = False
        else:
            for action in context.actions:
                if action.prediction_id != str(self._pending_id) or action.status == "planned":
                    continue
                if action.status != "published" or action.space != "command":
                    self._failed = True
                    raise RuntimeError("step was not published; reset context before advancing")
                if action.recorded_s is None:
                    self._failed = True
                    raise RuntimeError("step publication has no reception-clock timestamp")
                emitted = np.asarray(action.values)
                planned = np.asarray(action.planned_values if action.planned_values is not None else action.values)
                if (planned.shape != self._expected.shape or not np.array_equal(planned, self._expected)
                        or (self.require_exact_publication and not np.array_equal(emitted, self._expected))):
                    self._failed = True
                    raise RuntimeError("step publication differs from the returned action; reset required")
                # Repeated ZOH publications must not move the barrier forward on
                # every tick and starve a camera that updates at a slower rate.
                if not self._published:
                    self._observation_after_s = action.recorded_s
                else:
                    self._observation_after_s = min(self._observation_after_s, action.recorded_s)
                self._published = True
        self._context = context

    def predict(self, observation, prediction_id):
        if self._context is None or self._context.phase not in {"running", "syncing"}:
            raise RuntimeError("step prediction requires an active context")
        if self._failed:
            raise RuntimeError("step adapter failed; reset context first")
        if self._pending_id is not None and not self._published:
            raise RuntimeError("previous step has no publisher receipt")
        if type(prediction_id) is not int or prediction_id <= self._last_prediction:
            raise ValueError("step prediction IDs must increase")
        self._failed = True
        # Process the public (B,A) output before converting it to wire (1,A).
        # Do not fabricate repeated observations to fill a temporal model queue.
        action = self.policy.select_action(self.preprocessor(observation))
        action_shape = getattr(action, "shape", ())
        if len(action_shape) != 2 or action_shape[0] != 1 or action_shape[1] == 0:
            raise ValueError("select_action must produce one finite action vector with shape (1,A)")
        action = self.postprocessor(action)
        action_shape = getattr(action, "shape", ())
        if len(action_shape) != 2 or action_shape[0] != 1 or action_shape[1] == 0:
            raise ValueError("step postprocessor must preserve batch shape (1,A)")
        chunk = np.asarray(self.to_numpy(action))
        if (chunk.ndim != 2 or chunk.shape[0] != 1 or chunk.shape[1] == 0
                or chunk.dtype.kind not in "iuf" or not np.isfinite(chunk).all()):
            raise ValueError("select_action must produce one finite action vector")
        self._expected = chunk[0].copy()
        self._pending_id = prediction_id
        self._last_prediction = prediction_id
        self._published = False
        self._failed = False
        return chunk
