"""Publication-paced model steps. The ControlLoop lock owns this object."""

import math


class StepSchedule:
    def __init__(self, dataset_fps):
        if not math.isfinite(dataset_fps) or dataset_fps <= 0:
            raise ValueError("step execution requires a positive Dataset FPS")
        self.period_s = 1.0 / dataset_fps
        self.reset()

    def reset(self):
        self._prediction = None
        self._published = False
        self._next_request_s = 0.0

    def accepted(self, prediction_id, chunk):
        if chunk.ndim != 2 or chunk.shape[0] != 1:
            raise ValueError("step execution requires exactly one action per response")
        if self._prediction is not None and not self._published:
            raise RuntimeError("previous step has not been published")
        self._prediction = prediction_id
        self._published = False

    def published(self, prediction_id, now_s):
        if prediction_id == self._prediction and not self._published:
            self._published = True
            self._next_request_s = now_s + self.period_s

    def can_request(self, now_s, pending):
        return not pending and (
            self._prediction is None or (self._published and now_s >= self._next_request_s)
        )

    def velocity_expired(self, prediction_id, now_s):
        """A model step must not command continued motion during slow inference."""
        return (prediction_id == self._prediction and self._published
                and now_s >= self._next_request_s)
