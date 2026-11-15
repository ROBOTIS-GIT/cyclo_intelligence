"""Model-scoped left pose hold, applied before committing TT-RTC waypoints.

The planner state belongs to the END of the queued trajectory, not the current
robot time. Already committed waypoints are never rewritten. Consequently the
next model prefix describes exactly the held/released trajectory being executed.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import os

import numpy as np


MODEL_NAME = "Grootn17_Drill_Into_Dynamixel_VT_5K"
REFERENCE = np.array([-.7188857, .6100330, -1.3637808, -1.2060205,
                      .7054754, -.3251440, -.7922916])


@dataclass(frozen=True)
class HoldConfig:
    entry_rad: float = .010
    exit_rad: float = .025
    arrival_rad: float = .002
    alignment_speed_rad_s: float = .03
    stable_span_rad: float = .0035
    stable_s: float = .30
    exit_s: float = .20
    release_speed_rad_s: float = .15

    def __post_init__(self):
        values = list(vars(self).values())
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("pose hold parameters must be finite and positive")
        if not self.stable_span_rad < self.entry_rad < self.exit_rad <= .10:
            raise ValueError("pose hold requires stable_span < entry < exit <= 0.10rad")
        if not self.arrival_rad < self.entry_rad:
            raise ValueError("pose hold arrival tolerance must be smaller than entry band")
        if self.alignment_speed_rad_s > self.release_speed_rad_s:
            raise ValueError("pose alignment speed must not exceed release speed")
        if self.stable_s < .1 or self.exit_s < .1 or self.release_speed_rad_s > .5:
            raise ValueError("pose hold dwell too short or release speed too large")

    @classmethod
    def from_environment(cls):
        return cls(**{key: float(os.environ.get("CYCLO_LEFT_HOLD_" + key.upper(), value))
                      for key, value in vars(cls()).items()})


def enabled_for_model(model_path):
    enabled = os.environ.get("CYCLO_LEFT_ARM_POSE_HOLD", "false").lower().strip()
    if enabled not in {"true", "false", "1", "0"}:
        raise ValueError("CYCLO_LEFT_ARM_POSE_HOLD must be true or false")
    return enabled in {"true", "1"} and str(model_path).rstrip('/').split('/')[-1] == MODEL_NAME


class LeftPoseHold:
    def __init__(self, indices, lower, upper, config=None):
        self.indices = np.asarray(indices, dtype=int)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.config = config or HoldConfig()
        if (self.indices.shape != (7,) or len(set(self.indices)) != 7
                or np.any(self.indices < 0) or self.lower.shape != (7,)
                or self.upper.shape != (7,) or not np.isfinite(self.lower).all()
                or not np.isfinite(self.upper).all() or np.any(self.lower >= self.upper)
                or np.any(REFERENCE < self.lower) or np.any(REFERENCE > self.upper)):
            raise ValueError("pose hold requires seven ordered left joints with valid limits")
        self.clear()

    def clear(self):
        self.mode = "moving"
        self.held = None
        self.samples = deque()
        self.feedback_ready = False
        self.feedback_at_target = False
        self.last_feedback = None
        self.last_observed = None
        self.outside_s = np.zeros(7)
        self.outside_sign = np.zeros(7)

    def _left(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.ndim != 1 or a.size <= self.indices.max() or not np.isfinite(a).all():
            raise ValueError("pose hold action must be a finite flat vector")
        q = a[self.indices]
        if np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6):
            raise ValueError("pose hold action exceeds joint limits")
        return q

    def observe(self, feedback, now):
        """Fresh measured feedback only; called at 20 Hz by the control thread."""
        q = self._left(feedback).copy()
        if not np.isfinite(now):
            raise ValueError("pose hold observation time must be finite")
        if self.last_observed is not None and (now <= self.last_observed or now-self.last_observed > .15):
            self.samples.clear()
        self.last_observed = now
        self.last_feedback = q.copy()
        self.samples.append((now, q))
        while len(self.samples) > 1 and self.samples[1][0] <= now-self.config.stable_s:
            self.samples.popleft()
        values = np.stack([item[1] for item in self.samples])
        self.feedback_ready = bool(
            now-self.samples[0][0] >= self.config.stable_s-1e-9
            and np.all(np.abs(values-REFERENCE) <= self.config.entry_rad)
            and np.all(np.ptp(values, axis=0) <= self.config.stable_span_rad))
        self.feedback_at_target = bool(
            self.feedback_ready
            and np.all(np.abs(values-REFERENCE) <= self.config.arrival_rad))

    def plan(self, actions, anchor, now, source_hz=15., allow_entry=True):
        """Return a transactional candidate: (conditioned chunk, state, events).

        Caller installs the candidate state only AFTER a successful queue append.
        Dwell for release is measured on the source trajectory, so inference
        latency cannot count as persistent movement intent.
        """
        data = np.asarray(actions, dtype=np.float64)
        if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
            raise ValueError("pose hold requires a nonempty finite action chunk")
        if not np.isfinite(source_hz) or source_hz <= 0:
            raise ValueError("invalid pose hold source rate")
        for row in data:
            self._left(row)
        previous = self._left(anchor).copy() if anchor is not None else None
        state = deepcopy(self)
        fresh = (state.last_observed is not None
                 and -1e-9 <= now-state.last_observed <= .15)
        eligible = allow_entry and state.feedback_ready and fresh
        arrived = state.feedback_at_target and fresh
        out = data.copy()
        events = []
        dt = 1./source_hz
        for step, row in enumerate(data):
            target = row[self.indices]
            # A future approach is not measured arrival. Once this trajectory
            # leaves the entry region, wait for new feedback on a later refill.
            if (previous is None or np.any(np.abs(previous-REFERENCE) > self.config.entry_rad)
                    or np.any(np.abs(target-REFERENCE) > self.config.entry_rad)):
                eligible = False
            if (state.mode == "moving" and eligible and previous is not None
                    and np.all(np.abs(previous-REFERENCE) <= self.config.entry_rad)
                    and np.all(np.abs(target-REFERENCE) <= self.config.entry_rad)):
                state.mode = "aligning"
                # Finish the placement at the explicitly configured reference.
                # Slew from the queued anchor; never snap to this new target.
                state.held = REFERENCE.copy()
                state.outside_s.fill(0)
                state.outside_sign.fill(0)
                events.append((step, "align"))
            if state.mode in {"aligning", "holding"}:
                delta = target-state.held
                sign = np.where(np.abs(delta) > self.config.exit_rad, np.sign(delta), 0.)
                state.outside_s = np.where(sign == 0, 0.,
                    np.where(sign == state.outside_sign, state.outside_s+dt, 0.))
                state.outside_sign = sign
                if np.any(state.outside_s >= self.config.exit_s-1e-9):
                    state.mode = "releasing"
                    eligible = False
                    state.feedback_ready = False
                    state.feedback_at_target = False
                    state.samples.clear()
                    events.append((step, "release"))
                else:
                    if state.mode == "aligning":
                        if previous is None:
                            raise ValueError("pose alignment requires a queue anchor")
                        difference = state.held-previous
                        limit = self.config.alignment_speed_rad_s*dt
                        out[step, self.indices] = np.where(
                            np.abs(difference) <= limit, state.held,
                            previous+np.clip(difference, -limit, limit))
                        # A predicted waypoint reaching the reference is not
                        # evidence of actual arrival. Require fresh feedback
                        # stable at the reference as well as a settled anchor.
                        if arrived and np.all(np.abs(previous-state.held) <= 1e-12):
                            state.mode = "holding"
                            events.append((step, "hold"))
                    else:
                        out[step, self.indices] = state.held
            if state.mode == "releasing":
                if previous is None:
                    raise ValueError("pose hold release requires a queue anchor")
                difference = target-previous
                limit = self.config.release_speed_rad_s*dt
                out[step, self.indices] = previous+np.clip(difference, -limit, limit)
                if np.all(np.abs(difference) <= limit):
                    state.mode = "moving"
            previous = out[step, self.indices].copy()
        return out, state, events
