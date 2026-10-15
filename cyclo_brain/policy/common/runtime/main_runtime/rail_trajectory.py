"""VLA-RAIL cubic smoother and dual-quintic C2 reference transitions.

Equations 8-13, https://arxiv.org/html/2512.24673v1 . NumPy implementation;
FFW grippers remain discrete and are not polynomial fitted.
"""
from functools import lru_cache
import numpy as np

ARM_COLUMNS = np.array([i for i in range(16) if i not in (7, 15)])


def validate_rail_keys(action_keys):
    keys = [key.split(".")[-1] for key in action_keys]
    if len(keys) != 2 or set(keys) != {"arm_left", "arm_right"}:
        raise ValueError("VLA-RAIL requires FFW arm_left/arm_right (8+8) actions")


@lru_cache(maxsize=64)
def _cubic_fit_matrix(length):
    return np.linalg.pinv(np.polynomial.polynomial.polyvander(np.linspace(0, 1, length), 3))


class CubicChunk:
    def __init__(self, actions, dt, keys):
        validate_rail_keys(keys)
        self.actions = np.array(actions, dtype=float, copy=True)
        if (self.actions.ndim != 2 or self.actions.shape[1] != 16 or len(self.actions) < 4
                or not np.isfinite(self.actions).all() or not np.isfinite(dt) or dt <= 0):
            raise ValueError('VLA-RAIL requires at least four finite FFW 8+8 policy actions')
        self.dt = dt
        self.duration = (len(self.actions) - 1) * dt
        self.coefficients = _cubic_fit_matrix(len(self.actions)) @ self.actions[:, ARM_COLUMNS]

    def state(self, when):
        u = np.clip(when / self.duration, 0., 1.)
        c = self.coefficients
        q = c[0] + u * (c[1] + u * (c[2] + u * c[3]))
        v = (c[1] + u * (2*c[2] + 3*u*c[3])) / self.duration
        a = (2*c[2] + 6*u*c[3]) / self.duration**2
        return q, v, a

    def grippers(self, when):
        index = min(len(self.actions)-1, max(0, int(np.floor(when/self.dt + 1e-9))))
        return self.actions[index, [7, 15]].copy()


class Quintic:
    def __init__(self, start, end, duration):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Quintic duration must be positive')
        self.duration = duration
        q0, v0, a0 = start
        q1, v1, a1 = end
        c0, c1, c2 = np.asarray(q0), np.asarray(v0)*duration, .5*np.asarray(a0)*duration**2
        d, v, a = np.asarray(q1)-c0-c1-c2, np.asarray(v1)*duration-c1-2*c2, np.asarray(a1)*duration**2-2*c2
        self.c = (c0, c1, c2, 10*d-4*v+.5*a, -15*d+7*v-a, 6*d-3*v+.5*a)

    def state(self, when):
        u = np.clip(when/self.duration, 0., 1.)
        c = self.c
        q = sum(c[i]*u**i for i in range(6))
        v = sum(i*c[i]*u**(i-1) for i in range(1, 6))/self.duration
        a = sum(i*(i-1)*c[i]*u**(i-2) for i in range(2, 6))/self.duration**2
        return q, v, a


class DualQuintic:
    def __init__(self, start, end, duration):
        self.half = duration/2
        middle = ((start[0]+end[0])/2, (start[1]+end[1])/2, np.zeros_like(start[2]))
        self.left = Quintic(start, middle, self.half)
        self.right = Quintic(middle, end, self.half)

    def state(self, when):
        return self.left.state(when) if when <= self.half else self.right.state(when-self.half)


def alignment_offset(chunk, age, previous_state, window=.1, blend=.2):
    """Eq.10 direction score in a latency-centred, bounded search window.

    Cache snapshot age is a timing proxy, not hardware acquisition time. Ties
    prefer that proxy rather than arbitrarily skipping far into the trajectory.
    """
    nominal = max(0., age)
    upper = chunk.duration - min(blend, chunk.duration/3)
    if nominal >= upper:
        return None
    if previous_state is None:
        return nominal
    times = np.unique(np.r_[nominal, np.linspace(max(0., nominal-window), min(upper, nominal+window), 21)])
    q, v, _ = previous_state
    moving = abs(v) > 1e-5
    if not moving.any():
        return nominal
    score = [np.sign((chunk.state(t)[0]-q)[moving]*v[moving]).sum() for t in times]
    best = max(score)
    return float(min((t for t, s in zip(times, score) if s == best), key=lambda t: abs(t-nominal)))


class RailTrajectory:
    def __init__(self, chunk, start_time, offset=0., previous_state=None, blend=.2):
        self.chunk, self.start_time, self.offset = chunk, start_time, offset
        self.duration = chunk.duration-offset
        if self.duration <= 0:
            raise ValueError('VLA-RAIL chunk expired')
        self.blend = min(blend, self.duration/3)
        if previous_state is None:
            # Retain the bootstrap's first target; do not jump to the OLS fit.
            q = chunk.actions[0, ARM_COLUMNS]
            previous_state = (q, np.zeros_like(q), np.zeros_like(q))
        self.head = DualQuintic(previous_state, chunk.state(offset+self.blend), self.blend)
        # Local adaptation: finish at rest when no next chunk arrives, never
        # extrapolate a cubic past its prediction horizon.
        self.tail_start = self.duration-self.blend
        q_end = chunk.state(chunk.duration)[0]
        self.tail = DualQuintic(chunk.state(offset+self.tail_start),
                               (q_end, np.zeros_like(q_end), np.zeros_like(q_end)), self.blend)

    @property
    def end(self):
        return self.start_time+self.duration

    def state(self, now):
        t = max(0., now-self.start_time)
        if t < self.blend:
            return self.head.state(t)
        if t >= self.tail_start:
            return self.tail.state(min(t-self.tail_start, self.blend))
        return self.chunk.state(self.offset+t)

    def action(self, now):
        action = np.empty(16)
        action[ARM_COLUMNS] = self.state(now)[0]
        action[[7, 15]] = self.chunk.grippers(self.offset+max(0., now-self.start_time))
        return action
