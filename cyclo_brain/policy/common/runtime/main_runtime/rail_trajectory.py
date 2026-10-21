"""VLA-RAIL cubic least squares and dual-quintic C2 chunk fusion.

Paper: https://arxiv.org/html/2512.24673v1, equations 8-13. FFW grippers
remain discrete. Bootstrap blending, terminal stopping, a bounded alignment
search and short-horizon blend caps are explicit robot-runtime adaptations.
"""
from functools import lru_cache
import numpy as np

ARM_COLUMNS = np.array([i for i in range(16) if i not in (7, 15)])


def validate_rail_keys(action_keys):
    keys = [key.split(".")[-1] for key in action_keys]
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= {"arm_left", "arm_right"}:
        raise ValueError("VLA-RAIL requires one or both unique FFW arm_left/arm_right actions (8 per arm)")


@lru_cache(maxsize=64)
def _cubic_fit_matrix(length):
    # Normalizing time leaves the cubic least-squares space unchanged and
    # improves conditioning. SVD avoids explicitly inverting V.T @ V (Eq.9).
    return np.linalg.pinv(np.polynomial.polynomial.polyvander(np.linspace(0., 1., length), 3))


class CubicChunk:
    """One least-squares cubic per arm joint over the complete action chunk."""
    def __init__(self, actions, dt, keys):
        validate_rail_keys(keys)
        self.action_dim = 8 * len(keys)
        self.gripper_columns = np.arange(7, self.action_dim, 8)
        self.arm_columns = np.array([i for i in range(self.action_dim) if i % 8 != 7])
        self.actions = np.array(actions, dtype=float, copy=True)
        if (self.actions.ndim != 2 or self.actions.shape[1] != self.action_dim or len(self.actions) < 4
                or not np.isfinite(self.actions).all() or not np.isfinite(dt) or dt <= 0):
            raise ValueError('VLA-RAIL requires at least four finite policy actions with 8 values per FFW arm')
        self.dt = dt
        self.duration = (len(self.actions) - 1) * dt
        self.coefficients = _cubic_fit_matrix(len(self.actions)) @ self.actions[:, self.arm_columns]

    def state(self, when):
        u = np.asarray(np.clip(np.asarray(when)/self.duration, 0., 1.))[..., None]
        c = self.coefficients
        q = c[0]+u*(c[1]+u*(c[2]+u*c[3]))
        v = (c[1]+u*(2*c[2]+u*3*c[3]))/self.duration
        a = (2*c[2]+u*6*c[3])/self.duration**2
        return q, v, a

    def grippers(self, when):
        index = min(len(self.actions)-1, max(0, int(np.floor(when/self.dt + 1e-9))))
        return self.actions[index, self.gripper_columns].copy()


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
        u = np.asarray(np.clip(when/self.duration, 0., 1.))[..., None]
        c = self.c
        q = c[0]+u*(c[1]+u*(c[2]+u*(c[3]+u*(c[4]+u*c[5]))))
        v = (c[1]+u*(2*c[2]+u*(3*c[3]+u*(4*c[4]+u*5*c[5]))))/self.duration
        a = (2*c[2]+u*(6*c[3]+u*(12*c[4]+u*20*c[5])))/self.duration**2
        return q, v, a


class DualQuintic:
    """Paper Eq.11/12: average midpoint position/velocity, zero acceleration."""
    def __init__(self, start, end, duration):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Dual-quintic duration must be positive')
        self.half = duration/2
        middle = ((np.asarray(start[0])+np.asarray(end[0]))/2,
                  (np.asarray(start[1])+np.asarray(end[1]))/2,
                  np.zeros_like(start[2]))
        self.left = Quintic(start, middle, self.half)
        self.right = Quintic(middle, end, self.half)

    def state(self, when):
        when = np.asarray(when)
        if when.ndim == 0:
            return self.left.state(when) if when <= self.half else self.right.state(when-self.half)
        left, right = self.left.state(when), self.right.state(when-self.half)
        return tuple(np.where((when <= self.half)[..., None], a, b) for a, b in zip(left, right))


def trajectory_timing(chunk, offset=0., blend=.2, playback_rate=1.):
    """Fixed playback ratio; short-horizon blend caps are a local adaptation.

    Offsets use policy seconds and blend/duration use wall seconds. No inference
    latency or derivative limits alter the requested playback rate.
    """
    offset = np.asarray(offset)
    if (not np.isfinite(blend) or blend <= 0
            or not np.isfinite(playback_rate) or playback_rate <= 0
            or not np.isfinite(offset).all() or np.any(offset < 0)):
        raise ValueError('VLA-RAIL timing requires positive finite blend/rate and nonnegative finite offset')
    remaining = chunk.duration-offset
    if np.any(remaining <= 0):
        raise ValueError('VLA-RAIL chunk expired')
    duration = remaining/playback_rate
    effective_blend = np.minimum(blend, duration/3)
    scale = np.full_like(duration, playback_rate, dtype=float)
    if offset.ndim == 0:
        return float(duration), float(effective_blend), float(scale)
    return duration, effective_blend, scale


def alignment_offset(chunk, age, previous_state, blend=.2, playback_rate=1.):
    """Maximize the paper's Eq.10 direction score, including sign(0) = +1.

    ``age`` is policy seconds; callers convert observation age using their
    explicit playback ratio. Search the whole valid domain, partitioned by the
    cubic position-error roots where the sign score can change. Observation age
    only rejects expired responses and breaks equal-score ties. The end-of-chunk
    reservation and nominal-time tie breaking are local runtime choices. FFW
    grippers remain discrete and do not participate in this derivative score.
    """
    if not all(np.isfinite(value) and value > 0 for value in (blend, playback_rate)):
        raise ValueError('VLA-RAIL alignment blend and playback rate must be positive and finite')
    if not np.isfinite(age):
        raise ValueError('VLA-RAIL alignment age must be finite')
    nominal = max(0., age)
    upper = chunk.duration-min(blend*playback_rate, chunk.duration/3)
    if nominal >= upper:
        return None
    nominal = float(np.clip(nominal, np.nextafter(0., upper), np.nextafter(upper, 0.)))
    if previous_state is None:
        return nominal
    q, v, _ = previous_state
    q, v = np.asarray(q), np.asarray(v)
    root_joints = []
    for joint in range(len(q)):
        if v[joint] == 0.:
            # This term is +1 everywhere, including stationary joints.
            continue
        power = chunk.coefficients[:, joint].copy()
        power[0] -= q[joint]
        for root in np.roots(power[::-1]):
            u = float(root.real)
            if abs(root.imag) > 16*np.sqrt(np.finfo(float).eps)*max(1., abs(u)):
                continue
            when = u*chunk.duration
            if not 0. < when < upper:
                continue
            # A root is mathematically zero; a floating root solve may leave a
            # tiny signed residual. Validate its forward error before assigning
            # that joint Eq.10's +1 contribution at this root only. This is not
            # a velocity threshold or a tolerance on ordinary candidate scores.
            error_scale = np.sum(abs(power)*abs(u)**np.arange(4))
            residual = abs(np.polynomial.polynomial.polyval(u, power))
            if residual <= 64*np.finfo(float).eps*error_scale:
                root_joints.append((when, joint))

    breaks = np.unique([0., upper, *(when for when, _ in root_joints)])
    candidates = [nominal, *(when for when, _ in root_joints)]
    for left, right in zip(breaks[:-1], breaks[1:]):
        interior_left, interior_right = np.nextafter(left, right), np.nextafter(right, left)
        if interior_left > interior_right:
            continue
        # Score is constant inside each root interval. Include both a robust
        # interior point and the representable point nearest observation age.
        candidates.extend(((left+right)/2, float(np.clip(nominal, interior_left, interior_right))))
    times = np.unique(candidates)
    target_q = chunk.state(times)[0]
    contributions = np.where((target_q-q)*v >= 0., 1, -1)
    for when, joint in root_joints:
        contributions[np.searchsorted(times, when), joint] = 1
    scores = contributions.sum(axis=-1)
    return float(times[np.lexsort((abs(times-nominal), -scores))[0]])


class RailTrajectory:
    def __init__(self, chunk, start_time, offset=0., previous_state=None, blend=.2,
                 playback_rate=1., *, head_mode='dual', preserve_blend=False, tail_blend=None,
                 gripper_offset=None, gripper_playback_rate=None):
        self.duration, self.blend, self.time_scale = trajectory_timing(chunk, offset, blend, playback_rate)
        if head_mode not in ('dual', 'single'):
            raise ValueError('VLA-RAIL bridge must be dual or single quintic')
        if tail_blend is not None and (not np.isfinite(tail_blend) or tail_blend <= 0):
            raise ValueError('VLA-RAIL tail blend must be positive and finite')
        if preserve_blend:
            # An explicit planner must not obtain feasibility by silently
            # shortening its selected bridge. Reserve the independently
            # selected stop segment; they may meet with no middle segment.
            selected_tail = blend if tail_blend is None else tail_blend
            if self.duration + 1e-12 < blend+selected_tail:
                raise ValueError('VLA-RAIL horizon cannot contain the requested head and tail')
            self.blend = float(blend)
        self.tail_blend = float(self.blend if tail_blend is None else tail_blend)
        if self.blend+self.tail_blend > self.duration+1e-12:
            raise ValueError('VLA-RAIL head and tail must not overlap')
        self.head_mode = head_mode
        self.playback_rate = self.time_scale
        self.chunk, self.start_time, self.offset = chunk, start_time, offset
        self.policy_remaining_s = chunk.duration-offset
        # Spatial arm alignment must not rewind or skip gripper events solely
        # because another arm waypoint is closer. The runtime can retain its
        # nominal observation-based gripper clock independently of arm phase.
        self.gripper_offset = float(offset if gripper_offset is None else gripper_offset)
        self.gripper_time_scale = float(self.time_scale if gripper_playback_rate is None
                                        else gripper_playback_rate)
        if (not np.isfinite(self.gripper_offset) or not 0. <= self.gripper_offset < chunk.duration
                or not np.isfinite(self.gripper_time_scale) or self.gripper_time_scale <= 0.):
            raise ValueError('VLA-RAIL gripper timing requires a valid phase and positive finite rate')
        if previous_state is None:
            # Retain the bootstrap's first target and start at rest.
            q = chunk.actions[0, chunk.arm_columns]
            previous_state = (q, np.zeros_like(q), np.zeros_like(q))
        # Optional diagnostics only: global least squares does not constrain
        # its fitted curve to the raw waypoint range. Production explicitly
        # disables this envelope check; never clip and corrupt its derivatives.
        raw_arms = chunk.actions[:, chunk.arm_columns]
        self.lower = np.minimum(raw_arms.min(axis=0), previous_state[0])
        self.upper = np.maximum(raw_arms.max(axis=0), previous_state[0])
        bridge_type = DualQuintic if head_mode == 'dual' else Quintic
        self.head = bridge_type(previous_state, self.reference_state(self.blend), self.blend)
        # Local adaptation: finish at rest when no next chunk arrives, never
        # extrapolate a cubic past its prediction horizon.
        self.tail_start = self.duration-self.tail_blend
        q_end = chunk.state(chunk.duration)[0]
        self.tail = bridge_type(self.reference_state(self.tail_start),
                               (q_end, np.zeros_like(q_end), np.zeros_like(q_end)), self.tail_blend)

    def policy_time(self, elapsed):
        return min(self.chunk.duration, self.offset+max(0., elapsed)*self.time_scale)

    def gripper_policy_time(self, elapsed):
        return min(self.chunk.duration,
                   self.gripper_offset+max(0., elapsed)*self.gripper_time_scale)

    def reference_state(self, elapsed):
        q, v, a = self.chunk.state(self.policy_time(elapsed))
        return q, v*self.time_scale, a*self.time_scale**2

    @property
    def end(self):
        return self.start_time+self.duration

    def state(self, now):
        t = max(0., now-self.start_time)
        if t < self.blend:
            return self.head.state(t)
        if t >= self.tail_start:
            return self.tail.state(min(t-self.tail_start, self.tail_blend))
        return self.reference_state(t)

    def action(self, now, *, enforce_envelope=True):
        action = np.empty(self.chunk.action_dim)
        positions = self.state(now)[0]
        if not np.isfinite(positions).all():
            raise ValueError('VLA-RAIL produced nonfinite positions; stop required')
        if enforce_envelope and (np.any(positions < self.lower-1e-6)
                                 or np.any(positions > self.upper+1e-6)):
            raise ValueError('VLA-RAIL polynomial exceeds policy position envelope; stop required')
        action[self.chunk.arm_columns] = positions
        action[self.chunk.gripper_columns] = self.chunk.grippers(self.gripper_policy_time(now-self.start_time))
        return action
