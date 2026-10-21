"""Spatially aligned, curve-checked handoff for asynchronous VLA-RAIL chunks.

This is an execution-side extension of the paper implementation. The original
direction-score alignment and dual-quintic classes remain in rail_trajectory.
The planner checks the curve that would actually be executed, rather than only
the position of a candidate waypoint. All tolerances below are relative to the
motion in this observation/reference; they are not robot speed/acceleration
limits and no command or derivative is clipped.
"""
from dataclasses import dataclass
import math
import numpy as np

from .rail_trajectory import DualQuintic, Quintic, RailTrajectory
from .rail_alignment import nearest_tcp_phases


@dataclass(frozen=True)
class HandoffSelection:
    trajectory: object
    reason: str
    diagnostics: dict


def _state_arrays(state, dimension):
    if state is None:
        return None
    if len(state) != 3:
        raise ValueError('Handoff boundary requires q, v and a')
    result = tuple(np.asarray(value, dtype=float) for value in state)
    if any(value.shape != (dimension,) or not np.isfinite(value).all() for value in result):
        raise ValueError('Handoff boundary has invalid dimensions or nonfinite values')
    return result


def _signs(values, tolerance):
    return np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))


def _turns(signs):
    """Count meaningful direction changes, ignoring a zero-speed plateau."""
    last = np.zeros(signs.shape[::2], dtype=int)
    count = np.zeros_like(last)
    for index in range(signs.shape[1]):
        current = signs[:, index]
        count += (current != 0) & (last != 0) & (current != last)
        last = np.where(current != 0, current, last)
    return count


def _reference_samples(trajectory, times):
    """Vectorized evaluation for our known reference, scalar protocol fallback."""
    times = np.asarray(times, dtype=float)
    if isinstance(trajectory, RailTrajectory):
        elapsed = np.maximum(0., times-trajectory.start_time)
        policy = np.minimum(trajectory.chunk.duration, trajectory.offset+elapsed*trajectory.time_scale)
        q, v, a = trajectory.chunk.state(policy)
        v *= trajectory.time_scale
        a *= trajectory.time_scale**2
        for mask, segment, local_times in (
                (elapsed < trajectory.blend, trajectory.head, elapsed),
                (elapsed >= trajectory.tail_start, trajectory.tail,
                 np.minimum(elapsed-trajectory.tail_start, trajectory.tail_blend))):
            if np.any(mask):
                segment_state = segment.state(local_times[mask])
                for destination, source in zip((q, v, a), segment_state):
                    destination[mask] = source
        return q, v, a
    states = [trajectory.state(float(time)) for time in times.ravel()]
    return tuple(np.asarray([state[index] for state in states]).reshape(times.shape+(-1,))
                 for index in range(3))


def _exact_velocity_features(bridge, duration, dimension, velocity_tolerance):
    """Polynomial-root audit of the accepted bridge, independent of sampling.

    Every velocity extremum and zero crossing is inspected. Thus a narrow
    backward lobe cannot evade the coarse, vectorized candidate screening.
    """
    segments = ((bridge.left, 0.), (bridge.right, bridge.half)) if isinstance(bridge, DualQuintic) else ((bridge, 0.),)
    signs = [[] for _ in range(dimension)]
    positive = np.zeros(dimension)
    negative = np.zeros(dimension)
    vmin = np.full(dimension, math.inf)
    vmax = np.full(dimension, -math.inf)
    qmin = np.full(dimension, math.inf)
    qmax = np.full(dimension, -math.inf)
    for segment, _ in segments:
        for joint in range(dimension):
            coefficients = np.array([np.asarray(c)[joint] for c in segment.c])
            velocity = coefficients[1:] * np.arange(1, 6) / segment.duration
            acceleration = velocity[1:] * np.arange(1, 5)

            def real_roots(power):
                roots = np.polynomial.polynomial.polyroots(power)
                return [float(root.real) for root in roots
                        if abs(root.imag) < 1e-7 and 0. < root.real < 1.]

            zeros = sorted([0., 1., *real_roots(velocity)])
            extrema = [0., 1., *real_roots(acceleration)]
            velocities = np.polynomial.polynomial.polyval(extrema, velocity)
            positions = np.polynomial.polynomial.polyval(zeros, coefficients)
            vmin[joint] = min(vmin[joint], float(np.min(velocities)))
            vmax[joint] = max(vmax[joint], float(np.max(velocities)))
            qmin[joint] = min(qmin[joint], float(np.min(positions)))
            qmax[joint] = max(qmax[joint], float(np.max(positions)))
            for left, right, qa, qb in zip(zeros[:-1], zeros[1:], positions[:-1], positions[1:]):
                difference = qb-qa
                positive[joint] += max(0., difference)
                negative[joint] += max(0., -difference)
                interval = [left, right, (left+right)/2,
                            *(x for x in extrema if left < x < right)]
                values = np.polynomial.polynomial.polyval(interval, velocity)
                strongest = float(values[np.argmax(np.abs(values))])
                sign = int(_signs(strongest, velocity_tolerance[joint]))
                if sign and (not signs[joint] or signs[joint][-1] != sign):
                    signs[joint].append(sign)
    return {
        'turns': np.array([max(0, len(values)-1) for values in signs]),
        'positive_travel': positive, 'negative_travel': negative,
        'velocity_min': vmin, 'velocity_max': vmax,
        'position_min': qmin, 'position_max': qmax,
    }


def _select_joint_handoff(chunk, now, observation_age, previous_state, previous_trajectory=None,
                          blend=.2, playback_rate=1.):
    """Return a compatible causal trajectory or an explicit rejection.

    ``observation_age`` and ``blend`` use wall seconds; chunk offsets use policy
    seconds. Established handoffs never move to a phase before observation age.
    The configured blend is a minimum; a late chunk is rejected rather than
    joined with an automatically shortened, sharper bridge. Both single- and
    dual-quintic bridges are evaluated. A genuine change from the old velocity
    direction to the new trajectory direction is allowed, whereas extra turns
    invented by the bridge are rejected on materially moving joints.

    Bootstrap preserves the existing feedback-to-offset-zero behavior for a
    fresh response, including its short-horizon timing adaptation. An expired
    first response is still rejected. Gripper timing is unchanged.
    """
    if (not all(math.isfinite(float(value)) for value in (now, observation_age, blend, playback_rate))
            or observation_age < 0 or blend <= 0 or playback_rate <= 0):
        raise ValueError('Handoff timing must be finite, with positive blend/rate and nonnegative age')
    dimension = len(chunk.arm_columns)
    previous_state = _state_arrays(previous_state, dimension)
    nominal = float(observation_age*playback_rate)
    diagnostics = {
        'planner': 'causal_curve_checked_v1',
        'nominal_policy_offset_s': nominal,
        'observation_age_s': float(observation_age),
        'configured_blend_s': float(blend),
        'candidate_count': 0, 'rejected_candidate_count': 0,
        'rejection_counts': {},
    }

    def rejected(reason):
        return HandoffSelection(None, reason, diagnostics)

    if nominal >= chunk.duration:
        return rejected('expired_horizon')
    if previous_trajectory is None:
        trajectory = RailTrajectory(chunk, now, 0., previous_state, blend, playback_rate)
        diagnostics.update(offset_s=0., blend_s=float(trajectory.blend), head_mode='dual',
                           tail_mode='dual', bootstrap=True,
                           bootstrap_alignment='fresh_feedback_offset_zero')
        return HandoffSelection(trajectory, 'bootstrap', diagnostics)
    if previous_state is None:
        previous_state = _state_arrays(previous_trajectory.state(now), dimension)
    q0, v0, a0 = previous_state

    # A full head and a full stop segment must fit. Unlike the paper baseline's
    # duration/3 cap, candidate generation does not shorten the requested blend.
    if chunk.duration-nominal < 2*blend*playback_rate-1e-12:
        return rejected('insufficient_horizon')

    raw_range = np.ptp(chunk.actions[:, chunk.arm_columns], axis=0)
    motion = np.maximum.reduce((raw_range, np.abs(v0)*blend, .5*np.abs(a0)*blend**2))
    # A completely stationary prediction can still specify a new static goal.
    # Give that degenerate case a spatial scale rather than zero objective
    # weights; no speed sign is inferred from the goal displacement.
    if np.max(motion) < 1e-12:
        motion = np.maximum(motion, np.abs(chunk.state(nominal)[0]-q0))
    global_motion = max(float(np.max(motion)), 1e-12)
    scale = np.maximum(motion, .05*global_motion)
    # Tiny nearly stationary joints contribute smoothly to cost without vetoing
    # a large-arm movement on the basis of an arbitrary floating-point sign.
    material = motion >= .025*global_motion
    dominant_speed = float(np.max(np.abs(v0)))
    moving_boundary = material & (np.abs(v0) >= .05*dominant_speed)
    weights = motion**2/(motion**2+(.05*global_motion)**2)
    weights /= max(float(weights.sum()), 1e-12)
    start = tuple(value[None, None, :] for value in previous_state)
    records = []
    rejection_counts = {'behind_continuation': 0, 'extra_reversal': 0,
                        'backward_excursion': 0, 'nonfinite_bridge': 0,
                        'exact_curve_check': 0}
    fractions = np.linspace(0., 1., 25)
    best_rejected_score = math.inf

    for duration in (blend, 1.25*blend, 1.5*blend):
        upper = float(chunk.duration-2*duration*playback_rate)
        if upper < nominal-1e-12:
            continue
        upper = max(nominal, upper)
        offsets = np.unique(np.linspace(nominal, upper, 9))
        count = len(offsets)
        policy_times = offsets[:, None]+fractions[None, :]*duration*playback_rate
        reference_q, reference_v, _ = chunk.state(policy_times)
        reference_v *= playback_rate
        end = tuple(value[:, None, :] for value in chunk.state(offsets+duration*playback_rate))
        end = (end[0], end[1]*playback_rate, end[2]*playback_rate**2)
        target = end[0][:, 0]
        velocity_tolerance = .015*scale/duration
        old_sign = _signs(v0, velocity_tolerance)
        reference_signs = _signs(reference_v, velocity_tolerance)
        reference_turns = _turns(reference_signs)
        # Check sustained new direction beyond the proposed join, not a single
        # noisy derivative at one point. A sustained opposite direction is an
        # intended reversal and gets one braking/turnaround allowance.
        future_times = offsets[:, None]+duration*playback_rate*np.array([1., 1.25, 1.5, 1.75, 2.])[None, :]
        _, future_v, _ = chunk.state(future_times)
        future_v *= playback_rate
        future_signs = _signs(future_v, velocity_tolerance)
        new_sign = _signs(np.median(future_v, axis=1), velocity_tolerance)
        coherent = np.all((future_signs == new_sign[:, None]) | (future_signs == 0), axis=1)
        first_new_sign = np.zeros((count, dimension), dtype=int)
        for index in range(reference_signs.shape[1]):
            first_new_sign = np.where(first_new_sign == 0, reference_signs[:, index], first_new_sign)
        intentional_turn = old_sign[None, :]*first_new_sign < 0
        allowed_turns = reference_turns+intentional_turn.astype(int)
        continuation = (coherent & (new_sign != 0) & (first_new_sign == new_sign)
                        & (old_sign[None, :]*new_sign > 0) & (reference_turns == 0))
        # A joint at rest has no inherited progress direction to preserve. It
        # may approach a new path and then follow it (one turn). Near-static
        # wrist signs must not veto substantial ongoing shoulder/elbow motion.
        check_joints = moving_boundary & (old_sign != 0)
        allowed_turns += ((old_sign[None, :] == 0)
                          & ((target-q0)*new_sign < -.015*scale)).astype(int)
        behind = np.any(continuation & check_joints[None, :]
                        & ((target-q0)*new_sign < -.015*scale), axis=1)

        old_q, old_v, _ = _reference_samples(previous_trajectory, now+fractions*duration)
        if old_q.shape != (len(fractions), dimension) or not np.isfinite(old_q).all() or not np.isfinite(old_v).all():
            raise ValueError('Previous trajectory returned an invalid state')
        old_velocity_scale = duration/scale
        old_new_weight = fractions[None, :, None]
        desired_q = (1-old_new_weight)*old_q[None]+old_new_weight*reference_q
        desired_v = (1-old_new_weight)*old_v[None]+old_new_weight*reference_v

        for mode, bridge_type in (('single', Quintic), ('dual', DualQuintic)):
            bridge = bridge_type(start, end, duration)
            positions, velocities, accelerations = bridge.state(fractions*duration)
            delta = np.diff(positions, axis=1)
            negative = np.maximum(0., -delta*new_sign[:, None]).sum(axis=1)
            reference_delta = np.diff(reference_q, axis=1)
            reference_negative = np.maximum(0., -reference_delta*new_sign[:, None]).sum(axis=1)
            excess_backward = np.maximum(0., negative-reference_negative)/scale
            bridge_turns = _turns(_signs(velocities, velocity_tolerance))
            extra_turn = np.any((bridge_turns > allowed_turns) & check_joints[None, :], axis=1)
            backward_bad = np.any(continuation & check_joints[None, :]
                                  & (excess_backward > .025), axis=1)
            q_error = np.mean(((positions-desired_q)/scale)**2, axis=1)@weights
            v_error = np.mean(((velocities-desired_v)*old_velocity_scale)**2, axis=1)@weights
            jerk = np.diff(accelerations, axis=1)/(duration/(len(fractions)-1))
            jerk_cost = np.mean((jerk*(duration**3)/scale)**2, axis=1)@weights
            variation = np.maximum(0., np.abs(delta).sum(axis=1)-np.abs(target-q0))/scale
            score = (q_error+.3*v_error+.001*jerk_cost
                     + 2*(variation**2@weights)
                     + .15*((offsets-nominal)/(duration*playback_rate))**2
                     + .1*((duration/blend)-1)**2)
            finite = (np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(velocities).all(axis=(1, 2))
                      & np.isfinite(accelerations).all(axis=(1, 2)) & np.isfinite(score))
            invalid = behind | extra_turn | backward_bad | ~finite
            rejection_counts['behind_continuation'] += int(behind.sum())
            rejection_counts['extra_reversal'] += int(extra_turn.sum())
            rejection_counts['backward_excursion'] += int(backward_bad.sum())
            rejection_counts['nonfinite_bridge'] += int((~finite).sum())
            diagnostics['candidate_count'] += count
            diagnostics['rejected_candidate_count'] += int(invalid.sum())
            if np.any(invalid & finite):
                best_rejected_score = min(best_rejected_score, float(np.min(score[invalid & finite])))
            for index in np.flatnonzero(~invalid):
                records.append((float(score[index]), float(offsets[index]), float(duration), mode,
                                allowed_turns[index], continuation[index], new_sign[index], check_joints,
                                reference_negative[index], velocity_tolerance,
                                float(np.max(excess_backward[index])),
                                float(np.max(np.abs(velocities[index]))),
                                float(np.max(np.abs(accelerations[index])))))

    diagnostics['rejection_counts'] = rejection_counts
    if math.isfinite(best_rejected_score):
        diagnostics['best_rejected_score'] = best_rejected_score
    # Audit only the best few sampled survivors. Rejecting instead of falling
    # back to an unaudited candidate keeps computation bounded per control tick.
    for record in sorted(records, key=lambda value: value[0])[:4]:
        (score, offset, duration, mode, allowed_turns, continuation, new_sign, check_joints,
         reference_negative, velocity_tolerance, sampled_backward, max_v, max_a) = record
        trajectory = RailTrajectory(chunk, now, offset, previous_state, duration,
                                    playback_rate, head_mode=mode, preserve_blend=True)
        exact = _exact_velocity_features(trajectory.head, duration, dimension, velocity_tolerance)
        backwards = np.where(new_sign > 0, exact['negative_travel'],
                             np.where(new_sign < 0, exact['positive_travel'], 0.))
        excess = np.maximum(0., backwards-reference_negative)/scale
        if (np.any((exact['turns'] > allowed_turns) & check_joints)
                or np.any(continuation & check_joints & (excess > .025))):
            rejection_counts['exact_curve_check'] += 1
            diagnostics['rejected_candidate_count'] += 1
            continue
        target = trajectory.reference_state(duration)[0]
        lower, upper = np.minimum(q0, target), np.maximum(q0, target)
        overshoot = np.maximum(0., np.maximum(lower-exact['position_min'], exact['position_max']-upper))/scale
        diagnostics.update(offset_s=offset, blend_s=duration, head_mode=mode, tail_mode=mode,
                           bootstrap=False, score=score,
                           max_normalized_backtrack=float(np.max(excess)),
                           max_normalized_overshoot=float(np.max(overshoot)),
                           predicted_backward_travel_rad=float(np.max(backwards)),
                           predicted_velocity_abs_max_rad_s=float(max(np.max(np.abs(exact['velocity_min'])),
                                                                       np.max(np.abs(exact['velocity_max'])))),
                           predicted_acceleration_sampled_abs_max_rad_s2=max_a,
                           allowed_reversal_joint_count=int(np.count_nonzero((allowed_turns > 0) & material)),
                           checked_moving_joint_count=int(np.count_nonzero(check_joints)),
                           phase_advance_s=float(offset-nominal))
        return HandoffSelection(trajectory, 'accepted', diagnostics)
    return rejected('incompatible_bridge')


def _tcp_values(provider, positions):
    positions = np.asarray(positions, dtype=float)
    xyz = np.asarray(provider(positions), dtype=float)
    if (xyz.ndim != positions.ndim+1 or xyz.shape[:-2] != positions.shape[:-1]
            or xyz.shape[-1] != 3 or xyz.shape[-2] < 1 or not np.isfinite(xyz).all()):
        raise ValueError('TCP provider must return finite [..., arms, 3] positions')
    return xyz


def _tcp_curve_metrics(provider, positions, reference, boundary, blend,
                       absolute_allowance, relative_allowance):
    """Compare cumulative backward travel at every prefix, for each TCP.

    New model reversals are retained: only travel beyond its aligned reference
    allowance is rejected. A later intentional return cannot excuse an earlier
    invented return. A resting TCP has no inherited direction to preserve.
    """
    combined = _tcp_values(provider, np.concatenate((positions, reference), axis=0))
    xyz, reference_xyz = np.split(combined, 2, axis=0)
    q0, v0, _ = boundary
    # This is a numerical directional derivative, not a commanded displacement.
    epsilon = 1e-5
    derivative = _tcp_values(provider, np.stack((q0-epsilon*v0, q0+epsilon*v0)))
    old_velocity = (derivative[1]-derivative[0])/(2*epsilon)
    old_speed = np.linalg.norm(old_velocity, axis=-1)
    direction = old_velocity/np.maximum(old_speed[:, None], 1e-15)
    steps = np.diff(xyz, axis=1)
    reference_steps = np.diff(reference_xyz, axis=1)
    projected = np.sum(steps*direction[None, None], axis=-1)
    reference_projected = np.sum(reference_steps*direction[None, None], axis=-1)
    backward = np.cumsum(np.maximum(0., -projected), axis=1)
    reference_backward = np.cumsum(np.maximum(0., -reference_projected), axis=1)
    reference_arc = np.cumsum(np.linalg.norm(reference_steps, axis=-1), axis=1)
    extra = np.maximum(0., backward-reference_backward)
    allowance = absolute_allowance+relative_allowance*reference_arc
    # Avoid treating the numerical direction of a stopped TCP as task intent.
    # This relative significance threshold does not limit a command's speed.
    moving = old_speed[None, :]*blend > np.maximum(1e-10, .01*reference_arc[:, -1])
    margin = np.where(moving[:, None, :], extra-allowance, -math.inf)
    return {
        'rejected': np.any(margin > 1e-10, axis=(1, 2)),
        'extra': np.max(extra, axis=1),
        'backward': backward[:, -1],
        'reference_backward': reference_backward[:, -1],
        'reference_arc': reference_arc[:, -1],
        'moving': moving,
        'old_speed': old_speed,
        'maximum_margin': np.max(margin, axis=(1, 2)),
    }


def select_handoff(chunk, now, observation_age, previous_state, previous_trajectory=None,
                   blend=.2, playback_rate=1., *, tcp_positions=None,
                   tcp_backtrack_allowance_m=.005, tcp_backtrack_relative=.02,
                   minimum_horizon_s=None):
    """Join near the incoming path's closest TCP phase, preserving continuity.

    ``tcp_positions`` maps arm positions [..., D] to [..., n_arms, 3]. Without
    it the original, conservative joint-only admission remains available for
    standalone callers. Observation age still rejects expired responses, but
    spatial phase is estimated against the current command reference instead
    of advancing by elapsed wall time. ``minimum_horizon_s`` stretches the
    selected remaining interval without extrapolating the model or changing
    publication rate. Whole-curve admission retains intentional model turns.
    """
    if tcp_positions is None:
        return _select_joint_handoff(chunk, now, observation_age, previous_state,
                                     previous_trajectory, blend, playback_rate)
    values = (now, observation_age, blend, playback_rate,
              tcp_backtrack_allowance_m, tcp_backtrack_relative)
    coverage = 0. if minimum_horizon_s is None else float(minimum_horizon_s)
    if (not all(math.isfinite(float(value)) for value in values) or not math.isfinite(coverage)
            or observation_age < 0 or blend <= 0 or playback_rate <= 0 or coverage < 0
            or tcp_backtrack_allowance_m < 0 or tcp_backtrack_relative < 0):
        raise ValueError('VLA-RAIL TCP planning parameters must be finite and nonnegative; blend/rate positive')
    dimension = len(chunk.arm_columns)
    prior = _state_arrays(previous_state, dimension)
    nominal = float(observation_age*playback_rate)
    bootstrap = previous_trajectory is None
    diagnostics = dict(planner='tcp_nearest_handoff_v3',
                       nominal_policy_offset_s=nominal, observation_age_s=float(observation_age),
                       configured_blend_s=float(blend), configured_playback_rate=float(playback_rate),
                       requested_minimum_horizon_s=coverage,
                       tcp_backtrack_allowance_m=float(tcp_backtrack_allowance_m),
                       tcp_backtrack_relative=float(tcp_backtrack_relative),
                       candidate_count=0, rejected_candidate_count=0, rejection_counts={})
    if nominal >= chunk.duration:
        return HandoffSelection(None, 'expired_horizon', diagnostics)
    if prior is None:
        prior = (_state_arrays(previous_trajectory.state(now), dimension)
                 if previous_trajectory is not None else
                 (chunk.actions[0, chunk.arm_columns], np.zeros(dimension), np.zeros(dimension)))
    q0, v0, a0 = prior
    if bootstrap:
        phases = np.array([0.])
        diagnostics.update(alignment_method='bootstrap_offset_zero', nearest_policy_phase_s=0.,
                           alignment_candidate_phases_s=[0.], nearest_has_future=True)
    else:
        phases, alignment = nearest_tcp_phases(chunk, q0, tcp_positions,
                                               tcp_backtrack_allowance_m, tcp_backtrack_relative)
        diagnostics.update(alignment)
    earliest = float(phases[0])
    remaining = chunk.duration-earliest
    phase_ranks = np.arange(len(phases))
    usable = phases < chunk.duration-1e-9
    phases, phase_ranks = phases[usable], phase_ranks[usable]
    if not len(phases):
        return HandoffSelection(None, 'nearest_phase_has_no_future', diagnostics)
    rates = np.minimum(playback_rate, (chunk.duration-phases)/coverage) if coverage > 0 else np.full_like(phases, playback_rate)
    lifetimes = (chunk.duration-phases)/rates
    raw_range = np.ptp(chunk.actions[:, chunk.arm_columns], axis=0)
    motion = np.maximum.reduce((raw_range, np.abs(v0)*blend, .5*np.abs(a0)*blend**2,
                                .1*np.abs(chunk.state(earliest)[0]-q0)))
    global_motion = max(float(np.max(motion)), 1e-12)
    scale = np.maximum(motion, .05*global_motion)
    weights = motion**2/(motion**2+(.05*global_motion)**2)
    weights /= max(float(weights.sum()), 1e-12)
    boundary = tuple(value[None, None, :] for value in prior)
    groups = []
    records = []
    grid_head = np.linspace(0., 1., 17)
    grid_middle = np.linspace(0., 1., 9)[1:]
    grid_tail = np.linspace(0., 1., 17)[1:]
    for head_duration in (blend, 1.25*blend, 1.5*blend, 2*blend, 2.5*blend):
        for tail_duration in (blend, .5*blend):
            valid = lifetimes+1e-12 >= head_duration+tail_duration
            if not np.any(valid):
                continue
            offsets, local_rates, durations = phases[valid], rates[valid], lifetimes[valid]
            ranks = phase_ranks[valid]
            count = len(offsets)
            tail_starts = durations-tail_duration
            elapsed = np.concatenate((np.broadcast_to(grid_head*head_duration, (count, 17)),
                                      head_duration+(tail_starts-head_duration)[:, None]*grid_middle,
                                      tail_starts[:, None]+tail_duration*grid_tail), axis=1)
            policy_times = offsets[:, None]+elapsed*local_rates[:, None]
            reference_q, reference_v, reference_a = chunk.state(policy_times)
            reference_v *= local_rates[:, None, None]
            reference_a *= local_rates[:, None, None]**2
            end = chunk.state(offsets+head_duration*local_rates)
            head = Quintic(boundary, (end[0][:, None], end[1][:, None]*local_rates[:, None, None],
                                     end[2][:, None]*local_rates[:, None, None]**2), head_duration)
            head_state = head.state(elapsed)
            tail_initial = chunk.state(chunk.duration-tail_duration*local_rates)
            tail = Quintic((tail_initial[0][:, None], tail_initial[1][:, None]*local_rates[:, None, None],
                            tail_initial[2][:, None]*local_rates[:, None, None]**2),
                           (np.broadcast_to(chunk.state(chunk.duration)[0], (count, 1, dimension)),
                            np.zeros((count, 1, dimension)), np.zeros((count, 1, dimension))), tail_duration)
            tail_state = tail.state(elapsed-tail_starts[:, None])
            q, v, a = (np.where((elapsed < head_duration)[..., None], h,
                                np.where((elapsed >= tail_starts[:, None])[..., None], t, r))
                       for h, t, r in zip(head_state, tail_state, (reference_q, reference_v, reference_a)))
            if previous_trajectory is not None:
                old_q, old_v, _ = _reference_samples(previous_trajectory, now+elapsed)
            else:
                old_q = q0+elapsed[..., None]*v0+.5*elapsed[..., None]**2*a0
                old_v = v0+elapsed[..., None]*a0
            mix = np.minimum(1., elapsed/head_duration)[..., None]
            desired_q = (1-mix)*old_q+mix*reference_q
            desired_v = (1-mix)*old_v+mix*reference_v
            intervals = np.diff(elapsed, axis=1)
            jerk = np.divide(np.diff(a, axis=1), intervals[..., None],
                             out=np.zeros_like(a[:, 1:]), where=intervals[..., None] > 1e-9)
            q_cost = np.mean(((q-desired_q)/scale)**2, axis=1)@weights
            v_cost = np.mean(((v-desired_v)*blend/scale)**2, axis=1)@weights
            jerk_cost = np.mean((jerk*blend**3/scale)**2, axis=1)@weights
            actual_travel = np.cumsum(np.abs(np.diff(q, axis=1)), axis=1)
            model_travel = np.cumsum(np.abs(np.diff(reference_q, axis=1)), axis=1)
            join_distance = np.abs(reference_q[:, 0]-q0)
            extra_joint = np.maximum(0., actual_travel-model_travel-join_distance[:, None])/scale
            joint_excursion = np.max(np.sqrt(np.sum(extra_joint**2*weights, axis=-1)), axis=1)
            score = (q_cost+.3*v_cost+.001*jerk_cost+2*joint_excursion**2
                     + .4*((offsets-earliest)/max(remaining, chunk.dt))**2
                     + .015*(head_duration/blend-1)**2)
            finite = (np.isfinite(q).all(axis=(1, 2)) & np.isfinite(v).all(axis=(1, 2))
                      & np.isfinite(a).all(axis=(1, 2)) & np.isfinite(score))
            for index in range(count):
                records.append(dict(offset=float(offsets[index]), rate=float(local_rates[index]),
                                    phase_rank=int(ranks[index]),
                                    duration=float(durations[index]), head=float(head_duration), tail=float(tail_duration),
                                    score=float(score[index]), joint_excursion=float(joint_excursion[index]),
                                    finite=bool(finite[index])))
            groups.append((q, v, a, reference_q, elapsed))
    if not records:
        return HandoffSelection(None, 'insufficient_horizon', diagnostics)
    q, v, a, reference_q, elapsed = (np.concatenate([group[index] for group in groups]) for index in range(5))
    tcp = _tcp_curve_metrics(tcp_positions, q, reference_q, prior, blend,
                             tcp_backtrack_allowance_m, tcp_backtrack_relative)
    finite = np.array([record['finite'] for record in records])
    # Joint-space cost protects configuration/orientation as well as TCP
    # translation. This secondary rejection concerns *additional* travel,
    # allowing the displacement actually needed to enter the new path.
    joint_rejected = np.array([record['joint_excursion'] > .5 for record in records])
    invalid = tcp['rejected'] | joint_rejected | ~finite
    diagnostics.update(candidate_count=len(records), rejected_candidate_count=int(invalid.sum()),
                       rejection_counts=dict(tcp_excess_backtrack=int(tcp['rejected'].sum()),
                                             joint_excess_excursion=int(joint_rejected.sum()),
                                             nonfinite_bridge=int((~finite).sum()), dense_curve_check=0))
    # Both acceptance and cost see the full head/body/tail. A final denser check
    # prevents a narrow Cartesian lobe from hiding between candidate samples.
    costs = np.array([record['score'] for record in records])
    costs += 5*np.mean((tcp['extra']/np.maximum(tcp_backtrack_allowance_m+
                                               tcp_backtrack_relative*tcp['reference_arc'], 1e-9))**2,
                       axis=1)
    survivors = np.flatnonzero(~invalid)
    # Distance is primary: exhaust viable bridges at the closest phase before
    # using a nearby fallback. Different head/tail fits compete within a phase.
    ranks = np.array([record['phase_rank'] for record in records])
    ranked = survivors[np.lexsort((costs[survivors], ranks[survivors]))]
    checked_per_phase = {}
    gripper_offset = 0. if bootstrap else nominal
    gripper_rate = (min(playback_rate, (chunk.duration-gripper_offset)/coverage)
                    if coverage > 0 else playback_rate)
    for index in ranked:
        record = records[index]
        rank = record['phase_rank']
        if checked_per_phase.get(rank, 0) >= 4:
            continue
        checked_per_phase[rank] = checked_per_phase.get(rank, 0)+1
        trajectory = RailTrajectory(chunk, now, record['offset'], prior, record['head'], record['rate'],
                                    head_mode='single', preserve_blend=True, tail_blend=record['tail'],
                                    gripper_offset=gripper_offset, gripper_playback_rate=gripper_rate)
        dense_times = np.unique(np.concatenate((np.linspace(0., trajectory.duration, 193),
                                                np.linspace(0., trajectory.blend, 65),
                                                np.linspace(trajectory.tail_start, trajectory.duration, 65))))
        dense_q, dense_v, dense_a = _reference_samples(trajectory, now+dense_times)
        dense_reference = chunk.state(trajectory.offset+dense_times*trajectory.time_scale)[0]
        dense = _tcp_curve_metrics(tcp_positions, dense_q[None], dense_reference[None], prior, blend,
                                   tcp_backtrack_allowance_m, tcp_backtrack_relative)
        if dense['rejected'][0]:
            diagnostics['rejection_counts']['dense_curve_check'] += 1
            diagnostics['rejected_candidate_count'] += 1
            continue
        diagnostics.update(offset_s=trajectory.offset, blend_s=trajectory.blend,
                           tail_blend_s=trajectory.tail_blend, head_mode='single', tail_mode='single',
                           time_scale=trajectory.time_scale, duration_s=trajectory.duration,
                           bootstrap=bootstrap, score=float(costs[index]),
                           phase_advance_s=float(trajectory.offset-earliest),
                           selected_alignment_rank=rank,
                           alignment_fallback_used=bool(rank),
                           alignment_fallback_reason=(None if not rank else
                               ('nearest_phase_has_no_future' if not diagnostics['nearest_has_future']
                                else 'nearest_bridge_incompatible')),
                           phase_minus_observation_age_s=float(trajectory.offset-nominal),
                           gripper_alignment='nominal_observation_clock',
                           gripper_offset_s=trajectory.gripper_offset,
                           gripper_time_scale=trajectory.gripper_time_scale,
                           max_tcp_extra_backtrack_m=float(np.max(dense['extra'])),
                           tcp_backward_travel_m=dense['backward'][0].tolist(),
                           tcp_reference_backward_travel_m=dense['reference_backward'][0].tolist(),
                           tcp_reference_arc_m=dense['reference_arc'][0].tolist(),
                           checked_moving_tcp_count=int(dense['moving'][0].sum()),
                           joint_extra_excursion_normalized=record['joint_excursion'],
                           predicted_velocity_abs_max_rad_s=float(np.max(np.abs(dense_v))),
                           predicted_acceleration_sampled_abs_max_rad_s2=float(np.max(np.abs(dense_a))),
                           future_retimed=bool(trajectory.time_scale < playback_rate-1e-12))
        if bootstrap:
            diagnostics['bootstrap_alignment'] = 'fresh_feedback_offset_zero'
        return HandoffSelection(trajectory, 'bootstrap' if bootstrap else 'accepted', diagnostics)
    return HandoffSelection(None, 'incompatible_bridge', diagnostics)
