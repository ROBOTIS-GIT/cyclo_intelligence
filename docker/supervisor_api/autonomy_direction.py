"""Session-local patrol direction; saved route order remains the reference."""
from fastapi import HTTPException


class PatrolRedirect(RuntimeError):
    """Current patrol goal was cancelled and drained for a new route direction."""


def next_route_index(index, count, direction, closed):
    if count < 1: raise ValueError('Route is empty')
    candidate = index + direction
    return candidate % count if closed else min(count - 1, max(0, candidate))


def change_direction(controller, action):
    c = controller
    with c.control_lock, c.lock:
        if not c.state.get('active') or c.cancelled.is_set():
            raise HTTPException(409, 'Start autonomy before changing patrol direction.')
        if c.state.get('observation_mode') != 'patrol' or len(c.route_points) < 2:
            raise HTTPException(409, 'Patrol needs at least two observation points.')
        old = c.state.get('patrol_direction', 1)
        direction = -1 if action == 'reverse_patrol' else 1
        context = dict(c.state.get('observation_context') or {})
        index = c.state.get('patrol_index', 0)
        target = next_route_index(index, len(c.route_points), direction, c.plan_snapshot.get('route_closed', True))
        if context.get('phase') == 'observing':
            context.update(can_advance=target != index,
                           next_spot_id=c.route_points[target]['spot_id'] if target != index else None)
        c.instruction_version += 1
        c.transition(patrol_direction=direction, observation_context=context)
        moving = c.state.get('phase') == 'positioning' and direction != old
        if moving and c.runtime:
            # Worker cancels and drains only its navigation goal; no session Stop.
            c.runtime.patrol_redirect.set()
        label = 'reverse' if direction == -1 else 'forward'
        c.event('Patrol direction changed', label)
        if c.state.get('mode') == 'mission' or c.state.get('phase') == 'returning':
            return f'Patrol direction is now {label}. It will apply after the mission returns to its observation point.'
        if moving:
            return f'Patrol direction is now {label}. Navigation will turn back toward the last observation point after cancelling the current goal.'
        return f'Patrol direction is now {label}. The next patrol departure will use this order.'
