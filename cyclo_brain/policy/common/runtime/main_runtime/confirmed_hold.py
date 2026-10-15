"""Confirm policy motion holds from fresh feedback, not publisher delivery."""
import math
import time

HOLD_CAPABILITY = 'confirmed_pose_hold_v2'


def hold_current_pose(robot, action_keys, timeout=3.0):
    deadline = time.monotonic() + timeout
    receipt = robot.publish_current_pose_hold(action_keys, duration_s=.1)
    if not isinstance(receipt, dict) or not receipt.get('positions'):
        raise RuntimeError('RobotClient does not support confirmed pose holds')
    expected = receipt['positions']
    commanded = receipt['commanded_at']
    previous = None
    stable = 0
    while time.monotonic() < deadline:
        state = robot.current_hold_feedback()
        positions, received = state['positions'], state['received']
        fresh = all(received.get(n, 0) > commanded for n in expected)
        close = all(n in positions and math.isfinite(positions[n])
                    and abs(positions[n] - target) <= .03 for n, target in expected.items())
        moved = previous is None or all(received.get(n, 0) > previous['received'].get(n, 0) for n in expected)
        slow = previous is not None and all(
            abs(positions.get(n, float('inf')) - previous['positions'].get(n, 0))
            / max(1e-9, received.get(n, 0) - previous['received'].get(n, 0)) <= .08
            for n in expected)
        base_still = True
        if receipt.get('mobile'):
            odom = state.get('odom', {})
            base_still = state.get('odom_received', 0) > commanded and all(
                key in odom and len(odom[key]) == 3 and all(math.isfinite(float(v)) and abs(v) <= .03 for v in odom[key])
                for key in ('linear_velocity', 'angular_velocity'))
        if moved:
            stable = stable + 1 if fresh and close and slow and base_still else 0
            previous = state
        if stable >= 3:
            return
        time.sleep(.02)
    raise RuntimeError('Current-pose hold not confirmed by fresh joint/base feedback')
