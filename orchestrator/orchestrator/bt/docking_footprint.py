"""SG2 Autonomy Studio base outline, base_link metres; not an arm envelope."""
import math
import numpy as np

SG2_BASE_OUTLINE = [[-0.4029, -0.0522], [-0.3774, -0.1649], [-0.3626, -0.209], [-0.3241, -0.2352], [-0.0819, -0.301], [0.1516, -0.301], [0.1795, -0.2954], [0.2032, -0.2796], [0.219, -0.2559], [0.2246, -0.228], [0.2246, 0.228], [0.219, 0.2559], [0.2032, 0.2796], [0.1795, 0.2954], [0.1516, 0.301], [-0.0819, 0.301], [-0.3241, 0.2352], [-0.3626, 0.209], [-0.3774, 0.1649]]

# Convex envelope includes wheel corners protruding beyond the shell.
SG2_BASE_FOOTPRINT = [(-0.4029, -0.0522), (-0.3774, -0.1649), (-0.3626, -0.209), (-0.3241, -0.2352), (-0.0819, -0.301), (0.1516, -0.301), (0.1795, -0.2954), (0.2236, -0.28040000000000004), (0.2246, -0.228), (0.2246, 0.228), (0.2236, 0.28040000000000004), (0.1795, 0.2954), (0.1516, 0.301), (-0.0819, 0.301), (-0.3241, 0.2352), (-0.3626, 0.209), (-0.3774, 0.1649)]

def swept_obstacle(points, command, clearance=.01, horizon=.5):
    vx, vy, wz = map(float, command)
    if not all(math.isfinite(v) for v in (vx, vy, wz)):
        raise ValueError('Non-finite base command')
    if math.hypot(vx, vy) < 1e-5 and abs(wz) < 1e-5:
        return None
    scan = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(scan).all(): raise ValueError('Non-finite scan point')
    if not len(scan): return None
    polygon = np.asarray(SG2_BASE_FOOTPRINT)
    radius = float(np.linalg.norm(polygon, axis=1).max())
    travel = (math.hypot(vx, vy) + abs(wz)*radius)*horizon
    steps = max(1, math.ceil(travel/.005))
    padding = clearance + travel/steps/2
    edges = np.roll(polygon, -1, axis=0)-polygon
    lengths = np.linalg.norm(edges, axis=1)
    for t in np.linspace(0., horizon, steps+1):
        a = wz*t; c, s = math.cos(a), math.sin(a)
        if abs(wz) > 1e-8:
            x = (vx*s + vy*(c-1))/wz
            y = (vx*(1-c) + vy*s)/wz
        else: x, y = vx*t, vy*t
        local = (scan-[x,y]) @ np.array([[c,-s],[s,c]])
        inside = np.ones(len(scan), dtype=bool)
        for start, edge, length in zip(polygon, edges, lengths):
            delta = local-start
            inside &= (edge[0]*delta[:,1]-edge[1]*delta[:,0]) >= -padding*length
        found = np.flatnonzero(inside)
        if len(found):
            return {'point_base_m':scan[found[0]].tolist(), 'lookahead_s':float(t),
                    'command':list(command), 'clearance_m':clearance,
                    'footprint':'sg2_autonomy_base'}
    return None
