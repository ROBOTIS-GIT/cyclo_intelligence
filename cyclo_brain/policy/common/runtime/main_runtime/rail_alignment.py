"""Bounded NumPy-only spatial phase search for an incoming cubic chunk."""
import numpy as np


def nearest_tcp_phases(chunk, anchor, tcp_positions, absolute_allowance=.003,
                       relative_allowance=.02):
    """Return the closest phase followed by a few spatially nearby alternatives.

    Cartesian distance sums both arms' squared distances. A sub-micrometre
    numerical tie uses joint distance, so a stationary TCP does not select an
    arbitrary wrist configuration. Local minima are refined together, without
    SciPy or per-arm scalar optimizers. This is a bounded search, not a proof
    that an arbitrarily oscillatory FK path has no smaller minimum.

    Fallbacks are only used if every bridge at the closest phase fails. They
    include adjacent policy samples and other nearby minima, never an age-based
    catch-up point. A terminal closest point stays visible as rank zero; one
    native interval before the endpoint is an explicit fallback, not a clipped
    replacement for that closest point.
    """
    anchor = np.asarray(anchor, dtype=float)
    anchor_tcp = np.asarray(tcp_positions(anchor), dtype=float)
    if (anchor_tcp.ndim != 2 or anchor_tcp.shape[-1] != 3
            or not np.isfinite(anchor_tcp).all()):
        raise ValueError('TCP provider must return finite [..., arms, 3] positions')

    def values(phases):
        q = chunk.state(phases)[0]
        xyz = np.asarray(tcp_positions(q), dtype=float)
        if xyz.shape != q.shape[:-1]+anchor_tcp.shape or not np.isfinite(xyz).all():
            raise ValueError('TCP provider must return finite [..., arms, 3] positions')
        cartesian = np.sum((xyz-anchor_tcp)**2, axis=(-2, -1))
        joint = np.sum((q-anchor)**2, axis=-1)
        # At most (0.1 micrometre)^2: only numerical Cartesian ties use posture.
        ordered = cartesian+1e-14*joint/(1.+joint)
        return ordered, cartesian, joint, xyz

    grid = np.linspace(0., chunk.duration, 65)
    ordered, _, _, xyz = values(grid)
    minima = np.flatnonzero((ordered[1:-1] <= ordered[:-2])
                            & (ordered[1:-1] <= ordered[2:]))+1
    minima = minima[np.argsort(ordered[minima], kind='stable')[:8]]
    refined = np.array([], dtype=float)
    # An interior minimum in the first/last grid cell can be hidden by a
    # seemingly best endpoint; refine both edge cells as well as found basins.
    left = np.concatenate(([grid[0], grid[-2]], grid[minima-1]))
    right = np.concatenate(([grid[1], grid[-1]], grid[minima+1]))
    if len(left):
        ratio = (np.sqrt(5.)-1.)/2.
        xleft, xright = right-ratio*(right-left), left+ratio*(right-left)
        fleft, fright = values(xleft)[0], values(xright)[0]
        for _ in range(25):
            keep_left = fleft <= fright
            right = np.where(keep_left, xright, right)
            left = np.where(keep_left, left, xleft)
            xleft, xright = right-ratio*(right-left), left+ratio*(right-left)
            # A single batch FK call evaluates both points of every basin.
            costs = values(np.concatenate((xleft, xright)))[0]
            fleft, fright = np.split(costs, 2)
        refined = (left+right)/2.
    basins = np.unique(np.concatenate(([0., chunk.duration], refined)))
    basin_order, _, _, _ = values(basins)
    nearest = float(basins[np.argmin(basin_order)])
    nearest_order, nearest_distance2, _, nearest_xyz = values(np.array([nearest]))
    path_arc = np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum(axis=0)
    allowance = absolute_allowance+relative_allowance*path_arc
    distance_slack = float(np.linalg.norm(allowance))
    neighbours = np.clip(nearest+chunk.dt*np.array([-1., -.5, .5, 1.]),
                         0., chunk.duration)
    alternatives = np.unique(np.concatenate((basins, neighbours,
                                             [max(0., chunk.duration-chunk.dt)])))
    order, distance2, _, _ = values(alternatives)
    nearby = np.sqrt(distance2) <= np.sqrt(nearest_distance2[0])+distance_slack+1e-12
    useful = (alternatives < chunk.duration-1e-9) & (abs(alternatives-nearest) > 1e-7)
    if nearest >= chunk.duration-1e-9:
        useful &= alternatives <= chunk.duration-chunk.dt+1e-12
    alternatives = alternatives[nearby & useful]
    order = order[nearby & useful]
    alternatives = alternatives[np.argsort(order, kind='stable')[:3]]
    phases = np.concatenate(([nearest], alternatives))
    _, _, _, candidate_xyz = values(phases)
    return phases, {
        'alignment_method': 'continuous_both_tcp_nearest',
        'alignment_anchor_tcp_m': anchor_tcp.tolist(),
        'nearest_policy_phase_s': nearest,
        'nearest_tcp_gap_m': np.linalg.norm(nearest_xyz[0]-anchor_tcp, axis=-1).tolist(),
        'alignment_candidate_phases_s': phases.tolist(),
        'alignment_candidate_tcp_gaps_m': np.linalg.norm(candidate_xyz-anchor_tcp, axis=-1).tolist(),
        'alignment_distance_slack_m': distance_slack,
        'nearest_has_future': bool(nearest < chunk.duration-1e-9),
        'alignment_coarse_samples': len(grid),
        'alignment_refined_basins': len(minima)+2,
    }
