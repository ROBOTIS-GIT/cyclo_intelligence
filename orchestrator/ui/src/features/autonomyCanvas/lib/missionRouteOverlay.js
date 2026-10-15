// Record and Operate share the same observation/child-mission map geometry.
export const OBSERVATION_WAYPOINT_COLORS = { idle: 0x8d7899 };

export function missionRouteOverlay({ id, source, executionSpots = [], order, activeWaypointId }) {
  if (!executionSpots.length) return null;
  const spots = [...new Map(executionSpots.map(spot => [spot.id, spot])).values()];
  return {
    id, spots,
    pathSpots: source ? [source, ...executionSpots, source] : executionSpots,
    order: order || spots.map((spot, index) => ({ id: spot.id, order: index + 1 })),
    activeWaypointId,
  };
}
