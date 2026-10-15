import { missionRouteOverlay } from "./missionRouteOverlay";

// Temporary, read-only mission geometry. Map selection never creates an overlay.
export function recordMissionOverlay({ cycle, mission, routeView, mapName, activeWaypointId }) {
  if (!cycle?.isRunning || !cycle.pinnedSpot || mission?.mapName !== mapName
    || cycle.pinnedMissionId !== mission?.missionId) return null;
  const spots = routeView?.orderedSpots || [];
  if (!spots.length) return null;
  return missionRouteOverlay({
    id: mission.missionId, source: cycle.pinnedSpot,
    executionSpots: routeView.executionSpots || spots,
    order: routeView.order,
    activeWaypointId: cycle.returning || cycle.returnPending ? null : activeWaypointId,
  });
}
