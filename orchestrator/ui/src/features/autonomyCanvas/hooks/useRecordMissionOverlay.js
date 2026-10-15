import { useMemo } from "react";
import { recordMissionOverlay } from "../lib/recordMissionOverlay";

export default function useRecordMissionOverlay(cycle, mission, routeView, mapName, activeWaypointId) {
  const { isRunning, pinnedSpot, pinnedMissionId, returning, returnPending } = cycle;
  return useMemo(() => recordMissionOverlay({
    cycle: { isRunning, pinnedSpot, pinnedMissionId, returning, returnPending },
    mission, routeView, mapName, activeWaypointId,
  }), [isRunning, pinnedSpot, pinnedMissionId, returning, returnPending, mission, routeView, mapName, activeWaypointId]);
}
