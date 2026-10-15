import { STAGE_OPERATE } from "../../../../mapWorkspace/lib/stages";
// Feature-owned projection only; runtime controllers retain state and execution ownership.
export default function buildAutonomyScene({
  runtime,
  workspace,
  recordData
}) {
  const {
    autonomyOperation,
    collectWaypoints,
    collectObservations,
    collectExecution,
    recordMissionMapOverlay,
    autonomyMap
  } = runtime;
  const {
    workspaceStage
  } = workspace;
  const {
    recordAuthoringBusy
  } = recordData;
  return {
    observation: autonomyOperation,
    mission: {
      collectWaypoints: {
        ...collectWaypoints,
        routeMode: collectObservations.route?.routeMode,
        orderMode: collectObservations.route?.orderMode,
        navigationActive: collectExecution.navigationActive,
        missionOverlay: recordMissionMapOverlay,
        routeSourceId: collectObservations.route?.routeSourceId,
        routeView: collectObservations.route?.routeView,
        onRouteMapClick: collectObservations.route?.handleMapClick,
        activeWaypointId: collectExecution.returning ? collectExecution.cycle.pinnedSpot?.id : collectExecution.navigation.isRunning ? collectExecution.navigation.toSpotId : null,
        selectedId: collectObservations.draft ? collectObservations.draft.id || "observation-preview" : collectObservations.selectedId,
        onPoseChange: !recordAuthoringBusy && !collectObservations.blocked && !collectObservations.working && !collectObservations.route?.orderMode && (!!collectObservations.draft || collectObservations.route?.routeMode) ? collectObservations.move : undefined,
        spots: collectObservations.draft ? [...collectObservations.spots.filter(s => s.id !== collectObservations.draft.id).map(s => ({
          ...s,
          poseEditable: false
        })), ...(collectObservations.draft.pose ? [{
          id: collectObservations.draft.id || "observation-preview",
          label: collectObservations.draft.label,
          pose: collectObservations.draft.pose
        }] : [])] : collectObservations.spots,
        onClearSelection: () => {
          if (collectObservations.draft || collectObservations.working) return;
          collectObservations.select("");
        },
        onSelect: id => {
          if (collectObservations.draft || collectObservations.working) return;
          if (collectObservations.route?.routeMode) {
            collectObservations.route.handleSpotClick(id);
            return;
          }
          if (collectExecution.cycle.isRunning) return;
          collectObservations.select(id);
        }
      },
      operateWaypoints: autonomyMap,
      ...(workspaceStage === STAGE_OPERATE ? {
        overlayActive: true,
        renderedVisibleSpots: autonomyMap.spots,
        activeWaypointId: autonomyMap.activeWaypointId,
        followRobot: autonomyMap.navigationActive,
        renderedBehaviorNodes: [],
        renderedRouteView: autonomyMap.route
      } : {})
    }
  };
}
