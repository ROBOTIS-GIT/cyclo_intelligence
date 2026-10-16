// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { useRef } from "react";
import { STAGE_COLLECT, STAGE_OPERATE } from "../../mapWorkspace/lib/stages";
import { useAutonomyMapView } from "../components/autonomyMapView";
import useAutonomyOperation from "./useAutonomyOperation";
import useRecordObservationSpots from "./useRecordObservationSpots";
import useRecordExecution from "./useRecordExecution";
import useRecordMissionOverlay from "./useRecordMissionOverlay";

// Compose Autonomy-owned behavior around the host's single navigation/BT owner.
// Keep hooks mounted across stages so shutdown and pending requests retain ownership.
export default function useAutonomyWorkspaceRuntime({
  workspaceStage, missionMapLoaded, running, runPoseInitialized,
  runDisplayMapEditor, runShutdownPending, runtimeMapName, missionRunner, busy,
  runMissionSnapshot, currentPose, setInteractionMode, runStartRequestRef,
  setPendingRunStart, sendMissionGoal, cancelNavigateToPoseGoal, setMessage,
  btStatusRef, btStatusHistoryRef, callService, stopMissionBt,
  ensureMissionBtActive, releaseMissionBt, missionRunnerFlags, runRouteView,
}) {
  const autonomyOperation = useAutonomyOperation({
    enabled: workspaceStage === STAGE_OPERATE && missionMapLoaded && running && runPoseInitialized && !runDisplayMapEditor.busy && !runShutdownPending,
    mapName: missionMapLoaded ? runtimeMapName : null,
    busy: missionRunner.isRunning || !!busy,
  });
  const recordAuthoringLock = useRef(false);
  const collectObservations = useRecordObservationSpots({
    enabled: workspaceStage === STAGE_COLLECT && missionMapLoaded,
    mapName: runtimeMapName,
    missionId: runMissionSnapshot.missionId,
    currentPose,
    localized: running && runPoseInitialized,
    blocked: !!busy || runShutdownPending || missionRunner.isRunning || autonomyOperation.needsStop,
    isBlocked: () => recordAuthoringLock.current,
    onMapMode: setInteractionMode,
  });
  const recordMissionStartRef = useRef(null);
  const collectExecution = useRecordExecution({
    enabled: workspaceStage === STAGE_COLLECT && missionMapLoaded,
    mapName: runtimeMapName, localized: running && runPoseInitialized, currentPose,
    manager: collectObservations, mission: runMissionSnapshot, missionRunner,
    startMission: () => recordMissionStartRef.current?.(),
    cancelMissionStart: () => { runStartRequestRef.current += 1; setPendingRunStart(null); },
    sendGoal: sendMissionGoal, cancelGoal: cancelNavigateToPoseGoal, onMessage: setMessage,
    runtime: { btStatusRef, btStatusHistoryRef, callService, stopBt: stopMissionBt,
      ensureBtActive: ensureMissionBtActive, releaseBt: releaseMissionBt, getFlags: missionRunnerFlags },
  });
  const recordMissionMapOverlay = useRecordMissionOverlay(collectExecution.cycle, runMissionSnapshot,
    runRouteView, runtimeMapName, missionRunner.activeSpotId);
  const missionRunnerActive = missionRunner.isRunning || autonomyOperation.needsStop || collectExecution.isRunning;
  const recordNavigationMoving = collectExecution.navigation.isRunning && !collectExecution.navigation.dwelling;
  const collectWaypoints = {
    spots: collectObservations.spots, selectedId: collectObservations.selectedId,
    moving: collectExecution.navigation.isRunning && !collectExecution.navigation.dwelling,
    stopping: collectExecution.stopping,
    navigate: collectExecution.goToSpot, stop: collectExecution.stop,
    isMoving: () => collectExecution.navigation.isRunning && !collectExecution.navigation.dwelling,
  };
  const autonomyMap = useAutonomyMapView(autonomyOperation);

  return {
    autonomyOperation, recordAuthoringLock, collectObservations, recordMissionStartRef,
    collectExecution, recordMissionMapOverlay, missionRunnerActive,
    recordNavigationMoving, collectWaypoints, autonomyMap,
  };
}
