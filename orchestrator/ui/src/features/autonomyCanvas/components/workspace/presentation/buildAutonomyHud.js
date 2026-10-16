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

import { STAGE_OPERATE } from "../../../../mapWorkspace/lib/stages";
import { executionPhaseLabel } from "../../../lib/executionState";
// Feature-owned projection only; runtime controllers retain state and execution ownership.
export default function buildAutonomyHud({
  workspace,
  runtime,
  recordData
}) {
  const {
    busy,
    interactionMode,
    runShutdownPending,
    missionMapLoaded,
    setInteractionMode,
    runPoseInitialized,
    setMessage,
    workspaceStage
  } = workspace;
  const {
    missionRunnerActive,
    runRuntimeActive,
    collectObservations,
    collectExecution,
    linkedRecordMissionNames,
    recordMissionName,
    collectMission,
    runMapBusy,
    runtimeMapName,
    running,
    navGoalDriving,
    handleLocalize,
    autonomyOperation
  } = runtime;
  const {
    collectDraft,
    qwenObservation
  } = recordData;
  return {
    collect: {
      observations: collectObservations,
      observationName: collectExecution.activeObservation?.label,
      executionLabel: executionPhaseLabel(collectExecution.executionState, {
        compact: true
      }),
      missionNames: linkedRecordMissionNames,
      selectedMission: recordMissionName,
      missionLoading: collectMission.loading,
      missionDisabled: !missionMapLoaded || !!busy || runMapBusy || runShutdownPending || !collectExecution.observationReady || collectExecution.cycle.isRunning,
      onMissionChange: collectMission.select,
      canStartMission: collectExecution.canRunMission && !collectMission.loading && !collectObservations.planDirty,
      onRunMission: collectExecution.startChildMission,
      mapName: missionMapLoaded ? runtimeMapName : "",
      observationSelectionDisabled: collectExecution.cycle.isRunning,
      onGoToObservation: id => {
        setInteractionMode("view");
        return collectExecution.goToSpot(id);
      },
      clickNavigationDisabled: !!collectObservations.draft || collectObservations.planDirty || collectObservations.working || collectObservations.route?.routeMode || collectExecution.stopping || collectExecution.needsStop || !missionMapLoaded || !collectObservations.spots.length || !!busy || runMapBusy || runShutdownPending || collectExecution.cycle.isRunning || collectExecution.missionRunner.isRunning,
      navigationActive: running || runRuntimeActive || navGoalDriving,
      onStopRoute: collectExecution.stop,
      patrolRunning: collectExecution.navigation.kind === "patrol" && collectExecution.navigation.isRunning,
      patrolPaused: collectExecution.navigation.patrolPaused,
      onPausePatrol: collectExecution.pausePatrol,
      onResumePatrol: collectExecution.resumePatrol,
      onPreviousObservation: collectExecution.previousObservation,
      canPreviousObservation: collectExecution.canPreviousObservation && !busy && !runMapBusy && !runShutdownPending,
      onSkipObservation: collectExecution.skipObservation,
      canSkipObservation: collectExecution.canSkipObservation && !busy && !runMapBusy && !runShutdownPending,
      stopping: collectExecution.stopping,
      shutdownPending: runShutdownPending,
      onStart: () => {
        setInteractionMode("view");
        return collectExecution.startPatrol();
      },
      startDisabled: !collectObservations.configuration?.plan?.observations.length || collectObservations.planDirty || collectObservations.route?.routeMode || !!collectObservations.draft || collectObservations.working || !!busy || runMapBusy || runShutdownPending || collectExecution.stopping || collectExecution.needsStop || collectExecution.cycle.isRunning || collectExecution.missionRunner.isRunning,
      onLocalize: handleLocalize,
      localizing: interactionMode === "initial",
      localized: running && runPoseInitialized,
      localizeDisabled: !!busy || runMapBusy || !missionMapLoaded || runShutdownPending || missionRunnerActive,
      routeActive: collectExecution.isRunning,
      authoringDisabled: collectDraft.active || collectDraft.recording || collectDraft.busy,
      onMessage: setMessage
    },
    runObservation: workspaceStage === STAGE_OPERATE ? autonomyOperation : qwenObservation
  };
}
