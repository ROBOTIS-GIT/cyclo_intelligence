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

import { STAGE_COLLECT } from "../../../../mapWorkspace/lib/stages";
// Feature-owned projection only; runtime controllers retain state and execution ownership.
export default function buildAutonomyChrome({
  workspace,
  runtime,
  recordData
}) {
  const {
    workspaceStage,
    busy,
    runShutdownPending,
    setMessage,
    runPoseInitialized,
    missionMapLoaded
  } = workspace;
  const {
    mappingRuntimeActive,
    missionRunnerActive,
    openRecordMapPicker,
    handleOpenRunMapDialog,
    collectExecution,
    collectObservations,
    runMapBusy,
    running,
    publishTeleopCommand,
    runtimeMapName,
    handleTestModel,
    recordMissionName,
    autonomyOperation
  } = runtime;
  const {
    collectDraft,
    qwenObservation,
    recordSession,
    datasetSelection,
    setDatasetSelection,
    setCollectEditorBusy
  } = recordData;
  return {
    header: {
      onOpenRunMap: workspaceStage === STAGE_COLLECT ? openRecordMapPicker : handleOpenRunMapDialog,
      recordMapChangeBlocked: collectExecution.isRunning || collectExecution.stopping || collectObservations.working || !!collectObservations.draft || collectObservations.planDirty || collectObservations.routeEditor?.dirty
    },
    sidebar: {
      collect: {
        draft: collectDraft,
        observation: qwenObservation,
        recordSession,
        onTestModel: handleTestModel,
        datasetSelection,
        onDatasetSelectionChange: setDatasetSelection,
        onEditorBusyChange: setCollectEditorBusy,
        recordMission: recordMissionName,
        recordObservationSpotId: collectObservations.selectedId,
        captureDisabled: !!collectObservations.draft || collectObservations.working || collectObservations.planDirty || collectObservations.route?.routeMode,
        routeRunning: missionRunnerActive,
        onPublishTeleop: publishTeleopCommand,
        onMessage: setMessage,
        teleopDisabled: !!busy || runShutdownPending || missionRunnerActive || mappingRuntimeActive,
        observationDisabled: !!busy || runMapBusy || runShutdownPending,
        mapName: runtimeMapName,
        mapLoaded: missionMapLoaded,
        poseReady: runPoseInitialized,
        running
      },
      operate: {
        observation: autonomyOperation,
        mapName: runtimeMapName,
        mapLoaded: missionMapLoaded,
        poseReady: runPoseInitialized,
        running
      }
    }
  };
}
