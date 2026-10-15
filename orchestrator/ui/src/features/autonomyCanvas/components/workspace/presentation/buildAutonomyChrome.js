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
