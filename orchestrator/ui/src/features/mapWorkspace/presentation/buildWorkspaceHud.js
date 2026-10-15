import buildAutonomyHud from "../../autonomyCanvas/components/workspace/presentation/buildAutonomyHud";
// Presentational mapping only; execution state stays in its owning controller.
export default function buildWorkspaceHud({
  workspace,
  design,
  runtime,
  designEditing,
  mapping,
  recordData
}) {
  const autonomy = buildAutonomyHud({
    workspace,
    runtime,
    recordData
  });
  const {
    busy,
    interactionMode,
    runShutdownPending,
    labelToolsOpen,
    mapEditToolsOpen,
    setLabelToolsOpen,
    setMapEditToolsOpen,
    missionMapLoaded,
    setInteractionMode,
    runPoseInitialized
  } = workspace;
  const {
    canRedoDesign,
    canUndoDesign,
    designMapName,
    designMissionLoadError,
    designMapBusy,
    designMissionLoadPhase,
    missionName,
    designCatalog,
    openDeleteMissionDialog,
    handleOpenDuplicateMissionDialog,
    handleOpenRenameMissionDialog,
    showWaypointOptions
  } = design;
  const {
    designHistoryLocked,
    designMapActive,
    designDocumentReady,
    mappingRuntimeActive,
    missionRunnerActive,
    handleRedoDesign,
    handleUndoDesign,
    runRuntimeActive,
    dialogHost,
    mapEditor,
    runMapBusy,
    running,
    navGoalDriving,
    handleLocalize,
    btNodeBusy,
    missionRunnerStopping,
    handleRunMission,
    handleStopNavigation,
    runMapSnapshotInvalid
  } = runtime;
  const {
    missionRouteMode,
    handleCreateSpotAtRobot,
    importDesignWaypoint,
    runGuardedDesignAction,
    handleDesignMissionChange,
    startNewMission,
    handleSaveMission,
    handleToggleMissionRouteMode,
    handleToggleSpotMode,
    handleToggleWaypointOptions
  } = designEditing;
  const {
    handleDeleteSavedMap,
    handleOpenSaveMapDialog,
    handleStopMapping,
    protectedMapPaths,
    savedMaps,
    showSaveMapDialog
  } = mapping;
  return {
    hud: {
      design: {
        busy,
        canRedoDesign,
        canUndoDesign,
        currentMapName: designMapName,
        designHistoryLocked,
        designMapActive: designMapActive && (designDocumentReady || Boolean(designMissionLoadError)),
        designMapAvailable: designDocumentReady,
        designMapBusy: designMapBusy || designMissionLoadPhase === "loading",
        designMissionLoadError,
        interactionMode,
        mappingRuntimeActive,
        missionName,
        missionNames: designCatalog.names,
        missionRouteMode,
        missionRunnerActive,
        onCreateSpotAtRobot: handleCreateSpotAtRobot,
        onImportWaypoint: importDesignWaypoint,
        onDeleteMission: openDeleteMissionDialog,
        onDuplicateMission: handleOpenDuplicateMissionDialog,
        onMissionChange: name => runGuardedDesignAction(() => handleDesignMissionChange(name)),
        onNewMission: () => runGuardedDesignAction(startNewMission),
        onRedoDesign: handleRedoDesign,
        onRenameMission: handleOpenRenameMissionDialog,
        onSaveMission: handleSaveMission,
        onToggleMissionRouteMode: handleToggleMissionRouteMode,
        onToggleSpotMode: handleToggleSpotMode,
        onToggleWaypointOptions: handleToggleWaypointOptions,
        onUndoDesign: handleUndoDesign,
        runRuntimeActive,
        runShutdownPending,
        showWaypointOptions
      },
      mapping: {
        busy,
        dialogHost,
        mappingRuntimeActive,
        onDeleteSavedMap: handleDeleteSavedMap,
        onOpenSaveMapDialog: handleOpenSaveMapDialog,
        onStopNavigation: handleStopMapping,
        protectedPaths: protectedMapPaths,
        savedMaps,
        showSaveMapDialog
      },
      mapEdit: {
        labelToolsOpen,
        mapEditor,
        mapEditToolsOpen,
        setLabelToolsOpen,
        setMapEditToolsOpen
      },
      collect: autonomy.collect,
      run: {
        observation: autonomy.runObservation,
        btNodeBusy,
        busy,
        interactionMode,
        missionMapLoaded,
        missionRunnerActive,
        missionRunnerStopping,
        onLocalize: handleLocalize,
        onRunMission: handleRunMission,
        onStopNavigation: handleStopNavigation,
        runMapBusy,
        runMapSnapshotInvalid,
        runPoseInitialized,
        runShutdownPending,
        running
      },
      navigate: {
        busy,
        interactionMode,
        missionMapLoaded,
        navGoalDriving,
        onLocalize: handleLocalize,
        onStopNavigation: handleStopNavigation,
        runMapBusy,
        runPoseInitialized,
        runShutdownPending,
        running,
        setInteractionMode
      }
    }
  };
}
