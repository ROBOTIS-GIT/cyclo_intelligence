import buildAutonomyScene from "../../autonomyCanvas/components/workspace/presentation/buildAutonomyScene";
// Presentational mapping only; execution state stays in its owning controller.
export default function buildWorkspaceScene({
  runtime,
  workspace,
  recordData,
  design,
  designEditing
}) {
  const autonomy = buildAutonomyScene({
    runtime,
    workspace,
    recordData
  });
  const {
    mappingEditorActive,
    mappingTopicsActive,
    runTopicsActive,
    designLocalizationActive,
    stageNavigationTopicsActive,
    designMapActive,
    runFamilyStage,
    running,
    displayedMap,
    globalCostmap,
    localCostmap,
    scan,
    mappingPoseSync,
    runPoseSync,
    currentPose,
    navGoalPose,
    navGoalStatus,
    plan,
    footprint,
    displayTf,
    missionOverlayActive,
    renderedVisibleSpots,
    missionRunner,
    missionFollowRobot,
    renderedBehaviorNodes,
    designDocumentReady,
    mapEditor,
    designMapEditor,
    runDisplayMapEditor,
    handleCreateSpotAtPose,
    waypointBtLayer,
    runBtLayer,
    activeBtLayer,
    activeLayers,
    layerToggles,
    needsGlobalCostmap,
    needsLocalCostmap,
    needsScan,
    needsPlan,
    needsRobotModel,
    needsTf
  } = runtime;
  const {
    workspaceStage,
    robotType,
    missionMapLoaded,
    busy,
    interactionMode
  } = workspace;
  const {
    selectedSpotId,
    selectedBehaviorNodeId,
    behaviorPreviewNode,
    designMapPath,
    mapName,
    handleMoveBehaviorNode,
    setBtLayerSpotId
  } = design;
  const {
    renderedRouteView,
    missionRouteMode,
    missionRouteSourceId,
    handleMissionRouteSpotClick,
    handleSelectSpot,
    handleSelectBehaviorNode,
    handleMissionRouteMapClick,
    handleMoveSpot,
    handleClearMapSelection
  } = designEditing;
  return {
    observation: autonomy.observation,
    stage: {
      id: workspaceStage,
      mappingEditorActive,
      mappingTopicsActive,
      runTopicsActive,
      designLocalizationActive,
      navigationTopicsActive: stageNavigationTopicsActive,
      designMapActive,
      runFamily: runFamilyStage,
      running
    },
    scene: {
      robotType,
      displayedMap,
      globalCostmap,
      localCostmap,
      scan,
      mappingPoseSync,
      runPoseSync,
      currentPose,
      navGoalPose,
      navGoalStatus,
      plan,
      footprint,
      displayTf
    },
    mission: {
      overlayActive: missionOverlayActive,
      renderedVisibleSpots: renderedVisibleSpots,
      selectedSpotId,
      activeWaypointId: missionRunner.activeSpotId,
      followRobot: missionFollowRobot,
      renderedBehaviorNodes: renderedBehaviorNodes,
      selectedBehaviorNodeId,
      behaviorPreviewNode,
      renderedRouteView: renderedRouteView,
      routeMode: missionRouteMode,
      routeSourceId: missionRouteSourceId,
      mapLoaded: missionMapLoaded,
      documentReady: designDocumentReady,
      designMapPath,
      mapName,
      ...autonomy.mission
    },
    editors: {
      mapEditor,
      designMapEditor,
      runDisplayMapEditor
    },
    interaction: {
      busy,
      mode: interactionMode,
      onRouteSpotClick: handleMissionRouteSpotClick,
      onSpotClick: handleSelectSpot,
      onBehaviorNodeClick: handleSelectBehaviorNode,
      onRouteMapClick: handleMissionRouteMapClick,
      onSpotPoseChange: handleMoveSpot,
      onBehaviorNodePoseChange: handleMoveBehaviorNode,
      onMapClick: handleClearMapSelection,
      onMapPose: handleCreateSpotAtPose,
      onBtLayerClose: () => setBtLayerSpotId("")
    },
    bt: {
      waypointLayer: waypointBtLayer,
      runLayer: runBtLayer,
      activeLayer: activeBtLayer
    },
    layers: {
      active: activeLayers,
      toggles: layerToggles,
      needsGlobalCostmap,
      needsLocalCostmap,
      needsScan,
      needsPlan,
      needsRobotModel,
      needsTf
    }
  };
}
