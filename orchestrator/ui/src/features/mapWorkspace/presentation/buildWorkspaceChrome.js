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

import buildAutonomyChrome from "../../autonomyCanvas/components/workspace/presentation/buildAutonomyChrome";
// Presentational mapping only; execution state stays in its owning controller.
export default function buildWorkspaceChrome({
  workspace,
  runtime,
  design,
  mapping,
  designEditing,
  recordData
}) {
  const autonomy = buildAutonomyChrome({
    workspace,
    runtime,
    recordData
  });
  const {
    workspaceStage,
    busy,
    runShutdownPending,
    setMessage,
    runPoseInitialized,
    missionMapLoaded
  } = workspace;
  const {
    btNodeBusy,
    mapEditor,
    mappingRuntimeActive,
    missionRunnerActive,
    handleOpenEditMapDialog,
    runBtLayer,
    missionRunner,
    runFamilyStage,
    runMapBusy,
    runRuntimeActive,
    running,
    showEditMapDialog,
    showRunMapDialog,
    waypointBtLayer,
    mappingEditorActive,
    designDocumentReady,
    designPanelBehaviorNodes,
    designPanelSpots,
    publishTeleopCommand,
    teleopDisabled,
    topicRows,
    navGoalStatus,
    runtimeMapName,
    runMapSnapshotInvalid,
    runMapName,
    runMissionName,
    runCatalog,
    handleMissionChange
  } = runtime;
  const {
    designMapBusy,
    setBtLayerSpotId,
    handleOpenDesignMapDialog,
    showDesignMapDialog,
    editingSpotId,
    editingSpotLabel,
    missionFlowEdges,
    handleDeleteBehaviorNode,
    setEditingSpotLabel,
    selectedBehaviorNodeId,
    selectedSpotId
  } = design;
  const {
    handleStartMapping
  } = mapping;
  const {
    designPanelRouteClosed,
    designPanelRouteSpots,
    missionRouteMode,
    handleCancelSpotRename,
    handleClearMissionRoute,
    handleCommitSpotRename,
    handleDeleteSpot,
    handleMissionRouteSpotClick,
    handleMoveRouteSpot,
    handleOpenMissionRouteLoop,
    handleOpenWaypointBt,
    handleRemoveRouteSpot,
    handleSelectBehaviorNode,
    handleSelectSpot,
    handleStartRenameSpot
  } = designEditing;
  return {
    stage: workspaceStage,
    header: {
      btNodeBusy,
      busy,
      designMapBusy,
      mapEditorBusy: mapEditor.busy,
      mappingRuntimeActive,
      missionRunnerActive,
      onBackToDesignMap: () => setBtLayerSpotId(""),
      onOpenDesignMap: handleOpenDesignMapDialog,
      onOpenEditMap: handleOpenEditMapDialog,
      onStartMapping: handleStartMapping,
      runBtLayer,
      runCurrentIndex: missionRunner.currentIndex,
      runFamilyStage,
      runMapBusy,
      runRuntimeActive,
      runShutdownPending,
      runTotal: missionRunner.total,
      running,
      showDesignMapDialog,
      showEditMapDialog,
      showRunMapDialog,
      waypointBtLayer,
      ...autonomy.header
    },
    map: {
      mappingEditorActive,
      waypointBtLayer
    },
    sidebar: {
      design: {
        busy,
        designDocumentReady,
        designPanelBehaviorNodes,
        designPanelRouteClosed,
        designPanelRouteSpots,
        designPanelSpots,
        editingSpotId,
        editingSpotLabel,
        missionFlowEdges,
        missionRouteMode,
        onCancelSpotRename: handleCancelSpotRename,
        onClearMissionRoute: handleClearMissionRoute,
        onCommitSpotRename: handleCommitSpotRename,
        onDeleteBehaviorNode: handleDeleteBehaviorNode,
        onDeleteSpot: handleDeleteSpot,
        onEditingSpotLabelChange: setEditingSpotLabel,
        onMissionRouteSpotClick: handleMissionRouteSpotClick,
        onMoveRouteSpot: handleMoveRouteSpot,
        onOpenMissionRouteLoop: handleOpenMissionRouteLoop,
        onOpenWaypointBt: handleOpenWaypointBt,
        onRemoveRouteSpot: handleRemoveRouteSpot,
        onSelectBehaviorNode: handleSelectBehaviorNode,
        onSelectSpot: handleSelectSpot,
        onStartRenameSpot: handleStartRenameSpot,
        selectedBehaviorNodeId,
        selectedSpotId
      },
      mapping: {
        onMessage: setMessage,
        onPublishTeleop: publishTeleopCommand,
        teleopDisabled,
        topicRows
      },
      navigation: {
        goalStatus: navGoalStatus,
        mapName: runtimeMapName,
        poseReady: runPoseInitialized,
        topicRows
      },
      run: {
        mapName: missionMapLoaded && !runMapSnapshotInvalid ? runMapName : "",
        missionName: runMissionName,
        missionNames: runCatalog.names,
        missionSelectDisabled: !!busy || missionRunnerActive || runMapBusy || !missionMapLoaded,
        onMissionChange: handleMissionChange,
        onTaskParametersSaved: runtime.reloadRunTaskParameters,
        poseReady: runPoseInitialized,
        runner: missionRunner,
        running,
        topicRows
      },
      ...autonomy.sidebar
    }
  };
}
