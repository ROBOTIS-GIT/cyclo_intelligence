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

import { useCallback } from "react";
import { useMapEditor } from "../../../components/navigation/MapEditor";
import useMissionRosScene, { deriveMissionRosSceneGates } from "./useMissionRosScene";
import useMapEditStageCoordinator from "../../missionCanvas/hooks/useMapEditStageCoordinator";
import useMissionMapScenePresentation from "./useMissionMapScenePresentation";

export default function useWorkspaceMapScene({
  workspace, design, runMapName, running, runtimeMapName, btNodeIsUp, setBtStatusText, runSpots, runMissionFlowNodes, runMissionFlowEdges
}) {
  const {
    busy, layersByStage, missionMapLoaded, navigationRuntimeMode, runBtVisualizationActive,
    workspaceStage, setMessage, mapEditorReloadToken, tfBufferRef, tfBufferRevision, setLayersByStage,
  } = workspace;
  const {
    designMapPath, mapName, designMapName, invalidateDesignMission, missionName, setMapName,
    designMapReloadToken, designMapBusy, designMissionLoadPhase, designMissionLoadError, spots,
    designBehaviorNodes, runBehaviorNodes,
  } = design;
  const sceneGates = deriveMissionRosSceneGates({
    busy,
    designMapPath,
    layersByStage,
    mapName,
    missionMapLoaded,
    navigationRuntimeMode,
    runBtVisualizationActive,
    runMapName,
    running,
    workspaceStage,
  });

  const {
    activeLayers,
    designLocalizationActive,
    designMapActive,
    mappingEditorActive,
    mappingRuntimeActive,
    mappingTopicsActive,
    runFamilyStage,
    runRuntimeActive,
    runSessionActive,
    runTopicsActive,
    stageNavigationTopicsActive,
  } = sceneGates;

  const mapEditor = useMapEditor({
    open: mappingEditorActive,
    mapName: designMapName,
    onMessage: setMessage,
    reloadToken: mapEditorReloadToken,
    autoSelect: false,
  });

  const handleSelectedMapEditIdentity = useCallback(({ mapName: nextMapName }) => {
    invalidateDesignMission({ mapName: nextMapName, missionName });
    setMapName(nextMapName);
  }, [invalidateDesignMission, missionName, setMapName]);

  const {
    picker: {
      open: showEditMapDialog,
      pendingPath: pendingEditMapPath,
      setPendingPath: setPendingEditMapPath,
      openPicker: handleOpenEditMapDialog,
      cancelPicker: cancelEditMapDialog,
      confirmPicker: handleConfirmEditMap,
    },
  } = useMapEditStageCoordinator({
    active: mappingEditorActive,
    currentMapName: designMapName,
    currentMissionName: missionName,
    editor: mapEditor,
    onSelectedMapIdentity: handleSelectedMapEditIdentity,
  });

  const designMapEditor = useMapEditor({
    open: designMapActive,
    mapName: designMapName,
    onMessage: setMessage,
    reloadToken: designMapReloadToken,
  });

  const runDisplayMapEditor = useMapEditor({
    open: runFamilyStage && missionMapLoaded,
    mapName: runtimeMapName,
    onMessage: setMessage,
  });

  const {
    amclPose,
    btActiveNodesText,
    currentPose,
    designMapAvailable,
    displayedMap,
    displayTf,
    footprint,
    globalCostmap,
    latestTf,
    localCostmap,
    mappingPoseSync,
    needsGlobalCostmap,
    needsLocalCostmap,
    needsPlan,
    needsRobotModel,
    needsScan,
    needsTf,
    plan,
    resetMappingPoseSync,
    runPoseSync,
    scan,
    topicRows,
  } = useMissionRosScene({
    sceneGates,
    mapEditorMap: mapEditor.map,
    designMapEditorMap: designMapEditor.map,
    runDisplayMapEditorMap: runDisplayMapEditor.map,
    btNodeIsUp,
    onBtStatus: setBtStatusText,
    tfBufferRef,
    tfBufferRevision,
  });

  const {
    designDocumentReady,
    missionOverlayActive,
    designVisibleSpots,
    runVisibleSpots,
    runRouteView,
    renderedBehaviorNodes,
    renderedVisibleSpots,
    designPanelSpots,
    designPanelBehaviorNodes,
    layerToggles,
  } = useMissionMapScenePresentation({
    workspaceStage,
    designMapName,
    runtimeMapName,
    designMapSample: {
      identity: designMapEditor.image?.path,
      map: designMapEditor.map,
    },
    runMapSample: {
      identity: runDisplayMapEditor.image?.path,
      map: runDisplayMapEditor.map,
    },
    designMapAvailable,
    designMapBusy,
    designMissionLoadPhase,
    designMissionLoadError,
    displayedMap,
    runSessionActive,
    designSpots: spots,
    runSpots,
    designBehaviorNodes,
    runBehaviorNodes,
    runMissionFlowNodes,
    runMissionFlowEdges,
    activeLayers,
    setLayersByStage,
  });

  return {
    activeLayers, designLocalizationActive, designMapActive, mappingEditorActive,
    mappingRuntimeActive, mappingTopicsActive, runFamilyStage, runRuntimeActive, runSessionActive,
    runTopicsActive, stageNavigationTopicsActive, mapEditor, showEditMapDialog, pendingEditMapPath,
    setPendingEditMapPath, handleOpenEditMapDialog, cancelEditMapDialog, handleConfirmEditMap,
    designMapEditor, runDisplayMapEditor, amclPose, btActiveNodesText, currentPose, displayedMap,
    displayTf, footprint, globalCostmap, latestTf, localCostmap, mappingPoseSync,
    needsGlobalCostmap, needsLocalCostmap, needsPlan, needsRobotModel, needsScan, needsTf, plan,
    resetMappingPoseSync, runPoseSync, scan, topicRows, designDocumentReady, missionOverlayActive,
    designVisibleSpots, runVisibleSpots, runRouteView, renderedBehaviorNodes, renderedVisibleSpots,
    designPanelSpots, designPanelBehaviorNodes, layerToggles,
  };
}
