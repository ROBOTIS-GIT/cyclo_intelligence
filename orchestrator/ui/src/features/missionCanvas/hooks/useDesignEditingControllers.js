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

import useDesignDocumentPersistence from "./useDesignDocumentPersistence";
import useDesignWaypointInteraction from "./useDesignWaypointInteraction";
import { useCallback, useEffect } from "react";
import { STAGE_AUTHORING } from "../../mapWorkspace/lib/stages";
import { filterMissionFlowEdges, syncMissionFlowNodesWithSpots } from "../../mapWorkspace/lib/missionFlow";
import useWaypointTaskFileController from "./useWaypointTaskFileController";
import useDesignDocumentContentController from "./useDesignDocumentContentController";

export default function useDesignEditingControllers({
  design, workspace, designVisibleSpots, designDocumentReady, runCommand, runSessionActive, runRouteView, waitForAutoLocalizedPose, mappingRuntimeActive, runRuntimeActive, missionRunnerActive, designLocalizationActive, invalidateNavigationStatus, clearLocalizationPoseCache, handleStopNavigation
}) {
  const {
    designCatalog, mapName, missionName, btLayerSpotId, designMapBusy, designMissionLoadPhase,
    designMissionLoadError, missionBtFiles, recordBtEdit, replaceLiveBtFiles,
    checkpointPersistedBtFile, setPersistedRevision, reconcileDesignDirty, getLiveBtFiles,
    getPersistedRevision, getPersistedLocalBtPaths, hasPersistedLocalBtPath, hasActiveDesignSave,
    captureDesignMissionLease, isDesignMissionLeaseCurrent, captureDesignHistory, resetDesignHistory,
    saveDesignMissionRef, setSpots, setMissionFlowNodes, setMissionFlowEdges,
    resetMissionRouteEditingRef, setSelectedSpotId, setBtLayerSpotId, applyMissionMetadata,
    mergeBehaviorMapPatch, commitDesignLedgerSnapshot, resetNewDesignLedger, clearBehaviorSelection,
    clearPendingBehaviorPlacement, setEditingSpotId, setEditingSpotLabel, setDesignMissionLoadError,
  } = design;
  const { busy, setBusy, setMessage, workspaceStage } = workspace;
  const designMissionIsStored = (
    designCatalog.mapName === mapName
    && designCatalog.names.includes(missionName)
  );

  const {
    selectedSpot: selectedBtLayerSpot,
    defaultPath: selectedBtLayerDefaultPath,
    filePaths: selectedBtLayerPaths,
    selectedPath: selectedBtLayerPath,
    loadingPath: missionBtLoadingPath,
    fileActionsDisabled: localBtFileActionsDisabled,
    loadXml: loadMissionLocalBtXml,
    saveXml: saveMissionLocalBtXml,
    selectXml: selectMissionLocalBtXml,
    saveXmlAs: saveMissionLocalBtXmlAs,
    setDefaultXml: setMissionLocalBtDefault,
    onEditorXmlChange: handleMissionLocalBtXmlChange,
    resetSelectedPaths: resetMissionLocalBtSelections,
    forgetSpotSelection: forgetMissionLocalBtSelection,
    isBusy: isWaypointTaskFileBusy,
  } = useWaypointTaskFileController({
    spots: designVisibleSpots,
    selectedSpotId: btLayerSpotId,
    mapName,
    missionName,
    missionStored: designMissionIsStored,
    busy,
    operationsDisabled: designMapBusy || designMissionLoadPhase !== "idle",
    missionLoadError: designMissionLoadError,
    ledger: {
      missionBtFiles,
      recordBtEdit,
      replaceLiveBtFiles,
      checkpointPersistedBtFile,
      setPersistedRevision,
      reconcileDirty: reconcileDesignDirty,
      getLiveBtFiles,
      getPersistedRevision,
      getPersistedLocalBtPaths,
      hasPersistedLocalBtPath,
      hasActiveSave: hasActiveDesignSave,
    },
    captureDocumentLease: captureDesignMissionLease,
    isDocumentLeaseCurrent: isDesignMissionLeaseCurrent,
    captureHistory: captureDesignHistory,
    resetHistory: resetDesignHistory,
    saveMissionRef: saveDesignMissionRef,
    setSpots,
    setBusy,
    onMessage: setMessage,
  });

  useEffect(() => {
    if (workspaceStage !== STAGE_AUTHORING || !designDocumentReady) return;
    setMissionFlowNodes((current) => (
      syncMissionFlowNodesWithSpots(current, designVisibleSpots)
    ));
    setMissionFlowEdges((current) => (
      filterMissionFlowEdges(current, designVisibleSpots)
    ));
  }, [
    designDocumentReady,
    designVisibleSpots,
    setMissionFlowEdges,
    setMissionFlowNodes,
    workspaceStage,
  ]);

  useEffect(() => {
    if (workspaceStage === STAGE_AUTHORING && designDocumentReady) {
      return;
    }
    resetMissionRouteEditingRef.current();
  }, [designDocumentReady, resetMissionRouteEditingRef, workspaceStage]);

  const applySpots = useCallback((nextSpots) => {
    setSpots(nextSpots);
    setSelectedSpotId((current) => (
      nextSpots.some((spot) => spot.id === current) ? current : ""
    ));
    setBtLayerSpotId((current) => (
      nextSpots.some((spot) => spot.id === current) ? current : ""
    ));
  }, [setBtLayerSpotId, setSelectedSpotId, setSpots]);

  const {
    commitLoadedSnapshot: commitLoadedDesign,
    resetNewDocument: resetNewDesignDocument,
    applySavedCanonicalSpots,
  } = useDesignDocumentContentController({
    setMetadata: applyMissionMetadata,
    applySpots,
    updateSpots: setSpots,
    setFlowNodes: setMissionFlowNodes,
    setFlowEdges: setMissionFlowEdges,
    mergeBehaviorMapPatch,
    commitLedgerSnapshot: commitDesignLedgerSnapshot,
    resetLedgerNewDocument: resetNewDesignLedger,
    resetHistory: resetDesignHistory,
    resetLocalBtSelections: resetMissionLocalBtSelections,
    clearBehaviorSelection,
    clearPendingBehaviorPlacement,
    resetRouteEditing: () => resetMissionRouteEditingRef.current(),
    setEditingSpotId,
    setEditingSpotLabel,
    setLoadError: setDesignMissionLoadError,
  });

  useEffect(() => {
    if (!btLayerSpotId) return;
    if (
      workspaceStage !== STAGE_AUTHORING ||
      !designVisibleSpots.some((spot) => spot.id === btLayerSpotId)
    ) {
      setBtLayerSpotId("");
    }
  }, [btLayerSpotId, designVisibleSpots, setBtLayerSpotId, workspaceStage]);

  const {
    handleDesignMissionChange, startNewMission, showSaveMissionDialog, saveMissionName,
    setSaveMissionName, handleSaveMission, closeSaveMissionDialog, handleConfirmSaveMission,
    markDesignDirty, showUnsavedDialog, unsavedSavePending, runGuardedDesignAction,
    resolveUnsavedDialog, cancelUnsavedDialog,
  } = useDesignDocumentPersistence({
    workspace, design, commitLoadedDesign, resetNewDesignDocument, designVisibleSpots, applySavedCanonicalSpots, runCommand, isWaypointTaskFileBusy
  });

  const {
    cancelPendingDesignLocalization,
    handleSelectSpot,
    missionRouteMode,
    missionRouteSourceId,
    handleToggleMissionRouteMode,
    handleMissionRouteSpotClick,
    handleMissionRouteMapClick,
    handleClearMissionRoute,
    handleOpenMissionRouteLoop,
    handleMoveRouteSpot,
    handleRemoveRouteSpot,
    resetMissionRouteEditing,
    renderedRouteView,
    designPanelRouteSpots,
    designPanelRouteClosed,
    createDesignWaypointOnMap,
    importDesignWaypoint,
    handleMoveSpot,
    handleStartRenameSpot,
    handleCancelSpotRename,
    handleCommitSpotRename,
    handleDeleteSpot,
    handleOpenWaypointBt,
    handleClearMapSelection,
    handleSelectBehaviorNode,
    handleToggleWaypointOptions,
    handleToggleSpotMode,
    handleCreateSpotAtRobot,
    handleCompleteDesignAtRobotPose,
  } = useDesignWaypointInteraction({
    design, workspace, designVisibleSpots, designDocumentReady, markDesignDirty, runSessionActive, runRouteView, runCommand, forgetMissionLocalBtSelection, waitForAutoLocalizedPose, mappingRuntimeActive, runRuntimeActive, missionRunnerActive, designLocalizationActive, invalidateNavigationStatus, clearLocalizationPoseCache, handleStopNavigation
  });

  return {
    selectedBtLayerSpot, selectedBtLayerDefaultPath, selectedBtLayerPaths, selectedBtLayerPath,
    missionBtLoadingPath, localBtFileActionsDisabled, loadMissionLocalBtXml, saveMissionLocalBtXml,
    selectMissionLocalBtXml, saveMissionLocalBtXmlAs, setMissionLocalBtDefault,
    handleMissionLocalBtXmlChange, handleDesignMissionChange, startNewMission,
    showSaveMissionDialog, saveMissionName, setSaveMissionName, handleSaveMission,
    closeSaveMissionDialog, handleConfirmSaveMission, showUnsavedDialog, unsavedSavePending,
    runGuardedDesignAction, resolveUnsavedDialog, cancelUnsavedDialog,
    cancelPendingDesignLocalization, handleSelectSpot, missionRouteMode, missionRouteSourceId,
    handleToggleMissionRouteMode, handleMissionRouteSpotClick, handleMissionRouteMapClick,
    handleClearMissionRoute, handleOpenMissionRouteLoop, handleMoveRouteSpot, handleRemoveRouteSpot,
    resetMissionRouteEditing, renderedRouteView, designPanelRouteSpots, designPanelRouteClosed,
    createDesignWaypointOnMap, importDesignWaypoint, handleMoveSpot, handleStartRenameSpot,
    handleCancelSpotRename, handleCommitSpotRename, handleDeleteSpot, handleOpenWaypointBt,
    handleClearMapSelection, handleSelectBehaviorNode, handleToggleWaypointOptions,
    handleToggleSpotMode, handleCreateSpotAtRobot, handleCompleteDesignAtRobotPose,
  };
}
