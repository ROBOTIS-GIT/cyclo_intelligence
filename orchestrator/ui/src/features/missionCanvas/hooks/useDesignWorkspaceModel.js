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

import useDesignDocumentHistory from "./useDesignDocumentHistory";
import useDesignWorkspaceState from "./useDesignWorkspaceState";
import { STAGE_AUTHORING } from "../../mapWorkspace/lib/stages";
import { DEFAULT_MAP_NAME } from "../../mapWorkspace/lib/missionNames";
import useDesignBehaviorNodeController from "./useDesignBehaviorNodeController";
import useDesignMissionCatalogController from "./useDesignMissionCatalogController";

export default function useDesignWorkspaceModel({
  workspace, runtimeMapName, loadMissionBtFileOrDefault, fetchLegacySpotsForMap
}) {
  const { initialSession, setMessage, setWorkspaceStage, setInteractionMode } = workspace;
  const {
    cancelPendingDesignLocalizationRef, behaviorMarkDirtyRef, behaviorPrepareSelectRef,
    catalogRunCommandRef, confirmDesignSelectionRef, deletedDesignMissionRef, mapName, setMapName,
    missionName, setMissionName, designMapNameRef, designMissionNameRef, spots, setSpots,
    missionMetadata, applyMissionMetadata, editingSpotId, setEditingSpotId, editingSpotLabel,
    setEditingSpotLabel, selectedSpotId, setSelectedSpotId, resetMissionRouteEditingRef,
    clearMissionRouteSourceRef, saveDesignMissionRef, missionFlowNodes, setMissionFlowNodes,
    missionFlowEdges, setMissionFlowEdges, missionBtFiles, deletedMissionBtPaths, recordBtEdit,
    replaceLiveBtFiles, replaceDeletedBtPaths, markNonBtDirty, clearDesignDirtyLedger,
    reconcileDesignDirty, commitDesignLedgerSnapshot, resetNewDesignLedger,
    getDesignLedgerHistorySlice, restoreDesignLedgerHistorySlice, checkpointPersistedBtFile,
    setPersistedRevision, beginDesignSave, checkpointSaveUpload, checkpointSaveManifest,
    checkpointSaveCleanup, reconcileDesignSave, abortDesignSave, getLiveBtFiles,
    getPersistedRevision, getPersistedLocalBtPaths, hasPersistedLocalBtPath, isDesignDirty,
    hasActiveDesignSave, designMissionLoadPhase, loadDesignMissionSnapshot, invalidateDesignMission,
    captureDesignMissionLease, isDesignMissionLeaseCurrent, btLayerSpotId, setBtLayerSpotId,
    showWaypointOptions, setShowWaypointOptions, designPoseInitialized, setDesignPoseInitialized,
    designMapPath, getDesignDocumentIdentity, setDesignDocumentIdentity, designMissionLoadError,
    setDesignMissionLoadError, designMapReloadToken, setDesignMapReloadToken,
  } = useDesignWorkspaceState({
    initialSession, loadMissionBtFileOrDefault, fetchLegacySpotsForMap,
  });

  const {
    catalog: designCatalog,
    listMissionNames: fetchMissionNames,
    refreshCatalog: refreshDesignCatalog,
    recordSavedMission,
    beginDocumentRequest: beginDesignDocumentRequest,
    finishDocumentRequest: finishDesignDocumentRequest,
    invalidateRequests: invalidateDesignDocumentRequests,
    picker: designMissionPicker,
    rename: designMissionRename,
    duplicate: designMissionDuplicate,
    deletion: designMissionDeletion,
  } = useDesignMissionCatalogController({
    currentMapName: String(mapName || "").trim() || DEFAULT_MAP_NAME,
    currentMapPath: designMapPath,
    currentMissionName: missionName,
    getPersistedRevision,
    setPersistedRevision,
    invalidateDocument: invalidateDesignMission,
    runCommand: (...args) => catalogRunCommandRef.current(...args),
    onMessage: setMessage,
    onPrepareOpen: () => {
      setWorkspaceStage(STAGE_AUTHORING);
      setShowWaypointOptions(false);
    },
    onConfirmSelection: (...args) => confirmDesignSelectionRef.current(...args),
    onRenamed: (target) => {
      setMissionName(target);
    },
    onDeleted: (...args) => deletedDesignMissionRef.current(...args),
  });

  const {
    open: showDesignMapDialog,
    files: designMapFiles,
    missionNames: designMissionNames,
    pendingMapPath: pendingDesignMapPath,
    pendingMissionName: pendingDesignMissionName,
    busy: designMapBusy,
    catalogReady: designDialogCatalogReady,
    openPicker: handleOpenDesignMapDialog,
    changePendingMap: handleDesignMapChange,
    setPendingMissionName: setPendingDesignMissionName,
    cancelPicker: cancelDesignMapDialog,
    confirmSelection: handleConfirmDesignMap,
  } = designMissionPicker;

  const {
    open: showRenameMissionDialog, name: renameMissionName,
    setName: setRenameMissionName, openDialog: handleOpenRenameMissionDialog,
    close: closeRenameMissionDialog, confirm: handleConfirmRenameMission,
  } = designMissionRename;

  const {
    open: showDuplicateMissionDialog, name: duplicateMissionName,
    setName: setDuplicateMissionName, openDialog: handleOpenDuplicateMissionDialog,
    close: closeDuplicateMissionDialog, confirm: handleConfirmDuplicateMission,
  } = designMissionDuplicate;

  const {
    open: showDeleteMissionDialog, openDialog: openDeleteMissionDialog,
    close: closeDeleteMissionDialog, confirm: handleConfirmDeleteMission,
  } = designMissionDeletion;

  const designMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;

  const {
    designNodes: designBehaviorNodes,
    runNodes: runBehaviorNodes,
    selectedNodeId: selectedBehaviorNodeId,
    pendingTag: pendingBehaviorNodeTag,
    previewNode: behaviorPreviewNode,
    mergeMapPatch: mergeBehaviorMapPatch,
    getHistorySlice: getBehaviorHistorySlice,
    restoreHistorySlice: restoreBehaviorHistorySlice,
    clearSelection: clearBehaviorSelection,
    clearPendingPlacement: clearPendingBehaviorPlacement,
    selectNode: selectBehaviorNode,
    placePendingAtPose: placePendingBehaviorAtPose,
    moveNode: handleMoveBehaviorNode,
    deleteNode: handleDeleteBehaviorNode,
  } = useDesignBehaviorNodeController({
    designMapName,
    runtimeMapName,
    markDirty: () => behaviorMarkDirtyRef.current(),
    onMessage: setMessage,
    captureDocumentLease: captureDesignMissionLease,
    isDocumentLeaseCurrent: isDesignMissionLeaseCurrent,
    getCurrentIdentity: () => ({
      mapName: designMapNameRef.current,
      missionName: designMissionNameRef.current,
    }),
    onPrepareSelect: (nodeId) => behaviorPrepareSelectRef.current(nodeId),
    onPlaced: () => {
      setSelectedSpotId("");
      setInteractionMode("view");
    },
  });

  const {
    getDesignHistorySnapshot, captureDesignHistory, undoDesignHistory, redoDesignHistory,
    resetDesignHistory, rebaseDesignHistory, canUndoDesign, canRedoDesign,
  } = useDesignDocumentHistory({
    spots, getBehaviorHistorySlice, missionMetadata, missionFlowNodes, missionFlowEdges,
    selectedSpotId, getDesignLedgerHistorySlice, setSpots, applyMissionMetadata,
    restoreBehaviorHistorySlice, restoreDesignLedgerHistorySlice, setMissionFlowNodes,
    setMissionFlowEdges, setSelectedSpotId, clearMissionRouteSourceRef, setEditingSpotId,
    setEditingSpotLabel, setBtLayerSpotId, setInteractionMode, setShowWaypointOptions, setMessage,
    behaviorMarkDirtyRef, markNonBtDirty,
  });

  return {
    cancelPendingDesignLocalizationRef, behaviorPrepareSelectRef, catalogRunCommandRef,
    confirmDesignSelectionRef, deletedDesignMissionRef, mapName, setMapName, missionName,
    designMapNameRef, designMissionNameRef, spots, setSpots, missionMetadata, applyMissionMetadata,
    editingSpotId, setEditingSpotId, editingSpotLabel, setEditingSpotLabel, selectedSpotId,
    setSelectedSpotId, resetMissionRouteEditingRef, clearMissionRouteSourceRef,
    saveDesignMissionRef, missionFlowNodes, setMissionFlowNodes, missionFlowEdges,
    setMissionFlowEdges, missionBtFiles, deletedMissionBtPaths, recordBtEdit, replaceLiveBtFiles,
    replaceDeletedBtPaths, markNonBtDirty, clearDesignDirtyLedger, reconcileDesignDirty,
    commitDesignLedgerSnapshot, resetNewDesignLedger, checkpointPersistedBtFile,
    setPersistedRevision, beginDesignSave, checkpointSaveUpload, checkpointSaveManifest,
    checkpointSaveCleanup, reconcileDesignSave, abortDesignSave, getLiveBtFiles,
    getPersistedRevision, getPersistedLocalBtPaths, hasPersistedLocalBtPath, isDesignDirty,
    hasActiveDesignSave, designMissionLoadPhase, loadDesignMissionSnapshot, invalidateDesignMission,
    captureDesignMissionLease, isDesignMissionLeaseCurrent, btLayerSpotId, setBtLayerSpotId,
    showWaypointOptions, setShowWaypointOptions, designPoseInitialized, setDesignPoseInitialized,
    designMapPath, getDesignDocumentIdentity, setDesignDocumentIdentity, designMissionLoadError,
    setDesignMissionLoadError, designMapReloadToken, setDesignMapReloadToken, designCatalog,
    fetchMissionNames, refreshDesignCatalog, recordSavedMission, beginDesignDocumentRequest,
    finishDesignDocumentRequest, invalidateDesignDocumentRequests, showDesignMapDialog,
    designMapFiles, designMissionNames, pendingDesignMapPath, pendingDesignMissionName,
    designMapBusy, designDialogCatalogReady, handleOpenDesignMapDialog, handleDesignMapChange,
    setPendingDesignMissionName, cancelDesignMapDialog, handleConfirmDesignMap,
    showRenameMissionDialog, renameMissionName, setRenameMissionName, handleOpenRenameMissionDialog,
    closeRenameMissionDialog, handleConfirmRenameMission, showDuplicateMissionDialog,
    duplicateMissionName, setDuplicateMissionName, handleOpenDuplicateMissionDialog,
    closeDuplicateMissionDialog, handleConfirmDuplicateMission, showDeleteMissionDialog,
    openDeleteMissionDialog, closeDeleteMissionDialog, handleConfirmDeleteMission, designMapName,
    designBehaviorNodes, runBehaviorNodes, selectedBehaviorNodeId, pendingBehaviorNodeTag,
    behaviorPreviewNode, mergeBehaviorMapPatch, clearBehaviorSelection,
    clearPendingBehaviorPlacement, selectBehaviorNode, placePendingBehaviorAtPose,
    handleMoveBehaviorNode, handleDeleteBehaviorNode, getDesignHistorySnapshot,
    captureDesignHistory, undoDesignHistory, redoDesignHistory, resetDesignHistory,
    rebaseDesignHistory, canUndoDesign, canRedoDesign,
  };
}
