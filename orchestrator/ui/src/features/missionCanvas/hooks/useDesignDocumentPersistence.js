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

import { useCallback, useEffect } from "react";
import { STAGE_AUTHORING } from "../../mapWorkspace/lib/stages";
import { saveMissionSession } from "../../mapWorkspace/lib/session";
import { buildGlobalMissionXml } from "../../mapWorkspace/lib/missionBtFiles";
import useUnsavedDesignActionGuard from "./useUnsavedDesignActionGuard";
import useDesignMissionDocumentLifecycleController from "./useDesignMissionDocumentLifecycleController";
import useDesignMissionSaveController from "./useDesignMissionSaveController";
import useDesignMissionSaveDialog from "./useDesignMissionSaveDialog";

export default function useDesignDocumentPersistence({
  workspace, design, commitLoadedDesign, resetNewDesignDocument, designVisibleSpots, applySavedCanonicalSpots, runCommand, isWaypointTaskFileBusy
}) {
  const { setMissionMapLoaded, setWorkspaceStage, setInteractionMode, navigationRuntimeMode, setMessage } = workspace;
  const {
    clearBehaviorSelection, clearPendingBehaviorPlacement, resetMissionRouteEditingRef,
    setBtLayerSpotId, setDesignMapReloadToken, setSelectedSpotId, getDesignDocumentIdentity,
    setDesignDocumentIdentity, loadDesignMissionSnapshot, isDesignMissionLeaseCurrent,
    invalidateDesignMission, beginDesignDocumentRequest, finishDesignDocumentRequest,
    invalidateDesignDocumentRequests, designCatalog, setPendingDesignMissionName,
    clearDesignDirtyLedger, setDesignMissionLoadError, confirmDesignSelectionRef,
    deletedDesignMissionRef, missionMetadata, designBehaviorNodes, missionFlowNodes, missionFlowEdges,
    getDesignHistorySnapshot, designMissionLoadError, beginDesignSave, checkpointSaveUpload,
    checkpointSaveManifest, checkpointSaveCleanup, reconcileDesignSave, abortDesignSave,
    setPersistedRevision, resetDesignHistory, rebaseDesignHistory, captureDesignMissionLease,
    recordSavedMission, refreshDesignCatalog, hasActiveDesignSave, saveDesignMissionRef, missionName,
    captureDesignHistory, markNonBtDirty, isDesignDirty, designMapName,
  } = design;
  const prepareDesignDocumentChange = useCallback((kind, identity) => {
    if (kind === "picker") {
      setMissionMapLoaded(true);
      setWorkspaceStage(STAGE_AUTHORING);
      setInteractionMode("view");
      clearBehaviorSelection();
      clearPendingBehaviorPlacement();
      resetMissionRouteEditingRef.current();
      setBtLayerSpotId("");
      setDesignMapReloadToken((value) => value + 1);
      saveMissionSession({
        mapName: identity.mapName,
        workspaceStage: STAGE_AUTHORING,
        designMapPath: identity.mapPath,
        navigationRuntimeMode,
      });
      return;
    }
    if (kind === "switch") {
      setSelectedSpotId("");
      clearBehaviorSelection();
      clearPendingBehaviorPlacement();
      setBtLayerSpotId("");
      resetMissionRouteEditingRef.current();
    }
  }, [
    clearBehaviorSelection, clearPendingBehaviorPlacement, navigationRuntimeMode,
    resetMissionRouteEditingRef, setBtLayerSpotId, setDesignMapReloadToken, setInteractionMode,
    setMissionMapLoaded, setSelectedSpotId, setWorkspaceStage,
  ]);

  const {
    confirmPickerSelection: confirmDesignDocumentSelection,
    switchMission: handleDesignMissionChange,
    newDocument: newDesignDocument,
    continueAfterDelete: continueAfterDeletedDesignMission,
  } = useDesignMissionDocumentLifecycleController({
    getIdentity: getDesignDocumentIdentity,
    setIdentity: setDesignDocumentIdentity,
    loader: {
      load: loadDesignMissionSnapshot,
      isCurrent: isDesignMissionLeaseCurrent,
      invalidate: invalidateDesignMission,
    },
    requests: {
      begin: beginDesignDocumentRequest,
      finish: finishDesignDocumentRequest,
      invalidate: invalidateDesignDocumentRequests,
    },
    content: {
      commitLoadedSnapshot: commitLoadedDesign,
      resetNewDocument: resetNewDesignDocument,
    },
    getCatalogNames: () => designCatalog.names,
    setPendingMissionName: setPendingDesignMissionName,
    clearDirty: clearDesignDirtyLedger,
    setLoadError: setDesignMissionLoadError,
    onPrepareChange: prepareDesignDocumentChange,
    onMessage: setMessage,
  });

  confirmDesignSelectionRef.current = confirmDesignDocumentSelection;

  const startNewMission = useCallback(() => newDesignDocument({
    btFiles: { "global.xml": buildGlobalMissionXml([]) },
  }), [newDesignDocument]);

  deletedDesignMissionRef.current = ({ remainingNames }) => {
    // The deleted mission's unsaved edits are moot — no guard on the switch.
    void continueAfterDeletedDesignMission({
      remainingNames,
      newDocument: { btFiles: { "global.xml": buildGlobalMissionXml([]) } },
    });
  };

  const { saveMission: saveDesignMission } = useDesignMissionSaveController({
    getSnapshot: () => ({
      identity: getDesignDocumentIdentity(),
      catalog: {
        mapName: designCatalog.mapName,
        names: designCatalog.names,
      },
      content: {
        metadata: missionMetadata,
        visibleSpots: designVisibleSpots,
        behaviorNodes: designBehaviorNodes,
        missionFlowNodes,
        missionFlowEdges,
      },
      historyAtStart: getDesignHistorySnapshot(),
      loadError: designMissionLoadError,
    }),
    getIdentity: getDesignDocumentIdentity,
    ledger: {
      beginSave: beginDesignSave,
      checkpointSaveUpload,
      checkpointSaveManifest,
      checkpointSaveCleanup,
      reconcileSave: reconcileDesignSave,
      abortSave: abortDesignSave,
      setPersistedRevision,
    },
    content: {
      applySavedCanonicalSpots,
    },
    history: { reset: resetDesignHistory, rebase: rebaseDesignHistory },
    loader: {
      captureLease: captureDesignMissionLease,
      isCurrent: isDesignMissionLeaseCurrent,
      invalidate: invalidateDesignMission,
    },
    catalog: {
      record: recordSavedMission,
      refresh: refreshDesignCatalog,
    },
    setIdentity: setDesignDocumentIdentity,
    runCommand,
    onMessage: setMessage,
    isWaypointFileBusy: isWaypointTaskFileBusy,
    hasActiveSave: hasActiveDesignSave,
  });

  useEffect(() => {
    saveDesignMissionRef.current = saveDesignMission;
  }, [saveDesignMission, saveDesignMissionRef]);

  const {
    open: showSaveMissionDialog,
    name: saveMissionName,
    setName: setSaveMissionName,
    requestSave: handleSaveMission,
    cancel: closeSaveMissionDialog,
    confirm: handleConfirmSaveMission,
  } = useDesignMissionSaveDialog({
    missionName,
    existingNames: designCatalog.names,
    saveMission: saveDesignMission,
  });

  const markDesignDirty = useCallback(() => {
    captureDesignHistory();
    markNonBtDirty();
  }, [captureDesignHistory, markNonBtDirty]);

  const clearDesignDirty = useCallback(() => {
    clearDesignDirtyLedger();
  }, [clearDesignDirtyLedger]);

  const {
    open: showUnsavedDialog,
    saving: unsavedSavePending,
    runGuardedAction: runGuardedDesignAction,
    resolve: resolveUnsavedDialog,
    cancel: cancelUnsavedDialog,
  } = useUnsavedDesignActionGuard({
    isDirty: isDesignDirty,
    clearDirty: clearDesignDirty,
    save: () => saveDesignMission(missionName),
    documentKey: `${designMapName}\u0000${missionName}`,
  });

  return {
    handleDesignMissionChange, startNewMission, showSaveMissionDialog, saveMissionName,
    setSaveMissionName, handleSaveMission, closeSaveMissionDialog, handleConfirmSaveMission,
    markDesignDirty, showUnsavedDialog, unsavedSavePending, runGuardedDesignAction,
    resolveUnsavedDialog, cancelUnsavedDialog,
  };
}
