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

import { useCallback, useRef, useState } from "react";
import { initialNavigationRuntimeMode } from "../../mapWorkspace/lib/session";
import { DEFAULT_MAP_NAME, DEFAULT_MISSION_NAME } from "../../mapWorkspace/lib/missionNames";
import { savedBehaviorNodesForMap } from "../lib/designStore";
import useDesignMissionDocumentLedger from "./useDesignMissionDocumentLedger";
import useDesignMissionLoader from "./useDesignMissionLoader";

// Design owns document identity, edits and the ledger. Run keeps its separate read-only snapshot.
export default function useDesignWorkspaceState({
  initialSession, loadMissionBtFileOrDefault, fetchLegacySpotsForMap,
}) {
  const restoredDesignMapPath = (
    typeof initialSession.designMapPath === "string" && initialSession.designMapPath.trim()
      ? initialSession.designMapPath.trim()
      : ""
  );

  const cancelPendingDesignLocalizationRef = useRef(() => false);

  const behaviorMarkDirtyRef = useRef(() => {});

  const behaviorPrepareSelectRef = useRef(() => {});

  const catalogRunCommandRef = useRef(async () => {});

  const confirmDesignSelectionRef = useRef(async () => {});

  const deletedDesignMissionRef = useRef(async () => {});

  const [mapName, setMapName] = useState(() => (
    typeof initialSession.mapName === "string" && initialSession.mapName.trim()
      ? initialSession.mapName
      : DEFAULT_MAP_NAME
  ));

  const [missionName, setMissionName] = useState(() => (
    typeof initialSession.missionName === "string" && initialSession.missionName.trim()
      ? initialSession.missionName.trim()
      : DEFAULT_MISSION_NAME
  ));

  const designMapNameRef = useRef(mapName);

  const designMissionNameRef = useRef(missionName);

  designMapNameRef.current = mapName;

  designMissionNameRef.current = missionName;

  const [spots, setSpots] = useState([]);

  const [missionMetadata, setMissionMetadata] = useState({});

  const applyMissionMetadata = useCallback((metadata) => {
    setMissionMetadata(metadata);
  }, []);

  const [editingSpotId, setEditingSpotId] = useState("");

  const [editingSpotLabel, setEditingSpotLabel] = useState("");

  const [selectedSpotId, setSelectedSpotId] = useState("");

  const resetMissionRouteEditingRef = useRef(() => {});

  const clearMissionRouteSourceRef = useRef(() => {});

  const saveDesignMissionRef = useRef(null);

  const [missionFlowNodes, setMissionFlowNodes] = useState([]);

  const [missionFlowEdges, setMissionFlowEdges] = useState([]);

  const {
    missionBtFiles,
    deletedMissionBtPaths,
    recordBtEdit,
    replaceLiveBtFiles,
    replaceDeletedBtPaths,
    markNonBtDirty,
    clearDirty: clearDesignDirtyLedger,
    reconcileDirty: reconcileDesignDirty,
    commitSnapshot: commitDesignLedgerSnapshot,
    resetNewDocument: resetNewDesignLedger,
    getHistorySlice: getDesignLedgerHistorySlice,
    restoreHistorySlice: restoreDesignLedgerHistorySlice,
    checkpointPersistedBtFile,
    setPersistedRevision,
    beginSave: beginDesignSave,
    checkpointSaveUpload,
    checkpointSaveManifest,
    checkpointSaveCleanup,
    reconcileSave: reconcileDesignSave,
    abortSave: abortDesignSave,
    getLiveBtFiles,
    getPersistedRevision,
    getPersistedLocalBtPaths,
    hasPersistedLocalBtPath,
    isDirty: isDesignDirty,
    hasActiveSave: hasActiveDesignSave,
  } = useDesignMissionDocumentLedger();

  const {
    phase: designMissionLoadPhase,
    load: loadDesignMissionSnapshot,
    invalidate: invalidateDesignMission,
    captureLease: captureDesignMissionLease,
    isCurrent: isDesignMissionLeaseCurrent,
  } = useDesignMissionLoader({
    initialMapName: mapName,
    initialMissionName: missionName,
    loadBtFile: loadMissionBtFileOrDefault,
    loadLegacySpots: fetchLegacySpotsForMap,
    loadLegacyBehaviorNodes: savedBehaviorNodesForMap,
  });

  const [btLayerSpotId, setBtLayerSpotId] = useState("");

  const [showWaypointOptions, setShowWaypointOptions] = useState(false);

  const [designPoseInitialized, setDesignPoseInitialized] = useState(() => (
    initialNavigationRuntimeMode(initialSession) === "localization" &&
    initialSession.designPoseInitialized === true
  ));

  const [designMapPath, setDesignMapPath] = useState(restoredDesignMapPath);

  const designMapPathRef = useRef(designMapPath);

  designMapPathRef.current = designMapPath;

  const getDesignDocumentIdentity = useCallback(() => ({
    mapName: designMapNameRef.current,
    mapPath: designMapPathRef.current,
    missionName: designMissionNameRef.current,
  }), []);

  const setDesignDocumentIdentity = useCallback((identity = {}) => {
    if (typeof identity.mapName === "string") {
      designMapNameRef.current = identity.mapName;
      setMapName(identity.mapName);
    }
    if (typeof identity.mapPath === "string") {
      designMapPathRef.current = identity.mapPath;
      setDesignMapPath(identity.mapPath);
    }
    if (typeof identity.missionName === "string") {
      designMissionNameRef.current = identity.missionName;
      setMissionName(identity.missionName);
    }
  }, []);

  const [designMissionLoadError, setDesignMissionLoadError] = useState("");

  const [designMapReloadToken, setDesignMapReloadToken] = useState(0);

  return {
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
  };
}
