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

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useMapEditor } from "../../components/navigation/MapEditor";
import { RunnerStatus } from "../../hooks/missionRunnerCore";
import { useBTHistory } from "../../hooks/useBTHistory";
import { useMissionRunner } from "../../hooks/useMissionRunner";
import { useNavigationRosPublisher } from "../../hooks/useNavigationRosTopic";
import { useRosServiceCaller } from "../../hooks/useRosServiceCaller";
import {
  cancelNavigateToPoseGoal,
  configureDesignLocalizationAmcl,
  deletePgmMap,
  getPgmFiles,
  getServiceStatus,
  requestNoMotionUpdate,
  saveNavigationMap,
  sendInitialPoseEstimate,
  sendNavigateThroughPosesGoalsAndWait,
  sendNavigateToPoseGoalAndWait,
  startNavigation,
  stopNavigation,
} from "../../utils/navigationApi";
import { getNavigationMissionBtFile } from "../../utils/navigationMissionsApi";
import { getNavigationSpots } from "../../utils/navigationSpotsApi";
import {
  orientationFromYaw,
  updateTfBuffer,
  yawFromPose,
} from "../../utils/navigationTf";
import { rosTimestampNow } from "../../utils/rosTime";
import { formatTaskDisplayMessage } from "../../utils/taskTerminology";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_NAVIGATE,
  STAGE_RUN,
} from "./lib/stages";
import {
  initialNavigationRuntimeMode,
  initialRunRuntimeOwned,
  initialRunShutdownPending,
  initialWorkspaceStage,
  readMissionSession,
  recentRunShutdownMarker,
  saveMissionSession,
} from "./lib/session";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "./lib/missionNames";
import {
  buildGlobalMissionXml,
  localBtPathForSpot,
} from "./lib/missionBtFiles";
import {
  filterMissionFlowEdges,
  syncMissionFlowNodesWithSpots,
} from "./lib/missionFlow";
import useWaypointTaskFileController from "./hooks/useWaypointTaskFileController";
import useDesignWaypointController from "./hooks/useDesignWaypointController";
import useDesignBehaviorNodeController from "./hooks/useDesignBehaviorNodeController";
import {
  savedBehaviorNodesForMap,
} from "./lib/designStore";
import { LAYER_PRESETS } from "./lib/layers";
import useMissionBtNodeLease from "./hooks/useMissionBtNodeLease";
import useDesignMissionDocumentLedger from "./hooks/useDesignMissionDocumentLedger";
import useDesignMissionLoader from "./hooks/useDesignMissionLoader";
import useMissionRouteEditor from "./hooks/useMissionRouteEditor";
import useMissionRosScene, {
  deriveMissionRosSceneGates,
} from "./hooks/useMissionRosScene";
import useMissionWorkspaceExitGuard from "./hooks/useMissionWorkspaceExitGuard";
import useRunMissionSnapshot from "./hooks/useRunMissionSnapshot";
import useNavigateGoalController from "./hooks/useNavigateGoalController";
import useDesignMissionCatalogController from "./hooks/useDesignMissionCatalogController";
import useUnsavedDesignActionGuard from "./hooks/useUnsavedDesignActionGuard";
import useDesignDocumentContentController from "./hooks/useDesignDocumentContentController";
import useDesignMissionDocumentLifecycleController from "./hooks/useDesignMissionDocumentLifecycleController";
import useDesignMissionSaveController from "./hooks/useDesignMissionSaveController";
import useMappingLifecycleController from "./hooks/useMappingLifecycleController";
import useRunMapMissionSelectionController from "./hooks/useRunMapMissionSelectionController";
import useRunNavigationLocalizationController from "./hooks/useRunNavigationLocalizationController";
import useNavigationRuntimeSessionController from "./hooks/useNavigationRuntimeSessionController";
import useMapEditStageCoordinator from "./hooks/useMapEditStageCoordinator";
import useDesignAtRobotWaypointController from "./hooks/useDesignAtRobotWaypointController";
import useMissionBtLayerComposition from "./hooks/useMissionBtLayerComposition";
import useNavigationStopCoordinator from "./hooks/useNavigationStopCoordinator";
import useDesignMissionSaveDialog from "./hooks/useDesignMissionSaveDialog";
import useDesignHistoryInteractionController from "./hooks/useDesignHistoryInteractionController";
import useMissionMapScenePresentation, {
  deriveMissionRoutePresentation,
} from "./hooks/useMissionMapScenePresentation";
import StageRail from "./components/workspace/StageRail";
import MissionCanvasDialogLayer from "./components/workspace/MissionCanvasDialogLayer";
import MissionCanvasSceneSurface from "./components/workspace/MissionCanvasSceneSurface";
import MissionStageChrome from "./components/workspace/MissionStageChrome";
import { TELEOP_MESSAGE_TYPE, TELEOP_TOPIC, teleopTwist } from "./components/mapping/MappingTeleopPanel";

const STATUS_POLL_MS = 10000;

const NOMOTION_UPDATE_INTERVAL_MS = 1000;

const AUTO_LOCALIZE_MAX_UPDATES = 10;

const AUTO_LOCALIZE_MIN_UPDATES = 3;

const AUTO_LOCALIZE_UPDATE_DELAY_MS = 700;

const AUTO_LOCALIZE_XY_COVARIANCE_MAX = 0.6;

const AUTO_LOCALIZE_YAW_COVARIANCE_MAX = 0.5;

function taskDisplayMessage(value) {
  return formatTaskDisplayMessage(value, "Waypoint Task");
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function amclPoseLooksLocalized(amclPose) {
  const covariance = amclPose?.pose?.covariance;
  if (!Array.isArray(covariance) || covariance.length < 36) return false;
  const xVariance = Number(covariance[0]);
  const yVariance = Number(covariance[7]);
  const yawVariance = Number(covariance[35]);
  if (![xVariance, yVariance, yawVariance].every(Number.isFinite)) return false;
  return (
    Math.max(xVariance, yVariance) <= AUTO_LOCALIZE_XY_COVARIANCE_MAX &&
    yawVariance <= AUTO_LOCALIZE_YAW_COVARIANCE_MAX
  );
}

function navigationRuntimeModeFromStatus(status) {
  if (status?.is_up === false) return "idle";
  if (!status?.is_up) return "";
  if (status.mode === "map" || status.mode === "mapping") return "mapping";
  if (status.mode === "localize" || status.mode === "localization") return "localization";
  if (status.mode === "nav" || status.mode === "run") return "run";
  return "";
}

async function fetchLegacySpotsForMap(targetMapName) {
  const normalizedMapName = String(targetMapName || "").trim() || DEFAULT_MAP_NAME;
  const result = await getNavigationSpots(normalizedMapName);
  return result.spots || [];
}

async function loadMissionBtFileOrDefault(
  targetMapName,
  targetMissionName,
  path,
  fallback,
  expectedRevision,
) {
  const response = await getNavigationMissionBtFile(
    targetMapName,
    path,
    missionRequestName(targetMissionName),
  );
  if (
    Number.isInteger(expectedRevision)
    && Number.isInteger(response?.revision)
    && response.revision !== expectedRevision
  ) {
    throw new Error(
      `Mission changed while loading ${path}; reload the mission before editing or running it`,
    );
  }
  if (response?.exists && typeof response.content === "string") {
    return response.content;
  }
  return fallback;
}

// Mounted by the Autonomy Studio shell, which owns the page root, the app bar
// and the workspace choice. `dialogHost` is the shell's page root (dialogs must
// portal inside the --mc-* token scope), `onExitStateChange` publishes the
// reason the back button is blocked, and `exitHandleRef` receives
// `requestExit(onExit)` so leaving passes through the unsaved-Design guard.
export default function MissionCanvasWorkspace({
  dialogHost = null,
  onExitStateChange = null,
  exitHandleRef = null,
}) {
  const initialSessionRef = useRef(null);
  if (initialSessionRef.current === null) {
    initialSessionRef.current = readMissionSession();
  }
  const initialSession = initialSessionRef.current;
  const restoredDesignMapPath = (
    typeof initialSession.designMapPath === "string" && initialSession.designMapPath.trim()
      ? initialSession.designMapPath.trim()
      : ""
  );
  const invalidateRunPoseRequestRef = useRef(() => {});
  const cancelPendingDesignLocalizationRef = useRef(() => false);
  const nomotionUpdateBusyRef = useRef(false);
  const tfBufferRef = useRef(new Map());
  const currentPoseRef = useRef(null);
  const amclPoseRef = useRef(null);
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
  const [status, setStatus] = useState(null);
  const [spots, setSpots] = useState([]);
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
  // Run is deliberately a separate, read-only mission session. Loading a
  // mission to execute must never replace the mission currently being edited.
  const initialRunMissionName = (
    typeof initialSession.runMissionName === "string" && initialSession.runMissionName.trim()
      ? initialSession.runMissionName.trim()
      : DEFAULT_MISSION_NAME
  );
  const loadLegacyRunSpotsForMap = useCallback(async (targetMapName) => {
    return fetchLegacySpotsForMap(targetMapName);
  }, []);
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
  const {
    snapshot: runMissionSnapshot,
    load: loadRunMissionForMap,
    clear: clearRunMissionSnapshot,
    invalidate: invalidateRunMissionLoad,
    cancelAndClear: cancelAndClearRunMissionSnapshot,
  } = useRunMissionSnapshot({
    initialMissionName: initialRunMissionName,
    loadLegacySpotsForMap: loadLegacyRunSpotsForMap,
    loadMissionBtFileOrDefault,
  });
  const {
    mapName: runMapName,
    missionName: runMissionName,
    catalog: runCatalog,
    spots: runSpots,
    btFiles: runMissionBtFiles,
    flowNodes: runMissionFlowNodes,
    flowEdges: runMissionFlowEdges,
    invalid: runMapSnapshotInvalid,
  } = runMissionSnapshot;
  const [busy, setBusy] = useState("");
  // Status strings are no longer surfaced anywhere (the header status line
  // was removed on purpose); the state stays because ~70 flows and child
  // panels still report through setMessage, keeping the wiring in place if a
  // notification surface ever returns.
  const [message, setMessage] = useState("");
  void message;
  const [btLayerSpotId, setBtLayerSpotId] = useState("");
  // The read-only Run BT view only needs a stable map context. Suspending
  // high-frequency visual overlays while it is open avoids redrawing the
  // WebGL map underneath every ReactFlow update; Nav2 itself is unaffected.
  const [runBtVisualizationActive, setRunBtVisualizationActive] = useState(false);
  const [interactionMode, setInteractionMode] = useState("view");
  // Run stage: AMCL must be given an initial pose after nav bringup before the
  // mission runner may send goals — a lost robot stays still otherwise.
  const [runPoseInitialized, setRunPoseInitialized] = useState(false);
  const [showWaypointOptions, setShowWaypointOptions] = useState(false);
  const [designPoseInitialized, setDesignPoseInitialized] = useState(() => (
    initialNavigationRuntimeMode(initialSession) === "localization" &&
    initialSession.designPoseInitialized === true
  ));
  const [navigationRuntimeMode, setNavigationRuntimeMode] = useState(() => (
    initialNavigationRuntimeMode(initialSession)
  ));
  const [runRuntimeOwned, setRunRuntimeOwned] = useState(() => (
    initialRunRuntimeOwned(initialSession)
  ));
  const [runShutdownPending, setRunShutdownPending] = useState(() => (
    initialRunShutdownPending(initialSession)
  ));
  const [tfBufferRevision, setTfBufferRevision] = useState(0);
  const [workspaceStage, setWorkspaceStage] = useState(() => initialWorkspaceStage(initialSession));
  // Run and Navigate share one runtime. Async controllers read the current
  // stage from this ref so their requests capture the correct owner.
  const workspaceStageRef = useRef(workspaceStage);
  workspaceStageRef.current = workspaceStage;
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
  // Whether a map is loaded for display in Design/Run. Deliberately NOT restored
  // from session: a refresh, a stage switch, or the backend going down should
  // drop the stale map so it never lingers after the system is off.
  const [missionMapLoaded, setMissionMapLoaded] = useState(false);
  const prevRunningRef = useRef(false);
  const [designMissionLoadError, setDesignMissionLoadError] = useState("");
  const [designMapReloadToken, setDesignMapReloadToken] = useState(0);
  // Map Edit HUD tool-group popovers (the Design HUD's waypoint-options idiom).
  const [mapEditToolsOpen, setMapEditToolsOpen] = useState(false);
  const [labelToolsOpen, setLabelToolsOpen] = useState(false);
  // A fresh Run tab has no authoritative map/mission snapshot yet. Localize
  // becomes available only after the first successful Run map load.
  const [mapEditorReloadToken, setMapEditorReloadToken] = useState(0);
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
  const [layersByStage, setLayersByStage] = useState(() => ({
    [STAGE_MAPPING]: { ...LAYER_PRESETS[STAGE_MAPPING] },
    [STAGE_MAP_EDIT]: { ...LAYER_PRESETS[STAGE_MAP_EDIT] },
    [STAGE_NAVIGATE]: { ...LAYER_PRESETS[STAGE_NAVIGATE] },
    [STAGE_AUTHORING]: { ...LAYER_PRESETS[STAGE_AUTHORING] },
    [STAGE_RUN]: { ...LAYER_PRESETS[STAGE_RUN] },
  }));
  const publishRosTopic = useNavigationRosPublisher();


  const running = status?.is_up ?? false;
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
  const designMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
  const runtimeMapName = String(runMapName || "").trim() || DEFAULT_MAP_NAME;
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

  const getDesignHistorySnapshot = useCallback(() => JSON.stringify({
    spots,
    ...getBehaviorHistorySlice(),
    missionFlowNodes,
    missionFlowEdges,
    selectedSpotId,
    ...getDesignLedgerHistorySlice(),
  }), [
    getBehaviorHistorySlice,
    getDesignLedgerHistorySlice,
    missionFlowEdges,
    missionFlowNodes,
    selectedSpotId,
    spots,
  ]);

  const applyDesignHistorySnapshot = useCallback((snapshot) => {
    try {
      const restored = JSON.parse(snapshot);
      const restoredSpots = Array.isArray(restored.spots) ? restored.spots : [];
      setSpots(restoredSpots);
      restoreBehaviorHistorySlice(restored);
      restoreDesignLedgerHistorySlice(restored);
      setMissionFlowNodes(
        Array.isArray(restored.missionFlowNodes) ? restored.missionFlowNodes : [],
      );
      setMissionFlowEdges(
        Array.isArray(restored.missionFlowEdges) ? restored.missionFlowEdges : [],
      );
      setSelectedSpotId(
        restoredSpots.some((spot) => spot.id === restored.selectedSpotId)
          ? restored.selectedSpotId
          : "",
      );
      clearMissionRouteSourceRef.current();
      setEditingSpotId("");
      setEditingSpotLabel("");
      setBtLayerSpotId("");
      setInteractionMode("view");
      setShowWaypointOptions(false);
      setMessage("Design history restored");
    } catch {
      setMessage("Failed to restore design history");
    }
  }, [restoreBehaviorHistorySlice, restoreDesignLedgerHistorySlice]);

  const {
    capture: captureDesignHistory,
    undo: undoDesignHistory,
    redo: redoDesignHistory,
    reset: resetDesignHistory,
    rebase: rebaseDesignHistory,
    canUndo: canUndoDesign,
    canRedo: canRedoDesign,
  } = useBTHistory({
    getSnapshot: getDesignHistorySnapshot,
    applySnapshot: applyDesignHistorySnapshot,
  });
  behaviorMarkDirtyRef.current = () => {
    captureDesignHistory();
    markNonBtDirty();
  };

  // The three editors stay workspace-owned because their open gates feed from
  // sceneGates while their loaded maps feed back into the rendered ROS scene.
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
  }, [invalidateDesignMission, missionName]);
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

  const { callService } = useRosServiceCaller();
  const {
    btNodeBusy,
    btNodeIsUp,
    btStatusRef,
    ensureMissionBtActive,
    releaseMissionBt,
    setBtStatusText,
  } = useMissionBtNodeLease({
    callService,
    needsBtTopics: workspaceStage === STAGE_RUN,
    onMessage: setMessage,
  });
  const {
    amclPose,
    btActiveNodesText,
    btStatusText,
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
  // ── Mission runner (Run stage): navigate → run waypoint BT → advance ──────
  const resolveMissionBtXml = useCallback(
    (spot) => (spot ? runMissionBtFiles[localBtPathForSpot(spot)] : ""),
    [runMissionBtFiles],
  );
  const sendMissionGoal = useCallback(async (x, y, yaw, signal) => {
    const poseStamped = {
      header: { frame_id: "map", stamp: rosTimestampNow() },
      pose: { position: { x, y, z: 0 }, orientation: orientationFromYaw(yaw) },
    };
    return sendNavigateToPoseGoalAndWait({ pose: poseStamped }, signal);
  }, []);
  const sendMissionGoals = useCallback(async (goals, signal) => {
    const poses = goals.map(({ x, y, yaw }) => ({
      header: { frame_id: "map", stamp: rosTimestampNow() },
      pose: { position: { x, y, z: 0 }, orientation: orientationFromYaw(yaw) },
    }));
    return sendNavigateThroughPosesGoalsAndWait({ poses }, signal);
  }, []);

  // ── Navigate stage: direct point-to-point goal (no mission) ───────────────
  const {
    goalPose: navGoalPose,
    goalStatus: navGoalStatus,
    driving: navGoalDriving,
    sendGoal: handleSendNavGoal,
    invalidateGoal: invalidateNavGoal,
  } = useNavigateGoalController({
    sendGoalAndWait: sendNavigateToPoseGoalAndWait,
    onDisarm: () => setInteractionMode("view"),
    onMessage: setMessage,
  });
  const stopMissionBt = useCallback(async () => {
    const result = await callService(
      "/bt/set_running",
      "std_srvs/srv/SetBool",
      { data: false },
    );
    if (result?.success === false) {
      throw new Error(taskDisplayMessage(result.message) || "Task Engine stop rejected");
    }
    return result;
  }, [callService]);
  const missionRunnerFlags = useCallback(
    () => ({ navRunning: running, btNodeIsUp }),
    [btNodeIsUp, running],
  );
  const missionRunner = useMissionRunner({
    // A closed route intentionally contains the start waypoint a second time
    // as its final execution step. Keep the unique list for badges/editing,
    // but give the runner the full traversal so last -> start is sent to Nav2.
    orderedSpots: runRouteView.executionSpots,
    resolveBtXml: resolveMissionBtXml,
    btStatusRef,
    callService,
    sendGoal: sendMissionGoal,
    sendGoals: sendMissionGoals,
    cancelGoal: cancelNavigateToPoseGoal,
    stopBt: stopMissionBt,
    ensureBtActive: ensureMissionBtActive,
    releaseBt: releaseMissionBt,
    getFlags: missionRunnerFlags,
    onMessage: setMessage,
  });
  const missionRunnerActive = missionRunner.isRunning;
  const missionRunnerStopping = (
    missionRunnerActive && missionRunner.status === RunnerStatus.CANCELLED
  );
  const missionFollowRobot = (
    missionRunnerActive
    && (missionRunner.phase === "nav-sent" || missionRunner.phase === "awaiting-nav-result")
  ) || (workspaceStage === STAGE_NAVIGATE && navGoalStatus === "driving");

  const {
    waypointBtLayer,
    runBtLayer,
    activeBtLayer,
    runBtViewActive,
  } = useMissionBtLayerComposition({
    workspaceStage,
    selectedBtLayerSpot,
    selectedBtLayerPath,
    selectedBtLayerPaths,
    selectedBtLayerDefaultPath,
    missionBtFiles,
    missionBtLoadingPath,
    loadMissionLocalBtXml,
    saveMissionLocalBtXml,
    selectMissionLocalBtXml,
    saveMissionLocalBtXmlAs,
    setMissionLocalBtDefault,
    localBtFileActionsDisabled,
    handleMissionLocalBtXmlChange,
    missionRunner,
    runVisibleSpots,
    runMissionBtFiles,
    btActiveNodesText,
  });

  useEffect(() => {
    setRunBtVisualizationActive(runBtViewActive);
  }, [runBtViewActive]);

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
  }, [designDocumentReady, workspaceStage]);

  const teleopDisabled = !!busy || mappingEditorActive;

  const navigationRuntimeSession = useNavigationRuntimeSessionController({
    enabled: true,
    pollIntervalMs: STATUS_POLL_MS,
    state: {
      mapName,
      stage: workspaceStage,
      designMapPath,
      runtimeMode: navigationRuntimeMode,
      designPoseReady: designPoseInitialized,
      runtimeOwned: runRuntimeOwned,
      shutdownPending: runShutdownPending,
      missionName,
      runMissionName,
      status,
    },
    status: {
      get: getServiceStatus,
      modeOf: navigationRuntimeModeFromStatus,
    },
    session: {
      read: readMissionSession,
      save: saveMissionSession,
      recentShutdownMarker: recentRunShutdownMarker,
    },
    runtime: {
      stop: stopNavigation,
    },
    commits: {
      setStatus,
      setRuntimeMode: setNavigationRuntimeMode,
      setDesignPoseReady: setDesignPoseInitialized,
      setRuntimeOwned: setRunRuntimeOwned,
      confirmStopped: () => {
        invalidateRunPoseRequestRef.current();
        setStatus({ is_up: false, mode: "idle" });
        setNavigationRuntimeMode("idle");
        setDesignPoseInitialized(false);
        setRunPoseInitialized(false);
        setRunRuntimeOwned(false);
        setRunShutdownPending(false);
      },
    },
    onMessage: setMessage,
  });
  const {
    refreshStatus: loadStatus,
    invalidateStatus: invalidateNavigationStatus,
    isPageExitStopSent,
  } = navigationRuntimeSession;

  const applySpots = useCallback((nextSpots) => {
    setSpots(nextSpots);
    setSelectedSpotId((current) => (
      nextSpots.some((spot) => spot.id === current) ? current : ""
    ));
    setBtLayerSpotId((current) => (
      nextSpots.some((spot) => spot.id === current) ? current : ""
    ));
  }, []);

  const {
    commitLoadedSnapshot: commitLoadedDesign,
    resetNewDocument: resetNewDesignDocument,
    applySavedCanonicalSpots,
  } = useDesignDocumentContentController({
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
  }, [btLayerSpotId, designVisibleSpots, workspaceStage]);

  useEffect(() => {
    currentPoseRef.current = currentPose;
  }, [currentPose]);

  useEffect(() => {
    amclPoseRef.current = amclPose;
  }, [amclPose]);

  useEffect(() => {
    setBtStatusText(btStatusText);
  }, [btStatusText, setBtStatusText]);

  // Localization does not survive a nav restart; require a fresh pose each time.
  useEffect(() => {
    if (!runRuntimeActive) {
      invalidateRunPoseRequestRef.current();
      setRunPoseInitialized(false);
    }
  }, [runRuntimeActive]);

  // Drop live Run/Mapping maps when the backend goes down. Design uses the
  // selected PGM file, so stopping the temporary At Robot localization must
  // not unload that static map.
  useEffect(() => {
    if (
      prevRunningRef.current
      && !running
      && workspaceStage !== STAGE_AUTHORING
    ) {
      setMissionMapLoaded(false);
    }
    prevRunningRef.current = running;
  }, [running, workspaceStage]);

  useEffect(() => {
    if (!designLocalizationActive || !designPoseInitialized) {
      return undefined;
    }
    let cancelled = false;
    const tick = async () => {
      if (cancelled || nomotionUpdateBusyRef.current) return;
      nomotionUpdateBusyRef.current = true;
      try {
        await requestNoMotionUpdate();
      } catch (error) {
        console.warn("No-motion AMCL update failed:", error);
      } finally {
        nomotionUpdateBusyRef.current = false;
      }
    };
    void tick();
    const interval = window.setInterval(tick, NOMOTION_UPDATE_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [designLocalizationActive, designPoseInitialized]);

  useEffect(() => {
    if (updateTfBuffer(tfBufferRef.current, latestTf)) {
      setTfBufferRevision((value) => value + 1);
    }
  }, [latestTf]);

  const clearLocalizationPoseCache = useCallback(() => {
    tfBufferRef.current.clear();
    currentPoseRef.current = null;
    amclPoseRef.current = null;
    setTfBufferRevision((value) => value + 1);
  }, []);

  const runCommand = useCallback(async (label, action) => {
    invalidateNavigationStatus();
    setBusy(label);
    try {
      const result = await action();
      if (result?.stale || result?.skipped) return result;
      if (typeof result === "string") {
        setMessage(taskDisplayMessage(result));
      } else {
        setMessage(taskDisplayMessage(result?.message) || `${label} complete`);
      }
      return result;
    } catch (error) {
      setMessage(error instanceof Error ? taskDisplayMessage(error.message) : `${label} failed`);
    } finally {
      setBusy("");
      void loadStatus();
    }
  }, [invalidateNavigationStatus, loadStatus]);
  catalogRunCommandRef.current = runCommand;

  const runMapMissionSelection = useRunMapMissionSelectionController({
    getCurrentStage: () => workspaceStageRef.current,
    getDefaults: () => ({ mapName, missionName }),
    inventory: {
      listMaps: getPgmFiles,
      listMissions: fetchMissionNames,
    },
    snapshot: {
      get: () => runMissionSnapshot,
      load: loadRunMissionForMap,
      clear: clearRunMissionSnapshot,
      invalidate: invalidateRunMissionLoad,
      cancelAndClear: cancelAndClearRunMissionSnapshot,
    },
    commits: {
      setStage: setWorkspaceStage,
      setInteractionMode,
      setMapLoaded: setMissionMapLoaded,
      resetPose: () => {
        invalidateRunPoseRequestRef.current();
        setRunPoseInitialized(false);
      },
      invalidateGoal: invalidateNavGoal,
    },
    onMessage: setMessage,
  });
  const {
    open: showRunMapDialog,
    stage: runMapDialogStage,
    files: runMapFiles,
    missionNames: runMapDialogMissionNames,
    selectedPath: runMapPath,
    selectedMission: pendingRunMissionName,
    setSelectedMission: setPendingRunMissionName,
    busy: runMapBusy,
    openDialog: handleOpenRunMapDialog,
    changeMap: handleRunMapChange,
    cancel: cancelRunMapDialog,
    confirm: handleConfirmRunMap,
  } = runMapMissionSelection.dialog;
  const runMissionNames = runMapDialogMissionNames || [];
  const handleMissionChange = runMapMissionSelection.switchMission;
  const invalidateRunMapSelection = runMapMissionSelection.invalidate;

  const publishTeleopCommand = useCallback((motion) => (
    publishRosTopic(TELEOP_TOPIC, TELEOP_MESSAGE_TYPE, teleopTwist(motion))
  ), [publishRosTopic]);

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
  }, [clearBehaviorSelection, clearPendingBehaviorPlacement, navigationRuntimeMode]);
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
    content: { applySavedCanonicalSpots },
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
  }, [saveDesignMission]);

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

  // Destructive design-session actions (switch/new/load/exit) suspend behind
  // one shared guard so unsaved manifest edits are never silently discarded.
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

  // Step 2: with the robot localized, run the route. This only executes the
  // waypoint sequence — navigation is already up from the localize step.
  const handleRunMission = useCallback(() => {
    if (runMapBusy || !missionMapLoaded || runMapSnapshotInvalid) {
      setMessage("Wait for the selected mission to finish loading");
      return;
    }
    if (!runPoseInitialized) {
      setMessage("Localize the robot first");
      return;
    }
    if (!runRouteView.orderedSpots.length) {
      setMessage("No route to run — connect waypoints in Design first");
      return;
    }
    missionRunner.start();
  }, [
    missionMapLoaded,
    runRouteView.orderedSpots.length,
    missionRunner,
    runMapBusy,
    runMapSnapshotInvalid,
    runPoseInitialized,
  ]);

  const stopActiveNavigationSession = useNavigationStopCoordinator({
    getDefaultClearRunSnapshot: () => workspaceStageRef.current === STAGE_RUN,
    invalidate: {
      mapSelection: invalidateRunMapSelection,
      poseRequest: () => invalidateRunPoseRequestRef.current(),
      navGoal: invalidateNavGoal,
    },
    runner: { stop: missionRunner.stop },
    navigation: { stop: stopNavigation },
    snapshot: { cancelAndClear: cancelAndClearRunMissionSnapshot },
    pose: {
      clearCache: clearLocalizationPoseCache,
      resetMappingSync: resetMappingPoseSync,
    },
    commits: {
      setInteractionView: () => setInteractionMode("view"),
      hideMap: () => setMissionMapLoaded(false),
      setStatusIdle: () => setStatus({ is_up: false, mode: "idle" }),
      setRuntimeModeIdle: () => setNavigationRuntimeMode("idle"),
      setDesignPoseReady: setDesignPoseInitialized,
      setRunPoseReady: setRunPoseInitialized,
      setRuntimeOwned: setRunRuntimeOwned,
      setShutdownPending: setRunShutdownPending,
    },
    session: { save: saveMissionSession },
  });

  const handleStopNavigation = useCallback(() => runCommand(
    "Stop",
    () => stopActiveNavigationSession(),
  ), [runCommand, stopActiveNavigationSession]);

  const protectedMapPaths = useMemo(() => (
    (running || missionRunnerActive || runShutdownPending) && runMapPath
      ? [runMapPath]
      : []
  ), [missionRunnerActive, runMapPath, runShutdownPending, running]);

  const mappingLifecycle = useMappingLifecycleController({
    active: workspaceStage === STAGE_MAPPING,
    inventoryRefreshToken: mapEditorReloadToken,
    getMapName: () => designMapName,
    runtime: {
      prepareStart: () => {
        setWorkspaceStage(STAGE_MAPPING);
        clearLocalizationPoseCache();
        resetMappingPoseSync();
      },
      start: (targetMapName) => startNavigation("map", targetMapName),
      commitStarted: (targetMapName) => {
        setNavigationRuntimeMode("mapping");
        setDesignPoseInitialized(false);
        setRunRuntimeOwned(false);
        setRunShutdownPending(false);
      },
      persistStarted: (targetMapName) => {
        saveMissionSession({
          mapName: targetMapName,
          workspaceStage: STAGE_MAPPING,
          navigationRuntimeMode: "mapping",
          designPoseInitialized: false,
          runRuntimeOwned: false,
          runShutdownPending: false,
          runShutdownRequestedAt: null,
        });
      },
      stop: () => stopActiveNavigationSession({ clearRunSnapshot: false }),
    },
    document: {
      save: saveNavigationMap,
      commitSavedMap: (targetMapName) => {
        invalidateDesignMission({ mapName: targetMapName, missionName });
        setMapName(targetMapName);
        setWorkspaceStage(STAGE_MAPPING);
        setInteractionMode("view");
        setMapEditorReloadToken((value) => value + 1);
      },
    },
    inventory: {
      list: getPgmFiles,
      remove: deletePgmMap,
      isProtected: (path) => protectedMapPaths.includes(path),
    },
    runCommand,
    onMessage: setMessage,
  });
  const {
    savedMaps,
    start: handleStartMapping,
    stop: handleStopMapping,
    removeSavedMap: handleDeleteSavedMap,
    saveDialog: mappingSaveDialog,
  } = mappingLifecycle;
  const {
    open: showSaveMapDialog,
    name: saveMapName,
    setName: setSaveMapName,
    openDialog: handleOpenSaveMapDialog,
    close: closeSaveMapDialog,
    confirm: handleConfirmSaveMap,
  } = mappingSaveDialog;

  const cancelPendingDesignLocalization = useCallback(() => (
    cancelPendingDesignLocalizationRef.current()
  ), []);

  const handleSelectSpot = useCallback((spotId) => {
    setSelectedSpotId(spotId);
    clearBehaviorSelection();
    clearPendingBehaviorPlacement();
    setEditingSpotId("");
    setEditingSpotLabel("");
    setShowWaypointOptions(false);
    setInteractionMode("view");
  }, [clearBehaviorSelection, clearPendingBehaviorPlacement]);

  const prepareMissionRouteEditing = useCallback(() => {
    cancelPendingDesignLocalization();
    setWorkspaceStage(STAGE_AUTHORING);
    clearPendingBehaviorPlacement();
    setShowWaypointOptions(false);
    setBtLayerSpotId("");
    setInteractionMode("view");
  }, [cancelPendingDesignLocalization, clearPendingBehaviorPlacement]);

  const selectMissionRouteSpot = useCallback((spotId) => {
    handleSelectSpot(spotId);
    setBtLayerSpotId("");
  }, [handleSelectSpot]);

  const {
    routeMode: missionRouteMode,
    routeSourceId: missionRouteSourceId,
    routeView: designRouteView,
    toggleMode: handleToggleMissionRouteMode,
    handleSpotClick: handleMissionRouteSpotClick,
    handleMapClick: handleMissionRouteMapClick,
    clearRoute: handleClearMissionRoute,
    openLoop: handleOpenMissionRouteLoop,
    moveSpot: handleMoveRouteSpot,
    removeSpot: handleRemoveRouteSpot,
    resetEditing: resetMissionRouteEditing,
    clearSource: clearMissionRouteSource,
    tryAcquireMutationLock: tryAcquireRouteMutationLock,
    releaseMutationLock: releaseRouteMutationLock,
    setRouteSourceAfterExternalMutation,
  } = useMissionRouteEditor({
    spots: designVisibleSpots,
    flowNodes: missionFlowNodes,
    flowEdges: missionFlowEdges,
    setFlowNodes: setMissionFlowNodes,
    setFlowEdges: setMissionFlowEdges,
    busy: Boolean(busy),
    documentReady: designDocumentReady,
    markDirty: markDesignDirty,
    onPrepareEditMode: prepareMissionRouteEditing,
    onSelectSpot: selectMissionRouteSpot,
    onMessage: setMessage,
  });
  resetMissionRouteEditingRef.current = resetMissionRouteEditing;
  clearMissionRouteSourceRef.current = clearMissionRouteSource;

  const {
    renderedRouteView,
    designPanelRouteSpots,
    designPanelRouteClosed,
  } = deriveMissionRoutePresentation({
    runSessionActive,
    designDocumentReady,
    runRouteView,
    designRouteView,
  });

  const clearWaypointBehaviorSelection = useCallback(() => {
    clearBehaviorSelection();
    clearPendingBehaviorPlacement();
  }, [clearBehaviorSelection, clearPendingBehaviorPlacement]);
  const {
    createOnMap: createDesignWaypointOnMap,
    createAtRobot: createDesignWaypointAtRobot,
    moveWaypoint: handleMoveSpot,
    startRename: handleStartRenameSpot,
    cancelRename: handleCancelSpotRename,
    commitRename: handleCommitSpotRename,
    deleteWaypoint: handleDeleteSpot,
  } = useDesignWaypointController({
    spots,
    setSpots,
    editingSpotId,
    editingSpotLabel,
    setEditingSpotId,
    setEditingSpotLabel,
    mapName: designMapName,
    captureDocumentLease: captureDesignMissionLease,
    isDocumentLeaseCurrent: isDesignMissionLeaseCurrent,
    getCurrentIdentity: () => ({
      mapName: designMapNameRef.current,
      missionName: designMissionNameRef.current,
    }),
    runCommand,
    markDirty: markDesignDirty,
    ledger: {
      deletedPaths: deletedMissionBtPaths,
      getLiveBtFiles,
      getPersistedLocalBtPaths,
      replaceLiveBtFiles,
      replaceDeletedBtPaths,
    },
    routeView: designRouteView,
    setFlowEdges: setMissionFlowEdges,
    tryAcquireRouteMutationLock,
    releaseRouteMutationLock,
    setRouteSourceAfterExternalMutation,
    forgetTaskSelection: forgetMissionLocalBtSelection,
    setSelectedSpotId,
    clearBehaviorSelection: clearWaypointBehaviorSelection,
    setTaskLayerSpotId: setBtLayerSpotId,
    setInteractionMode,
    setShowWaypointOptions,
    setBusy,
    onMessage: setMessage,
  });

  const handleOpenWaypointBt = useCallback((spotId) => {
    const spot = designVisibleSpots.find((item) => item.id === spotId);
    if (!spot) return;
    cancelPendingDesignLocalization();
    handleSelectSpot(spotId);
    resetMissionRouteEditing();
    setBtLayerSpotId(spotId);
    setMessage(`Editing ${spot.label || spot.id} Waypoint Task`);
  }, [
    cancelPendingDesignLocalization,
    designVisibleSpots,
    handleSelectSpot,
    resetMissionRouteEditing,
  ]);

  const handleClearMapSelection = useCallback(() => {
    if (btLayerSpotId) {
      setBtLayerSpotId("");
      return;
    }
    setSelectedSpotId("");
    clearBehaviorSelection();
    clearPendingBehaviorPlacement();
    setEditingSpotId("");
    setEditingSpotLabel("");
  }, [btLayerSpotId, clearBehaviorSelection, clearPendingBehaviorPlacement]);

  behaviorPrepareSelectRef.current = () => {
    cancelPendingDesignLocalization();
    setSelectedSpotId("");
    setEditingSpotId("");
    setEditingSpotLabel("");
    setShowWaypointOptions(false);
    setBtLayerSpotId("");
    setInteractionMode("view");
  };
  const handleSelectBehaviorNode = useCallback((nodeId) => {
    selectBehaviorNode(nodeId);
  }, [selectBehaviorNode]);

  const handleToggleWaypointOptions = useCallback(() => {
    cancelPendingDesignLocalization();
    setWorkspaceStage(STAGE_AUTHORING);
    setShowWaypointOptions((value) => !value);
  }, [
    cancelPendingDesignLocalization,
  ]);

  const handleToggleSpotMode = useCallback(() => {
    cancelPendingDesignLocalization();
    setWorkspaceStage(STAGE_AUTHORING);
    clearPendingBehaviorPlacement();
    clearBehaviorSelection();
    setBtLayerSpotId("");
    setEditingSpotId("");
    setEditingSpotLabel("");
    setShowWaypointOptions(false);
    setInteractionMode((value) => (value === "spot" ? "view" : "spot"));
  }, [
    cancelPendingDesignLocalization,
    clearBehaviorSelection,
    clearPendingBehaviorPlacement,
  ]);

  const waitForAutoLocalizedPose = useCallback(async () => {
    let latestPose = null;
    for (let attempt = 0; attempt < AUTO_LOCALIZE_MAX_UPDATES; attempt += 1) {
      await requestNoMotionUpdate();
      await delay(AUTO_LOCALIZE_UPDATE_DELAY_MS);
      const amclPoseMessage = amclPoseRef.current;
      const pose = amclPoseMessage?.pose?.pose;
      if (pose?.position) {
        latestPose = pose;
        if (
          attempt + 1 >= AUTO_LOCALIZE_MIN_UPDATES &&
          amclPoseLooksLocalized(amclPoseMessage)
        ) {
          return pose;
        }
      }
    }
    if (latestPose?.position) return latestPose;
    throw new Error("Robot pose unavailable after automatic localization");
  }, []);

  const designAtRobotWaypoint = useDesignAtRobotWaypointController({
    getState: () => ({
      stage: workspaceStageRef.current,
      interactionMode,
      documentReady: designDocumentReady,
      mapName: designMapName,
      mapPath: designMapPath,
      mappingActive: mappingRuntimeActive,
      runActive: runRuntimeActive,
      runnerActive: missionRunnerActive,
      designLocalizationActive,
    }),
    document: {
      captureLease: captureDesignMissionLease,
      isLeaseCurrent: isDesignMissionLeaseCurrent,
    },
    runtime: {
      startLocalization: (targetMapName) => startNavigation("localize", targetMapName),
      configureAmcl: configureDesignLocalizationAmcl,
      cleanupStartedRuntime: () => {
        invalidateNavigationStatus();
        return stopNavigation();
      },
      clearPoseCache: clearLocalizationPoseCache,
      publishInitialPose: sendInitialPoseEstimate,
      waitForConvergence: waitForAutoLocalizedPose,
      poseCoordinates: (pose) => ({
        x: Number(pose?.position?.x ?? 0),
        y: Number(pose?.position?.y ?? 0),
        yaw: yawFromPose(pose),
      }),
      invalidateStatus: invalidateNavigationStatus,
      stop: stopNavigation,
      stopShared: handleStopNavigation,
    },
    waypoint: {
      createAtRobot: createDesignWaypointAtRobot,
    },
    ui: {
      prepareBegin: () => {
        setWorkspaceStage(STAGE_AUTHORING);
        clearPendingBehaviorPlacement();
        clearBehaviorSelection();
        setSelectedSpotId("");
        resetMissionRouteEditing();
        setShowWaypointOptions(false);
        setDesignPoseInitialized(false);
        clearLocalizationPoseCache();
      },
      setRuntimeMode: setNavigationRuntimeMode,
      setRuntimeOwned: setRunRuntimeOwned,
      setShutdownPending: setRunShutdownPending,
      setInteractionMode,
      setWaypointOptionsOpen: setShowWaypointOptions,
      setDesignPoseReady: setDesignPoseInitialized,
    },
    session: {
      save: saveMissionSession,
    },
    runCommand,
    onMessage: setMessage,
  });
  const {
    begin: handleCreateSpotAtRobot,
    completeAtPose: handleCompleteDesignAtRobotPose,
    cancelPending: cancelPendingAtRobotLocalization,
  } = designAtRobotWaypoint;
  cancelPendingDesignLocalizationRef.current = cancelPendingAtRobotLocalization;

  const runNavigationLocalization = useRunNavigationLocalizationController({
    getState: () => ({
      stage: workspaceStageRef.current,
      runStage: STAGE_RUN,
      mapBusy: runMapBusy,
      snapshotInvalid: runMapSnapshotInvalid,
      interactionMode,
      running,
      runtimeMode: navigationRuntimeMode,
      targetMapName: runtimeMapName,
      sessionMapName: mapName,
    }),
    runtime: {
      start: startNavigation,
      stop: stopNavigation,
      pageExitStopSent: isPageExitStopSent,
    },
    commits: {
      setStage: setWorkspaceStage,
      setRuntimeMode: setNavigationRuntimeMode,
      setDesignPoseReady: setDesignPoseInitialized,
      setRunPoseReady: setRunPoseInitialized,
      setRuntimeOwned: setRunRuntimeOwned,
      setShutdownPending: setRunShutdownPending,
      setInteractionMode,
      persistClaim: saveMissionSession,
      persistRollback: saveMissionSession,
    },
    pose: {
      clearCache: clearLocalizationPoseCache,
      publish: sendInitialPoseEstimate,
      waitForConvergence: waitForAutoLocalizedPose,
    },
    runCommand,
    onMessage: setMessage,
  });
  const {
    localize: handleLocalize,
    estimatePose: handleRunPoseEstimate,
    invalidatePoseRequest: invalidateRunPoseRequest,
  } = runNavigationLocalization;
  invalidateRunPoseRequestRef.current = invalidateRunPoseRequest;

  const handleCreateSpotAtPose = useCallback(async (x, y, yaw) => {
    if (workspaceStage === STAGE_NAVIGATE) {
      if (interactionMode === "initial") {
        void handleRunPoseEstimate(x, y, yaw);
      } else if (interactionMode === "goal") {
        void handleSendNavGoal(x, y, yaw);
      }
      return;
    }
    if (workspaceStage === STAGE_RUN) {
      // The BT node being up is expected here; run pose estimation must not
      // fall through to the design-stage guards below.
      if (interactionMode === "initial") void handleRunPoseEstimate(x, y, yaw);
      return;
    }
    if (workspaceStage === STAGE_AUTHORING && !designDocumentReady) return;
    if (interactionMode === "initial") {
      void handleCompleteDesignAtRobotPose(x, y, yaw);
      return;
    }
    if (interactionMode === "behavior" && pendingBehaviorNodeTag) {
      placePendingBehaviorAtPose(x, y, yaw);
      return;
    }
    if (interactionMode !== "spot") return;
    await createDesignWaypointOnMap(x, y, yaw);
  }, [handleSendNavGoal,
    placePendingBehaviorAtPose,
    designDocumentReady,
    createDesignWaypointOnMap,
    handleCompleteDesignAtRobotPose,
    handleRunPoseEstimate,
    interactionMode,
    pendingBehaviorNodeTag,
    workspaceStage,
  ]);

  const waypointBtLayerOpen = !!waypointBtLayer;
  const {
    locked: designHistoryLocked,
    undoAction: handleUndoDesign,
    redoAction: handleRedoDesign,
  } = useDesignHistoryInteractionController({
    active: workspaceStage === STAGE_AUTHORING,
    documentReady: designDocumentReady,
    busy,
    mapBusy: designMapBusy,
    taskLayerOpen: waypointBtLayerOpen,
    canUndo: canUndoDesign,
    canRedo: canRedoDesign,
    undo: undoDesignHistory,
    redo: redoDesignHistory,
    onMessage: setMessage,
  });

  useMissionWorkspaceExitGuard({
    busy,
    btNodeBusy,
    designMapBusy,
    runMapBusy,
    mapEditorBusy: mapEditor.busy,
    mapEditorDirty: mapEditor.dirty,
    designMapEditorDirty: designMapEditor.dirty,
    mappingRuntimeActive,
    runRuntimeActive,
    designLocalizationActive,
    navigationRuntimeMode,
    missionRunnerActive,
    runShutdownPending,
    navGoalDriving,
    onExitStateChange,
    exitHandleRef,
    runGuardedDesignAction,
  });

  const handleSelectStageTab = (stageId) => {
    if (stageId === workspaceStage) return;

    const applyStageSelection = () => {
      // Run and Navigate share the nav runtime AND the loaded map snapshot:
      // an idle switch between them may retain the loaded map. Active runtime
      // switches are stopped before this cleanup runs.
      const runFamilySwitch = (
        (stageId === STAGE_RUN || stageId === STAGE_NAVIGATE)
        && (workspaceStage === STAGE_RUN || workspaceStage === STAGE_NAVIGATE)
      );
      invalidateRunMapSelection();
      invalidateRunPoseRequestRef.current();
      cancelPendingDesignLocalization();
      if (workspaceStage === STAGE_NAVIGATE) invalidateNavGoal();
      if (!runFamilySwitch) setMissionMapLoaded(false);
      setInteractionMode("view");
      clearPendingBehaviorPlacement();
      setShowWaypointOptions(false);
      resetMissionRouteEditing();
      setBtLayerSpotId("");
      setMapEditToolsOpen(false);
      setLabelToolsOpen(false);
      setWorkspaceStage(stageId);
    };

    const leavingRunFamilyStage = (
      workspaceStage === STAGE_RUN || workspaceStage === STAGE_NAVIGATE
    );
    const runFamilyRuntimeNeedsStop = (
      navigationRuntimeMode === "run"
      || runRuntimeActive
      || runRuntimeOwned
      || runShutdownPending
      || missionRunnerActive
      || navGoalDriving
    );

    if (leavingRunFamilyStage && runFamilyRuntimeNeedsStop) {
      const clearRunSnapshot = workspaceStage === STAGE_RUN;
      void runCommand("Stop", async () => {
        const result = await stopActiveNavigationSession({ clearRunSnapshot });
        applyStageSelection();
        return result;
      });
      return;
    }

    applyStageSelection();
  };

  return (
    <>
      <MissionCanvasDialogLayer
        saveMap={{
          open: showSaveMapDialog,
          value: saveMapName,
          busy: !!busy,
          onChange: setSaveMapName,
          onCancel: closeSaveMapDialog,
          onSubmit: handleConfirmSaveMap,
        }}
        saveMission={{
          open: showSaveMissionDialog,
          value: saveMissionName,
          existingNames: designCatalog.names,
          currentName: missionName,
          busy: !!busy,
          onChange: setSaveMissionName,
          onCancel: closeSaveMissionDialog,
          onSubmit: handleConfirmSaveMission,
        }}
        renameMission={{
          open: showRenameMissionDialog,
          value: renameMissionName,
          existingNames: designCatalog.names.filter((name) => name !== missionName),
          busy: !!busy,
          onChange: setRenameMissionName,
          onCancel: closeRenameMissionDialog,
          onSubmit: handleConfirmRenameMission,
        }}
        duplicateMission={{
          open: showDuplicateMissionDialog,
          value: duplicateMissionName,
          existingNames: designCatalog.names,
          busy: !!busy,
          onChange: setDuplicateMissionName,
          onCancel: closeDuplicateMissionDialog,
          onSubmit: handleConfirmDuplicateMission,
        }}
        deleteMission={{
          open: showDeleteMissionDialog,
          body: `Delete mission "${missionName}"? This permanently removes its waypoints, route, and Waypoint Tasks.`,
          busy: !!busy,
          onConfirm: handleConfirmDeleteMission,
          onCancel: closeDeleteMissionDialog,
        }}
        unsaved={{
          open: showUnsavedDialog,
          body: `"${missionName}" has unsaved changes.`,
          altLabel: designCatalog.names.includes(missionName) ? "Save & continue" : "",
          hint: designCatalog.names.includes(missionName)
            ? ""
            : "Use Save Mission to name this mission first.",
          busy: !!busy || unsavedSavePending,
          onConfirm: () => resolveUnsavedDialog("discard"),
          onAlt: () => resolveUnsavedDialog("save"),
          onCancel: cancelUnsavedDialog,
        }}
        designLoad={{
          open: showDesignMapDialog && !showUnsavedDialog,
          files: designMapFiles,
          selectedPath: pendingDesignMapPath,
          missionNames: designMissionNames,
          selectedMissionName: pendingDesignMissionName,
          busy: designMapBusy,
          catalogReady: designDialogCatalogReady,
          onChange: handleDesignMapChange,
          onMissionChange: setPendingDesignMissionName,
          onCancel: cancelDesignMapDialog,
          onSubmit: () => runGuardedDesignAction(handleConfirmDesignMap),
        }}
        runLoad={{
          open: showRunMapDialog,
          files: runMapFiles,
          selectedPath: runMapPath,
          missionNames: runMapDialogStage === STAGE_NAVIGATE ? null : runMissionNames,
          selectedMissionName: pendingRunMissionName,
          busy: runMapBusy,
          navigationOnly: runMapDialogStage === STAGE_NAVIGATE,
          onChange: handleRunMapChange,
          onMissionChange: setPendingRunMissionName,
          onCancel: cancelRunMapDialog,
          onSubmit: handleConfirmRunMap,
        }}
        editLoad={{
          open: showEditMapDialog,
          files: mapEditor.files,
          selectedPath: pendingEditMapPath,
          busy: mapEditor.busy,
          onChange: setPendingEditMapPath,
          onCancel: cancelEditMapDialog,
          onSubmit: handleConfirmEditMap,
        }}
      />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <StageRail
          busy={busy}
          mappingRuntimeActive={mappingRuntimeActive}
          onSelectStage={handleSelectStageTab}
          workspaceStage={workspaceStage}
        />

        <MissionStageChrome
          stage={workspaceStage}
          header={{
            btNodeBusy,
            busy,
            designMapBusy,
            mapEditorBusy: mapEditor.busy,
            mappingRuntimeActive,
            missionRunnerActive,
            onBackToDesignMap: () => setBtLayerSpotId(""),
            onOpenDesignMap: handleOpenDesignMapDialog,
            onOpenEditMap: handleOpenEditMapDialog,
            onOpenRunMap: handleOpenRunMapDialog,
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
          }}
          map={{
            mappingEditorActive,
            waypointBtLayer,
          }}
          sidebar={{
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
              selectedSpotId,
            },
            mapping: {
              onMessage: setMessage,
              onPublishTeleop: publishTeleopCommand,
              teleopDisabled,
              topicRows,
            },
            navigation: {
              goalStatus: navGoalStatus,
              mapName: runtimeMapName,
              poseReady: runPoseInitialized,
              topicRows,
            },
            run: {
              mapName: missionMapLoaded && !runMapSnapshotInvalid ? runMapName : "",
              missionName: runMissionName,
              missionNames: runCatalog.names,
              missionSelectDisabled: !!busy || missionRunnerActive || runMapBusy || !missionMapLoaded,
              onMissionChange: handleMissionChange,
              poseReady: runPoseInitialized,
              runner: missionRunner,
              running,
              topicRows,
            },
          }}
        >
          <MissionCanvasSceneSurface
            stage={{
              id: workspaceStage,
              mappingEditorActive,
              mappingTopicsActive,
              runTopicsActive,
              designLocalizationActive,
              navigationTopicsActive: stageNavigationTopicsActive,
              designMapActive,
              runFamily: runFamilyStage,
              running,
            }}
            scene={{
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
            }}
            mission={{
              overlayActive: missionOverlayActive,
              renderedVisibleSpots,
              selectedSpotId,
              activeWaypointId: missionRunner.activeSpotId,
              followRobot: missionFollowRobot,
              renderedBehaviorNodes,
              selectedBehaviorNodeId,
              behaviorPreviewNode,
              renderedRouteView,
              routeMode: missionRouteMode,
              routeSourceId: missionRouteSourceId,
              mapLoaded: missionMapLoaded,
              documentReady: designDocumentReady,
              designMapPath,
              mapName,
            }}
            editors={{ mapEditor, designMapEditor, runDisplayMapEditor }}
            interaction={{
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
              onBtLayerClose: () => setBtLayerSpotId(""),
            }}
            bt={{
              waypointLayer: waypointBtLayer,
              runLayer: runBtLayer,
              activeLayer: activeBtLayer,
            }}
            layers={{
              active: activeLayers,
              toggles: layerToggles,
              needsGlobalCostmap,
              needsLocalCostmap,
              needsScan,
              needsPlan,
              needsRobotModel,
              needsTf,
            }}
            hud={{
              design: {
                busy,
                canRedoDesign,
                canUndoDesign,
                currentMapName: designMapName,
                designHistoryLocked,
                designMapActive: designMapActive
                  && (designDocumentReady || Boolean(designMissionLoadError)),
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
                onDeleteMission: openDeleteMissionDialog,
                onDuplicateMission: handleOpenDuplicateMissionDialog,
                onMissionChange: (name) => (
                  runGuardedDesignAction(() => handleDesignMissionChange(name))
                ),
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
                showWaypointOptions,
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
                showSaveMapDialog,
              },
              mapEdit: {
                labelToolsOpen,
                mapEditor,
                mapEditToolsOpen,
                setLabelToolsOpen,
                setMapEditToolsOpen,
              },
              run: {
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
                running,
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
                setInteractionMode,
              },
            }}
          />
        </MissionStageChrome>
      </div>
    </>
  );
}
