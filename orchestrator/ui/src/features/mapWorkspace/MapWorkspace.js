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

import { taskDisplayMessage, navigationRuntimeModeFromStatus, fetchLegacySpotsForMap, loadMissionBtFileOrDefault } from "./lib/workspaceRuntime";
import useRecordMapChange from "../autonomyCanvas/hooks/useRecordMapChange";
import useMapPoseInteraction from "./hooks/useMapPoseInteraction";
import useWorkspaceLocalize from "./hooks/useWorkspaceLocalize";
import useWorkspaceMapSelection from "./hooks/useWorkspaceMapSelection";
import useWorkspaceNavigationSession from "./hooks/useWorkspaceNavigationSession";
import useWorkspaceMissionRunner from "./hooks/useWorkspaceMissionRunner";
import useWorkspaceMapScene from "./hooks/useWorkspaceMapScene";
import useDesignEditingControllers from "../missionCanvas/hooks/useDesignEditingControllers";
import useDesignWorkspaceModel from "../missionCanvas/hooks/useDesignWorkspaceModel";
import useMapWorkspaceState from "./hooks/useMapWorkspaceState";
import useWorkspaceMapping from "../missionCanvas/hooks/useWorkspaceMapping";

import useWorkspaceStageTransition from "./hooks/useWorkspaceStageTransition";
import useWorkspaceLocalizationEffects from "./hooks/useWorkspaceLocalizationEffects";

import useRunMissionStart from "./hooks/useRunMissionStart";
import useRecordWorkspaceData from "../autonomyCanvas/hooks/useRecordWorkspaceData";

import MapWorkspaceView from "./components/workspace/MapWorkspaceView";
import useAutoLocalizedPose from "./hooks/useAutoLocalizedPose";
import useAutonomyWorkspaceRuntime from "../autonomyCanvas/hooks/useAutonomyWorkspaceRuntime";

import useCollectMissionSelection from "../autonomyCanvas/hooks/useCollectMissionSelection";

import { useCallback, useEffect } from "react";

import { RunnerStatus } from "../../hooks/missionRunnerCore";

import { useNavigationRosPublisher } from "../../hooks/useNavigationRosTopic";
import { useRosServiceCaller } from "../../hooks/useRosServiceCaller";
import { cancelNavigateToPoseGoal, stopNavigation } from "../../utils/navigationApi";

import { WORKSPACE_MISSION, STAGE_AUTHORING, STAGE_NAVIGATE, STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN } from "./lib/stages";
import { saveMissionSession } from "./lib/session";
import { DEFAULT_MAP_NAME, DEFAULT_MISSION_NAME } from "./lib/missionNames";

import useMissionBtNodeLease from "./hooks/useMissionBtNodeLease";

import useMissionWorkspaceExitGuard from "./hooks/useMissionWorkspaceExitGuard";
import useRunMissionSnapshot from "./hooks/useRunMissionSnapshot";

import useMissionBtLayerComposition from "./hooks/useMissionBtLayerComposition";
import useNavigationStopCoordinator from "./hooks/useNavigationStopCoordinator";

import useDesignHistoryInteractionController from "../missionCanvas/hooks/useDesignHistoryInteractionController";

import { TELEOP_MESSAGE_TYPE, TELEOP_TOPIC, teleopTwist } from "./components/MappingTeleopPanel";

const STATUS_POLL_MS = 10000;

const NOMOTION_UPDATE_INTERVAL_MS = 1000;

// Mounted by the Autonomy Studio shell, which owns the page root, the app bar
// and the workspace choice. `dialogHost` is the shell's page root (dialogs must
// portal inside the --mc-* token scope), `onExitStateChange` publishes the
// reason the back button is blocked, and `exitHandleRef` receives
// `requestExit(onExit)` so leaving passes through the unsaved-Design guard.
export default function MapWorkspace({
  workspaceKind = WORKSPACE_MISSION,
  StageChrome,
  SceneSurface,
  railFooter = null,
  dialogHost = null,
  onExitStateChange = null,
  exitHandleRef = null,
}) {
  const workspace = useMapWorkspaceState({
    workspaceKind,
  });
  const {
    initialSession, invalidateRunPoseRequestRef, amclPoseRef, status, setStatus, setPendingRunStart,
    runStartRequestRef, busy, setMessage, setRunBtVisualizationActive, setInteractionMode,
    runPoseInitialized, setRunPoseInitialized, navigationRuntimeMode, setNavigationRuntimeMode,
    setRunRuntimeOwned, runShutdownPending, setRunShutdownPending, workspaceStage, dataStageBusy,
    workspaceStageRef, missionMapLoaded, setMissionMapLoaded,
  } = workspace;

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
    revision: runMissionRevision,
    missionId: runMissionId,
    invalid: runMapSnapshotInvalid,
  } = runMissionSnapshot;

  const runtimeMapName = String(runMapName || "").trim() || DEFAULT_MAP_NAME;
  const design = useDesignWorkspaceModel({
    workspace, runtimeMapName, loadMissionBtFileOrDefault, fetchLegacySpotsForMap
  });
  const {
    missionBtFiles, setDesignPoseInitialized, fetchMissionNames, designMapBusy, undoDesignHistory,
    redoDesignHistory, canUndoDesign, canRedoDesign,
  } = design;

  const publishRosTopic = useNavigationRosPublisher();

  const running = status?.is_up ?? false;
  const { callService } = useRosServiceCaller();

  const {
    btNodeBusy,
    btNodeIsUp,
    btStatusRef,
    btStatusHistoryRef,
    ensureMissionBtActive,
    releaseMissionBt,
    setBtStatusText,
  } = useMissionBtNodeLease({
    callService,
    needsBtTopics: [STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT].includes(workspaceStage),
    onMessage: setMessage,
  });

  const {
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
  } = useWorkspaceMapScene({
    workspace, design, runMapName, running, runtimeMapName, btNodeIsUp, setBtStatusText, runSpots, runMissionFlowNodes, runMissionFlowEdges
  });

  // ── Mission runner (Run stage): navigate → run waypoint BT → advance ──────
  const {
    sendMissionGoal, navGoalPose, navGoalStatus, navGoalDriving, handleSendNavGoal,
    invalidateNavGoal, stopMissionBt, missionRunnerFlags, missionRunner,
  } = useWorkspaceMissionRunner({
    workspace, runMissionBtFiles, callService, running, btNodeIsUp, runRouteView, btStatusRef, btStatusHistoryRef, ensureMissionBtActive, releaseMissionBt, taskDisplayMessage
  });

  const { autonomyOperation, recordAuthoringLock, collectObservations, recordMissionStartRef,
    collectExecution, recordMissionMapOverlay, missionRunnerActive, recordNavigationMoving,
    collectWaypoints, autonomyMap } = useAutonomyWorkspaceRuntime({
    workspaceStage, missionMapLoaded, running, runPoseInitialized, runDisplayMapEditor,
    runShutdownPending, runtimeMapName, missionRunner, busy, runMissionSnapshot, currentPose,
    setInteractionMode, runStartRequestRef, setPendingRunStart, sendMissionGoal,
    cancelNavigateToPoseGoal, setMessage, btStatusRef, btStatusHistoryRef, callService,
    stopMissionBt, ensureMissionBtActive, releaseMissionBt, missionRunnerFlags, runRouteView,
  });

  const missionRunnerStopping = (
    missionRunner.isRunning && missionRunner.status === RunnerStatus.CANCELLED
  );
  const missionFollowRobot = (
    missionRunnerActive
    && (missionRunner.phase === "nav-sent" || missionRunner.phase === "awaiting-nav-result")
  ) || (workspaceStage === STAGE_NAVIGATE && navGoalStatus === "driving");

  const teleopDisabled = !!busy || mappingEditorActive;

  const {
    invalidateNavigationStatus, isPageExitStopSent, runCommand,
  } = useWorkspaceNavigationSession({
    design, workspace, autonomyOperation, runMissionName, STATUS_POLL_MS, navigationRuntimeModeFromStatus, taskDisplayMessage
  });

  const {
    clearLocalizationPoseCache,
  } = useWorkspaceLocalizationEffects({
    workspace, design, currentPose, amclPose, runRuntimeActive, running, designLocalizationActive, latestTf, NOMOTION_UPDATE_INTERVAL_MS
  });

  const {
    showRunMapDialog, runMapDialogStage, runMapFiles, runMapPath, pendingRunMissionName,
    setPendingRunMissionName, runMapBusy, handleOpenRunMapDialog, handleRunMapChange,
    cancelRunMapDialog, handleConfirmRunMap, runMissionNames, handleMissionChange,
    invalidateRunMapSelection,
  } = useWorkspaceMapSelection({
    workspace, design, runMissionSnapshot, loadRunMissionForMap, clearRunMissionSnapshot, invalidateRunMissionLoad, cancelAndClearRunMissionSnapshot, invalidateNavGoal, collectExecution, runtimeMapName, collectObservations
  });

  const publishTeleopCommand = useCallback((motion) => (
    publishRosTopic(TELEOP_TOPIC, TELEOP_MESSAGE_TYPE, teleopTwist(motion))
  ), [publishRosTopic]);

  const collectMission = useCollectMissionSelection({
    enabled: [STAGE_COLLECT, STAGE_LEARN].includes(workspaceStage) && missionMapLoaded,
    mapName: runtimeMapName,
    listMissions: fetchMissionNames,
    loadMission: loadRunMissionForMap,
    onMessage: setMessage,
  });
  const linkedRecordMissionNames = collectExecution.linkedMissions.map(mission => mission.mission_name)
    .filter(name => collectMission.names.includes(name));
  const recordMissionName = linkedRecordMissionNames.includes(collectMission.name) ? collectMission.name : "";

  // Step 2: with the robot localized, run the saved mission, including each
  // waypoint task after its navigation goal succeeds.
  const {
    handleRunMission,
  } = useRunMissionStart({
    workspace, collectMission, recordNavigationMoving, runMapBusy, runMapSnapshotInvalid, runMissionName, runtimeMapName, runMapName, runMissionSnapshot, loadRunMissionForMap, recordMissionStartRef, runMissionId, runMissionRevision, missionRunner
  });

  const stopActiveNavigationSession = useNavigationStopCoordinator({
    autonomy: { needsStop: autonomyOperation.needsStop, stop: autonomyOperation.stop },
    getDefaultClearRunSnapshot: () => (workspaceStageRef.current === STAGE_RUN || [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(workspaceStageRef.current)),
    invalidate: {
      mapSelection: invalidateRunMapSelection,
      poseRequest: () => invalidateRunPoseRequestRef.current(),
      navGoal: invalidateNavGoal,
    },
    runner: { stop: () => workspaceStageRef.current === STAGE_COLLECT ? collectExecution.stop() : missionRunner.stop() },
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

  const mapping = useWorkspaceMapping({
    workspace, design, running, missionRunnerActive, runMapPath, clearLocalizationPoseCache, resetMappingPoseSync, stopActiveNavigationSession, runCommand
  });

  const waitForAutoLocalizedPose = useAutoLocalizedPose(amclPoseRef);
  const designEditing = useDesignEditingControllers({
    design, workspace, designVisibleSpots, designDocumentReady, runCommand, runSessionActive, runRouteView, waitForAutoLocalizedPose, mappingRuntimeActive, runRuntimeActive, missionRunnerActive, designLocalizationActive, invalidateNavigationStatus, clearLocalizationPoseCache, handleStopNavigation
  });
  const {
    selectedBtLayerSpot, selectedBtLayerDefaultPath, selectedBtLayerPaths, selectedBtLayerPath,
    missionBtLoadingPath, localBtFileActionsDisabled, loadMissionLocalBtXml, saveMissionLocalBtXml,
    selectMissionLocalBtXml, saveMissionLocalBtXmlAs, setMissionLocalBtDefault,
    handleMissionLocalBtXmlChange, runGuardedDesignAction,
  } = designEditing;

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
    autonomyOperation,
    runVisibleSpots,
    runMissionBtFiles,
    btActiveNodesText,
  });

  useEffect(() => {
    setRunBtVisualizationActive(runBtViewActive);
  }, [runBtViewActive, setRunBtVisualizationActive]);

  const recordData = useRecordWorkspaceData({
    workspace, runtimeMapName, collectExecution, runMissionSnapshot, running, collectObservations, recordMissionName, currentPose, recordAuthoringLock
  });
  const { collectEditorBusy, collectDraft, qwenObservation } = recordData;

  const {
    openRecordMapPicker,
  } = useRecordMapChange({
    runCommand, collectDraft, collectExecution, stopActiveNavigationSession, qwenObservation,
    collectObservations, setMessage, running, runRuntimeActive, missionMapLoaded,
    handleOpenRunMapDialog,
  });
  const {
    handleLocalize, handleRunPoseEstimate,
  } = useWorkspaceLocalize({
    workspace, design, runMapBusy, runMapSnapshotInvalid, running, runtimeMapName, isPageExitStopSent, clearLocalizationPoseCache, waitForAutoLocalizedPose, runCommand
  });

  const {
    handleCreateSpotAtPose,
  } = useMapPoseInteraction({
    workspace, designEditing, design, handleRunPoseEstimate, handleSendNavGoal, collectObservations, designDocumentReady
  });

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

  const collectionBlockReason = collectDraft.active || collectDraft.recording
    ? "Save or discard the current episode before going back"
    : collectObservations.draft || collectObservations.planDirty
      ? "Save or discard the observation changes before going back"
      : collectEditorBusy || dataStageBusy
        ? "Finish the current annotation or operation before going back"
        : qwenObservation.needsStop ? "Stop reasoning before going back" : "";
  useMissionWorkspaceExitGuard({
    collectionBlockReason,
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

  const {
    handleSelectStageTab, handleTestModel,
  } = useWorkspaceStageTransition({
    workspace, recordData, designEditing, design, workspaceKind, collectObservations, invalidateRunMapSelection, invalidateNavGoal, runRuntimeActive, missionRunnerActive, navGoalDriving, runCommand, stopActiveNavigationSession
  });

  return <MapWorkspaceView
    SceneSurface={SceneSurface}
    StageChrome={StageChrome}
    model={{
      workspace, design, designEditing, recordData, mapping,
      runtime: {
        activeBtLayer, activeLayers, autonomyMap, autonomyOperation, btNodeBusy, cancelEditMapDialog,
        cancelRunMapDialog, collectExecution, collectMission, collectObservations, collectWaypoints,
        currentPose, designDocumentReady, designHistoryLocked, designLocalizationActive,
        designMapActive, designMapEditor, designPanelBehaviorNodes, designPanelSpots, dialogHost,
        displayTf, displayedMap, footprint, globalCostmap, handleConfirmEditMap, handleConfirmRunMap,
        handleCreateSpotAtPose, handleLocalize, handleMissionChange, handleOpenEditMapDialog,
        handleOpenRunMapDialog, handleRedoDesign, handleRunMapChange, handleRunMission,
        handleSelectStageTab, handleStopNavigation, handleTestModel, handleUndoDesign, layerToggles,
        linkedRecordMissionNames, localCostmap, mapEditor, mappingEditorActive, mappingPoseSync,
        mappingRuntimeActive, mappingTopicsActive, missionFollowRobot, missionOverlayActive,
        missionRunner, missionRunnerActive, missionRunnerStopping, navGoalDriving, navGoalPose,
        navGoalStatus, needsGlobalCostmap, needsLocalCostmap, needsPlan, needsRobotModel, needsScan,
        needsTf, openRecordMapPicker, pendingEditMapPath, pendingRunMissionName, plan,
        publishTeleopCommand, railFooter, recordMissionMapOverlay, recordMissionName,
        renderedBehaviorNodes, renderedVisibleSpots, runBtLayer, runCatalog, runDisplayMapEditor,
        runFamilyStage, runMapBusy, runMapDialogStage, runMapFiles, runMapName, runMapPath,
        runMapSnapshotInvalid, runMissionName, runMissionNames, runPoseSync, runRuntimeActive,
        runTopicsActive, running, runtimeMapName, scan, setPendingEditMapPath,
        setPendingRunMissionName, showEditMapDialog, showRunMapDialog, stageNavigationTopicsActive,
        teleopDisabled, topicRows, waypointBtLayer, workspaceKind,
      },
    }}
  />;
}
