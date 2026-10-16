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
import {
  configureDesignLocalizationAmcl,
  sendInitialPoseEstimate,
  startNavigation,
  stopNavigation,
} from "../../../utils/navigationApi";
import { yawFromPose } from "../../../utils/navigationTf";
import { STAGE_AUTHORING } from "../../mapWorkspace/lib/stages";
import { saveMissionSession } from "../../mapWorkspace/lib/session";
import useMissionRouteEditor from "../../mapWorkspace/hooks/useMissionRouteEditor";
import { deriveMissionRoutePresentation } from "../../mapWorkspace/hooks/useMissionMapScenePresentation";
import useDesignWaypointController from "./useDesignWaypointController";
import useDesignAtRobotWaypointController from "./useDesignAtRobotWaypointController";

// Design-only interaction composition. The shared workspace supplies map and
// navigation state; this hook owns route editing, waypoint selection/placement,
// and the temporary localization flow used by At Robot.
export default function useDesignWaypointInteraction({
  design, workspace, designVisibleSpots, designDocumentReady, markDesignDirty, runSessionActive, runRouteView, runCommand, forgetMissionLocalBtSelection, waitForAutoLocalizedPose, mappingRuntimeActive, runRuntimeActive, missionRunnerActive, designLocalizationActive, invalidateNavigationStatus, clearLocalizationPoseCache, handleStopNavigation
}) {
  const {
    cancelPendingDesignLocalizationRef, setSelectedSpotId, clearBehaviorSelection,
    clearPendingBehaviorPlacement, setEditingSpotId, setEditingSpotLabel, setShowWaypointOptions,
    setBtLayerSpotId, missionFlowNodes, missionFlowEdges, setMissionFlowNodes, setMissionFlowEdges,
    resetMissionRouteEditingRef, clearMissionRouteSourceRef, spots, setSpots, editingSpotId,
    editingSpotLabel, designMapName, captureDesignMissionLease, isDesignMissionLeaseCurrent,
    designMapNameRef, designMissionNameRef, deletedMissionBtPaths, getLiveBtFiles,
    getPersistedLocalBtPaths, replaceLiveBtFiles, replaceDeletedBtPaths, btLayerSpotId,
    behaviorPrepareSelectRef, selectBehaviorNode, designMapPath, setDesignPoseInitialized,
  } = design;
  const {
    setInteractionMode, setWorkspaceStage, busy, setMessage, setBusy, workspaceStageRef,
    interactionMode, setNavigationRuntimeMode, setRunRuntimeOwned, setRunShutdownPending,
  } = workspace;
  const cancelPendingDesignLocalization = useCallback(() => (
    cancelPendingDesignLocalizationRef.current()
  ), [cancelPendingDesignLocalizationRef]);

  const handleSelectSpot = useCallback((spotId) => {
    setSelectedSpotId(spotId);
    clearBehaviorSelection();
    clearPendingBehaviorPlacement();
    setEditingSpotId("");
    setEditingSpotLabel("");
    setShowWaypointOptions(false);
    setInteractionMode("view");
  }, [
    clearBehaviorSelection, clearPendingBehaviorPlacement, setSelectedSpotId,
    setEditingSpotId, setEditingSpotLabel, setShowWaypointOptions, setInteractionMode,
  ]);

  const prepareMissionRouteEditing = useCallback(() => {
    cancelPendingDesignLocalization();
    setWorkspaceStage(STAGE_AUTHORING);
    clearPendingBehaviorPlacement();
    setShowWaypointOptions(false);
    setBtLayerSpotId("");
    setInteractionMode("view");
  }, [
    cancelPendingDesignLocalization, clearPendingBehaviorPlacement, setWorkspaceStage,
    setShowWaypointOptions, setBtLayerSpotId, setInteractionMode,
  ]);

  const selectMissionRouteSpot = useCallback((spotId) => {
    handleSelectSpot(spotId);
    setBtLayerSpotId("");
  }, [handleSelectSpot, setBtLayerSpotId]);

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
    createFromMission: importDesignWaypoint,
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
    setBtLayerSpotId,
    setMessage,
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
  }, [
    btLayerSpotId, clearBehaviorSelection, clearPendingBehaviorPlacement,
    setBtLayerSpotId, setSelectedSpotId, setEditingSpotId, setEditingSpotLabel,
  ]);

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
    cancelPendingDesignLocalization, setWorkspaceStage, setShowWaypointOptions,
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
    setWorkspaceStage, setBtLayerSpotId, setEditingSpotId, setEditingSpotLabel,
    setShowWaypointOptions, setInteractionMode,
  ]);

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

  return {
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
  };
}
