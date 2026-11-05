// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef } from "react";
import {
  createNavigationSpot,
  deleteNavigationSpot,
  updateNavigationSpot,
} from "../../../utils/navigationSpotsApi";
import {
  defaultLocalBtXml,
  initializeCreatedWaypointLocalBt,
  localBtPathForSpot,
  localBtPathsForSpot,
  withLocalBtLibrary,
} from "../lib/missionBtFiles";
import { missionFlowEdgesForRouteOrder } from "../lib/missionFlow";
import {
  isMissionManifestSpot,
  nextWaypointLabel,
  spotPoseFromMapPose,
} from "../lib/missionSpots";

export default function useDesignWaypointController({
  spots,
  setSpots,
  editingSpotId,
  editingSpotLabel,
  setEditingSpotId,
  setEditingSpotLabel,
  mapName,
  captureDocumentLease,
  isDocumentLeaseCurrent,
  getCurrentIdentity,
  runCommand,
  markDirty,
  ledger,
  routeView,
  setFlowEdges,
  tryAcquireRouteMutationLock,
  releaseRouteMutationLock,
  setRouteSourceAfterExternalMutation,
  forgetTaskSelection,
  setSelectedSpotId,
  clearBehaviorSelection,
  setTaskLayerSpotId,
  setInteractionMode,
  setShowWaypointOptions,
  setBusy,
  onMessage,
  api = {},
}) {
  const createSpot = api.createSpot || createNavigationSpot;
  const updateSpot = api.updateSpot || updateNavigationSpot;
  const deleteSpot = api.deleteSpot || deleteNavigationSpot;
  const createPendingRef = useRef(false);
  const moveGenerationRef = useRef(new Map());
  const renameGenerationRef = useRef(new Map());
  const moveQueueRef = useRef(new Map());
  const renameQueueRef = useRef(new Map());

  const documentIsCurrent = useCallback((lease, identity) => {
    const current = getCurrentIdentity();
    return (
      isDocumentLeaseCurrent(lease)
      && current.mapName === identity.mapName
      && current.missionName === identity.missionName
    );
  }, [getCurrentIdentity, isDocumentLeaseCurrent]);

  const commitCreatedWaypoint = useCallback((createdSpot) => {
    const initialized = initializeCreatedWaypointLocalBt(spots, createdSpot, [
      ...Object.keys(ledger.getLiveBtFiles()),
      ...ledger.getPersistedLocalBtPaths(),
      ...ledger.deletedPaths,
    ]);
    const files = ledger.getLiveBtFiles();
    const emptyXml = defaultLocalBtXml(initialized.spot);
    initialized.paths.forEach((path) => { files[path] = emptyXml; });
    markDirty();
    ledger.replaceLiveBtFiles(files, { bumpBtEpoch: true });
    setSpots((current) => [...current, initialized.spot]);
    setSelectedSpotId(initialized.spot.id);
    clearBehaviorSelection();
    return initialized.spot;
  }, [clearBehaviorSelection, ledger, markDirty, setSelectedSpotId, setSpots, spots]);

  const createWaypoint = useCallback(async ({
    commandLabel,
    resolvePose,
    finalize,
    resultMessage,
  }) => {
    if (createPendingRef.current) return undefined;
    createPendingRef.current = true;
    setShowWaypointOptions(false);
    setInteractionMode("view");
    const label = nextWaypointLabel(spots);
    const targetMapName = mapName;
    const identity = { ...getCurrentIdentity() };
    const lease = captureDocumentLease();
    try {
      return await runCommand(commandLabel, async () => {
        try {
          const { x, y, yaw } = await resolvePose();
          const created = await createSpot({
            map_name: targetMapName,
            label,
            pose: spotPoseFromMapPose(x, y, yaw),
            metadata: { source: "mission_canvas", coordinate_space: "map" },
          });
          if (!documentIsCurrent(lease, identity)) {
            try { await deleteSpot(created.id, targetMapName); } catch { /* best effort */ }
            throw new Error("Map or mission changed while the waypoint was being created");
          }
          const initialized = commitCreatedWaypoint(created);
          return resultMessage
            ? resultMessage(initialized)
            : `Created ${initialized.label}`;
        } finally {
          if (finalize) {
            await finalize({ documentCurrent: documentIsCurrent(lease, identity) });
          }
        }
      });
    } finally {
      createPendingRef.current = false;
    }
  }, [
    captureDocumentLease, commitCreatedWaypoint, createSpot, deleteSpot,
    documentIsCurrent, getCurrentIdentity, mapName, runCommand,
    setInteractionMode, setShowWaypointOptions, spots,
  ]);

  const createOnMap = useCallback((x, y, yaw) => createWaypoint({
    commandLabel: "Create Waypoint",
    resolvePose: async () => ({ x, y, yaw }),
  }), [createWaypoint]);

  const createAtRobot = useCallback((options) => createWaypoint({
    ...options,
    commandLabel: "At Robot",
  }), [createWaypoint]);
  const moveWaypoint = useCallback((spotId, x, y, yaw) => {
    const spot = spots.find((item) => item.id === spotId);
    if (!spot) return Promise.resolve();
    const nextPose = spotPoseFromMapPose(x, y, yaw ?? spot.pose?.yaw ?? 0);
    const lease = captureDocumentLease();
    const identity = { ...getCurrentIdentity() };
    const leaseToken = typeof lease === "object"
      ? lease?.generation ?? lease?.id ?? JSON.stringify(lease)
      : lease;
    const identityKey = `${identity.mapName}::${identity.missionName}::${leaseToken}`;
    const existing = moveQueueRef.current.get(spotId);
    const coordinator = existing?.identityKey === identityKey
      ? existing
      : { identityKey, confirmed: spot.pose, tail: Promise.resolve() };
    const generation = (moveGenerationRef.current.get(spotId) || 0) + 1;
    moveGenerationRef.current.set(spotId, generation);
    moveQueueRef.current.set(spotId, coordinator);
    markDirty();
    setSpots((current) => current.map((item) => (
      item.id === spotId ? { ...item, pose: nextPose } : item
    )));
    if (isMissionManifestSpot(spot)) {
      coordinator.confirmed = nextPose;
      onMessage(`Moved ${spot.label || spot.id}`);
      return Promise.resolve();
    }
    const request = async () => {
      if (!documentIsCurrent(lease, identity)) return;
      try {
        const updated = await updateSpot(spotId, {
          map_name: spot.map_name,
          pose: nextPose,
          metadata: { ...(spot.metadata ?? {}), coordinate_space: "map" },
        });
        if (!documentIsCurrent(lease, identity)) return;
        coordinator.confirmed = updated.pose || nextPose;
        if (moveGenerationRef.current.get(spotId) === generation) {
          setSpots((current) => current.map((item) => (
            item.id === spotId
              ? {
                ...item,
                pose: coordinator.confirmed,
                metadata: { ...(item.metadata ?? {}), ...(updated.metadata ?? {}) },
              }
              : item
          )));
          onMessage(`Moved ${spot.label || spot.id}`);
        }
      } catch (error) {
        if (
          documentIsCurrent(lease, identity)
          && moveGenerationRef.current.get(spotId) === generation
        ) {
          setSpots((current) => current.map((item) => (
            item.id === spotId ? { ...item, pose: coordinator.confirmed } : item
          )));
          onMessage(error instanceof Error ? error.message : "Failed to move waypoint");
        }
      }
    };
    coordinator.tail = coordinator.tail.catch(() => {}).then(request);
    return coordinator.tail;
  }, [
    captureDocumentLease, documentIsCurrent, getCurrentIdentity, markDirty,
    onMessage, setSpots, spots, updateSpot,
  ]);
  const startRename = useCallback((spot) => {
    if (!spot) return;
    setSelectedSpotId(spot.id);
    clearBehaviorSelection();
    setShowWaypointOptions(false);
    setInteractionMode("view");
    setEditingSpotId(spot.id);
    setEditingSpotLabel(spot.label || spot.id);
  }, [
    clearBehaviorSelection, setEditingSpotId, setEditingSpotLabel,
    setInteractionMode, setSelectedSpotId, setShowWaypointOptions,
  ]);

  const cancelRename = useCallback(() => {
    setEditingSpotId("");
    setEditingSpotLabel("");
  }, [setEditingSpotId, setEditingSpotLabel]);

  const commitRename = useCallback((spot) => {
    if (!spot) return Promise.resolve();
    const label = editingSpotLabel.trim() || spot.label || spot.id;
    cancelRename();
    if (label === spot.label) return Promise.resolve();
    const lease = captureDocumentLease();
    const identity = { ...getCurrentIdentity() };
    const leaseToken = typeof lease === "object"
      ? lease?.generation ?? lease?.id ?? JSON.stringify(lease)
      : lease;
    const identityKey = `${identity.mapName}::${identity.missionName}::${leaseToken}`;
    const existing = renameQueueRef.current.get(spot.id);
    const coordinator = existing?.identityKey === identityKey
      ? existing
      : { identityKey, confirmed: spot.label, tail: Promise.resolve() };
    const generation = (renameGenerationRef.current.get(spot.id) || 0) + 1;
    renameGenerationRef.current.set(spot.id, generation);
    renameQueueRef.current.set(spot.id, coordinator);
    markDirty();
    setSpots((current) => current.map((item) => (
      item.id === spot.id ? { ...item, label } : item
    )));
    if (isMissionManifestSpot(spot)) {
      coordinator.confirmed = label;
      onMessage(`Renamed ${label}`);
      return Promise.resolve();
    }
    const request = async () => {
      if (!documentIsCurrent(lease, identity)) return;
      try {
        const updated = await updateSpot(spot.id, { map_name: spot.map_name, label });
        if (!documentIsCurrent(lease, identity)) return;
        coordinator.confirmed = updated.label || label;
        if (renameGenerationRef.current.get(spot.id) === generation) {
          setSpots((current) => current.map((item) => (
            item.id === spot.id
              ? withLocalBtLibrary({
                ...item,
                label: coordinator.confirmed,
                metadata: { ...(item.metadata ?? {}), ...(updated.metadata ?? {}) },
              }, localBtPathForSpot(item), localBtPathsForSpot(item))
              : item
          )));
          onMessage(`Renamed ${coordinator.confirmed}`);
        }
      } catch (error) {
        if (
          documentIsCurrent(lease, identity)
          && renameGenerationRef.current.get(spot.id) === generation
        ) {
          setSpots((current) => current.map((item) => (
            item.id === spot.id ? { ...item, label: coordinator.confirmed } : item
          )));
          onMessage(error instanceof Error ? error.message : "Failed to update waypoint");
        }
      }
    };
    coordinator.tail = coordinator.tail.catch(() => {}).then(request);
    return coordinator.tail;
  }, [
    cancelRename, captureDocumentLease, documentIsCurrent, editingSpotLabel,
    getCurrentIdentity, markDirty, onMessage, setSpots, updateSpot,
  ]);
  const deleteWaypoint = useCallback(async (spot) => {
    if (!spot || !tryAcquireRouteMutationLock()) return;
    setBusy("Delete Waypoint");
    const routeIds = routeView.treeSpots.map((item) => item.id);
    const lease = captureDocumentLease();
    const identity = { ...getCurrentIdentity() };
    try {
      const paths = localBtPathsForSpot(spot);
      if (!isMissionManifestSpot(spot)) await deleteSpot(spot.id, spot.map_name);
      if (!documentIsCurrent(lease, identity)) return;
      markDirty();
      setSpots((current) => current.filter((item) => item.id !== spot.id));
      if (routeIds.includes(spot.id)) {
        const remaining = routeIds.filter((id) => id !== spot.id);
        const keepClosed = routeView.closed && remaining.length > 1;
        setFlowEdges(missionFlowEdgesForRouteOrder(remaining, keepClosed));
        setRouteSourceAfterExternalMutation(
          !keepClosed && remaining.length > 1 ? remaining[remaining.length - 1] : "",
        );
      }
      ledger.replaceDeletedBtPaths((current) => [...new Set([...current, ...paths])], {
        bumpNonBtEpoch: false,
        markDirty: false,
      });
      forgetTaskSelection(spot.id);
      setSelectedSpotId((current) => (current === spot.id ? "" : current));
      setTaskLayerSpotId((current) => (current === spot.id ? "" : current));
      setEditingSpotId((current) => (current === spot.id ? "" : current));
      setEditingSpotLabel("");
      onMessage(`Deleted ${spot.label || spot.id}`);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Failed to delete waypoint");
    } finally {
      releaseRouteMutationLock();
      setBusy((current) => (current === "Delete Waypoint" ? "" : current));
    }
  }, [
    captureDocumentLease, deleteSpot, documentIsCurrent, forgetTaskSelection,
    getCurrentIdentity, ledger, markDirty, onMessage,
    releaseRouteMutationLock, routeView, setBusy, setFlowEdges,
    setRouteSourceAfterExternalMutation, setSelectedSpotId, setSpots,
    setTaskLayerSpotId, setEditingSpotId, setEditingSpotLabel,
    tryAcquireRouteMutationLock,
  ]);

  return {
    createOnMap,
    createAtRobot,
    moveWaypoint,
    startRename,
    cancelRename,
    commitRename,
    deleteWaypoint,
  };
}
