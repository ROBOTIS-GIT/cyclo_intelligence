// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { useCallback, useMemo, useRef, useState } from "react";
import {
  missionFlowEdgesForRouteOrder,
  syncMissionFlowNodesWithSpots,
} from "../lib/missionFlow";
import { deriveMissionRouteView } from "../lib/missionRouteView";

const noop = () => {};

export default function useMissionRouteEditor({
  spots = [],
  flowNodes = [],
  flowEdges = [],
  setFlowNodes = noop,
  setFlowEdges = noop,
  busy = false,
  documentReady = false,
  markDirty = noop,
  onPrepareEditMode = noop,
  onSelectSpot = noop,
  onMessage = noop,
} = {}) {
  const mutationLockRef = useRef(false);
  const [routeMode, setRouteMode] = useState(false);
  const [routeSourceId, setRouteSourceIdState] = useState("");

  const routeView = useMemo(() => deriveMissionRouteView({
    spots,
    flowNodes,
    flowEdges,
    routeSourceId,
  }), [flowEdges, flowNodes, routeSourceId, spots]);

  const resetEditing = useCallback(() => {
    setRouteMode(false);
    setRouteSourceIdState("");
  }, []);

  const clearSource = useCallback(() => {
    setRouteSourceIdState("");
  }, []);

  // Waypoint deletion is asynchronous and owns the same route document. The
  // lock prevents clicks/reorders from changing its captured route while the
  // backend request is pending.
  const tryAcquireMutationLock = useCallback(() => {
    if (busy || mutationLockRef.current) return false;
    mutationLockRef.current = true;
    return true;
  }, [busy]);

  const releaseMutationLock = useCallback(() => {
    mutationLockRef.current = false;
  }, []);

  const isMutationLocked = useCallback(() => mutationLockRef.current, []);

  const setRouteSourceAfterExternalMutation = useCallback((spotId) => {
    setRouteSourceIdState(String(spotId || ""));
  }, []);

  const setRouteOrder = useCallback((orderedIds) => {
    if (!routeMode || mutationLockRef.current || busy) return false;
    const requestedIds = Array.isArray(orderedIds) ? orderedIds : [];
    const validSpotIds = new Set(spots.map((spot) => spot.id));
    const validIds = requestedIds.filter((id, index) => (
      validSpotIds.has(id) && requestedIds.indexOf(id) === index
    ));
    const currentIds = routeView.treeSpots.map((spot) => spot.id);
    if (
      currentIds.length === validIds.length
      && currentIds.every((id, index) => id === validIds[index])
    ) {
      return false;
    }
    markDirty();
    setFlowNodes((current) => syncMissionFlowNodesWithSpots(current, spots));
    const keepClosed = routeView.closed && validIds.length > 1;
    setFlowEdges(missionFlowEdgesForRouteOrder(validIds, keepClosed));
    setRouteSourceIdState(
      !keepClosed && validIds.length > 1 ? validIds[validIds.length - 1] : "",
    );
    return true;
  }, [busy, markDirty, routeMode, routeView.closed, routeView.treeSpots, setFlowEdges, setFlowNodes, spots]);

  const appendSpot = useCallback((spotId, { select = true } = {}) => {
    if (!routeMode || mutationLockRef.current || busy) return false;
    const spot = spots.find((item) => item.id === spotId);
    if (!spot) return false;
    const currentIds = routeView.treeSpots.map((item) => item.id);
    if (currentIds.includes(spotId)) {
      onMessage(`${spot.label || spot.id} is already in the route`);
      return false;
    }
    if (select) onSelectSpot(spotId);
    if (currentIds.length === 0) {
      setRouteSourceIdState(spotId);
      onMessage(`Route start: ${spot.label || spot.id}`);
      return true;
    }
    if (!setRouteOrder([...currentIds, spotId])) return false;
    onMessage(`${spot.label || spot.id} added to route`);
    return true;
  }, [busy, onMessage, onSelectSpot, routeMode, routeView.treeSpots, setRouteOrder, spots]);

  const toggleMode = useCallback(() => {
    if (mutationLockRef.current || busy) return false;
    if (!documentReady) {
      onMessage("Load a mission before editing mission route");
      return false;
    }
    onPrepareEditMode();
    setRouteMode((value) => {
      const next = !value;
      setRouteSourceIdState("");
      onMessage(next
        ? "Click a waypoint to append it to the mission route"
        : "Mission route editing finished");
      return next;
    });
    return true;
  }, [busy, documentReady, onMessage, onPrepareEditMode]);

  const handleSpotClick = useCallback((spotId) => {
    if (!routeMode || mutationLockRef.current || busy) return false;
    const spot = spots.find((item) => item.id === spotId);
    if (!spot) return false;
    onSelectSpot(spotId);

    const routeIds = routeView.treeSpots.map((item) => item.id);
    const targetIndex = routeIds.indexOf(spotId);
    if (targetIndex < 0) return appendSpot(spotId, { select: false });

    const sourceSpot = spots.find((item) => item.id === routeSourceId);
    const closesOpenRoute = (
      !routeView.closed
      && routeIds.length > 1
      && routeSourceId === routeIds[routeIds.length - 1]
      && targetIndex === 0
    );
    if (closesOpenRoute) {
      markDirty();
      setFlowEdges(missionFlowEdgesForRouteOrder(routeIds, true));
      setRouteSourceIdState("");
      onMessage(
        `Route closed: ${sourceSpot?.label || routeSourceId} -> ${spot.label || spot.id}`,
      );
      return true;
    }

    if (!routeView.closed && targetIndex === routeIds.length - 1) {
      setRouteSourceIdState(spotId);
      onMessage(`Route end: ${spot.label || spot.id}`);
      return true;
    }

    onMessage(`${spot.label || spot.id} is already in the route`);
    return false;
  }, [appendSpot, busy, markDirty, onMessage, onSelectSpot, routeMode, routeSourceId, routeView.closed, routeView.treeSpots, setFlowEdges, spots]);

  const handleMapClick = useCallback(() => {
    if (!routeMode) return false;
    setRouteSourceIdState("");
    onMessage("Route selection cleared");
    return true;
  }, [onMessage, routeMode]);

  const clearRoute = useCallback(() => {
    if (!routeMode || mutationLockRef.current || busy || !flowEdges.length) return false;
    markDirty();
    setFlowEdges([]);
    setRouteSourceIdState("");
    onMessage("Route cleared");
    return true;
  }, [busy, flowEdges.length, markDirty, onMessage, routeMode, setFlowEdges]);

  const openLoop = useCallback(() => {
    if (!routeMode || mutationLockRef.current || busy || !routeView.closed) return false;
    const routeIds = routeView.treeSpots.map((spot) => spot.id);
    if (routeIds.length < 2) return false;
    markDirty();
    setFlowEdges(missionFlowEdgesForRouteOrder(routeIds, false));
    setRouteSourceIdState(routeIds[routeIds.length - 1]);
    onMessage("Loop opened");
    return true;
  }, [busy, markDirty, onMessage, routeMode, routeView.closed, routeView.treeSpots, setFlowEdges]);

  const moveSpot = useCallback((spotId, direction) => {
    if (!routeMode || mutationLockRef.current || busy) return false;
    const currentIds = routeView.treeSpots.map((spot) => spot.id);
    const index = currentIds.indexOf(spotId);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= currentIds.length) return false;
    const nextIds = [...currentIds];
    [nextIds[index], nextIds[nextIndex]] = [nextIds[nextIndex], nextIds[index]];
    return setRouteOrder(nextIds);
  }, [busy, routeMode, routeView.treeSpots, setRouteOrder]);

  const removeSpot = useCallback((spotId) => {
    if (!routeMode || mutationLockRef.current || busy) return false;
    const currentIds = routeView.treeSpots.map((spot) => spot.id);
    if (!currentIds.includes(spotId)) return false;
    const spot = routeView.treeSpots.find((item) => item.id === spotId);
    if (!setRouteOrder(currentIds.filter((id) => id !== spotId))) return false;
    onMessage(`${spot?.label || spotId} removed from route`);
    return true;
  }, [busy, onMessage, routeMode, routeView.treeSpots, setRouteOrder]);

  return {
    routeMode,
    routeSourceId,
    routeView,
    setRouteOrder,
    appendSpot,
    toggleMode,
    handleSpotClick,
    handleMapClick,
    clearRoute,
    openLoop,
    moveSpot,
    removeSpot,
    resetEditing,
    clearSource,
    tryAcquireMutationLock,
    releaseMutationLock,
    isMutationLocked,
    setRouteSourceAfterExternalMutation,
  };
}
