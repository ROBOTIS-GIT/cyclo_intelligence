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
import { localBtPathsForSpot } from "../../../mapWorkspace/lib/missionBtFiles";
import { missionFlowEdgesForRouteOrder } from "../../../mapWorkspace/lib/missionFlow";
import { isMissionManifestSpot } from "../../../mapWorkspace/lib/missionSpots";

export default function useDesignWaypointDeletion({ route, execution, document, api, persistence, waypoints, selection }) {
  const {
    tryAcquireRouteMutationLock,
    routeView,
    setFlowEdges,
    setRouteSourceAfterExternalMutation,
    releaseRouteMutationLock,
  } = route;
  const { setBusy, onMessage } = execution;
  const { captureDocumentLease, getCurrentIdentity, documentIsCurrent } = document;
  const { deleteSpot } = api;
  const { markDirty, ledger } = persistence;
  const { setSpots, setEditingSpotId, setEditingSpotLabel } = waypoints;
  const { forgetTaskSelection, setSelectedSpotId, setTaskLayerSpotId } = selection;
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

  return deleteWaypoint;
}
