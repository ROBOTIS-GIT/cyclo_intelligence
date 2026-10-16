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
  localBtPathForSpot,
  localBtPathsForSpot,
  withLocalBtLibrary,
} from "../../../mapWorkspace/lib/missionBtFiles";
import { isMissionManifestSpot, spotPoseFromMapPose } from "../../../mapWorkspace/lib/missionSpots";

export default function useDesignWaypointEdits({ waypoints, document, coordinator, persistence, execution, api, selection }) {
  const { spots, setSpots, setEditingSpotId, setEditingSpotLabel, editingSpotLabel } = waypoints;
  const { captureDocumentLease, getCurrentIdentity, documentIsCurrent } = document;
  const { mutationQueue } = coordinator;
  const { markDirty } = persistence;
  const { onMessage } = execution;
  const { updateSpot } = api;
  const { setSelectedSpotId, clearBehaviorSelection, setShowWaypointOptions, setInteractionMode } = selection;
  const moveWaypoint = useCallback((spotId, x, y, yaw) => {
    const spot = spots.find((item) => item.id === spotId);
    if (!spot) return Promise.resolve();
    const nextPose = spotPoseFromMapPose(x, y, yaw ?? spot.pose?.yaw ?? 0);
    const lease = captureDocumentLease();
    const identity = { ...getCurrentIdentity() };
    const mutation = mutationQueue.capture("move", spotId, lease, identity, spot.pose);
    markDirty();
    setSpots((current) => current.map((item) => (
      item.id === spotId ? { ...item, pose: nextPose } : item
    )));
    if (isMissionManifestSpot(spot)) {
      mutation.confirm(nextPose);
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
        mutation.confirm(updated.pose || nextPose);
        if (mutation.isLatest()) {
          setSpots((current) => current.map((item) => (
            item.id === spotId
              ? {
                ...item,
                pose: mutation.confirmed(),
                metadata: { ...(item.metadata ?? {}), ...(updated.metadata ?? {}) },
              }
              : item
          )));
          onMessage(`Moved ${spot.label || spot.id}`);
        }
      } catch (error) {
        if (
          documentIsCurrent(lease, identity)
          && mutation.isLatest()
        ) {
          setSpots((current) => current.map((item) => (
            item.id === spotId ? { ...item, pose: mutation.confirmed() } : item
          )));
          onMessage(error instanceof Error ? error.message : "Failed to move waypoint");
        }
      }
    };
    return mutation.enqueue(request);
  }, [
    captureDocumentLease,
    documentIsCurrent,
    getCurrentIdentity,
    markDirty,
    mutationQueue,
    onMessage,
    setSpots,
    spots,
    updateSpot,
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
    const mutation = mutationQueue.capture("rename", spot.id, lease, identity, spot.label);
    markDirty();
    setSpots((current) => current.map((item) => (
      item.id === spot.id ? { ...item, label } : item
    )));
    if (isMissionManifestSpot(spot)) {
      mutation.confirm(label);
      onMessage(`Renamed ${label}`);
      return Promise.resolve();
    }
    const request = async () => {
      if (!documentIsCurrent(lease, identity)) return;
      try {
        const updated = await updateSpot(spot.id, { map_name: spot.map_name, label });
        if (!documentIsCurrent(lease, identity)) return;
        mutation.confirm(updated.label || label);
        if (mutation.isLatest()) {
          setSpots((current) => current.map((item) => (
            item.id === spot.id
              ? withLocalBtLibrary({
                ...item,
                label: mutation.confirmed(),
                metadata: { ...(item.metadata ?? {}), ...(updated.metadata ?? {}) },
              }, localBtPathForSpot(item), localBtPathsForSpot(item))
              : item
          )));
          onMessage(`Renamed ${mutation.confirmed()}`);
        }
      } catch (error) {
        if (
          documentIsCurrent(lease, identity)
          && mutation.isLatest()
        ) {
          setSpots((current) => current.map((item) => (
            item.id === spot.id ? { ...item, label: mutation.confirmed() } : item
          )));
          onMessage(error instanceof Error ? error.message : "Failed to update waypoint");
        }
      }
    };
    return mutation.enqueue(request);
  }, [
    cancelRename,
    captureDocumentLease,
    documentIsCurrent,
    editingSpotLabel,
    getCurrentIdentity,
    markDirty,
    mutationQueue,
    onMessage,
    setSpots,
    updateSpot,
  ]);
  return { moveWaypoint, startRename, cancelRename, commitRename };
}
