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
  canonicalLocalBtPathForSpot,
  canonicalLocalBtPathsForSpot,
  localBtDirectoriesForSpots,
  localBtPathForSpot,
  localBtPathsForSpot,
  withLocalBtLibrary,
} from "../../mapWorkspace/lib/missionBtFiles";
import { DEFAULT_MAP_NAME } from "../../mapWorkspace/lib/missionNames";

export default function useDesignDocumentContentController({
  setMetadata,
  applySpots,
  updateSpots,
  setFlowNodes,
  setFlowEdges,
  mergeBehaviorMapPatch,
  commitLedgerSnapshot,
  resetLedgerNewDocument,
  resetHistory,
  resetLocalBtSelections,
  clearBehaviorSelection,
  clearPendingBehaviorPlacement,
  resetRouteEditing,
  setEditingSpotId,
  setEditingSpotLabel,
  setLoadError,
}) {
  const resetTransientSelections = useCallback(() => {
    resetLocalBtSelections();
    clearBehaviorSelection();
    clearPendingBehaviorPlacement();
  }, [clearBehaviorSelection, clearPendingBehaviorPlacement, resetLocalBtSelections]);

  const commitLoadedSnapshot = useCallback((snapshot = {}) => {
    const spots = Array.isArray(snapshot?.spots) ? snapshot.spots : [];
    const flowNodes = Array.isArray(snapshot?.flowNodes) ? snapshot.flowNodes : [];
    const flowEdges = Array.isArray(snapshot?.flowEdges) ? snapshot.flowEdges : [];
    const btFiles = snapshot?.btFiles && typeof snapshot.btFiles === "object"
      && !Array.isArray(snapshot.btFiles)
      ? { ...snapshot.btFiles }
      : {};
    const baseline = snapshot?.baseline && typeof snapshot.baseline === "object"
      ? snapshot.baseline
      : {};

    if (Array.isArray(snapshot?.behaviorNodesPatch)) {
      mergeBehaviorMapPatch(
        snapshot?.identity?.mapName || DEFAULT_MAP_NAME,
        snapshot.behaviorNodesPatch,
      );
    }
    setMetadata?.({ ...(snapshot.metadata || {}) });
    applySpots(spots);
    setFlowNodes(flowNodes);
    setFlowEdges(flowEdges);
    commitLedgerSnapshot({ btFiles, baseline });
    resetHistory();
    resetTransientSelections();
    return { spots, flowNodes, flowEdges, btFiles, baseline };
  }, [
    setMetadata,
    applySpots,
    commitLedgerSnapshot,
    mergeBehaviorMapPatch,
    resetHistory,
    resetTransientSelections,
    setFlowEdges,
    setFlowNodes,
  ]);

  const resetNewDocument = useCallback(({ btFiles = {} } = {}) => {
    setMetadata?.({});
    applySpots([]);
    setFlowNodes([]);
    setFlowEdges([]);
    resetLocalBtSelections();
    resetLedgerNewDocument({ btFiles });
    resetHistory();
    clearBehaviorSelection();
    clearPendingBehaviorPlacement();
    resetRouteEditing();
    setEditingSpotId("");
    setEditingSpotLabel("");
    setLoadError("");
  }, [
    setMetadata,
    applySpots,
    clearBehaviorSelection,
    clearPendingBehaviorPlacement,
    resetHistory,
    resetLedgerNewDocument,
    resetLocalBtSelections,
    resetRouteEditing,
    setEditingSpotId,
    setEditingSpotLabel,
    setFlowEdges,
    setFlowNodes,
    setLoadError,
  ]);

  const applySavedCanonicalSpots = useCallback((canonicalSpots) => {
    if (!Array.isArray(canonicalSpots)) {
      throw new TypeError("Saved canonical spots must be an array");
    }
    const canonicalDirectories = localBtDirectoriesForSpots(canonicalSpots);
    updateSpots((current) => current.map((spot) => {
      const savedSpot = canonicalSpots.find(({ id }) => id === spot.id);
      if (savedSpot) {
        return withLocalBtLibrary(
          spot,
          localBtPathForSpot(savedSpot),
          localBtPathsForSpot(savedSpot),
        );
      }
      const directory = canonicalDirectories.get(spot.id);
      return withLocalBtLibrary(
        spot,
        canonicalLocalBtPathForSpot(spot, directory),
        canonicalLocalBtPathsForSpot(spot, directory),
      );
    }));
  }, [updateSpots]);

  return {
    commitLoadedSnapshot,
    resetNewDocument,
    applySavedCanonicalSpots,
    resetTransientSelections,
  };
}
