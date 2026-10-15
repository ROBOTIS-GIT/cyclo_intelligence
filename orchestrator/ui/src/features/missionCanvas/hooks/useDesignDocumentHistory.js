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
import { useBTHistory } from "../../../hooks/useBTHistory";

export default function useDesignDocumentHistory({
  spots, getBehaviorHistorySlice, missionMetadata, missionFlowNodes, missionFlowEdges,
  selectedSpotId, getDesignLedgerHistorySlice, setSpots, applyMissionMetadata,
  restoreBehaviorHistorySlice, restoreDesignLedgerHistorySlice, setMissionFlowNodes,
  setMissionFlowEdges, setSelectedSpotId, clearMissionRouteSourceRef, setEditingSpotId,
  setEditingSpotLabel, setBtLayerSpotId, setInteractionMode, setShowWaypointOptions, setMessage,
  behaviorMarkDirtyRef, markNonBtDirty,
}) {
  const getDesignHistorySnapshot = useCallback(() => JSON.stringify({
    spots,
    ...getBehaviorHistorySlice(),
    missionMetadata,
    missionFlowNodes,
    missionFlowEdges,
    selectedSpotId,
    ...getDesignLedgerHistorySlice(),
  }), [
    getBehaviorHistorySlice,
    missionMetadata,
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
      applyMissionMetadata(restored.missionMetadata || {});
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
  }, [
    applyMissionMetadata, clearMissionRouteSourceRef, restoreBehaviorHistorySlice,
    restoreDesignLedgerHistorySlice, setBtLayerSpotId, setEditingSpotId, setEditingSpotLabel,
    setInteractionMode, setMessage, setMissionFlowEdges, setMissionFlowNodes, setSelectedSpotId,
    setShowWaypointOptions, setSpots,
  ]);

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

  return {
    getDesignHistorySnapshot, captureDesignHistory, undoDesignHistory, redoDesignHistory,
    resetDesignHistory, rebaseDesignHistory, canUndoDesign, canRedoDesign,
  };
}
