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

import { useCallback, useEffect } from 'react';
import toast from 'react-hot-toast';

export default function useMissionBtClear({ graph, history, confirmation,
  loading, fileActionsDisabled, fileAction, hydratedPath, filePath, setSelectedNodeId }) {
  const { nodes, nodesRef, parseError, setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const { captureHistory, getHistorySnapshot, coalescedEditRef } = history;
  const { clearTreeArmed, disarmClearTree, armClearTree, clearTreeTargetRef } = confirmation;
  useEffect(() => {
    if (!clearTreeArmed) return;
    if (
      nodes.length === 0
      || loading
      || fileActionsDisabled
      || Boolean(fileAction)
      || Boolean(parseError)
      || hydratedPath !== filePath
    ) {
      disarmClearTree();
    }
  }, [
    clearTreeArmed,
    disarmClearTree,
    fileAction,
    fileActionsDisabled,
    filePath,
    hydratedPath,
    loading,
    nodes.length,
    parseError,
  ]);

  const handleClearTree = useCallback(() => {
    if (
      nodesRef.current.length === 0
      || loading
      || fileActionsDisabled
      || Boolean(fileAction)
      || Boolean(parseError)
      || hydratedPath !== filePath
    ) {
      return;
    }

    const currentSnapshot = getHistorySnapshot();
    if (!clearTreeArmed) {
      armClearTree(currentSnapshot);
      return;
    }
    if (clearTreeTargetRef.current !== currentSnapshot) {
      armClearTree(currentSnapshot);
      return;
    }

    captureHistory();
    setNodes([]);
    setEdges([]);
    setNodeDataMap(new Map());
    setSelectedNodeId(null);
    setParseError(null);
    coalescedEditRef.current = { key: "", lastChangeAt: 0 };
    disarmClearTree();
    toast.success("Waypoint task cleared");
  }, [
    armClearTree,
    captureHistory,
    clearTreeArmed,
    clearTreeTargetRef,
    coalescedEditRef,
    disarmClearTree,
    fileAction,
    fileActionsDisabled,
    filePath,
    getHistorySnapshot,
    hydratedPath,
    loading,
    nodesRef,
    parseError,
    setEdges,
    setNodeDataMap,
    setNodes,
    setParseError,
    setSelectedNodeId,
  ]);

  return handleClearTree;
}
