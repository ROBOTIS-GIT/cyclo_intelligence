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

import { useRef, useCallback } from 'react';
import { useBTHistory } from '../../../hooks/useBTHistory';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';

export default function useMissionBtHistory({ graph, setSelectedNodeId }) {
  const { nodes, edges, nodeDataMap, setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const coalescedEditRef = useRef({ key: '', lastChangeAt: 0 });
  const getHistorySnapshot = useCallback(() => {
    return JSON.stringify({
      nodes: nodes.map(({ data: { isActive: _active, isSelected: _selected, ...data }, ...node }) => ({
        ...node,
        data,
      })),
      edges,
      nodeDataMap: [...nodeDataMap.entries()],
    });
  }, [edges, nodeDataMap, nodes]);

  const applyHistorySnapshot = useCallback((snapshot) => {
    try {
      const parsed = JSON.parse(snapshot);
      setNodes(parsed.nodes || []);
      setEdges(parsed.edges || []);
      setNodeDataMap(new Map(parsed.nodeDataMap || []));
      setSelectedNodeId(null);
      setParseError(null);
      coalescedEditRef.current = { key: "", lastChangeAt: 0 };
    } catch (error) {
      setParseError(error instanceof Error
        ? formatTaskDisplayMessage(error.message, "Waypoint Task")
        : "Failed to restore history");
    }
  }, [setEdges, setNodeDataMap, setNodes, setParseError, setSelectedNodeId]);

  const {
    capture: captureHistory,
    undo: undoHistory,
    redo: redoHistory,
    reset: resetHistory,
    canUndo,
    canRedo,
  } = useBTHistory({
    getSnapshot: getHistorySnapshot,
    applySnapshot: applyHistorySnapshot,
  });

  const captureCoalescedEditHistory = useCallback((key) => {
    const now = Date.now();
    const previous = coalescedEditRef.current;
    if (previous.key !== key || now - previous.lastChangeAt > 750) {
      captureHistory();
    }
    coalescedEditRef.current = { key, lastChangeAt: now };
  }, [captureHistory]);

  return { captureHistory, undoHistory, redoHistory, resetHistory, canUndo, canRedo,
    getHistorySnapshot, captureCoalescedEditHistory, coalescedEditRef };
}
