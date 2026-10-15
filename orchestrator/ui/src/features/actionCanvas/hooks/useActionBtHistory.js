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

import { useCallback } from 'react';
import { useBTHistory } from '../../../hooks/useBTHistory';
import { setTreeFileName, setSelectedNodeId } from '../actionCanvasSlice';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';

export default function useActionBtHistory({ graph, treeFileName, dispatch }) {
  const { nodes, edges, nodeDataMap, setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const getHistorySnapshot = useCallback(() => {
    return JSON.stringify({
      nodes: nodes.map(({ data: { isActive: _a, isSelected: _s, ...d }, ...n }) => ({
        ...n,
        data: d,
      })),
      edges,
      nodeDataMap: [...nodeDataMap.entries()],
      treeFileName,
    });
  }, [nodes, edges, nodeDataMap, treeFileName]);

  const applyHistorySnapshot = useCallback((snap) => {
    try {
      const {
        nodes: n,
        edges: e,
        nodeDataMap: ndm,
        treeFileName: restoredFileName = '',
      } = JSON.parse(snap);
      setNodes(n);
      setEdges(e);
      setNodeDataMap(new Map(ndm));
      setParseError(null);
      dispatch(setSelectedNodeId(null));
      dispatch(setTreeFileName(restoredFileName));
    } catch (err) {
      setParseError(formatTaskDisplayMessage(err.message));
    }
  }, [setNodes, setEdges, setNodeDataMap, setParseError, dispatch]);

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

  return { captureHistory, undoHistory, redoHistory, resetHistory,
    canUndo, canRedo, getHistorySnapshot };
}
