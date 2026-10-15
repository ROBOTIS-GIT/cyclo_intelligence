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

import { useEffect } from 'react';
import { findDeletionLayoutAnchor } from '../../../utils/btTreeParser';
import { layoutVisibleOnly } from '../lib/btGraph';

export default function useBtGraphDeletion({ graph, captureHistory, selectedNodeId,
  setSelectedNodeId, includeInspectorSelection = false }) {
  const { nodesRef, edgesRef, setNodes, setEdges, setNodeDataMap } = graph;
  useEffect(() => {
    const handler = (e) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;
      const selectedNodeIds = new Set(
        currentNodes.filter((n) => n.selected || (includeInspectorSelection && n.id === selectedNodeId)).map((n) => n.id)
      );
      const selectedEdgeIds = new Set(
        currentEdges.filter((eg) => eg.selected).map((eg) => eg.id)
      );
      if (selectedNodeIds.size === 0 && selectedEdgeIds.size === 0) return;

      captureHistory();
      const remainingNodes = currentNodes.filter((n) => !selectedNodeIds.has(n.id));
      const remainingEdges = currentEdges.filter(
        (eg) =>
          !selectedEdgeIds.has(eg.id) &&
          !selectedNodeIds.has(eg.source) &&
          !selectedNodeIds.has(eg.target)
      );
      // Re-flow what's left so the deleted node/edge's old slot doesn't
      // leave a visible gap in the tree. Keep a surviving neighbor (or the
      // first remaining node for a disconnected deletion) at its old canvas
      // coordinates so dagre cannot move the graph back near (0, 0).
      const anchorNodeId = findDeletionLayoutAnchor(
        currentNodes,
        currentEdges,
        selectedNodeIds,
        selectedEdgeIds,
      );
      setNodes(layoutVisibleOnly(remainingNodes, remainingEdges, { anchorNodeId }));
      setEdges(remainingEdges);
      if (selectedNodeIds.size > 0) {
        setNodeDataMap((prev) => {
          const next = new Map(prev);
          selectedNodeIds.forEach((id) => next.delete(id));
          return next;
        });
      }
      if (selectedNodeIds.has(selectedNodeId)) {
        setSelectedNodeId(null);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [
    setNodes,
    setEdges,
    setSelectedNodeId,
    captureHistory,
    selectedNodeId,
    includeInspectorSelection,
    nodesRef,
    edgesRef,
    setNodeDataMap,
  ]);

}
