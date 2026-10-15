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
import { addEdge } from '@xyflow/react';
import { PALETTE_DRAG_MIME } from '../../../components/bt/BTNodePalette';
import { isValidBtConnection } from '../../../utils/btConnection';
import { catalogEntryToParams, layoutVisibleOnly } from '../lib/btGraph';

export default function useBtGraphCommands({ graph, nodeCatalog, captureHistory,
  captureEditHistory = captureHistory, setSelectedNodeId, onInvalidConnection }) {
  const { nodesRef, edgesRef, nodeDataMapRef, reactFlowRef,
    setNodes, setEdges, setNodeDataMap } = graph;
  const handleCanvasDragOver = useCallback((event) => {
    if (event.dataTransfer.types.includes(PALETTE_DRAG_MIME)) {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    }
  }, []);

  const handleCanvasDrop = useCallback((event) => {
    const tag =
      event.dataTransfer.getData(PALETTE_DRAG_MIME) ||
      event.dataTransfer.getData('text/plain');
    const meta = nodeCatalog.find((entry) => entry.tag === tag);
    if (!tag || !meta) return;
    event.preventDefault();

    // Convert screen coordinates to ReactFlow canvas coordinates
    const position = reactFlowRef.current
      ? reactFlowRef.current.screenToFlowPosition({ x: event.clientX, y: event.clientY })
      : { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 };

    // Auto-name: {tag}_{n}
    let maxIdx = 0;
    for (const { name } of nodeDataMapRef.current.values()) {
      const m = String(name || '').match(new RegExp(`^${tag}_(\\d+)$`));
      if (m) maxIdx = Math.max(maxIdx, parseInt(m[1], 10));
    }
    const autoName = `${tag}_${maxIdx + 1}`;
    const id = `bt_${Date.now()}`;
    const params = catalogEntryToParams(meta);

    captureHistory();
    const isControl = meta.category === 'control';
    const newNode = {
      id,
      type: isControl ? 'btControl' : 'btAction',
      position,
      // Control nodes carry a collapsed flag so the +/- toggle has somewhere
      // to write. Action nodes don't need it.
      data: isControl
        ? { label: autoName, nodeType: tag, params, collapsed: false }
        : { label: autoName, nodeType: tag, params },
    };
    // Skip auto-dagre on drop. The new node has no edges yet, so dagre
    // treats it as a disconnected component and parks it off to the side
    // — overwriting the cursor coords the user just chose. That makes
    // the sibling-x sort in handleConnect later route it to the end of
    // the parent's children regardless of where the user dropped it.
    // Instead we keep the cursor position; the real re-flow happens when
    // the user wires up the edge (handleConnect), and at that point the
    // dropped x is what feeds into the sibling sort.
    setNodes((prev) => [...prev, newNode]);
    setNodeDataMap((prev) =>
      new Map(prev).set(
        id,
        isControl
          ? { tag, name: autoName, params, collapsed: false }
          : { tag, name: autoName, params },
      )
    );
    setSelectedNodeId(id);
  }, [
    nodeCatalog,
    reactFlowRef,
    captureHistory,
    setNodes,
    setNodeDataMap,
    setSelectedNodeId,
    nodeDataMapRef,
  ]);

  // ── Manual edge connection ────────────────────────────────────────────────
  const handleConnect = useCallback((connection) => {
    // Belt-and-braces alongside the isValidConnection prop: actions are BT
    // leaves — the engine loads children of control nodes only, so an edge
    // out of an action would render but never execute.
    if (!isValidBtConnection(connection, nodesRef.current, edgesRef.current)) {
      onInvalidConnection?.();
      return;
    }
    captureHistory();
    const nextEdges = addEdge(
      { ...connection, type: 'smoothstep', animated: false },
      edgesRef.current
    );
    // After the new edge lands the topology changed, so re-flow nodes around
    // it using the same edge list that will be committed. Keep the source in
    // place: dagre otherwise resets the graph near (0, 0), making the nodes
    // disappear from the user's current viewport.
    const laidOut = layoutVisibleOnly(nodesRef.current, nextEdges, {
      anchorNodeId: connection.source,
    });
    setEdges(nextEdges);
    setNodes(laidOut);
  }, [captureHistory, edgesRef, nodesRef, onInvalidConnection, setEdges, setNodes]);

  const handleAutoLayout = useCallback(() => {
    if (nodesRef.current.length === 0) return;
    captureHistory();
    setNodes(layoutVisibleOnly(nodesRef.current, edgesRef.current));
  }, [captureHistory, edgesRef, nodesRef, setNodes]);

  const handleToggleCollapse = useCallback((nodeId) => {
    const target = nodesRef.current.find((n) => n.id === nodeId);
    if (!target || target.type !== 'btControl') return;
    captureHistory();
    const nextCollapsed = !target.data?.collapsed;
    setNodeDataMap((prev) => {
      const next = new Map(prev);
      const entry = next.get(nodeId);
      if (entry) next.set(nodeId, { ...entry, collapsed: nextCollapsed });
      return next;
    });
    setNodes((ns) => {
      const flipped = ns.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, collapsed: nextCollapsed } }
          : n
      );
      return layoutVisibleOnly(flipped, edgesRef.current);
    });
  }, [nodesRef, captureHistory, setNodeDataMap, setNodes, edgesRef]);

  // ── Node name change: update nodeDataMap.name + nodes[].data.label ────────
  // Empty input is ignored (the inspector resets to the previous value via
  // its localName state reset on selection change).
  const handleNameChange = useCallback((nodeId, newName) => {
    const trimmed = (newName ?? '').trim();
    if (!trimmed) return;
    captureEditHistory(`name:${nodeId}`);
    setNodeDataMap((prev) => {
      const next = new Map(prev);
      const entry = next.get(nodeId);
      if (entry) next.set(nodeId, { ...entry, name: trimmed });
      return next;
    });
    setNodes((ns) =>
      ns.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, label: trimmed } } : n
      )
    );
  }, [captureEditHistory, setNodeDataMap, setNodes]);

  // ── Param change: update nodeDataMap + nodes state ────────────────────────
  const handleParamChange = useCallback((nodeId, paramName, value) => {
    captureEditHistory(`param:${nodeId}:${paramName}`);
    setNodeDataMap((prev) => {
      const next = new Map(prev);
      const entry = next.get(nodeId);
      if (entry) next.set(nodeId, { ...entry, params: { ...entry.params, [paramName]: value } });
      return next;
    });
    setNodes((ns) =>
      ns.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [paramName]: value } } }
          : n
      )
    );
  }, [captureEditHistory, setNodeDataMap, setNodes]);

  return { handleCanvasDragOver, handleCanvasDrop, handleConnect,
    handleAutoLayout, handleToggleCollapse, handleNameChange, handleParamChange };
}
