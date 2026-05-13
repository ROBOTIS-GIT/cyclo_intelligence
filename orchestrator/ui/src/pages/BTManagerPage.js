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

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  ReactFlow,
  Controls,
  Background,
  addEdge,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdPlayArrow, MdStop, MdUploadFile, MdSave, MdUndo, MdRedo, MdAutoFixHigh } from 'react-icons/md';

import BTControlNode from '../components/bt/BTControlNode';
import BTActionNode from '../components/bt/BTActionNode';
import BTParamPanel from '../components/bt/BTParamPanel';
import BTNodePalette, { PALETTE_DRAG_MIME } from '../components/bt/BTNodePalette';
import TreeListModal from '../features/btmanager/components/TreeListModal';
import { parseBTXml, applyDagreLayout } from '../utils/btTreeParser';
import { serializeFromGraph } from '../utils/btXmlSerializer';
import { setTreeXml, setTreeFileName, setBtStatus, setActiveNodeNames, setSelectedNodeId } from '../features/btmanager/btmanagerSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { useBTHistory } from '../hooks/useBTHistory';
import { findNodeMeta, isControlTag } from '../constants/btNodeCatalog';

const nodeTypes = {
  btControl: BTControlNode,
  btAction: BTActionNode,
};

// BFS down the edges to enumerate every node reachable from `rootId`.
// Used to mark a collapsed Control node's whole subtree as hidden.
function collectDescendants(rootId, edges) {
  const out = new Set();
  const queue = [rootId];
  while (queue.length) {
    const id = queue.shift();
    for (const e of edges) {
      if (e.source === id && !out.has(e.target)) {
        out.add(e.target);
        queue.push(e.target);
      }
    }
  }
  return out;
}

// Walk the graph and return the set of node ids that should be rendered
// hidden because some ancestor Control node is collapsed.
function computeHiddenIds(nodes, edges) {
  const hidden = new Set();
  for (const n of nodes) {
    if (n.type === 'btControl' && n.data && n.data.collapsed) {
      for (const id of collectDescendants(n.id, edges)) hidden.add(id);
    }
  }
  return hidden;
}

// Run dagre over just the visible slice of the tree, then splice the
// resulting positions back into the full nodes array. Hidden nodes keep
// their old coords so they sit ready underneath the collapsed parent for
// when the user expands it again.
function layoutVisibleOnly(nodes, edges) {
  const hidden = computeHiddenIds(nodes, edges);
  const visibleNodes = nodes.filter((n) => !hidden.has(n.id));
  const visibleEdges = edges.filter(
    (e) => !hidden.has(e.source) && !hidden.has(e.target)
  );
  const laid = applyDagreLayout(visibleNodes, visibleEdges, { respectStored: false });
  const byId = new Map(laid.nodes.map((n) => [n.id, n]));
  return nodes.map((n) => (byId.has(n.id) ? byId.get(n.id) : n));
}

export default function BTManagerPage({ isActive = true }) {
  const dispatch = useDispatch();
  const { callService } = useRosServiceCaller();
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);

  const treeXml = useSelector((state) => state.btmanager.treeXml);
  const treeFileName = useSelector((state) => state.btmanager.treeFileName);
  const btStatus = useSelector((state) => state.btmanager.btStatus);
  const activeNodeNames = useSelector((state) => state.btmanager.activeNodeNames);
  const selectedNodeId = useSelector((state) => state.btmanager.selectedNodeId);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  // nodeDataMap: Map<id, {tag, name, params}> — primary source of truth for node content
  const [nodeDataMap, setNodeDataMap] = useState(new Map());
  const [parseError, setParseError] = useState(null);
  const [showTreeList, setShowTreeList] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveFileName, setSaveFileName] = useState('');

  // ReactFlow instance for coordinate conversion on drop
  const reactFlowRef = useRef(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const nodeDataMapRef = useRef(nodeDataMap);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  nodeDataMapRef.current = nodeDataMap;

  // ── History ──────────────────────────────────────────────────────────────
  // Snapshots are JSON strings encoding {nodes, edges, nodeDataMap}.
  // isActive / isSelected are annotation-only and excluded.

  const getHistorySnapshot = useCallback(() => {
    if (nodes.length === 0) return null;
    return JSON.stringify({
      nodes: nodes.map(({ data: { isActive: _a, isSelected: _s, ...d }, ...n }) => ({
        ...n,
        data: d,
      })),
      edges,
      nodeDataMap: [...nodeDataMap.entries()],
    });
  }, [nodes, edges, nodeDataMap]);

  const applyHistorySnapshot = useCallback((snap) => {
    try {
      const { nodes: n, edges: e, nodeDataMap: ndm } = JSON.parse(snap);
      setNodes(n);
      setEdges(e);
      setNodeDataMap(new Map(ndm));
      setParseError(null);
      dispatch(setSelectedNodeId(null));
    } catch (err) {
      setParseError(err.message);
    }
  }, [setNodes, setEdges, dispatch]);

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

  // ── Initial load from Redux treeXml (e.g. on page mount) ─────────────────
  useEffect(() => {
    if (!treeXml) {
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
      setParseError(null);
      return;
    }
    try {
      const { nodes: n, edges: e, nodeDataMap: ndm } = parseBTXml(treeXml);
      setNodes(n);
      setEdges(e);
      setNodeDataMap(ndm);
      setParseError(null);
    } catch (err) {
      setParseError(err.message);
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount to restore Redux-persisted tree

  // ── Persist working tree to Redux so it survives page switches ────────────
  // The graph state (nodes/edges/nodeDataMap) lives in local useState, which
  // is torn down on unmount. Without this, navigating away and back to the BT
  // Manager wipes the user's in-progress tree. We serialise on a small debounce
  // so a flurry of edits doesn't dispatch on every keystroke, and again on
  // unmount to catch whatever's still in the debounce window.
  useEffect(() => {
    if (nodes.length === 0) return;
    const t = setTimeout(() => {
      try {
        dispatch(setTreeXml(serializeFromGraph(nodes, edges, nodeDataMap)));
      } catch {
        // Partial graphs (e.g. mid-drag, disconnected nodes) can throw here.
        // Drop the snapshot rather than nuking the previously-good treeXml.
      }
    }, 400);
    return () => clearTimeout(t);
  }, [nodes, edges, nodeDataMap, dispatch]);

  useEffect(() => {
    return () => {
      const n = nodesRef.current;
      const e = edgesRef.current;
      const m = nodeDataMapRef.current;
      if (n.length === 0) return;
      try {
        dispatch(setTreeXml(serializeFromGraph(n, e, m)));
      } catch {
        // Same swallow as the debounced path — preserve last good state.
      }
    };
  }, [dispatch]);

  // ── Handle tree selection from TreeListModal ──────────────────────────────
  const handleServerFileSelect = useCallback(async (item) => {
    if (!item || !item.full_path) return;
    try {
      const urlMatch = rosbridgeUrl.match(/ws:\/\/([^:]+):/);
      const host = urlMatch ? urlMatch[1] : 'localhost';
      const fileUrl = `http://${host}:8082${item.full_path}`;
      const response = await fetch(fileUrl);
      if (!response.ok) throw new Error(`Failed to fetch file: ${response.status}`);

      const xmlContent = await response.text();
      const fileName = item.name || item.full_path.split('/').pop();

      const { nodes: n, edges: e, nodeDataMap: ndm } = parseBTXml(xmlContent);
      setNodes(n);
      setEdges(e);
      setNodeDataMap(ndm);
      setParseError(null);

      resetHistory();
      dispatch(setSelectedNodeId(null));
      dispatch(setTreeXml(xmlContent));
      dispatch(setTreeFileName(fileName));
      toast.success(`Loaded: ${fileName}`);
    } catch (err) {
      toast.error(`Failed to load file: ${err.message}`);
    }
  }, [rosbridgeUrl, dispatch, setNodes, setEdges, resetHistory]);

  // ── Node click handler ────────────────────────────────────────────────────
  const handleNodeClick = useCallback((event, node) => {
    dispatch(setSelectedNodeId(node.id));
  }, [dispatch]);

  // ── Drag-and-drop from palette: drop anywhere to create a disconnected node
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
    if (!tag || !findNodeMeta(tag)) return;
    event.preventDefault();

    // Convert screen coordinates to ReactFlow canvas coordinates
    const position = reactFlowRef.current
      ? reactFlowRef.current.screenToFlowPosition({ x: event.clientX, y: event.clientY })
      : { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 };

    // Auto-name: {tag}_{n}
    let maxIdx = 0;
    for (const { name } of nodeDataMapRef.current.values()) {
      const m = name.match(new RegExp(`^${tag}_(\\d+)$`));
      if (m) maxIdx = Math.max(maxIdx, parseInt(m[1], 10));
    }
    const autoName = `${tag}_${maxIdx + 1}`;
    const id = `bt_${Date.now()}`;
    const meta = findNodeMeta(tag);
    const params = meta ? { ...meta.params } : {};

    captureHistory();
    const isControl = isControlTag(tag);
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
    dispatch(setSelectedNodeId(id));
  }, [captureHistory, setNodes, dispatch]);

  // ── Manual edge connection ────────────────────────────────────────────────
  const handleConnect = useCallback((connection) => {
    captureHistory();
    setEdges((prev) => {
      const nextEdges = addEdge({ ...connection, type: 'smoothstep', animated: false }, prev);
      // After the new edge lands the topology changed, so re-flow nodes
      // around it. setNodes runs separately because we need the latest
      // edge list to compute the layout.
      const laidOut = layoutVisibleOnly(nodesRef.current, nextEdges);
      setNodes(laidOut);
      return nextEdges;
    });
  }, [captureHistory, setEdges, setNodes]);

  // ── Node drag stop: just capture history (ReactFlow updates position) ─────
  const handleNodeDragStop = useCallback(() => {
    captureHistory();
  }, [captureHistory]);

  // ── Manual auto-layout (toolbar button) ───────────────────────────────────
  // respectStored:false discards both XML-loaded coords and any manual drags
  // so a fresh dagre pass wins. Undo restores the prior coords because we
  // capture history first. Hidden nodes (under collapsed parents) skip
  // layout so they don't get re-positioned out from under the user.
  const handleAutoLayout = useCallback(() => {
    if (nodesRef.current.length === 0) return;
    captureHistory();
    setNodes(layoutVisibleOnly(nodesRef.current, edgesRef.current));
  }, [captureHistory, setNodes]);

  // ── Collapse/expand toggle on Control nodes ───────────────────────────────
  // Flips data.collapsed on the target Control node in both nodes[] and
  // nodeDataMap, then re-flows the now-visible slice so the layout closes
  // up the gap (collapse) or fans the children back out (expand).
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
  }, [setNodes, captureHistory]);

  // ── Node name change: update nodeDataMap.name + nodes[].data.label ────────
  // Empty input is ignored (the inspector resets to the previous value via
  // its localName state reset on selection change).
  const handleNameChange = useCallback((nodeId, newName) => {
    const trimmed = (newName ?? '').trim();
    if (!trimmed) return;
    captureHistory();
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
  }, [setNodes, captureHistory]);

  // ── Param change: update nodeDataMap + nodes state ────────────────────────
  const handleParamChange = useCallback((nodeId, paramName, value) => {
    captureHistory();
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
  }, [setNodes, captureHistory]);

  // ── Delete key: remove selected nodes and/or edges ────────────────────────
  // ReactFlow's default onEdgesChange manages `edge.selected` for us, so we
  // just have to read both selection flags here. Edge-only and mixed
  // selections are handled in a single transaction so undo restores both
  // at once.
  useEffect(() => {
    const handler = (e) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;
      const selectedNodeIds = new Set(
        currentNodes.filter((n) => n.selected).map((n) => n.id)
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
      // leave a visible gap in the tree.
      setNodes(layoutVisibleOnly(remainingNodes, remainingEdges));
      setEdges(remainingEdges);
      if (selectedNodeIds.size > 0) {
        setNodeDataMap((prev) => {
          const next = new Map(prev);
          selectedNodeIds.forEach((id) => next.delete(id));
          return next;
        });
      }
      if (selectedNodeIds.has(selectedNodeId)) {
        dispatch(setSelectedNodeId(null));
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [setNodes, setEdges, dispatch, captureHistory, selectedNodeId]);

  // ── Undo/redo keybindings ─────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (!(e.ctrlKey || e.metaKey)) return;
      const key = e.key.toLowerCase();
      if (key === 'z') {
        e.preventDefault();
        if (e.shiftKey) redoHistory();
        else undoHistory();
      } else if (key === 'y' && !e.shiftKey) {
        e.preventDefault();
        redoHistory();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [undoHistory, redoHistory]);

  // ── HTTP base URL helper ──────────────────────────────────────────────────
  const getHttpBaseUrl = useCallback(() => {
    const urlMatch = rosbridgeUrl.match(/ws:\/\/([^:]+):/);
    const host = urlMatch ? urlMatch[1] : 'localhost';
    return `http://${host}:8082`;
  }, [rosbridgeUrl]);

  // ── Serialize current graph to BT XML ────────────────────────────────────
  const getSerializedXml = useCallback(() => {
    return serializeFromGraph(nodes, edges, nodeDataMap);
  }, [nodes, edges, nodeDataMap]);

  // ── Save As ───────────────────────────────────────────────────────────────
  const handleSaveAs = useCallback(async () => {
    const name = saveFileName.trim();
    if (!name) return;

    const content = getSerializedXml();
    if (!content) return;

    try {
      const baseUrl = getHttpBaseUrl();
      const res = await fetch(`${baseUrl}/bt/save_tree`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: name, content }),
      });
      const data = await res.json();
      if (data.success) {
        toast.success(data.message);
        setShowSaveDialog(false);
        setSaveFileName('');
      } else {
        toast.error(data.message || 'Save failed');
      }
    } catch (err) {
      toast.error(`Save failed: ${err.message}`);
    }
  }, [saveFileName, getSerializedXml, getHttpBaseUrl]);

  // ── BT Start ──────────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (nodes.length === 0) {
      toast.error('No tree loaded');
      return;
    }
    try {
      const currentXml = getSerializedXml();

      const baseUrl = getHttpBaseUrl();
      const launchRes = await fetch(`${baseUrl}/bt/launch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const launchData = await launchRes.json();
      if (!launchData.success) {
        toast.error(`Failed to launch BT node: ${launchData.message}`);
        return;
      }

      const isAlreadyRunning = launchData.message.includes('already running');
      if (!isAlreadyRunning) {
        await new Promise((resolve) => setTimeout(resolve, 8000));
      }

      const result = await callService(
        '/bt/load_and_run',
        'interfaces/srv/LoadAndRunTree',
        { tree_xml: currentXml },
        30000
      );
      if (result.success) {
        dispatch(setBtStatus('running'));
        dispatch(setSelectedNodeId(null));
        toast.success('BT started');
      } else {
        toast.error(`Failed: ${result.message}`);
      }
    } catch (err) {
      toast.error(`Failed to start BT: ${err.message}`);
    }
  }, [callService, dispatch, nodes.length, getSerializedXml, getHttpBaseUrl]);

  // ── BT Stop ───────────────────────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    try {
      try {
        await callService('/bt/set_running', 'std_srvs/srv/SetBool', { data: false });
      } catch {
        // BT node may already be gone
      }
      const baseUrl = getHttpBaseUrl();
      await fetch(`${baseUrl}/bt/shutdown`, { method: 'POST' });
      dispatch(setBtStatus('stopped'));
      dispatch(setActiveNodeNames([]));
      toast.success('BT stopped');
    } catch (err) {
      toast.error(`Failed to stop BT: ${err.message}`);
    }
  }, [callService, dispatch, getHttpBaseUrl]);

  // ── Auto-stop when the tree finishes on its own ──────────────────────────
  // BehaviorTreeNode publishes /bt/status='completed' once the root returns
  // SUCCESS or FAILURE, but leaves the BT process alive — the user otherwise
  // has to click Stop manually to free the slot. Treat 'completed' as an
  // implicit Stop so the button state flips back to Start-ready and the BT
  // node is torn down at the same time.
  useEffect(() => {
    if (btStatus === 'completed') {
      handleStop();
    }
  }, [btStatus, handleStop]);

  // ── BT status / active-nodes subscription ────────────────────────────────
  useEffect(() => {
    if (!rosbridgeUrl || !isActive) return;

    let ros = null;
    let statusTopic = null;
    let activeNodesTopic = null;

    const setupSubscription = async () => {
      try {
        const ROSLIB = (await import('roslib')).default;
        const { default: rosConnectionManager } = await import('../utils/rosConnectionManager');
        ros = await rosConnectionManager.getConnection(rosbridgeUrl);

        statusTopic = new ROSLIB.Topic({
          ros,
          name: '/bt/status',
          messageType: 'std_msgs/msg/String',
        });
        statusTopic.subscribe((msg) => {
          dispatch(setBtStatus(msg.data));
          if (msg.data !== 'running') dispatch(setActiveNodeNames([]));
        });

        activeNodesTopic = new ROSLIB.Topic({
          ros,
          name: '/bt/active_nodes',
          messageType: 'std_msgs/msg/String',
        });
        activeNodesTopic.subscribe((msg) => {
          const names = msg.data ? msg.data.split(',') : [];
          dispatch(setActiveNodeNames(names));
        });
      } catch (err) {
        console.debug('BT status subscription not available:', err.message);
      }
    };

    setupSubscription();
    return () => {
      if (statusTopic) statusTopic.unsubscribe();
      if (activeNodesTopic) activeNodesTopic.unsubscribe();
    };
  }, [rosbridgeUrl, isActive, dispatch]);

  // ── Annotate nodes for ReactFlow render ──────────────────────────────────
  // Layers on:
  //   isActive / isSelected — visual highlight from BT runtime + inspector
  //   hidden                — ReactFlow skips the node and its edges; flipped
  //                           on for any descendant of a collapsed Control
  //   childCount            — drives the BTControlNode +/- button disabled
  //                           state and the "N hidden" badge
  //   onToggleCollapse      — pass-through so BTControlNode can call back
  //                           without prop drilling
  const annotatedNodes = useMemo(() => {
    const activeSet = new Set(activeNodeNames);
    const hiddenIds = computeHiddenIds(nodes, edges);
    const childrenById = new Map(nodes.map((n) => [n.id, []]));
    const childCount = new Map();
    for (const e of edges) {
      if (childrenById.has(e.source)) childrenById.get(e.source).push(e.target);
      childCount.set(e.source, (childCount.get(e.source) ?? 0) + 1);
    }
    // Bubble active-state up from leaves to ancestor Control nodes so the
    // user can tell a Loop/Sequence is "live" even when collapsed — the
    // active leaf itself is hidden under the +/- toggle, but the Control
    // wrapper still pulses.
    const hasActiveDescendant = (rootId) => {
      const queue = [...(childrenById.get(rootId) || [])];
      while (queue.length) {
        const id = queue.shift();
        if (activeSet.has(id)) return true;
        const kids = childrenById.get(id);
        if (kids && kids.length) queue.push(...kids);
      }
      return false;
    };
    return nodes.map((node) => {
      const directly = activeSet.has(node.id);
      const isControl = node.type === 'btControl';
      const isActive = directly || (isControl && hasActiveDescendant(node.id));
      return {
        ...node,
        hidden: hiddenIds.has(node.id),
        data: {
          ...node.data,
          isActive,
          isSelected: node.id === selectedNodeId,
          childCount: childCount.get(node.id) ?? 0,
          onToggleCollapse: handleToggleCollapse,
        },
      };
    });
  }, [nodes, edges, activeNodeNames, selectedNodeId, handleToggleCollapse]);

  const statusColor =
    btStatus === 'running' ? 'bg-green-500' :
    btStatus === 'completed' ? 'bg-yellow-400' : 'bg-gray-400';
  const statusLabel =
    btStatus === 'running' ? 'Running' :
    btStatus === 'completed' ? 'Completed' : 'Stopped';

  const hasTree = nodes.length > 0;

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <h1 className="text-xl font-bold text-gray-800">BT Manager</h1>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            {treeFileName || 'No file loaded'}
          </span>
          <button
            onClick={undoHistory}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              canUndo
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdUndo size={18} />
          </button>
          <button
            onClick={redoHistory}
            disabled={!canRedo}
            title="Redo (Ctrl+Shift+Z)"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              canRedo
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdRedo size={18} />
          </button>
          <button
            onClick={handleAutoLayout}
            disabled={!hasTree}
            title="Auto Layout"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              hasTree
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdAutoFixHigh size={18} />
          </button>
          <button
            onClick={() => {
              setSaveFileName(treeFileName ? treeFileName.replace(/\.xml$/i, '') : '');
              setShowSaveDialog(true);
            }}
            disabled={!hasTree}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg',
              'text-sm font-medium transition-colors duration-150',
              hasTree
                ? 'bg-blue-50 hover:bg-blue-100 text-blue-700 cursor-pointer'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
          >
            <MdSave size={18} />
            Save As
          </button>
          <button
            onClick={() => setShowTreeList(true)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer',
              'bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium',
              'transition-colors duration-150'
            )}
          >
            <MdUploadFile size={18} />
            Load XML
          </button>
        </div>
      </div>

      {/* React Flow Canvas */}
      <div className="flex-1 relative flex">
        <BTNodePalette />
        <div
          className="flex-1 relative"
          onDragOver={handleCanvasDragOver}
          onDrop={handleCanvasDrop}
        >
          {parseError ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-red-500 text-center">
                <p className="font-semibold">Parse Error</p>
                <p className="text-sm mt-1">{parseError}</p>
              </div>
            </div>
          ) : nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <p className="text-lg">No behavior tree loaded</p>
                <p className="text-sm mt-1">Click "Load XML" or drag nodes from the palette</p>
              </div>
            </div>
          ) : (
            <ReactFlow
              nodes={annotatedNodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodeTypes={nodeTypes}
              onInit={(instance) => { reactFlowRef.current = instance; }}
              onConnect={handleConnect}
              onNodeClick={handleNodeClick}
              onNodeDragStop={handleNodeDragStop}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              nodesDraggable={true}
              nodesConnectable={true}
              elementsSelectable={true}
              deleteKeyCode={null}
              minZoom={0.3}
              maxZoom={2}
              zoomOnScroll={false}
              panOnScroll={true}
              zoomOnPinch={true}
              zoomActivationKeyCode="Control"
            >
              <Controls showInteractive={false} />
              <Background color="#e5e7eb" gap={16} />
            </ReactFlow>
          )}
        </div>
        {selectedNodeId && (
          <BTParamPanel
            nodes={annotatedNodes}
            selectedNodeId={selectedNodeId}
            onParamChange={handleParamChange}
            onNameChange={handleNameChange}
          />
        )}
      </div>

      {/* Bottom Control Bar */}
      <div className="flex items-center justify-between px-6 py-3 border-t border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <button
            onClick={handleStart}
            disabled={btStatus === 'running' || btStatus === 'completed' || !hasTree}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              (btStatus === 'running' || btStatus === 'completed' || !hasTree)
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            )}
          >
            <MdPlayArrow size={20} />
            Start
          </button>
          <button
            onClick={handleStop}
            disabled={btStatus === 'stopped'}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              btStatus === 'stopped'
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 text-white'
            )}
          >
            <MdStop size={20} />
            Stop
          </button>
        </div>

        <div className="flex items-center gap-2">
          <div className={clsx('w-3 h-3 rounded-full', statusColor)} />
          <span className="text-sm text-gray-600">{statusLabel}</span>
        </div>
      </div>

      {/* Tree List Modal */}
      <TreeListModal
        isOpen={showTreeList}
        onClose={() => setShowTreeList(false)}
        onSelect={handleServerFileSelect}
      />

      {/* Save As Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white rounded-xl shadow-xl p-6 w-80">
            <h2 className="text-base font-semibold text-gray-800 mb-4">Save Tree As</h2>
            <div className="flex items-center gap-1 border border-gray-300 rounded-lg px-3 py-2 focus-within:ring-2 focus-within:ring-blue-400">
              <input
                autoFocus
                type="text"
                value={saveFileName}
                onChange={(e) => setSaveFileName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveAs();
                  if (e.key === 'Escape') setShowSaveDialog(false);
                }}
                placeholder="filename"
                className="flex-1 text-sm outline-none"
              />
              <span className="text-sm text-gray-400">.xml</span>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveAs}
                disabled={!saveFileName.trim()}
                className={clsx(
                  'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                  saveFileName.trim()
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                )}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
