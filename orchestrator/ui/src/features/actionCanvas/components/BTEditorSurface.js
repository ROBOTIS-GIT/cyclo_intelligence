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
import {
  MdPlayArrow,
  MdStop,
  MdUploadFile,
  MdSave,
  MdUndo,
  MdRedo,
  MdAutoFixHigh,
  MdDeleteSweep,
  MdPowerSettingsNew,
} from 'react-icons/md';

import BTControlNode from '../../../components/bt/BTControlNode';
import BTActionNode from '../../../components/bt/BTActionNode';
import BTParamPanel from '../../../components/bt/BTParamPanel';
import BTNodePalette, { PALETTE_DRAG_MIME } from '../../../components/bt/BTNodePalette';
import TreeListModal from './TreeListModal';
import {
  parseBTXml,
  applyDagreLayout,
  findDeletionLayoutAnchor,
} from '../../../utils/btTreeParser';
import { isValidBtConnection } from '../../../utils/btConnection';
import { serializeFromGraph } from '../../../utils/btXmlSerializer';
import { setTreeXml, setTreeFileName, setBtStatus, setActiveNodeNames, setSelectedNodeId } from '../actionCanvasSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import { useBTHistory } from '../../../hooks/useBTHistory';
import { useBTNodeCatalog } from '../../../hooks/useBTNodeCatalog';
import { formatBtSupportedRobotTypes, isBtRobotSupported } from '../../../constants/btSupport';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';
import { selectBtSupportedRobotTypes } from '../btSupportSlice';
import { readBtTree, saveBtTree } from '../btTreesApi';

const nodeTypes = {
  btControl: BTControlNode,
  btAction: BTActionNode,
};

const API_BASE = '/api';

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

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
function layoutVisibleOnly(nodes, edges, { anchorNodeId = null } = {}) {
  const hidden = computeHiddenIds(nodes, edges);
  const visibleNodes = nodes.filter((n) => !hidden.has(n.id));
  const visibleEdges = edges.filter(
    (e) => !hidden.has(e.source) && !hidden.has(e.target)
  );
  const laid = applyDagreLayout(visibleNodes, visibleEdges, {
    respectStored: false,
    anchorNodeId,
  });
  const byId = new Map(laid.nodes.map((n) => [n.id, n]));
  return nodes.map((n) => (byId.has(n.id) ? byId.get(n.id) : n));
}

function catalogEntryToParams(entry) {
  return Object.fromEntries(
    (entry?.ports || []).map((port) => [port.name, port.default]),
  );
}

function normalizeBtStatus(status) {
  return String(status || 'stopped').trim().toLowerCase();
}

function getBtStatusLabel(status) {
  switch (normalizeBtStatus(status)) {
    case 'running':
      return 'Running';
    case 'completed':
      return 'Completed';
    case 'failed':
    case 'failure':
      return 'Failed';
    case 'stopping':
      return 'Stopping';
    default:
      return 'Stopped';
  }
}

function getSimulationInferenceNodeNames(nodeDataMap) {
  return Array.from(nodeDataMap.values())
    .filter(({ tag, params = {} }) => {
      if (tag !== 'SendCommand') return false;
      const command = String(params.command || 'LOAD').toUpperCase();
      if (command !== 'LOAD') return false;
      const mode = String(params.inference_mode || 'simulation').toLowerCase();
      return mode !== 'robot';
    })
    .map(({ name }) => name)
    .filter(Boolean);
}

export default function BTEditorSurface({
  isActive = true,
  title = 'Action Canvas',
  subtitle = '',
  className = 'w-full h-full',
  variant = 'legacy',
  onExitStateChange,
}) {
  const dispatch = useDispatch();
  const { callService } = useRosServiceCaller();
  const { catalog: nodeCatalog = [], refreshCatalog } = useBTNodeCatalog();
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const robotType = useSelector((state) => state.tasks.robotType);
  const supportedRobotTypes = useSelector(selectBtSupportedRobotTypes);
  const autonomyStudioVariant = variant === 'autonomy-studio';

  const treeXml = useSelector((state) => state.actionCanvas.treeXml);
  const treeFileName = useSelector((state) => state.actionCanvas.treeFileName);
  const btStatus = useSelector((state) => state.actionCanvas.btStatus);
  const activeNodeNames = useSelector((state) => state.actionCanvas.activeNodeNames);
  const selectedNodeId = useSelector((state) => state.actionCanvas.selectedNodeId);
  // A non-empty Redux document is parsed asynchronously after the first
  // render. Until that graph is actually installed, an early workspace switch
  // must not persist the temporary [] state over the saved draft.
  const graphHydratedRef = useRef(!treeXml);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  // nodeDataMap: Map<id, {tag, name, params}> — primary source of truth for node content
  const [nodeDataMap, setNodeDataMap] = useState(new Map());
  const [parseError, setParseError] = useState(null);
  const [showTreeList, setShowTreeList] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveFileName, setSaveFileName] = useState('');
  const [saveConflict, setSaveConflict] = useState(null);
  const [clearTreeArmed, setClearTreeArmed] = useState(false);
  const [btNodeStatus, setBtNodeStatus] = useState({
    state: 'unknown',
    raw: 'not checked',
  });
  const [btNodePendingAction, setBtNodePendingAction] = useState(null);
  const [btExecutionPending, setBtExecutionPending] = useState(null);

  useEffect(() => {
    if (typeof onExitStateChange !== 'function') return;
    const normalizedStatus = normalizeBtStatus(btStatus);
    onExitStateChange({
      active: ['running', 'stopping'].includes(normalizedStatus),
      busy: Boolean(btNodePendingAction || btExecutionPending),
    });
  }, [btExecutionPending, btNodePendingAction, btStatus, onExitStateChange]);

  // ReactFlow instance for coordinate conversion on drop
  const reactFlowRef = useRef(null);
  const clearTreeTimerRef = useRef(null);
  const clearTreeTargetRef = useRef(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const nodeDataMapRef = useRef(nodeDataMap);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  nodeDataMapRef.current = nodeDataMap;

  // ── History ──────────────────────────────────────────────────────────────
  // Snapshots encode the graph plus its file identity. Empty graphs are valid
  // snapshots so clearing the canvas (and adding the first node) remains fully
  // undoable and redoable.
  // isActive / isSelected are annotation-only and excluded.

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

  const disarmClearTree = useCallback(() => {
    if (clearTreeTimerRef.current) {
      clearTimeout(clearTreeTimerRef.current);
      clearTreeTimerRef.current = null;
    }
    clearTreeTargetRef.current = null;
    setClearTreeArmed(false);
  }, []);

  const armClearTree = useCallback((snapshot) => {
    if (clearTreeTimerRef.current) clearTimeout(clearTreeTimerRef.current);
    clearTreeTargetRef.current = snapshot;
    setClearTreeArmed(true);
    clearTreeTimerRef.current = setTimeout(() => {
      clearTreeTimerRef.current = null;
      clearTreeTargetRef.current = null;
      setClearTreeArmed(false);
    }, 4000);
  }, []);

  useEffect(() => () => {
    if (clearTreeTimerRef.current) clearTimeout(clearTreeTimerRef.current);
  }, []);

  useEffect(() => {
    if (!clearTreeArmed) return;
    if (
      btExecutionPending ||
      btNodePendingAction ||
      ['running', 'stopping'].includes(normalizeBtStatus(btStatus))
    ) {
      disarmClearTree();
    }
  }, [
    btExecutionPending,
    btNodePendingAction,
    btStatus,
    clearTreeArmed,
    disarmClearTree,
  ]);

  // ── Initial load from Redux treeXml (e.g. on page mount) ─────────────────
  useEffect(() => {
    if (!treeXml) {
      graphHydratedRef.current = true;
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
      setParseError(null);
      return;
    }
    try {
      const { nodes: n, edges: e, nodeDataMap: ndm } = parseBTXml(treeXml);
      graphHydratedRef.current = n.length === 0;
      setNodes(n);
      setEdges(e);
      setNodeDataMap(ndm);
      setParseError(null);
    } catch (err) {
      graphHydratedRef.current = false;
      setParseError(formatTaskDisplayMessage(err.message));
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount to restore Redux-persisted tree

  useEffect(() => {
    if (!graphHydratedRef.current && nodes.length > 0) {
      graphHydratedRef.current = true;
    }
  }, [nodes.length]);

  // ── Persist working tree to Redux so it survives page switches ────────────
  // The graph state (nodes/edges/nodeDataMap) lives in local useState, which
  // is torn down on unmount. Without this, navigating away and back to the BT
  // Manager wipes the user's in-progress tree. We serialise on a small debounce
  // so a flurry of edits doesn't dispatch on every keystroke, and again on
  // unmount to catch whatever's still in the debounce window.
  useEffect(() => {
    if (!graphHydratedRef.current) return undefined;
    const t = setTimeout(() => {
      try {
        dispatch(setTreeXml(
          nodes.length === 0 ? '' : serializeFromGraph(nodes, edges, nodeDataMap),
        ));
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
      if (!graphHydratedRef.current) return;
      try {
        dispatch(setTreeXml(n.length === 0 ? '' : serializeFromGraph(n, e, m)));
      } catch {
        // Same swallow as the debounced path — preserve last good state.
      }
    };
  }, [dispatch]);

  // ── Handle tree selection from TreeListModal ──────────────────────────────
  const handleServerFileSelect = useCallback(async (item) => {
    const requestedName = item?.name || String(item?.full_path || '').split('/').pop();
    if (!requestedName) return;
    try {
      const tree = await readBtTree(requestedName);
      const xmlContent = tree.content || '';
      const fileName = tree.name || requestedName;

      const { nodes: n, edges: e, nodeDataMap: ndm } = parseBTXml(xmlContent);
      setNodes(n);
      setEdges(e);
      setNodeDataMap(ndm);
      setParseError(null);

      disarmClearTree();
      resetHistory();
      dispatch(setSelectedNodeId(null));
      dispatch(setTreeXml(xmlContent));
      dispatch(setTreeFileName(fileName));
      toast.success(`Opened: ${fileName}`);
    } catch (err) {
      toast.error(`Failed to load file: ${formatTaskDisplayMessage(err.message)}`);
    }
  }, [disarmClearTree, dispatch, setNodes, setEdges, resetHistory]);

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
      const m = name.match(new RegExp(`^${tag}_(\\d+)$`));
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
    dispatch(setSelectedNodeId(id));
  }, [captureHistory, setNodes, dispatch, nodeCatalog]);

  // ── Manual edge connection ────────────────────────────────────────────────
  const handleConnect = useCallback((connection) => {
    // Belt-and-braces alongside the isValidConnection prop: actions are BT
    // leaves — the engine loads children of control nodes only, so an edge
    // out of an action would render but never execute.
    if (!isValidBtConnection(connection, nodesRef.current, edgesRef.current)) {
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

  const handleClearTree = useCallback(() => {
    if (nodesRef.current.length === 0) return;
    if (
      ['running', 'stopping'].includes(normalizeBtStatus(btStatus)) ||
      btNodePendingAction ||
      btExecutionPending
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
    setParseError(null);
    dispatch(setSelectedNodeId(null));
    dispatch(setTreeXml(''));
    dispatch(setTreeFileName(''));
    disarmClearTree();
    toast.success('Task cleared');
  }, [
    armClearTree,
    btExecutionPending,
    btNodePendingAction,
    btStatus,
    captureHistory,
    clearTreeArmed,
    disarmClearTree,
    dispatch,
    getHistorySnapshot,
    setEdges,
    setNodes,
  ]);

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

  // ── Serialize current graph to BT XML ────────────────────────────────────
  const getSerializedXml = useCallback(() => {
    return serializeFromGraph(nodes, edges, nodeDataMap);
  }, [nodes, edges, nodeDataMap]);

  // ── Save As ───────────────────────────────────────────────────────────────
  const handleSaveAs = useCallback(async ({ overwrite = false } = {}) => {
    const name = saveFileName.trim();
    if (!name) return;

    const content = getSerializedXml();
    if (!content) return;

    try {
      const data = await saveBtTree({ filename: name, content, overwrite });
      toast.success(formatTaskDisplayMessage(data.message));
      setShowSaveDialog(false);
      setSaveFileName('');
      setSaveConflict(null);
    } catch (err) {
      const conflict = err.status === 409 || err.detail?.code === 'file_exists';
      if (conflict) {
        const detail = err.detail && typeof err.detail === 'object' ? err.detail : {};
        setSaveConflict({ ...detail, message: err.message });
        toast.error(formatTaskDisplayMessage(err.message) || 'File already exists');
        return;
      }
      toast.error(`Save failed: ${formatTaskDisplayMessage(err.message)}`);
    }
  }, [saveFileName, getSerializedXml]);

  // ── BT Start ──────────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (nodes.length === 0) {
      toast.error('No task to run');
      return;
    }
    if (btNodeStatus.state !== 'up') {
      toast.error('Task Engine is off');
      return;
    }
    if (btExecutionPending) return;
    setBtExecutionPending('start');
    try {
      const executionStatus = normalizeBtStatus(btStatus);
      if (['completed', 'failed', 'failure'].includes(executionStatus)) {
        const cleanupResult = await callService(
          '/bt/set_running',
          'std_srvs/srv/SetBool',
          { data: false },
        );
        if (!cleanupResult?.success) {
          throw new Error(formatTaskDisplayMessage(cleanupResult?.message) || 'Failed to reset the completed task');
        }
      }

      const currentXml = getSerializedXml();
      const simulationInferenceNodes = getSimulationInferenceNodeNames(nodeDataMap);
      if (simulationInferenceNodes.length > 0) {
        toast(
          `SendCommand is in simulation mode: ${simulationInferenceNodes.join(', ')}. Robot command topics will not be published.`,
          { duration: 7000 },
        );
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
        toast.success('Task started');
      } else {
        toast.error(`Failed: ${formatTaskDisplayMessage(result.message)}`);
      }
    } catch (err) {
      toast.error(`Failed to start task: ${formatTaskDisplayMessage(err.message)}`);
    } finally {
      setBtExecutionPending(null);
    }
  }, [btExecutionPending, btNodeStatus.state, btStatus, callService, dispatch, getSerializedXml, nodeDataMap, nodes.length]);

  // ── BT Stop ───────────────────────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    if (btExecutionPending) return;
    setBtExecutionPending('stop');
    try {
      const result = await callService('/bt/set_running', 'std_srvs/srv/SetBool', { data: false });
      if (!result.success) {
        toast.error(`Failed: ${formatTaskDisplayMessage(result.message)}`);
        return;
      }
      dispatch(setBtStatus('stopped'));
      dispatch(setActiveNodeNames([]));
      toast.success('Task stopped');
    } catch (err) {
      toast.error(`Failed to stop task: ${formatTaskDisplayMessage(err.message)}`);
    } finally {
      setBtExecutionPending(null);
    }
  }, [btExecutionPending, callService, dispatch]);

  // ── BT node process lifecycle via supervisor API ─────────────────────────
  const refreshBtNodeStatus = useCallback(async ({ quiet = false } = {}) => {
    try {
      const response = await fetch(`${API_BASE}/services/bt_node/status`);
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || `status failed (${response.status})`);
      }
      setBtNodeStatus(data);
      if (data.state === 'down') {
        dispatch(setBtStatus('stopped'));
        dispatch(setActiveNodeNames([]));
      }
      return data;
    } catch (err) {
      const next = {
        state: 'unknown',
        raw: err.message,
      };
      setBtNodeStatus(next);
      if (!quiet) toast.error(`Task Engine status unavailable: ${formatTaskDisplayMessage(err.message)}`);
      return next;
    }
  }, [dispatch]);

  useEffect(() => {
    if (!isActive) return undefined;
    refreshBtNodeStatus({ quiet: true });
    const id = setInterval(
      () => refreshBtNodeStatus({ quiet: true }),
      5000,
    );
    return () => clearInterval(id);
  }, [isActive, refreshBtNodeStatus]);

  const callBtNodeService = useCallback(async (action) => {
    setBtNodePendingAction(action);
    try {
      const init = { method: 'POST' };
      if (action === 'start') {
        init.headers = { 'Content-Type': 'application/json' };
        init.body = JSON.stringify({ robot_type: robotType || '' });
      }
      const response = await fetch(`${API_BASE}/services/bt_node/${action}`, init);
      const data = await readJsonResponse(response);
      if (!response.ok || data.ok === false) {
        throw new Error(data.detail || data.message || `${action} failed`);
      }
      return data;
    } finally {
      setBtNodePendingAction(null);
    }
  }, [robotType]);

  const handleBtNodeOn = useCallback(async () => {
    if (robotType && !isBtRobotSupported(robotType, supportedRobotTypes)) {
      toast.error(`Action Canvas currently supports only ${formatBtSupportedRobotTypes(supportedRobotTypes)}`);
      return;
    }

    try {
      await callBtNodeService('start');
      toast.success('Task Engine started');
      await refreshBtNodeStatus({ quiet: true });
      try {
        await refreshCatalog({ force: true });
      } catch (err) {
        console.debug('BT catalog refresh after node start failed:', err.message);
      }
    } catch (err) {
      toast.error(`Failed to start Task Engine: ${formatTaskDisplayMessage(err.message)}`);
      await refreshBtNodeStatus({ quiet: true });
    }
  }, [callBtNodeService, refreshBtNodeStatus, refreshCatalog, robotType, supportedRobotTypes]);

  const handleBtNodeOff = useCallback(async () => {
    try {
      const executionStatus = normalizeBtStatus(btStatus);
      if (['completed', 'failed', 'failure'].includes(executionStatus)) {
        const result = await callService(
          '/bt/set_running',
          'std_srvs/srv/SetBool',
          { data: false },
        );
        if (!result?.success) {
          throw new Error(formatTaskDisplayMessage(result?.message) || 'Failed to reset the completed task');
        }
      }
      await callBtNodeService('stop');
      dispatch(setBtStatus('stopped'));
      dispatch(setActiveNodeNames([]));
      toast.success('Task Engine stopped');
      await refreshBtNodeStatus({ quiet: true });
    } catch (err) {
      toast.error(`Failed to stop Task Engine: ${formatTaskDisplayMessage(err.message)}`);
      await refreshBtNodeStatus({ quiet: true });
    }
  }, [btStatus, callBtNodeService, callService, dispatch, refreshBtNodeStatus]);

  // ── BT status / active-nodes subscription ────────────────────────────────
  useEffect(() => {
    if (!rosbridgeUrl || !isActive) return;

    let ros = null;
    let statusTopic = null;
    let activeNodesTopic = null;

    const setupSubscription = async () => {
      try {
        const ROSLIB = (await import('roslib')).default;
        const { default: rosConnectionManager } = await import('../../../utils/rosConnectionManager');
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

  const hasTree = nodes.length > 0;
  const normalizedBtStatus = normalizeBtStatus(btStatus);
  const isBtNodeUp = btNodeStatus.state === 'up';
  const isBtNodeBusy = Boolean(btNodePendingAction);
  const isBtExecutionBusy = Boolean(btExecutionPending);
  const isBtRunning = normalizedBtStatus === 'running';
  const isBtBusy = isBtRunning || normalizedBtStatus === 'stopping';
  const isBtTerminal = ['completed', 'failed', 'failure'].includes(normalizedBtStatus);
  const canStartBt = hasTree && isBtNodeUp && !isBtBusy && !isBtNodeBusy && !isBtExecutionBusy;
  const canStopBt = isBtNodeUp && isBtRunning && !isBtNodeBusy && !isBtExecutionBusy;
  const canClearTree = hasTree && !isBtBusy && !isBtNodeBusy && !isBtExecutionBusy;
  const canStartBtNode = !isBtNodeUp && !isBtNodeBusy && !isBtExecutionBusy;
  const canStopBtNode = isBtNodeUp &&
    (normalizedBtStatus === 'stopped' || isBtTerminal) &&
    !isBtNodeBusy &&
    !isBtExecutionBusy;
  const statusColor = autonomyStudioVariant
    ? isBtRunning ? 'bg-[var(--mc-success)]' :
      normalizedBtStatus === 'completed' ? 'bg-[var(--mc-warning)]' :
      ['failed', 'failure'].includes(normalizedBtStatus) ? 'bg-[var(--mc-danger)]' :
      normalizedBtStatus === 'stopping' ? 'bg-[var(--mc-warning)]' :
      'bg-[var(--mc-text-subtle)]'
    : isBtRunning ? 'bg-green-500' :
      normalizedBtStatus === 'completed' ? 'bg-yellow-400' :
      ['failed', 'failure'].includes(normalizedBtStatus) ? 'bg-red-500' :
      normalizedBtStatus === 'stopping' ? 'bg-orange-400' :
      'bg-gray-400';
  const statusLabel = getBtStatusLabel(btStatus);
  const btNodeStatusColor = autonomyStudioVariant
    ? isBtNodeUp ? 'bg-[var(--mc-success)]' :
      btNodeStatus.state === 'down' ? 'bg-[var(--mc-text-subtle)]' :
      'bg-[var(--mc-warning)]'
    : isBtNodeUp ? 'bg-green-500' :
      btNodeStatus.state === 'down' ? 'bg-gray-400' :
      'bg-yellow-400';
  const btNodeStatusLabel =
    isBtNodeUp ? 'On' :
    btNodeStatus.state === 'down' ? 'Off' :
    'Unknown';

  return (
    <div
      data-variant={variant}
      className={clsx(
        className,
        'bt-editor-surface flex flex-col',
        autonomyStudioVariant && 'bg-[var(--mc-bg)] text-[var(--mc-text)]',
      )}
    >
      {/* Header */}
      <div className={clsx(
        'flex items-center justify-between px-6 border-b',
        autonomyStudioVariant
          ? 'h-14 border-[var(--mc-border)] bg-[var(--mc-surface-2)]'
          : 'py-4 border-black bg-white',
      )}>
        <div className="min-w-0">
          <h1 className={clsx(
            'font-bold truncate',
            autonomyStudioVariant
              ? 'text-[16px] tracking-tight text-[var(--mc-text)]'
              : 'text-xl text-gray-800',
          )}>
            {title}
          </h1>
          {subtitle && (
            <p className={clsx(
              'mt-0.5 truncate text-[10px] font-mono',
              autonomyStudioVariant
                ? 'text-[var(--mc-text-muted)]'
                : 'text-gray-500',
            )}>
              {subtitle}
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className={clsx(
            'max-w-[220px] truncate font-mono',
            autonomyStudioVariant
              ? 'text-[11px] text-[var(--mc-text-muted)]'
              : 'text-sm text-gray-500',
          )}>
            {treeFileName || 'No file loaded'}
          </span>
          <button
            onClick={undoHistory}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              autonomyStudioVariant
                ? canUndo
                  ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : canUndo
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
              autonomyStudioVariant
                ? canRedo
                  ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : canRedo
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
              autonomyStudioVariant
                ? hasTree
                  ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : hasTree
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdAutoFixHigh size={18} />
          </button>
          <button
            type="button"
            onClick={handleClearTree}
            disabled={!canClearTree}
            aria-label={clearTreeArmed ? 'Confirm clear current task' : 'Clear current task'}
            title={clearTreeArmed ? 'Click again to clear the current task' : 'Clear current task'}
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              autonomyStudioVariant
                ? !canClearTree
                  ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                  : clearTreeArmed
                    ? 'border border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] cursor-pointer'
                    : 'border border-[var(--mc-danger-border)] bg-[var(--mc-surface)] text-[var(--mc-danger)] shadow-[var(--mc-shadow)] hover:bg-[var(--mc-surface-hover)] cursor-pointer'
                : !canClearTree
                  ? 'bg-gray-100 text-gray-300 cursor-not-allowed'
                  : clearTreeArmed
                    ? 'bg-red-600 text-white cursor-pointer'
                    : 'bg-red-50 hover:bg-red-100 text-red-600 cursor-pointer',
            )}
          >
            <MdDeleteSweep size={18} aria-hidden="true" />
          </button>
          <button
            onClick={() => {
              setSaveFileName(treeFileName ? treeFileName.replace(/\.xml$/i, '') : '');
              setSaveConflict(null);
              setShowSaveDialog(true);
            }}
            disabled={!hasTree}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg',
              'text-sm font-medium transition-colors duration-150',
              autonomyStudioVariant
                ? hasTree
                  ? 'border border-[var(--mc-accent)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-accent-hover)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : hasTree
                ? 'bg-blue-50 hover:bg-blue-100 text-blue-700 cursor-pointer'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
          >
            <MdSave size={18} />
            Save Task As
          </button>
          <button
            onClick={() => setShowTreeList(true)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer',
              autonomyStudioVariant
                ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] text-sm font-medium'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium',
              'transition-colors duration-150'
            )}
          >
            <MdUploadFile size={18} />
            Open Task
          </button>
        </div>
      </div>

      {/* React Flow Canvas */}
      <div className={clsx(
        'flex-1 relative flex min-h-0',
        autonomyStudioVariant && 'bg-[var(--mc-bg)]',
      )}>
        <BTNodePalette canUpdateCatalog={isBtNodeUp} />
        <div
          className={clsx(
            'flex-1 relative',
            autonomyStudioVariant && 'bg-[var(--mc-canvas)]',
          )}
          onDragOver={handleCanvasDragOver}
          onDrop={handleCanvasDrop}
        >
          {parseError ? (
            <div className="flex items-center justify-center h-full">
              <div className={clsx(
                'text-center',
                autonomyStudioVariant ? 'text-[var(--mc-danger)]' : 'text-red-500',
              )}>
                <p className="font-semibold">Task File Error</p>
                <p className="text-sm mt-1">{parseError}</p>
              </div>
            </div>
          ) : nodes.length === 0 ? (
            <div className={clsx(
              'flex items-center justify-center h-full',
              autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
            )}>
              <div className="text-center">
                <p className="text-lg">No task yet</p>
                <p className="text-sm mt-1">Open a saved task or drag steps from the left</p>
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
              isValidConnection={(connection) => (
                isValidBtConnection(connection, nodesRef.current, edgesRef.current)
              )}
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
              autoPanOnConnect={false}
            >
              <Controls showInteractive={false} />
              <Background color={autonomyStudioVariant ? 'var(--mc-border)' : '#e5e7eb'} gap={16} />
            </ReactFlow>
          )}
        </div>
        {selectedNodeId && (
          <BTParamPanel
            nodes={annotatedNodes}
            selectedNodeId={selectedNodeId}
            onParamChange={handleParamChange}
            onNameChange={handleNameChange}
            variant={variant}
          />
        )}
      </div>

      {/* Bottom Control Bar */}
      <div className={clsx(
        'flex items-center justify-between px-6 border-t',
        autonomyStudioVariant
          ? 'h-14 border-[var(--mc-border)] bg-[var(--mc-surface-2)]'
          : 'py-3 border-black bg-white',
      )}>
        <div className="flex items-center gap-3">
          <div className={clsx(
            'flex items-center gap-2 pr-3 mr-1 border-r',
            autonomyStudioVariant ? 'border-[var(--mc-border)]' : 'border-gray-200',
          )}>
            <div className={clsx('w-3 h-3 rounded-full', btNodeStatusColor)} />
            <span className={clsx(
              'text-sm',
              autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-600',
            )}>
              Task Engine {btNodeStatusLabel}
            </span>
            <button
              onClick={handleBtNodeOn}
              disabled={!canStartBtNode}
              title="Turn on Task Engine"
              className={clsx(
                'flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                autonomyStudioVariant
                  ? !canStartBtNode
                    ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                    : 'border border-[var(--mc-success)] bg-[var(--mc-success)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                  : !canStartBtNode
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              )}
            >
              <MdPowerSettingsNew size={18} />
              Turn On
            </button>
            <button
              onClick={handleBtNodeOff}
              disabled={!canStopBtNode}
              title="Turn off Task Engine"
              className={clsx(
                'flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                autonomyStudioVariant
                  ? !canStopBtNode
                    ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                    : 'border border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                  : !canStopBtNode
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-700 text-white'
              )}
            >
              <MdStop size={18} />
              Turn Off
            </button>
          </div>
          <button
            onClick={handleStart}
            disabled={!canStartBt}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              autonomyStudioVariant
                ? !canStartBt
                  ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                  : 'border border-[var(--mc-success)] bg-[var(--mc-success)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                : !canStartBt
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            )}
          >
            <MdPlayArrow size={20} />
            Run Task
          </button>
          <button
            onClick={handleStop}
            disabled={!canStopBt}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              autonomyStudioVariant
                ? !canStopBt
                  ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                  : 'border border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                : !canStopBt
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 text-white'
            )}
          >
            <MdStop size={20} />
            Stop Task
          </button>
        </div>

        <div className="flex items-center gap-2">
          <div className={clsx('w-3 h-3 rounded-full', statusColor)} />
          <span className={clsx(
            'text-sm',
            autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-600',
          )}>
            {statusLabel}
          </span>
        </div>
      </div>

      {/* Tree List Modal */}
      <TreeListModal
        isOpen={showTreeList}
        onClose={() => setShowTreeList(false)}
        onSelect={handleServerFileSelect}
        variant={variant}
      />

      {/* Save As Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className={clsx(
            'rounded-2xl p-6 w-80',
            autonomyStudioVariant
              ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text)] shadow-[var(--mc-shadow)]'
              : 'bg-white shadow-xl',
          )}>
            <h2 className={clsx(
              'text-base font-semibold mb-4',
              autonomyStudioVariant ? 'text-[var(--mc-text)]' : 'text-gray-800',
            )}>
              Save Task As
            </h2>
            <div className={clsx(
              'flex items-center gap-1 border rounded-lg px-3 py-2 focus-within:ring-2',
              autonomyStudioVariant
                ? 'border-[var(--mc-border-strong)] bg-[var(--mc-surface-2)] focus-within:ring-[var(--mc-accent)]'
                : 'border-gray-300 focus-within:ring-blue-400',
            )}>
              <input
                autoFocus
                type="text"
                value={saveFileName}
                onChange={(e) => {
                  setSaveFileName(e.target.value);
                  setSaveConflict(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveAs();
                  if (e.key === 'Escape') {
                    setShowSaveDialog(false);
                    setSaveConflict(null);
                  }
                }}
                placeholder="filename"
                className={clsx(
                  'flex-1 min-w-0 text-sm outline-none bg-transparent',
                  autonomyStudioVariant && 'text-[var(--mc-text)] placeholder:text-[var(--mc-text-subtle)]',
                )}
              />
              <span className={clsx(
                'text-sm',
                autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
              )}>
                .xml
              </span>
            </div>
            {saveConflict && (
              <div className={clsx(
                'mt-3 rounded-lg border px-3 py-2 text-sm',
                autonomyStudioVariant
                  ? 'border-[var(--mc-warning)] bg-[var(--mc-surface-2)] text-[var(--mc-warning)]'
                  : 'border-amber-200 bg-amber-50 text-amber-800',
              )}>
                <div className="font-medium">File already exists</div>
                <div className="mt-1">
                  Choose another name or overwrite {saveConflict.filename || 'this file'}.
                </div>
              </div>
            )}
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => {
                  setShowSaveDialog(false);
                  setSaveConflict(null);
                }}
                className={clsx(
                  'px-4 py-2 text-sm rounded-lg transition-colors',
                  autonomyStudioVariant
                    ? 'text-[var(--mc-text-muted)] hover:bg-[var(--mc-surface-hover)]'
                    : 'text-gray-600 hover:bg-gray-100',
                )}
              >
                Cancel
              </button>
              {saveConflict && (
                <button
                  onClick={() => handleSaveAs({ overwrite: true })}
                  className={clsx(
                    'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                    autonomyStudioVariant
                      ? 'border border-[var(--mc-danger-border)] bg-[var(--mc-surface)] text-[var(--mc-danger)] hover:bg-[var(--mc-surface-hover)]'
                      : 'bg-red-50 hover:bg-red-100 text-red-700',
                  )}
                >
                  Overwrite
                </button>
              )}
              <button
                onClick={handleSaveAs}
                disabled={!saveFileName.trim()}
                className={clsx(
                  'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                  autonomyStudioVariant
                    ? saveFileName.trim()
                      ? 'bg-[var(--mc-accent)] hover:bg-[var(--mc-accent-hover)] text-[var(--mc-accent-fg)]'
                      : 'bg-[var(--mc-surface-hover)] text-[var(--mc-text-subtle)] cursor-not-allowed'
                    : saveFileName.trim()
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
