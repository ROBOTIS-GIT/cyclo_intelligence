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

import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ReactFlow,
  Controls,
  Background,
  addEdge,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import toast from "react-hot-toast";
import {
  MdAutoFixHigh,
  MdDeleteSweep,
  MdDriveFileRenameOutline,
  MdRedo,
  MdSave,
  MdStar,
  MdUndo,
  MdUploadFile,
} from "react-icons/md";

import BTActionNode from "../bt/BTActionNode";
import BTControlNode from "../bt/BTControlNode";
import BTNodePalette, { PALETTE_DRAG_MIME } from "../bt/BTNodePalette";
import BTParamPanel from "../bt/BTParamPanel";
import { useBTHistory } from "../../hooks/useBTHistory";
import { useBTNodeCatalog } from "../../hooks/useBTNodeCatalog";
import {
  parseBTXml,
  applyDagreLayout,
  findDeletionLayoutAnchor,
} from "../../utils/btTreeParser";
import { isValidBtConnection } from "../../utils/btConnection";
import { formatTaskDisplayMessage } from "../../utils/taskTerminology";
import { serializeFromGraph } from "../../utils/btXmlSerializer";

const nodeTypes = {
  btControl: BTControlNode,
  btAction: BTActionNode,
};

const reactFlowProOptions = { hideAttribution: true };

function catalogEntryToParams(entry) {
  return Object.fromEntries(
    (entry?.ports || []).map((port) => [port.name, port.default]),
  );
}

function collectDescendants(rootId, edges) {
  const out = new Set();
  const queue = [rootId];
  while (queue.length) {
    const id = queue.shift();
    for (const edge of edges) {
      if (edge.source === id && !out.has(edge.target)) {
        out.add(edge.target);
        queue.push(edge.target);
      }
    }
  }
  return out;
}

function computeHiddenIds(nodes, edges) {
  const hidden = new Set();
  nodes.forEach((node) => {
    if (node.type !== "btControl" || !node.data?.collapsed) return;
    collectDescendants(node.id, edges).forEach((id) => hidden.add(id));
  });
  return hidden;
}

// Shared with the Action Canvas editor; re-exported to keep existing imports.
export { isValidBtConnection } from "../../utils/btConnection";

function layoutVisibleOnly(nodes, edges, { anchorNodeId = null } = {}) {
  const hidden = computeHiddenIds(nodes, edges);
  const visibleNodes = nodes.filter((node) => !hidden.has(node.id));
  const visibleEdges = edges.filter((edge) => (
    !hidden.has(edge.source) && !hidden.has(edge.target)
  ));
  const laidOut = applyDagreLayout(visibleNodes, visibleEdges, {
    respectStored: false,
    anchorNodeId,
  });
  const byId = new Map(laidOut.nodes.map((node) => [node.id, node]));
  return nodes.map((node) => byId.get(node.id) || node);
}

export default function MissionBtEditor({
  filePath,
  fileOptions = [],
  defaultFilePath = "",
  xml,
  loading = false,
  activeNodeNames = [],
  onXmlChange,
  onLoadXml,
  onSaveXml,
  onFilePathChange,
  onSaveXmlAs,
  onSetDefaultXml,
  fileActionsDisabled = false,
}) {
  const { catalog: nodeCatalog = [] } = useBTNodeCatalog();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [nodeDataMap, setNodeDataMap] = useState(new Map());
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [parseError, setParseError] = useState(null);
  const [hydratedPath, setHydratedPath] = useState("");
  const [fileAction, setFileAction] = useState("");
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [pendingLoadPath, setPendingLoadPath] = useState(filePath || "");
  const [showSaveAsDialog, setShowSaveAsDialog] = useState(false);
  const [saveAsName, setSaveAsName] = useState("");
  const [clearTreeArmed, setClearTreeArmed] = useState(false);
  const reactFlowRef = useRef(null);
  const clearTreeTimerRef = useRef(null);
  const clearTreeTargetRef = useRef(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const nodeDataMapRef = useRef(nodeDataMap);
  const lastEmittedXmlRef = useRef(null);
  const lastEmittedPathRef = useRef("");
  const onXmlChangeRef = useRef(onXmlChange);
  const fileActionRequestRef = useRef(0);
  const coalescedEditRef = useRef({ key: "", lastChangeAt: 0 });

  nodesRef.current = nodes;
  edgesRef.current = edges;
  nodeDataMapRef.current = nodeDataMap;
  onXmlChangeRef.current = onXmlChange;

  useEffect(() => {
    fileActionRequestRef.current += 1;
    if (clearTreeTimerRef.current) {
      clearTimeout(clearTreeTimerRef.current);
      clearTreeTimerRef.current = null;
    }
    clearTreeTargetRef.current = null;
    setClearTreeArmed(false);
    setFileAction("");
    setShowLoadDialog(false);
    setPendingLoadPath(filePath || "");
    setShowSaveAsDialog(false);
    setSaveAsName("");
  }, [filePath]);

  const availableFileOptions = useMemo(() => (
    [filePath, ...fileOptions]
      .map((path) => String(path || "").trim())
      .filter((path, index, paths) => path && paths.indexOf(path) === index)
  ), [fileOptions, filePath]);

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
  }, [setEdges, setNodes]);

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

  const captureCoalescedEditHistory = useCallback((key) => {
    const now = Date.now();
    const previous = coalescedEditRef.current;
    if (previous.key !== key || now - previous.lastChangeAt > 750) {
      captureHistory();
    }
    coalescedEditRef.current = { key, lastChangeAt: now };
  }, [captureHistory]);

  useEffect(() => {
    if (
      filePath === lastEmittedPathRef.current &&
      xml === lastEmittedXmlRef.current
    ) {
      return;
    }
    try {
      const parsed = parseBTXml(xml || "");
      // The prop already is parent-owned state. Treat its normalized graph as
      // the baseline instead of emitting the component's initial empty graph
      // (or marking formatting-only normalization as a user edit).
      lastEmittedXmlRef.current = serializeFromGraph(
        parsed.nodes || [],
        parsed.edges || [],
        parsed.nodeDataMap || new Map(),
      );
      lastEmittedPathRef.current = filePath;
      setNodes(parsed.nodes || []);
      setEdges(parsed.edges || []);
      setNodeDataMap(parsed.nodeDataMap || new Map());
      setSelectedNodeId(null);
      setParseError(null);
      setHydratedPath(filePath);
      coalescedEditRef.current = { key: "", lastChangeAt: 0 };
      resetHistory();
    } catch (error) {
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
      setSelectedNodeId(null);
      setParseError(error instanceof Error
        ? formatTaskDisplayMessage(error.message, "Waypoint Task")
        : "Could not read this waypoint task file");
      setHydratedPath(filePath);
      coalescedEditRef.current = { key: "", lastChangeAt: 0 };
    }
  }, [filePath, resetHistory, setEdges, setNodes, xml]);

  // Persist tree edits to the parent immediately (not debounced): a debounce
  // whose timer was reset by every parent re-render never fired, and switching
  // waypoints cleared the pending timer, silently dropping the whole tree.
  // serializeFromGraph omits node positions, so drags produce identical XML and
  // are skipped by the guard below — only real structural edits re-emit. filePath
  // is intentionally excluded from the deps: on a waypoint switch it changes one
  // render before `nodes` reloads, and emitting in that gap would write the old
  // tree to the new path.
  useLayoutEffect(() => {
    if (parseError || hydratedPath !== filePath) return;
    const emit = onXmlChangeRef.current;
    if (typeof emit !== "function") return;
    let serialized;
    try {
      serialized = serializeFromGraph(nodes, edges, nodeDataMap);
    } catch {
      return; // partial graph mid-edit; wait for the next change
    }
    if (filePath === lastEmittedPathRef.current && serialized === lastEmittedXmlRef.current) {
      return;
    }
    lastEmittedXmlRef.current = serialized;
    lastEmittedPathRef.current = filePath;
    emit(filePath, serialized);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [edges, hydratedPath, nodeDataMap, nodes, parseError]);

  const handleCanvasDragOver = useCallback((event) => {
    if (event.dataTransfer.types.includes(PALETTE_DRAG_MIME)) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    }
  }, []);

  const handleCanvasDrop = useCallback((event) => {
    const tag = event.dataTransfer.getData(PALETTE_DRAG_MIME)
      || event.dataTransfer.getData("text/plain");
    const meta = nodeCatalog.find((entry) => entry.tag === tag);
    if (!tag || !meta) return;
    event.preventDefault();

    const position = reactFlowRef.current
      ? reactFlowRef.current.screenToFlowPosition({ x: event.clientX, y: event.clientY })
      : { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 };

    let maxIndex = 0;
    for (const { name } of nodeDataMapRef.current.values()) {
      const match = String(name || "").match(new RegExp(`^${tag}_(\\d+)$`));
      if (match) maxIndex = Math.max(maxIndex, parseInt(match[1], 10));
    }
    const name = `${tag}_${maxIndex + 1}`;
    const id = `bt_${Date.now()}`;
    const params = catalogEntryToParams(meta);
    const isControl = meta.category === "control";
    const nextNode = {
      id,
      type: isControl ? "btControl" : "btAction",
      position,
      data: isControl
        ? { label: name, nodeType: tag, params, collapsed: false }
        : { label: name, nodeType: tag, params },
    };

    captureHistory();
    setNodes((current) => [...current, nextNode]);
    setNodeDataMap((current) => {
      const next = new Map(current);
      next.set(
        id,
        isControl
          ? { tag, name, params, collapsed: false }
          : { tag, name, params },
      );
      return next;
    });
    setSelectedNodeId(id);
  }, [captureHistory, nodeCatalog, setNodes]);

  const handleConnect = useCallback((connection) => {
    if (!isValidBtConnection(connection, nodesRef.current, edgesRef.current)) {
      toast.error("Each step can have only one parent, and connections cannot form a loop");
      return;
    }
    captureHistory();
    const nextEdges = addEdge(
      { ...connection, type: "smoothstep", animated: false },
      edgesRef.current,
    );
    setEdges(nextEdges);
    setNodes(layoutVisibleOnly(nodesRef.current, nextEdges, {
      anchorNodeId: connection.source,
    }));
  }, [captureHistory, setEdges, setNodes]);

  const handleAutoLayout = useCallback(() => {
    if (!nodesRef.current.length) return;
    captureHistory();
    setNodes(layoutVisibleOnly(nodesRef.current, edgesRef.current));
  }, [captureHistory, setNodes]);

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
    disarmClearTree,
    fileAction,
    fileActionsDisabled,
    filePath,
    getHistorySnapshot,
    hydratedPath,
    loading,
    parseError,
    setEdges,
    setNodes,
  ]);

  // Local waypoint XML is owned by the mission store. Keep file I/O behind
  // parent callbacks so this editor never reaches into the standalone library's global
  // orchestrator/bt/trees template directory.
  const serializeCurrentXml = useCallback(() => serializeFromGraph(
    nodesRef.current,
    edgesRef.current,
    nodeDataMapRef.current,
  ), []);

  const handleLoadXml = useCallback(async (targetPath) => {
    if (!targetPath || typeof onLoadXml !== "function") return;
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("load");
    try {
      await onLoadXml(targetPath);
      if (fileActionRequestRef.current !== requestId) return;
      setShowLoadDialog(false);
      toast.success(`Opened: ${targetPath}`);
      if (typeof onFilePathChange === "function") onFilePathChange(targetPath);
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to load ${targetPath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [onFilePathChange, onLoadXml]);

  const handleSaveXml = useCallback(async () => {
    if (!filePath || typeof onSaveXml !== "function") return;
    let serialized;
    try {
      serialized = serializeCurrentXml();
    } catch (error) {
      toast.error(`Failed to serialize ${filePath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
      return;
    }
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("save");
    try {
      await onSaveXml(filePath, serialized);
      if (fileActionRequestRef.current !== requestId) return;
      toast.success(`Saved: ${filePath}`);
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to save ${filePath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [filePath, onSaveXml, serializeCurrentXml]);

  const handleSaveXmlAs = useCallback(async (event) => {
    event.preventDefault();
    const targetName = saveAsName.trim();
    if (!filePath || !targetName || typeof onSaveXmlAs !== "function") return;
    let serialized;
    try {
      serialized = serializeCurrentXml();
    } catch (error) {
      toast.error(`Failed to serialize ${filePath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
      return;
    }
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("save-as");
    try {
      const response = await onSaveXmlAs(filePath, targetName, serialized);
      if (fileActionRequestRef.current !== requestId) return;
      const nextPath = String(response?.path || "").trim();
      setShowSaveAsDialog(false);
      setSaveAsName("");
      toast.success(`Saved as: ${nextPath || targetName}`);
      if (nextPath && response?.selected !== true && typeof onFilePathChange === "function") {
        onFilePathChange(nextPath);
      }
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to save as ${targetName}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [filePath, onFilePathChange, onSaveXmlAs, saveAsName, serializeCurrentXml]);

  const handleSetDefaultXml = useCallback(async () => {
    if (!filePath || typeof onSetDefaultXml !== "function") return;
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("set-default");
    try {
      await onSetDefaultXml(filePath);
      if (fileActionRequestRef.current !== requestId) return;
      toast.success(`Used for Run: ${filePath}`);
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to use ${filePath} for Run: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [filePath, onSetDefaultXml]);

  const handleToggleCollapse = useCallback((nodeId) => {
    const target = nodesRef.current.find((node) => node.id === nodeId);
    if (!target || target.type !== "btControl") return;
    const nextCollapsed = !target.data?.collapsed;
    captureHistory();
    setNodeDataMap((current) => {
      const next = new Map(current);
      const entry = next.get(nodeId);
      if (entry) next.set(nodeId, { ...entry, collapsed: nextCollapsed });
      return next;
    });
    setNodes((current) => {
      const flipped = current.map((node) => (
        node.id === nodeId
          ? { ...node, data: { ...node.data, collapsed: nextCollapsed } }
          : node
      ));
      return layoutVisibleOnly(flipped, edgesRef.current);
    });
  }, [captureHistory, setNodes]);

  const handleNameChange = useCallback((nodeId, name) => {
    const trimmed = String(name || "").trim();
    if (!trimmed) return;
    captureCoalescedEditHistory(`name:${nodeId}`);
    setNodeDataMap((current) => {
      const next = new Map(current);
      const entry = next.get(nodeId);
      if (entry) next.set(nodeId, { ...entry, name: trimmed });
      return next;
    });
    setNodes((current) => current.map((node) => (
      node.id === nodeId
        ? { ...node, data: { ...node.data, label: trimmed } }
        : node
    )));
  }, [captureCoalescedEditHistory, setNodes]);

  const handleParamChange = useCallback((nodeId, paramName, value) => {
    captureCoalescedEditHistory(`param:${nodeId}:${paramName}`);
    setNodeDataMap((current) => {
      const next = new Map(current);
      const entry = next.get(nodeId);
      if (entry) next.set(nodeId, {
        ...entry,
        params: { ...entry.params, [paramName]: value },
      });
      return next;
    });
    setNodes((current) => current.map((node) => (
      node.id === nodeId
        ? {
          ...node,
          data: {
            ...node.data,
            params: { ...node.data.params, [paramName]: value },
          },
        }
        : node
    )));
  }, [captureCoalescedEditHistory, setNodes]);

  useEffect(() => {
    const handleDelete = (event) => {
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;

      const selectedNodeIds = new Set(
        nodesRef.current
          .filter((node) => node.selected || node.id === selectedNodeId)
          .map((node) => node.id),
      );
      const selectedEdgeIds = new Set(
        edgesRef.current.filter((edge) => edge.selected).map((edge) => edge.id),
      );
      if (selectedNodeIds.size === 0 && selectedEdgeIds.size === 0) return;

      captureHistory();
      const remainingNodes = nodesRef.current.filter((node) => !selectedNodeIds.has(node.id));
      const remainingEdges = edgesRef.current.filter((edge) => (
        !selectedEdgeIds.has(edge.id) &&
        !selectedNodeIds.has(edge.source) &&
        !selectedNodeIds.has(edge.target)
      ));
      const anchorNodeId = findDeletionLayoutAnchor(
        nodesRef.current,
        edgesRef.current,
        selectedNodeIds,
        selectedEdgeIds,
      );
      setNodes(layoutVisibleOnly(remainingNodes, remainingEdges, { anchorNodeId }));
      setEdges(remainingEdges);
      setNodeDataMap((current) => {
        const next = new Map(current);
        selectedNodeIds.forEach((id) => next.delete(id));
        return next;
      });
      if (selectedNodeIds.has(selectedNodeId)) setSelectedNodeId(null);
    };
    document.addEventListener("keydown", handleDelete);
    return () => document.removeEventListener("keydown", handleDelete);
  }, [captureHistory, selectedNodeId, setEdges, setNodes]);

  const annotatedNodes = useMemo(() => {
    const activeSet = new Set(activeNodeNames);
    const hiddenIds = computeHiddenIds(nodes, edges);
    const childrenById = new Map(nodes.map((node) => [node.id, []]));
    const childCount = new Map();
    edges.forEach((edge) => {
      if (childrenById.has(edge.source)) childrenById.get(edge.source).push(edge.target);
      childCount.set(edge.source, (childCount.get(edge.source) || 0) + 1);
    });

    const hasActiveDescendant = (nodeId) => {
      const queue = [...(childrenById.get(nodeId) || [])];
      while (queue.length) {
        const id = queue.shift();
        if (activeSet.has(id)) return true;
        queue.push(...(childrenById.get(id) || []));
      }
      return false;
    };

    return nodes.map((node) => {
      const isControl = node.type === "btControl";
      return {
        ...node,
        hidden: hiddenIds.has(node.id),
        data: {
          ...node.data,
          isActive: activeSet.has(node.id) || (isControl && hasActiveDescendant(node.id)),
          isSelected: node.id === selectedNodeId,
          childCount: childCount.get(node.id) || 0,
          onToggleCollapse: handleToggleCollapse,
        },
      };
    });
  }, [activeNodeNames, edges, handleToggleCollapse, nodes, selectedNodeId]);

  const canClearTree = (
    nodes.length > 0
    && !loading
    && !fileActionsDisabled
    && !fileAction
    && !parseError
    && hydratedPath === filePath
  );

  return (
    <div className="h-full min-h-0 relative flex bg-[var(--mc-bg)] text-[var(--mc-text)]">
      <div
        aria-label="Waypoint Task file actions"
        className="absolute top-3 z-20 flex items-center gap-1 rounded-[10px] border border-[var(--mc-border)] bg-[var(--mc-surface)]/95 p-1 shadow-sm transition-[right]"
        style={{ right: selectedNodeId ? 332 : 12 }}
      >
          <button
            type="button"
            onClick={undoHistory}
            disabled={!canUndo}
            title="Undo"
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              canUndo
                ? "bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)]"
                : "bg-[var(--mc-surface-2)] text-[var(--mc-text-subtle)] opacity-50",
            )}
          >
            <MdUndo size={18} />
          </button>
          <button
            type="button"
            onClick={redoHistory}
            disabled={!canRedo}
            title="Redo"
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              canRedo
                ? "bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)]"
                : "bg-[var(--mc-surface-2)] text-[var(--mc-text-subtle)] opacity-50",
            )}
          >
            <MdRedo size={18} />
          </button>
          <button
            type="button"
            onClick={handleAutoLayout}
            disabled={!nodes.length}
            title="Auto layout"
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              nodes.length
                ? "bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)]"
                : "bg-[var(--mc-surface-2)] text-[var(--mc-text-subtle)] opacity-50",
            )}
          >
            <MdAutoFixHigh size={18} />
          </button>
          <button
            type="button"
            onClick={handleClearTree}
            disabled={!canClearTree}
            aria-label={clearTreeArmed
              ? "Confirm clear current waypoint task"
              : "Clear current waypoint task"}
            title={clearTreeArmed
              ? "Click again to clear the current waypoint task"
              : "Clear current waypoint task"}
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center border transition-colors",
              !canClearTree
                ? "border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50"
                : clearTreeArmed
                  ? "border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)]"
                  : "border-[var(--mc-danger-border)] bg-[var(--mc-surface)] text-[var(--mc-danger)] shadow-[var(--mc-shadow)] hover:bg-[var(--mc-surface-hover)]",
            )}
          >
            <MdDeleteSweep size={18} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => {
              setPendingLoadPath(filePath || availableFileOptions[0] || "");
              setShowLoadDialog(true);
            }}
            disabled={loading || fileActionsDisabled || Boolean(fileAction) || !onLoadXml}
            title={filePath ? `Open ${filePath}` : "Open Task"}
            aria-label="Open Task"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (loading || fileActionsDisabled || fileAction || !onLoadXml) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdUploadFile size={18} />
            {fileAction === "load" ? "Opening..." : "Open Task"}
          </button>
          <button
            type="button"
            onClick={handleSaveXml}
            disabled={(
              loading ||
              fileActionsDisabled ||
              Boolean(fileAction) ||
              Boolean(parseError) ||
              hydratedPath !== filePath ||
              !onSaveXml
            )}
            title={filePath ? `Save Task to ${filePath}` : "Save Task"}
            aria-label="Save Task"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (
                loading ||
                fileActionsDisabled ||
                fileAction ||
                parseError ||
                hydratedPath !== filePath ||
                !onSaveXml
              ) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdSave size={18} />
            {fileAction === "save" ? "Saving..." : "Save"}
          </button>
          <button
            type="button"
            onClick={() => setShowSaveAsDialog(true)}
            disabled={(
              loading
              || fileActionsDisabled
              || Boolean(fileAction)
              || Boolean(parseError)
              || hydratedPath !== filePath
              || !onSaveXmlAs
            )}
            title="Save the current waypoint task as another file"
            aria-label="Save Task As"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (
                loading
                || fileActionsDisabled
                || fileAction
                || parseError
                || hydratedPath !== filePath
                || !onSaveXmlAs
              ) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdDriveFileRenameOutline size={18} />
            {fileAction === "save-as" ? "Saving..." : "Save Task As"}
          </button>
          <button
            type="button"
            onClick={handleSetDefaultXml}
            disabled={(
              loading
              || fileActionsDisabled
              || Boolean(fileAction)
              || !filePath
              || filePath === defaultFilePath
              || !onSetDefaultXml
            )}
            title={filePath === defaultFilePath
              ? "This task is already used when running the mission"
              : "Use this task when the mission runs"}
            aria-label="Use for Run"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (
                loading
                || fileActionsDisabled
                || fileAction
                || !filePath
                || filePath === defaultFilePath
                || !onSetDefaultXml
              ) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdStar size={18} />
            {fileAction === "set-default" ? "Updating..." : "Use for Run"}
          </button>
      </div>

        <BTNodePalette canUpdateCatalog={false} />
        <div
          className="flex-1 min-w-0 relative"
          onDragOver={handleCanvasDragOver}
          onDrop={handleCanvasDrop}
        >
          {loading ? (
            <div className="h-full flex items-center justify-center text-sm text-[var(--mc-text-muted)]">
              Loading waypoint task...
            </div>
          ) : parseError ? (
            <div className="h-full flex items-center justify-center text-center text-[var(--mc-danger)]">
              <div>
                <div className="font-semibold">Parse Error</div>
                <div className="mt-1 text-xs">{parseError}</div>
              </div>
            </div>
          ) : nodes.length === 0 ? (
            <div className="h-full flex items-center justify-center text-center text-[var(--mc-text-subtle)]">
              <div>
                <div className="text-sm font-semibold">No waypoint task</div>
                <div className="mt-1 text-xs">Drag steps from the palette.</div>
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
              onNodeClick={(_event, node) => setSelectedNodeId(node.id)}
              onNodeDragStop={captureHistory}
              fitView
              fitViewOptions={{ padding: 0.18 }}
              nodesDraggable
              nodesConnectable
              elementsSelectable
              deleteKeyCode={null}
              minZoom={0.3}
              maxZoom={2}
              zoomOnScroll
              panOnScroll={false}
              zoomOnPinch
              zoomActivationKeyCode={null}
              autoPanOnConnect={false}
              proOptions={reactFlowProOptions}
            >
              <Controls showInteractive={false} />
              <Background color="#dcd7ca" gap={16} />
            </ReactFlow>
          )}
        </div>
        {selectedNodeId && (
          <BTParamPanel
            nodes={annotatedNodes}
            selectedNodeId={selectedNodeId}
            onParamChange={handleParamChange}
            onNameChange={handleNameChange}
            onClose={() => setSelectedNodeId(null)}
            variant="autonomy-studio"
          />
        )}
        {showLoadDialog && (
          <div
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/45 p-6"
            role="dialog"
            aria-modal="true"
            aria-label="Waypoint Task files"
          >
            <form
              className="w-full max-w-md rounded-xl border border-[var(--mc-border)] bg-[var(--mc-surface)] p-4 shadow-xl"
              onSubmit={(event) => {
                event.preventDefault();
                void handleLoadXml(pendingLoadPath);
              }}
            >
              <div className="text-sm font-semibold">Open Waypoint Task</div>
              <div className="mt-1 text-xs text-[var(--mc-text-subtle)]">
                Opening another task changes the editor only. Choose Use for Run to change mission behavior.
              </div>
              <div className="mt-3 max-h-64 space-y-2 overflow-auto">
                {availableFileOptions.map((path) => (
                  <label
                    key={path}
                    className={clsx(
                      "flex cursor-pointer items-start gap-2 rounded-lg border p-2.5",
                      pendingLoadPath === path
                        ? "border-[var(--mc-accent-hover)] bg-[var(--mc-accent-soft)]"
                        : "border-[var(--mc-border)] bg-[var(--mc-surface-2)]",
                    )}
                  >
                    <input
                      type="radio"
                      name="local-bt-xml"
                      value={path}
                      checked={pendingLoadPath === path}
                      onChange={() => setPendingLoadPath(path)}
                      className="mt-0.5 accent-[var(--mc-accent-hover)]"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-1 text-xs font-semibold">
                        <span className="truncate">{path.split("/").pop()}</span>
                        {path === defaultFilePath && (
                          <span className="rounded bg-[var(--mc-accent-soft)] px-1 py-0.5 text-[9px] text-[var(--mc-accent-hover)]">
                            Run Task
                          </span>
                        )}
                      </span>
                      <span className="block truncate text-[10px] text-[var(--mc-text-subtle)]">
                        {path}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <button
                  type="button"
                  className="rounded-lg border border-[var(--mc-border)] px-3 py-2 text-xs"
                  onClick={() => setShowLoadDialog(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!pendingLoadPath || Boolean(fileAction)}
                  className="rounded-lg bg-[var(--mc-accent-hover)] px-3 py-2 text-xs font-semibold text-[var(--mc-accent-fg)] hover:brightness-90 disabled:opacity-50"
                >
                  {fileAction === "load" ? "Opening..." : "Open Selected"}
                </button>
              </div>
            </form>
          </div>
        )}
        {showSaveAsDialog && (
          <div
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/45 p-6"
            role="dialog"
            aria-modal="true"
            aria-label="Save Waypoint Task As"
          >
            <form
              className="w-full max-w-sm rounded-xl border border-[var(--mc-border)] bg-[var(--mc-surface)] p-4 shadow-xl"
              onSubmit={handleSaveXmlAs}
            >
              <div className="text-sm font-semibold">Save Waypoint Task As</div>
              <div className="mt-1 text-xs text-[var(--mc-text-subtle)]">
                A new task file is added to this waypoint. The task used for Run does not change.
              </div>
              <label className="mt-4 block text-xs font-medium" htmlFor="local-bt-save-as-name">
                New task file name
              </label>
              <input
                id="local-bt-save-as-name"
                aria-label="New task file name"
                value={saveAsName}
                onChange={(event) => setSaveAsName(event.target.value)}
                placeholder="alternate.xml"
                className="mt-1 w-full rounded-lg border border-[var(--mc-border)] bg-[var(--mc-surface-2)] px-3 py-2 text-sm outline-none focus:border-[var(--mc-accent-hover)]"
                autoFocus
              />
              <div className="mt-1 text-[10px] text-[var(--mc-text-subtle)]">
                .xml is added automatically. Use letters, numbers, dot, underscore or hyphen.
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <button
                  type="button"
                  className="rounded-lg border border-[var(--mc-border)] px-3 py-2 text-xs"
                  onClick={() => setShowSaveAsDialog(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!saveAsName.trim() || Boolean(fileAction)}
                  className="rounded-lg bg-[var(--mc-accent-hover)] px-3 py-2 text-xs font-semibold text-[var(--mc-accent-fg)] hover:brightness-90 disabled:opacity-50"
                >
                  {fileAction === "save-as" ? "Saving..." : "Save Task As"}
                </button>
              </div>
            </form>
          </div>
        )}
    </div>
  );
}
