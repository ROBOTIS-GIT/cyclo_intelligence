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
import { MdPlayArrow, MdStop, MdUploadFile, MdSave, MdUndo, MdRedo } from 'react-icons/md';

import BTControlNode from '../components/bt/BTControlNode';
import BTActionNode from '../components/bt/BTActionNode';
import BTParamPanel from '../components/bt/BTParamPanel';
import BTNodePalette, { PALETTE_DRAG_MIME } from '../components/bt/BTNodePalette';
import TreeListModal from '../features/btmanager/components/TreeListModal';
import { parseBTXml } from '../utils/btTreeParser';
import { serializeFromGraph } from '../utils/btXmlSerializer';
import { setTreeXml, setTreeFileName, setBtStatus, setActiveNodeNames, setSelectedNodeId } from '../features/btmanager/btmanagerSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { useBTHistory } from '../hooks/useBTHistory';
import { findNodeMeta, isControlTag } from '../constants/btNodeCatalog';

const nodeTypes = {
  btControl: BTControlNode,
  btAction: BTActionNode,
};

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
    setNodes((prev) => [
      ...prev,
      {
        id,
        type: isControlTag(tag) ? 'btControl' : 'btAction',
        position,
        data: { label: autoName, nodeType: tag, params },
      },
    ]);
    setNodeDataMap((prev) => new Map(prev).set(id, { tag, name: autoName, params }));
    dispatch(setSelectedNodeId(id));
  }, [captureHistory, setNodes, dispatch]);

  // ── Manual edge connection ────────────────────────────────────────────────
  const handleConnect = useCallback((connection) => {
    captureHistory();
    setEdges((prev) => addEdge({ ...connection, type: 'smoothstep', animated: false }, prev));
  }, [captureHistory, setEdges]);

  // ── Node drag stop: just capture history (ReactFlow updates position) ─────
  const handleNodeDragStop = useCallback(() => {
    captureHistory();
  }, [captureHistory]);

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

  // ── Delete key: remove selected nodes + their edges ───────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;
      const selectedIds = new Set(currentNodes.filter((n) => n.selected).map((n) => n.id));
      if (selectedIds.size === 0) return;

      captureHistory();
      setNodes((ns) => ns.filter((n) => !selectedIds.has(n.id)));
      setEdges((es) =>
        es.filter((e) => !selectedIds.has(e.source) && !selectedIds.has(e.target))
      );
      setNodeDataMap((prev) => {
        const next = new Map(prev);
        selectedIds.forEach((id) => next.delete(id));
        return next;
      });
      dispatch(setSelectedNodeId(null));
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [setNodes, setEdges, dispatch, captureHistory]);

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

  // ── Annotate nodes with isActive / isSelected ────────────────────────────
  const annotatedNodes = useMemo(() => {
    const activeSet = new Set(activeNodeNames);
    return nodes.map((node) => ({
      ...node,
      data: {
        ...node.data,
        isActive: activeSet.has(node.id),
        isSelected: node.id === selectedNodeId,
      },
    }));
  }, [nodes, activeNodeNames, selectedNodeId]);

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
