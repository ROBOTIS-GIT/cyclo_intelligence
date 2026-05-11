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
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdPlayArrow, MdStop, MdUploadFile, MdSave } from 'react-icons/md';

import BTControlNode from '../components/bt/BTControlNode';
import BTActionNode from '../components/bt/BTActionNode';
import BTParamPanel from '../components/bt/BTParamPanel';
import TreeListModal from '../features/btmanager/components/TreeListModal';
import { parseBTXml } from '../utils/btTreeParser';
import { setTreeXml, setTreeFileName, setBtStatus, setActiveNodeNames, setSelectedNodeId } from '../features/btmanager/btmanagerSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

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
  const [parseError, setParseError] = useState(null);
  const [showTreeList, setShowTreeList] = useState(false);
  const [xmlDoc, setXmlDoc] = useState(null);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveFileName, setSaveFileName] = useState('');
  const nodeElementMapRef = useRef(new Map());
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  // Parse XML and update flow whenever treeXml changes
  useEffect(() => {
    if (!treeXml) {
      setNodes([]);
      setEdges([]);
      setXmlDoc(null);
      nodeElementMapRef.current = new Map();
      setParseError(null);
      return;
    }

    try {
      const { nodes: newNodes, edges: newEdges, xmlDoc: doc, nodeElementMap } = parseBTXml(treeXml);
      setNodes(newNodes);
      setEdges(newEdges);
      setXmlDoc(doc);
      nodeElementMapRef.current = nodeElementMap;
      setParseError(null);
    } catch (err) {
      setParseError(err.message);
      setNodes([]);
      setEdges([]);
      setXmlDoc(null);
      nodeElementMapRef.current = new Map();
    }
  }, [treeXml, setNodes, setEdges]);

  // Handle tree selection from TreeListModal
  const handleServerFileSelect = useCallback(async (item) => {
    if (!item || !item.full_path) return;

    try {
      const urlMatch = rosbridgeUrl.match(/ws:\/\/([^:]+):/);
      const host = urlMatch ? urlMatch[1] : 'localhost';
      const fileUrl = `http://${host}:8082${item.full_path}`;
      const response = await fetch(fileUrl);

      if (!response.ok) {
        throw new Error(`Failed to fetch file: ${response.status}`);
      }

      const xmlContent = await response.text();
      const fileName = item.name || item.full_path.split('/').pop();

      // Parse directly so the canvas always resets even when treeXml content is unchanged
      try {
        const { nodes: newNodes, edges: newEdges, xmlDoc: doc, nodeElementMap } = parseBTXml(xmlContent);
        setNodes(newNodes);
        setEdges(newEdges);
        setXmlDoc(doc);
        nodeElementMapRef.current = nodeElementMap;
        setParseError(null);
      } catch (parseErr) {
        setParseError(parseErr.message);
        setNodes([]);
        setEdges([]);
        setXmlDoc(null);
        nodeElementMapRef.current = new Map();
      }

      dispatch(setSelectedNodeId(null));
      dispatch(setTreeXml(xmlContent));
      dispatch(setTreeFileName(fileName));
      toast.success(`Loaded: ${fileName}`);
    } catch (err) {
      toast.error(`Failed to load file: ${err.message}`);
    }
  }, [rosbridgeUrl, dispatch, setNodes, setEdges]);

  // Node click handler for param editing
  const handleNodeClick = useCallback((event, node) => {
    dispatch(setSelectedNodeId(node.id));
  }, [dispatch]);

  // Node drag stop: reorder XML DOM children to match visual left-to-right order
  const handleNodeDragStop = useCallback((_event, draggedNode) => {
    const parentEdge = edgesRef.current.find((e) => e.target === draggedNode.id);
    if (!parentEdge) return; // root node has no siblings

    const parentId = parentEdge.source;
    const siblingIds = edgesRef.current
      .filter((e) => e.source === parentId)
      .map((e) => e.target);

    const sorted = nodesRef.current
      .filter((n) => siblingIds.includes(n.id))
      .sort((a, b) => a.position.x - b.position.x);

    const parentEl = nodeElementMapRef.current.get(parentId);
    if (!parentEl) return;

    sorted.forEach((n) => {
      const el = nodeElementMapRef.current.get(n.id);
      if (el) parentEl.appendChild(el); // moves existing child to end
    });
  }, []);

  // Param change handler: update xmlDoc DOM + nodes state directly (no Redux treeXml change)
  const handleParamChange = useCallback((nodeId, paramName, value) => {
    // Update XML DOM
    const el = nodeElementMapRef.current.get(nodeId);
    if (el) {
      el.setAttribute(paramName, value);
    }
    // Update nodes state in-place so BTParamPanel reflects the value without re-parse
    setNodes((ns) =>
      ns.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [paramName]: value } } }
          : n
      )
    );
  }, [setNodes]);

  // Delete key handler: cascade-delete selected nodes and sync XML DOM
  useEffect(() => {
    const handler = (e) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;
      const selectedIds = new Set(currentNodes.filter((n) => n.selected).map((n) => n.id));
      if (selectedIds.size === 0) return;

      // BFS to find all descendants
      const toDeleteIds = new Set(selectedIds);
      let changed = true;
      while (changed) {
        changed = false;
        for (const edge of currentEdges) {
          if (toDeleteIds.has(edge.source) && !toDeleteIds.has(edge.target)) {
            toDeleteIds.add(edge.target);
            changed = true;
          }
        }
      }

      // Remove top-level deleted elements from XML DOM (children cascade automatically)
      const map = nodeElementMapRef.current;
      toDeleteIds.forEach((id) => {
        const el = map.get(id);
        if (el && el.parentNode) {
          const parentEl = el.parentNode;
          const parentId = [...map.entries()].find(([, e]) => e === parentEl)?.[0];
          if (!parentId || !toDeleteIds.has(parentId)) {
            parentEl.removeChild(el);
          }
        }
        map.delete(id);
      });

      setNodes((ns) => ns.filter((n) => !toDeleteIds.has(n.id)));
      setEdges((es) => es.filter((e) => !toDeleteIds.has(e.source) && !toDeleteIds.has(e.target)));
      dispatch(setSelectedNodeId(null));
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [setNodes, setEdges, dispatch]);

  // Helper: get HTTP server base URL from rosbridgeUrl
  const getHttpBaseUrl = useCallback(() => {
    const urlMatch = rosbridgeUrl.match(/ws:\/\/([^:]+):/);
    const host = urlMatch ? urlMatch[1] : 'localhost';
    return `http://${host}:8082`;
  }, [rosbridgeUrl]);

  // Save As handler
  const handleSaveAs = useCallback(async () => {
    const name = saveFileName.trim();
    if (!name) return;

    const content = xmlDoc
      ? new XMLSerializer().serializeToString(xmlDoc)
      : treeXml;
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
  }, [saveFileName, xmlDoc, treeXml, getHttpBaseUrl]);

  // BT Start - launch node, load XML, and run
  const handleStart = useCallback(async () => {
    if (!treeXml) {
      toast.error('No tree loaded');
      return;
    }
    try {
      // Serialize current xmlDoc (reflects any node deletions) or fall back to raw XML
      const currentXml = xmlDoc
        ? new XMLSerializer().serializeToString(xmlDoc)
        : treeXml;

      // 1. Launch BT node if not running
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

      // 2. Wait for BT node to fully initialize
      const isAlreadyRunning = launchData.message.includes('already running');
      if (!isAlreadyRunning) {
        await new Promise((resolve) => setTimeout(resolve, 8000));
      }

      // 3. Load XML and start execution
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
  }, [callService, dispatch, treeXml, xmlDoc, getHttpBaseUrl]);

  // BT Stop - stop tree and shutdown node
  const handleStop = useCallback(async () => {
    try {
      // 1. Stop BT execution
      try {
        await callService(
          '/bt/set_running',
          'std_srvs/srv/SetBool',
          { data: false }
        );
      } catch {
        // BT node may already be gone, continue to shutdown
      }

      // 2. Shutdown BT node process
      const baseUrl = getHttpBaseUrl();
      await fetch(`${baseUrl}/bt/shutdown`, { method: 'POST' });

      dispatch(setBtStatus('stopped'));
      dispatch(setActiveNodeNames([]));
      toast.success('BT stopped');
    } catch (err) {
      toast.error(`Failed to stop BT: ${err.message}`);
    }
  }, [callService, dispatch, getHttpBaseUrl]);

  // Subscribe to BT status topic
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
          if (msg.data !== 'running') {
            dispatch(setActiveNodeNames([]));
          }
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

  // Annotate nodes with isActive flag based on active node IDs
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
            onClick={() => {
              setSaveFileName(treeFileName ? treeFileName.replace(/\.xml$/i, '') : '');
              setShowSaveDialog(true);
            }}
            disabled={!treeXml}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg',
              'text-sm font-medium transition-colors duration-150',
              treeXml
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
        <div className="flex-1 relative">
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
              <p className="text-sm mt-1">Click "Load XML" to select a tree file</p>
            </div>
          </div>
        ) : (
          <ReactFlow
            nodes={annotatedNodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            nodesDraggable={true}
            nodesConnectable={false}
            elementsSelectable={true}
            onNodeClick={handleNodeClick}
            onNodeDragStop={handleNodeDragStop}
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
            disabled={btStatus === 'running' || btStatus === 'completed' || !treeXml}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              (btStatus === 'running' || btStatus === 'completed' || !treeXml)
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
