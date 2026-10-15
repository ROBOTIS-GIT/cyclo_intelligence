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

import React, { useCallback, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import '@xyflow/react/dist/style.css';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import { useBTNodeCatalog } from '../../../hooks/useBTNodeCatalog';
import { serializeFromGraph } from '../../../utils/btXmlSerializer';
import { setSelectedNodeId } from '../actionCanvasSlice';
import { selectBtSupportedRobotTypes } from '../btSupportSlice';
import { normalizeBtStatus } from '../lib/btRuntimeStatus';
import btRuntimeView from '../lib/btRuntimeView';
import useBtGraph from '../../btEditor/hooks/useBtGraph';
import useBtGraphCommands from '../../btEditor/hooks/useBtGraphCommands';
import useBtGraphDeletion from '../../btEditor/hooks/useBtGraphDeletion';
import useBtUndoKeys from '../../btEditor/hooks/useBtUndoKeys';
import useBtRenderedNodes from '../../btEditor/hooks/useBtRenderedNodes';
import useBtClearConfirmation from '../../btEditor/hooks/useBtClearConfirmation';
import useActionBtPersistence from '../hooks/useActionBtPersistence';
import useActionBtFiles from '../hooks/useActionBtFiles';
import useActionBtHistory from '../hooks/useActionBtHistory';
import useActionBtClear from '../hooks/useActionBtClear';
import useActionBtExecution from '../hooks/useActionBtExecution';
import useBtEngineLifecycle from '../hooks/useBtEngineLifecycle';
import useBtRuntimeSubscription from '../hooks/useBtRuntimeSubscription';
import ActionBtHeader from './ActionBtHeader';
import ActionBtGraph from './ActionBtGraph';
import ActionBtRuntimeBar from './ActionBtRuntimeBar';
import ActionBtDialogs from './ActionBtDialogs';

// This composition owns no ROS lifecycle or file I/O; each adapter shares the
// same graph, while Mission's editor supplies a different document adapter.
export default function BTEditorSurface({ isActive = true, title = 'Action Canvas',
  subtitle = '', className = 'w-full h-full', variant = 'legacy', onExitStateChange }) {
  const dispatch = useDispatch();
  const { callService } = useRosServiceCaller();
  const { catalog: nodeCatalog = [], refreshCatalog } = useBTNodeCatalog();
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const robotType = useSelector((state) => state.tasks.robotType);
  const supportedRobotTypes = useSelector(selectBtSupportedRobotTypes);
  const { treeXml, treeFileName, btStatus, activeNodeNames, selectedNodeId } =
    useSelector((state) => state.actionCanvas);
  const selectNode = useCallback((id) => dispatch(setSelectedNodeId(id)), [dispatch]);
  const graph = useBtGraph();
  const history = useActionBtHistory({ graph, treeFileName, dispatch });
  const confirmation = useBtClearConfirmation();
  useActionBtPersistence({ graph, treeXml, dispatch });
  const getSerializedXml = useCallback(() => serializeFromGraph(
    graph.nodes, graph.edges, graph.nodeDataMap,
  ), [graph.nodes, graph.edges, graph.nodeDataMap]);
  const files = useActionBtFiles({ graph, dispatch, getSerializedXml,
    disarmClearTree: confirmation.disarmClearTree, resetHistory: history.resetHistory });
  const engine = useBtEngineLifecycle({ dispatch, isActive, robotType,
    supportedRobotTypes, btStatus, callService, refreshCatalog });
  const execution = useActionBtExecution({ nodes: graph.nodes, nodeDataMap: graph.nodeDataMap,
    btNodeStatus: engine.btNodeStatus, btStatus, dispatch, callService, getSerializedXml });
  useBtRuntimeSubscription({ rosbridgeUrl, isActive, dispatch });
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const hasTree = graph.nodes.length > 0;
  const runtime = { ...engine, ...execution,
    ...btRuntimeView({ ...engine, ...execution, btStatus, hasTree, autonomyStudioVariant }) };
  const handleClearTree = useActionBtClear({ graph, history, confirmation,
    runtime, dispatch, selectNode, btStatus });
  const graphCommands = useBtGraphCommands({ graph, nodeCatalog,
    captureHistory: history.captureHistory, setSelectedNodeId: selectNode });
  useBtGraphDeletion({ graph, captureHistory: history.captureHistory,
    selectedNodeId, setSelectedNodeId: selectNode });
  useBtUndoKeys(history);
  const commands = { ...graphCommands, handleClearTree,
    handleNodeClick: (_event, node) => selectNode(node.id),
    handleNodeDragStop: history.captureHistory };
  const annotatedNodes = useBtRenderedNodes({ nodes: graph.nodes, edges: graph.edges,
    activeNodeNames, selectedNodeId, handleToggleCollapse: commands.handleToggleCollapse });
  const view = { title, subtitle, variant, className, autonomyStudioVariant,
    treeFileName, selectedNodeId, setSelectedNodeId: selectNode, annotatedNodes, hasTree,
    canClearTree: runtime.canClearTree, clearTreeArmed: confirmation.clearTreeArmed };

  useEffect(() => {
    if (typeof onExitStateChange !== 'function') return;
    onExitStateChange({ active: ['running', 'stopping'].includes(normalizeBtStatus(btStatus)),
      busy: Boolean(engine.btNodePendingAction || execution.btExecutionPending) });
  }, [btStatus, engine.btNodePendingAction, execution.btExecutionPending, onExitStateChange]);

  return (
    <div data-variant={variant} className={clsx(className, 'bt-editor-surface flex flex-col',
      autonomyStudioVariant && 'bg-[var(--mc-bg)] text-[var(--mc-text)]')}>
      <ActionBtHeader view={view} history={history} commands={commands} files={files} />
      <ActionBtGraph view={view} runtime={runtime} commands={commands} graph={graph} />
      <ActionBtRuntimeBar view={view} runtime={runtime} />
      <ActionBtDialogs files={files} view={view} />
    </div>
  );
}
