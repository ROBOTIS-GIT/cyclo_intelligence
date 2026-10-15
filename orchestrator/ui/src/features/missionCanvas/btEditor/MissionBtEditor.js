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

import React, { useState } from 'react';
import toast from 'react-hot-toast';
import '@xyflow/react/dist/style.css';
import { useBTNodeCatalog } from '../../../hooks/useBTNodeCatalog';
import useBtGraph from '../../btEditor/hooks/useBtGraph';
import useBtGraphCommands from '../../btEditor/hooks/useBtGraphCommands';
import useBtGraphDeletion from '../../btEditor/hooks/useBtGraphDeletion';
import useBtRenderedNodes from '../../btEditor/hooks/useBtRenderedNodes';
import useBtClearConfirmation from '../../btEditor/hooks/useBtClearConfirmation';
import useMissionBtHistory from './useMissionBtHistory';
import useMissionBtDocument from './useMissionBtDocument';
import useMissionBtFiles from './useMissionBtFiles';
import useMissionBtClear from './useMissionBtClear';
import MissionBtToolbar from './MissionBtToolbar';
import MissionBtGraph from './MissionBtGraph';
import MissionBtFileDialogs from './MissionBtFileDialogs';

const invalidConnection = () => toast.error('Each step can have only one parent, and connections cannot form a loop');

export default function MissionBtEditor({ filePath, fileOptions = [], defaultFilePath = '',
  xml, loading = false, activeNodeNames = [], onXmlChange, onLoadXml, onSaveXml,
  onFilePathChange, onSaveXmlAs, onSetDefaultXml, fileActionsDisabled = false }) {
  const { catalog: nodeCatalog = [] } = useBTNodeCatalog();
  const graph = useBtGraph();
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const history = useMissionBtHistory({ graph, setSelectedNodeId });
  const confirmation = useBtClearConfirmation();
  const hydratedPath = useMissionBtDocument({
    graph,
    history,
    filePath,
    xml,
    onXmlChange,
    setSelectedNodeId,
  });
  const files = useMissionBtFiles({ graph, filePath, fileOptions, onLoadXml,
    onSaveXml, onFilePathChange, onSaveXmlAs, onSetDefaultXml,
    disarmClearTree: confirmation.disarmClearTree });
  const handleClearTree = useMissionBtClear({ graph, history, confirmation, loading,
    fileActionsDisabled, fileAction: files.fileAction, hydratedPath, filePath, setSelectedNodeId });
  const graphCommands = useBtGraphCommands({ graph, nodeCatalog, setSelectedNodeId,
    captureHistory: history.captureHistory, captureEditHistory: history.captureCoalescedEditHistory,
    onInvalidConnection: invalidConnection });
  useBtGraphDeletion({ graph, captureHistory: history.captureHistory,
    selectedNodeId, setSelectedNodeId, includeInspectorSelection: true });
  const commands = { ...graphCommands, handleClearTree };
  const annotatedNodes = useBtRenderedNodes({ nodes: graph.nodes, edges: graph.edges,
    activeNodeNames, selectedNodeId, handleToggleCollapse: commands.handleToggleCollapse });
  const canClearTree = graph.nodes.length > 0 && !loading && !fileActionsDisabled
    && !files.fileAction && !graph.parseError && hydratedPath === filePath;
  const view = { selectedNodeId, setSelectedNodeId, annotatedNodes, canClearTree,
    hydratedPath, clearTreeArmed: confirmation.clearTreeArmed };
  const options = { filePath, defaultFilePath, loading, fileActionsDisabled,
    onLoadXml, onSaveXml, onSaveXmlAs, onSetDefaultXml };
  return (
    <div className="h-full min-h-0 relative flex bg-[var(--mc-bg)] text-[var(--mc-text)]">
      <MissionBtToolbar view={view} history={history} commands={commands} graph={graph} files={files} options={options} />
      <MissionBtGraph commands={commands} options={options} graph={graph} view={view} history={history} />
      <MissionBtFileDialogs files={files} options={options} />
    </div>
  );
}
