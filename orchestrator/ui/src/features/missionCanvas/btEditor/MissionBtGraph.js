import BtExecutionFocus from "../../btEditor/components/BtExecutionFocus";
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

import React from 'react';
import { ReactFlow, Controls, Background } from "@xyflow/react";
import BTNodePalette from "../../../components/bt/BTNodePalette";
import BTParamPanel from "../../../components/bt/BTParamPanel";
import { isValidBtConnection } from "../../../utils/btConnection";
import nodeTypes from '../../btEditor/lib/nodeTypes';

export default function MissionBtGraph({ commands, options, graph, view, history }) {
  const {
    handleCanvasDragOver,
    handleCanvasDrop,
    handleConnect,
    handleParamChange,
    handleNameChange,
  } = commands;
  const { loading } = options;
  const { parseError, nodes, edges, onNodesChange, onEdgesChange, reactFlowRef, nodesRef, edgesRef } = graph;
  const { annotatedNodes, setSelectedNodeId, selectedNodeId } = view;
  const { captureHistory } = history;
  return (<>
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
              proOptions={{ hideAttribution: true }}
            >
              <BtExecutionFocus nodes={annotatedNodes} edges={edges} />
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

  </>);
}
