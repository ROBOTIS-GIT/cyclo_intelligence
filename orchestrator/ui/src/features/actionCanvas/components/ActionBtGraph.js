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
import clsx from "clsx";
import BTParamPanel from "../../../components/bt/BTParamPanel";
import BTNodePalette from "../../../components/bt/BTNodePalette";
import { isValidBtConnection } from "../../../utils/btConnection";
import nodeTypes from '../../btEditor/lib/nodeTypes';

export default function ActionBtGraph({ view, runtime, commands, graph }) {
  const { autonomyStudioVariant, annotatedNodes, selectedNodeId, setSelectedNodeId, variant } = view;
  const { isBtNodeUp } = runtime;
  const {
    handleCanvasDragOver,
    handleCanvasDrop,
    handleConnect,
    handleNodeClick,
    handleNodeDragStop,
    handleParamChange,
    handleNameChange,
  } = commands;
  const { parseError, nodes, edges, onNodesChange, onEdgesChange, reactFlowRef, nodesRef, edgesRef } = graph;
  return (<>
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
              <BtExecutionFocus nodes={annotatedNodes} edges={edges} />
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
            onClose={() => setSelectedNodeId(null)}
            variant={variant}
          />
        )}
      </div>


  </>);
}
