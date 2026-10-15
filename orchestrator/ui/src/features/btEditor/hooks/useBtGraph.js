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

import { useRef, useState } from 'react';
import { useNodesState, useEdgesState } from '@xyflow/react';

// One graph owner per editor. Document and command adapters share these refs;
// neither creates a second copy of the graph while an async file load completes.
export default function useBtGraph() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [nodeDataMap, setNodeDataMap] = useState(new Map());
  const [parseError, setParseError] = useState(null);
  const reactFlowRef = useRef(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const nodeDataMapRef = useRef(nodeDataMap);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  nodeDataMapRef.current = nodeDataMap;
  return { nodes, edges, nodeDataMap, parseError,
    setNodes, setEdges, setNodeDataMap, setParseError,
    onNodesChange, onEdgesChange, reactFlowRef, nodesRef, edgesRef, nodeDataMapRef };
}
