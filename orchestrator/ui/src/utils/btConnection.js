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

/**
 * Shared BT edge validation for the Autonomy Studio Behavior Tree editors.
 *
 * Standard behavior-tree structure: only control nodes are internal — actions
 * are always leaves, so an edge may only leave a btControl node. The engine
 * enforces the same rule at load time (bt_nodes_loader recurses into children
 * for BaseControl only and silently drops children of actions), so an edge
 * out of an action would draw fine but never execute.
 *
 * Also rejects a second parent for the target and any edge that would close
 * a cycle.
 */
export function isValidBtConnection(connection, nodes, edges) {
  const source = connection?.source;
  const target = connection?.target;
  if (!source || !target || source === target) return false;
  const sourceNode = nodes.find((node) => node.id === source);
  if (!sourceNode || sourceNode.type !== "btControl") return false;
  if (edges.some((edge) => edge.target === target)) return false;

  // Adding source -> target is cyclic when target already reaches source.
  const queue = [target];
  const visited = new Set();
  while (queue.length) {
    const nodeId = queue.shift();
    if (nodeId === source) return false;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    edges.forEach((edge) => {
      if (edge.source === nodeId) queue.push(edge.target);
    });
  }
  return true;
}
