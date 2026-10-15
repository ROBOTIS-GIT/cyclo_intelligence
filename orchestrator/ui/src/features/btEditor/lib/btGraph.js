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

import { applyDagreLayout } from '../../../utils/btTreeParser';

export function collectDescendants(rootId, edges) {
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
export function computeHiddenIds(nodes, edges) {
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
export function layoutVisibleOnly(nodes, edges, { anchorNodeId = null } = {}) {
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

export function catalogEntryToParams(entry) {
  return Object.fromEntries(
    (entry?.ports || []).map((port) => [port.name, port.default]),
  );
}
