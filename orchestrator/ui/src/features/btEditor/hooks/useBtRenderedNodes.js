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

import { useMemo } from 'react';
import { computeHiddenIds } from '../lib/btGraph';

export default function useBtRenderedNodes({ nodes, edges, activeNodeNames, selectedNodeId, handleToggleCollapse }) {
  const annotatedNodes = useMemo(() => {
    const activeSet = new Set(activeNodeNames);
    const hiddenIds = computeHiddenIds(nodes, edges);
    const childrenById = new Map(nodes.map((n) => [n.id, []]));
    const childCount = new Map();
    for (const e of edges) {
      if (childrenById.has(e.source)) childrenById.get(e.source).push(e.target);
      childCount.set(e.source, (childCount.get(e.source) ?? 0) + 1);
    }
    // Bubble active-state up from leaves to ancestor Control nodes so the
    // user can tell a Loop/Sequence is "live" even when collapsed — the
    // active leaf itself is hidden under the +/- toggle, but the Control
    // wrapper still pulses.
    const hasActiveDescendant = (rootId) => {
      const queue = [...(childrenById.get(rootId) || [])];
      while (queue.length) {
        const id = queue.shift();
        if (activeSet.has(id)) return true;
        const kids = childrenById.get(id);
        if (kids && kids.length) queue.push(...kids);
      }
      return false;
    };
    return nodes.map((node) => {
      const directly = activeSet.has(node.id);
      const isControl = node.type === 'btControl';
      const isActive = directly || (isControl && hasActiveDescendant(node.id));
      return {
        ...node,
        hidden: hiddenIds.has(node.id),
        data: {
          ...node.data,
          isActive,
          isSelected: node.id === selectedNodeId,
          childCount: childCount.get(node.id) ?? 0,
          onToggleCollapse: handleToggleCollapse,
        },
      };
    });
  }, [nodes, edges, activeNodeNames, selectedNodeId, handleToggleCollapse]);

  return annotatedNodes;
}
