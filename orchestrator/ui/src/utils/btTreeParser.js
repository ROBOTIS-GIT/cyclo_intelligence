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

import dagre from 'dagre';

const CONTROL_TYPES = new Set(['Sequence', 'Loop', 'Fallback', 'Parallel']);

const NODE_WIDTH = 200;
const NODE_HEIGHT = 80;

// Attributes that are internal metadata, not BT node parameters.
// bt_collapsed is a UI-only flag persisted on Control nodes so the
// collapse/expand state survives Save As, page navigation, and file
// reloads. BT.cpp doesn't recognise it and ignores unknown attributes.
const INTERNAL_ATTRS = new Set(['ID', 'name', 'bt_x', 'bt_y', 'bt_collapsed']);

/**
 * Parse BT XML string into React Flow nodes, edges, and a nodeDataMap.
 *
 * nodeDataMap: Map<id, {tag, name, params}> — used as primary state after load.
 * bt_x / bt_y XML attributes override dagre positions when present (set by
 * serializeFromGraph on save, so position is preserved across load/save cycles).
 */
export function parseBTXml(xmlString) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xmlString, 'text/xml');

  const parseError = doc.querySelector('parsererror');
  if (parseError) {
    throw new Error('Invalid XML: ' + parseError.textContent);
  }

  const mainTreeId = doc.documentElement.getAttribute('main_tree_to_execute');
  const behaviorTrees = doc.querySelectorAll('BehaviorTree');

  let rootElement = null;
  for (const bt of behaviorTrees) {
    if (bt.getAttribute('ID') === mainTreeId) {
      rootElement = bt.children[0];
      break;
    }
  }

  if (!rootElement) {
    const firstBT = behaviorTrees[0];
    if (firstBT && firstBT.children.length > 0) {
      rootElement = firstBT.children[0];
    }
  }

  if (!rootElement) {
    return { nodes: [], edges: [], xmlDoc: doc, nodeElementMap: new Map(), nodeDataMap: new Map() };
  }

  const nodes = [];
  const edges = [];
  let nodeIdCounter = 0;
  const nodeElementMap = new Map();
  const nodeDataMap = new Map();

  function traverse(element, parentId) {
    const id = `bt_${nodeIdCounter++}`;
    nodeElementMap.set(id, element);
    const tag = element.tagName;
    const name = element.getAttribute('name') || tag;
    const isControl = CONTROL_TYPES.has(tag);

    const params = {};
    for (const attr of element.attributes) {
      if (!INTERNAL_ATTRS.has(attr.name)) {
        params[attr.name] = attr.value;
      }
    }

    const storedX = element.getAttribute('bt_x');
    const storedY = element.getAttribute('bt_y');
    // Only Control nodes carry a collapsed flag — action nodes have no
    // children to hide. Default false when the attribute is absent.
    const collapsed = isControl && element.getAttribute('bt_collapsed') === 'true';

    nodes.push({
      id,
      type: isControl ? 'btControl' : 'btAction',
      data: isControl
        ? { label: name, nodeType: tag, params, collapsed }
        : { label: name, nodeType: tag, params },
      position: { x: 0, y: 0 },
      _storedX: storedX,
      _storedY: storedY,
    });

    nodeDataMap.set(
      id,
      isControl ? { tag, name, params, collapsed } : { tag, name, params },
    );

    if (parentId) {
      edges.push({
        id: `e_${parentId}_${id}`,
        source: parentId,
        target: id,
        type: 'smoothstep',
        animated: false,
      });
    }

    for (const child of element.children) {
      traverse(child, id);
    }
  }

  traverse(rootElement, null);

  const layout = applyDagreLayout(nodes, edges);
  return { ...layout, xmlDoc: doc, nodeElementMap, nodeDataMap };
}

/**
 * Run dagre on the (nodes, edges) pair and return a new nodes array with
 * computed positions. With `respectStored: true` (the load-from-XML path)
 * the parser-injected `_storedX/_storedY` win — that's how saved layouts
 * survive a reload. With `respectStored: false` (in-app re-layout after
 * a structural change) the stored hints are ignored so dagre is the sole
 * source of truth and the graph stays tidy.
 */
export function applyDagreLayout(nodes, edges, { respectStored = true } = {}) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 60 });

  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  // dagre's child ordering tracks the order edges are inserted into the
  // graph. Sort sibling edges (same source) by the target's current x so
  // users who hand-arranged children left-to-right keep that order after
  // Auto Layout. Edges with different sources compare equal — the sort is
  // stable, so cross-parent ordering stays as-is.
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const xOf = (id) => {
    const n = nodeById.get(id);
    if (!n) return 0;
    if (n.position && typeof n.position.x === 'number') return n.position.x;
    if (
      n._storedX !== null && n._storedX !== undefined && n._storedX !== ''
    ) {
      const v = parseFloat(n._storedX);
      return Number.isNaN(v) ? 0 : v;
    }
    return 0;
  };
  const sortedEdges = [...edges].sort((a, b) => {
    if (a.source !== b.source) return 0;
    return xOf(a.target) - xOf(b.target);
  });
  sortedEdges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  const layoutNodes = nodes.map(({ _storedX, _storedY, ...node }) => {
    if (
      respectStored &&
      _storedX !== null && _storedX !== undefined && _storedX !== '' &&
      _storedY !== null && _storedY !== undefined && _storedY !== ''
    ) {
      return { ...node, position: { x: parseFloat(_storedX), y: parseFloat(_storedY) } };
    }
    const pos = g.node(node.id);
    if (!pos) return node;
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    };
  });

  return { nodes: layoutNodes, edges };
}
