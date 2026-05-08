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

/**
 * Parse BT XML string into React Flow nodes and edges.
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
    // Fallback: use the first BehaviorTree
    const firstBT = behaviorTrees[0];
    if (firstBT && firstBT.children.length > 0) {
      rootElement = firstBT.children[0];
    }
  }

  if (!rootElement) {
    return { nodes: [], edges: [] };
  }

  const nodes = [];
  const edges = [];
  let nodeIdCounter = 0;

  function traverse(element, parentId) {
    const id = `bt_${nodeIdCounter++}`;
    const tag = element.tagName;
    const name = element.getAttribute('name') || tag;
    const isControl = CONTROL_TYPES.has(tag);

    // Collect params (skip ID and name)
    const params = {};
    for (const attr of element.attributes) {
      if (attr.name !== 'ID' && attr.name !== 'name') {
        params[attr.name] = attr.value;
      }
    }

    nodes.push({
      id,
      type: isControl ? 'btControl' : 'btAction',
      data: {
        label: name,
        nodeType: tag,
        params,
      },
      position: { x: 0, y: 0 }, // Will be set by dagre
    });

    if (parentId) {
      edges.push({
        id: `e_${parentId}_${id}`,
        source: parentId,
        target: id,
        type: 'smoothstep',
        animated: false,
      });
    }

    // Recurse into children (skip non-element nodes)
    for (const child of element.children) {
      traverse(child, id);
    }
  }

  traverse(rootElement, null);

  // Apply dagre layout
  return applyDagreLayout(nodes, edges);
}

function applyDagreLayout(nodes, edges) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 60 });

  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  const layoutNodes = nodes.map((node) => {
    const pos = g.node(node.id);
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
