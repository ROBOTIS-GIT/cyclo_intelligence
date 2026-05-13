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

const INDENT = '  ';
const WRAP_THRESHOLD = 100;

function escapeAttr(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/"/g, '&quot;');
}

function escapeText(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function isWhitespaceOnly(text) {
  return !/\S/.test(text);
}

function elementChildren(node) {
  const out = [];
  for (const child of node.childNodes) {
    if (child.nodeType === 1) out.push(child); // ELEMENT_NODE
  }
  return out;
}

function renderableChildren(node) {
  // Children we emit: elements, comments, and non-whitespace text nodes.
  const out = [];
  for (const child of node.childNodes) {
    if (child.nodeType === 1) {
      out.push(child); // element
    } else if (child.nodeType === 8) {
      out.push(child); // comment
    } else if (child.nodeType === 3 && !isWhitespaceOnly(child.nodeValue)) {
      out.push(child); // non-whitespace text
    }
  }
  return out;
}

function attrPairs(el) {
  const pairs = [];
  for (const attr of el.attributes) {
    pairs.push(`${attr.name}="${escapeAttr(attr.value)}"`);
  }
  return pairs;
}

function serializeElement(el, depth, lines) {
  const indent = INDENT.repeat(depth);
  const tag = el.tagName;
  const pairs = attrPairs(el);
  const children = renderableChildren(el);
  const hasElementChildren = elementChildren(el).length > 0;

  if (children.length === 0) {
    // Self-closing
    const singleLine = pairs.length
      ? `<${tag} ${pairs.join(' ')}/>`
      : `<${tag}/>`;
    if (indent.length + singleLine.length <= WRAP_THRESHOLD || pairs.length <= 1) {
      lines.push(indent + singleLine);
      return;
    }
    const contIndent = indent + INDENT;
    lines.push(`${indent}<${tag} ${pairs[0]}`);
    for (let i = 1; i < pairs.length - 1; i++) {
      lines.push(`${contIndent}${pairs[i]}`);
    }
    lines.push(`${contIndent}${pairs[pairs.length - 1]}/>`);
    return;
  }

  // Has children: decide open-tag layout
  const openSingle = pairs.length
    ? `<${tag} ${pairs.join(' ')}>`
    : `<${tag}>`;
  const closeTag = `</${tag}>`;

  // Pure non-whitespace text content (e.g. <input_port>...</input_port>) — inline if it fits.
  if (!hasElementChildren && children.length === 1 && children[0].nodeType === 3) {
    const text = escapeText(children[0].nodeValue.trim());
    const singleLine = `${openSingle}${text}${closeTag}`;
    if (indent.length + singleLine.length <= WRAP_THRESHOLD && !text.includes('\n')) {
      lines.push(indent + singleLine);
      return;
    }
    // Wrap: text on its own indented lines.
    if (indent.length + openSingle.length <= WRAP_THRESHOLD || pairs.length <= 1) {
      lines.push(indent + openSingle);
    } else {
      const contIndent = indent + INDENT;
      lines.push(`${indent}<${tag} ${pairs[0]}`);
      for (let i = 1; i < pairs.length - 1; i++) {
        lines.push(`${contIndent}${pairs[i]}`);
      }
      lines.push(`${contIndent}${pairs[pairs.length - 1]}>`);
    }
    const textIndent = indent + INDENT;
    for (const line of text.split('\n')) {
      lines.push(textIndent + line.trim());
    }
    lines.push(indent + closeTag);
    return;
  }

  // Element/mixed children — open tag, recurse, close tag.
  if (indent.length + openSingle.length <= WRAP_THRESHOLD || pairs.length <= 1) {
    lines.push(indent + openSingle);
  } else {
    const contIndent = indent + INDENT;
    lines.push(`${indent}<${tag} ${pairs[0]}`);
    for (let i = 1; i < pairs.length - 1; i++) {
      lines.push(`${contIndent}${pairs[i]}`);
    }
    lines.push(`${contIndent}${pairs[pairs.length - 1]}>`);
  }
  for (const child of children) {
    if (child.nodeType === 1) {
      serializeElement(child, depth + 1, lines);
    } else if (child.nodeType === 8) {
      lines.push(`${indent}${INDENT}<!--${child.nodeValue}-->`);
    } else if (child.nodeType === 3) {
      const textIndent = indent + INDENT;
      for (const line of escapeText(child.nodeValue.trim()).split('\n')) {
        lines.push(textIndent + line.trim());
      }
    }
  }
  lines.push(indent + closeTag);
}

/**
 * Build a BT XML string from a free-form ReactFlow graph.
 *
 * nodes      — ReactFlow node array (need .id and .position)
 * edges      — ReactFlow edge array (source → target = parent → child)
 * nodeDataMap — Map<id, {tag, name, params}>
 *
 * The node with in-degree 0 that has children (or is a control type) becomes
 * the main tree root.  Nodes not reachable from the main root are stored in a
 * separate <BehaviorTree ID="__pending__"> section that BT.cpp ignores.
 * Positions are saved as bt_x / bt_y attributes so the next load restores the
 * free-form layout instead of running dagre.
 */
export function serializeFromGraph(nodes, edges, nodeDataMap) {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  // children map: parentId → [childId] (will be sorted by x position)
  const children = new Map(nodes.map((n) => [n.id, []]));
  const inDegree = new Map(nodes.map((n) => [n.id, 0]));

  edges.forEach((e) => {
    if (children.has(e.source)) children.get(e.source).push(e.target);
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
  });

  // Sort children left-to-right by x position
  children.forEach((childIds) => {
    childIds.sort(
      (a, b) => (nodeById.get(a)?.position.x ?? 0) - (nodeById.get(b)?.position.x ?? 0)
    );
  });

  // Find roots (no incoming edges)
  const roots = nodes.filter((n) => (inDegree.get(n.id) ?? 0) === 0);
  // Main root: prefer the root that has outgoing edges (i.e., is a parent)
  const mainRoot =
    roots.find((r) => (children.get(r.id)?.length ?? 0) > 0) ?? roots[0] ?? null;

  // BFS to find all nodes reachable from main root
  function getReachable(rootId) {
    const visited = new Set();
    const queue = rootId ? [rootId] : [];
    while (queue.length) {
      const id = queue.shift();
      if (visited.has(id)) continue;
      visited.add(id);
      (children.get(id) ?? []).forEach((c) => queue.push(c));
    }
    return visited;
  }

  const reachableFromMain = mainRoot ? getReachable(mainRoot.id) : new Set();
  const pendingNodes = nodes.filter((n) => !reachableFromMain.has(n.id));

  const xmlDoc = new DOMParser().parseFromString('<root/>', 'text/xml');
  const rootEl = xmlDoc.documentElement;
  rootEl.setAttribute('BTCPP_format', '4');
  rootEl.setAttribute('main_tree_to_execute', 'MainTree');

  function buildEl(nodeId) {
    const data = nodeDataMap.get(nodeId);
    if (!data) return null;
    const { tag, name, params, collapsed } = data;
    const el = xmlDoc.createElement(tag);
    el.setAttribute('name', name);
    const node = nodeById.get(nodeId);
    if (node) {
      el.setAttribute('bt_x', Math.round(node.position.x).toString());
      el.setAttribute('bt_y', Math.round(node.position.y).toString());
    }
    // UI-only metadata: persist collapsed Control nodes so reload picks the
    // same state. Default false → omit the attribute to keep XML compact.
    if (collapsed === true) {
      el.setAttribute('bt_collapsed', 'true');
    }
    Object.entries(params).forEach(([k, v]) => el.setAttribute(k, v));
    (children.get(nodeId) ?? []).forEach((childId) => {
      const childEl = buildEl(childId);
      if (childEl) el.appendChild(childEl);
    });
    return el;
  }

  const mainBT = xmlDoc.createElement('BehaviorTree');
  mainBT.setAttribute('ID', 'MainTree');
  if (mainRoot) {
    const mainEl = buildEl(mainRoot.id);
    if (mainEl) mainBT.appendChild(mainEl);
  }
  rootEl.appendChild(mainBT);

  if (pendingNodes.length > 0) {
    const pendingBT = xmlDoc.createElement('BehaviorTree');
    pendingBT.setAttribute('ID', '__pending__');
    pendingNodes.forEach((n) => {
      const el = buildEl(n.id);
      if (el) pendingBT.appendChild(el);
    });
    rootEl.appendChild(pendingBT);
  }

  return serializeBTXml(xmlDoc);
}

/**
 * Pretty-print a BehaviorTree.CPP XML document.
 *
 * Drops whitespace-only text nodes, emits one element per line with 2-space
 * indent, and wraps attributes onto continuation lines (parent indent + 2)
 * when the single-line form would exceed 100 chars.
 */
export function serializeBTXml(xmlDoc) {
  if (!xmlDoc || !xmlDoc.documentElement) return '';

  const lines = [];
  // Preserve the XML declaration if the original had one. DOMParser does not
  // expose it via xmlVersion in all browsers, so emit a standard one whenever
  // the input contains an XML doc — matches the on-disk format of existing trees.
  lines.push('<?xml version="1.0" encoding="UTF-8"?>');

  // Top-level comments / PIs that sit before the root are rare here; skip them.
  serializeElement(xmlDoc.documentElement, 0, lines);

  return lines.join('\n') + '\n';
}
