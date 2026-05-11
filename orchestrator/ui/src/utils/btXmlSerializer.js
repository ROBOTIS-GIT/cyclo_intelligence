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
