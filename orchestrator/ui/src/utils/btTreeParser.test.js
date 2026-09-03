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

import { applyDagreLayout, findDeletionLayoutAnchor } from './btTreeParser';

describe('applyDagreLayout', () => {
  const nodes = [
    {
      id: 'sequence',
      type: 'btControl',
      position: { x: 900, y: 700 },
      data: { label: 'Sequence_1', nodeType: 'Sequence' },
    },
    {
      id: 'command',
      type: 'btAction',
      position: { x: 1100, y: 720 },
      data: { label: 'SendCommand_1', nodeType: 'SendCommand' },
    },
  ];
  const edges = [
    { id: 'e_sequence_command', source: 'sequence', target: 'command' },
  ];

  it('keeps the requested anchor at the same canvas position', () => {
    const result = applyDagreLayout(nodes, edges, {
      respectStored: false,
      anchorNodeId: 'sequence',
    });

    const sequence = result.nodes.find((node) => node.id === 'sequence');
    const command = result.nodes.find((node) => node.id === 'command');

    expect(sequence.position).toEqual(nodes[0].position);
    expect(command.position.x).toBe(sequence.position.x);
    expect(command.position.y).toBeGreaterThan(sequence.position.y);
  });

  it('uses dagre coordinates when no anchor is requested', () => {
    const result = applyDagreLayout(nodes, edges, { respectStored: false });
    const sequence = result.nodes.find((node) => node.id === 'sequence');

    expect(sequence.position).not.toEqual(nodes[0].position);
  });
});

describe('findDeletionLayoutAnchor', () => {
  const nodes = [
    { id: 'sequence', position: { x: 900, y: 700 } },
    { id: 'command', position: { x: 900, y: 840 } },
    { id: 'new_node', position: { x: 1300, y: 900 } },
  ];
  const edges = [
    { id: 'e_sequence_command', source: 'sequence', target: 'command' },
  ];

  it('anchors a surviving neighbor when deleting a connected node', () => {
    expect(findDeletionLayoutAnchor(
      nodes,
      edges,
      new Set(['command']),
      new Set(),
    )).toBe('sequence');
  });

  it('anchors the existing graph when deleting a disconnected node', () => {
    expect(findDeletionLayoutAnchor(
      nodes,
      edges,
      new Set(['new_node']),
      new Set(),
    )).toBe('sequence');
  });
});
