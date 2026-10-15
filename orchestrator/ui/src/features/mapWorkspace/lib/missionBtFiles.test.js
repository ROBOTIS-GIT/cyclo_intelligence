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

import { assembleMissionBtFilesForSave } from './missionBtFiles';

describe('assembleMissionBtFilesForSave', () => {
  const EDITED = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait duration="2.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');

  test('moves a flat waypoint BT into its stable waypoint directory', () => {
    const spot = {
      id: 'spot_a',
      label: 'Bay',
      linked_bt_tree: 'locals/dock.xml',
      metadata: { local_bt: 'locals/dock.xml' },
    };
    const { files, stalePaths } = assembleMissionBtFilesForSave(
      [spot],
      { 'locals/dock.xml': EDITED },
      [],
      'global.xml',
      '<global/>',
    );
    expect(files['locals/spot_a/main.xml']).toBe(EDITED);
    expect(files['locals/dock.xml']).toBeUndefined();
    expect(files['global.xml']).toBe('<global/>');
    expect(stalePaths).toEqual(['locals/dock.xml']);
  });

  test('preserves names and edited content already stored in the waypoint directory', () => {
    const spot = {
      id: 'spot_a',
      label: 'Dock',
      metadata: { local_bt: 'locals/spot_a/run.xml' },
    };
    const { files, stalePaths } = assembleMissionBtFilesForSave(
      [spot],
      { 'locals/spot_a/run.xml': EDITED },
      [],
      'global.xml',
      '<global/>',
    );
    expect(files['locals/spot_a/run.xml']).toBe(EDITED);
    expect(stalePaths).toEqual([]);
  });

  test('persists every owned XML while local_bt only selects the default', () => {
    const alternate = EDITED.replace('2.0', '4.0');
    const spot = {
      id: 'spot_a',
      label: 'Dock',
      linked_bt_tree: 'locals/spot_a/alternate.xml',
      local_bt_files: ['locals/spot_a/alternate.xml', 'locals/spot_a/default.xml'],
      metadata: {
        local_bt: 'locals/spot_a/alternate.xml',
        local_bt_files: ['locals/spot_a/alternate.xml', 'locals/spot_a/default.xml'],
      },
    };
    const { files, stalePaths } = assembleMissionBtFilesForSave(
      [spot],
      {
        'locals/spot_a/default.xml': EDITED,
        'locals/spot_a/alternate.xml': alternate,
      },
      [],
      'global.xml',
      '<global/>',
    );

    expect(files['locals/spot_a/default.xml']).toBe(EDITED);
    expect(files['locals/spot_a/alternate.xml']).toBe(alternate);
    expect(stalePaths).toEqual([]);
  });

  test('collects a mixed flat and nested library into one waypoint directory', () => {
    const alternate = EDITED.replace('2.0', '4.0');
    const spot = {
      id: 'Waypoint_1_ee992021',
      label: 'Waypoint 1',
      linked_bt_tree: 'locals/waypoint_1.xml',
      local_bt_files: [
        'locals/waypoint_1.xml',
        'locals/waypoint_1_ee992021/test.xml',
      ],
      metadata: { local_bt: 'locals/waypoint_1.xml' },
    };
    const { files, stalePaths } = assembleMissionBtFilesForSave(
      [spot],
      {
        'locals/waypoint_1.xml': EDITED,
        'locals/waypoint_1_ee992021/test.xml': alternate,
      },
      [],
      'global.xml',
      '<global/>',
    );

    expect(files['locals/waypoint_1/main.xml']).toBe(EDITED);
    expect(files['locals/waypoint_1/test.xml']).toBe(alternate);
    expect(stalePaths).toEqual([
      'locals/waypoint_1.xml',
      'locals/waypoint_1_ee992021/test.xml',
    ]);
  });

  test('removes the waypoint token without renaming an existing XML file', () => {
    const spot = {
      id: 'Waypoint_1_ee992021',
      label: 'Waypoint 1',
      linked_bt_tree: 'locals/waypoint_1_ee992021/test.xml',
      local_bt_files: ['locals/waypoint_1_ee992021/test.xml'],
      metadata: { local_bt: 'locals/waypoint_1_ee992021/test.xml' },
    };
    const { files, stalePaths } = assembleMissionBtFilesForSave(
      [spot],
      { 'locals/waypoint_1_ee992021/test.xml': EDITED },
      [],
      'global.xml',
      '<global/>',
    );

    expect(files['locals/waypoint_1/test.xml']).toBe(EDITED);
    expect(stalePaths).toEqual(['locals/waypoint_1_ee992021/test.xml']);
  });

  test('migrates a legacy root-level BT file into the waypoint main XML', () => {
    const spot = {
      id: 'spot_legacy',
      label: 'Prep table',
      linked_bt_tree: 'prep_table.xml',
      metadata: { local_bt: 'prep_table.xml' },
    };
    const { files, stalePaths } = assembleMissionBtFilesForSave(
      [spot],
      { 'prep_table.xml': EDITED },
      [],
      'global.xml',
      '<global/>',
    );
    const localEntries = Object.entries(files).filter(([path]) => path.startsWith('locals/'));

    expect(localEntries).toHaveLength(1);
    expect(localEntries[0][0]).toBe('locals/spot_legacy/main.xml');
    expect(localEntries[0][1]).toBe(EDITED);
    expect(files['prep_table.xml']).toBeUndefined();
    expect(stalePaths).toEqual(['prep_table.xml']);
  });

  test('writes an empty default for an unedited waypoint', () => {
    const spot = { id: 'spot_a', label: 'Dock', metadata: {} };
    const { files } = assembleMissionBtFilesForSave([spot], {}, [], 'global.xml', '<global/>');
    expect(files['locals/spot_a/main.xml']).toContain('<BehaviorTree ID="MainTree"/>');
  });

  test('keeps distinct BT files when labels normalize to the same slug', () => {
    const spots = [
      {
        id: 'spot_a',
        label: '픽업',
        metadata: { local_bt: 'locals/pickup_a.xml' },
      },
      {
        id: 'spot_b',
        label: '도착',
        metadata: { local_bt: 'locals/dropoff_b.xml' },
      },
    ];
    const second = EDITED.replace('2.0', '3.0');
    const { files } = assembleMissionBtFilesForSave(
      spots,
      {
        'locals/pickup_a.xml': EDITED,
        'locals/dropoff_b.xml': second,
      },
      [],
      'global.xml',
      '<global/>',
    );

    expect(files['locals/spot_a/main.xml']).toBe(EDITED);
    expect(files['locals/spot_b/main.xml']).toBe(second);
  });

  test('uses readable numeric suffixes when token-free waypoint folders collide', () => {
    const startOnly = EDITED.replace('2.0', '7.0').replace('<Wait', '<Wait name="StartOnly"');
    const spots = [
      {
        id: 'Waypoint_1_ee992021',
        label: 'Start',
        linked_bt_tree: 'locals/waypoint_1/main.xml',
        local_bt_files: ['locals/waypoint_1/main.xml'],
        metadata: {
          local_bt: 'locals/waypoint_1/main.xml',
          local_bt_files: ['locals/waypoint_1/main.xml'],
        },
      },
      { id: 'Waypoint_1_d58de4b1', label: 'Waypoint 1', metadata: {} },
    ];

    const { files } = assembleMissionBtFilesForSave(
      spots,
      { 'locals/waypoint_1/main.xml': startOnly },
      [],
      'global.xml',
      '<global/>',
    );

    expect(files['locals/waypoint_1/main.xml']).toBe(startOnly);
    expect(files['locals/waypoint_1_2/main.xml']).toContain('<BehaviorTree ID="MainTree"/>');
    expect(files['locals/waypoint_1_2/main.xml']).not.toContain('StartOnly');
  });
});
