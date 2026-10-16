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

import { initialWorkspaceStage, initialWorkspaceKind } from './session';
import {
  STAGE_LEARN, WORKSPACE_NAV_GROUPS, WORKSPACE_STAGES,
  normalizeWorkspaceStage, workspaceHasStage, workspaceNavGroups,
} from './stages';

describe('Autonomy workflow stages', () => {
  test('keeps Record, Train, and Operate in workflow order', () => {
    const group = WORKSPACE_NAV_GROUPS.find(g => g.caption === 'AUTONOMY');
    expect(group.stageIds).toEqual(['collect', STAGE_LEARN, 'operate']);
    expect(group.stageIds.map(id => WORKSPACE_STAGES.find(stage => stage.id === id).label))
      .toEqual(['Record', 'Train', 'Operate']);
    expect(workspaceNavGroups('autonomy')).toEqual([group]);
  });

  test.each(['annotate', 'train', 'reasoning'])('restores former %s sessions into Train', stage => {
    expect(initialWorkspaceStage({ workspaceStage: stage })).toBe(STAGE_LEARN);
    expect(initialWorkspaceStage({ workspaceStage: stage }, 'autonomy')).toBe(STAGE_LEARN);
  });

  test.each([
    { workspaceStage: 'collect', recordMode: 'dataset' },
    { workspaceStage: 'collect', recordMode: 'dataset', autonomyLayoutVersion: 1 },
    { workspaceStage: 'operate', workspaceStages: { autonomy: 'collect' }, recordMode: 'dataset' },
  ])('migrates legacy Record dataset selection: %o', session => {
    expect(initialWorkspaceStage(session, 'autonomy')).toBe(STAGE_LEARN);
  });

  test('also migrates legacy dataset sessions without an explicit workspace kind', () => {
    expect(initialWorkspaceStage({ workspaceStage: 'collect', recordMode: 'dataset' })).toBe(STAGE_LEARN);
  });

  test.each([
    { workspaceStage: 'collect', recordMode: 'dataset' },
    { workspaceStage: 'operate', workspaceStages: { autonomy: 'collect' }, recordMode: 'dataset' },
  ])('does not let stale recordMode override Record after migration: %o', session => {
    expect(initialWorkspaceStage({ ...session, autonomyLayoutVersion: 2 }, 'autonomy')).toBe('collect');
  });

  test('keeps a new Record session selected without a workspace-specific stage', () => {
    expect(initialWorkspaceStage({
      workspaceStage: 'collect', recordMode: 'dataset', autonomyLayoutVersion: 2,
    })).toBe('collect');
  });

  test.each(['annotate', 'train', 'reasoning'])('restores a remembered %s stage into Train', stage => {
    expect(initialWorkspaceStage({ workspaceStages: { autonomy: stage } }, 'autonomy'))
      .toBe(STAGE_LEARN);
  });

  test.each(['collect', STAGE_LEARN, 'operate'])('remembers %s independently of Mission Canvas', stage => {
    const session = {
      autonomyLayoutVersion: 2,
      workspaceStage: 'operate',
      workspaceStages: { mission: 'authoring', autonomy: stage },
    };
    expect(initialWorkspaceStage(session, 'mission')).toBe('authoring');
    expect(initialWorkspaceStage(session, 'autonomy')).toBe(stage);
    expect(workspaceHasStage('autonomy', stage)).toBe(true);
    expect(workspaceHasStage('mission', stage)).toBe(false);
  });

  test.each([STAGE_LEARN, 'annotate', 'train', 'reasoning'])('normalizes %s links and infers Autonomy Canvas', stage => {
    expect(normalizeWorkspaceStage(stage)).toBe(STAGE_LEARN);
    expect(workspaceHasStage('autonomy', stage)).toBe(true);
    expect(workspaceHasStage('mission', stage)).toBe(false);
    expect(initialWorkspaceKind({ workspaceStage: stage })).toBe('autonomy');
  });

  test('rejects stages from the other canvas and unknown stages', () => {
    expect(initialWorkspaceStage({ workspaceStage: 'operate' }, 'mission')).toBe('mapping');
    expect(initialWorkspaceStage({ workspaceStage: 'run' }, 'autonomy')).toBe('collect');
    expect(initialWorkspaceStage({ workspaceStages: { autonomy: 'authoring' } }, 'autonomy')).toBe('collect');
    expect(initialWorkspaceStage({ workspaceStages: { mission: 'train' } }, 'mission')).toBe('mapping');
    expect(initialWorkspaceStage({ workspaceStages: { autonomy: 'missing' } }, 'autonomy')).toBe('collect');
  });

  test('migrates former autonomy sessions without changing Action Canvas selection', () => {
    expect(initialWorkspaceKind({ workspaceKind: 'mission', workspaceStage: 'collect' })).toBe('autonomy');
    expect(initialWorkspaceKind({ workspaceKind: 'autonomy', workspaceStage: 'operate' })).toBe('autonomy');
    expect(initialWorkspaceKind({ workspaceKind: 'action_canvas', workspaceStage: 'train' })).toBe('action_canvas');
  });
});
