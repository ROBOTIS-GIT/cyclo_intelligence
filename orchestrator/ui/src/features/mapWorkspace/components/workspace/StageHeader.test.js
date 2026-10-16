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

import { render, screen, fireEvent } from '@testing-library/react';
import StageHeader from './StageHeader';
import { STAGE_COLLECT, STAGE_RUN, STAGE_OPERATE } from '../../lib/stages';

test('Record permits changing a localized map only after execution and edits finish', () => {
  const open = jest.fn();
  const props = {workspaceStage:STAGE_COLLECT,runFamilyStage:true,running:true,onOpenRunMap:open};
  const view = render(<StageHeader {...props}/>);
  fireEvent.click(screen.getByRole('button',{name:'Load Map'}));
  expect(open).toHaveBeenCalledTimes(1);
  view.rerender(<StageHeader {...props} recordMapChangeBlocked/>);
  expect(screen.getByRole('button',{name:'Load Map'})).toBeDisabled();
  view.rerender(<StageHeader {...props} runShutdownPending/>);
  expect(screen.getByRole('button',{name:'Load Map'})).toBeDisabled();
});

test.each([STAGE_RUN, STAGE_OPERATE])('other stages retain the running map lock: %s', stage => {
  render(<StageHeader workspaceStage={stage} runFamilyStage running/>);
  expect(screen.getByRole('button',{name:'Load Map'})).toBeDisabled();
});
