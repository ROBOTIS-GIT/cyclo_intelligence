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

import { fireEvent, render, screen } from '@testing-library/react';
import OperateStageHud from './OperateStageHud';


test('Operate Stop uses the same ownership-aware teardown as stage changes', () => {
  const stop = jest.fn();
  const stopShared = jest.fn();
  render(<OperateStageHud observation={{ needsStop: true, stop }} onStopNavigation={stopShared} />);
  fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
  expect(stopShared).toHaveBeenCalledTimes(1);
  expect(stop).not.toHaveBeenCalled();
});


test('Operate starts reasoning without requiring a selected mission snapshot', () => {
  const start = jest.fn();
  render(<OperateStageHud observation={{start}} missionMapLoaded runPoseInitialized running runMapSnapshotInvalid />);
  fireEvent.click(screen.getByRole('button', {name: 'Run autonomy'}));
  expect(start).toHaveBeenCalledTimes(1);
});

test('Operate cannot start while localization is missing or a stop is pending', () => {
  const props = { observation: {start: jest.fn()}, missionMapLoaded: true, running: true };
  const view = render(<OperateStageHud {...props}/>);
  expect(screen.getByRole('button', {name: 'Run autonomy'})).toBeDisabled();
  view.rerender(<OperateStageHud {...props} runPoseInitialized observation={{phase: 'stopping'}}/>);
  expect(screen.getByRole('button', {name: 'Run autonomy'})).toBeDisabled();
});
