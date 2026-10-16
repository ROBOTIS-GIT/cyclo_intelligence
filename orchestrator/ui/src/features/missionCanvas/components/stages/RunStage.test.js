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
import { RunStageHud } from './RunStage';

jest.mock('../../../mapWorkspace/components/mapChrome', () => ({ TopicStatusPanel: () => null }));
jest.mock('../run/RunSessionPanel', () => () => null);

test('Run requires the loaded mission snapshot and invokes the mission runner', () => {
  const onRunMission = jest.fn();
  const props = {missionMapLoaded: true, runPoseInitialized: true, onRunMission};
  const view = render(<RunStageHud {...props} runMapSnapshotInvalid />);
  expect(screen.getByRole('button', {name: 'Run Mission'})).toBeDisabled();
  view.rerender(<RunStageHud {...props} />);
  fireEvent.click(screen.getByRole('button', {name: 'Run Mission'}));
  expect(onRunMission).toHaveBeenCalledTimes(1);
});
