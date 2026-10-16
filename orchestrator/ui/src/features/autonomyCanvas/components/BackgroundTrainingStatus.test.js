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

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import BackgroundTrainingStatus from "./BackgroundTrainingStatus";
import StageRail from '../../mapWorkspace/components/workspace/StageRail';
import { collectionRequest } from '../../../utils/collectionApi';

jest.mock('../../../utils/collectionApi', () => ({ collectionRequest: jest.fn() }));

test('Qwen training remains visible while Operate stays selectable', async () => {
  collectionRequest.mockImplementation(async path => path === 'qwen/workflow'
    ? { reasoning_training: { id:'job1', status:'training', model_name:'Home 2B', step:210, steps:1000 } }
    : { training: { status:'idle' } });
  const onSelectStage = jest.fn();
  render(<StageRail workspaceKind="autonomy" workspaceStage="learn" onSelectStage={onSelectStage} footer={<BackgroundTrainingStatus />} />);
  const status = await screen.findByRole('status', { name:'Background training' });
  expect(status).toHaveTextContent('Qwen · Home 2B');
  expect(status).toHaveTextContent('210/1000');
  expect(screen.getByRole('progressbar', { name:'Qwen training progress' })).toHaveAttribute('value', '210');
  const operate = screen.getByRole('tab', { name:'Operate' });
  expect(operate).toBeEnabled();
  fireEvent.click(operate);
  expect(onSelectStage).toHaveBeenCalled();
  await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith('segmentation/workflow'));
});
