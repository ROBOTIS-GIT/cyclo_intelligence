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

import { render, screen } from '@testing-library/react';
import CollectDecisionPreview from './CollectDecisionPreview';

const result = {
  image_base64:'abc', analysis_id:'pinned-1', captured_at:1,
  decision:{decision:'start',mission_id:'m2',reason:'A red stick is visible.'},
  selected_mission:{mission_name:'Mission2'}, map_name:'warehouse',
  model:'Qwen/Qwen3.5-4B', adapter:'/model/red-stick-lora', localized:false,
};

test('shows only decision, mission and reason for a successful prediction', () => {
  render(<CollectDecisionPreview result={result} phase="Observing" />);
  expect(screen.getByText('start')).toBeInTheDocument();
  expect(screen.getByText('Mission2')).toBeInTheDocument();
  expect(screen.getByText('A red stick is visible.')).toBeInTheDocument();
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
  expect(screen.queryByRole('button')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Mission decision preview')).toHaveTextContent(/^DecisionstartMissionMission2A red stick is visible\.$/);
});

test('shows an error instead of presenting an old decision as a successful result', () => {
  render(<CollectDecisionPreview result={result} error="Camera unavailable" phase="Waiting" />);
  expect(screen.getByRole('alert')).toHaveTextContent('Camera unavailable');
  expect(screen.queryByText('Mission2')).not.toBeInTheDocument();
});

test('shows an active mission for supervision without dispatching an action',()=>{
 render(<CollectDecisionPreview result={{reasoning_task:'mission_supervision',
  decision:{decision:'continue',mission_id:'clean',reason:'The path is clear.'},
  context:{active_mission:{mission_id:'clean',mission_name:'CleanTable'}}}} />);
 expect(screen.getByText('Supervision')).toBeInTheDocument();
 expect(screen.getByText('CleanTable')).toBeInTheDocument();
 expect(screen.queryByRole('button')).not.toBeInTheDocument();
});
