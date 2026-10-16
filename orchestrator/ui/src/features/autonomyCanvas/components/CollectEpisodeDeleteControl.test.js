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

import { act, fireEvent, render, screen } from '@testing-library/react';
import CollectEpisodeDeleteControl from './CollectEpisodeDeleteControl';

const episodes = [
 {id:'one',map_name:'house',episode_type:'observation',storage_path:'house/raw_episode_0001',observation_spot:{id:'table',label:'Table'},frame_count:12,ended_at:1},
 {id:'two',map_name:'house',episode_type:'mission',storage_path:'house/raw_episode_0002',recording_mission_name:'CleanTable',ended_at:1},
 {id:'three',map_name:'house',episode_type:'patrol',storage_path:'house/raw_episode_0003',capture_segment:'mission_return',return_mission_name:'CleanTable',return_observation_spot:{id:'table',label:'Table'},ended_at:1},
 {id:'other',map_name:'other',scenario:'Other map'},
];
function setup(remove=jest.fn(async()=>true)) {
 const draft={episodes,deleteEpisode:remove};
 const view=render(<CollectEpisodeDeleteControl draft={draft} mapName="house"/>);
 fireEvent.click(screen.getByRole('button',{name:'Delete episode'}));
 return {draft,view};
}
afterEach(()=>jest.restoreAllMocks());
test('filters current-map episodes and shows targets and return context',()=>{
 setup();
 expect(screen.queryByText('Other map')).not.toBeInTheDocument();
 expect(screen.getByText('Observation · Table')).toBeInTheDocument();
 expect(screen.getByText('Mission · CleanTable')).toBeInTheDocument();
 expect(screen.getByText('Patrol · Return · CleanTable')).toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Mission',exact:true}));
 expect(screen.queryByText('raw_episode_0001')).not.toBeInTheDocument();
 expect(screen.getByText('raw_episode_0002')).toBeInTheDocument();
});
test('bulk delete confirms once and deletes only checked episodes',async()=>{
 const confirm=jest.spyOn(window,'confirm').mockReturnValue(false);
 const {draft}=setup();
 fireEvent.click(screen.getByLabelText('Select raw_episode_0001'));
 fireEvent.click(screen.getByLabelText('Select raw_episode_0002'));
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete (2)'})));
 expect(draft.deleteEpisode).not.toHaveBeenCalled();
 confirm.mockReturnValue(true);
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete (2)'})));
 expect(draft.deleteEpisode.mock.calls).toEqual([['one'],['two']]);
 expect(screen.queryByText('raw_episode_0001')).not.toBeInTheDocument();
 expect(screen.getByText('raw_episode_0003')).toBeInTheDocument();
});
test('switching filters clears selection and select all covers only visible rows',async()=>{
 jest.spyOn(window,'confirm').mockReturnValue(true);
 const {draft}=setup();
 fireEvent.click(screen.getByLabelText('Select raw_episode_0001'));
 fireEvent.click(screen.getByRole('button',{name:'Patrol',exact:true}));
 expect(screen.getByRole('button',{name:'Delete (0)'})).toBeDisabled();
 fireEvent.click(screen.getByLabelText('Select all visible episodes'));
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete (1)'})));
 expect(draft.deleteEpisode.mock.calls).toEqual([['three']]);
});
test('partial failure leaves failed selections available for retry',async()=>{
 jest.spyOn(window,'confirm').mockReturnValue(true);
 setup(jest.fn(async id=>id!=='two'));
 fireEvent.click(screen.getByLabelText('Select all visible episodes'));
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete (3)'})));
 expect(screen.getByRole('alert')).toHaveTextContent('raw_episode_0002: Delete failed');
 expect(screen.getByLabelText('Select raw_episode_0002')).toBeChecked();
 expect(screen.getByRole('button',{name:'Delete (1)'})).toBeEnabled();
});

test('delete stays locked between continuous frame writes and unlocks after capture stops', () => {
  const draft = { episodes: [], recording: true, busy: false };
  const view = render(<CollectEpisodeDeleteControl draft={draft} mapName="warehouse"/>);
  for (const busy of [true, false, true, false]) {
    view.rerender(<CollectEpisodeDeleteControl draft={{...draft, busy}} mapName="warehouse"/>);
    expect(screen.getByRole('button', {name:'Delete episode'})).toBeDisabled();
  }
  view.rerender(<CollectEpisodeDeleteControl draft={{...draft, recording:false}} mapName="warehouse"/>);
  expect(screen.getByRole('button', {name:'Delete episode'})).toBeEnabled();
});
