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
import CollectEpisodeLibrary from './CollectEpisodeLibrary';

const episodes = [
  {id:'a',ended_at:1,map_name:'warehouse',storage_path:'warehouse/raw_episode_0001',scenario:'Red stick',frame_count:5,unlabeled_count:3},
  {id:'b',ended_at:1,map_name:'office',storage_path:'office/raw_episode_0001',scenario:'Empty desk',frame_count:2,unlabeled_count:0},
  {id:'open',map_name:'warehouse',scenario:'Still recording'},
];
test('shows saved episodes, filters by map and scene, and selects the exact episode', () => {
  const draft = {episodes,load:jest.fn(),refreshEpisodes:jest.fn(),episode:{id:'b'}};
  render(<CollectEpisodeLibrary draft={draft}/>);
  expect(screen.getAllByRole('listitem')).toHaveLength(2);
  expect(screen.queryByText('Still recording')).not.toBeInTheDocument();
  expect(screen.getByRole('button',{name:/Empty desk/})).toHaveAttribute('aria-pressed','true');
  fireEvent.change(screen.getByLabelText('Filter episodes by map'),{target:{value:'warehouse'}});
  expect(screen.getAllByRole('listitem')).toHaveLength(1);
  fireEvent.click(screen.getByRole('button',{name:/Red stick/}));
  expect(draft.load).toHaveBeenCalledWith('a');
  fireEvent.change(screen.getByLabelText('Search episodes'),{target:{value:'desk'}});
  expect(screen.getByText('No observation episodes match this filter.')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Filter episodes by map'),{target:{value:''}});
  expect(screen.getByText('Labeled')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Refresh episodes'}));
  expect(draft.refreshEpisodes).toHaveBeenCalledTimes(1);
});
test('shows the episode category and searches within it', () => {
 const draft={episodes:[{...episodes[0],episode_type:'observation'}, {...episodes[1],episode_type:'mission'}],load:jest.fn()};
 render(<CollectEpisodeLibrary draft={draft}/>);
 expect(screen.getByText('warehouse / Observation')).toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Mission 1'}));
 expect(screen.getByText('office / Mission')).toBeInTheDocument();
 expect(draft.load).toHaveBeenCalledWith('b');
 fireEvent.change(screen.getByLabelText('Search episodes'),{target:{value:'Mission'}});
 expect(screen.getAllByRole('listitem')).toHaveLength(1);
});
test('blocks episode changes during annotation saves or segmentation', () => {
  const draft = {episodes,load:jest.fn()};
  render(<CollectEpisodeLibrary draft={draft} disabled/>);
  const item = screen.getByRole('button',{name:/Red stick/});
  expect(item).toBeDisabled();
  fireEvent.click(item);
  expect(draft.load).not.toHaveBeenCalled();
});

test('fully rejected frames count as reviewed but not labeled for training',()=>{
 const draft={episodes:[{...episodes[0],mask_reviewed_count:3,mask_labeled_count:1,mask_pending_count:2}]};
 const view=render(<CollectEpisodeLibrary draft={draft} masks/>);
 expect(screen.getByText('1 labeled')).toBeInTheDocument();
 view.rerender(<CollectEpisodeLibrary draft={draft} masks reviewMode/>);
 expect(screen.getByText('3 reviewed · 2 pending')).toBeInTheDocument();
});

test('displays and searches current recording mission names without duplicate automatic titles',()=>{
 const draft={episodes:[{...episodes[0],recording_mission_name:'CleanTable',scenario:'warehouse / CleanTable'}]};
 render(<CollectEpisodeLibrary draft={draft}/>);
 expect(screen.getByText('warehouse / Mission / CleanTable')).toBeInTheDocument();
 expect(screen.queryByText('warehouse / CleanTable')).not.toBeInTheDocument();
 fireEvent.change(screen.getByLabelText('Search episodes'),{target:{value:'CleanTable'}});
 expect(screen.getAllByRole('listitem')).toHaveLength(1);
 fireEvent.change(screen.getByLabelText('Search episodes'),{target:{value:'Mission1'}});
 expect(screen.queryByRole('listitem')).not.toBeInTheDocument();
});


test('deletes only a selected saved episode and locks deletion during edits', () => {
  const onDelete = jest.fn();
  const draft = {episodes, episode: episodes[0]};
  const view = render(<CollectEpisodeLibrary draft={draft} onDelete={onDelete}/>);
  fireEvent.click(screen.getByRole('button', {name:'Delete selected episode'}));
  expect(onDelete).toHaveBeenCalledTimes(1);
  view.rerender(<CollectEpisodeLibrary draft={draft} onDelete={onDelete} disabled/>);
  expect(screen.getByRole('button', {name:'Delete selected episode'})).toBeDisabled();
  view.rerender(<CollectEpisodeLibrary draft={{episodes}} onDelete={onDelete}/>);
  expect(screen.getByRole('button', {name:'Delete selected episode'})).toBeDisabled();
});


test('repeated clicks on the selected episode leave its editor open', () => {
  const draft={episodes,episode:episodes[0],load:jest.fn()};
  const view=render(<CollectEpisodeLibrary draft={draft}/>);
  const selected=screen.getByRole('button',{name:/Red stick/});
  fireEvent.click(selected);fireEvent.click(selected);
  expect(draft.load).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:/Empty desk/}));
  expect(draft.load).toHaveBeenCalledWith('b');
  view.rerender(<CollectEpisodeLibrary draft={{...draft,episode:episodes[1]}}/>);
  fireEvent.click(screen.getByRole('button',{name:/Empty desk/}));
  expect(draft.load).toHaveBeenCalledTimes(1);
});

test('groups observations by map and point without filtering patrol episodes', () => {
 const entries=[
  {...episodes[0],observation_spot_id:'tv',observation_spot:{label:'TVObservation'}},
  {...episodes[0],id:'kitchen',observation_spot_id:'kitchen',observation_spot:{label:'KitchenObservation'}},
  {...episodes[1],observation_spot_id:'tv',observation_spot:{label:'TVObservation'}},
  {...episodes[0],id:'patrol',episode_type:'patrol'},
 ];
 const draft={episodes:entries,load:jest.fn()};
 render(<CollectEpisodeLibrary draft={draft}/>);
 const filter=screen.getByLabelText('Filter episodes by observation point');
 fireEvent.change(filter,{target:{value:JSON.stringify(['warehouse','tv'])}});
 expect(screen.getAllByRole('listitem')).toHaveLength(1);
 expect(screen.getByText('warehouse / Observation / TVObservation')).toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Patrol 1'}));
 expect(screen.queryByLabelText('Filter episodes by observation point')).not.toBeInTheDocument();
 expect(screen.getAllByRole('listitem')).toHaveLength(1);
 fireEvent.click(screen.getByRole('button',{name:'Observation 3'}));
 fireEvent.change(screen.getByLabelText('Filter episodes by map'),{target:{value:'office'}});
 expect(screen.getByLabelText('Filter episodes by observation point')).toHaveValue('');
 expect(screen.getByText('office / Observation / TVObservation')).toBeInTheDocument();
});
