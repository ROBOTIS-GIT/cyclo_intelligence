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

import { useState } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import CollectSidebar from './CollectSidebar';
import CollectStageHud from './CollectStageHud';
import { collectionRequest } from '../../../utils/collectionApi';
const mockSaveMask = jest.fn();
let mockOnMaskSaved;
let mockReviewMode;
jest.mock('./LiveHeadCamera', () => () => null);
jest.mock('../../mapWorkspace/components/MappingTeleopPanel', () => () => null);
// Workflow polling is covered by its own suites; keep save/review tests focused on editor operations.
jest.mock('../../../utils/collectionApi', () => ({collectionRequest:jest.fn(() => new Promise(() => {}))}));
jest.mock('./Sam2Annotation', () => {
  const React = require('react');
  return React.forwardRef(({frame, onFrameLoaded, onMaskSaved, renderLayout, reviewMode}, ref) => {
    mockOnMaskSaved = onMaskSaved;
    mockReviewMode = reviewMode;
    React.useImperativeHandle(ref, () => ({save: mockSaveMask}));
    React.useEffect(() => {onFrameLoaded(frame.frame_id);}, [frame.frame_id, onFrameLoaded]);
    return renderLayout ? renderLayout({imageEditor:<div>Mask image</div>}) : null;
  });
});
function setup(overrides = {}, maskMode = false) {
  const frame={frame_id:'frame1',capture:{received_at_ms:1000}};
  const draft={episodes:[],episode:{id:'ep1',ended_at:1000,observations:[frame],annotations:[],catalog:{missions:[]}},
    selectedFrame:frame,active:false,busy:false,annotate:jest.fn().mockResolvedValue(true),...overrides};
  draft.episode = {scenario:"Shelf", group_id:"scene-a", robot_type:"ffw_sg2", environment:"simulation", ...draft.episode};
  const props = {mapCanvas:<CollectStageHud/>, draft, mapName:"warehouse", mapLoaded:true};
  const view = render(<CollectSidebar {...props} mode="dataset" />);
  draft.changeMode = mode => view.rerender(<CollectSidebar {...props} mode={mode}/>);
  if (maskMode) {
    fireEvent.click(screen.getByRole("button", {name:"Object masks (optional)"}));
  } else {
    fireEvent.load(screen.getByAltText('Saved frame for reasoning annotation'));
    fireEvent.change(screen.getByRole('textbox',{name:'Reason'}),{target:{value:'No stick visible.'}});
  }
  return draft;
}
beforeEach(()=>{collectionRequest.mockImplementation(() => new Promise(() => {}));mockSaveMask.mockReset().mockResolvedValue(true);global.fetch=jest.fn(async()=>({ok:true,json:async()=>({training:{status:'idle'},batch:{status:'idle'},models:[],missions:[]})}));});
afterEach(()=>{delete global.fetch;});
test('reasoning Save approves the decision independently of object masks',async()=>{
  const draft=setup();
  expect(screen.queryByRole('button',{name:'Save label'})).not.toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Save mask'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await screen.findByText('Annotation saved for training.');
  expect(mockSaveMask).not.toHaveBeenCalled();
  expect(draft.annotate).toHaveBeenCalledWith(expect.objectContaining({reason:'No stick visible.',review_status:'approved'}));
});
test('reasoning labels can save without a SAM mask',async()=>{
  mockSaveMask.mockResolvedValue(false);const draft=setup();
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await screen.findByText('Annotation saved for training.');
  expect(mockSaveMask).not.toHaveBeenCalled();
  expect(draft.annotate).toHaveBeenCalledTimes(1);
});
test('label failure does not report success and allows retry',async()=>{
  const draft=setup();draft.annotate.mockResolvedValueOnce(false);
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await screen.findByText(/Decision was not saved/);
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await screen.findByText('Annotation saved for training.');
  expect(draft.annotate).toHaveBeenCalledTimes(2);
});


test('review filters approved masks and navigates all unreviewed frames without a capture button',()=>{
  const first={frame_id:'first',capture:{received_at_ms:1000}};
  const second={frame_id:'second',auto_labeling:{sam:{status:'generated'}},capture:{received_at_ms:2000}};
  const third={frame_id:'third',auto_labeling:{sam:{status:'generated'}},capture:{received_at_ms:3000}};
  const selectFrame=jest.fn();
  setup({selectedFrame:first,selectFrame,episode:{id:'ep1',ended_at:1000,observations:[first,second,third],regions:[{frame_id:'first',review_status:'approved'}],annotations:[],catalog:{missions:[]}}}, true);
  expect(screen.queryByRole('button',{name:'Capture frame'})).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox',{name:'Unreviewed masks only'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Next frame'}));
  expect(selectFrame).toHaveBeenLastCalledWith(second);
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true');
  expect(selectFrame).toHaveBeenLastCalledWith(second);
});

test('Record uses the map mission without a duplicate mission selector', () => {
  const draft={episodes:[],episode:null,active:false,busy:false,selectFrame:jest.fn(),start:jest.fn()};
  render(<CollectSidebar mapCanvas={<CollectStageHud/>} draft={draft} mapName="warehouse" mapLoaded recordMission="Mission2" recordSession={{type:"mission",ready:true,missionId:"m2",missionName:"Mission2",observationSpotId:"a",returnSpotId:"a",planRevision:1}} />);
  expect(screen.queryByLabelText('Recording mission')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Capture mode')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Start episode'}));
  expect(draft.start).toHaveBeenCalledWith(expect.objectContaining({recording_mission_name:'Mission2'}));
});

test('Reasoning saves scene evidence as reason without frame metadata', async()=>{
  const draft=setup();
  fireEvent.change(screen.getByLabelText('Reason'),{target:{value:'Empty plate on table'}});
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await screen.findByText('Annotation saved for training.');
  expect(draft.annotate).toHaveBeenCalledWith(expect.objectContaining({reason:'Empty plate on table'}));
  expect(draft.annotate.mock.calls[0][0]).not.toHaveProperty('metadata');
});

test.each(['done', 'failed', 'cancelled'])('navigation %s leaves independent capture and the episode open', async status => {
  const draft = {episodes:[], episode:null, active:false, busy:false, selectedFrame:null,
    selectFrame:jest.fn(), stopRecording:jest.fn(), finish:jest.fn().mockResolvedValue(true),
    startRecording:jest.fn(() => new Promise(() => {}))};
  const props = {draft, mapName:'warehouse', mapLoaded:true, routeReady:true,
    recordMission:'Mission2',
    onStartRoute:jest.fn().mockReturnValue(true), onStopRoute:jest.fn().mockResolvedValue([])};
  const view=render(<CollectSidebar mapCanvas={<CollectStageHud/>} {...props} routeStatus="done"/>);
  draft.active=true;
  draft.episode={id:'ep',observations:[],annotations:[],catalog:{missions:[]}};
  view.rerender(<CollectSidebar mapCanvas={<CollectStageHud/>} {...props} routeStatus="done"/>);
  fireEvent.click(screen.getByRole('button',{name:'Capture'}));
  expect(draft.finish).not.toHaveBeenCalled(); // Previous run's completed status is ignored.
  view.rerender(<CollectSidebar mapCanvas={<CollectStageHud/>} {...props} routeStatus="navigating" routeRunning/>);
  view.rerender(<CollectSidebar mapCanvas={<CollectStageHud/>} {...props} routeStatus={status}/>);
  expect(draft.stopRecording).not.toHaveBeenCalled();
  expect(draft.finish).not.toHaveBeenCalled();
  expect(draft.startRecording).toHaveBeenCalledWith(1000, expect.any(Function), expect.any(Function));
  expect(props.onStartRoute).not.toHaveBeenCalled();
  expect(props.onStopRoute).not.toHaveBeenCalled();
});

test('remounting Dataset after Live preserves unsaved mission label edits', () => {
  const draft = setup();
  fireEvent.change(screen.getByLabelText('Reason'), {target:{value:'Unsaved observation'}});
  draft.changeMode('live');
  draft.changeMode('dataset');
  expect(screen.getByLabelText('Reason')).toHaveValue('Unsaved observation');
});

test('Reasoning does not require frame information or overwrite historical metadata', async () => {
  const frame = {frame_id:'frame1'};
  const draft=setup({selectedFrame:frame,episode:{id:'ep1',observations:[frame],annotations:[],catalog:{missions:[]}}});
  for(const name of ['Scenario','Scene group','Robot','Environment']) expect(screen.queryByLabelText(name)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await screen.findByText('Annotation saved for training.');
  expect(draft.annotate.mock.calls[0][0]).not.toHaveProperty('metadata');
});

test('mask save waits for episode details before list refresh takes the operation lock',async()=>{
  let finishDetails;
  const refresh=jest.fn(()=>new Promise(resolve=>{finishDetails=resolve;}));
  const refreshEpisodes=jest.fn().mockResolvedValue(true);
  setup({refresh,refreshEpisodes}, true);
  const completion=mockOnMaskSaved();
  expect(refresh).toHaveBeenCalledTimes(1);
  expect(refreshEpisodes).not.toHaveBeenCalled();
  finishDetails();
  await completion;
  expect(refreshEpisodes).toHaveBeenCalledTimes(1);
});

test('successful mask save advances within sampled frames only',async()=>{
 const frames=['first','middle','last'].map(frame_id=>({frame_id,capture:{received_at_ms:1000}}));
 const selectFrame=jest.fn();
 setup({selectedFrame:frames[0],selectFrame,refresh:jest.fn().mockResolvedValue(),refreshEpisodes:jest.fn().mockResolvedValue(),
   episode:{id:'ep1',ended_at:1000,observations:frames,annotations:[],regions:[],sam_sample:{schema_version:'user_count_v1',frame_ids:['first','last']},catalog:{missions:[]}}}, true);
 fireEvent.click(screen.getByRole("button",{name:"Show sampled frames"}));
 await mockOnMaskSaved({advance:true});
 expect(selectFrame).toHaveBeenLastCalledWith(frames[2]);
});

test('last sampled frame stays selected after save',async()=>{
 const frame={frame_id:'last',capture:{received_at_ms:1000}};
 const selectFrame=jest.fn();
 setup({selectedFrame:frame,selectFrame,refresh:jest.fn().mockResolvedValue(),refreshEpisodes:jest.fn().mockResolvedValue(),episode:{id:'ep1',ended_at:1000,observations:[frame],annotations:[],sam_sample:{schema_version:'user_count_v1',frame_ids:['last']},catalog:{missions:[]}}}, true);
 await mockOnMaskSaved({advance:true});
 expect(selectFrame).not.toHaveBeenCalled();
});

test('Review stays disabled before any automatic result exists',()=>{
 setup({},true);
 expect(screen.getByRole('button',{name:'Review'})).toBeDisabled();
});

test('a frame with all masks rejected is reviewed but has no training labels',()=>{
 const first={frame_id:'first'}, second={frame_id:'second',auto_labeling:{sam:{status:'generated'}}};
 const selectFrame=jest.fn();
 setup({selectedFrame:first,selectFrame,episode:{id:'ep1',ended_at:1000,observations:[first,second],annotations:[],
   regions:[{frame_id:'first',review_status:'rejected'},{frame_id:'second',review_status:'draft',annotator:'sam-auto'}],catalog:{missions:[]}}}, true);
 expect(mockReviewMode).toBe(false);
 fireEvent.click(screen.getByRole('button',{name:'Review'}));
 expect(selectFrame).toHaveBeenLastCalledWith(second);
});

test('Dataset tools show one editor and the same workflow controls',()=>{
 setup({},true);
 expect(screen.queryByText('Frame information')).not.toBeInTheDocument();
 expect(screen.queryByRole('textbox',{name:'Reason'})).not.toBeInTheDocument();
 expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Reasoning labels'}));
 expect(screen.getByLabelText('Sample count')).toBeVisible();
 expect(screen.getByRole('button',{name:'Auto-label'})).toBeVisible();
 expect(screen.queryByRole('button',{name:'Manual',exact:true})).not.toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Auto Labeling',exact:true})).not.toBeInTheDocument();
 expect(screen.queryByText('Frame information')).not.toBeInTheDocument();
 expect(screen.getByLabelText('Expected decision')).toBeInTheDocument();
 expect(screen.getByAltText('Saved frame for reasoning annotation')).toBeInTheDocument();
});


test('empty reviewed frames are excluded from the unreviewed filter',()=>{
 const first={frame_id:'first',mask_review:{status:'empty'}},second={frame_id:'second',auto_labeling:{sam:{status:'generated'}}};
 const selectFrame=jest.fn();
 setup({selectedFrame:first,selectFrame,episode:{id:'ep1',ended_at:1000,observations:[first,second],annotations:[],regions:[],catalog:{missions:[]}}},true);
 fireEvent.click(screen.getByRole('button',{name:'Review'}));
 expect(selectFrame).toHaveBeenLastCalledWith(second);
});

test('approving a reasoning draft advances and removes the reviewed frame from the review filter', async()=>{
  const frames=['first','second'].map(frame_id=>({frame_id,auto_labeling:{sam:{status:'generated'},qwen:{status:'generated'}}}));
  const annotate=jest.fn();
  function ReviewWorkspace() {
    const [selectedFrame,selectFrame]=useState(frames[0]);
    const [annotations,setAnnotations]=useState(frames.map(frame=>({frame_ids:[frame.frame_id],review_status:'draft',annotator:'sam-auto',revision:1,
      target:{decision:'wait',reason:`Evidence for ${frame.frame_id}`}})));
    annotate.mockImplementation(async label=>{
      setAnnotations(previous=>previous.map(item=>item.frame_ids[0]===selectedFrame.frame_id
        ? {...item,review_status:label.review_status,revision:item.revision+1,target:{decision:label.decision,reason:label.reason}}:item));
      return true;
    });
    const draft={episodes:[],active:false,busy:false,selectedFrame,selectFrame,annotate,
      episode:{id:'review',ended_at:1,observations:frames,annotations,regions:[],catalog:{missions:[]},
        qwen_sample:{schema_version:'user_count_v1',frame_ids:['first']}}};
    return <CollectSidebar mapCanvas={<CollectStageHud/>} draft={draft} mode="dataset" mapName="warehouse" mapLoaded/>;
  }
  render(<ReviewWorkspace/>);
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument();
  fireEvent.load(screen.getByAltText('Saved frame for reasoning annotation'));
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await waitFor(()=>expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument());
  expect(annotate).toHaveBeenCalledWith({decision:'wait',mission_id:null,reason:'Evidence for first',annotator:'collect-ui',review_status:'approved'});
  await waitFor(()=>expect(screen.getByLabelText('Reason')).toHaveValue('Evidence for second'));
  fireEvent.load(screen.getByAltText('Saved frame for reasoning annotation'));
  fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
  await waitFor(()=>expect(screen.getByText('All generated labels reviewed.')).toBeVisible());
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true');
  expect(await screen.findByText('All generated labels reviewed.')).toBeVisible();
});

test('saving a mask review advances outside the sample and excludes the approved frame', async()=>{
  const frames=['first','second'].map(frame_id=>({frame_id,auto_labeling:{sam:{status:'generated'},qwen:{status:'generated'}}}));
  const refresh=jest.fn(), refreshEpisodes=jest.fn().mockResolvedValue();
  function MaskReviewWorkspace() {
    const [selectedFrame,selectFrame]=useState(frames[0]);
    const [regions,setRegions]=useState(frames.map(frame=>({frame_id:frame.frame_id,review_status:'draft',annotator:'sam-auto'})));
    refresh.mockImplementation(async()=>{
      const approved=regions.map(region=>region.frame_id===selectedFrame.frame_id?{...region,review_status:'approved'}:region);
      setRegions(approved);
      return {observations:frames,regions:approved};
    });
    const draft={episodes:[],active:false,busy:false,selectedFrame,selectFrame,refresh,refreshEpisodes,
      episode:{id:'review',ended_at:1,observations:frames,annotations:[],regions,catalog:{missions:[]},
        sam_sample:{schema_version:'user_count_v1',frame_ids:['first']}}};
    return <CollectSidebar mapCanvas={<CollectStageHud/>} draft={draft} mode="dataset" mapName="warehouse" mapLoaded/>;
  }
  render(<MaskReviewWorkspace/>);
  fireEvent.click(screen.getByRole('button',{name:'Object masks (optional)'}));
  expect(mockReviewMode).toBe(true);
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  await act(async()=>{await mockOnMaskSaved({advance:true});});
  expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true');
  expect(mockReviewMode).toBe(true);
  expect(refresh.mock.invocationCallOrder[0]).toBeLessThan(refreshEpisodes.mock.invocationCallOrder[0]);
});

test.each(['draft','approved'])('the same mask editor uses per-frame %s status before entering review', status=>{
  const frame={frame_id:'frame1'};
  setup({selectedFrame:frame,episode:{id:'ep1',ended_at:1,observations:[frame],annotations:[],
    regions:[{frame_id:'frame1',review_status:status}],catalog:{missions:[]}}},true);
  expect(screen.getByLabelText('Sample count')).toBeVisible();
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  expect(mockReviewMode).toBe(status==='draft');
});

test('review enables saving an empty mask result without a manual/automatic mode switch',()=>{
  const frame={frame_id:'frame1',auto_labeling:{sam:{status:'generated'}}};
  setup({selectedFrame:frame,episode:{id:'ep1',ended_at:1,observations:[frame],annotations:[],regions:[],catalog:{missions:[]}}},true);
  expect(mockReviewMode).toBe(false);
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true');
  expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument();
  expect(mockReviewMode).toBe(true);
});
