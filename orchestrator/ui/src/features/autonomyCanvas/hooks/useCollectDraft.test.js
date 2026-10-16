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

import { act, renderHook, waitFor } from '@testing-library/react';
import useCollectDraft from './useCollectDraft';
let ep;
const savedFrame = id => ({frame_id:id,capture:{received_at_ms:1000},context:{active_mission:null}});
beforeEach(() => {
  ep = {id:'ep1',scenario:'Shelf',group_id:'room',map_name:'warehouse',started_at:Date.now(),ended_at:null,observations:[savedFrame('first')],annotations:[],catalog:{missions:[]}};
  global.fetch = jest.fn(async (url, options) => {
    const body = options?.body ? JSON.parse(options.body) : null;
    if (url.endsWith('/annotations')) {
      ep.annotations=[{sample_id:'sample1',frame_ids:[body.frame_id],revision:body.expected_revision+1,target:body,review_status:body.review_status}];
    }
    if(url.endsWith('/end'))ep.ended_at=Date.now();
    return {ok:true,json:async()=>url.includes('/reason-presets?')?{map_name:'warehouse',presets:[]}:
      url.endsWith('/episodes')&&!body?{episodes:[]}:{...ep,annotations:[...ep.annotations],observations:[...ep.observations]}};
  });
});
const fields = {scenario:'Shelf',group_id:'room',robot_type:'ffw_sg2',environment:'unknown'};

test('server stores episodes and pins the original frame across newer observations', async () => {
  const hook = renderHook(props => useCollectDraft(props), {initialProps:{enabled:true,mapName:'warehouse'}});
  await act(async()=>{await hook.result.current.start(fields);});
  act(()=>hook.result.current.selectFrame(hook.result.current.episode.observations[0]));
  ep.observations.push(savedFrame('second'));
  await act(async()=>{await hook.result.current.refresh();});
  expect(hook.result.current.selectedFrame.frame_id).toBe('first');
  await act(async()=>{await hook.result.current.annotate({decision:'wait',mission_id:null,reason:'Empty',annotator:'tester',review_status:'approved'});});
  expect(JSON.parse(global.fetch.mock.calls.find(([url])=>url.endsWith('/annotations'))[1].body)).toMatchObject({frame_id:'first',expected_revision:0,review_status:'approved'});
  expect(hook.result.current.selectedFrame.labelRevision).toBe(1);
  hook.rerender({enabled:false,mapName:'warehouse'});
  await waitFor(()=>expect(hook.result.current.active).toBe(false));
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/finish'))).toBe(false);
});

test('loads reusable reasons across all episodes on the selected map', async () => {
  global.fetch.mockImplementation(async (url) => ({
    ok: true,
    json: async () => url.includes('/reason-presets?')
      ? { map_name: 'warehouse', presets: [{
        decision: 'wait', mission_id: null, reason: 'Shared map reason.', count: 4,
      }] }
      : url.endsWith('/episodes') ? { episodes: [] } : ep,
  }));
  const hook = renderHook(() => useCollectDraft({ enabled: true, mapName: 'warehouse' }));
  await waitFor(() => expect(hook.result.current.reasonPresets).toEqual([{
    decision: 'wait', mission_id: null, reason: 'Shared map reason.', count: 4,
  }]));
  expect(fetch.mock.calls.some(([url]) => (
    url.endsWith('/reason-presets?map_name=warehouse')
  ))).toBe(true);
});

test('uses the selected episode map for presets when no runtime map is loaded', async () => {
  global.fetch.mockImplementation(async (url) => ({
    ok: true,
    json: async () => url.includes('/reason-presets?')
      ? { map_name: 'warehouse', presets: [{
        decision: 'wait', mission_id: null, reason: 'Episode map reason.', count: 2,
      }] }
      : url.endsWith('/episodes') ? { episodes: [] } : { ...ep, ended_at: 1 },
  }));
  const hook = renderHook(() => useCollectDraft({ enabled: true, mapName: null }));
  await act(async () => { await hook.result.current.load('ep1'); });
  await waitFor(() => expect(hook.result.current.reasonPresets).toEqual([{
    decision: 'wait', mission_id: null, reason: 'Episode map reason.', count: 2,
  }]));
  expect(fetch.mock.calls.some(([url]) => (
    url.endsWith('/reason-presets?map_name=warehouse')
  ))).toBe(true);
});

test('storage failures are visible and do not create a successful local episode', async () => {
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  global.fetch.mockImplementation(async()=>({ok:false,json:async()=>({detail:'Disk full'})}));
  await act(async()=>{expect(await hook.result.current.start(fields)).toBe(false);});
  expect(hook.result.current.episode).toBeNull();
  expect(hook.result.current.error).toBe('Disk full');
});

test('revision stays pinned when a different reviewer saves during editing', async () => {
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  act(()=>hook.result.current.selectFrame(hook.result.current.episode.observations[0]));
  ep.annotations=[{sample_id:'sample1',frame_ids:['first'],revision:3,target:{decision:'start'}}];
  await act(async()=>{await hook.result.current.refresh();});
  await act(async()=>{await hook.result.current.annotate({decision:'unknown',reason:'Occluded',annotator:'tester'});});
  const call=global.fetch.mock.calls.find(([url])=>url.endsWith('/annotations'));
  expect(JSON.parse(call[1].body).expected_revision).toBe(0);
});


test('saving an episode closes it without adding an extra frame', async () => {
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  await act(async()=>{expect(await hook.result.current.finish()).toBe(true);});
  expect(hook.result.current.active).toBe(false);
  expect(hook.result.current.episode.observations).toHaveLength(1);
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/capture') || url.endsWith('/finish'))).toBe(false);
});

test('save analyzed frame submits the selected analysis ID without fresh camera context', async () => {
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse',context:{pose:{x:99}}}));
  await act(async()=>{await hook.result.current.start(fields);});
  global.fetch.mockImplementation(async()=>({ok:true,json:async()=>({episode:ep,saved_frame:ep.observations[0]})}));
  await act(async()=>{await hook.result.current.saveAnalysis('analysis-original');});
  const call=global.fetch.mock.calls.find(([url])=>url.endsWith('/save-analysis'));
  expect(JSON.parse(call[1].body)).toEqual({analysis_id:'analysis-original'});
});


test('unmapped collection cannot start',async()=>{
  const hook=renderHook(props=>useCollectDraft(props),{initialProps:{enabled:true,mapName:null}});
  await act(async()=>{expect(await hook.result.current.start(fields)).toBe(false);});
  expect(hook.result.current.error).toMatch(/Load a map/);
  expect(fetch.mock.calls.some(([,options])=>options?.method==='POST')).toBe(false);

});


test('deletion clears the selected frame and list only after server success',async()=>{
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  await act(async()=>{await hook.result.current.end();});
  act(()=>hook.result.current.selectFrame(ep.observations[0]));
  global.fetch.mockImplementation(async()=>({ok:false,json:async()=>({detail:'Deletion failed'})}));
  await act(async()=>{expect(await hook.result.current.discardEpisode()).toBe(false);});
  expect(hook.result.current.selectedFrame.frame_id).toBe('first');
  expect(hook.result.current.episode.id).toBe('ep1');
  global.fetch.mockImplementation(async(url,opts)=>({ok:true,json:async()=>opts?.method==='DELETE'?{deleted:true}:{episodes:[]}}));
  await act(async()=>{expect(await hook.result.current.discardEpisode()).toBe(true);});
  expect(hook.result.current.episode).toBeNull();
  expect(hook.result.current.selectedFrame).toBeNull();
  expect(hook.result.current.episodes).toEqual([]);
  expect(fetch.mock.calls.some(([url,opts])=>url.endsWith('/episodes/ep1')&&opts?.method==='DELETE')).toBe(true);
});


test('failed episode save leaves the episode active and allows retry', async () => {
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  global.fetch.mockImplementation(async()=>({ok:false,json:async()=>({detail:'Camera unavailable'})}));
  await act(async()=>{expect(await hook.result.current.finish()).toBe(false);});
  expect(hook.result.current.active).toBe(true);
  expect(hook.result.current.error).toBe('Camera unavailable');
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/capture'))).toBe(false);
});


test('continuous capture is serial, snapshots metadata, and stops scheduling', async()=>{
  jest.useFakeTimers();
  try {
    const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
    await act(async()=>{await hook.result.current.start(fields);});
    const metadata={...fields}; let complete;
    fetch.mockImplementation(url=>url.endsWith('/capture') ? new Promise(resolve=>{complete=()=>resolve({ok:true,json:async()=>({episode:ep,saved_frame:ep.observations[0]})});}) : Promise.resolve({ok:true,json:async()=>url.endsWith('/episodes')?{episodes:[]}:ep}));
    act(()=>{hook.result.current.startRecording(1000,()=>({...metadata}));});
    await act(async()=>{jest.advanceTimersByTime(5000);});
    expect(fetch.mock.calls.filter(([url])=>url.endsWith('/capture'))).toHaveLength(1);
    await act(async()=>{complete();});
    metadata.scenario='New scene';
    await act(async()=>{jest.advanceTimersByTime(1000);});
    const calls=fetch.mock.calls.filter(([url])=>url.endsWith('/capture'));
    expect(calls).toHaveLength(2);
    expect(JSON.parse(calls[0][1].body).metadata.scenario).toBe('Shelf');
    expect(JSON.parse(calls[1][1].body).metadata.scenario).toBe('New scene');
    act(()=>hook.result.current.stopRecording());
    await act(async()=>{complete();});
    await act(async()=>{jest.advanceTimersByTime(10000);});
    expect(fetch.mock.calls.filter(([url])=>url.endsWith('/capture'))).toHaveLength(2);
    hook.unmount();
  } finally {jest.useRealTimers();}
});

test('camera error stops continuous capture without closing the episode', async()=>{
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  fetch.mockImplementation(async()=>({ok:false,json:async()=>({detail:'Camera unavailable'})}));
  await act(async()=>{await hook.result.current.startRecording(1000,()=>fields);});
  expect(hook.result.current.recording).toBe(false);
  expect(hook.result.current.active).toBe(true);
  expect(hook.result.current.error).toBe('Camera unavailable');
});


test('a failed segment boundary prevents camera capture', async()=>{
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  const prepare=jest.fn().mockResolvedValue(false);
  await act(async()=>{await hook.result.current.startRecording(1000,()=>fields,prepare);});
  expect(prepare).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/capture'))).toBe(false);
  expect(hook.result.current.recording).toBe(false);
});

test('Stop during a pending segment boundary never captures a late frame', async()=>{
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  let resolve, recording;
  const prepare=()=>new Promise(done=>{resolve=done;});
  act(()=>{recording=hook.result.current.startRecording(1000,()=>fields,prepare);});
  act(()=>hook.result.current.stopRecording());
  await act(async()=>{resolve(true);await recording;});
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/capture'))).toBe(false);
});

test('discard closes even an empty active episode before deleting it', async()=>{
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  ep.observations=[];
  await act(async()=>{await hook.result.current.start(fields);});
  fetch.mockClear();
  await act(async()=>{await hook.result.current.discardEpisode();});
  const calls=fetch.mock.calls;
  const end=calls.findIndex(([url])=>url.endsWith('/end'));
  const deletion=calls.findIndex(([,options])=>options?.method==='DELETE');
  expect(end).toBeGreaterThanOrEqual(0); expect(deletion).toBeGreaterThan(end);
  expect(hook.result.current.episode).toBeNull();
  hook.unmount();
});

test('deleting another saved episode preserves the active episode and selected frame', async () => {
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>{await hook.result.current.start(fields);});
  act(()=>hook.result.current.selectFrame(hook.result.current.episode.observations[0]));
  global.fetch.mockClear();
  await act(async()=>{expect(await hook.result.current.deleteEpisode('other')).toBe(true);});
  expect(global.fetch.mock.calls.some(([url,opts])=>url.endsWith('/episodes/other') && opts.method==='DELETE')).toBe(true);
  expect(global.fetch.mock.calls.some(([url])=>url.endsWith('/end'))).toBe(false);
  expect(hook.result.current.episode.id).toBe('ep1');
  expect(hook.result.current.active).toBe(true);
  expect(hook.result.current.selectedFrame.frame_id).toBe('first');
});

test('strict episodes skip pending execution before capture and retry only a cached-frame context mismatch',async()=>{
 jest.useFakeTimers();
 try {
  ep.capture_contract='task_aligned_v1';
  let ready=false,captures=0;
  const getExecutionContext=()=>({reasoning_task:'patrol_supervision',recording_execution:ready?{status:'running'}:null});
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse',getExecutionContext}));
  await act(async()=>{await hook.result.current.start(fields);});
  const previous=fetch.getMockImplementation();
  fetch.mockImplementation(async(url,options)=>{
   if(!url.endsWith('/capture'))return previous(url,options);
   captures++;
   if(captures===1)return {ok:false,status:422,json:async()=>({detail:'Capture-time execution context could not be matched. Record a new frame after the current state is available.'})};
   return {ok:true,json:async()=>({episode:ep,saved_frame:ep.observations[0]})};
  });
  let recording; await act(async()=>{recording=hook.result.current.startRecording(1000,()=>null);});
  expect(captures).toBe(0);
  ready=true;await act(async()=>jest.advanceTimersByTime(250));
  expect(captures).toBe(1);expect(hook.result.current.recording).toBe(true);expect(hook.result.current.error).toBe('');
  await act(async()=>jest.advanceTimersByTime(250));
  expect(captures).toBe(2);expect(hook.result.current.recording).toBe(true);
  act(()=>hook.result.current.stopRecording());
  await act(async()=>{jest.advanceTimersByTime(1000);await recording;});
  hook.unmount();
 } finally {jest.useRealTimers();}
});

test.each([[422,'The current patrol segment does not belong to this episode route.'],[500,'Disk full'],[503,'Head camera unavailable']])('continuous capture surfaces non-transient %s errors',async(status,message)=>{
 const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
 await act(async()=>hook.result.current.start(fields));
 const previous=fetch.getMockImplementation();let captures=0;
 fetch.mockImplementation(async(url,options)=>url.endsWith('/capture')?(captures++,{ok:false,status,json:async()=>({detail:message})}):previous(url,options));
 await act(async()=>hook.result.current.startRecording(1000,()=>null));
 expect(captures).toBe(1);expect(hook.result.current.recording).toBe(false);expect(hook.result.current.error).toBe(message);
 hook.unmount();
});

test('persistent capture-time mismatch stops after five seconds with an actionable error',async()=>{
 jest.useFakeTimers();
 try {
  ep.capture_contract='task_aligned_v1';
  const getExecutionContext=()=>({reasoning_task:'patrol_supervision',recording_execution:{status:'running'}});
  const hook=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse',getExecutionContext}));
  await act(async()=>hook.result.current.start(fields));
  const previous=fetch.getMockImplementation();let captures=0;
  fetch.mockImplementation(async(url,options)=>{
   if(!url.endsWith('/capture'))return previous(url,options);
   captures++;
   return {ok:false,status:422,json:async()=>({detail:'Capture-time execution context could not be matched. Record a new frame after the current state is available.'})};
  });
  let recording;await act(async()=>{recording=hook.result.current.startRecording(1000,()=>null);});
  for(let i=0;i<19;i++)await act(async()=>jest.advanceTimersByTime(250));
  expect(hook.result.current.recording).toBe(true);expect(hook.result.current.error).toBe('');
  await act(async()=>{jest.advanceTimersByTime(250);await recording;});
  expect(captures).toBe(21);
  expect(hook.result.current.recording).toBe(false);
  expect(hook.result.current.error).toMatch(/5 seconds.*camera updates and system clocks/);
  await act(async()=>jest.advanceTimersByTime(1000));expect(captures).toBe(21);
  hook.unmount();
 } finally {jest.useRealTimers();}
});

test('return transition publishes the linked Patrol child before another capture can start',async()=>{
 const child={...ep,id:'return-ep',episode_type:'patrol',capture_segment:'mission_return',source_episode_id:'ep1',observations:[],annotations:[]};
 global.fetch.mockImplementation(async(url,options)=>({ok:true,json:async()=>url.endsWith('/return')?child:url.endsWith('/episodes')&&!options?.body?{episodes:[]}:url.includes('/reason-presets')?{presets:[]}:ep}));
 const h=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
 await act(async()=>h.result.current.start(fields));
 let result;await act(async()=>{result=await h.result.current.startReturnEpisode();});
 expect(result.id).toBe('return-ep');
 expect(h.result.current.episode).toMatchObject({id:'return-ep',episode_type:'patrol',source_episode_id:'ep1'});
 expect(fetch.mock.calls.filter(([url])=>url.endsWith('/return'))).toHaveLength(1);
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/capture'))).toBe(false);
});

test('failed return transition leaves the original episode available and reports error',async()=>{
 const h=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
 await act(async()=>h.result.current.start(fields));
 global.fetch.mockImplementation(async()=>({ok:false,status:422,json:async()=>({detail:'The return observation was deleted.'})}));
 await act(async()=>expect(await h.result.current.startReturnEpisode()).toBeNull());
 expect(h.result.current.episode.id).toBe('ep1');
 expect(h.result.current.error).toBe('The return observation was deleted.');
});


test('segment switches the active episode without issuing any motion command', async()=>{
 const h=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
 await act(async()=>h.result.current.start(fields));
 const old=fetch.getMockImplementation();
 fetch.mockImplementation((url,options)=>url.endsWith('/segment')?Promise.resolve({ok:true,json:async()=>({...ep,id:'segment2',episode_type:'observation',observations:[]})}):old(url,options));
 let child;await act(async()=>{child=await h.result.current.startSegment({episode_type:'observation',observation_spot_id:'table'});});
 expect(child.id).toBe('segment2');expect(h.result.current.episode.id).toBe('segment2');
 expect(fetch).toHaveBeenCalledWith('/api/collection/episodes/ep1/segment',expect.objectContaining({body:JSON.stringify({episode_type:'observation',observation_spot_id:'table',map_name:'warehouse'})}));
});


test('Save during segment publication waits and closes the new episode',async()=>{
 const h=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
 await act(async()=>h.result.current.start(fields));
 const original=fetch.getMockImplementation();let publish;
 fetch.mockImplementation((url,options)=>url.endsWith('/segment')?new Promise(resolve=>{
  publish=()=>{ep={...ep,id:'next',observations:[],annotations:[]};resolve({ok:true,json:async()=>ep});};
 }):original(url,options));
 let transition,save;
 act(()=>{transition=h.result.current.startSegment({episode_type:'patrol',patrol_spot_ids:['a']});});
 act(()=>{save=h.result.current.finish({allowEmpty:true});});
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/end'))).toBe(false);
 await act(async()=>{publish();await transition;await save;});
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/episodes/next/end'))).toBe(true);
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/episodes/ep1/end'))).toBe(false);
 expect(h.result.current.episode.id).toBe('next');expect(h.result.current.active).toBe(false);
});


test('continuous capture waits for a storage lock and resumes after it is released',async()=>{
 jest.useFakeTimers();
 try {
  const h=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>h.result.current.start(fields));
  const previous=fetch.getMockImplementation();let release,captures=0;
  fetch.mockImplementation(async(url,options)=>{
   if(url.endsWith('/episodes')&&!options?.body)return new Promise(resolve=>{release=()=>resolve({ok:true,json:async()=>({episodes:[]})});});
   if(url.endsWith('/capture')){captures++;return {ok:true,json:async()=>({episode:ep,saved_frame:savedFrame(`f${captures}`)})};}
   return previous(url,options);
  });
  let refresh,recording;
  act(()=>{refresh=h.result.current.refreshEpisodes();recording=h.result.current.startRecording(500,()=>null);});
  await act(async()=>jest.advanceTimersByTime(250));
  expect(captures).toBe(0);expect(h.result.current.recording).toBe(true);
  await act(async()=>{release();await refresh;});
  await act(async()=>jest.advanceTimersByTime(25));
  expect(captures).toBe(1);expect(h.result.current.recording).toBe(true);
  await act(async()=>jest.advanceTimersByTime(500));
  expect(captures).toBe(2);
  act(()=>h.result.current.stopRecording());
  await act(async()=>{jest.advanceTimersByTime(500);await recording;});
  h.unmount();
 } finally {jest.useRealTimers();}
});

test('an operation that acquires the lock during a pre-frame wait does not end capture',async()=>{
 jest.useFakeTimers();
 try {
  const h=renderHook(()=>useCollectDraft({enabled:true,mapName:'warehouse'}));
  await act(async()=>h.result.current.start(fields));
  const previous=fetch.getMockImplementation();let releaseList,releaseBoundary,captures=0;
  fetch.mockImplementation(async(url,options)=>{
   if(url.endsWith('/episodes')&&!options?.body)return new Promise(resolve=>{releaseList=()=>resolve({ok:true,json:async()=>({episodes:[]})});});
   if(url.endsWith('/capture')){captures++;return {ok:true,json:async()=>({episode:ep,saved_frame:savedFrame(`f${captures}`)})};}
   return previous(url,options);
  });
  const boundary=jest.fn().mockImplementationOnce(()=>new Promise(resolve=>{releaseBoundary=resolve;})).mockResolvedValue(true);
  let recording,refresh;
  await act(async()=>{recording=h.result.current.startRecording(500,()=>null,boundary);});
  act(()=>{refresh=h.result.current.refreshEpisodes();});
  await act(async()=>releaseBoundary(true));
  expect(captures).toBe(0);expect(h.result.current.recording).toBe(true);
  await act(async()=>{releaseList();await refresh;});
  await act(async()=>jest.advanceTimersByTime(25));
  expect(captures).toBe(1);expect(h.result.current.recording).toBe(true);
  await act(async()=>jest.advanceTimersByTime(500));expect(captures).toBe(2);
  act(()=>h.result.current.stopRecording());
  await act(async()=>{jest.advanceTimersByTime(500);await recording;});h.unmount();
 } finally {jest.useRealTimers();}
});


test('reselecting a loaded episode preserves the selected frame without refetching or becoming busy', async () => {
  ep.ended_at = 1;
  ep.observations.push(savedFrame('second'));
  const hook = renderHook(() => useCollectDraft({enabled:true,mapName:null}));
  await act(async () => { await hook.result.current.load('ep1'); });
  act(() => hook.result.current.selectFrame(hook.result.current.episode.observations[1]));
  const selected = hook.result.current.selectedFrame;
  const loaded = hook.result.current.episode;
  fetch.mockClear();
  for (let click=0;click<3;click++) {
    await act(async () => {
      const pending = hook.result.current.load('ep1');
      expect(hook.result.current.busy).toBe(false);
      expect(await pending).toBe(true);
    });
    expect(hook.result.current.selectedFrame).toBe(selected);
    expect(hook.result.current.episode).toBe(loaded);
  }
  expect(fetch).not.toHaveBeenCalled();
  const next = {...ep,id:'ep2',observations:[savedFrame('third')]};
  fetch.mockImplementation(async url => ({ok:true,json:async()=>url.includes('/reason-presets?') ? {presets:[]} : next}));
  await act(async () => { await hook.result.current.load('ep2'); });
  expect(fetch.mock.calls.some(([url]) => url.endsWith('/episodes/ep2'))).toBe(true);
  expect(hook.result.current.episode.id).toBe('ep2');
  expect(hook.result.current.selectedFrame).toBeNull();
});
