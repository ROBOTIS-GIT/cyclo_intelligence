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

import { act, renderHook, waitFor } from "@testing-library/react";
import useAutonomyOperation from "./useAutonomyOperation";
import useOperateModel from "./useOperateModel";
jest.mock("./useOperateModel");
let server, settings;
beforeEach(() => {
 localStorage.clear();
 server = {phase:'idle', active:false, needsStop:false, events:[]};
 settings = {plan:{revision:1, mode:'fixed', observations:[{spot_id:'a',mission_ids:[]}],excluded_mission_ids:[]},spots:[],missions:[]};
 useOperateModel.mockReturnValue({selected:{id:'trained'}, models:[], select:jest.fn()});
 global.fetch=jest.fn(async (url, options) => {
   if(url.endsWith('/start')) server={...server,phase:'positioning',active:true,needsStop:true};
   if(url.endsWith('/stop')) server={...server,phase:'stopped',active:false,needsStop:false};
   if(url.includes('/maps/') && options?.method === 'PUT') settings={...settings,plan:{...JSON.parse(options.body),revision:settings.plan.revision+1}};
   return {ok:true,json:async()=>url.includes('/maps/') ? settings : server};
 });
});
const mount = () => renderHook(()=>useAutonomyOperation({enabled:true,mapName:'Home'}));
test('starts the backend session with plan revision and selected model',async()=>{
 const h=mount(); await waitFor(()=>expect(h.result.current.canStart).toBe(true));
 await act(async()=>h.result.current.start());
 expect(fetch).toHaveBeenCalledWith('/api/autonomy/start',expect.objectContaining({body:JSON.stringify({map_name:'Home',model_id:'trained',localized:true,plan_revision:1})}));
 expect(h.result.current.active).toBe(true);
 expect(h.result.current.modelChoice.locked).toBe(true);
 h.unmount();
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/stop'))).toBe(false);
});
test('reconnects to server execution without loading an ObservationMode mission',async()=>{
 server={...server,active:true,needsStop:true,phase:'executing',mission:{mission_name:'CleanTable'}};
 const h=mount();await waitFor(()=>expect(h.result.current.mission?.mission_name).toBe('CleanTable'));
 expect(localStorage.getItem('autonomy.serverOwned')).toBe('true');
 expect(h.result.current.canStart).toBe(false);h.unmount();
});
test('Operate edits and saves the shared plan before allowing Run',async()=>{
 const h=mount();await waitFor(()=>expect(h.result.current.config.plan).toBeTruthy());
 act(()=>h.result.current.config.edit({mode:'patrol'}));
 expect(h.result.current.config.draft.mode).toBe('patrol');
 expect(h.result.current.canStart).toBe(false);
 let started;await act(async()=>{started=await h.result.current.start();});
 expect(started).toBe(false);
 await act(async()=>h.result.current.config.save());
 await waitFor(()=>expect(h.result.current.config.plan.revision).toBe(2));
 expect(h.result.current.config.dirty).toBe(false);
 expect(h.result.current.canStart).toBe(true);h.unmount();
});
test('active autonomy locks plan editing and saving',async()=>{
 server={...server,active:true,needsStop:true,phase:'watching'};
 const h=mount();await waitFor(()=>expect(h.result.current.config.plan).toBeTruthy());
 await waitFor(()=>expect(h.result.current.config.locked).toBe(true));
 act(()=>h.result.current.config.edit({mode:'patrol'}));
 expect(h.result.current.config.draft.mode).toBe('fixed');
 let saved;await act(async()=>{saved=await h.result.current.config.save();});
 expect(saved).toBe(false);
 expect(fetch.mock.calls.some(([,options])=>options?.method==='PUT')).toBe(false);h.unmount();
});
test('Stop clears server ownership only after acknowledgement',async()=>{
 server={...server,active:true,needsStop:true};const h=mount();
 await waitFor(()=>expect(h.result.current.needsStop).toBe(true));
 let success;await act(async()=>{success=await h.result.current.stop();});
 expect(success).toBe(true);expect(h.result.current.needsStop).toBe(false);
 expect(localStorage.getItem('autonomy.serverOwned')).toBe('false');h.unmount();
});
test('failed Stop keeps controls locked and exposes error',async()=>{
 server={...server,active:true,needsStop:true};const h=mount();
 await waitFor(()=>expect(h.result.current.needsStop).toBe(true));
 fetch.mockImplementation(async()=>({ok:false,json:async()=>({detail:'Cannot stop navigation'})}));
 let success;await act(async()=>{success=await h.result.current.stop();});
 expect(success).toBe(false);expect(h.result.current.needsStop).toBe(true);
 expect(h.result.current.error).toBe('Cannot stop navigation');h.unmount();
});

test('a shared plan update is refreshed in Operate before the next Run',async()=>{
 const h=mount();await waitFor(()=>expect(h.result.current.config.plan?.revision).toBe(1));
 settings={...settings,plan:{...settings.plan,revision:2,mode:'patrol'}};
 act(()=>window.dispatchEvent(new CustomEvent('observation-spots-changed',{detail:{mapName:'Home'}})));
 await waitFor(()=>expect(h.result.current.config.plan.revision).toBe(2));
 await act(async()=>h.result.current.start());
 expect(fetch).toHaveBeenCalledWith('/api/autonomy/start',expect.objectContaining({body:JSON.stringify({map_name:'Home',model_id:'trained',localized:true,plan_revision:2})}));
 h.unmount();
});

test('manual pause and resume use dedicated endpoints without stopping the server session',async()=>{
 const h=mount();await waitFor(()=>expect(h.result.current.config.draft).toBeTruthy());
 await act(async()=>h.result.current.pause());
 await act(async()=>h.result.current.resume());
 expect(fetch).toHaveBeenCalledWith('/api/autonomy/pause',expect.objectContaining({method:'POST'}));
 expect(fetch).toHaveBeenCalledWith('/api/autonomy/resume',expect.objectContaining({method:'POST'}));
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/stop'))).toBe(false);h.unmount();
});

test('Stop waits for an in-flight Start before releasing server and navigation ownership', async () => {
 const h=mount();await waitFor(()=>expect(h.result.current.canStart).toBe(true));
 let resolveStart;
 const original=fetch.getMockImplementation();
 fetch.mockImplementation((url,options)=>url.endsWith('/start')
  ? new Promise(resolve=>{resolveStart=()=>{server={...server,active:true,needsStop:true,phase:'positioning'};resolve({ok:true,json:async()=>server});};})
  : original(url,options));
 let starting,stopping;
 act(()=>{starting=h.result.current.start();});
 expect(h.result.current.needsStop).toBe(true);
 expect(h.result.current.canStart).toBe(false);
 expect(localStorage.getItem('autonomy.serverOwned')).toBe('true');
 act(()=>{stopping=h.result.current.stop();});
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/stop'))).toBe(false);
 await act(async()=>{resolveStart();await starting;await stopping;});
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/stop'))).toBe(true);
 expect(h.result.current.needsStop).toBe(false);
 expect(h.result.current.active).toBe(false);
 expect(localStorage.getItem('autonomy.serverOwned')).toBe('false');h.unmount();
});

test('a failed Start response retains ownership until an authoritative status or Stop confirms idle', async () => {
 const h=mount();await waitFor(()=>expect(h.result.current.canStart).toBe(true));
 const original=fetch.getMockImplementation();
 fetch.mockImplementation((url,options)=>url.endsWith('/start')
  ? Promise.reject(new Error('Connection dropped after Start')) : original(url,options));
 await act(async()=>{await h.result.current.start();});
 expect(h.result.current.needsStop).toBe(true);
 expect(h.result.current.error).toBe('Connection dropped after Start');
 await act(async()=>{await h.result.current.stop();});
 expect(h.result.current.needsStop).toBe(false);h.unmount();
});


test('late lower-revision polling cannot release current server ownership', async () => {
 server={...server,phase:'executing',active:true,needsStop:true,
   execution:{epoch:'boot',session_id:'s',revision:8,lifecycle:'active',owns_execution:true}};
 const h=mount();await waitFor(()=>expect(h.result.current.execution?.revision).toBe(8));
 const count=fetch.mock.calls.filter(([url])=>url.endsWith('/status')).length;
 server={...server,phase:'stopped',active:false,needsStop:false,
   execution:{...server.execution,revision:7,lifecycle:'stopped',owns_execution:false}};
 await waitFor(()=>expect(fetch.mock.calls.filter(([url])=>url.endsWith('/status')).length).toBeGreaterThan(count),{timeout:1600});
 expect(h.result.current.execution.revision).toBe(8);
 expect(h.result.current.needsStop).toBe(true);
 expect(localStorage.getItem('autonomy.serverOwned')).toBe('true');
 h.unmount();
});


test('identical status polls and parent renders preserve formatted event references', async () => {
 jest.useFakeTimers();
 const formatTime=jest.spyOn(Date.prototype,'toLocaleTimeString');
 server={...server,events:[{time_ms:1000,title:'Ready',detail:''}]};
 const h=mount();
 try {
  await waitFor(()=>expect(h.result.current.events).toHaveLength(1));
  const events=h.result.current.events;
  const formatted=formatTime.mock.calls.length;
  h.rerender();
  expect(h.result.current.events).toBe(events);
  expect(formatTime).toHaveBeenCalledTimes(formatted);
  const polls=fetch.mock.calls.filter(([url])=>url.endsWith('/status')).length;
  server=JSON.parse(JSON.stringify(server));
  await act(async()=>{jest.advanceTimersByTime(750);});
  expect(fetch.mock.calls.filter(([url])=>url.endsWith('/status')).length).toBeGreaterThan(polls);
  expect(h.result.current.events).toBe(events);
  expect(formatTime).toHaveBeenCalledTimes(formatted);
 } finally {
  h.unmount();formatTime.mockRestore();jest.useRealTimers();
 }
});

test('polls while Operate is disabled and accepts changed fields at the same execution revision', async () => {
 jest.useFakeTimers();
 server={...server,execution:{epoch:'boot',session_id:'s',revision:8,lifecycle:'idle',owns_execution:false},
  loadedModel:{phase:'loading'},result:{decision_id:'first'}};
 const h=renderHook(()=>useAutonomyOperation({enabled:false,mapName:'Home'}));
 try {
  await waitFor(()=>expect(h.result.current.execution?.revision).toBe(8));
  server={...server,loadedModel:{phase:'ready'},result:{decision_id:'second'},
   events:[{time_ms:1000,title:'Ready',detail:''}]};
  await act(async()=>{jest.advanceTimersByTime(750);});
  expect(h.result.current.loadedModel.phase).toBe('ready');
  expect(h.result.current.result.decision_id).toBe('second');
  expect(h.result.current.events).toHaveLength(1);
  server={...server,needsStop:true,active:true,
   execution:{...server.execution,revision:9,lifecycle:'active',owns_execution:true}};
  await act(async()=>{jest.advanceTimersByTime(750);});
  expect(h.result.current.needsStop).toBe(true);
  expect(localStorage.getItem('autonomy.serverOwned')).toBe('true');
 } finally {
  h.unmount();jest.useRealTimers();
 }
});
