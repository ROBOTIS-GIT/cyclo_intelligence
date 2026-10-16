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

import { act, renderHook } from '@testing-library/react';
import useQwenObservation from './useQwenObservation';
const ok = data => Promise.resolve({ ok: true, json: async () => data });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
beforeEach(() => { jest.useFakeTimers(); global.fetch = jest.fn(); });
afterEach(() => { jest.useRealTimers(); delete global.fetch; });

test('only starts when localized and waits for model readiness before camera inference', async () => {
  const view = renderHook(({enabled}) => useQwenObservation(enabled), {initialProps:{enabled:false}});
  await act(async () => view.result.current.start());
  expect(fetch).not.toHaveBeenCalled();
  view.rerender({enabled:true});
  fetch.mockImplementation(url => ok(url.endsWith('status') ? {phase:'loading'} : {}));
  await act(async () => view.result.current.start());
  expect(view.result.current.phase).toBe('loading');
  expect(fetch.mock.calls.some(([url]) => url.endsWith('observe'))).toBe(false);
  act(() => view.result.current.pause());
  await act(async () => jest.advanceTimersByTime(5000));
  expect(fetch).toHaveBeenCalledTimes(2);
});

test('uses sequential frames and discards an in-flight result after pause', async () => {
  const pending = deferred();
  fetch.mockImplementation(url => url.endsWith('observe') ? pending.promise : ok({phase:'ready'}));
  const view = renderHook(() => useQwenObservation(true));
  let started;
  await act(async () => { started = view.result.current.start(); });
  expect(fetch).toHaveBeenCalledTimes(3);
  await act(async () => jest.advanceTimersByTime(10000));
  expect(fetch).toHaveBeenCalledTimes(3);
  act(() => view.result.current.pause());
  await act(async () => { pending.resolve({ok:true,json:async()=>({text:'late description',frame_stamp:'1'})}); await started; });
  expect(view.result.current.result).toBeNull();
  expect(view.result.current.phase).toBe('paused');
});

test('shows actual text, requests the next frame after completion, and stops on lost localization', async () => {
  fetch.mockImplementation(url => ok(url.endsWith('observe') ? {text:'카메라 설명',frame_stamp:'12'} : {phase:'ready'}));
  const view = renderHook(({enabled})=>useQwenObservation(enabled),{initialProps:{enabled:true}});
  await act(async () => view.result.current.start());
  expect(view.result.current.result.text).toBe('카메라 설명');
  await act(async () => jest.advanceTimersByTime(199));
  expect(fetch).toHaveBeenCalledTimes(3);
  await act(async () => jest.advanceTimersByTime(1));
  expect(JSON.parse(fetch.mock.calls[3][1].body).previous_frame_stamp).toBe('12');
  view.rerender({enabled:false});
  const count=fetch.mock.calls.length;
  await act(async () => jest.advanceTimersByTime(10000));
  expect(fetch).toHaveBeenCalledTimes(count);
  expect(view.result.current.active).toBe(false);
});

test('missing camera shows an error without inventing a description', async () => {
  fetch.mockImplementation(url => url.endsWith('observe') ? Promise.resolve({ok:false,status:503,json:async()=>({detail:'Head camera unavailable'})}) : ok({phase:'ready'}));
  const view = renderHook(()=>useQwenObservation(true));
  await act(async () => view.result.current.start());
  expect(view.result.current.phase).toBe('waiting');
  expect(view.result.current.error).toBe('Head camera unavailable');
  expect(view.result.current.result).toBeNull();
  view.unmount();
  const count=fetch.mock.calls.length;
  await act(async () => jest.advanceTimersByTime(10000));
  expect(fetch).toHaveBeenCalledTimes(count);
});


test('Stop waits for startup acknowledgement, stops Docker, and blocks Run until shutdown completes', async () => {
  const starting = deferred(); const stopping = deferred();
  fetch.mockImplementation(url => url.endsWith('/start') ? starting.promise : stopping.promise);
  const view = renderHook(() => useQwenObservation(true));
  let run; await act(async () => { run = view.result.current.start(); });
  let stop; await act(async () => { stop = view.result.current.stop(); });
  expect(view.result.current.phase).toBe('stopping');
  expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/reasoning/start']);
  await act(async () => view.result.current.start());
  expect(fetch).toHaveBeenCalledTimes(1);
  await act(async () => { starting.resolve({ok:true,json:async()=>({phase:'starting'})}); await run; });
  expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/reasoning/start','/api/reasoning/stop']);
  await act(async () => { stopping.resolve({ok:true,json:async()=>({phase:'stopped'})}); await stop; });
  expect(view.result.current.phase).toBe('stopped');
  expect(view.result.current.needsStop).toBe(false);
  await act(async () => jest.advanceTimersByTime(5000));
  expect(fetch).toHaveBeenCalledTimes(2);
});

test('failed Docker stop remains retryable', async () => {
  fetch.mockResolvedValueOnce({ok:false,status:503,json:async()=>({detail:'Docker stop failed'})});
  const view = renderHook(()=>useQwenObservation(false));
  await act(async () => view.result.current.stop());
  expect(view.result.current.phase).toBe('error');
  expect(view.result.current.needsStop).toBe(true);
  fetch.mockImplementation(()=>ok({phase:'stopped'}));
  await act(async () => view.result.current.stop());
  expect(view.result.current.phase).toBe('stopped');
});


test('discards decisions when the operator context changes during inference', async () => {
  let finish;
  fetch.mockImplementation(url => url.endsWith('/decide') ? new Promise(resolve=>{finish=resolve;}) : ok({phase:'ready'}));
  const onResult=jest.fn();
  const h=renderHook(({key})=>useQwenObservation(true,{}, {endpoint:'decide',payload:{map_name:'map'},contextKey:key,onResult}),{initialProps:{key:'old'}});
  let pending;act(()=>{pending=h.result.current.start();});
  await act(async()=>{await Promise.resolve();await Promise.resolve();await Promise.resolve();await Promise.resolve();});
  h.rerender({key:'new'});
  await act(async()=>{finish({ok:true,json:async()=>({frame_stamp:'1',decision:{decision:'start'}})});await pending;});
  expect(onResult).not.toHaveBeenCalled();
  expect(h.result.current.result).toBeNull();h.unmount();
});


test('Collect discards an in-flight preview when its map or instruction changes', async () => {
  const pending=deferred();
  fetch.mockImplementation(url=>url.endsWith('/observe') ? pending.promise : ok({phase:'ready'}));
  const view=renderHook(({key})=>useQwenObservation(true,{context:{map_name:'warehouse'}},{contextKey:key}),{initialProps:{key:'old'}});
  let started;
  await act(async()=>{started=view.result.current.start();});
  view.rerender({key:'new'});
  await act(async()=>{pending.resolve({ok:true,json:async()=>({analysis_id:'old',decision:{decision:'start'},frame_stamp:'1'})}); await started;});
  expect(view.result.current.result).toBeNull();
  expect(fetch.mock.calls.every(([url])=>url.startsWith('/api/reasoning/'))).toBe(true);
  view.unmount();
});

test('loads the selected adapter and binds decisions to the acknowledged server session',async()=>{
 fetch.mockImplementation(url=>ok(url.endsWith('/decide') ? {frame_stamp:'1'} : {phase:'ready',session_id:'session',selected_model_id:'tuned'}));
 const h=renderHook(()=>useQwenObservation(true,{}, {endpoint:'decide',startPayload:{model_id:'tuned',map_name:'map'},payload:{map_name:'map'}}));
 await act(async()=>h.result.current.start());
 expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({model_id:'tuned',map_name:'map'});
 expect(JSON.parse(fetch.mock.calls[2][1].body).expected_session_id).toBe('session');
 expect(h.result.current.loadedModel.selected_model_id).toBe('tuned');h.unmount();
});
test('a different loaded model stops before any decision request',async()=>{
 fetch.mockImplementation(()=>ok({phase:'ready',session_id:'session',selected_model_id:'wrong'}));
 const h=renderHook(()=>useQwenObservation(true,{}, {endpoint:'decide',startPayload:{model_id:'tuned'}}));
 await act(async()=>h.result.current.start());
 expect(h.result.current.phase).toBe('error');
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/decide'))).toBe(false);h.unmount();
});

 test('pause retains the loaded model and Stop still requests container shutdown',async()=>{
 fetch.mockImplementation(url=>ok(url.endsWith('/decide')?{frame_stamp:'1'}:{phase:'ready',session_id:'s',selected_model_id:'tuned'}));
 const h=renderHook(()=>useQwenObservation(true,{}, {endpoint:'decide',keepLoadedSession:true,startPayload:{model_id:'tuned'},payload:{map_name:'map'}}));
 await act(async()=>h.result.current.start());
 act(()=>h.result.current.pause());
 expect(h.result.current.loadedModel.selected_model_id).toBe('tuned');
 expect(h.result.current.needsStop).toBe(true);
 const count=fetch.mock.calls.length;
 await act(async()=>jest.advanceTimersByTime(10000));
 expect(fetch).toHaveBeenCalledTimes(count);
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/stop'))).toBe(false);
 await act(async()=>h.result.current.start());
 expect(h.result.current.active).toBe(true);
 expect(fetch.mock.calls.filter(([url])=>url.endsWith('/start'))).toHaveLength(1);
 act(()=>h.result.current.pause());
 await act(async()=>h.result.current.stop());
 expect(fetch.mock.calls.filter(([url])=>url.endsWith('/stop'))).toHaveLength(1);
 expect(h.result.current.needsStop).toBe(false);h.unmount();
 });
