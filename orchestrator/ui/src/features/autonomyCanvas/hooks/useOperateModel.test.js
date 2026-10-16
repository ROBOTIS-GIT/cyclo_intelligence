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
import useOperateModel from './useOperateModel';
const model={id:'one',model_name:'Custom'};
beforeEach(()=>{localStorage.clear();global.fetch=jest.fn().mockResolvedValue({ok:true,json:async()=>({models:[model]})});});
afterEach(()=>{delete global.fetch;});
test('remembers selection per map and clears unavailable models after refresh',async()=>{
 const h=renderHook(({map})=>useOperateModel(map),{initialProps:{map:'map'}});
 await waitFor(()=>expect(h.result.current.models).toHaveLength(1));
 expect(h.result.current.selected).toBeNull();
 act(()=>h.result.current.select('one'));
 expect(localStorage.getItem('autonomy.operate.model.map')).toBe('one');
 h.rerender({map:'another'});
 expect(h.result.current.selected).toBeNull();
 await waitFor(()=>expect(h.result.current.loading).toBe(false));
 expect(h.result.current.selected).toBeNull();
 h.rerender({map:'map'});
 await waitFor(()=>expect(h.result.current.selected?.id).toBe('one'));
 fetch.mockResolvedValue({ok:true,json:async()=>({models:[]})});
 await act(async()=>h.result.current.refresh());
 expect(h.result.current.selected).toBeNull();h.unmount();
});
test('ignores a stale map response',async()=>{
 let finish;fetch.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
 const h=renderHook(({map})=>useOperateModel(map),{initialProps:{map:'old'}});
 h.rerender({map:'new'});
 await waitFor(()=>expect(h.result.current.models).toHaveLength(1));
 await act(async()=>finish({ok:true,json:async()=>({models:[{id:'old'}]})}));
 expect(h.result.current.models).toEqual([model]);h.unmount();
});
