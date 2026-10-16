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

import {act,renderHook,waitFor} from '@testing-library/react';
import useObservationPlan from './useObservationPlan';
import {autonomyRequest} from './autonomyApi';
jest.mock('./autonomyApi',()=>({autonomyRequest:jest.fn(),OBSERVATION_PLAN_CHANGED:'observation-spots-changed'}));
let view;
beforeEach(()=>{
 view={plan:{revision:1,mode:'fixed',observations:[],excluded_mission_ids:[]},spots:[],missions:[]};
 autonomyRequest.mockReset();
 autonomyRequest.mockImplementation(async(path,method,body)=>{
  if(method==='PUT') view={...view,plan:{...body,revision:2}};
  return view;
 });
});
test('Record saves mission bindings and patrol order once, then readers see the same version',async()=>{
 const record=renderHook(()=>useObservationPlan({mapName:'house',editable:true}));
 const operate=renderHook(()=>useObservationPlan({mapName:'house'}));
 await waitFor(()=>expect(record.result.current.plan).toBeTruthy());
 await waitFor(()=>expect(operate.result.current.plan).toBeTruthy());
 act(()=>record.result.current.edit({mode:'patrol',observations:[{spot_id:'table',mission_ids:['clean']},{spot_id:'laundry',mission_ids:['wash']}]}));
 expect(record.result.current.dirty).toBe(true);
 await act(async()=>expect(await record.result.current.save()).toBe(true));
 await waitFor(()=>expect(operate.result.current.plan.revision).toBe(2));
 expect(operate.result.current.plan.observations.map(p=>p.spot_id)).toEqual(['table','laundry']);
 expect(record.result.current.dirty).toBe(false);
});
test('failed revision save keeps the editor draft and reports the conflict',async()=>{
 const h=renderHook(()=>useObservationPlan({mapName:'house',editable:true}));
 await waitFor(()=>expect(h.result.current.plan).toBeTruthy());
 act(()=>h.result.current.edit({mode:'patrol'}));
 autonomyRequest.mockRejectedValue(new Error('Settings changed. Reload.'));
 await act(async()=>expect(await h.result.current.save()).toBe(false));
 expect(h.result.current.dirty).toBe(true);
 expect(h.result.current.draft.mode).toBe('patrol');
 expect(h.result.current.error).toMatch(/Settings changed/);
});
test('execution locks editing and saving',async()=>{
 const h=renderHook(props=>useObservationPlan({mapName:'house',editable:true,...props}),{initialProps:{locked:false}});
 await waitFor(()=>expect(h.result.current.plan).toBeTruthy());
 act(()=>h.result.current.edit({mode:'patrol'}));h.rerender({locked:true});
 await act(async()=>expect(await h.result.current.save()).toBe(false));
 expect(autonomyRequest.mock.calls.some(call=>call[1]==='PUT')).toBe(false);
});
