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
import useRecordNavigation from "./useRecordNavigation";
const spots=[{id:"a",label:"Dining",pose:{x:1,y:0,yaw:0}},{id:"b",label:"Laundry",pose:{x:2,y:0,yaw:1}}];
function initial(overrides={}) {
 const plan={route_closed:false,observations:spots.map(spot=>({spot_id:spot.id,dwell_seconds:0})),...overrides};
 plan.observations=plan.observations.map(point=>({departure_policy:"timed",...point}));
 const resolves=[];
 const sendGoal=jest.fn((x,y,yaw,signal)=>new Promise(resolve=>{
  resolves.push(resolve); signal.addEventListener("abort",()=>resolve({status:"CANCELED"}),{once:true});
 }));
 return {enabled:true,mapName:"house",localized:true,manager:{mode:"mission",spots,configuration:{plan},
  select:jest.fn(),refresh:jest.fn().mockResolvedValue({spots,plan})},sendGoal,resolves,
  cancelGoal:jest.fn().mockResolvedValue({ok:true}),onMessage:jest.fn()};
}
async function start(hook,method="start",id) {
 let pending; act(()=>{pending=hook.result.current[method](id);});
 await waitFor(()=>expect(["switching","preparing","launching"]).not.toContain(hook.result.current.status));
 let value;await act(async()=>{value=await pending;});return value;
}
test("Start follows saved observation route even when Mission capture was selected",async()=>{
 const p=initial(),h=renderHook(useRecordNavigation,{initialProps:p});
 expect(await start(h)).toBe(true);
 expect(p.sendGoal).toHaveBeenCalledWith(1,0,0,expect.any(AbortSignal));
 expect(h.result.current.fromSpotId).toBeNull();expect(h.result.current.toSpotId).toBe("a");
 await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(p.sendGoal).toHaveBeenCalledTimes(2));
 expect(h.result.current.fromSpotId).toBe("a");expect(h.result.current.toSpotId).toBe("b");
 await act(async()=>p.resolves[1]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(h.result.current.status).toBe("done"));
 expect(p.sendGoal).toHaveBeenCalledTimes(2);
});
test("Closed loop wraps to first observation until Stop",async()=>{
 const p=initial({route_closed:true}),h=renderHook(useRecordNavigation,{initialProps:p});
 await start(h);await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(p.sendGoal).toHaveBeenCalledTimes(2));
 await act(async()=>p.resolves[1]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(p.sendGoal).toHaveBeenCalledTimes(3));
 expect(h.result.current.fromSpotId).toBe("b");expect(h.result.current.toSpotId).toBe("a");
 await act(async()=>h.result.current.stop());
 expect(h.result.current.held).toBe(true);
 expect(h.result.current.isRunning).toBe(false);
});
test("Stop preserves unfinished waypoint and Start resumes its cursor",async()=>{
 const p=initial(),h=renderHook(useRecordNavigation,{initialProps:p});
 await start(h);await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(p.sendGoal).toHaveBeenCalledTimes(2));
 await act(async()=>h.result.current.stop());
 expect(h.result.current.status).toBe("paused");
 await start(h);
 expect(p.sendGoal).toHaveBeenLastCalledWith(2,0,1,expect.any(AbortSignal));
 await act(async()=>h.result.current.stop());
});
test("Stop during route refresh prevents late navigation",async()=>{
 const p=initial();let resolve;p.manager.refresh.mockImplementation(()=>new Promise(r=>{resolve=r;}));
 const h=renderHook(useRecordNavigation,{initialProps:p});let pending;act(()=>{pending=h.result.current.start();});
 await waitFor(()=>expect(p.manager.refresh).toHaveBeenCalled());
 await act(async()=>h.result.current.stop());
 await act(async()=>{resolve({spots,plan:p.manager.configuration.plan});expect(await pending).toBe(false);});
 expect(p.sendGoal).not.toHaveBeenCalled();expect(h.result.current.held).toBe(false);
});
test("changed route requires review and unsaved route refuses to start",async()=>{
 const p=initial();p.manager.refresh.mockResolvedValue({spots:[spots[0]],plan:p.manager.configuration.plan});
 const h=renderHook(useRecordNavigation,{initialProps:p});
 await act(async()=>expect(await h.result.current.start()).toBe(false));
 expect(p.sendGoal).not.toHaveBeenCalled();expect(p.onMessage).toHaveBeenCalledWith(expect.stringContaining("changed"));
 h.rerender({...p,manager:{...p.manager,planDirty:true}});
 await act(async()=>expect(await h.result.current.start()).toBe(false));
});
test("failed cancellation blocks subsequent goals until Stop succeeds",async()=>{
 const p=initial(),h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 p.cancelGoal.mockRejectedValueOnce(new Error("cancel transport failed"));
 await act(async()=>expect(h.result.current.stop()).rejects.toThrow("cancel transport failed"));
 await act(async()=>expect(await h.result.current.start()).toBe(false));
 expect(p.sendGoal).toHaveBeenCalledTimes(1);
 await act(async()=>h.result.current.stop());await start(h);
 expect(p.sendGoal).toHaveBeenCalledTimes(2);await act(async()=>h.result.current.stop());
});
test("manual go-to reaches saved observation outside route without running the full route",async()=>{
 const p=initial({observations:[{spot_id:"a",dwell_seconds:0}]}),h=renderHook(useRecordNavigation,{initialProps:p});
 await start(h,"goToSpot","b");expect(p.sendGoal).toHaveBeenCalledWith(2,0,1,expect.any(AbortSignal));
 await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(h.result.current.status).toBe("done"));expect(p.sendGoal).toHaveBeenCalledTimes(1);
});
test("arrival posture runs before dwelling and never launches a child mission",async()=>{
 const xml='<root><BehaviorTree ID="Main"><Wait duration="1"/></BehaviorTree></root>';
 const p=initial({observations:[{spot_id:"a",arrival_bt:xml,dwell_seconds:10}]});
 p.runtime={getFlags:()=>({btNodeIsUp:true}),btStatusRef:{current:"running"},
 callService:jest.fn().mockResolvedValue({success:true}),stopBt:jest.fn().mockResolvedValue(true)};
 const h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(p.runtime.callService).toHaveBeenCalledWith('/bt/load_and_run','interfaces/srv/LoadAndRunTree',{tree_xml:xml},30000));
 expect(h.result.current.dwelling).toBe(false);await act(async()=>h.result.current.stop());
});


test("Pause at a dwell keeps its remaining time and Resume does not replay the arrival goal", async()=>{
 jest.useFakeTimers();
 const p=initial({route_closed:true,observations:[{spot_id:"a",dwell_seconds:10},{spot_id:"b",dwell_seconds:10}]}),h=renderHook(useRecordNavigation,{initialProps:p});
 try {
  await start(h);await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
  expect(h.result.current.dwelling).toBe(true);
  await act(async()=>jest.advanceTimersByTime(4000));
  await act(async()=>h.result.current.pause());
  const remaining=h.result.current.getPatrolContinuation().dwellRemainingMs;
  expect(remaining).toBeGreaterThan(0);expect(remaining).toBeLessThanOrEqual(6000);
  await act(async()=>jest.advanceTimersByTime(15000));
  expect(p.sendGoal).toHaveBeenCalledTimes(1);
  await act(async()=>expect(await h.result.current.resume()).toBe(true));
  expect(h.result.current.dwelling).toBe(true);expect(p.sendGoal).toHaveBeenCalledTimes(1);
  await act(async()=>jest.advanceTimersByTime(remaining-1));expect(p.sendGoal).toHaveBeenCalledTimes(1);
  await act(async()=>jest.advanceTimersByTime(1));
  expect(p.sendGoal).toHaveBeenCalledTimes(2);expect(h.result.current.toSpotId).toBe("b");
  await act(async()=>h.result.current.stop());
 } finally { h.unmount();jest.useRealTimers(); }
});


test("Next observation skips a 120-second dwell once and cancels the old timer",async()=>{
 jest.useFakeTimers();
 const p=initial({route_closed:true,observations:spots.map(spot=>({spot_id:spot.id,dwell_seconds:120}))});
 const h=renderHook(useRecordNavigation,{initialProps:p});
 try {
  expect(h.result.current.canSkipDwell).toBe(false);
  await start(h);expect(h.result.current.canSkipDwell).toBe(true);
  await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
  expect(h.result.current.canSkipDwell).toBe(true);
  await act(async()=>jest.advanceTimersByTime(30000));
  let next,duplicate;act(()=>{next=h.result.current.skipDwell();duplicate=h.result.current.skipDwell();});
  await waitFor(()=>expect(p.sendGoal).toHaveBeenCalledTimes(2));
  await act(async()=>{expect(await next).toBe(true);expect(await duplicate).toBe(false);});
  expect(h.result.current.fromSpotId).toBe("a");expect(h.result.current.toSpotId).toBe("b");
  expect(h.result.current.canSkipDwell).toBe(true);
  await act(async()=>p.resolves[1]({status:"SUCCEEDED"}));
  await act(async()=>jest.advanceTimersByTime(90000));
  expect(p.sendGoal).toHaveBeenCalledTimes(2);
  await act(async()=>jest.advanceTimersByTime(30000));
  expect(p.sendGoal).toHaveBeenCalledTimes(3);expect(h.result.current.toSpotId).toBe("a");
  await act(async()=>h.result.current.stop());
 } finally {h.unmount();jest.useRealTimers();}
});
test("Next observation also leaves a paused dwell and wraps a closed route",async()=>{
 const p=initial({route_closed:true,observations:spots.map(spot=>({spot_id:spot.id,dwell_seconds:120}))});
 const h=renderHook(useRecordNavigation,{initialProps:p});
 await start(h);await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 await act(async()=>h.result.current.pause());expect(h.result.current.canSkipDwell).toBe(true);
 expect(await start(h,"skipDwell")).toBe(true);
 expect(h.result.current.patrolPaused).toBe(false);expect(h.result.current.toSpotId).toBe("b");
 await act(async()=>p.resolves[1]({status:"SUCCEEDED"}));
 expect(h.result.current.canSkipDwell).toBe(true);
 expect(await start(h,"skipDwell")).toBe(true);expect(h.result.current.toSpotId).toBe("a");
 await act(async()=>h.result.current.stop());
});
test.each([false,true])("no next observation exists for a single-point route (closed=%s)",async closed=>{
 const p=initial({route_closed:closed,observations:[{spot_id:"a",dwell_seconds:120}]});
 const h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 expect(h.result.current.canSkipDwell).toBe(false);
 await act(async()=>expect(await h.result.current.skipDwell()).toBe(false));
 expect(p.sendGoal).toHaveBeenCalledTimes(1);await act(async()=>h.result.current.stop());
});
test("Next observation is unavailable at a one-way final point or after Stop",async()=>{
 const p=initial({observations:spots.map(spot=>({spot_id:spot.id,dwell_seconds:120}))});
 const h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));await start(h,"skipDwell");
 await act(async()=>p.resolves[1]({status:"SUCCEEDED"}));
 expect(h.result.current.canSkipDwell).toBe(false);
 await act(async()=>expect(await h.result.current.skipDwell()).toBe(false));
 await act(async()=>h.result.current.stop());expect(h.result.current.canSkipDwell).toBe(false);
 expect(p.sendGoal).toHaveBeenCalledTimes(2);
});
test("Stop cancels a Next observation waiting for route refresh",async()=>{
 const p=initial({observations:spots.map(spot=>({spot_id:spot.id,dwell_seconds:120}))});
 const h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
 let release;p.manager.refresh.mockImplementationOnce(()=>new Promise(resolve=>{release=resolve;}));
 let pending;act(()=>{pending=h.result.current.skipDwell();});
 await waitFor(()=>expect(release).toBeDefined());
 await act(async()=>h.result.current.stop());
 await act(async()=>{release({spots,plan:p.manager.configuration.plan});expect(await pending).toBe(false);});
 expect(p.sendGoal).toHaveBeenCalledTimes(1);
});


test("Next and Previous replace an in-flight patrol goal using the destination index",async()=>{
 const p=initial(),h=renderHook(useRecordNavigation,{initialProps:p});
 await start(h);
 expect(h.result.current.canPreviousObservation).toBe(false);
 const firstSignal=p.sendGoal.mock.calls[0][3];
 expect(await start(h,"skipDwell")).toBe(true);
 expect(firstSignal.aborted).toBe(true);
 expect(h.result.current.toSpotId).toBe("b");
 expect(h.result.current.canSkipDwell).toBe(false);
 expect(h.result.current.canPreviousObservation).toBe(true);
 const secondSignal=p.sendGoal.mock.calls[1][3];
 expect(await start(h,"previousObservation")).toBe(true);
 expect(secondSignal.aborted).toBe(true);
 expect(h.result.current.toSpotId).toBe("a");
 await act(async()=>p.resolves[1]({status:"SUCCEEDED"}));
 expect(h.result.current.toSpotId).toBe("a");
 await act(async()=>h.result.current.stop());
});

test("Previous wraps a closed route from its first destination",async()=>{
 const p=initial({route_closed:true}),h=renderHook(useRecordNavigation,{initialProps:p});
 await start(h);expect(await start(h,"previousObservation")).toBe(true);
 expect(h.result.current.toSpotId).toBe("b");
 await act(async()=>h.result.current.stop());
});

test('replacement waits for cancellation and Stop invalidates the pending switch',async()=>{
 const p=initial(),h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 let release;p.cancelGoal.mockImplementationOnce(()=>new Promise(resolve=>{release=resolve;}));
 let switching;act(()=>{switching=h.result.current.skipDwell();});
 await waitFor(()=>expect(release).toBeDefined());
 expect(p.sendGoal).toHaveBeenCalledTimes(1);
 await act(async()=>h.result.current.stop());
 await act(async()=>{release({ok:true});expect(await switching).toBe(false);});
 expect(p.sendGoal).toHaveBeenCalledTimes(1);
});

test('failed cancellation never dispatches a replacement goal',async()=>{
 const p=initial(),h=renderHook(useRecordNavigation,{initialProps:p});await start(h);
 p.cancelGoal.mockResolvedValueOnce({ok:false});
 await act(async()=>expect(await h.result.current.skipDwell()).toBe(false));
 expect(p.sendGoal).toHaveBeenCalledTimes(1);expect(h.result.current.status).toBe('failed');
 await act(async()=>h.result.current.stop());
});


test('mission time does not consume the saved 600-second observation dwell',async()=>{
 jest.useFakeTimers();
 const p=initial({observations:spots.map(spot=>({spot_id:spot.id,dwell_seconds:600}))});
 const h=renderHook(useRecordNavigation,{initialProps:p});
 try {
  await start(h);await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
  await act(async()=>jest.advanceTimersByTime(120000));
  const snapshot=h.result.current.getPatrolContinuation();
  expect(snapshot.dwellRemainingMs).toBe(480000);
  await act(async()=>h.result.current.stop());
  await act(async()=>jest.advanceTimersByTime(900000));
  await start(h,'resumeAfterMission',snapshot);
  expect(h.result.current.dwelling).toBe(true);
  await act(async()=>jest.advanceTimersByTime(479000));
  expect(p.sendGoal).toHaveBeenCalledTimes(1);
  await act(async()=>jest.advanceTimersByTime(1000));
  expect(p.sendGoal).toHaveBeenCalledTimes(2);
  expect(h.result.current.toSpotId).toBe('b');
  await act(async()=>h.result.current.stop());
 } finally {h.unmount();jest.useRealTimers();}
});

test("decision departure holds indefinitely and Next manually advances without capture", async()=>{
 jest.useFakeTimers();
 const p=initial({route_closed:true,observations:spots.map(spot=>({spot_id:spot.id,departure_policy:"reasoning",dwell_seconds:2}))});
 const h=renderHook(useRecordNavigation,{initialProps:p});
 try {
  await start(h);await act(async()=>p.resolves[0]({status:"SUCCEEDED"}));
  await act(async()=>jest.advanceTimersByTime(600000));
  expect(h.result.current.dwelling).toBe(true);expect(p.sendGoal).toHaveBeenCalledTimes(1);
  await start(h,"skipDwell");expect(p.sendGoal).toHaveBeenCalledTimes(2);
  expect(h.result.current.toSpotId).toBe("b");
  await act(async()=>h.result.current.stop());
 } finally {h.unmount();jest.useRealTimers();}
});


test("waypoint navigation preserves the real refresh failure for its caller", async () => {
 const p=initial();p.manager.refresh.mockRejectedValueOnce(new Error("Observation service unavailable"));
 const h=renderHook(useRecordNavigation,{initialProps:p});
 await act(async()=>expect(h.result.current.goToSpot("b")).rejects.toThrow("Observation service unavailable"));
 expect(p.sendGoal).not.toHaveBeenCalled();
 expect(h.result.current.reason).toBe("Observation service unavailable");
});
