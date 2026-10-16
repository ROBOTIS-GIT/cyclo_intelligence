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
import useRecordExecution from "./useRecordExecution";
const a={id:"a",label:"Dining",pose:{x:1,y:0,yaw:0}},b={id:"b",label:"Laundry",pose:{x:2,y:0,yaw:0}};
const pose={position:{x:1,y:0},orientation:{w:1}};
function props(){
 const plan={route_closed:false,observations:[a,b].map(spot=>({spot_id:spot.id,dwell_seconds:0,mission_ids:["clean"]}))};
 const goals=[];
 return {enabled:true,mapName:"house",localized:true,currentPose:pose,
  manager:{selectedId:"a",returnSpotId:"a",spots:[a,b],configuration:{plan,missions:[{mission_id:"clean",mission_name:"CleanTable"}]},select:jest.fn(),
   refresh:jest.fn().mockResolvedValue({spots:[a,b],plan}),resolveReturnSpot:jest.fn().mockResolvedValue(a)},
  mission:{mapName:"house",missionId:"clean"},missionRunner:{isRunning:false,status:"idle",phase:"none",currentIndex:-1,stop:jest.fn().mockResolvedValue([])},
  startMission:jest.fn().mockResolvedValue(true),cancelMissionStart:jest.fn(),cancelGoal:jest.fn().mockResolvedValue({ok:true}),onMessage:jest.fn(),goals,
  sendGoal:jest.fn((x,y,yaw,signal)=>new Promise(resolve=>{goals.push(resolve);signal.addEventListener("abort",()=>resolve({status:"CANCELED"}),{once:true});})),
 };
}
async function launch(h,fn="startPatrol",id){let pending;act(()=>{pending=h.result.current[fn](id);});
 await waitFor(()=>expect(h.result.current.navigation.phase).toBe("awaiting-nav-result"));
 await act(async()=>expect(await pending).toBe(true));}
test("Patrol and manual go-to execute without any episode and retarget only after cancelling",async()=>{
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});
 await launch(h);expect(p.sendGoal).toHaveBeenCalledWith(1,0,0,expect.any(AbortSignal));
 const signal=p.sendGoal.mock.calls[0][3];
 await launch(h,"goToSpot","b");expect(signal.aborted).toBe(true);
 expect(p.sendGoal).toHaveBeenLastCalledWith(2,0,0,expect.any(AbortSignal));
 const waypointSignal=p.sendGoal.mock.calls.at(-1)[3];
 await launch(h);expect(waypointSignal.aborted).toBe(true);
 expect(p.sendGoal).toHaveBeenLastCalledWith(1,0,0,expect.any(AbortSignal));
 expect(h.result.current.owner).toBe("patrol");
 expect(p.startMission).not.toHaveBeenCalled();expect(h.result.current.executionType).toBe("patrol");
 await act(async()=>h.result.current.stop());
 expect(h.result.current.navigation.held).toBe(true);
});
test("Stop invalidates deferred refresh and no goal starts afterward",async()=>{
 const p=props();let release;p.manager.refresh.mockImplementation(()=>new Promise(r=>{release=r;}));
 const h=renderHook(useRecordExecution,{initialProps:p});let pending;act(()=>{pending=h.result.current.startPatrol();});
 await waitFor(()=>expect(release).toBeDefined());await act(async()=>h.result.current.stop());
 await act(async()=>{release({spots:[a,b],plan:p.manager.configuration.plan});expect(await pending).toBe(false);});
 expect(p.sendGoal).not.toHaveBeenCalled();
});
test("child mission requires settled source and returns there automatically without recording",async()=>{
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});
 await act(async()=>expect(await h.result.current.startChildMission()).toBe(false));
 await waitFor(()=>expect(h.result.current.observationReady).toBe(true),{timeout:1500});
 await act(async()=>expect(await h.result.current.startChildMission()).toBe(true));
 expect(p.startMission).toHaveBeenCalledTimes(1);
 h.rerender({...p,missionRunner:{...p.missionRunner,isRunning:true,status:"navigating",phase:"awaiting-nav-result",currentIndex:0}});
 expect(h.result.current.executionType).toBe("mission");
 expect(h.result.current.navigationActive).toBe(true);
 await act(async()=>expect(h.result.current.goToSpot("b")).rejects.toThrow(/Stop the running mission/));
 h.rerender({...p,missionRunner:{...p.missionRunner,status:"done",phase:"done",currentIndex:0}});
 await waitFor(()=>expect(p.sendGoal).toHaveBeenCalledWith(1,0,0,expect.any(AbortSignal)));
 expect(h.result.current.executionType).toBe("patrol");
 expect(h.result.current.navigationActive).toBe(true);
 await act(async()=>p.goals[0]({status:"SUCCEEDED"}));
 await waitFor(()=>expect(h.result.current.cycle.status).toBe("done"),{timeout:1800});
 await waitFor(()=>expect(h.result.current.executionType).toBe("observation"),{timeout:1500});
 expect(h.result.current.navigationActive).toBe(false);
 expect(p.startMission).toHaveBeenCalledTimes(1);
});
test("a selected observation never authorizes a task from a different robot pose",async()=>{
 const p={...props(),currentPose:{...pose,position:{x:8,y:8}}},h=renderHook(useRecordExecution,{initialProps:p});
 await act(async()=>expect(await h.result.current.startChildMission()).toBe(false));
 expect(p.startMission).not.toHaveBeenCalled();expect(h.result.current.canRunMission).toBe(false);
});


test("retargeting refuses a new goal when navigation cancellation fails",async()=>{
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});await launch(h);
 p.cancelGoal.mockRejectedValueOnce(new Error("cancel unavailable"));
 await act(async()=>expect(h.result.current.goToSpot("b")).rejects.toThrow("cancel unavailable"));
 expect(p.sendGoal).toHaveBeenCalledTimes(1);
 expect(h.result.current.cancelFailed).toBe(true);expect(h.result.current.isRunning).toBe(true);
 await act(async()=>expect(await h.result.current.startPatrol()).toBe(false));
 await act(async()=>h.result.current.stop());
 expect(h.result.current.cancelFailed).toBe(false);
 await launch(h,"goToSpot","b");expect(p.sendGoal).toHaveBeenCalledTimes(2);
 await act(async()=>h.result.current.stop());
});


test("failed Stop remains actionable and blocks observation and commands until retry succeeds",async()=>{
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});await launch(h);
 p.cancelGoal.mockRejectedValueOnce(new Error("stop not confirmed"));
 await act(async()=>expect(h.result.current.stop()).rejects.toThrow("stop not confirmed"));
 expect(h.result.current.stopping).toBe(false);expect(h.result.current.needsStop).toBe(true);
 expect(h.result.current.isRunning).toBe(true);expect(h.result.current.observationReady).toBe(false);
 expect(h.result.current.executionType).toBeNull();
 await act(async()=>expect(h.result.current.goToSpot("b")).rejects.toThrow(/cancellation was not confirmed/));
 await act(async()=>expect(await h.result.current.startChildMission()).toBe(false));
 expect(p.sendGoal).toHaveBeenCalledTimes(1);expect(p.startMission).not.toHaveBeenCalled();
 await act(async()=>h.result.current.stop());
 expect(h.result.current.needsStop).toBe(false);expect(h.result.current.cancelFailed).toBe(false);
 expect(h.result.current.isRunning).toBe(false);
});

test("loading a map near an observation does not select it or own execution",async()=>{
 const p=props();p.manager={...p.manager,selectedId:"",returnSpotId:""};
 const h=renderHook(useRecordExecution,{initialProps:p});
 await waitFor(()=>expect(h.result.current.observationReady).toBe(true),{timeout:1500});
 expect(p.manager.select).not.toHaveBeenCalled();expect(p.sendGoal).not.toHaveBeenCalled();
 expect(h.result.current.isRunning).toBe(false);expect(h.result.current.owner).toBeNull();
});
test("a new map load clears stopped navigation markers but cannot reset an active run",async()=>{
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});await launch(h,"goToSpot","b");
 act(()=>expect(h.result.current.resetIdle()).toBe(false));
 expect(h.result.current.navigation.toSpotId).toBe("b");
 await act(async()=>h.result.current.stop());
 act(()=>expect(h.result.current.resetIdle()).toBe(true));
 expect(h.result.current.navigation.toSpotId).toBeNull();expect(h.result.current.navigation.held).toBe(false);
 expect(h.result.current.cycle.pinnedSpot).toBeNull();expect(h.result.current.owner).toBeNull();
 expect(h.result.current.isRunning).toBe(false);
});

test("an open map edit session cannot launch navigation or a child mission",async()=>{
 const p=props();p.manager={...p.manager,route:{routeMode:true}};
 const h=renderHook(useRecordExecution,{initialProps:p});
 await waitFor(()=>expect(h.result.current.observationReady).toBe(true),{timeout:1500});
 await act(async()=>{
  expect(await h.result.current.startPatrol()).toBe(false);
  await expect(h.result.current.goToSpot("b")).rejects.toThrow(/Finish editing/);
  expect(await h.result.current.startChildMission()).toBe(false);
 });
 expect(h.result.current.canRunMission).toBe(false);
 expect(p.sendGoal).not.toHaveBeenCalled();expect(p.startMission).not.toHaveBeenCalled();
});


test("navigation overlay state ends on pause and Stop, and resumes with navigation",async()=>{
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});
 expect(h.result.current.navigationActive).toBe(false);
 await launch(h);expect(h.result.current.navigationActive).toBe(true);
 await act(async()=>h.result.current.pausePatrol());
 expect(h.result.current.navigationActive).toBe(false);
 await launch(h,"resumePatrol");
 expect(h.result.current.navigationActive).toBe(true);
 await act(async()=>h.result.current.stop());
 expect(h.result.current.navigationActive).toBe(false);
 expect(h.result.current.navigation.held).toBe(true);
});


test("child missions use the reached observation regardless of map selection",async()=>{
 const p=props();
 p.manager.configuration.plan.observations[1].mission_ids=["wash"];
 p.manager.configuration.missions.push({mission_id:"wash",mission_name:"DoLaundry"});
 p.manager.selectedId="b";p.manager.returnSpotId="b";
 const h=renderHook(useRecordExecution,{initialProps:p});
 await waitFor(()=>expect(h.result.current.observationReady).toBe(true),{timeout:1500});
 expect(h.result.current.observationSpotId).toBe("a");
 expect(h.result.current.linkedMissions.map(m=>m.mission_id)).toEqual(["clean"]);
 expect(h.result.current.canRunMission).toBe(true);
 h.rerender({...p,manager:{...p.manager,selectedId:"",returnSpotId:""}});
 expect(h.result.current.canRunMission).toBe(true);
 h.rerender({...p,mission:{...p.mission,missionId:"wash"}});
 expect(h.result.current.canRunMission).toBe(false);
 await act(async()=>expect(await h.result.current.startChildMission()).toBe(false));
 expect(p.startMission).not.toHaveBeenCalled();
 h.rerender(p);
 await act(async()=>expect(await h.result.current.startChildMission()).toBe(true));
 expect(p.manager.resolveReturnSpot).toHaveBeenCalledWith("a","clean");
 expect(h.result.current.cycle.pinnedSpot.id).toBe("a");
});


test("normalized Record state reports browser ownership without inheriting capture flags", async () => {
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});
 expect(h.result.current.executionState).toMatchObject({source:"record",authority:"browser",lifecycle:"idle",owns_execution:false});
 await waitFor(()=>expect(h.result.current.observationReady).toBe(true),{timeout:1500});
 expect(h.result.current.executionState).toMatchObject({activity:"observing",observation_spot_id:"a",owns_execution:false});
 const observed=h.result.current.executionState;
 h.rerender({...p,capture:{active:true,paused:true,detached:true},episode:{phase:"mission",mission_id:"old"}});
 expect(h.result.current.executionState).toEqual(observed);
 await launch(h,"goToSpot","b");
 expect(h.result.current.executionState).toMatchObject({lifecycle:"active",activity:"patrol_navigation",control:"running",owns_execution:true,target_spot_id:"b"});
 await act(async()=>h.result.current.stop());
 expect(h.result.current.executionState).toMatchObject({lifecycle:"stopped",control:"idle",owns_execution:false});
});

test("normalized pause is acknowledged only after cancellation settles", async () => {
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});await launch(h);
 let acknowledge;p.cancelGoal.mockImplementationOnce(()=>new Promise(resolve=>{acknowledge=resolve;}));
 let pause;act(()=>{pause=h.result.current.pausePatrol();});
 await waitFor(()=>expect(acknowledge).toBeDefined());
 expect(h.result.current.executionState).toMatchObject({lifecycle:"active",control:"pausing",owns_execution:true,activity:"patrol_navigation"});
 await act(async()=>{acknowledge({ok:true});await pause;});
 expect(h.result.current.executionState).toMatchObject({lifecycle:"active",control:"paused",owns_execution:true});
 await act(async()=>h.result.current.stop());
});

test("normalized Stop failure retains target and ownership until a successful retry", async () => {
 const p=props(),h=renderHook(useRecordExecution,{initialProps:p});await launch(h,"goToSpot","b");
 p.cancelGoal.mockRejectedValueOnce(new Error("stop not confirmed"));
 await act(async()=>expect(h.result.current.stop()).rejects.toThrow("stop not confirmed"));
 expect(h.result.current.executionState).toMatchObject({lifecycle:"failed",control:"cancelling",owns_execution:true,
  activity:"patrol_navigation",target_spot_id:"b",error:"stop not confirmed"});
 await act(async()=>h.result.current.stop());
 expect(h.result.current.executionState).toMatchObject({lifecycle:"stopped",control:"idle",owns_execution:false,error:null});
});
