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

import { act, renderHook } from "@testing-library/react";
import useRecordMissionCycle, { returnSnapshotForEpisode } from "./useRecordMissionCycle";
const target={id:"table",label:"Dining observation",pose:{x:3,y:4,yaw:0}};
const robot={position:{x:3,y:4},orientation:{w:1}};
const idle=()=>({isRunning:false,status:"idle",phase:"none",currentIndex:-1,stop:jest.fn().mockResolvedValue([]),pause:jest.fn(),resume:jest.fn()});
function options(){return {enabled:true,mapName:"house",localized:true,currentPose:robot,
 mission:{mapName:"house",missionId:"clean"},missionRunner:idle(),
 manager:{returnSpotId:"table",resolveReturnSpot:jest.fn().mockResolvedValue(target)},
 startMission:jest.fn().mockResolvedValue(true),cancelMissionStart:jest.fn(),
 sendGoal:jest.fn(),cancelGoal:jest.fn().mockResolvedValue({ok:true}),onMessage:jest.fn()};}
function running(props){return {...props,missionRunner:{...props.missionRunner,isRunning:true,status:"navigating",phase:"awaiting-nav-result",currentIndex:0}};}
function done(props,status="done"){return {...props,missionRunner:{...props.missionRunner,isRunning:false,status,currentIndex:0}};}
async function startReturn(hook){let pending;act(()=>{pending=hook.result.current.startReturn();});await act(async()=>{jest.advanceTimersByTime(0);await pending;});}
beforeEach(()=>jest.useFakeTimers());afterEach(()=>jest.useRealTimers());

test("mission completes only after capture-gated return, stable arrival and final frame acknowledgement",async()=>{
 const props=options();let arrive;props.sendGoal.mockImplementation(()=>new Promise(resolve=>{arrive=resolve;}));
 const hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>{expect(await hook.result.current.start()).toBe(true);});
 expect(props.startMission).toHaveBeenCalledTimes(1);expect(props.sendGoal).not.toHaveBeenCalled();
 hook.rerender(running(props));
 hook.rerender({...done(props),manager:{...props.manager,returnSpotId:"other"}});
 expect(hook.result.current.returnPending).toBe(true);expect(hook.result.current.isRunning).toBe(true);
 expect(props.sendGoal).not.toHaveBeenCalled();
 await startReturn(hook);
 expect(props.sendGoal).toHaveBeenCalledWith(3,4,0,expect.any(AbortSignal));
 expect(hook.result.current.returning).toBe(true);expect(hook.result.current.returnSpotId).toBe("table");
 await act(async()=>arrive({status:"SUCCEEDED"}));
 expect(hook.result.current.isRunning).toBe(true);expect(hook.result.current.returnPhase).toBe("settling");
 expect(hook.result.current.returnArrived).toBe(false);
 await act(async()=>jest.advanceTimersByTime(750));
 expect(hook.result.current.returnArrived).toBe(true);expect(hook.result.current.isRunning).toBe(true);
 act(()=>hook.result.current.completeReturn());
 expect(hook.result.current.isRunning).toBe(false);expect(hook.result.current.status).toBe("done");
});

test.each(["failed","cancelled"])("%s mission never returns",async(status)=>{
 const props=options(),hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>hook.result.current.start());hook.rerender(running(props));hook.rerender(done(props,status));
 expect(hook.result.current.status).toBe(status);expect(hook.result.current.returnPending).toBe(false);
 await act(async()=>expect(await hook.result.current.startReturn()).toBe(false));
 expect(props.sendGoal).not.toHaveBeenCalled();
});

test("Stop during return handoff prevents a queued return from starting",async()=>{
 const props=options(),hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>hook.result.current.start());hook.rerender(running(props));hook.rerender(done(props));
 let pending;act(()=>{pending=hook.result.current.startReturn();});
 await act(async()=>hook.result.current.stop());
 await act(async()=>{jest.advanceTimersByTime(0);expect(await pending).toBe(false);});
 expect(props.sendGoal).not.toHaveBeenCalled();expect(hook.result.current.status).toBe("cancelled");
});

test("Stop during snapshot refresh cannot start a late mission",async()=>{
 const props=options();let resolved;props.manager.resolveReturnSpot.mockImplementation(()=>new Promise(resolve=>{resolved=resolve;}));
 const hook=renderHook(useRecordMissionCycle,{initialProps:props});
 let pending;act(()=>{pending=hook.result.current.start();});
 await act(async()=>hook.result.current.stop());
 await act(async()=>{resolved(target);expect(await pending).toBe(false);});
 expect(props.startMission).not.toHaveBeenCalled();expect(props.cancelMissionStart).toHaveBeenCalled();
});

test("pose settling times out rather than silently finishing away from its observation",async()=>{
 const props={...options(),currentPose:{...robot,position:{x:8,y:9}}};props.sendGoal.mockResolvedValue({status:"SUCCEEDED"});
 const hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>hook.result.current.start());hook.rerender(running(props));hook.rerender(done(props));
 await startReturn(hook);await act(async()=>jest.advanceTimersByTime(10000));
 expect(hook.result.current.status).toBe("failed");expect(hook.result.current.returnArrived).toBe(false);
 expect(hook.result.current.reason).toMatch(/pose did not settle/);
});

test("saved episode return pose changes are rejected before the mission starts",async()=>{
 const props=options(),hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>expect(await hook.result.current.start({...target,pose:{...target.pose,x:20}})).toBe(false));
 expect(props.startMission).not.toHaveBeenCalled();expect(props.sendGoal).not.toHaveBeenCalled();
 expect(hook.result.current.reason).toMatch(/moved after this episode/);
});

test("queued mission startup has a bounded wait",async()=>{
 const props=options(),hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>hook.result.current.start());
 await act(async()=>jest.advanceTimersByTime(10000));
 expect(hook.result.current.status).toBe("failed");expect(props.cancelMissionStart).toHaveBeenCalled();
});

test("resuming a return child never reruns the source mission",async()=>{
 const props=options();props.sendGoal.mockImplementation(()=>new Promise(()=>{}));
 const hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>expect(await hook.result.current.restoreReturn("table","clean")).toBe(true));
 expect(props.sendGoal).not.toHaveBeenCalled();expect(props.startMission).not.toHaveBeenCalled();
 await startReturn(hook);expect(props.sendGoal).toHaveBeenCalledTimes(1);
 expect(props.startMission).not.toHaveBeenCalled();
 await act(async()=>hook.result.current.stop());
});


test("return restores the observation arrival posture before final frame acknowledgement",async()=>{
 const props=options(),xml='<root><BehaviorTree ID="Main"><JointControl duration="1"/></BehaviorTree></root>';
 props.manager.resolveReturnSpot.mockResolvedValue({...target,arrival_bt:xml});
 props.sendGoal.mockResolvedValue({status:"SUCCEEDED"});
 const btStatusRef={current:"running"};
 props.returnRuntime={btStatusRef,getFlags:()=>({btNodeIsUp:true}),
  callService:jest.fn().mockResolvedValue({success:true}),stopBt:jest.fn().mockResolvedValue(true),
  config:{pollMs:10,btStartTimeoutMs:500,btTimeoutMs:1000}};
 const hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>hook.result.current.start());hook.rerender(running(props));hook.rerender(done(props));
 await startReturn(hook);
 expect(props.returnRuntime.callService).toHaveBeenCalledWith("/bt/load_and_run","interfaces/srv/LoadAndRunTree",{tree_xml:xml},30000);
 expect(hook.result.current.returnPhase).toBe("settling");
 expect(hook.result.current.returnArrived).toBe(false);
 await act(async()=>expect(hook.result.current.pause()).rejects.toThrow(/posture is being restored/));
 await act(async()=>{btStatusRef.current="completed";jest.advanceTimersByTime(10);});
 expect(hook.result.current.returnArrived).toBe(false);
 await act(async()=>jest.advanceTimersByTime(750));
 expect(hook.result.current.returnArrived).toBe(true);
 act(()=>hook.result.current.completeReturn());
 expect(hook.result.current.status).toBe("done");
});

test("resuming rejects a changed pinned arrival posture before return movement",async()=>{
 const props=options(),hook=renderHook(useRecordMissionCycle,{initialProps:props});
 await act(async()=>expect(hook.result.current.restoreReturn("table","clean",{...target,arrival_bt:"<old/>"})).rejects.toThrow(/arrival posture changed/));
 expect(props.sendGoal).not.toHaveBeenCalled();expect(props.startMission).not.toHaveBeenCalled();
 expect(hook.result.current.returnPending).toBe(false);
});

test("older episodes resolve arrival posture from their original plan snapshot",()=>{
 expect(returnSnapshotForEpisode({return_observation_spot:target,recording_plan:{observations:[{spot_id:"table",arrival_bt:"<saved/>"}]}}))
  .toEqual({...target,arrival_bt:"<saved/>"});
 expect(returnSnapshotForEpisode({return_observation_spot:{...target,arrival_bt:""},recording_plan:{observations:[{spot_id:"table",arrival_bt:"<saved/>"}]}}).arrival_bt).toBe("");
});
