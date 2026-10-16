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
import useRecordExecutionContext from "./useRecordExecutionContext";
import useCollectDraft from "./useCollectDraft";

const mission = { mapName: "Home", missionId: "clean-id", missionName: "CleanTable" };
const idle = { isRunning: false, status: "idle", phase: "none", currentIndex: -1 };
const navigating = { isRunning: true, status: "navigating", phase: "awaiting-nav-result", currentIndex: 0 };
let now;
beforeEach(() => { now = 10000; jest.spyOn(Date, "now").mockImplementation(() => now); });
afterEach(() => { jest.restoreAllMocks(); });

function mount(initial = {}) {
 return renderHook(props => useRecordExecutionContext(props), { initialProps: {
  enabled: true, mapName: "Home", mission, runner: idle, ...initial,
 }});
}

test("selection alone does not report an active mission; actual runner transitions have fresh provenance", () => {
 const hook = mount();
 expect(hook.result.current.getContext()).toEqual({reasoning_task:"mission_selection",recording_execution:null});
 now = 11000;
 hook.rerender({enabled:true,mapName:"Home",mission,runner:navigating});
 const running = hook.result.current.getContext().recording_execution;
 expect(running).toMatchObject({mission_id:"clean-id",status:"running",step:1,action:"awaiting-nav-result",
  context_valid_from_ms:11000,sampled_at_ms:11000});
 now = 11500;
 expect(hook.result.current.getContext().recording_execution).toMatchObject({context_version:running.context_version,
  context_valid_from_ms:11000,sampled_at_ms:11500});
 now = 12000;
 hook.rerender({enabled:true,mapName:"Home",mission,runner:{...navigating,paused:true}});
 const paused = hook.result.current.getContext().recording_execution;
 expect(paused).toMatchObject({status:"paused",context_valid_from_ms:12000});
 expect(paused.context_version).toBeGreaterThan(running.context_version);
 hook.rerender({enabled:true,mapName:"Home",mission,runner:{...navigating,status:"done"}});
 expect(hook.result.current.getContext().recording_execution).toBeNull();
});

test("locks identity to the actual started mission despite later selection or catalog refresh", () => {
 const hook=mount({runner:navigating});
 hook.rerender({enabled:true,mapName:"Home",mission:{...mission,missionId:"other-id"},runner:{...navigating,currentIndex:1,phase:"bt-running"}});
 expect(hook.result.current.getContext().recording_execution).toMatchObject({mission_id:"clean-id",step:2,action:"bt-running"});
 hook.rerender({enabled:true,mapName:"Other",mission:{mapName:"Other",missionId:"other-id"},runner:navigating});
 expect(hook.result.current.getContext().recording_execution.mission_id).toBeNull();
});

test("does not invent stable identity for an unmatched running mission", () => {
 const hook=mount({runner:navigating,mission:{mapName:"Home",missionName:"CleanTable"}});
 expect(hook.result.current.getContext().recording_execution.mission_id).toBeNull();
});

test("capture requests follow actual before/during/paused/after Record mission state", async () => {
 const ep={id:"episode",map_name:"Home",observations:[],annotations:[],ended_at:null};
 const frame={frame_id:"frame",capture:{received_at_ms:10000},context:{}};
 global.fetch=jest.fn(async(url,options)=>({ok:true,json:async()=>
  url.includes("reason-presets")?{presets:[]}:url.endsWith("/capture")?{episode:ep,saved_frame:frame}:
   url.endsWith("/episodes")&&options?.method!=="POST"?{episodes:[]}:ep}));
 const hook=renderHook(({runner})=>{
  const recording=useRecordExecutionContext({enabled:true,mapName:"Home",mission,runner});
  return useCollectDraft({enabled:true,mapName:"Home",getExecutionContext:recording.getContext});
 },{initialProps:{runner:idle}});
 await act(async()=>hook.result.current.start({}));
 const capture=async()=>{
  await act(async()=>hook.result.current.capture());
  const requests=fetch.mock.calls.filter(([url])=>url.endsWith("/capture"));
  return JSON.parse(requests.at(-1)[1].body).context;
 };
 expect((await capture()).reasoning_task).toBe("mission_selection");
 now=11000;hook.rerender({runner:navigating});
 expect((await capture()).recording_execution).toMatchObject({mission_id:"clean-id",status:"running",step:1});
 now=12000;hook.rerender({runner:{...navigating,paused:true}});
 expect((await capture()).recording_execution.status).toBe("paused");
 now=13000;hook.rerender({runner:{...navigating,status:"done"}});
 const after=await capture();
 expect(after.recording_execution).toBeNull();
 expect(after.reasoning_task).toBe("mission_selection");
 hook.unmount();
});

test("explicit observation capture waits for actual arrival and never borrows a running mission", () => {
 const initial={enabled:true,mapName:"Home",mission,runner:idle,captureType:"observation",observationSpotId:"table",localized:true,observationReady:false};
 const hook=mount(initial);
 expect(hook.result.current.ready).toBe(false);
 expect(hook.result.current.getContext().recording_execution).toBeNull();
 now=11000; hook.rerender({...initial,observationReady:true});
 expect(hook.result.current.ready).toBe(true);
 expect(hook.result.current.getContext()).toMatchObject({reasoning_task:"mission_selection",recording_execution:{reasoning_task:"mission_selection",observation_spot_id:"table",status:"arrived",localized:true}});
 hook.rerender({...initial,runner:navigating});
 expect(hook.result.current.getContext().recording_execution).toBeNull();
});

test("patrol excludes repositioning, records moving and paused legs, and ends without changing task", () => {
 const initial={enabled:true,mapName:"Home",mission,runner:idle,captureType:"patrol",localized:true,patrol:{...navigating,fromSpotId:null,toSpotId:"a"}};
 const hook=mount(initial);
 expect(hook.result.current.ready).toBe(false);
 const patrol={...navigating,currentIndex:1,fromSpotId:"a",toSpotId:"b"};
 hook.rerender({...initial,patrol});
 expect(hook.result.current.getContext()).toMatchObject({reasoning_task:"patrol_supervision",recording_execution:{status:"running",from_spot_id:"a",to_spot_id:"b",mission_id:null}});
 hook.rerender({...initial,patrol:{...patrol,paused:true}});
 expect(hook.result.current.getContext().recording_execution.status).toBe("paused");
 hook.rerender({...initial,patrol:{...patrol,status:"done",isRunning:false}});
 expect(hook.result.current.getContext()).toEqual({reasoning_task:"patrol_supervision",recording_execution:null});
});

test("strict mission cannot be captured while idle or without stable identity", () => {
 const initial={enabled:true,mapName:"Home",mission,runner:idle,captureType:"mission",localized:true};
 const hook=mount(initial);
 expect(hook.result.current.ready).toBe(false);
 expect(hook.result.current.getContext()).toEqual({reasoning_task:"mission_supervision",recording_execution:null});
 hook.rerender({...initial,runner:navigating});
 expect(hook.result.current.ready).toBe(true);
 expect(hook.result.current.getContext().recording_execution.mission_id).toBe("clean-id");
});

test("task engine startup is not a captured mission step",()=>{
 const initial={enabled:true,mapName:"Home",mission,runner:{isRunning:true,status:"starting",phase:"none",currentIndex:-1},captureType:"mission",localized:true};
 const hook=mount(initial);
 expect(hook.result.current.ready).toBe(false);
 expect(hook.result.current.getContext().recording_execution).toBeNull();
 hook.rerender({...initial,runner:navigating});
 expect(hook.result.current.ready).toBe(true);
});

test("mission return changes to Patrol and keeps the final settled frame eligible without an active mission",()=>{
 const initial={enabled:true,mapName:"Home",mission,runner:{...idle,status:"done"},captureType:"mission",localized:true,
   missionCycle:{returning:true,isRunning:true,toSpotId:"table",returnPhase:"returning",phase:"awaiting-nav-result"}};
 const hook=mount(initial);
 expect(hook.result.current.ready).toBe(true);
 expect(hook.result.current.getContext()).toMatchObject({reasoning_task:"patrol_supervision",recording_execution:{mission_id:null,from_spot_id:null,to_spot_id:"table",phase:"returning"}});
 hook.rerender({...initial,missionCycle:{...initial.missionCycle,returnPhase:"settling",phase:"settling",returnArrived:true}});
 expect(hook.result.current.ready).toBe(true);
 expect(hook.result.current.getContext().recording_execution.phase).toBe("settling");
 hook.rerender({...initial,missionCycle:{returning:false,isRunning:false}});
 expect(hook.result.current.ready).toBe(false);
 expect(hook.result.current.getContext().recording_execution).toBeNull();
});


test("independent execution uses actual type and first-approach/held context, not Capture type",()=>{
 const navigation={...navigating,fromSpotId:null,toSpotId:"table"};
 const execution={executionType:"patrol",navigation,missionRunner:idle,cycle:{},observationReady:false};
 const initial={enabled:true,mapName:"Home",mission,runner:idle,localized:true,captureType:"observation",execution};
 const h=mount(initial);
 expect(h.result.current.getContext()).toMatchObject({reasoning_task:"patrol_supervision",recording_execution:{phase:"approaching",from_spot_id:null,to_spot_id:"table"}});
 h.rerender({...initial,execution:{...execution,navigation:{...navigation,isRunning:false,status:"paused",paused:true,held:true}}});
 expect(h.result.current.getContext().recording_execution.status).toBe("paused");
 h.rerender({...initial,execution:{...execution,executionType:"observation",observationReady:true,observationSpotId:"table",navigation:idle}});
 expect(h.result.current.getContext()).toMatchObject({reasoning_task:"mission_selection",recording_execution:{phase:"observing",observation_spot_id:"table"}});
});
