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
import useCollectRecording from "./useCollectRecording";
const makeDraft=()=>({active:true,episode:{id:"ep",episode_type:"patrol",patrol_spot_ids:["a","b"],observations:[{frame_id:"f"}]},
  recording:false,stopRecording:jest.fn(),startRecording:jest.fn(),finish:jest.fn().mockResolvedValue(true),start:jest.fn(),capture:jest.fn(),load:jest.fn()});
const patrol=()=>({type:"patrol",ready:true,planRevision:2,patrolSpotIds:["a","b"],start:jest.fn(),stop:jest.fn()});
test("capture and save never start or stop navigation",async()=>{
 const draft=makeDraft(),session=patrol();
 const {result}=renderHook(()=>useCollectRecording({draft,session}));
 await act(async()=>result.current.startCapture());
 expect(draft.startRecording).toHaveBeenCalledWith(1000,expect.any(Function),expect.any(Function));
 await act(async()=>result.current.saveEpisode());
 expect(draft.finish).toHaveBeenCalledWith({allowEmpty:true});
 expect(session.start).not.toHaveBeenCalled();expect(session.stop).not.toHaveBeenCalled();
});
test("route completion alone neither closes data nor creates another episode",async()=>{
 const draft=makeDraft(),session=patrol();
 const {rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 rerender({draft,session:{...session,type:"observation",observationSpotId:"b",isRunning:false,status:"done"}});
 expect(draft.finish).not.toHaveBeenCalled();expect(draft.start).not.toHaveBeenCalled();
});
test("continuous pre-frame boundary saves a new observation segment before its image",async()=>{
 const draft=makeDraft(),session=patrol();let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{before=callback;});
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 await act(async()=>result.current.startCapture());
 const child={id:"obs",episode_type:"observation",observation_spot_id:"b"};
 draft.startSegment=jest.fn(async()=>child);
 rerender({draft,session:{...session,type:"observation",observationSpotId:"b"}});
 let ready;await act(async()=>{ready=await before();});
 expect(ready).toBe(true);expect(draft.startSegment).toHaveBeenCalledWith(expect.objectContaining({episode_type:"observation",observation_spot_id:"b"}));
 expect(session.start).not.toHaveBeenCalled();
});
test("mission return uses the pinned return endpoint without issuing a movement command",async()=>{
 const draft={...makeDraft(),episode:{id:"task",episode_type:"mission",recording_mission_id:"clean",return_observation_spot_id:"a"},startReturnEpisode:jest.fn(async()=>({id:"return",episode_type:"patrol",capture_segment:"mission_return"}))};
 const session={...patrol(),returning:true,patrolSpotIds:["a"],missionId:"clean",returnSpotId:"a"};let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{before=callback;});
 const {result}=renderHook(()=>useCollectRecording({draft,session}));
 await act(async()=>result.current.startCapture());
 await act(async()=>expect(await before()).toBe(true));
 expect(draft.startReturnEpisode).toHaveBeenCalledWith({keepRecording:true});
 expect(session.start).not.toHaveBeenCalled();
});
test("a return captured from the middle stays in its ordinary Patrol episode",async()=>{
 const draft={...makeDraft(),episode:{id:"p",episode_type:"patrol",patrol_spot_ids:["a"]},startSegment:jest.fn()};
 const session={...patrol(),returning:true,patrolSpotIds:["a"],missionId:"clean",returnSpotId:"a"};let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{before=callback;});
 const {result}=renderHook(()=>useCollectRecording({draft,session}));
 await act(async()=>result.current.startCapture());
 await act(async()=>expect(await before()).toBe(true));expect(draft.startSegment).not.toHaveBeenCalled();
});
test("failed segment storage prevents the next frame but leaves motion running",async()=>{
 const draft={...makeDraft(),startSegment:jest.fn(async()=>null)},session={...patrol(),type:"observation",observationSpotId:"b"};let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{before=callback;});
 const {result}=renderHook(()=>useCollectRecording({draft,session}));
 await act(async()=>result.current.startCapture());
 await act(async()=>expect(await before()).toBe(false));
 expect(result.current.recordError).toMatch(/Earlier frames/);expect(session.stop).not.toHaveBeenCalled();
});
test("loading an unfinished episode never restores movement",async()=>{
 const draft=makeDraft(),session=patrol();
 const {result}=renderHook(()=>useCollectRecording({draft,session}));
 await act(async()=>result.current.resumeEpisode("old"));
 expect(draft.load).toHaveBeenCalledWith("old");expect(session.start).not.toHaveBeenCalled();
});

const observationWithTargets = () => ({
 mapName:"house",type:"observation",ready:true,observationSpotId:"table",planRevision:3,
 captureTargets:{
  mission:{type:"mission",ready:true,missionId:"clean",missionName:"CleanTable",observationSpotId:"table",returnSpotId:"table",planRevision:3},
  patrol:{type:"patrol",ready:true,patrolSpotIds:["table","laundry"],planRevision:3},
 }
});
function pendingDraft() {
 const draft={...makeDraft(),active:false,episode:null,startSegment:jest.fn()};
 draft.stopRecording=jest.fn(()=>{draft.recording=false;});
 draft.start=jest.fn(async fields=>{draft.episode={id:"new",...fields,observations:[]};draft.active=true;return true;});
 return draft;
}
test("new capture starts with the actual observation, even when a child mission is selected",async()=>{
 const draft=pendingDraft(),session=observationWithTargets();let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{draft.recording=true;before=callback;});
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 expect(result.current.targetReady).toBe(true);
 await act(async()=>result.current.startEpisode());rerender({draft,session});
 expect(draft.start).toHaveBeenCalledWith(expect.objectContaining({episode_type:"observation",recording_mission_id:null,observation_spot_id:"table"}));
 await act(async()=>result.current.startCapture());rerender({draft,session});
 await act(async()=>expect(await before()).toBe(true));
 expect(draft.startSegment).not.toHaveBeenCalled();
 expect(result.current.captureType).toBe("observation");
});
test("capture can be armed before execution and saves nothing until an actual state is ready",async()=>{
 const draft=pendingDraft(),session={...observationWithTargets(),type:null,ready:false};let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{draft.recording=true;before=callback;});
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 await act(async()=>result.current.startEpisode());rerender({draft,session});
 expect(draft.episode.episode_type).toBe("patrol");
 await act(async()=>result.current.startCapture());rerender({draft,session});
 await act(async()=>expect(await before()).toBeNull());
 expect(result.current.awaitingTarget).toBe(true);expect(result.current.captureType).toBeNull();
 expect(draft.startSegment).not.toHaveBeenCalled();
 rerender({draft,session:{...session,type:"patrol",ready:true,patrolSpotIds:["table","laundry"]}});
 await act(async()=>expect(await before()).toBe(true));
 expect(result.current.awaitingTarget).toBe(false);
});
test("one continuous recording follows patrol, observation, mission, return and resumed patrol",async()=>{
 const session=patrol(),draft={...makeDraft(),recording:true};let before,sequence=0;
 draft.startRecording.mockImplementation((_,__,callback)=>{before=callback;});
 draft.startSegment=jest.fn(async fields=>{draft.episode={id:`segment-${++sequence}`,...fields};return draft.episode;});
 draft.startReturnEpisode=jest.fn(async()=>{
  draft.episode={id:"return",episode_type:"patrol",capture_segment:"mission_return",patrol_spot_ids:["a"]};
  return draft.episode;
 });
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 await act(async()=>result.current.startCapture());
 const states=[
  session,
  {...session,type:"observation",observationSpotId:"a"},
  {...session,type:"mission",observationSpotId:"a",missionId:"clean",missionName:"CleanTable",returnSpotId:"a"},
  {...session,returning:true,missionId:"clean",returnSpotId:"a",patrolSpotIds:["a"]},
  {...session,type:"observation",observationSpotId:"a",returning:false},
  session,
 ];
 for(const state of states){
  rerender({draft,session:state});
  await act(async()=>expect(await before()).toBe(true));
  expect(result.current.captureType).toBe(state.type);
 }
 expect(draft.startSegment.mock.calls.map(([fields])=>fields.episode_type)).toEqual(["observation","mission","observation","patrol"]);
 expect(draft.startReturnEpisode).toHaveBeenCalledTimes(1);
 expect(draft.startRecording).toHaveBeenCalledTimes(1);
 expect(draft.stopRecording).not.toHaveBeenCalled();expect(draft.finish).not.toHaveBeenCalled();
 expect(session.start).not.toHaveBeenCalled();expect(session.stop).not.toHaveBeenCalled();
});
test("single frame Capture waits for verified execution and saves exactly once",async()=>{
 const draft=pendingDraft(),session={...observationWithTargets(),type:null,ready:false};
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 act(()=>result.current.setCaptureMode("single"));
 await act(async()=>result.current.startEpisode());rerender({draft,session});
 await act(async()=>result.current.captureSingle());
 expect(result.current.singlePending).toBe(true);expect(draft.capture).not.toHaveBeenCalled();
 const running={...session,type:"patrol",ready:true,patrolSpotIds:["table","laundry"]};
 rerender({draft,session:running});
 await waitFor(()=>expect(draft.capture).toHaveBeenCalledTimes(1));
 expect(result.current.singlePending).toBe(false);
 rerender({draft,session:{...running}});
 expect(draft.capture).toHaveBeenCalledTimes(1);
});
test("Stop capture cancels a queued single frame without stopping execution",async()=>{
 const draft=pendingDraft(),session={...observationWithTargets(),type:null,ready:false,stop:jest.fn()};
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 await act(async()=>result.current.startEpisode());rerender({draft,session});
 await act(async()=>result.current.captureSingle());
 act(()=>result.current.stopCapture());
 rerender({draft,session:{...session,type:"patrol",ready:true,patrolSpotIds:["table","laundry"]}});
 expect(result.current.singlePending).toBe(false);expect(draft.capture).not.toHaveBeenCalled();
 expect(session.stop).not.toHaveBeenCalled();
});
test("selecting another mission does not change capture until that mission actually runs",async()=>{
 const draft=pendingDraft(),session=observationWithTargets();let before;
 draft.startRecording.mockImplementation((_,__,callback)=>{draft.recording=true;before=callback;});
 draft.startSegment=jest.fn(async fields=>{draft.episode={id:"mission",...fields};return draft.episode;});
 const {result,rerender}=renderHook(useCollectRecording,{initialProps:{draft,session}});
 await act(async()=>result.current.startEpisode());rerender({draft,session});
 await act(async()=>result.current.startCapture());
 const selected={...session,captureTargets:{...session.captureTargets,mission:{...session.captureTargets.mission,missionId:"serve",missionName:"ServeDish"}}};
 rerender({draft,session:selected});
 await act(async()=>expect(await before()).toBe(true));
 expect(draft.startSegment).not.toHaveBeenCalled();
 rerender({draft,session:{...selected,type:"mission",missionId:"serve",missionName:"ServeDish",returnSpotId:"table"}});
 await act(async()=>expect(await before()).toBe(true));
 expect(draft.startSegment).toHaveBeenCalledWith(expect.objectContaining({episode_type:"mission",recording_mission_id:"serve"}));
});
