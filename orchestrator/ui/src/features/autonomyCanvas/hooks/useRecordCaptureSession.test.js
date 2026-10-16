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

import { renderHook } from "@testing-library/react";
import useRecordCaptureSession from "./useRecordCaptureSession";
import { episodeFieldsForPreparation } from "../utils/recordCapture";
function props(){return {mapName:"house",selectedMissionName:"CleanTable",
 execution:{cycle:{},executionType:"observation",observationSpotId:"table"},context:{ready:true},
 manager:{ready:true,selectedId:"table",planRevision:3,linkedMissions:[{mission_id:"clean",mission_name:"CleanTable"}],
  configuration:{plan:{observations:[{spot_id:"table",mission_ids:["clean"]},{spot_id:"laundry",mission_ids:["wash"]}]},missions:[{mission_id:"clean",mission_name:"CleanTable"},{mission_id:"wash",mission_name:"DoLaundry"}]}},
 mission:{missionId:"stale",missionName:"OldMission"}};}
test("preparation uses the selected linked mission instead of a previously loaded snapshot",()=>{
 const {result}=renderHook(useRecordCaptureSession,{initialProps:props()});
 expect(result.current.type).toBe("observation");
 expect(episodeFieldsForPreparation(result.current,"mission")).toMatchObject({episode_type:"mission",recording_mission_id:"clean",return_observation_spot_id:"table"});
 expect(episodeFieldsForPreparation(result.current,"patrol")).toMatchObject({episode_type:"patrol",patrol_spot_ids:["table","laundry"]});
});
test("unsaved routes and unlinked missions cannot be prepared",()=>{
 const base=props(),{result,rerender}=renderHook(useRecordCaptureSession,{initialProps:base});
 rerender({...base,manager:{...base.manager,planDirty:true}});
 expect(episodeFieldsForPreparation(result.current,"patrol")).toBeNull();
 rerender({...base,manager:{...base.manager,route:{routeMode:true}}});
 for (const type of ["patrol","observation","mission"]) expect(episodeFieldsForPreparation(result.current,type)).toBeNull();
 rerender({...base,selectedMissionName:"OtherMission"});
 expect(episodeFieldsForPreparation(result.current,"mission")).toBeNull();
});


test("failed execution exposes its cause only while capture is awaiting context",()=>{
 const base=props();
 const execution={...base.execution,executionType:null,cycle:{cyclePhase:"failed",reason:"Waypoint Task failed at LaundryBasket"}};
 const h=renderHook(useRecordCaptureSession,{initialProps:{...base,execution,context:{ready:false}}});
 expect(h.result.current.waitingReason).toBe("Waypoint Task failed at LaundryBasket");
 h.rerender({...base,execution,context:{ready:true}});
 expect(h.result.current.waitingReason).toBe("");
});


test("map selection cannot change the mission capture source",()=>{
 const p=props();p.manager.selectedId="laundry";
 p.manager.linkedMissions=[{mission_id:"wash",mission_name:"DoLaundry"}];
 const h=renderHook(useRecordCaptureSession,{initialProps:p});
 expect(episodeFieldsForPreparation(h.result.current,"mission")).toMatchObject({recording_mission_id:"clean",return_observation_spot_id:"table"});
 h.rerender({...p,selectedMissionName:"DoLaundry"});
 expect(episodeFieldsForPreparation(h.result.current,"mission")).toBeNull();
 h.rerender({...p,execution:{...p.execution,observationSpotId:null}});
 expect(episodeFieldsForPreparation(h.result.current,"mission")).toBeNull();
 expect(episodeFieldsForPreparation(h.result.current,"patrol")).not.toBeNull();
});
