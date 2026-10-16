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
import useRecordObservationSpots, { observationSpots, missionsAtObservation } from "./useRecordObservationSpots";
import { autonomyRequest } from "./autonomyApi";
jest.mock("./autonomyApi", () => ({ autonomyRequest: jest.fn(), OBSERVATION_PLAN_CHANGED: "observation-spots-changed" }));
const original = {id:"legacy",label:"Table view",pose:{x:1,y:2,yaw:0},metadata:{}};
let data;
beforeEach(() => {
  jest.resetAllMocks();
  data = {plan:{revision:1,observations:[{spot_id:"legacy"}]},spots:[original,{id:"station",label:"Sink"}]};
  autonomyRequest.mockImplementation(async (path, method = "GET", body) => {
    if (method === "GET") return data;
    if (method === "PUT") { data = { ...data, plan: { ...body, revision: body.revision + 1 } }; return data; }
    const id = method === "POST" ? "new" : decodeURIComponent(path.split("/").at(-1).split("?")[0]);
    if (method === "DELETE") {
      const observations = data.plan.observations.filter(point => point.spot_id !== id);
      data = { ...data, spots: data.spots.filter(spot => spot.id !== id), plan: { ...data.plan, observations,
        revision: data.plan.revision + (observations.length !== data.plan.observations.length ? 1 : 0) } };
      return data;
    }
    const spot = {id,label:body.label,pose:body.pose,metadata:{role:"observation"}};
    data = {...data,spots:method === "POST" ? [...data.spots,spot] : data.spots.map(value=>value.id===id ? spot : value),plan:{...data.plan,revision:data.plan.revision+1}};
    return spot;
  });
});
const props = () => ({enabled:true,mapName:"Home",localized:true,blocked:false,
  currentPose:{position:{x:3,y:4},orientation:{w:1}},onMapMode:jest.fn()});
test("shows role-tagged or legacy plan spots without task stations", () => {
  expect(observationSpots({...data,spots:[...data.spots,{id:"new",metadata:{role:"observation"}}]}).map(s=>s.id)).toEqual(["legacy","new"]);
});
test("creates a named observation at the robot and notifies Operate", async () => {
  const listener = jest.fn(); window.addEventListener("observation-spots-changed",listener);
  const {result} = renderHook(useRecordObservationSpots,{initialProps:props()});
  await waitFor(()=>expect(result.current.ready).toBe(true));
  act(()=>result.current.begin());
  act(()=>result.current.atRobot());
  act(()=>result.current.setDraft({...result.current.draft,label:"Laundry view"}));
  await act(async()=>{expect(await result.current.save()).toBe(true);});
  expect(autonomyRequest).toHaveBeenCalledWith("/autonomy/maps/Home/spots","POST",{
    label:"Laundry view",pose:{frame_id:"map",x:3,y:4,yaw:0},plan_revision:1});
  expect(result.current.selectedId).toBe("new");
  expect(listener).toHaveBeenCalledTimes(1);
  expect(result.current.draft).toBeNull();
  window.removeEventListener("observation-spots-changed",listener);
});
test("edits a legacy spot in place and deletes with the current revision", async()=>{
  const {result} = renderHook(useRecordObservationSpots,{initialProps:props()});
  await waitFor(()=>expect(result.current.ready).toBe(true));
  act(()=>result.current.select("legacy")); act(()=>result.current.begin(true));
  act(()=>result.current.place({x:9,y:8,yaw:1}));
  await act(async()=>{await result.current.save();});
  expect(autonomyRequest).toHaveBeenCalledWith("/autonomy/maps/Home/spots/legacy","PATCH",expect.objectContaining({pose:{frame_id:"map",x:9,y:8,yaw:1}}));
  await act(async()=>{await result.current.remove("legacy");});
  expect(autonomyRequest).toHaveBeenCalledWith("/autonomy/maps/Home/spots/legacy?plan_revision=2","DELETE");
  expect(result.current.spots).toEqual([]);
});
test("does not apply an old map response after switching maps", async()=>{
  let complete;
  autonomyRequest.mockImplementation(path=>path.endsWith("Home") ? new Promise(resolve=>{complete=resolve;}) : Promise.resolve({plan:{revision:0,observations:[]},spots:[]}));
  const options=props(); const {result,rerender}=renderHook(useRecordObservationSpots,{initialProps:options});
  rerender({...options,mapName:"house"});
  await waitFor(()=>expect(result.current.ready).toBe(true));
  await act(async()=>complete(data));
  expect(result.current.spots).toEqual([]);
  expect(result.current.error).toBeNull();
});
test("blocked editing never sends a mutation", async()=>{
  const {result,rerender}=renderHook(useRecordObservationSpots,{initialProps:props()});
  await waitFor(()=>expect(result.current.ready).toBe(true));
  act(()=>result.current.begin()); act(()=>result.current.atRobot());
  rerender({...props(),blocked:true});
  await act(async()=>{expect(await result.current.save()).toBe(false);});
  expect(autonomyRequest.mock.calls.every(call=>!call[1])).toBe(true);
});


test("the source observation determines eligible missions and the return target",async()=>{
 data={plan:{revision:1,observations:[{spot_id:"legacy",mission_ids:["clean"]},{spot_id:"other",mission_ids:["wash"]}]},
  missions:[{mission_id:"clean",mission_name:"CleanTable"},{mission_id:"wash",mission_name:"Laundry"}],
  spots:[original,{...original,id:"other",metadata:{role:"observation"}}]};
 const options={...props(),missionId:"clean"};const view=renderHook(useRecordObservationSpots,{initialProps:options});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 expect(view.result.current.returnSpotId).toBe("");
 act(()=>view.result.current.select("legacy"));
 expect(view.result.current.linkedMissions.map(m=>m.mission_id)).toEqual(["clean"]);
 expect(view.result.current.returnSpotId).toBe("legacy");
 act(()=>view.result.current.select("other"));
 expect(view.result.current.returnSpotId).toBe("");
 view.rerender({...options,missionId:"wash"});
 expect(view.result.current.returnSpotId).toBe("other");
 await act(async()=>expect(view.result.current.resolveReturnSpot("other","clean")).rejects.toThrow(/not linked/));
});
test("excluded and deleted missions are never offered by an observation",()=>{
 const config={plan:{excluded_mission_ids:["excluded"],observations:[{spot_id:"a",mission_ids:["ok","excluded","deleted"]}]},
  missions:[{mission_id:"ok"},{mission_id:"excluded"}]};
 expect(missionsAtObservation(config,"a")).toEqual([{mission_id:"ok"}]);
 expect(missionsAtObservation(config,"missing")).toEqual([]);
});
test("patrol defaults to the saved plan order and can restore it after a capture override",async()=>{
 const view=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 expect(view.result.current.patrolSpotIds).toEqual(["legacy"]);
 act(()=>view.result.current.setPatrolSpotIds(["legacy","legacy"]));
 act(()=>view.result.current.useSavedPatrol());
 expect(view.result.current.patrolSpotIds).toEqual(["legacy"]);
});

function routeFixture() {
 data = { plan: { revision: 8, mode: "patrol", route_closed: false, excluded_mission_ids: [], observations: [
   { spot_id: "a", mission_ids: ["clean"], dwell_seconds: 25, arrival_bt: "table.xml" },
   { spot_id: "b", mission_ids: ["wash"], dwell_seconds: 12, arrival_bt: "laundry.xml" },
 ] }, missions: [{ mission_id: "clean", mission_name: "CleanTable" }, { mission_id: "wash", mission_name: "DoLaundry" }],
 spots: ["a", "b", "c"].map((id, i) => ({ id, label: id.toUpperCase(), pose: {x:i,y:0,yaw:0}, metadata: {role:"observation"} })) };
}
test("map clicks only select; explicit inclusion and loop controls save the shared plan", async()=>{
 routeFixture(); const view=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 act(()=>view.result.current.route.toggleMode());
 act(()=>expect(view.result.current.route.handleSpotClick("c")).toBe(true));
 expect(view.result.current.selectedId).toBe("c");
 expect(view.result.current.route.routeView.treeSpots.map(s=>s.id)).toEqual(["a","b"]);
 expect(view.result.current.planDirty).toBe(false);
 act(()=>view.result.current.route.setIncluded("c",true));
 expect(view.result.current.route.routeView.treeSpots.map(s=>s.id)).toEqual(["a","b","c"]);
 expect(view.result.current.patrolSpotIds).toEqual(["a","b"]); // Execution never consumes an unsaved route.
 act(()=>expect(view.result.current.route.handleSpotClick("a")).toBe(true));
 expect(view.result.current.route.routeView.closed).toBe(false);
 act(()=>view.result.current.route.closeLoop());
 expect(view.result.current.route.routeView.closed).toBe(true);
 act(()=>view.result.current.routeEditor.updatePoint("c",{mission_ids:["wash"]}));
 await act(async()=>expect(await view.result.current.routeEditor.save()).toBe(true));
 const put=autonomyRequest.mock.calls.find(call=>call[1]==="PUT");
 expect(put[2]).toMatchObject({revision:8,route_closed:true,observations:[
   {spot_id:"a",arrival_bt:"table.xml",dwell_seconds:25,mission_ids:["clean"]},
   {spot_id:"b",arrival_bt:"laundry.xml",dwell_seconds:12,mission_ids:["wash"]},
   {spot_id:"c",mission_ids:["wash"]},
 ]});
 expect(view.result.current.planDirty).toBe(false);
 expect(view.result.current.patrolSpotIds).toEqual(["a","b","c"]);
});
test("a one-way route can shrink to a single observation and be rebuilt from an empty route",async()=>{
 routeFixture();const view=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 act(()=>view.result.current.route.toggleMode());
 act(()=>view.result.current.route.removeSpot("b"));
 expect(view.result.current.route.routeView.treeSpots.map(s=>s.id)).toEqual(["a"]);
 act(()=>view.result.current.route.clearRoute());
 expect(view.result.current.route.routeView.treeSpots).toEqual([]);
 act(()=>view.result.current.route.setIncluded("b",true));
 expect(view.result.current.route.routeView.treeSpots.map(s=>s.id)).toEqual(["b"]);
 act(()=>view.result.current.route.setIncluded("a",true));
 expect(view.result.current.route.routeView.treeSpots.map(s=>s.id)).toEqual(["b","a"]);
 expect(view.result.current.route.routeView.closed).toBe(false);
 expect(view.result.current.routeEditor.plan.observations[1].arrival_bt).toBe("table.xml");
});
test("clicking the first waypoint never implicitly closes an open loop",async()=>{
 routeFixture();data.plan.route_closed=true;
 const view=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 act(()=>view.result.current.route.toggleMode()); act(()=>view.result.current.route.openLoop());
 expect(view.result.current.route.routeView.closed).toBe(false);
 act(()=>view.result.current.route.handleSpotClick("a"));
 expect(view.result.current.route.routeView.closed).toBe(false);
 act(()=>view.result.current.route.closeLoop());
 expect(view.result.current.route.routeView.closed).toBe(true);
});
test("creation options prepare an On Map draft or an At Robot pose in one action",async()=>{
 const options=props();const view=renderHook(useRecordObservationSpots,{initialProps:options});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 act(()=>view.result.current.createOnMap());
 expect(view.result.current.draft.pose).toBeNull();expect(options.onMapMode).toHaveBeenCalledWith("spot");
 act(()=>view.result.current.cancel()); act(()=>view.result.current.createAtRobot());
 expect(view.result.current.draft.pose).toEqual({frame_id:"map",x:3,y:4,yaw:0});
});
test("the dynamic runtime lock blocks route and spot mutations without waiting for a rerender",async()=>{
 routeFixture();let active=false;
 const view=renderHook(useRecordObservationSpots,{initialProps:{...props(),isBlocked:()=>active}});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 act(()=>view.result.current.route.toggleMode());
 act(()=>view.result.current.routeEditor.updatePoint("a",{mission_ids:[]}));
 active=true;
 act(()=>expect(view.result.current.route.handleSpotClick("c")).toBe(false));
 await act(async()=>expect(await view.result.current.routeEditor.save()).toBe(false));
 act(()=>view.result.current.createAtRobot());
 expect(view.result.current.draft).toBeNull();
 expect(autonomyRequest.mock.calls.every(call=>!call[1])).toBe(true);
});

test('dragging and rotating a saved observation persists its pose without localization',async()=>{
 const view=renderHook(useRecordObservationSpots,{initialProps:{...props(),localized:false}});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 await act(async()=>expect(await view.result.current.move('legacy',4,5,Math.PI/2)).toBe(true));
 expect(autonomyRequest).toHaveBeenCalledWith('/autonomy/maps/Home/spots/legacy','PATCH',{
  label:'Table view',pose:{frame_id:'map',x:4,y:5,yaw:Math.PI/2},plan_revision:1});
 expect(view.result.current.spots[0].pose).toEqual({frame_id:'map',x:4,y:5,yaw:Math.PI/2});
 expect(view.result.current.selectedId).toBe('legacy');expect(view.result.current.draft).toBeNull();
});
test('dragging a new or edited observation changes only its draft until Save',async()=>{
 const view=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 act(()=>view.result.current.createOnMap());
 await act(async()=>expect(await view.result.current.move('observation-preview',6,7,-0.5)).toBe(true));
 expect(view.result.current.draft.pose).toEqual({frame_id:'map',x:6,y:7,yaw:-0.5});
 await act(async()=>expect(await view.result.current.move('legacy',10,20,1)).toBe(false));
 expect(autonomyRequest.mock.calls.every(call=>!call[1])).toBe(true);
 act(()=>view.result.current.cancel());act(()=>view.result.current.select('legacy'));act(()=>view.result.current.begin(true));
 await act(async()=>expect(await view.result.current.move('legacy',8,9,1)).toBe(true));
 expect(view.result.current.draft).toMatchObject({id:'legacy',label:'Table view',pose:{x:8,y:9,yaw:1}});
 expect(autonomyRequest.mock.calls.every(call=>!call[1])).toBe(true);
 await act(async()=>expect(await view.result.current.save()).toBe(true));
 expect(view.result.current.spots[0].pose).toMatchObject({x:8,y:9,yaw:1});
});
test('route editing stages pose edits while capture locks reject stale drag callbacks',async()=>{
 let locked=false;const view=renderHook(useRecordObservationSpots,{initialProps:{...props(),isBlocked:()=>locked}});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 const move=view.result.current.move;
 locked=true;
 await act(async()=>expect(await move('legacy',4,5,0)).toBe(false));
 locked=false;act(()=>view.result.current.route.toggleMode());
 await act(async()=>expect(await move('legacy',4,5,0)).toBe(true));
 expect(view.result.current.spots[0].pose).toMatchObject({x:4,y:5});
 expect(view.result.current.configuration.spots[0].pose).toEqual(original.pose);
 expect(autonomyRequest.mock.calls.every(call=>!call[1])).toBe(true);
});
test('failed drag save preserves the saved observation and reports the error',async()=>{
 const view=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(view.result.current.ready).toBe(true));
 autonomyRequest.mockRejectedValueOnce(new Error('Plan changed; refresh first.'));
 await act(async()=>expect(await view.result.current.move('legacy',4,5,0)).toBe(false));
 expect(view.result.current.spots[0].pose).toEqual(original.pose);
 expect(view.result.current.error).toBe('Plan changed; refresh first.');
 expect(view.result.current.working).toBe(false);
});

test("saved waypoint pose edits undo and redo without changing navigation", async () => {
 const {result}=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(result.current.ready).toBe(true));
 await act(async()=>expect(await result.current.move('legacy',7,8,1)).toBe(true));
 expect(result.current.canUndo).toBe(true);
 await act(async()=>expect(await result.current.undo()).toBe(true));
 expect(result.current.spots[0].pose).toEqual(original.pose);
 expect(result.current.canRedo).toBe(true);
 await act(async()=>expect(await result.current.redo()).toBe(true));
 expect(result.current.spots[0].pose).toEqual({frame_id:'map',x:7,y:8,yaw:1});
 await act(async()=>result.current.undo());
 await act(async()=>result.current.move('legacy',4,5,0));
 expect(result.current.canRedo).toBe(false);
});

test("route changes can be undone and redone before saving", async () => {
 const {result}=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(result.current.ready).toBe(true));
 act(()=>result.current.route.toggleMode());
 act(()=>result.current.route.clearRoute());
 expect(result.current.routeEditor.plan.observations).toEqual([]);
 await act(async()=>expect(await result.current.undo()).toBe(true));
 expect(result.current.routeEditor.plan.observations).toEqual([{spot_id:'legacy'}]);
 await act(async()=>expect(await result.current.redo()).toBe(true));
 expect(result.current.routeEditor.plan.observations).toEqual([]);
 expect(autonomyRequest.mock.calls.every(call=>!call[1])).toBe(true);
});

test("failed undo retains history and blocked execution forbids undo", async () => {
 const options=props(),{result,rerender}=renderHook(useRecordObservationSpots,{initialProps:options});
 await waitFor(()=>expect(result.current.ready).toBe(true));
 await act(async()=>result.current.move('legacy',7,8,1));
 rerender({...options,blocked:true});
 await act(async()=>expect(await result.current.undo()).toBe(false));
 rerender(options);
 autonomyRequest.mockRejectedValueOnce(new Error('Conflict'));
 await act(async()=>expect(await result.current.undo()).toBe(false));
 expect(result.current.canUndo).toBe(true);
 expect(result.current.canRedo).toBe(false);
 expect(result.current.error).toBe('Conflict');
});

test("external edits invalidate old undo and map reload clears history", async () => {
 const {result}=renderHook(useRecordObservationSpots,{initialProps:props()});
 await waitFor(()=>expect(result.current.ready).toBe(true));
 await act(async()=>result.current.move('legacy',7,8,1));
 data={...data,spots:data.spots.map(spot=>({...spot,label:'Changed elsewhere'}))};
 await act(async()=>result.current.refresh());
 await act(async()=>expect(await result.current.undo()).toBe(false));
 expect(result.current.canUndo).toBe(false);
 expect(result.current.spots[0].label).toBe('Changed elsewhere');
 await act(async()=>result.current.move('legacy',3,4,0));
 expect(result.current.canUndo).toBe(true);
 await act(async()=>result.current.reload());
 expect(result.current.canUndo).toBe(false);
});
