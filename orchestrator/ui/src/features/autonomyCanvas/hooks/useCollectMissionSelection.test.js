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
import useCollectMissionSelection from "./useCollectMissionSelection";
import { NAVIGATION_MISSIONS_CHANGED_EVENT } from "../../../utils/navigationMissionsApi";

function publish(detail) {
  window.dispatchEvent(new CustomEvent(NAVIGATION_MISSIONS_CHANGED_EVENT, { detail }));
}

test("selects one mission snapshot and clears it on map changes", async () => {
  const options = { enabled:true, mapName:"map-a", listMissions:jest.fn(async()=>["Mission1", "Mission2"]), loadMission:jest.fn(async()=>({exists:true})), onMessage:jest.fn() };
  const {result,rerender}=renderHook(p=>useCollectMissionSelection(p),{initialProps:options});
  await waitFor(()=>expect(result.current.names).toHaveLength(2));
  expect(result.current.name).toBe("");
  await act(async()=>{await result.current.select("Mission2");});
  expect(options.loadMission).toHaveBeenCalledWith("map-a","Mission2");
  expect(result.current.name).toBe("Mission2");
  rerender({...options,mapName:"map-b"});
  expect(result.current.name).toBe("");
  await waitFor(()=>expect(result.current.loading).toBe(false));
});

test("ignores an in-flight selection after switching maps", async()=>{
  let finish;
  const options={enabled:true,mapName:"map-a",listMissions:jest.fn(async()=>["Mission1"]),loadMission:jest.fn(()=>new Promise(resolve=>{finish=resolve;})),onMessage:jest.fn()};
  const {result,rerender}=renderHook(p=>useCollectMissionSelection(p),{initialProps:options});
  await waitFor(()=>expect(result.current.loading).toBe(false));
  let request;
  act(()=>{request=result.current.select("Mission1");});
  rerender({...options,mapName:"map-b"});
  await act(async()=>{finish({exists:true});await request;});
  expect(result.current.name).toBe("");
});

test("refreshes a matching Design mutation and follows a selected mission rename", async () => {
  const listMissions = jest.fn()
    .mockResolvedValueOnce(["Mission1", "Mission2"])
    .mockResolvedValueOnce(["Mission1", "ObservationMode"]);
  const loadMission = jest.fn(async () => ({ exists: true }));
  const options = {
    enabled: true,
    mapName: "reasoning",
    listMissions,
    loadMission,
    onMessage: jest.fn(),
  };
  const { result } = renderHook(() => useCollectMissionSelection(options));
  await waitFor(() => expect(result.current.names).toEqual(["Mission1", "Mission2"]));
  await act(async () => { await result.current.select("Mission2"); });

  act(() => publish({
    mapName: "reasoning",
    mutation: "rename-mission",
    previousMissionName: "Mission2",
    missionName: "ObservationMode",
  }));

  await waitFor(() => expect(result.current.names).toEqual(["Mission1", "ObservationMode"]));
  expect(result.current.name).toBe("ObservationMode");
  expect(loadMission).toHaveBeenLastCalledWith("reasoning", "ObservationMode");
});

test("clears a deleted selected mission and ignores changes for another map", async () => {
  const listMissions = jest.fn()
    .mockResolvedValueOnce(["Mission1", "Mission2"])
    .mockResolvedValueOnce(["Mission1"]);
  const options = {
    enabled: true,
    mapName: "reasoning",
    listMissions,
    loadMission: jest.fn(async () => ({ exists: true })),
    onMessage: jest.fn(),
  };
  const { result } = renderHook(() => useCollectMissionSelection(options));
  await waitFor(() => expect(result.current.names).toHaveLength(2));
  await act(async () => { await result.current.select("Mission2"); });

  act(() => publish({ mapName: "warehouse", mutation: "delete-mission", missionName: "Mission2" }));
  expect(listMissions).toHaveBeenCalledTimes(1);
  act(() => publish({ mapName: "reasoning", mutation: "delete-mission", missionName: "Mission2" }));

  await waitFor(() => expect(result.current.names).toEqual(["Mission1"]));
  expect(result.current.name).toBe("");
});

test("reloads the selected mission when the window regains focus", async () => {
  const options = {
    enabled: true,
    mapName: "reasoning",
    listMissions: jest.fn(async () => ["ObservationMode"]),
    loadMission: jest.fn(async () => ({ exists: true })),
    onMessage: jest.fn(),
  };
  const { result } = renderHook(() => useCollectMissionSelection(options));
  await waitFor(() => expect(result.current.names).toEqual(["ObservationMode"]));
  await act(async () => { await result.current.select("ObservationMode"); });
  options.loadMission.mockClear();

  act(() => window.dispatchEvent(new Event("focus")));

  await waitFor(() => expect(options.loadMission).toHaveBeenCalledWith(
    "reasoning",
    "ObservationMode",
  ));
  expect(result.current.name).toBe("ObservationMode");
});
