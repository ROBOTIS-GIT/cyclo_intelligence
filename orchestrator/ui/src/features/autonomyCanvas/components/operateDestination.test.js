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
import { autonomyMapView, useAutonomyMapView } from "./autonomyMapView";

const kitchen = { id: "kitchen", label: "Kitchen", pose: { x: 1, y: 2 } };
const laundry = { id: "laundry", label: "Laundry", pose: { x: 4, y: 5 } };
const execution = { lifecycle: "active", owns_execution: true, control: "running",
  activity: "patrol_navigation", target_spot_id: kitchen.id };
const state = { active: true, phase: "positioning", mode: "observation", execution,
  observation: { ...kitchen, spot_id: kitchen.id },
  config: { draft: { mode: "patrol", observations: [{ spot_id: kitchen.id }, { spot_id: laundry.id }] },
    spots: [kitchen, laundry] } };

test("navigation flag follows execution target through arrival, departure, and return", () => {
  const view = renderHook(({ value }) => useAutonomyMapView(value), { initialProps: { value: state } });
  expect(view.result.current.navigationDestination).toMatchObject({ ...kitchen, scope: "observation" });
  view.rerender({ value: { ...state, execution: { ...execution, activity: "observation_posture" } } });
  expect(view.result.current.navigationDestination).toBeNull();
  view.rerender({ value: { ...state, execution: { ...execution, target_spot_id: laundry.id } } });
  expect(view.result.current.navigationDestination).toMatchObject(laundry);
  view.rerender({ value: { ...state, execution: { ...execution, activity: "return_navigation", target_spot_id: laundry.id },
    resume_checkpoint: { return_spot_id: kitchen.id } } });
  expect(view.result.current.navigationDestination).toMatchObject(laundry);
  view.rerender({ value: { ...state, execution: { ...execution, activity: "return_posture" } } });
  expect(view.result.current.navigationDestination).toBeNull();
});

test.each([
  { activity: "observing" }, { activity: "mission_action" }, { activity: "idle" },
  { control: "pausing" }, { control: "paused" }, { control: "resuming" }, { control: "cancelling" },
  { lifecycle: "stopping" }, { lifecycle: "stopped" }, { lifecycle: "interrupted" },
  { lifecycle: "failed" }, { owns_execution: false }, { target_spot_id: null },
  { target_spot_id: "unknown" },
])("execution %j clears the flag despite retained compatibility fields", change => {
  expect(autonomyMapView({ ...state, execution: { ...execution, ...change } }).navigationDestination).toBeNull();
});

test("mission target comes from normalized execution even when runner and selection are stale", () => {
  const task = { ...laundry, id: kitchen.id };
  const value = { ...state, mode: "mission", phase: "executing", mission_route: [task],
    selectedId: laundry.id, runner: { phase: "navigation", activeSpotId: laundry.id },
    execution: { ...execution, activity: "mission_navigation" } };
  expect(autonomyMapView(value).navigationDestination).toEqual({ ...task, scope: "mission" });
  expect(autonomyMapView({ ...value, execution: { ...value.execution, target_spot_id: null } }).navigationDestination).toBeNull();
});

test("legacy snapshots support patrol, mission, and return but no retained decision destination", () => {
  const legacy = { ...state, execution: null };
  expect(autonomyMapView(legacy).navigationDestination).toMatchObject(kitchen);
  expect(autonomyMapView({ ...legacy, mode: "mission", phase: "executing", mission_route: [laundry],
    runner: { phase: "navigation", activeSpotId: laundry.id } }).navigationDestination).toMatchObject(laundry);
  expect(autonomyMapView({ ...legacy, phase: "returning", resume_checkpoint: { return_spot_id: laundry.id },
    runner: { phase: "return-navigation" } }).navigationDestination).toMatchObject(laundry);
  expect(autonomyMapView({ ...legacy, active: false, phase: "idle", selectedId: laundry.id,
    result: { target_spot_id: laundry.id } }).navigationDestination).toBeNull();
});

test("execution snapshot supplies destination during config hydration without inventing coordinates", () => {
  expect(autonomyMapView({ ...state, config: null }).navigationDestination).toMatchObject(kitchen);
  expect(autonomyMapView({ ...state, config: null, observation: { spot_id: kitchen.id } }).navigationDestination).toBeNull();
});


test("acknowledged running control overrides a stale compatibility pause", () => {
  expect(autonomyMapView({ ...state, paused: true }).navigationDestination).toMatchObject(kitchen);
  expect(autonomyMapView({ ...state, execution: null, paused: true }).navigationDestination).toBeNull();
});
