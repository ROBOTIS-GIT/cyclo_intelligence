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
import useRecordObservationSpots from "./useRecordObservationSpots";
import { autonomyRequest } from "./autonomyApi";

jest.mock("./autonomyApi", () => ({ autonomyRequest: jest.fn(), OBSERVATION_PLAN_CHANGED: "observation-spots-changed" }));
let data, backend;
const options = () => ({ enabled: true, mapName: "Home", localized: false, blocked: false, onMapMode: jest.fn(), missionId: "clean" });
beforeEach(() => {
  jest.resetAllMocks();
  data = { plan: { revision: 8, mode: "patrol", route_closed: false, excluded_mission_ids: [], observations: [
    { spot_id: "a", dwell_seconds: 25, mission_ids: ["clean"], arrival_bt: "table.xml" },
    { spot_id: "b", dwell_seconds: 12, mission_ids: ["wash"], arrival_bt: "laundry.xml" },
  ] }, spots: ["a", "b", "c"].map((id, index) => ({ id, label: id.toUpperCase(),
    pose: { frame_id: "map", x: index, y: 0, yaw: 0 }, metadata: { role: "observation" } })),
  missions: [{ mission_id: "clean" }, { mission_id: "wash" }] };
  backend = async (path, method = "GET", body) => {
    if (method === "GET") return data;
    const revision = method === "DELETE" ? Number(path.split("plan_revision=")[1]) : body.plan_revision ?? body.revision;
    if (revision !== data.plan.revision) throw new Error("Plan revision conflict");
    if (method === "PUT") { data = { ...data, plan: { ...body, revision: revision + 1 } }; return data; }
    const id = decodeURIComponent(path.split("/").at(-1).split("?")[0]);
    if (method === "DELETE") {
      const observations = data.plan.observations.filter(point => point.spot_id !== id);
      data = { ...data, spots: data.spots.filter(spot => spot.id !== id), plan: { ...data.plan, observations,
        revision: revision + (observations.length !== data.plan.observations.length ? 1 : 0) } };
      return data;
    }
    const spot = { ...data.spots.find(value => value.id === id), label: body.label.trim(), pose: body.pose };
    data = { ...data, spots: data.spots.map(value => value.id === id ? spot : value), plan: { ...data.plan, revision: revision + 1 } };
    return spot;
  };
  autonomyRequest.mockImplementation(backend);
});
async function openEditor(extra = {}) {
  const view = renderHook(useRecordObservationSpots, { initialProps: { ...options(), ...extra } });
  await waitFor(() => expect(view.result.current.ready).toBe(true));
  act(() => view.result.current.route.toggleMode());
  return view;
}
const mutations = () => autonomyRequest.mock.calls.filter(call => call[1] && call[1] !== "GET");

test("selection does not edit inclusion or loop state and a dirty toggle keeps the editor open", async () => {
  const { result } = await openEditor();
  const plan = result.current.routeEditor.plan;
  for (const id of ["c", "a", "b"]) {
    act(() => expect(result.current.route.handleSpotClick(id)).toBe(true));
    expect(result.current.selectedId).toBe(id);
    expect(result.current.routeEditor.plan).toEqual(plan);
    expect(result.current.planDirty).toBe(false);
  }
  act(() => expect(result.current.route.toggleMode()).toBe(true));
  expect(result.current.route.routeMode).toBe(false);
  act(() => result.current.route.toggleMode());
  act(() => result.current.route.setIncluded("c", true));
  act(() => expect(result.current.route.toggleMode()).toBe(false));
  expect(result.current.route.routeMode).toBe(true);
  expect(mutations()).toEqual([]);
});

test("inclusion and reordering retain dwell, missions and arrival settings, including unsaved values", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updatePoint("a", { dwell_seconds: 42, mission_ids: ["wash"] }));
  act(() => result.current.route.setIncluded("a", false));
  expect(result.current.routeEditor.pointFor("a")).toMatchObject({ dwell_seconds: 42, mission_ids: ["wash"] });
  act(() => result.current.route.setIncluded("a", true));
  expect(result.current.routeEditor.plan.observations.map(point => point.spot_id)).toEqual(["b", "a"]);
  expect(result.current.routeEditor.plan.observations[1]).toEqual({ spot_id: "a", dwell_seconds: 42, mission_ids: ["wash"], arrival_bt: "table.xml" });
  act(() => result.current.route.moveSpot("a", -1));
  act(() => result.current.route.closeLoop());
  await act(async () => expect(await result.current.undo()).toBe(true));
  expect(result.current.route.routeView.closed).toBe(false);
  await act(async () => expect(await result.current.redo()).toBe(true));
  expect(result.current.route.routeView.closed).toBe(true);
  expect(mutations()).toEqual([]);
});

test("drag, rotation and name changes are staged, undoable, and discarded without changing saved execution poses", async () => {
  const { result } = await openEditor();
  act(() => result.current.select("a"));
  const originalPose = result.current.configuration.spots[0].pose;
  await act(async () => expect(await result.current.move("a", 6, 7, 1.5)).toBe(true));
  act(() => result.current.routeEditor.updateSpot("a", { label: "New name" }));
  expect(result.current.spots[0]).toMatchObject({ label: "New name", pose: { x: 6, y: 7, yaw: 1.5 } });
  expect(result.current.configuration.spots[0].pose).toEqual(originalPose);
  expect(result.current.returnSpots[0].pose).toEqual(originalPose);
  expect(result.current.planDirty).toBe(true);
  await act(async () => expect(await result.current.undo()).toBe(true));
  expect(result.current.spots[0].label).toBe("A");
  await act(async () => result.current.undo());
  expect(result.current.spots[0].pose).toEqual(originalPose);
  expect(result.current.planDirty).toBe(false);
  await act(async () => result.current.redo());
  await act(async () => expect(await result.current.routeEditor.discard()).toBe(true));
  expect(result.current.spots[0].pose).toEqual(originalPose);
  expect(result.current.route.routeMode).toBe(false);
  expect(result.current.canUndo).toBe(false);
  expect(mutations()).toEqual([]);
});

test("Save uses each current server revision and closes after applying waypoints and route settings", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  await act(async () => result.current.move("b", 4, 5, 0.5));
  act(() => result.current.routeEditor.updatePoint("a", { dwell_seconds: 40 }));
  act(() => result.current.route.setRouteOrder(["b", "a"]));
  act(() => result.current.route.closeLoop());
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  const writes = mutations();
  expect(writes.map(call => call[1])).toEqual(["PATCH", "PATCH", "PUT"]);
  expect(writes.slice(0, 2).map(call => call[2].plan_revision)).toEqual([8, 9]);
  expect(writes[2][2]).toMatchObject({ revision: 10, route_closed: true, observations: [
    { spot_id: "b", dwell_seconds: 12, mission_ids: ["wash"], arrival_bt: "laundry.xml" },
    { spot_id: "a", dwell_seconds: 40, mission_ids: ["clean"], arrival_bt: "table.xml" },
  ] });
  expect(result.current.configuration.plan.revision).toBe(11);
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.planDirty).toBe(false);
  expect(result.current.route.routeMode).toBe(false);
});

test("failed first waypoint save keeps the full draft and can be retried", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  act(() => result.current.routeEditor.updatePoint("a", { dwell_seconds: 40 }));
  let fail = true;
  autonomyRequest.mockImplementation((path, method, body) => {
    if (method === "PATCH" && fail) { fail = false; return Promise.reject(new Error("Write unavailable")); }
    return backend(path, method, body);
  });
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(result.current.route.routeMode).toBe(true);
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.configuration.spots[0].label).toBe("A");
  expect(result.current.routeEditor.plan.observations[0].dwell_seconds).toBe(40);
  expect(result.current.error).toMatch(/Write unavailable/);
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(result.current.configuration.spots[0].label).toBe("Table");
});

test("a partial waypoint failure preserves the remaining draft and retries only remaining writes at the new revision", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  act(() => result.current.routeEditor.updateSpot("b", { label: "Laundry" }));
  act(() => result.current.routeEditor.updatePoint("a", { mission_ids: ["wash"] }));
  let fail = true;
  autonomyRequest.mockImplementation((path, method, body) => {
    if (method === "PATCH" && path.endsWith("/b") && fail) { fail = false; return Promise.reject(new Error("Second write failed")); }
    return backend(path, method, body);
  });
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(result.current.error).toMatch(/Some changes were saved/);
  expect(result.current.configuration.plan.revision).toBe(9);
  expect(result.current.configuration.spots.map(spot => spot.label)).toEqual(["Table", "B", "C"]);
  expect(result.current.spots.map(spot => spot.label)).toEqual(["Table", "Laundry", "C"]);
  expect(result.current.routeEditor.plan.observations[0].mission_ids).toEqual(["wash"]);
  expect(result.current.routeEditor.dirty).toBe(true);
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations().filter(call => call[0].endsWith("/a"))).toHaveLength(1);
  expect(mutations().filter(call => call[0].endsWith("/b")).map(call => call[2].plan_revision)).toEqual([9, 9]);
  expect(mutations().find(call => call[1] === "PUT")[2].revision).toBe(10);
});

test("a route PUT failure after a waypoint PATCH leaves bindings staged and retry does not repeat the PATCH", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  act(() => result.current.routeEditor.updatePoint("a", { dwell_seconds: 60 }));
  let fail = true;
  autonomyRequest.mockImplementation((path, method, body) => {
    if (method === "PUT" && fail) { fail = false; return Promise.reject(new Error("Route write failed")); }
    return backend(path, method, body);
  });
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(result.current.configuration.spots[0].label).toBe("Table");
  expect(result.current.configuration.plan.observations[0].dwell_seconds).toBe(25);
  expect(result.current.routeEditor.plan.observations[0].dwell_seconds).toBe(60);
  expect(result.current.route.routeMode).toBe(true);
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations().map(call => call[1])).toEqual(["PATCH", "PUT", "PUT"]);
  expect(mutations().filter(call => call[1] === "PUT").map(call => call[2].revision)).toEqual([9, 9]);
});

test("a successful PATCH followed by failed refresh is reconciled on retry without repeating the write", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  let failRefresh = false;
  autonomyRequest.mockImplementation(async (path, method, body) => {
    if ((!method || method === "GET") && failRefresh) { failRefresh = false; throw new Error("Refresh unavailable"); }
    const value = await backend(path, method, body);
    if (method === "PATCH") failRefresh = true;
    return value;
  });
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(result.current.routeEditor.dirty).toBe(true);
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.error).toMatch(/Some changes were saved/);
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations()).toHaveLength(1);
  expect(result.current.configuration.spots[0].label).toBe("Table");
});

test("waypoint deletion is staged with its route removal and supports undo, redo and Save", async () => {
  const { result } = await openEditor();
  act(() => expect(result.current.routeEditor.removeWaypoint("a")).toBe(true));
  expect(result.current.spots.map(spot => spot.id)).toEqual(["b", "c"]);
  expect(result.current.routeEditor.plan.observations.map(point => point.spot_id)).toEqual(["b"]);
  expect(result.current.configuration.spots).toHaveLength(3);
  expect(mutations()).toEqual([]);
  await act(async () => expect(await result.current.undo()).toBe(true));
  expect(result.current.spots).toHaveLength(3);
  expect(result.current.routeEditor.plan.observations[0].mission_ids).toEqual(["clean"]);
  await act(async () => expect(await result.current.redo()).toBe(true));
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations()[0]).toEqual(["/autonomy/maps/Home/spots/a?plan_revision=8", "DELETE"]);
  expect(result.current.configuration.spots.map(spot => spot.id)).toEqual(["b", "c"]);
  expect(result.current.route.routeMode).toBe(false);
});

test("Cancel after partial Save discards only the unsaved changes and reports persisted waypoint edits", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  act(() => result.current.routeEditor.updatePoint("a", { dwell_seconds: 60 }));
  autonomyRequest.mockImplementation((path, method, body) => method === "PUT"
    ? Promise.reject(new Error("Route write failed")) : backend(path, method, body));
  await act(async () => result.current.routeEditor.save());
  await act(async () => expect(await result.current.routeEditor.discard()).toBe(true));
  expect(result.current.route.routeMode).toBe(false);
  expect(result.current.planDirty).toBe(false);
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.configuration.plan.observations[0].dwell_seconds).toBe(25);
  expect(result.current.error).toMatch(/already saved/);
});

test("external revision changes reject Save without losing the draft or overwriting newer data", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  data = { ...data, plan: { ...data.plan, revision: 9, mode: "observe" } };
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(result.current.error).toMatch(/changed elsewhere/);
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.route.routeMode).toBe(true);
  expect(mutations()).toEqual([]);
});

test("runtime locks block all staging, history and save even through stale callbacks", async () => {
  let active = false;
  const { result } = await openEditor({ isBlocked: () => active });
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  const editor = result.current.routeEditor, route = result.current.route, move = result.current.move;
  active = true;
  act(() => expect(editor.updateSpot("a", { label: "Locked" })).toBe(false));
  act(() => expect(editor.updatePoint("a", { dwell_seconds: 90 })).toBe(false));
  act(() => expect(editor.removeWaypoint("a")).toBe(false));
  act(() => expect(route.setIncluded("c", true)).toBe(false));
  await act(async () => expect(await move("a", 9, 9, 0)).toBe(false));
  await act(async () => expect(await editor.save()).toBe(false));
  await act(async () => expect(await editor.discard()).toBe(false));
  await act(async () => expect(await result.current.undo()).toBe(false));
  expect(result.current.spots[0].label).toBe("Table");
  expect(mutations()).toEqual([]);
});

test.each([
  ["zero dwell", editor => editor.updatePoint("a", { dwell_seconds: 0 })],
  ["nonfinite dwell", editor => editor.updatePoint("a", { dwell_seconds: Infinity })],
  ["excessive dwell", editor => editor.updatePoint("a", { dwell_seconds: 3601 })],
  ["missing mission", editor => editor.updatePoint("a", { mission_ids: ["deleted"] })],
  ["duplicate waypoint", editor => editor.updatePoint("a", { spot_id: "b" })],
  ["long name", editor => editor.updateSpot("b", { label: "x".repeat(129) })],
  ["non-map pose", editor => editor.updateSpot("b", { pose: { frame_id: "odom" } })],
])("invalid %s blocks all writes before any valid staged waypoint can partially persist", async (_, invalidate) => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  act(() => invalidate(result.current.routeEditor));
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(mutations()).toEqual([]);
  expect(result.current.configuration.spots[0].label).toBe("A");
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.routeEditor.dirty).toBe(true);
  expect(result.current.route.routeMode).toBe(true);
});

test.each(["a", "c"])("a lost DELETE response for %s is reconciled using the backend's included/unincluded revision rules", async id => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.removeWaypoint(id));
  let loseResponse = true;
  autonomyRequest.mockImplementation(async (path, method, body) => {
    const value = await backend(path, method, body);
    if (method === "DELETE" && loseResponse) { loseResponse = false; throw new Error("Response disconnected"); }
    return value;
  });
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(result.current.routeEditor.dirty).toBe(true);
  expect(result.current.spots.some(spot => spot.id === id)).toBe(false);
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations()).toHaveLength(1);
  expect(result.current.configuration.plan.revision).toBe(id === "a" ? 9 : 8);
  expect(result.current.configuration.spots.some(spot => spot.id === id)).toBe(false);
});

test("an episode starting during Save stops further requests while preserving the confirmed partial write", async () => {
  let active = false;
  const { result } = await openEditor({ isBlocked: () => active });
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  act(() => result.current.routeEditor.updateSpot("b", { label: "Laundry" }));
  autonomyRequest.mockImplementation(async (path, method, body) => {
    const value = await backend(path, method, body);
    if (method === "PATCH") active = true;
    return value;
  });
  await act(async () => expect(await result.current.routeEditor.save()).toBe(false));
  expect(mutations()).toHaveLength(1);
  expect(result.current.error).toMatch(/Some changes were saved/);
  expect(result.current.routeEditor.dirty).toBe(true);
  expect(result.current.route.routeMode).toBe(true);
  active = false;
  autonomyRequest.mockImplementation(backend);
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations().map(call => call[2].plan_revision)).toEqual([8, 9]);
});

test("failed Cancel keeps the draft and editing mode open until the saved map can be reloaded", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updateSpot("a", { label: "Table" }));
  autonomyRequest.mockRejectedValueOnce(new Error("Map unavailable"));
  await act(async () => expect(await result.current.routeEditor.discard()).toBe(false));
  expect(result.current.spots[0].label).toBe("Table");
  expect(result.current.route.routeMode).toBe(true);
  expect(result.current.routeEditor.dirty).toBe(true);
  expect(result.current.error).toMatch(/Map unavailable/);
  await act(async () => expect(await result.current.routeEditor.discard()).toBe(true));
  expect(result.current.spots[0].label).toBe("A");
  expect(result.current.route.routeMode).toBe(false);
});

test("order mode starts a fresh staged route and rapid clicks preserve settings without selecting or saving", async () => {
  const { result } = await openEditor();
  act(() => result.current.routeEditor.updatePoint("a", { dwell_seconds: 42, mission_ids: ["wash"] }));
  act(() => result.current.select("b"));
  const saved = result.current.configuration.plan;
  act(() => expect(result.current.route.toggleOrderMode()).toBe(true));
  expect(result.current.route.orderMode).toBe(true);
  expect(result.current.selectedId).toBe("");
  expect(result.current.route.routeView.order).toEqual([]);
  expect(result.current.configuration.plan).toEqual(saved);
  act(() => {
    const route = result.current.route;
    expect(route.handleSpotClick("c")).toBe(true);
    expect(route.handleSpotClick("a")).toBe(true);
    expect(route.handleSpotClick("b")).toBe(true);
  });
  expect(result.current.route.routeView.order).toEqual([
    { id: "c", order: 1 }, { id: "a", order: 2 }, { id: "b", order: 3 },
  ]);
  expect(result.current.routeEditor.plan.observations).toEqual([
    { spot_id: "c", departure_policy: "reasoning", dwell_seconds: 15, mission_ids: [], arrival_bt: "" },
    { spot_id: "a", dwell_seconds: 42, mission_ids: ["wash"], arrival_bt: "table.xml" },
    { spot_id: "b", dwell_seconds: 12, mission_ids: ["wash"], arrival_bt: "laundry.xml" },
  ]);
  expect(result.current.selectedId).toBe("");
  expect(result.current.route.routeView.closed).toBe(false);
  expect(mutations()).toEqual([]);
});

test("clicking the first point closes a loop after two distinct points and consumes duplicate clicks", async () => {
  const { result } = await openEditor();
  act(() => {
    const route = result.current.route;
    route.toggleOrderMode();
    route.handleSpotClick("b");
    route.handleSpotClick("b");
    expect(route.handleSpotClick("missing")).toBe(false);
  });
  expect(result.current.route.orderMode).toBe(true);
  expect(result.current.route.routeView.order).toEqual([{ id: "b", order: 1 }]);
  act(() => {
    const route = result.current.route;
    route.handleSpotClick("a");
    route.handleSpotClick("a");
    route.handleSpotClick("b");
  });
  expect(result.current.route.orderMode).toBe(false);
  expect(result.current.route.routeMode).toBe(true);
  expect(result.current.selectedId).toBe("");
  expect(result.current.route.routeView.closed).toBe(true);
  expect(result.current.route.routeView.executionSpots.map(spot => spot.id)).toEqual(["b", "a", "b"]);
  expect(mutations()).toEqual([]);
});

test("finishing order mode leaves a one-way route and restores waypoint settings selection", async () => {
  const { result } = await openEditor();
  act(() => {
    result.current.route.toggleOrderMode();
    result.current.route.handleSpotClick("c");
    result.current.route.handleSpotClick("a");
    result.current.route.toggleOrderMode();
  });
  expect(result.current.route.orderMode).toBe(false);
  expect(result.current.route.routeView.closed).toBe(false);
  act(() => result.current.route.handleSpotClick("a"));
  expect(result.current.selectedId).toBe("a");
  expect(result.current.routeEditor.plan.observations.map(point => point.spot_id)).toEqual(["c", "a"]);
});

test("route order and its initial reset are undoable without losing point settings", async () => {
  const { result } = await openEditor();
  const original = result.current.routeEditor.plan;
  act(() => result.current.route.toggleOrderMode());
  act(() => result.current.route.handleSpotClick("b"));
  await act(async () => expect(await result.current.undo()).toBe(true));
  expect(result.current.routeEditor.plan.observations).toEqual([]);
  await act(async () => expect(await result.current.undo()).toBe(true));
  expect(result.current.routeEditor.plan).toEqual(original);
  await act(async () => expect(await result.current.redo()).toBe(true));
  expect(result.current.routeEditor.plan.observations).toEqual([]);
  await act(async () => expect(await result.current.redo()).toBe(true));
  expect(result.current.routeEditor.plan.observations).toEqual([original.observations[1]]);
  expect(result.current.route.orderMode).toBe(true);
  expect(mutations()).toEqual([]);
});

test("order mode blocks stale drag and settings callbacks and rejects saving an empty route", async () => {
  const { result } = await openEditor();
  const move = result.current.move, editor = result.current.routeEditor;
  const originalSpots = result.current.spots;
  act(() => result.current.route.toggleOrderMode());
  await act(async () => expect(await move("a", 9, 9, 1)).toBe(false));
  act(() => expect(editor.updateSpot("a", { label: "Changed" })).toBe(false));
  act(() => expect(editor.updatePoint("a", { dwell_seconds: 90 })).toBe(false));
  act(() => expect(editor.removeWaypoint("a")).toBe(false));
  await act(async () => expect(await editor.save()).toBe(false));
  expect(result.current.routeEditor.error).toMatch(/at least one waypoint/);
  expect(result.current.spots).toEqual(originalSpots);
  expect(mutations()).toEqual([]);
  act(() => result.current.route.handleSpotClick("a"));
  expect(result.current.routeEditor.error).toBeNull();
  await act(async () => expect(await result.current.routeEditor.save()).toBe(true));
  expect(mutations()).toHaveLength(1);
  expect(mutations()[0][1]).toBe("PUT");
  expect(data.plan.observations.map(point => point.spot_id)).toEqual(["a"]);
  expect(data.plan.route_closed).toBe(false);
  expect(result.current.route.routeMode).toBe(false);
  expect(result.current.route.orderMode).toBe(false);
});

test("discarding order mode restores the saved route and clears both editing modes", async () => {
  const { result } = await openEditor();
  const original = result.current.configuration.plan;
  act(() => result.current.route.toggleOrderMode());
  act(() => result.current.route.handleSpotClick("c"));
  await act(async () => expect(await result.current.routeEditor.discard()).toBe(true));
  expect(result.current.routeEditor.plan).toEqual(original);
  expect(result.current.route.routeMode).toBe(false);
  expect(result.current.route.orderMode).toBe(false);
  expect(result.current.canUndo).toBe(false);
  expect(mutations()).toEqual([]);
});
