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
import useReasoningObservation from "./useReasoningObservation";

const model = { id: "one", model_name: "Warehouse model", map_name: "warehouse", base_model: "Qwen", sample_count: 24 };
let models;
let started;
const settle = async () => { await act(async () => { await Promise.resolve(); }); };
const calls = suffix => fetch.mock.calls.filter(([url]) => url.endsWith(suffix));
beforeEach(() => {
  jest.useFakeTimers(); localStorage.clear();
  models = [model, { ...model, id: "two", model_name: "New model" }]; started = {};
  global.fetch = jest.fn(async (url, options) => {
    let data = {};
    if (url.includes("/models?")) data = { models: models.filter(item => item.map_name === new URL(url, "http://localhost").searchParams.get("map_name")) };
    if (url.endsWith("/start")) { started = JSON.parse(options.body); data = { session_id: "session" }; }
    if (url.endsWith("/status")) data = { phase: "ready", session_id: "session", selected_model_id: started.model_id };
    if (url.endsWith("/observe")) data = { frame_stamp: "1", decision: { decision: "start", mission_id: "mission", reason: "Task object is visible." }, selected_mission: { mission_name: "Pickup" } };
    return { ok: true, json: async () => data };
  });
});
afterEach(() => { jest.useRealTimers(); delete global.fetch; });

test("model selection requires explicit Start and preview never records or runs a mission", async () => {
  const view = renderHook(() => useReasoningObservation({ enabled: true, episode: { id: "saved", map_name: "warehouse" } }));
  await settle(); expect(view.result.current.available).toBe(false);
  await act(async () => view.result.current.start()); expect(calls("/start")).toHaveLength(0);
  act(() => view.result.current.modelChoice.select("one"));
  expect(view.result.current.available).toBe(true); expect(calls("/start")).toHaveLength(0);
  await act(async () => view.result.current.start());
  expect(JSON.parse(calls("/start")[0][1].body)).toEqual({ model_id: "one", map_name: "warehouse" });
  const body = JSON.parse(calls("/observe")[0][1].body);
  expect(body.episode_id).toBeNull();
  expect(body.context).toMatchObject({ map_name: "warehouse", pose: null, localized: null, preview_active_mission: null });
  expect(view.result.current.result.decision.decision).toBe("start");
  expect(fetch.mock.calls.every(([url]) => /^\/api\/reasoning\/(models\?|start$|status$|observe$)/.test(url))).toBe(true);
  view.unmount();
});

test("leaving Live pauses polling, retains the model lock, and can resume the same model", async () => {
  localStorage.setItem("autonomy.operate.model.warehouse", "one");
  const view = renderHook(({ enabled }) => useReasoningObservation({ enabled, loadedMapName: "warehouse" }), { initialProps: { enabled: false } });
  await settle(); await act(async () => view.result.current.start()); expect(calls("/start")).toHaveLength(0);
  view.rerender({ enabled: true }); await act(async () => view.result.current.start());
  view.rerender({ enabled: false }); const count = fetch.mock.calls.length;
  await act(async () => jest.advanceTimersByTime(2000));
  expect(fetch).toHaveBeenCalledTimes(count); expect(view.result.current.active).toBe(false);
  expect(view.result.current.modelChoice.locked).toBe(true);
  view.rerender({ enabled: true }); await act(async () => view.result.current.start());
  expect(calls("/start")).toHaveLength(2);
  await act(async () => view.result.current.stop()); expect(view.result.current.modelChoice.locked).toBe(false);
  view.unmount();
});

test("the live map and its pose take precedence over an episode from another map", async () => {
  const pose = { x: 1, y: 2 };
  const view = renderHook(() => useReasoningObservation({ enabled: true, loadedMapName: "warehouse", pose, localized: true, episode: { id: "old", map_name: "other" } }));
  await settle(); act(() => view.result.current.modelChoice.select("one"));
  await act(async () => view.result.current.start());
  expect(view.result.current.mapName).toBe("warehouse");
  expect(JSON.parse(calls("/observe")[0][1].body).context).toMatchObject({ map_name: "warehouse", pose, localized: true });
  view.unmount();
  const { result, unmount } = renderHook(() => useReasoningObservation({ enabled: true }));
  expect(result.current.available).toBe(false); unmount();
});

test("Dataset handoff refreshes and selects the requested model without starting reasoning", async () => {
  const view = renderHook(() => useReasoningObservation({ enabled: true, requestedModel: model }));
  expect(view.result.current.available).toBe(false); await settle();
  expect(fetch.mock.calls.filter(([url]) => url.includes("/models?"))).toHaveLength(2);
  expect(view.result.current.modelChoice.selected).toEqual(model); expect(view.result.current.available).toBe(true);
  expect(calls("/start")).toHaveLength(0); view.unmount();
});

test("handoff for another live map is rejected until a compatible model is explicitly selected", async () => {
  localStorage.setItem("autonomy.operate.model.warehouse", "one");
  const requestedModel = { ...model, id: "other", map_name: "other" };
  const view = renderHook(() => useReasoningObservation({ enabled: true, loadedMapName: "warehouse", requestedModel }));
  await settle(); expect(view.result.current.available).toBe(false);
  expect(view.result.current.modelChoice.error).toContain("belongs to other");
  await act(async () => view.result.current.start()); expect(calls("/start")).toHaveLength(0);
  act(() => view.result.current.modelChoice.select("one"));
  expect(view.result.current.available).toBe(true); expect(view.result.current.modelChoice.error).toBeNull(); view.unmount();
});

test("a missing handoff blocks a remembered selection and refreshes after publication", async () => {
  localStorage.setItem("autonomy.operate.model.warehouse", "one");
  const requestedModel = { ...model, id: "new" };
  const view = renderHook(() => useReasoningObservation({ enabled: true, loadedMapName: "warehouse", requestedModel }));
  await settle(); expect(view.result.current.available).toBe(false);
  expect(view.result.current.modelChoice.error).toContain("not available");
  models = [...models, requestedModel]; await act(async () => view.result.current.modelChoice.refresh());
  expect(view.result.current.modelChoice.selected.id).toBe("new"); expect(view.result.current.available).toBe(true);
  expect(calls("/start")).toHaveLength(0); view.unmount();
});

test("selection and refresh stay locked until Stop succeeds, including a paused preview", async () => {
  const view = renderHook(() => useReasoningObservation({ enabled: true, loadedMapName: "warehouse" }));
  await settle(); act(() => view.result.current.modelChoice.select("one"));
  await act(async () => view.result.current.start()); act(() => view.result.current.pause());
  const count = fetch.mock.calls.length;
  act(() => { view.result.current.modelChoice.select("two"); view.result.current.modelChoice.refresh(); });
  expect(view.result.current.modelChoice.selected.id).toBe("one"); expect(fetch).toHaveBeenCalledTimes(count);
  fetch.mockResolvedValueOnce({ ok: false, json: async () => ({ detail: "Stop failed" }) });
  await act(async () => expect(view.result.current.stop()).resolves.toBe(false));
  expect(view.result.current.modelChoice.locked).toBe(true);
  await act(async () => view.result.current.stop()); act(() => view.result.current.modelChoice.select("two"));
  expect(view.result.current.modelChoice.selected.id).toBe("two"); view.unmount();
});

test("changing the live map pauses inference and requires Stop before using its model", async () => {
  models.push({ ...model, id: "other", map_name: "other" });
  localStorage.setItem("autonomy.operate.model.warehouse", "one"); localStorage.setItem("autonomy.operate.model.other", "other");
  const view = renderHook(({ map }) => useReasoningObservation({ enabled: true, loadedMapName: map }), { initialProps: { map: "warehouse" } });
  await settle(); await act(async () => view.result.current.start());
  view.rerender({ map: "other" }); await settle();
  expect(view.result.current.active).toBe(false); expect(view.result.current.available).toBe(false);
  expect(view.result.current.selectionError).toContain("Stop reasoning");
  await act(async () => view.result.current.start()); expect(calls("/start")).toHaveLength(1);
  await act(async () => view.result.current.stop()); expect(view.result.current.available).toBe(true);
  await act(async () => view.result.current.start()); expect(started).toEqual({ model_id: "other", map_name: "other" });
  view.unmount();
});


test("a handoff received during preview waits for Stop before replacing the model", async () => {
  localStorage.setItem("autonomy.operate.model.warehouse", "one");
  const view = renderHook(({ requestedModel }) => useReasoningObservation({ enabled: true, loadedMapName: "warehouse", requestedModel }), { initialProps: { requestedModel: null } });
  await settle(); await act(async () => view.result.current.start());
  view.rerender({ requestedModel: models[1] }); await settle();
  expect(view.result.current.modelChoice.selected.id).toBe("one");
  expect(view.result.current.modelChoice.locked).toBe(true);
  expect(view.result.current.active).toBe(false);
  expect(view.result.current.available).toBe(false);
  expect(view.result.current.modelChoice.error).toContain("Stop reasoning");
  await act(async () => view.result.current.stop()); await settle();
  expect(view.result.current.modelChoice.selected.id).toBe("two");
  expect(view.result.current.available).toBe(true);
  expect(calls("/start")).toHaveLength(1);
  view.unmount();
});

test("Record preview samples actual runner context on each request without dispatching mission commands", async () => {
 localStorage.setItem("autonomy.operate.model.warehouse", "one");
 let reported = {reasoning_task:"mission_selection",recording_execution:null};
 const getExecutionContext = jest.fn(() => reported);
 const view=renderHook(()=>useReasoningObservation({enabled:true,loadedMapName:"warehouse",getExecutionContext}));
 await settle();await act(async()=>view.result.current.start());
 expect(JSON.parse(calls("/observe")[0][1].body).context.reasoning_task).toBe("mission_selection");
 reported={reasoning_task:"mission_supervision",recording_execution:{mission_id:"mission",status:"paused",step:2,
  action:"bt-running",map_name:"warehouse",sampled_at_ms:Date.now(),context_version:3,context_valid_from_ms:Date.now()}};
 await act(async()=>jest.advanceTimersByTime(200));
 expect(JSON.parse(calls("/observe")[1][1].body).context).toMatchObject(reported);
 expect(fetch.mock.calls.every(([url])=>/^\/api\/reasoning\/(models\?|start$|status$|observe$)/.test(url))).toBe(true);
 view.unmount();
});
