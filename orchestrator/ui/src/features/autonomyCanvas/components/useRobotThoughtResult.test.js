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
import useRobotThoughtResult from "./useRobotThoughtResult";

const reason = "The table is clean. I will move on to the next observation point.";
const createState = () => ({ active: true, session_id: "run-1", phase: "positioning",
  context_version: 5, reasoning_task: "patrol_supervision",
  execution: { lifecycle: "active", activity: "patrol_navigation", control: "running", target_spot_id: "laundry" },
  events: [
    { title: "Moving to observation", detail: "Laundry", time_ms: Date.now() },
    { title: "Observation complete", detail: `Kitchen · ${reason}`, time_ms: Date.now() - 100 },
    { title: "AI · next_observation", detail: reason, time_ms: Date.now() - 200 },
  ], result: null,
});
const inference = (decision = "continue") => ({ context_version: 5, reasoning_task: "patrol_supervision",
  action_allowed: true, decision: { decision, mission_id: null, reason: decision === "pause" ? "A person is ahead. Stop." : "No person is nearby. Continue." } });
beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test("recovers accepted departure missed between polls, then shows live patrol reasoning", () => {
  const state = createState();
  const view = renderHook(s => useRobotThoughtResult(s), { initialProps: state });
  expect(view.result.current.decision).toEqual({ decision: "next_observation", mission_id: null, reason });
  const fresh = inference();
  view.rerender({ ...state, result: fresh });
  expect(view.result.current.decision.decision).toBe("next_observation");
  act(() => jest.advanceTimersByTime(3200));
  expect(view.result.current).toBe(fresh);
  view.rerender({ ...state, events: [...state.events], result: fresh });
  expect(view.result.current).toBe(fresh);
});

test.each(["pause", "unknown"])("%s overrides the departure immediately and it never resurfaces", decision => {
  const state = createState();
  const view = renderHook(s => useRobotThoughtResult(s), { initialProps: state });
  const fresh = inference(decision);
  view.rerender({ ...state, result: fresh });
  expect(view.result.current).toBe(fresh);
  const continued = inference();
  view.rerender({ ...state, result: continued });
  expect(view.result.current).toBe(continued);
});

test("does not replay a proposal without acceptance, a stale event, or another session's held text", () => {
  const state = createState();
  const view = renderHook(s => useRobotThoughtResult(s), { initialProps: { ...state, events: [state.events[2]] } });
  expect(view.result.current).toBeNull();
  view.rerender(state);
  expect(view.result.current.decision.reason).toBe(reason);
  view.rerender({ ...state, session_id: "run-2", events: [] });
  expect(view.result.current).toBeNull();
  act(() => jest.advanceTimersByTime(15000));
  view.rerender({ ...state, session_id: "run-3" });
  expect(view.result.current).toBeNull();
});

test("arrival, Stop and errors suppress the held departure", () => {
  const state = createState();
  const view = renderHook(s => useRobotThoughtResult(s), { initialProps: state });
  view.rerender({ ...state, phase: "stopping", execution: { lifecycle: "stopping" } });
  expect(view.result.current).toBeNull();
  view.rerender({ ...state, session_id: "run-2" });
  expect(view.result.current.decision.reason).toBe(reason);
  view.rerender({ ...state, phase: "observing", execution: { lifecycle: "active", activity: "observing" } });
  expect(view.result.current).toBeNull();
  view.rerender({ ...state, session_id: "run-3", error: "Camera unavailable" });
  expect(view.result.current).toBeNull();
});
