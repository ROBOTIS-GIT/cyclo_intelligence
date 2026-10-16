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
import useRecordExecution from "./useRecordExecution";

const a = { id: "a", label: "Dining", pose: { x: 1, y: 0, yaw: 0 } };
const b = { id: "b", label: "Laundry", pose: { x: 2, y: 0, yaw: 0 } };
const poseAt = spot => ({ position: { x: spot.pose.x, y: spot.pose.y }, orientation: { w: 1 } });
const arrivalXml = '<root><BehaviorTree ID="Main"><Wait duration="0.01"/></BehaviorTree></root>';

function fixture({ closed = true, points = [a, b], dwell = {}, arrivalBt = false } = {}) {
  const plan = { route_closed: closed, observations: points.map(spot => ({
    spot_id: spot.id, departure_policy: "timed", dwell_seconds: dwell[spot.id] ?? 60, mission_ids: ["clean"],
    arrival_bt: arrivalBt ? arrivalXml : "",
  })) };
  const goals = [];
  const btStatusRef = { current: "idle" };
  const runtime = arrivalBt ? {
    btStatusRef, getFlags: () => ({ btNodeIsUp: true }),
    callService: jest.fn(async () => {
      btStatusRef.current = "running";
      setTimeout(() => { btStatusRef.current = "completed"; }, 30);
      return { success: true };
    }),
    stopBt: jest.fn().mockResolvedValue(true),
    config: { pollMs: 10, btStartTimeoutMs: 500, btTimeoutMs: 1000 },
  } : {};
  let props = {
    enabled: true, mapName: "house", localized: true, currentPose: poseAt(points[0]),
    manager: {
      selectedId: points[0].id, returnSpotId: points[0].id, spots: points,
      configuration: { plan, missions: [{mission_id:"clean",mission_name:"CleanTable"}] }, select: jest.fn(),
      refresh: jest.fn().mockResolvedValue({ spots: points, plan }),
      resolveReturnSpot: jest.fn(async id => ({ ...points.find(spot => spot.id === id),
        arrival_bt: arrivalBt ? arrivalXml : "" })),
    },
    mission: { mapName: "house", missionId: "clean" },
    missionRunner: { isRunning: false, status: "idle", phase: "none", currentIndex: -1,
      stop: jest.fn().mockResolvedValue([]), pause: jest.fn(), resume: jest.fn() },
    startMission: jest.fn().mockResolvedValue(true), cancelMissionStart: jest.fn(),
    cancelGoal: jest.fn().mockResolvedValue({ ok: true }), onMessage: jest.fn(), runtime,
    sendGoal: jest.fn((x, y, yaw, signal) => new Promise(resolve => {
      goals.push({ x, y, yaw, signal, resolve });
      signal.addEventListener("abort", () => resolve({ status: "CANCELED" }), { once: true });
    })),
  };
  const hook = renderHook(useRecordExecution, { initialProps: props });
  const update = changes => { props = { ...props, ...changes }; hook.rerender(props); };
  return { hook, goals, runtime, get props() { return props; }, update,
    at: spot => update({ currentPose: poseAt(spot), manager: { ...props.manager,
      selectedId: spot.id, returnSpotId: spot.id } }),
    missionState: (status, running = false) => update({ missionRunner: { ...props.missionRunner,
      isRunning: running, status, phase: running ? "awaiting-nav-result" : "done", currentIndex: 0 } }),
  };
}
async function start(f, method = "startPatrol", id) {
  let pending;
  act(() => { pending = f.hook.result.current[method](id); });
  await waitFor(() => expect(f.goals.length).toBeGreaterThan(0));
  await act(async () => expect(await pending).toBe(true));
}
async function arrive(f, goalIndex, spot) {
  f.at(spot);
  await act(async () => f.goals[goalIndex].resolve({ status: "SUCCEEDED" }));
}
async function patrolAt(f, spot = a) {
  await start(f);
  await arrive(f, 0, a);
  if (spot.id === b.id) {
    await waitFor(() => expect(f.goals).toHaveLength(2), { timeout: 2000 });
    await arrive(f, 1, b);
  }
  await waitFor(() => expect(f.hook.result.current.navigation.dwelling).toBe(true), { timeout: 2000 });
  expect(f.hook.result.current.observationReady).toBe(true);
}
async function startMission(f) {
  await act(async () => expect(await f.hook.result.current.startChildMission()).toBe(true));
  f.missionState("navigating", true);
  expect(f.hook.result.current.executionType).toBe("mission");
}
async function completeMission(f) {
  const previous = f.goals.length;
  f.missionState("done");
  await waitFor(() => expect(f.goals).toHaveLength(previous + 1));
  expect(f.hook.result.current.returning).toBe(true);
  return previous;
}
async function dispose(f) {
  await act(async () => f.hook.result.current.stop());
  f.hook.unmount();
}

test("paused closed patrol runs a child, returns once, and resumes the remaining dwell without replaying arrival BT", async () => {
  const f = fixture({ arrivalBt: true });
  await patrolAt(f);
  expect(f.runtime.callService).toHaveBeenCalledTimes(1);
  await act(async () => expect(await f.hook.result.current.pausePatrol()).toBe(true));
  expect(f.hook.result.current.navigation.patrolPaused).toBe(true);
  await waitFor(() => expect(f.hook.result.current.canRunMission).toBe(true), { timeout: 1500 });
  await startMission(f);
  const returning = await completeMission(f);
  expect(f.goals[returning].x).toBe(a.pose.x);
  await arrive(f, returning, a);
  await waitFor(() => expect(f.hook.result.current.navigation.dwelling).toBe(true), { timeout: 2500 });
  expect(f.goals.map(goal => goal.x)).toEqual([a.pose.x, a.pose.x]);
  expect(f.runtime.callService).toHaveBeenCalledTimes(2);
  expect(f.hook.result.current.owner).toBe("patrol");
  expect(f.props.startMission).toHaveBeenCalledTimes(1);
  await dispose(f);
}, 10000);

test("a child at the last one-way observation returns there without restarting the route", async () => {
  const f = fixture({ closed: false, dwell: { a: 0 } });
  await patrolAt(f, b);
  await startMission(f);
  const returning = await completeMission(f);
  await arrive(f, returning, b);
  await waitFor(() => expect(f.hook.result.current.navigation.dwelling).toBe(true), { timeout: 2000 });
  expect(f.goals.map(goal => goal.x)).toEqual([a.pose.x, b.pose.x, b.pose.x]);
  expect(f.hook.result.current.executionType).toBe("observation");
  expect(f.hook.result.current.isRunning).toBe(true);
  await dispose(f);
}, 10000);

test("a single-point loop resumes observing after return without a redundant navigation goal", async () => {
  const f = fixture({ points: [a] });
  await patrolAt(f);
  await startMission(f);
  const returning = await completeMission(f);
  await arrive(f, returning, a);
  await waitFor(() => expect(f.hook.result.current.owner).toBe("patrol"), { timeout: 2000 });
  expect(f.hook.result.current.navigation.dwelling).toBe(true);
  expect(f.hook.result.current.executionType).toBe("observation");
  expect(f.goals).toHaveLength(2);
  await dispose(f);
}, 10000);

test.each(["failed", "cancelled"])("%s child does not resume patrol or initiate return", async status => {
  const f = fixture();
  await patrolAt(f);
  await startMission(f);
  f.missionState(status);
  await waitFor(() => expect(f.hook.result.current.cycle.status).toBe(status));
  expect(f.goals).toHaveLength(1);
  expect(f.hook.result.current.navigation.isRunning).toBe(false);
  await dispose(f);
});

test("Stop during return clears queued patrol continuation even if navigation succeeds late", async () => {
  const f = fixture();
  await patrolAt(f);
  await startMission(f);
  const returning = await completeMission(f);
  await act(async () => f.hook.result.current.stop());
  await arrive(f, returning, a);
  expect(f.hook.result.current.cycle.status).toBe("cancelled");
  expect(f.hook.result.current.owner).toBeNull();
  expect(f.hook.result.current.navigation.isRunning).toBe(false);
  expect(f.goals).toHaveLength(2);
  f.hook.unmount();
});

test("a manually reached observation does not create patrol intent after a child returns", async () => {
  const f = fixture();
  await start(f, "goToSpot", a.id);
  await arrive(f, 0, a);
  await waitFor(() => expect(f.hook.result.current.observationReady).toBe(true), { timeout: 2000 });
  await startMission(f);
  const returning = await completeMission(f);
  await arrive(f, returning, a);
  await waitFor(() => expect(f.hook.result.current.cycle.status).toBe("done"), { timeout: 2000 });
  expect(f.goals).toHaveLength(2);
  expect(f.hook.result.current.navigation.isRunning).toBe(false);
  expect(f.hook.result.current.executionType).toBe("observation");
  await dispose(f);
}, 10000);


test("Stop during the deferred patrol refresh invalidates automatic continuation", async () => {
  const f = fixture();
  await patrolAt(f);
  await startMission(f);
  const returning = await completeMission(f);
  let resolveRefresh;
  f.props.manager.refresh.mockImplementationOnce(() => new Promise(resolve => { resolveRefresh = resolve; }));
  await arrive(f, returning, a);
  await waitFor(() => expect(resolveRefresh).toBeDefined(), { timeout: 2000 });
  await act(async () => f.hook.result.current.stop());
  await act(async () => resolveRefresh({ spots: f.props.manager.spots, plan: f.props.manager.configuration.plan }));
  expect(f.goals).toHaveLength(2);
  expect(f.hook.result.current.owner).toBeNull();
  expect(f.hook.result.current.navigation.isRunning).toBe(false);
  f.hook.unmount();
}, 10000);

test("a changed saved route blocks automatic continuation after mission return", async () => {
  const f = fixture();
  await patrolAt(f);
  await startMission(f);
  const returning = await completeMission(f);
  const changed = { ...f.props.manager.configuration.plan,
    observations: [...f.props.manager.configuration.plan.observations].reverse() };
  f.props.manager.refresh.mockResolvedValueOnce({ spots: f.props.manager.spots, plan: changed });
  await arrive(f, returning, a);
  await waitFor(() => expect(f.hook.result.current.navigation.status).toBe("failed"), { timeout: 2000 });
  expect(f.props.onMessage).toHaveBeenCalledWith(expect.stringContaining("route changed"));
  expect(f.goals).toHaveLength(2);
  expect(f.hook.result.current.navigation.isRunning).toBe(false);
  await dispose(f);
}, 10000);
