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

import { buildRecordExecutionState, recordExecutionType, recordNavigationActive } from "./recordExecutionState";

const navigating = { isRunning: true, phase: "awaiting-nav-result", status: "navigating", toSpotId: "dining", currentIndex: 0 };
const record = changes => buildRecordExecutionState({ mapName: "house", planRevision: 7, ...changes });

test("a selected editor marker does not establish observation or ownership", () => {
  const state = record({ selectedId: "dining", manager: { selectedId: "dining" } });
  expect(state).toMatchObject({ schema_version: 1, source: "record", authority: "browser",
    session_id: null, epoch: null, revision: null, lifecycle: "idle", activity: "idle", control: "idle",
    owns_execution: false, observation_spot_id: null, target_spot_id: null, mission_id: null });
  const reached = record({ observationReady: true, observationSpotId: "dining", selectedId: "laundry" });
  expect(reached).toMatchObject({ lifecycle: "idle", activity: "observing", owns_execution: false, observation_spot_id: "dining" });
  expect(recordExecutionType(reached)).toBe("observation");
});

test("patrol navigation and posture describe distinct active operations", () => {
  const state = record({ navigation: navigating });
  expect(state).toMatchObject({ lifecycle: "active", activity: "patrol_navigation", control: "running",
    owns_execution: true, target_spot_id: "dining", step_index: 0 });
  expect(recordNavigationActive(state)).toBe(true);
  const posture = record({ navigation: { ...navigating, phase: "bt-running", status: "running-bt" } });
  expect(posture.activity).toBe("observation_posture");
  expect(recordNavigationActive(posture)).toBe(false);
  const preparing = record({ navigation: { ...navigating, phase: "preparing", status: "preparing" } });
  expect(preparing.lifecycle).toBe("starting");
  expect(recordNavigationActive(preparing)).toBe(false);
});

test("a pause request remains pausing until cancellation acknowledges", () => {
  const retainedExecution = record({ navigation: navigating });
  const navigation = { ...navigating, isRunning: false, phase: "paused", status: "paused", paused: true, held: true, patrolPaused: true };
  const request = record({ navigation, stopping: true, pendingControl: { kind: "pausing" }, retainedExecution });
  expect(request).toMatchObject({ lifecycle: "active", activity: "patrol_navigation", control: "pausing", owns_execution: true });
  expect(recordNavigationActive(request)).toBe(false);
  const acknowledged = record({ navigation });
  expect(acknowledged).toMatchObject({ lifecycle: "active", control: "paused", pause_causes: { manual: "Operator pause" } });
  expect(record({ navigation, pendingControl: { kind: "resuming" } }).control).toBe("resuming");
});

test("an unconfirmed Stop retains execution context and ownership until acknowledged", () => {
  const retainedExecution = record({ navigation: navigating });
  const navigation = { ...navigating, isRunning: false, phase: "cancelled", status: "paused", held: true, paused: true };
  const failure = record({ navigation, cancelFailed: true, controlError: "cancel unavailable", retainedExecution });
  expect(failure).toMatchObject({ lifecycle: "failed", control: "cancelling", owns_execution: true,
    activity: "patrol_navigation", target_spot_id: "dining", error: "cancel unavailable" });
  expect(recordExecutionType(failure, true)).toBeNull();
  expect(recordNavigationActive(failure)).toBe(false);
  const stopped = record({ navigation });
  expect(stopped).toMatchObject({ lifecycle: "stopped", control: "idle", activity: "idle", owns_execution: false, error: null });
});

test("child navigation uses the dispatched mission and target, not editor selection", () => {
  const state = record({ owner: "mission", mission: { missionId: "edited" },
    cycle: { isRunning: true, cyclePhase: "mission", phase: "awaiting-nav-result", currentIndex: 2,
      pinnedMissionId: "clean", pinnedSpot: { id: "dining" } },
    missionRunner: { isRunning: true, phase: "awaiting-nav-result", activeSpotId: "table", currentIndex: 2 } });
  expect(state).toMatchObject({ activity: "mission_navigation", mission_id: "clean", target_spot_id: "table", step_index: 2,
    return_checkpoint: { return_spot_id: "dining", plan_revision: 7 } });
  expect(recordNavigationActive(state)).toBe(true);
});

test.each([
  ["return-pending", "none", "starting", "return_navigation", false],
  ["return-starting", "none", "starting", "return_navigation", false],
  ["returning", "awaiting-nav-result", "active", "return_navigation", true],
  ["returning", "bt-running", "active", "return_posture", false],
  ["return-settling", "settling", "active", "return_posture", false],
])("return %s/%s keeps the child mission and pinned observation", (cyclePhase, phase, lifecycle, activity, moving) => {
  const state = record({ owner: "mission", cycle: { isRunning: true, cyclePhase, phase,
    pinnedMissionId: "clean", pinnedSpot: { id: "dining" }, currentIndex: 0 },
    patrolContinuation: { route: [{ id: "dining" }, { id: "laundry" }], index: 0, closed: false, dwellRemainingMs: 4321 } });
  expect(state).toMatchObject({ lifecycle, activity, mission_id: "clean", target_spot_id: "dining",
    return_checkpoint: { return_spot_id: "dining", next_spot_id: "laundry", route_index: 0, dwell_remaining_ms: 4321 } });
  expect(recordExecutionType(state)).toBe("patrol");
  expect(recordNavigationActive(state)).toBe(moving);
});

test("detached capture sessions and capture flags never establish motion or ownership", () => {
  const inputs = { observationReady: true, observationSpotId: "dining" };
  const state = record(inputs);
  expect(record({ ...inputs, capture: { active: true, paused: true, detached: true }, recording: true,
    episode: { session_id: "old-session", mission_id: "old-mission", phase: "mission" } })).toEqual(state);
  expect(state).toMatchObject({ owns_execution: false, mission_id: null, return_checkpoint: null, control: "idle" });
});


test("navigation completion before pose settling does not revive a motion arrow", () => {
  const state = record({ navigation: { ...navigating, phase: "none", status: "done" } });
  expect(state.activity).toBe("observation_posture");
  expect(recordNavigationActive(state)).toBe(false);
  const returning = record({ cycle: { isRunning: true, cyclePhase: "returning", phase: "none", pinnedSpot: { id: "dining" } } });
  expect(returning.activity).toBe("return_posture");
  expect(recordNavigationActive(returning)).toBe(false);
});

test("historical pause errors cannot reacquire ownership after an acknowledged Stop", () => {
  const state = record({ navigation: { isRunning: false, phase: "cancelled", status: "paused", held: true, pauseFailed: true } });
  expect(state).toMatchObject({ lifecycle: "stopped", control: "idle", owns_execution: false, error: null });
});
