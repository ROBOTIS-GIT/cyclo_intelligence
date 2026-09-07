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

import {
  DEFAULT_RUNNER_CONFIG,
  RunnerPhase,
  RunnerStatus,
  WaypointState,
  goalFromSpot,
  initialRunnerState,
  isEmptyBt,
  isRunnerActive,
  missionRunnerReducer,
  navigationBatchFromIndex,
  requiresBackendTaskTimeout,
} from "./missionRunnerCore";


const SPOTS = [
  { id: "a", label: "Dock", pose: { x: 0, y: 0, yaw: 0 } },
  { id: "b", label: "Bay", pose: { x: 5, y: 0, yaw: Math.PI / 2 } },
];

describe("goalFromSpot", () => {
  test("reads the pose sent to Nav2", () => {
    expect(goalFromSpot(SPOTS[1])).toEqual({ x: 5, y: 0, yaw: Math.PI / 2 });
  });
});
describe("isEmptyBt", () => {
  const emptyXml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"/>',
    "</root>",
  ].join("\n");
  const filledXml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait duration="1.0"/></BehaviorTree>',
    "</root>",
  ].join("\n");

  test("blank / whitespace is empty", () => {
    expect(isEmptyBt("")).toBe(true);
    expect(isEmptyBt("   \n ")).toBe(true);
  });

  test("childless MainTree is empty", () => {
    expect(isEmptyBt(emptyXml)).toBe(true);
  });

  test("MainTree with a child is not empty", () => {
    expect(isEmptyBt(filledXml)).toBe(false);
  });

  test("unparseable XML is treated as non-empty so the error surfaces", () => {
    expect(isEmptyBt("<root><unclosed></root>")).toBe(false);
  });
});

describe("requiresBackendTaskTimeout", () => {
  const treeWith = (attributes) => [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    `  <BehaviorTree ID="MainTree"><SendCommand ${attributes}/></BehaviorTree>`,
    "</root>",
  ].join("\n");

  test("extends legacy and explicit inference LOAD tasks", () => {
    expect(requiresBackendTaskTimeout(treeWith('command="LOAD" model="groot"'))).toBe(true);
    expect(requiresBackendTaskTimeout(
      treeWith('target="INFERENCE" command="LOAD" model="lerobot"'),
    )).toBe(true);
  });

  test("extends Docker START and RESTART tasks", () => {
    expect(requiresBackendTaskTimeout(
      treeWith('target="DOCKER" command="START" model="groot"'),
    )).toBe(true);
    expect(requiresBackendTaskTimeout(
      treeWith('target="docker" command="restart" model="lerobot"'),
    )).toBe(true);
  });

  test("keeps regular and non-provisioning tasks on the normal timeout", () => {
    expect(requiresBackendTaskTimeout(treeWith('target="INFERENCE" command="STOP"'))).toBe(false);
    expect(requiresBackendTaskTimeout(treeWith('target="DOCKER" command="STOP"'))).toBe(false);
    expect(requiresBackendTaskTimeout('<root><BehaviorTree ID="MainTree"><Wait/></BehaviorTree></root>')).toBe(false);
    expect(requiresBackendTaskTimeout("<broken>")).toBe(false);
  });
});

describe("navigationBatchFromIndex", () => {
  const empty = '<root><BehaviorTree ID="MainTree"/></root>';
  const filled = '<root><BehaviorTree ID="MainTree"><Wait/></BehaviorTree></root>';
  const spots = ["a", "b", "c", "d"].map((id) => ({ id }));

  test("includes up to two empty waypoints and the following BT endpoint", () => {
    const xmlById = { a: empty, b: empty, c: filled, d: filled };
    expect(navigationBatchFromIndex(
      spots,
      0,
      (spot) => xmlById[spot.id],
    )).toEqual({ indices: [0, 1, 2], useThroughPoses: true });
  });

  test("splits more than two consecutive empty waypoints", () => {
    const xmlById = { a: empty, b: empty, c: empty, d: filled };
    expect(navigationBatchFromIndex(
      spots,
      0,
      (spot) => xmlById[spot.id],
    )).toEqual({ indices: [0, 1], useThroughPoses: true });
    expect(navigationBatchFromIndex(
      spots,
      2,
      (spot) => xmlById[spot.id],
    )).toEqual({ indices: [2, 3], useThroughPoses: true });
  });

  test("keeps a BT waypoint and a lone trailing empty waypoint single", () => {
    const xmlById = { a: filled, b: empty };
    expect(navigationBatchFromIndex(
      spots.slice(0, 2),
      0,
      (spot) => xmlById[spot.id],
    )).toEqual({ indices: [0], useThroughPoses: false });
    expect(navigationBatchFromIndex(
      spots.slice(0, 2),
      1,
      (spot) => xmlById[spot.id],
    )).toEqual({ indices: [1], useThroughPoses: false });
  });

  test("treats malformed XML as a BT endpoint", () => {
    const xmlById = { a: empty, b: "<root><broken></root>" };
    expect(navigationBatchFromIndex(
      spots.slice(0, 2),
      0,
      (spot) => xmlById[spot.id],
    )).toEqual({ indices: [0, 1], useThroughPoses: true });
  });
});

describe("missionRunnerReducer", () => {
  test("initial state marks every waypoint pending", () => {
    const state = initialRunnerState(SPOTS);
    expect(state.status).toBe(RunnerStatus.IDLE);
    expect(state.total).toBe(2);
    expect(state.progress.map((p) => p.state)).toEqual([
      WaypointState.PENDING,
      WaypointState.PENDING,
    ]);
  });

  test("happy path: start → navigate → runBt → finish → advance → done", () => {
    let state = initialRunnerState(SPOTS);
    state = missionRunnerReducer(state, { type: "start" });
    expect(state.status).toBe(RunnerStatus.STARTING);

    state = missionRunnerReducer(state, { type: "navigate", index: 0 });
    expect(state.status).toBe(RunnerStatus.NAVIGATING);
    expect(state.currentIndex).toBe(0);
    expect(state.progress[0].state).toBe(WaypointState.NAVIGATING);

    state = missionRunnerReducer(state, { type: "phase", phase: RunnerPhase.ARRIVED });
    expect(state.phase).toBe(RunnerPhase.ARRIVED);

    state = missionRunnerReducer(state, { type: "runBt", index: 0 });
    expect(state.status).toBe(RunnerStatus.RUNNING_BT);
    expect(state.progress[0].state).toBe(WaypointState.RUNNING_BT);

    state = missionRunnerReducer(state, { type: "finish", index: 0, skipped: false });
    expect(state.progress[0].state).toBe(WaypointState.DONE);

    state = missionRunnerReducer(state, { type: "advance" });
    expect(state.status).toBe(RunnerStatus.ADVANCING);

    state = missionRunnerReducer(state, { type: "navigate", index: 1 });
    state = missionRunnerReducer(state, { type: "finish", index: 1, skipped: true });
    expect(state.progress[1].state).toBe(WaypointState.SKIPPED);

    state = missionRunnerReducer(state, { type: "done" });
    expect(state.status).toBe(RunnerStatus.DONE);
    expect(state.currentIndex).toBe(-1);
  });

  test("fail marks the active waypoint failed and records a reason", () => {
    let state = initialRunnerState(SPOTS);
    state = missionRunnerReducer(state, { type: "navigate", index: 1 });
    state = missionRunnerReducer(state, { type: "fail", reason: "nav timeout at Bay", index: 1 });
    expect(state.status).toBe(RunnerStatus.FAILED);
    expect(state.reason).toBe("nav timeout at Bay");
    expect(state.progress[1].state).toBe(WaypointState.FAILED);
  });

  test("cancel rolls the active waypoint back to pending", () => {
    let state = initialRunnerState(SPOTS);
    state = missionRunnerReducer(state, { type: "navigate", index: 0 });
    state = missionRunnerReducer(state, { type: "cancel" });
    expect(state.status).toBe(RunnerStatus.CANCELLED);
    expect(state.currentIndex).toBe(-1);
    expect(state.activeIndices).toEqual([]);
    expect(state.progress[0].state).toBe(WaypointState.PENDING);
  });

  test("group navigation updates, fails, and cancels every active waypoint", () => {
    let state = initialRunnerState(SPOTS);
    state = missionRunnerReducer(state, {
      type: "navigate",
      index: 0,
      indices: [0, 1],
    });
    expect(state.progress.map((entry) => entry.state)).toEqual([
      WaypointState.NAVIGATING,
      WaypointState.NAVIGATING,
    ]);

    const failed = missionRunnerReducer(state, {
      type: "fail",
      reason: "batch aborted",
      index: 0,
      indices: [0, 1],
    });
    expect(failed.progress.map((entry) => entry.state)).toEqual([
      WaypointState.FAILED,
      WaypointState.FAILED,
    ]);

    const cancelled = missionRunnerReducer(state, { type: "cancel" });
    expect(cancelled.progress.map((entry) => entry.state)).toEqual([
      WaypointState.PENDING,
      WaypointState.PENDING,
    ]);
  });

  test("start resets progress after a prior failed run", () => {
    let state = initialRunnerState(SPOTS);
    state = missionRunnerReducer(state, { type: "navigate", index: 0 });
    state = missionRunnerReducer(state, { type: "fail", reason: "x", index: 0 });
    state = missionRunnerReducer(state, { type: "start" });
    expect(state.progress.every((p) => p.state === WaypointState.PENDING)).toBe(true);
    expect(state.reason).toBe("");
  });
});

describe("isRunnerActive", () => {
  test("true only for in-flight statuses", () => {
    expect(isRunnerActive(RunnerStatus.NAVIGATING)).toBe(true);
    expect(isRunnerActive(RunnerStatus.RUNNING_BT)).toBe(true);
    expect(isRunnerActive(RunnerStatus.IDLE)).toBe(false);
    expect(isRunnerActive(RunnerStatus.DONE)).toBe(false);
  });
});

test("DEFAULT_RUNNER_CONFIG only contains BT polling timeouts", () => {
  expect(DEFAULT_RUNNER_CONFIG).toEqual({
    btStartTimeoutMs: 5000,
    btTimeoutMs: 300000,
    backendTaskTimeoutMs: 23520000,
    pollMs: 250,
  });
});
