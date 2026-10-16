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

import buildAutonomyScene from "./buildAutonomyScene";
import buildAutonomyHud from "./buildAutonomyHud";
import buildAutonomyChrome from "./buildAutonomyChrome";
import { STAGE_COLLECT, STAGE_OPERATE } from "../../../../mapWorkspace/lib/stages";

function model() {
  return {
    workspace: { workspaceStage: STAGE_COLLECT, missionMapLoaded: true, runPoseInitialized: true,
      setInteractionMode: jest.fn(), setMessage: jest.fn() },
    recordData: { recordAuthoringBusy: false, collectDraft: {}, qwenObservation: { source: "preview" } },
    runtime: {
      running: true, runtimeMapName: "house", collectWaypoints: {},
      collectObservations: { spots: [{ id: "a" }, { id: "b" }], selectedId: "a", select: jest.fn(),
        move: jest.fn(), route: { routeMode: false, orderMode: false, handleSpotClick: jest.fn() },
        configuration: { plan: { observations: [{ id: "a" }] } } },
      collectExecution: { cycle: { isRunning: false }, navigation: { isRunning: false, toSpotId: "b" },
        missionRunner: { isRunning: false }, canRunMission: true, observationReady: true,
        startPatrol: jest.fn(), goToSpot: jest.fn(), stop: jest.fn() },
      collectMission: { loading: false, select: jest.fn() },
      autonomyOperation: { source: "operate" },
      autonomyMap: { spots: [{ id: "operating" }], activeWaypointId: "operating", navigationActive: true, route: { order: ["operating"] } },
      openRecordMapPicker: jest.fn(), handleOpenRunMapDialog: jest.fn(),
    },
  };
}

test("selected Record waypoint remains distinct from the currently executing destination", () => {
  const input = model();
  let points = buildAutonomyScene(input).mission.collectWaypoints;
  expect(points.selectedId).toBe("a");
  expect(points.activeWaypointId).toBeNull();
  input.runtime.collectExecution.navigation.isRunning = true;
  points = buildAutonomyScene(input).mission.collectWaypoints;
  expect(points.activeWaypointId).toBe("b");
  input.runtime.collectExecution.returning = true;
  input.runtime.collectExecution.cycle.pinnedSpot = { id: "return-origin" };
  expect(buildAutonomyScene(input).mission.collectWaypoints.activeWaypointId).toBe("return-origin");
});

test("route order clicks and idle selection retain separate semantics", () => {
  const input = model();
  let points = buildAutonomyScene(input).mission.collectWaypoints;
  points.onSelect("b"); points.onClearSelection();
  expect(input.runtime.collectObservations.select.mock.calls).toEqual([["b"], [""]]);
  input.runtime.collectObservations.route.routeMode = true;
  points = buildAutonomyScene(input).mission.collectWaypoints;
  points.onSelect("b");
  expect(input.runtime.collectObservations.route.handleSpotClick).toHaveBeenCalledWith("b");
  input.runtime.collectObservations.draft = { id: "a", label: "Draft", pose: { x: 1 } };
  points = buildAutonomyScene(input).mission.collectWaypoints;
  points.onSelect("b"); points.onClearSelection();
  expect(input.runtime.collectObservations.select).toHaveBeenCalledTimes(2);
  expect(points.spots).toEqual([{ id: "b", poseEditable: false }, { id: "a", label: "Draft", pose: { x: 1 } }]);
});

test("waypoint dragging is available only in an unlocked authoring mode", () => {
  const input = model();
  expect(buildAutonomyScene(input).mission.collectWaypoints.onPoseChange).toBeUndefined();
  input.runtime.collectObservations.route.routeMode = true;
  expect(buildAutonomyScene(input).mission.collectWaypoints.onPoseChange).toBe(input.runtime.collectObservations.move);
  input.runtime.collectObservations.route.orderMode = true;
  expect(buildAutonomyScene(input).mission.collectWaypoints.onPoseChange).toBeUndefined();
  input.runtime.collectObservations.route.orderMode = false;
  input.recordData.recordAuthoringBusy = true;
  expect(buildAutonomyScene(input).mission.collectWaypoints.onPoseChange).toBeUndefined();
});

test("Operate alone overrides mission visualization and follows the autonomy destination", () => {
  const input = model();
  expect(buildAutonomyScene(input).mission).not.toHaveProperty("followRobot");
  input.workspace.workspaceStage = STAGE_OPERATE;
  expect(buildAutonomyScene(input).mission).toMatchObject({
    overlayActive: true, renderedVisibleSpots: [{ id: "operating" }], activeWaypointId: "operating",
    followRobot: true, renderedBehaviorNodes: [], renderedRouteView: { order: ["operating"] },
  });
});

test("capture state locks authoring but never creates a prerequisite for starting patrol", () => {
  const input = model();
  input.recordData.collectDraft = { active: true, recording: true, busy: true };
  const hud = buildAutonomyHud(input);
  expect(hud.collect.authoringDisabled).toBe(true);
  expect(hud.collect.startDisabled).toBeFalsy();
  hud.collect.onStart();
  expect(input.workspace.setInteractionMode).toHaveBeenCalledWith("view");
  expect(input.runtime.collectExecution.startPatrol).toHaveBeenCalledTimes(1);
  hud.collect.onGoToObservation("b");
  expect(input.runtime.collectExecution.goToSpot).toHaveBeenCalledWith("b");
  expect(hud.runObservation).toBe(input.recordData.qwenObservation);
  input.workspace.workspaceStage = STAGE_OPERATE;
  expect(buildAutonomyHud(input).runObservation).toBe(input.runtime.autonomyOperation);
});

test("Record map switching stays blocked by execution and uncommitted route edits", () => {
  const input = model();
  expect(buildAutonomyChrome(input).header.onOpenRunMap).toBe(input.runtime.openRecordMapPicker);
  input.runtime.collectObservations.routeEditor = { dirty: true };
  expect(buildAutonomyChrome(input).header.recordMapChangeBlocked).toBe(true);
  input.runtime.collectObservations.routeEditor.dirty = false;
  input.runtime.collectExecution.stopping = true;
  expect(buildAutonomyChrome(input).header.recordMapChangeBlocked).toBe(true);
  input.workspace.workspaceStage = STAGE_OPERATE;
  expect(buildAutonomyChrome(input).header.onOpenRunMap).toBe(input.runtime.handleOpenRunMapDialog);
});
