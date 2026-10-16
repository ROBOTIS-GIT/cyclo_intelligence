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
import useAutonomyWorkspaceRuntime from "./useAutonomyWorkspaceRuntime";
import useAutonomyOperation from "./useAutonomyOperation";
import useRecordObservationSpots from "./useRecordObservationSpots";
import useRecordExecution from "./useRecordExecution";
import useRecordMissionOverlay from "./useRecordMissionOverlay";
import { useAutonomyMapView } from "../components/autonomyMapView";

jest.mock("./useAutonomyOperation");
jest.mock("./useRecordObservationSpots");
jest.mock("./useRecordExecution");
jest.mock("./useRecordMissionOverlay");
jest.mock("../components/autonomyMapView");

function inputs() {
  return {
    workspaceStage: "collect", missionMapLoaded: true, running: true,
    runPoseInitialized: true, runDisplayMapEditor: { busy: false }, runShutdownPending: false,
    runtimeMapName: "house", missionRunner: { isRunning: false, activeSpotId: "table" }, busy: false,
    runMissionSnapshot: { missionId: "clean" }, currentPose: { x: 1 },
    setInteractionMode: jest.fn(), runStartRequestRef: { current: 7 }, setPendingRunStart: jest.fn(),
    sendMissionGoal: jest.fn(), cancelNavigateToPoseGoal: jest.fn(), setMessage: jest.fn(),
    btStatusRef: { current: null }, btStatusHistoryRef: { current: [] }, callService: jest.fn(),
    stopMissionBt: jest.fn(), ensureMissionBtActive: jest.fn(), releaseMissionBt: jest.fn(),
    missionRunnerFlags: jest.fn(), runRouteView: {},
  };
}
const latest = mock => mock.mock.calls.at(-1)[0];

beforeEach(() => {
  jest.clearAllMocks();
  useAutonomyOperation.mockReturnValue({ needsStop: false });
  useRecordObservationSpots.mockReturnValue({ spots: [], selectedId: "observation" });
  useRecordExecution.mockReturnValue({
    isRunning: false, stopping: false, cycle: {}, navigation: { isRunning: false, dwelling: false },
    goToSpot: jest.fn(), stop: jest.fn(),
  });
  useRecordMissionOverlay.mockReturnValue({ points: [] });
  useAutonomyMapView.mockReturnValue({ spots: [] });
});

test("Record and Operate switch enablement without replacing navigation or BT ownership", () => {
  const props = inputs();
  const hook = renderHook(useAutonomyWorkspaceRuntime, { initialProps: props });
  expect(latest(useAutonomyOperation).enabled).toBe(false);
  expect(latest(useRecordExecution)).toMatchObject({ enabled: true, missionRunner: props.missionRunner });
  expect(latest(useRecordExecution).runtime).toEqual({
    btStatusRef: props.btStatusRef, btStatusHistoryRef: props.btStatusHistoryRef, callService: props.callService,
    stopBt: props.stopMissionBt, ensureBtActive: props.ensureMissionBtActive,
    releaseBt: props.releaseMissionBt, getFlags: props.missionRunnerFlags,
  });
  hook.rerender({ ...props, workspaceStage: "operate" });
  expect(latest(useAutonomyOperation).enabled).toBe(true);
  expect(latest(useRecordExecution).enabled).toBe(false);
  expect(latest(useRecordObservationSpots).enabled).toBe(false);
  expect(useRecordExecution).toHaveBeenCalledTimes(2);
  hook.rerender({ ...props, workspaceStage: "operate", runShutdownPending: true });
  expect(latest(useAutonomyOperation).enabled).toBe(false);
});

test("pending mission cancellation and authoring locks use the host's live refs", () => {
  const props = inputs();
  const hook = renderHook(useAutonomyWorkspaceRuntime, { initialProps: props });
  const start = latest(useRecordExecution).startMission;
  expect(start()).toBeUndefined();
  const first = jest.fn(() => "first"), second = jest.fn(() => "second");
  hook.result.current.recordMissionStartRef.current = first;
  expect(start()).toBe("first");
  const authoringLock = hook.result.current.recordAuthoringLock;
  authoringLock.current = true;
  expect(latest(useRecordObservationSpots).isBlocked()).toBe(true);
  hook.rerender({ ...props, workspaceStage: "operate" });
  expect(hook.result.current.recordAuthoringLock).toBe(authoringLock);
  hook.result.current.recordMissionStartRef.current = second;
  expect(start()).toBe("second");
  latest(useRecordExecution).cancelMissionStart();
  expect(props.runStartRequestRef.current).toBe(8);
  expect(props.setPendingRunStart).toHaveBeenCalledWith(null);
});

test("autonomy shutdown locks authoring and dwelling does not count as navigation motion", () => {
  const props = inputs();
  useAutonomyOperation.mockReturnValue({ needsStop: true });
  const execution = {
    isRunning: true, stopping: false, cycle: {}, navigation: { isRunning: true, dwelling: true },
    goToSpot: jest.fn(), stop: jest.fn(),
  };
  useRecordExecution.mockReturnValue(execution);
  const hook = renderHook(useAutonomyWorkspaceRuntime, { initialProps: props });
  expect(latest(useRecordObservationSpots).blocked).toBe(true);
  expect(hook.result.current.missionRunnerActive).toBe(true);
  expect(hook.result.current.recordNavigationMoving).toBe(false);
  expect(hook.result.current.collectWaypoints.isMoving()).toBe(false);
  expect(hook.result.current.collectWaypoints.navigate).toBe(execution.goToSpot);
  expect(hook.result.current.collectWaypoints.stop).toBe(execution.stop);
});
