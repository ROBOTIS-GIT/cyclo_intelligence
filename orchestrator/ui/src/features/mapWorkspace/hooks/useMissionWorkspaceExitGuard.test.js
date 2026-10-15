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

import { createRef } from "react";
import { act, renderHook } from "@testing-library/react";
import useMissionWorkspaceExitGuard, { missionWorkspaceExitBlockReason } from "./useMissionWorkspaceExitGuard";

const idleState = {
  busy: "",
  btNodeBusy: "",
  designMapBusy: false,
  runMapBusy: false,
  mapEditorBusy: false,
  mapEditorDirty: false,
  designMapEditorDirty: false,
  mappingRuntimeActive: false,
  runRuntimeActive: false,
  designLocalizationActive: false,
  navigationRuntimeMode: "idle",
  missionRunnerActive: false,
  runShutdownPending: false,
  navGoalDriving: false,
};

describe("missionWorkspaceExitBlockReason", () => {
  test("allows a clean idle workspace to exit", () => {
    expect(missionWorkspaceExitBlockReason(idleState)).toBe("");
  });

  test("prioritizes an in-flight operation over edits and runtimes", () => {
    expect(missionWorkspaceExitBlockReason({
      ...idleState,
      busy: "Save mission",
      mapEditorDirty: true,
      runRuntimeActive: true,
    })).toBe("Wait for the current operation to finish before going back");
  });

  test("prioritizes unsaved map edits over an active runtime", () => {
    expect(missionWorkspaceExitBlockReason({
      ...idleState,
      designMapEditorDirty: true,
      mappingRuntimeActive: true,
    })).toBe("Save the current map edits before going back");
  });

  test.each([
    { mappingRuntimeActive: true },
    { runRuntimeActive: true },
    { designLocalizationActive: true },
    { navigationRuntimeMode: "localization" },
    { missionRunnerActive: true },
    { runShutdownPending: true },
    { navGoalDriving: true },
  ])("blocks each active runtime condition: %o", (activeState) => {
    expect(missionWorkspaceExitBlockReason({
      ...idleState,
      ...activeState,
    })).toBe("Stop the active runtime before going back");
  });
});

describe("collection exit protection", () => {
  test.each([
    "Save or discard annotation changes before going back.",
    "Save the active episode before going back.",
  ])("prioritizes collection state over general operation and runtime guards: %s", collectionBlockReason => {
    expect(missionWorkspaceExitBlockReason({
      ...idleState, collectionBlockReason, busy: "Save mission", mapEditorDirty: true,
      runRuntimeActive: true,
    })).toBe(collectionBlockReason);
  });

  test("an empty collection reason preserves the existing guards", () => {
    expect(missionWorkspaceExitBlockReason({ ...idleState, collectionBlockReason: "" })).toBe("");
    expect(missionWorkspaceExitBlockReason({
      ...idleState, collectionBlockReason: "", runRuntimeActive: true,
    })).toBe("Stop the active runtime before going back");
  });

  test("publishes the collection lock to the shell and prevents Back until edits are saved", () => {
    const exitHandleRef = createRef();
    const onExitStateChange = jest.fn();
    const runGuardedDesignAction = jest.fn(action => action());
    const onExit = jest.fn();
    const collectionBlockReason = "Save or discard annotation changes before going back.";
    const { rerender } = renderHook(({ reason }) => useMissionWorkspaceExitGuard({
      ...idleState, collectionBlockReason: reason, exitHandleRef,
      onExitStateChange, runGuardedDesignAction,
    }), { initialProps: { reason: collectionBlockReason } });

    expect(onExitStateChange).toHaveBeenLastCalledWith({ blockReason: collectionBlockReason });
    expect(exitHandleRef.current.requestExit(onExit)).toBe(false);
    expect(runGuardedDesignAction).not.toHaveBeenCalled();
    expect(onExit).not.toHaveBeenCalled();

    rerender({ reason: "" });
    expect(onExitStateChange).toHaveBeenLastCalledWith({ blockReason: "" });
    act(() => { expect(exitHandleRef.current.requestExit(onExit)).toBe(true); });
    expect(runGuardedDesignAction).toHaveBeenCalledWith(onExit);
    expect(onExit).toHaveBeenCalledTimes(1);
  });
});
