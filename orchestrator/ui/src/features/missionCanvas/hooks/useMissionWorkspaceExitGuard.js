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

import { useImperativeHandle, useLayoutEffect } from "react";

export function missionWorkspaceExitBlockReason({
  busy,
  btNodeBusy,
  designMapBusy,
  runMapBusy,
  mapEditorBusy,
  mapEditorDirty,
  designMapEditorDirty,
  mappingRuntimeActive,
  runRuntimeActive,
  designLocalizationActive,
  navigationRuntimeMode,
  missionRunnerActive,
  runShutdownPending,
  navGoalDriving,
}) {
  if (busy || btNodeBusy || designMapBusy || runMapBusy || mapEditorBusy) {
    return "Wait for the current operation to finish before going back";
  }
  if (mapEditorDirty || designMapEditorDirty) {
    return "Save the current map edits before going back";
  }
  if (
    mappingRuntimeActive
    || runRuntimeActive
    || designLocalizationActive
    || navigationRuntimeMode !== "idle"
    || missionRunnerActive
    || runShutdownPending
    || navGoalDriving
  ) {
    return "Stop the active runtime before going back";
  }
  return "";
}

export default function useMissionWorkspaceExitGuard({
  busy,
  btNodeBusy,
  designMapBusy,
  runMapBusy,
  mapEditorBusy,
  mapEditorDirty,
  designMapEditorDirty,
  mappingRuntimeActive,
  runRuntimeActive,
  designLocalizationActive,
  navigationRuntimeMode,
  missionRunnerActive,
  runShutdownPending,
  navGoalDriving,
  onExitStateChange,
  exitHandleRef,
  runGuardedDesignAction,
}) {
  const blockReason = missionWorkspaceExitBlockReason({
    busy,
    btNodeBusy,
    designMapBusy,
    runMapBusy,
    mapEditorBusy,
    mapEditorDirty,
    designMapEditorDirty,
    mappingRuntimeActive,
    runRuntimeActive,
    designLocalizationActive,
    navigationRuntimeMode,
    missionRunnerActive,
    runShutdownPending,
    navGoalDriving,
  });

  // The shell renders the app bar, so publish the block reason before paint:
  // the back button must already show it when the next click lands.
  useLayoutEffect(() => {
    if (typeof onExitStateChange === "function") {
      onExitStateChange({ blockReason });
    }
  }, [blockReason, onExitStateChange]);

  // The unsaved-Design dialog is owned by the workspace. Expose a request
  // instead of letting the shell unmount it directly, so a clean exit proceeds
  // immediately while a dirty Design can suspend the callback behind its guard.
  useImperativeHandle(exitHandleRef, () => ({
    requestExit(onExit) {
      if (blockReason) return false;
      runGuardedDesignAction(onExit);
      return true;
    },
  }), [blockReason, runGuardedDesignAction]);

  return blockReason;
}
