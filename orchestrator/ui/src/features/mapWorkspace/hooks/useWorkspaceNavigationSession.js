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

import { useCallback } from "react";
import { getServiceStatus, stopNavigation } from "../../../utils/navigationApi";
import { readMissionSession, recentRunShutdownMarker, saveMissionSession } from "../lib/session";
import useNavigationRuntimeSessionController from "./useNavigationRuntimeSessionController";

export default function useWorkspaceNavigationSession({
  design, workspace, autonomyOperation, runMissionName, STATUS_POLL_MS, navigationRuntimeModeFromStatus, taskDisplayMessage
}) {
  const {
    mapName, designMapPath, designPoseInitialized, missionName, setDesignPoseInitialized,
    catalogRunCommandRef,
  } = design;
  const {
    workspaceStage, navigationRuntimeMode, runRuntimeOwned, runShutdownPending, status, setStatus,
    setNavigationRuntimeMode, setRunRuntimeOwned, invalidateRunPoseRequestRef, setRunPoseInitialized,
    setRunShutdownPending, setMessage, setBusy,
  } = workspace;
  const navigationRuntimeSession = useNavigationRuntimeSessionController({
    enabled: true,
    pollIntervalMs: STATUS_POLL_MS,
    state: {
      mapName,
      stage: workspaceStage,
      designMapPath,
      runtimeMode: navigationRuntimeMode,
      designPoseReady: designPoseInitialized,
      runtimeOwned: runRuntimeOwned,
      serverOwned: autonomyOperation.needsStop,
      shutdownPending: runShutdownPending,
      missionName,
      runMissionName,
      status,
    },
    status: {
      get: getServiceStatus,
      modeOf: navigationRuntimeModeFromStatus,
    },
    session: {
      read: readMissionSession,
      save: saveMissionSession,
      recentShutdownMarker: recentRunShutdownMarker,
    },
    runtime: {
      stop: stopNavigation,
    },
    commits: {
      setStatus,
      setRuntimeMode: setNavigationRuntimeMode,
      setDesignPoseReady: setDesignPoseInitialized,
      setRuntimeOwned: setRunRuntimeOwned,
      confirmStopped: () => {
        invalidateRunPoseRequestRef.current();
        setStatus({ is_up: false, mode: "idle" });
        setNavigationRuntimeMode("idle");
        setDesignPoseInitialized(false);
        setRunPoseInitialized(false);
        setRunRuntimeOwned(false);
        setRunShutdownPending(false);
      },
    },
    onMessage: setMessage,
  });

  const {
    refreshStatus: loadStatus,
    invalidateStatus: invalidateNavigationStatus,
    isPageExitStopSent,
  } = navigationRuntimeSession;

  const runCommand = useCallback(async (label, action) => {
    invalidateNavigationStatus();
    setBusy(label);
    try {
      const result = await action();
      if (result?.stale || result?.skipped) return result;
      if (typeof result === "string") {
        setMessage(taskDisplayMessage(result));
      } else {
        setMessage(taskDisplayMessage(result?.message) || `${label} complete`);
      }
      return result;
    } catch (error) {
      setMessage(error instanceof Error ? taskDisplayMessage(error.message) : `${label} failed`);
    } finally {
      setBusy("");
      void loadStatus();
    }
  }, [invalidateNavigationStatus, loadStatus, setBusy, setMessage, taskDisplayMessage]);

  catalogRunCommandRef.current = runCommand;

  return {
    invalidateNavigationStatus, isPageExitStopSent, runCommand,
  };
}
