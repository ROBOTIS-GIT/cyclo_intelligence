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

import { sendInitialPoseEstimate, startNavigation, stopNavigation } from "../../../utils/navigationApi";
import { STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN } from "../lib/stages";
import { saveMissionSession } from "../lib/session";
import useRunNavigationLocalizationController from "./useRunNavigationLocalizationController";

export default function useWorkspaceLocalize({
  workspace, design, runMapBusy, runMapSnapshotInvalid, running, runtimeMapName, isPageExitStopSent, clearLocalizationPoseCache, waitForAutoLocalizedPose, runCommand
}) {
  const {
    workspaceStageRef, interactionMode, navigationRuntimeMode, setWorkspaceStage,
    setNavigationRuntimeMode, setRunPoseInitialized, setRunRuntimeOwned, setRunShutdownPending,
    setInteractionMode, setMessage, invalidateRunPoseRequestRef,
  } = workspace;
  const { mapName, setDesignPoseInitialized } = design;
  const runNavigationLocalization = useRunNavigationLocalizationController({
    getState: () => ({
      stage: workspaceStageRef.current,
      runStage: [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(workspaceStageRef.current) ? workspaceStageRef.current : STAGE_RUN,
      mapBusy: runMapBusy,
      snapshotInvalid: runMapSnapshotInvalid,
      interactionMode,
      running,
      runtimeMode: navigationRuntimeMode,
      targetMapName: runtimeMapName,
      sessionMapName: mapName,
    }),
    runtime: {
      start: startNavigation,
      stop: stopNavigation,
      pageExitStopSent: isPageExitStopSent,
    },
    commits: {
      setStage: setWorkspaceStage,
      setRuntimeMode: setNavigationRuntimeMode,
      setDesignPoseReady: setDesignPoseInitialized,
      setRunPoseReady: setRunPoseInitialized,
      setRuntimeOwned: setRunRuntimeOwned,
      setShutdownPending: setRunShutdownPending,
      setInteractionMode,
      persistClaim: saveMissionSession,
      persistRollback: saveMissionSession,
    },
    pose: {
      clearCache: clearLocalizationPoseCache,
      publish: sendInitialPoseEstimate,
      waitForConvergence: waitForAutoLocalizedPose,
    },
    runCommand,
    onMessage: setMessage,
  });

  const {
    localize: handleLocalize,
    estimatePose: handleRunPoseEstimate,
    invalidatePoseRequest: invalidateRunPoseRequest,
  } = runNavigationLocalization;

  invalidateRunPoseRequestRef.current = invalidateRunPoseRequest;

  return {
    handleLocalize, handleRunPoseEstimate,
  };
}
