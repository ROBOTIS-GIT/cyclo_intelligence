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
import { useMissionRunner } from "../../../hooks/useMissionRunner";
import { cancelNavigateToPoseGoal, sendNavigateThroughPosesGoalsAndWait, sendNavigateToPoseGoalAndWait } from "../../../utils/navigationApi";
import { orientationFromYaw } from "../../../utils/navigationTf";
import { rosTimestampNow } from "../../../utils/rosTime";
import { STAGE_COLLECT } from "../lib/stages";
import { localBtPathForSpot } from "../lib/missionBtFiles";
import useNavigateGoalController from "./useNavigateGoalController";

export default function useWorkspaceMissionRunner({
  workspace, runSpots, runMissionBtFiles, callService, running, btNodeIsUp, runRouteView, btStatusRef, btStatusHistoryRef, ensureMissionBtActive, releaseMissionBt, taskDisplayMessage
}) {
  const { setInteractionMode, setMessage, workspaceStage } = workspace;
  const resolveMissionBtXml = useCallback(
    (spot) => (spot ? runMissionBtFiles[localBtPathForSpot(spot)] : ""),
    [runMissionBtFiles],
  );

  const sendMissionGoal = useCallback(async (x, y, yaw, signal) => {
    const poseStamped = {
      header: { frame_id: "map", stamp: rosTimestampNow() },
      pose: { position: { x, y, z: 0 }, orientation: orientationFromYaw(yaw) },
    };
    return sendNavigateToPoseGoalAndWait({ pose: poseStamped }, signal);
  }, []);

  const sendMissionGoals = useCallback(async (goals, signal) => {
    const poses = goals.map(({ x, y, yaw }) => ({
      header: { frame_id: "map", stamp: rosTimestampNow() },
      pose: { position: { x, y, z: 0 }, orientation: orientationFromYaw(yaw) },
    }));
    return sendNavigateThroughPosesGoalsAndWait({ poses }, signal);
  }, []);

  const {
    goalPose: navGoalPose,
    goalStatus: navGoalStatus,
    driving: navGoalDriving,
    sendGoal: handleSendNavGoal,
    invalidateGoal: invalidateNavGoal,
  } = useNavigateGoalController({
    sendGoalAndWait: sendNavigateToPoseGoalAndWait,
    onDisarm: () => setInteractionMode("view"),
    onMessage: setMessage,
  });

  const stopMissionBt = useCallback(async () => {
    const result = await callService(
      "/bt/set_running",
      "std_srvs/srv/SetBool",
      { data: false },
    );
    if (result?.success === false) {
      throw new Error(taskDisplayMessage(result.message) || "Task Engine stop rejected");
    }
    return result;
  }, [callService, taskDisplayMessage]);

  const missionRunnerFlags = useCallback(
    () => ({ navRunning: running, btNodeIsUp }),
    [btNodeIsUp, running],
  );

  const missionRunner = useMissionRunner({
    // A closed route intentionally contains the start waypoint a second time
    // as its final execution step. Keep the unique list for badges/editing,
    // but give the runner the full traversal so last -> start is sent to Nav2.
    orderedSpots: runRouteView.executionSpots,
    missionWaypoints: runSpots,
    pausableNavigation: workspaceStage === STAGE_COLLECT,
    resolveBtXml: resolveMissionBtXml,
    btStatusRef,
    btStatusHistoryRef,
    callService,
    sendGoal: sendMissionGoal,
    sendGoals: sendMissionGoals,
    cancelGoal: cancelNavigateToPoseGoal,
    stopBt: stopMissionBt,
    ensureBtActive: ensureMissionBtActive,
    releaseBt: releaseMissionBt,
    getFlags: missionRunnerFlags,
    onMessage: setMessage,
  });

  return {
    sendMissionGoal, navGoalPose, navGoalStatus, navGoalDriving, handleSendNavGoal,
    invalidateNavGoal, stopMissionBt, missionRunnerFlags, missionRunner,
  };
}
