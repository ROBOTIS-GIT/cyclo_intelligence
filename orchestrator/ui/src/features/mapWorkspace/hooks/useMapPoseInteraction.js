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
import { STAGE_AUTHORING, STAGE_NAVIGATE, STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN } from "../lib/stages";

export default function useMapPoseInteraction({
  workspace, designEditing, design, handleRunPoseEstimate, handleSendNavGoal, collectObservations, designDocumentReady
}) {
  const { workspaceStage, interactionMode } = workspace;
  const { handleCompleteDesignAtRobotPose, createDesignWaypointOnMap } = designEditing;
  const { pendingBehaviorNodeTag, placePendingBehaviorAtPose } = design;
  const handleCreateSpotAtPose = useCallback(async (x, y, yaw) => {
    if (workspaceStage === STAGE_NAVIGATE) {
      if (interactionMode === "initial") {
        void handleRunPoseEstimate(x, y, yaw);
      } else if (interactionMode === "goal") {
        void handleSendNavGoal(x, y, yaw);
      }
      return;
    }
    if (workspaceStage === STAGE_RUN || [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(workspaceStage)) {
      // The BT node being up is expected here; run pose estimation must not
      // fall through to the design-stage guards below.
      if (interactionMode === "initial") void handleRunPoseEstimate(x, y, yaw);
      if (workspaceStage === STAGE_COLLECT && interactionMode === "spot") {
        collectObservations.place({ x, y, yaw });
      }
      return;
    }
    if (workspaceStage === STAGE_AUTHORING && !designDocumentReady) return;
    if (interactionMode === "initial") {
      void handleCompleteDesignAtRobotPose(x, y, yaw);
      return;
    }
    if (interactionMode === "behavior" && pendingBehaviorNodeTag) {
      placePendingBehaviorAtPose(x, y, yaw);
      return;
    }
    if (interactionMode !== "spot") return;
    await createDesignWaypointOnMap(x, y, yaw);
  }, [collectObservations, handleSendNavGoal,
    placePendingBehaviorAtPose,
    designDocumentReady,
    createDesignWaypointOnMap,
    handleCompleteDesignAtRobotPose,
    handleRunPoseEstimate,
    interactionMode,
    pendingBehaviorNodeTag,
    workspaceStage,
  ]);

  return {
    handleCreateSpotAtPose,
  };
}
