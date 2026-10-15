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

import useCollectDraft from "./useCollectDraft";
import useRecordExecutionContext from "./useRecordExecutionContext";
import useRecordCaptureSession from "./useRecordCaptureSession";
import { useState } from "react";
import { STAGE_COLLECT, STAGE_LEARN } from "../../mapWorkspace/lib/stages";
import useReasoningObservation from "./useReasoningObservation";

export default function useRecordWorkspaceData({
  workspace, runtimeMapName, collectExecution, runMissionSnapshot, running, collectObservations, recordMissionName, currentPose, recordAuthoringLock
}) {
  const { workspaceStage, runPoseInitialized, missionMapLoaded } = workspace;
  const [datasetSelection, setDatasetSelection] = useState({
    tool: "mission",
  });

  const [requestedPreviewModel, setRequestedPreviewModel] = useState(null);

  const [collectEditorBusy, setCollectEditorBusy] = useState(false);

  const recordExecution = useRecordExecutionContext({
    enabled: workspaceStage === STAGE_COLLECT, mapName: runtimeMapName,
    execution: collectExecution, mission: runMissionSnapshot, localized: running && runPoseInitialized,
  });

  const recordSession = useRecordCaptureSession({ execution: collectExecution, context: recordExecution,
    manager: collectObservations, mission: runMissionSnapshot, selectedMissionName: recordMissionName, mapName: runtimeMapName });

  const collectDraft = useCollectDraft({
    enabled: [STAGE_COLLECT, STAGE_LEARN].includes(workspaceStage),
    mapName: missionMapLoaded ? runtimeMapName : null,
    context: { pose: currentPose || null, localized: missionMapLoaded ? !!(running && runPoseInitialized) : null },
    getExecutionContext: recordExecution.getContext,
  });

  const recordAuthoringBusy = collectExecution.isRunning || collectDraft.active || collectDraft.recording || collectDraft.busy;

  recordAuthoringLock.current = recordAuthoringBusy;

  const qwenObservation = useReasoningObservation({
    enabled: workspaceStage === STAGE_COLLECT,
    requestedModel: requestedPreviewModel,
    loadedMapName: missionMapLoaded ? runtimeMapName : null,
    episode: collectDraft.episode,
    pose: currentPose || null,
    localized: missionMapLoaded ? !!(running && runPoseInitialized) : null,
    contextVersion: `${collectDraft.reasoningContextVersion}:${recordExecution.version}`,
    getExecutionContext: recordExecution.getContext,
  });

  return {
    datasetSelection, setDatasetSelection, setRequestedPreviewModel, collectEditorBusy,
    setCollectEditorBusy, recordSession, collectDraft, recordAuthoringBusy, qwenObservation,
  };
}
