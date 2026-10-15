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

import { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import { WORKSPACE_MISSION, WORKSPACE_AUTONOMY, STAGE_AUTHORING, STAGE_MAPPING, STAGE_MAP_EDIT, STAGE_NAVIGATE, STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN } from "../lib/stages";
import { initialNavigationRuntimeMode, initialRunRuntimeOwned, initialRunShutdownPending, initialWorkspaceStage, readMissionSession, saveMissionSession } from "../lib/session";
import { LAYER_PRESETS } from "../lib/layers";

// Shared map and navigation UI state. Feature documents and execution controllers own their own state.
export default function useMapWorkspaceState({
  workspaceKind,
}) {
  const robotType = useSelector(state => state.tasks?.robotType || "");

  const initialSessionRef = useRef(null);

  if (initialSessionRef.current === null) {
    initialSessionRef.current = readMissionSession();
  }

  const initialSession = initialSessionRef.current;

  const invalidateRunPoseRequestRef = useRef(() => {});

  const nomotionUpdateBusyRef = useRef(false);

  const tfBufferRef = useRef(new Map());

  const currentPoseRef = useRef(null);

  const amclPoseRef = useRef(null);

  const [status, setStatus] = useState(null);

  const [pendingRunStart, setPendingRunStart] = useState(null);

  const runStartRequestRef = useRef(0);

  const [busy, setBusy] = useState("");

  const [message, setMessage] = useState("");

  void message;

  const [runBtVisualizationActive, setRunBtVisualizationActive] = useState(false);

  const [interactionMode, setInteractionMode] = useState("view");

  const [runPoseInitialized, setRunPoseInitialized] = useState(false);

  const [navigationRuntimeMode, setNavigationRuntimeMode] = useState(() => (
    initialNavigationRuntimeMode(initialSession)
  ));

  const [runRuntimeOwned, setRunRuntimeOwned] = useState(() => (
    initialRunRuntimeOwned(initialSession)
  ));

  const [runShutdownPending, setRunShutdownPending] = useState(() => (
    initialRunShutdownPending(initialSession)
  ));

  const [tfBufferRevision, setTfBufferRevision] = useState(0);

  const [workspaceStage, setWorkspaceStage] = useState(() => initialWorkspaceStage(initialSession, workspaceKind));

  const [dataStageBusy, setDataStageBusy] = useState(false);

  useEffect(() => {
    const saved = readMissionSession();
    // Migrate both canvases before another workspace overwrites the old global stage.
    const rememberedStages = saved.autonomyLayoutVersion === 2 ? saved.workspaceStages : {
      [WORKSPACE_MISSION]: initialWorkspaceStage(initialSession, WORKSPACE_MISSION),
      [WORKSPACE_AUTONOMY]: initialWorkspaceStage(initialSession, WORKSPACE_AUTONOMY),
    };
    saveMissionSession({ autonomyLayoutVersion: 2,
      workspaceStages: { ...rememberedStages, [workspaceKind]: workspaceStage } });
  }, [initialSession, workspaceKind, workspaceStage]);

  const workspaceStageRef = useRef(workspaceStage);

  workspaceStageRef.current = workspaceStage;

  const [missionMapLoaded, setMissionMapLoaded] = useState(false);

  const prevRunningRef = useRef(false);

  const [mapEditToolsOpen, setMapEditToolsOpen] = useState(false);

  const [labelToolsOpen, setLabelToolsOpen] = useState(false);

  const [mapEditorReloadToken, setMapEditorReloadToken] = useState(0);

  const [layersByStage, setLayersByStage] = useState(() => ({
    [STAGE_MAPPING]: { ...LAYER_PRESETS[STAGE_MAPPING] },
    [STAGE_MAP_EDIT]: { ...LAYER_PRESETS[STAGE_MAP_EDIT] },
    [STAGE_NAVIGATE]: { ...LAYER_PRESETS[STAGE_NAVIGATE] },
    [STAGE_AUTHORING]: { ...LAYER_PRESETS[STAGE_AUTHORING] },
    [STAGE_RUN]: { ...LAYER_PRESETS[STAGE_RUN] },
    [STAGE_OPERATE]: { ...LAYER_PRESETS[STAGE_OPERATE] },
    [STAGE_COLLECT]: { ...LAYER_PRESETS[STAGE_COLLECT] },
    [STAGE_LEARN]: {},
  }));

  return {
    robotType, initialSession, invalidateRunPoseRequestRef, nomotionUpdateBusyRef, tfBufferRef,
    currentPoseRef, amclPoseRef, status, setStatus, pendingRunStart, setPendingRunStart,
    runStartRequestRef, busy, setBusy, setMessage, runBtVisualizationActive,
    setRunBtVisualizationActive, interactionMode, setInteractionMode, runPoseInitialized,
    setRunPoseInitialized, navigationRuntimeMode, setNavigationRuntimeMode, runRuntimeOwned,
    setRunRuntimeOwned, runShutdownPending, setRunShutdownPending, tfBufferRevision,
    setTfBufferRevision, workspaceStage, setWorkspaceStage, dataStageBusy, setDataStageBusy,
    workspaceStageRef, missionMapLoaded, setMissionMapLoaded, prevRunningRef, mapEditToolsOpen,
    setMapEditToolsOpen, labelToolsOpen, setLabelToolsOpen, mapEditorReloadToken,
    setMapEditorReloadToken, layersByStage, setLayersByStage,
  };
}
