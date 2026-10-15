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

export const STAGE_MAPPING = "mapping";

export const STAGE_MAP_EDIT = "map_edit";

export const STAGE_AUTHORING = "authoring";

export const STAGE_RUN = "run";

export const STAGE_OPERATE = "operate";

// Retain the persisted Record stage ID for existing sessions.
export const STAGE_COLLECT = "collect";
export const STAGE_LEARN = "learn";

// Legacy links and saved sessions are normalized to Train before use.
export const STAGE_ANNOTATE = "annotate";
export const STAGE_TRAIN = "train";
export const STAGE_REASONING = "reasoning";

// Direct point-to-point driving: localize on a saved map, click a goal, go.
export const STAGE_NAVIGATE = "navigate";

export const WORKSPACE_MISSION = "mission";
export const WORKSPACE_AUTONOMY = "autonomy";

export const WORKSPACE_ACTION_CANVAS = "action_canvas";

// Sessions saved while the workspace was still called "standalone BT".
export const LEGACY_WORKSPACE_STANDALONE_BT = "standalone_bt";

export const RUN_SHUTDOWN_RETRY_MAX_AGE_MS = 60_000;

export const WORKSPACE_STAGES = [
  { id: STAGE_MAPPING, label: "Mapping" },
  { id: STAGE_MAP_EDIT, label: "Map Edit" },
  { id: STAGE_NAVIGATE, label: "Navigation" },
  { id: STAGE_AUTHORING, label: "Design" },
  { id: STAGE_RUN, label: "Run" },
  { id: STAGE_COLLECT, label: "Record" },
  { id: STAGE_LEARN, label: "Train" },
  { id: STAGE_OPERATE, label: "Operate" },
];

// The Mission Canvas rail groups map-bound stages by the asset they manage.
// Action Canvas is selected from the chooser and intentionally has no rail.
export const WORKSPACE_NAV_GROUPS = [
  { caption: "NAVIGATION", stageIds: [STAGE_MAPPING, STAGE_MAP_EDIT, STAGE_NAVIGATE] },
  { caption: "MISSION", stageIds: [STAGE_AUTHORING, STAGE_RUN] },
  { caption: "AUTONOMY", stageIds: [STAGE_COLLECT, STAGE_LEARN, STAGE_OPERATE] },
];


export function workspaceNavGroups(kind = WORKSPACE_MISSION) {
  return WORKSPACE_NAV_GROUPS.filter(group =>
    kind === WORKSPACE_AUTONOMY ? group.caption === "AUTONOMY" : group.caption !== "AUTONOMY");
}
export function normalizeWorkspaceStage(stage) {
  return [STAGE_ANNOTATE, STAGE_TRAIN, STAGE_REASONING].includes(stage) ? STAGE_LEARN : stage;
}

export function workspaceHasStage(kind, stage) {
  return workspaceNavGroups(kind).some(group => group.stageIds.includes(normalizeWorkspaceStage(stage)));
}
