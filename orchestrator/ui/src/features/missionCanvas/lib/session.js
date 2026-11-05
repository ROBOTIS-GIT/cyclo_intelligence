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

import {
  LEGACY_WORKSPACE_STANDALONE_BT,
  RUN_SHUTDOWN_RETRY_MAX_AGE_MS,
  STAGE_MAPPING,
  STAGE_RUN,
  WORKSPACE_ACTION_CANVAS,
  WORKSPACE_MISSION,
  WORKSPACE_STAGES,
} from "./stages";

export const MISSION_SESSION_STORAGE_KEY = "autonomy_studio_session";

// Sessions saved before the page became Autonomy Studio.
export const LEGACY_MISSION_SESSION_STORAGE_KEY = "mission_canvas_session";

export function readMissionSession() {
  if (typeof window === "undefined" || !window.sessionStorage) return {};
  try {
    const raw = window.sessionStorage.getItem(MISSION_SESSION_STORAGE_KEY)
      ?? window.sessionStorage.getItem(LEGACY_MISSION_SESSION_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function saveMissionSession(patch) {
  if (typeof window === "undefined" || !window.sessionStorage) return;
  const current = readMissionSession();
  const next = {
    ...current,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  window.sessionStorage.setItem(MISSION_SESSION_STORAGE_KEY, JSON.stringify(next));
}

export function initialWorkspaceStage(session) {
  const stage = session?.workspaceStage;
  return WORKSPACE_STAGES.some((item) => item.id === stage) ? stage : STAGE_MAPPING;
}

export function initialWorkspaceKind(session) {
  const kind = session?.workspaceKind;
  return kind === WORKSPACE_ACTION_CANVAS || kind === LEGACY_WORKSPACE_STANDALONE_BT
    ? WORKSPACE_ACTION_CANVAS
    : WORKSPACE_MISSION;
}

export function recentRunShutdownMarker(session) {
  if (session?.runShutdownPending !== true) return false;
  const requestedAt = Number(session.runShutdownRequestedAt);
  if (!Number.isFinite(requestedAt) || requestedAt <= 0) return true;
  return Date.now() - requestedAt <= RUN_SHUTDOWN_RETRY_MAX_AGE_MS;
}

export function initialRunShutdownPending(session) {
  if (recentRunShutdownMarker(session)) return true;
  // One-time migration for a Run session saved before ownership and page-exit
  // markers were introduced. Once rewritten, explicit ownership controls the
  // behavior and ordinary SPA remounts do not stop the runtime.
  return (
    session?.runRuntimeOwned === undefined
    && initialWorkspaceStage(session) === STAGE_RUN
    && session?.navigationRuntimeMode === "run"
  );
}

export function initialNavigationRuntimeMode(session) {
  if (initialRunShutdownPending(session)) return "idle";
  const mode = session?.navigationRuntimeMode;
  return ["idle", "mapping", "localization", "run"].includes(mode) ? mode : "idle";
}

export function initialRunRuntimeOwned(session) {
  if (session?.runShutdownPending === true && !recentRunShutdownMarker(session)) return false;
  if (session?.runRuntimeOwned === true) return true;
  if (session?.runRuntimeOwned === false) return false;
  // One-time migration for Run sessions saved before ownership tracking was
  // introduced. A Run page that restored an active Run mode was the owner in
  // the previous implementation.
  return initialWorkspaceStage(session) === STAGE_RUN && session?.navigationRuntimeMode === "run";
}
