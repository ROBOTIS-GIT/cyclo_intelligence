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

import { executionPhaseLabel } from "../lib/executionState";

// Presentation helpers deliberately do not infer or dispatch robot actions.
export function autonomyPhaseLabel(state = {}) {
  if (state.execution) return executionPhaseLabel(state.execution, { reasoningActive: !!state.active });
  const terminalLabel = { error: "Failed", stopping: "Stopping", stopped: "Stopped", idle: "Ready" }[state.phase];
  if (terminalLabel) return terminalLabel;
  if (state.paused || state.phase === "paused") return "Paused · reasoning active";
  return {
    settling: "Settling camera", positioning: "Patrol · moving", executing: "Mission supervision",
    returning: "Returning to observation", resuming: "Resuming patrol", idle: "Ready",
    starting: "Starting Qwen", loading: "Loading model", observing: "Observing",
    watching: "Observing", waiting: "Waiting", stopping: "Stopping", stopped: "Stopped", error: "Failed",
  }[state.phase] || "Ready";
}

export function resultTask(result) {
  return result?.reasoning_task || result?.context?.reasoning_task || "mission_selection";
}

export function displayedReasoningResult(state = {}) {
  if (["idle", "stopping", "stopped", "error", "interrupted"].includes(state.phase)
    || ["idle", "stopping", "stopped", "failed", "interrupted"].includes(state.execution?.lifecycle)) return null;
  if (state.result?.context_version != null && state.context_version != null
    && state.result.context_version !== state.context_version) return null;
  const task = state.reasoning_task || (state.mode === "mission" ? "mission_supervision" : "mission_selection");
  return state.result && resultTask(state.result) === task ? state.result : null;
}

export function resultMissionName(result) {
  if (resultTask(result) === "patrol_supervision") return "None";
  const activeMission = result?.context?.active_mission || result?.active_mission;
  return result?.selected_mission?.mission_name || activeMission?.mission_name ||
    (resultTask(result) === "mission_supervision" ? activeMission?.mission_id || result?.decision?.mission_id : null) || "None";
}

export function returnSpotLabel(state = {}) {
  const id = state.resume_checkpoint?.return_spot_id;
  if (!id) return null;
  return state.config?.spots?.find(spot => spot.id === id)?.label ||
    (state.observation?.spot_id === id ? state.observation.label : null) || id;
}
