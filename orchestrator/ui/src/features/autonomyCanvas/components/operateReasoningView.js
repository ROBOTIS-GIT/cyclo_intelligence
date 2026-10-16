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

// Present acknowledged execution state, never an AI proposal, as the current mode.
export function operateReasoningView(state = {}) {
  const execution = state.execution;
  const terminal = { idle: "Ready", starting: "Starting", stopping: "Stopping", stopped: "Stopped", failed: "Failed", interrupted: "Interrupted" };
  const lifecycle = execution?.lifecycle || ({ error: "failed", loading: "starting" }[state.phase] || state.phase || "idle");
  if (terminal[lifecycle]) return { mode: null, status: terminal[lifecycle], detail: "", tone: lifecycle === "failed" ? "danger" : "neutral" };
  const activity = execution?.activity;
  let mode = ({ observing: "observation", patrol_navigation: "patrol", observation_posture: "patrol",
    return_navigation: "patrol", return_posture: "patrol", mission_navigation: "mission", mission_action: "mission" })[activity];
  if (!mode) mode = ["returning", "positioning", "resuming", "settling"].includes(state.phase) ? "patrol"
    : execution?.mission_id || state.mode === "mission" || state.reasoning_task === "mission_supervision" ? "mission"
    : state.reasoning_task === "patrol_supervision" ? "patrol" : "observation";
  const returning = activity?.startsWith("return_") || state.phase === "returning";
  const control = execution?.control || (state.paused || state.phase === "paused" ? "paused" : "running");
  const status = ({ paused: "Paused", pausing: "Pausing", resuming: "Resuming", cancelling: "Cancelling" })[control]
    || (returning ? "Returning" : mode === "patrol" ? (activity?.includes("posture") || state.phase === "settling" ? "Settling" : "Moving")
      : mode === "mission" ? (activity === "mission_action" ? "Running task" : activity === "mission_navigation" ? "Navigating" : "Preparing") : "Observing");
  const spot = state.observation?.label;
  const mission = state.mission?.mission_name;
  const index = state.runner?.currentIndex;
  const waypoint = Number.isInteger(index) && index >= 0 ? state.mission_route?.[index]?.label : null;
  const atSpot = mode === "observation" || activity?.includes("posture") || state.phase === "settling";
  const detail = mode === "mission" ? [mission, waypoint].filter(Boolean).join(" · ")
    : spot ? `${atSpot ? "At " : ["paused", "pausing"].includes(control) ? "Destination: " : returning ? "Returning to " : "Moving to "}${spot}` : "";
  return { mode, status, detail, tone: ["paused", "pausing", "cancelling"].includes(control) ? "warning" : "active" };
}

export function decisionLabel(decision) {
  return ({ start: "Start mission", continue: "Continue", pause: "Pause", cancel: "Cancel mission",
    wait: "Wait", unknown: "Uncertain", next_observation: "Next observation" })[decision] || decision;
}
