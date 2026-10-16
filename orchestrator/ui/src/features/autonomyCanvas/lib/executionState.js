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

// Shared presentation contract. Execution owners publish snapshots; UI never
// turns a displayed phase or a Qwen proposal into an actuator command.
export function executionPhaseLabel(execution, { reasoningActive = false, compact = false } = {}) {
  if (!execution) return null;
  if (execution.lifecycle === "idle" && execution.activity === "observing") return "Observing";
  const terminal = { failed: "Failed", interrupted: "Interrupted", stopping: "Stopping",
    stopped: "Stopped", idle: compact ? "Idle" : "Ready", starting: "Starting" }[execution.lifecycle];
  if (terminal) return terminal;
  const control = { pausing: "Pausing", resuming: "Resuming", cancelling: "Cancelling" }[execution.control];
  if (control) return control;
  if (execution.control === "paused") {
    if (reasoningActive) return "Paused · reasoning active";
    return execution.activity === "patrol_navigation" || execution.activity === "observing" ? "Patrol paused" : "Paused";
  }
  return {
    patrol_navigation: compact ? "Patrol" : "Patrol · moving",
    observation_posture: "Settling camera",
    observing: "Observing",
    mission_navigation: reasoningActive ? "Mission supervision" : "Mission",
    mission_action: reasoningActive ? "Mission supervision" : "Mission",
    return_navigation: compact ? "Returning" : "Returning to observation",
    return_posture: compact ? "Returning" : "Returning to observation",
    idle: compact ? "Idle" : "Ready",
  }[execution.activity] || (compact ? "Idle" : "Ready");
}

export function executionOwnsRobot(status = {}) {
  // needsStop also represents an unacknowledged Start request in the UI.
  return Boolean(status.execution?.owns_execution || status.needsStop);
}

export function latestExecutionStatus(current, incoming) {
  const before = current?.execution, after = incoming?.execution;
  // Revisions are local to the server process and session. Reconnect after a
  // restart must accept the new interrupted snapshot even with a lower revision.
  if (before && after && before.epoch === after.epoch && before.session_id === after.session_id
    && Number.isInteger(before.revision) && Number.isInteger(after.revision)
    && after.revision < before.revision) return current;
  return incoming;
}
