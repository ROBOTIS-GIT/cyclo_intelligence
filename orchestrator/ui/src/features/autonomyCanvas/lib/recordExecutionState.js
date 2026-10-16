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

// An observed browser execution snapshot, never a server ownership claim.
const NAVIGATION_PHASES = new Set(["nav-sent", "awaiting-nav-result"]);
const POSTURE_PHASES = new Set(["arrived", "bt-loading", "bt-running", "bt-done", "settling"]);
const STARTING_PHASES = new Set(["preparing", "launching", "switching", "starting", "return-pending", "return-starting"]);
const RETURN_PHASES = new Set(["return-pending", "return-starting", "returning", "return-settling", "return-arrived"]);
const indexOrNull = value => Number.isInteger(value) && value >= 0 ? value : null;

function returnCheckpoint(cycle, continuation, planRevision) {
  if (!cycle.pinnedSpot?.id) return null;
  const checkpoint = { return_spot_id: cycle.pinnedSpot.id, plan_revision: planRevision };
  if (continuation) {
    const next = continuation.index + 1;
    checkpoint.next_spot_id = continuation.route?.[next]?.id
      || (continuation.closed ? continuation.route?.[0]?.id : null) || null;
    checkpoint.route_index = indexOrNull(continuation.index);
    checkpoint.dwell_remaining_ms = Number.isFinite(continuation.dwellRemainingMs) ? continuation.dwellRemainingMs : null;
  }
  return checkpoint;
}

// Editor selection and capture/episode state are intentionally not inputs.
export function buildRecordExecutionState({ mapName, planRevision = null, enabled = true,
  navigation = {}, cycle = {}, missionRunner = {}, owner = null,
  observationReady = false, observationSpotId = null, stopping = false,
  cancelFailed = false, pendingControl = null, controlError = null,
  retainedExecution = null, patrolContinuation = null } = {}) {
  const operation = pendingControl?.kind || null;
  const returning = RETURN_PHASES.has(cycle.cyclePhase) || cycle.returning || cycle.returnPending;
  const childActive = Boolean(cycle.isRunning || missionRunner.isRunning || returning);
  const runner = childActive ? cycle.isRunning ? cycle : missionRunner : navigation;
  const running = Boolean(navigation.isRunning || navigation.patrolPaused || childActive);
  const pauseFailed = running && Boolean(runner.pauseFailed);
  const ownsExecution = Boolean(cancelFailed || stopping || operation || running);
  const retain = Boolean((cancelFailed || stopping || operation === "pausing") && retainedExecution?.owns_execution);
  const failure = cancelFailed || pauseFailed || (owner === "mission" ? cycle.cyclePhase === "failed" : navigation.status === "failed");
  const starting = operation === "starting" || STARTING_PHASES.has(childActive ? cycle.cyclePhase : navigation.phase)
    || (ownsExecution && runner.status === "starting");
  const terminal = ["cancelled", "done"].includes(owner === "mission" ? cycle.cyclePhase : navigation.phase)
    || (!owner && cycle.cyclePhase === "cancelled");
  const lifecycle = failure ? "failed"
    : operation === "pausing" || operation === "resuming" ? "active"
      : operation === "starting" ? "starting"
        : stopping ? "stopping"
          : !enabled && ownsExecution ? "interrupted"
            : starting ? "starting" : ownsExecution ? "active" : terminal ? "stopped" : "idle";
  const paused = ownsExecution && (runner.paused || (!childActive && navigation.patrolPaused));
  const control = cancelFailed || (stopping && operation === "cancelling") ? "cancelling"
    : operation === "pausing" || (ownsExecution && runner.pausing) || pauseFailed ? "pausing"
      : operation === "resuming" ? "resuming"
        : stopping || operation === "starting" ? "cancelling"
          : paused ? "paused" : ownsExecution ? "running" : "idle";

  let activity = "idle";
  if (returning) {
    activity = NAVIGATION_PHASES.has(cycle.phase) || ["return-pending", "return-starting"].includes(cycle.cyclePhase)
      ? "return_navigation" : "return_posture";
  } else if (childActive) {
    activity = NAVIGATION_PHASES.has(runner.phase) ? "mission_navigation" : "mission_action";
  } else if (observationReady && observationSpotId) {
    activity = "observing";
  } else if (navigation.isRunning || navigation.patrolPaused) {
    activity = NAVIGATION_PHASES.has(navigation.phase) || STARTING_PHASES.has(navigation.phase)
      || navigation.status === "starting" || navigation.patrolPaused ? "patrol_navigation"
        : POSTURE_PHASES.has(navigation.phase) || ["done", "advancing", "running-bt"].includes(navigation.status)
          ? "observation_posture" : "idle";
  }
  if (retain) activity = retainedExecution.activity;

  const childContext = childActive || (owner === "mission" && failure);
  const checkpoint = childContext ? returnCheckpoint(cycle, patrolContinuation, planRevision) : null;
  const target = returning ? cycle.pinnedSpot?.id || cycle.toSpotId
    : childActive ? missionRunner.activeSpotId : navigation.toSpotId;
  const error = cancelFailed ? controlError || "Execution cancellation was not confirmed. Retry Stop."
    : pauseFailed ? runner.reason || "Execution pause was not confirmed."
      : failure ? (owner === "mission" ? cycle.reason : navigation.reason) || "Execution failed." : null;
  return {
    schema_version: 1, source: "record", authority: "browser", session_id: null, epoch: null, revision: null,
    lifecycle, activity, control, owns_execution: ownsExecution,
    map_name: mapName || null, plan_revision: planRevision,
    observation_spot_id: activity === "observing" ? observationSpotId : retain ? retainedExecution.observation_spot_id : null,
    target_spot_id: retain ? retainedExecution.target_spot_id : activity === "idle" || activity === "observing" ? null : target || null,
    mission_id: retain ? retainedExecution.mission_id : childContext ? cycle.pinnedMissionId || null : null,
    step_index: retain ? retainedExecution.step_index : ownsExecution ? indexOrNull(runner.currentIndex) : null,
    pause_causes: control === "paused" || control === "pausing" ? { manual: "Operator pause" } : {},
    return_checkpoint: retain ? retainedExecution.return_checkpoint : checkpoint,
    error,
  };
}

// Legacy consumers still use these labels for recording metadata and controls.
export function recordExecutionType(state, held = false) {
  if (state.lifecycle === "failed" && state.control === "cancelling") return null;
  if (state.activity.startsWith("return_")) return "patrol";
  if (state.activity.startsWith("mission_")) return "mission";
  if (state.activity === "observing") return "observation";
  if (state.activity.startsWith("patrol_") || state.activity === "observation_posture" || held) return "patrol";
  return null;
}

export function recordNavigationActive(state) {
  return state.lifecycle === "active" && state.control === "running"
    && ["patrol_navigation", "mission_navigation", "return_navigation"].includes(state.activity);
}
