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

import { useCallback, useRef } from "react";

const TERMINAL = new Set(["idle", "done", "failed", "cancelled"]);

// This is a browser runner report, never an authoritative ROS snapshot. The
// server correlates its validity interval with frame receipt and validates IDs.
export default function useRecordExecutionContext(props) {
  const { enabled, mapName, localized, execution } = props;
  const runner = execution?.missionRunner || props.runner || {};
  const missionCycle = execution?.cycle || props.missionCycle || {};
  const mission = execution && missionCycle.pinnedMissionId
    ? { mapName, missionId: missionCycle.pinnedMissionId } : props.mission || {};
  const captureType = execution ? execution.executionType : props.captureType;
  const observationSpotId = execution ? execution.observationSpotId : props.observationSpotId;
  const observationReady = execution ? execution.observationReady : props.observationReady;
  const patrol = execution?.navigation || props.patrol || {};
  const state = useRef({ active: false, identity: null, signature: null, version: 0, since: Date.now(), report: null });
  const active = Boolean(enabled && (runner.isRunning || runner.paused) && !TERMINAL.has(runner.status));
  const current = state.current;
  if (active && !current.active) {
    current.identity = mission.mapName === mapName && mission.missionId
      ? { map_name: mapName, mission_id: mission.missionId } : null;
  }
  if (!active) current.identity = null;
  const missionReport = active ? {
    map_name: mapName,
    // A missing/mismatched identity becomes unverified on the server. Never
    // substitute a selected mission's name or invent a stable ID.
    mission_id: current.identity?.map_name === mapName ? current.identity.mission_id : null,
    status: runner.paused ? "paused" : "running",
    step: Math.max(0, Number.isInteger(runner.currentIndex) ? runner.currentIndex : 0) + 1,
    action: runner.phase && runner.phase !== "none" ? runner.phase : runner.status || "starting",
  } : null;
  const returning = !!missionCycle.returning && (captureType === "mission" || !!execution);
  const reasoningTask = captureType === "patrol" || returning ? "patrol_supervision"
    : captureType === "mission" || (!captureType && active) ? "mission_supervision" : "mission_selection";
  const patrolActive = enabled && captureType === "patrol" && (patrol.isRunning || patrol.paused)
    && !TERMINAL.has(patrol.status) && (patrol.fromSpotId || !!execution) && patrol.toSpotId
    && (!execution || patrol.held || !["preparing", "launching"].includes(patrol.phase));
  let report = null;
  if (returning && enabled && missionCycle.toSpotId && missionCycle.isRunning) report = {
    map_name: mapName, mission_id: null, status: missionCycle.paused ? "paused" : "running",
    phase: missionCycle.returnPhase || "returning", from_spot_id: null,
    to_spot_id: missionCycle.toSpotId, step: 1, action: missionCycle.phase,
  };
  else if (!captureType && !execution) report = missionReport;
  else if (captureType === "mission") report = missionReport?.mission_id && runner.currentIndex >= 0 && runner.phase && runner.phase !== "none" ? missionReport : null;
  else if (patrolActive) report = {
    map_name: mapName, mission_id: null, status: patrol.paused ? "paused" : "running", phase: ["arrived", "bt-loading", "bt-running", "bt-done", "settling"].includes(patrol.phase) ? "settling" : patrol.fromSpotId ? "moving" : "approaching",
    from_spot_id: patrol.fromSpotId, to_spot_id: patrol.toSpotId,
    step: patrol.currentIndex + 1, action: patrol.phase,
  };
  else if (captureType === "observation" && enabled && observationReady && observationSpotId && localized) report = {
    map_name: mapName, mission_id: null, status: "arrived", phase: "observing",
    observation_spot_id: observationSpotId,
  };
  if (report && captureType) { report.reasoning_task = reasoningTask; report.localized = !!localized; }
  const signature = JSON.stringify([reasoningTask, report]);
  if (signature !== current.signature) {
    current.signature = signature;
    current.version += 1;
    current.since = Date.now();
  }
  current.active = active;
  current.report = report;
  current.reasoningTask = reasoningTask;
  const getContext = useCallback(() => {
    const latest = state.current;
    return latest.report ? {
      reasoning_task: latest.reasoningTask,
      recording_execution: { ...latest.report, sampled_at_ms: Date.now(),
        context_version: latest.version, context_valid_from_ms: latest.since },
    } : { reasoning_task: latest.reasoningTask, recording_execution: null };
  }, []);
  return { getContext, version: current.version, ready: !!report && (returning || !captureType || captureType !== "mission" || !!report.mission_id) };
}
