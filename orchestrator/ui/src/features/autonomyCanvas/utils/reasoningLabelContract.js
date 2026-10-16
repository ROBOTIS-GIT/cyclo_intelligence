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

export const SELECTION_TASK = "mission_selection";
export const PATROL_TASK = "patrol_supervision";
export const SUPERVISION_TASK = "mission_supervision";
export const DECISION_CHOICES = {
  [SELECTION_TASK]: ["start", "wait", "next_observation", "unknown"],
  [SUPERVISION_TASK]: ["continue", "pause", "cancel"],
  [PATROL_TASK]: ["pause", "continue"],
};

// Only recorded server context selects the task. Older active-mission fields
// must not silently turn historical selection labels into supervision labels.
export function frameReasoningContract(frame) {
  const context = frame?.context || {};
  const task = context.reasoning_task || SELECTION_TASK;
  const supervision = task === SUPERVISION_TASK;
  // Historical return frames used mission_supervision. They can pause/resume
  // the return trip, but there is no active child task left to cancel.
  const returning = supervision && (context.active_mission?.status === "returning" ||
    context.observation?.phase === "returning" ||
    ["return-navigation", "return-posture"].includes(context.active_mission?.action));
  return {
    task,
    supervision,
    patrol: task === PATROL_TASK,
    decisions: (returning ? DECISION_CHOICES[PATROL_TASK] : DECISION_CHOICES[task] || [])
      .filter(value => value !== "next_observation" || context.observation?.can_advance !== false),
    activeMission: context.active_mission || null,
    candidateIds: Array.isArray(context.candidate_mission_ids)
      ? context.candidate_mission_ids : null,
  };
}

export function decisionNeedsMission(decision, task = SELECTION_TASK) {
  if (task === SUPERVISION_TASK) return DECISION_CHOICES[SUPERVISION_TASK].includes(decision);
  return task === SELECTION_TASK && decision === "start";
}

export function labelMatchesFrame(frame, target) {
  const contract = frameReasoningContract(frame);
  if (!contract.decisions.includes(target?.decision)) return false;
  if (contract.patrol) return !contract.activeMission && !target.mission_id;
  if (contract.supervision) {
    return Boolean(contract.activeMission?.mission_id &&
      target.mission_id === contract.activeMission.mission_id);
  }
  if (target.decision === "start") {
    return Boolean(!contract.activeMission && target.mission_id &&
      (contract.candidateIds === null || contract.candidateIds.includes(target.mission_id)));
  }
  return target.decision !== "next_observation" || (!target.mission_id && !contract.activeMission
    && frame?.context?.observation?.can_advance !== false);
}

export function episodeReasonPresets(annotations, frames) {
  const observations = new Map(frames.map(frame => [frame.frame_id, frame]));
  const presets = new Map();
  for (const annotation of annotations) {
    const frame = observations.get(annotation.frame_ids?.[0]);
    const target = annotation.target;
    const reason = target?.reason?.trim();
    if (annotation.review_status !== "approved" || !reason || !labelMatchesFrame(frame, target)) continue;
    const missionId = decisionNeedsMission(target.decision, frameReasoningContract(frame).task) ? target.mission_id || "" : "";
    const task = frameReasoningContract(frame).task;
    const key = `${task}\u0000${target.decision}\u0000${missionId}\u0000${reason}`;
    const existing = presets.get(key);
    presets.set(key, existing ? { ...existing, count: existing.count + 1 } : {
      reasoning_task: task, decision: target.decision, mission_id: missionId, reason, count: 1,
    });
  }
  return [...presets.values()].sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}
