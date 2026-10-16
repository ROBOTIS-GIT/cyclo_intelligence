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

import { frameReasoningContract, SELECTION_TASK, PATROL_TASK, SUPERVISION_TASK } from "./reasoningLabelContract";

export const EPISODE_TYPES = ["observation", "patrol", "mission"];
export const EPISODE_TASKS = { observation: SELECTION_TASK, patrol: PATROL_TASK, mission: SUPERVISION_TASK };
export const TYPE_LABELS = { observation: "Observation", patrol: "Patrol", mission: "Mission" };

export function episodeCategory(episode) {
  if (EPISODE_TYPES.includes(episode?.episode_type)) return episode.episode_type;
  const path = episode?.storage_path?.split("/") || [];
  const folder = path[path.length - 2];
  if (EPISODE_TYPES.includes(folder)) return folder;
  return episode?.recording_mission_name && episode.recording_mission_name !== "ObservationMode" ? "mission" : "observation";
}
export function categoryLabel(episode) { return TYPE_LABELS[episodeCategory(episode)]; }

// Recorded task owns decisions. Category never fabricates an execution context.
export function frameMatchesEpisode(episode, frame) {
  if (!frame) return false;
  const context = frame.context || {}, contract = frameReasoningContract(frame), type = episodeCategory(episode);
  if (contract.task !== EPISODE_TASKS[type] || context.runtime_source === "unverified") return false;
  if (type === "mission") {
    if (context.observation?.phase === "returning" || ["return-navigation", "return-posture"].includes(contract.activeMission?.action)) return false;
    if (!contract.activeMission?.mission_id || !["running", "paused"].includes(contract.activeMission.status)) return false;
    if (episode.recording_mission_id && contract.activeMission.mission_id !== episode.recording_mission_id) return false;
  } else if (contract.activeMission) return false;
  if (episode?.capture_contract !== "task_aligned_v1") return true;
  if (context.reasoning_task !== EPISODE_TASKS[type] || context.map_name !== episode.map_name) return false;
  if (!["server_at_capture", "record_runner_reported"].includes(context.source)
      || ["unavailable", "unverified"].includes(context.runtime_source)) return false;
  const observation = context.observation || {};
  if (type === "observation") return Boolean(episode.observation_spot_id) && context.localized === true
    && ["observing", "arrived"].includes(observation.phase)
    && (observation.at_spot_id || observation.spot_id) === episode.observation_spot_id;
  if (type === "patrol") {
    if (episode.capture_segment === "mission_return" && !["returning", "settling"].includes(observation.phase)) return false;
    const route = episode.patrol_spot_ids || [];
    return ["moving", "returning", "approaching", "settling"].includes(observation.phase)
      && route.includes(observation.target_spot_id || observation.at_spot_id)
      && (!observation.from_spot_id || route.includes(observation.from_spot_id));
  }
  return true;
}

export function approvedReasoningCount(episode) {
  if (episode?.reasoning_approved_count != null) return episode.reasoning_approved_count;
  if (episode?.observations) {
    const valid = new Set(episode.observations.filter(frame => frameMatchesEpisode(episode, frame)).map(frame => frame.frame_id));
    return new Set((episode.annotations || []).filter(label => label.review_status === "approved" && valid.has(label.frame_ids?.[0])).map(label => label.frame_ids[0])).size;
  }
  return Math.max(0, (episode?.frame_count || 0) - (episode?.unlabeled_count ?? episode?.frame_count ?? 0));
}
