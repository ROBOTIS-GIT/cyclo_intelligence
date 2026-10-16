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

// Use recorded metadata; a recorded patrol route is not proof of visits.
export const EPISODE_TYPES = {observation: "Observation", mission: "Mission", patrol: "Patrol"};
export function episodeListDetails(episode, spotNames = {}) {
  const type = episode.episode_type || (episode.recording_mission_name && episode.recording_mission_name !== "ObservationMode" ? "mission" : "observation");
  const spot = spotNames[episode.observation_spot_id || episode.return_observation_spot_id] || episode.observation_spot?.label || episode.return_observation_spot?.label
    || episode.observation_spot_id || episode.return_observation_spot_id;
  const mission = episode.recording_mission_name || "Unspecified mission";
  const route = (episode.patrol_spot_ids || []).map(id =>
    spotNames[id] || [episode.observation_spot, episode.return_observation_spot].find(s => s?.id === id)?.label || id).join(" → ");
  const target = type === "mission" ? mission : type === "observation" ? spot || "Unspecified observation"
    : episode.capture_segment === "mission_return" ? `Return · ${episode.return_mission_name || "Mission"}` : "Observation route";
  const detail = type === "mission" ? spot || "" : type === "patrol"
    ? episode.capture_segment === "mission_return" ? `To ${spot || "observation"}` : route ? `Route: ${route}` : "" : "";
  return {type, group: `${EPISODE_TYPES[type] || "Other"} · ${target}`, detail,
    name: episode.storage_path?.split("/").filter(Boolean).pop() || episode.scenario || episode.id,
    frames: episode.frame_count ?? episode.observations?.length ?? 0};
}
