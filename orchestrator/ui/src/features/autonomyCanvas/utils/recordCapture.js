// Capture identities follow execution; UI selections never invent runtime state.
export function episodeFieldsForSession(session, options = {}) {
  if (!session?.ready || !session.type) return null;
  const type = session.type;
  return {
    capture_contract: "task_aligned_v1", episode_type: type,
    plan_revision: session.planRevision,
    observation_spot_id: type !== "patrol" ? session.observationSpotId || null : null,
    patrol_spot_ids: type === "patrol" ? session.patrolSpotIds || [] : [],
    recording_mission_id: type === "mission" ? session.missionId : null,
    recording_mission_name: type === "mission" ? session.missionName : null,
    return_observation_spot_id: type === "mission" ? session.returnSpotId || session.observationSpotId : null,
    ...options,
  };
}
export function captureScopeMatches(episode, fields, returning = false) {
  if (!episode || !fields || episode.episode_type !== fields.episode_type) return false;
  if (fields.episode_type === "mission") return episode.recording_mission_id === fields.recording_mission_id
    && episode.observation_spot_id === fields.observation_spot_id;
  if (fields.episode_type === "observation") return episode.observation_spot_id === fields.observation_spot_id;
  if (returning && episode.capture_segment === "mission_return") return true;
  if (episode.capture_segment === "mission_return") return false;
  return JSON.stringify(episode.patrol_spot_ids || []) === JSON.stringify(fields.patrol_spot_ids || []);
}

// Preparation uses saved plan bindings; actual frame metadata still uses execution.
export function episodeFieldsForPreparation(session, type, options = {}) {
  if (!session) return { episode_type: type, recording_mission_name: null, ...options };
  const target = session.captureTargets?.[type];
  if (target) return episodeFieldsForSession(target, options);
  return session.type === type ? episodeFieldsForSession(session, options) : null;
}
