// The flag describes the current navigation command, never UI selection or a
// retained reasoning result. Normalized execution is authoritative when present.
export function operateDestination(state, spots, navigationActive) {
  if (!navigationActive) return null;
  const execution = state.execution;
  const returning = state.phase === "returning" || state.runner?.phase === "return-navigation";
  const mission = execution ? execution.activity === "mission_navigation"
    : !returning && state.mode === "mission";
  const id = execution ? execution.target_spot_id
    : mission ? state.runner?.activeSpotId
      : returning ? state.resume_checkpoint?.return_spot_id || state.observation?.spot_id
        : state.observation?.spot_id;
  if (!id) return null;
  const target = (mission ? state.mission_route || [] : spots).find(spot => spot.id === id);
  if (!target?.pose || !Number.isFinite(target.pose.x) || !Number.isFinite(target.pose.y)) return null;
  return { ...target, scope: mission ? "mission" : "observation" };
}
