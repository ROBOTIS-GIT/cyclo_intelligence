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
