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

import { FiAlertCircle, FiClock, FiEye, FiNavigation, FiPause, FiPlay, FiRotateCcw, FiSquare, FiTool } from "react-icons/fi";
import { MdOutlineRestaurant } from "react-icons/md";
import "./OperateThoughtAction.css";

const action = (label, kind, tone = "active") => ({ label, kind, tone });
const TASK_LABELS = {
  servedish: "Serving food", servefood: "Serving food", foodserving: "Serving food",
  cleantable: "Cleaning the table", cleardish: "Clearing dishes", cleandish: "Cleaning dishes",
  cleartable: "Clearing the table", dolaundry: "Doing laundry", trashdisposal: "Taking out the trash",
  turnontv: "Turning on the TV", turnofftv: "Turning off the TV",
};
const RUNNER_ACTIVITIES = {
  navigation: "mission_navigation", "waypoint-task": "mission_action", "task-completed": "mission_action",
  "patrol-navigation": "patrol_navigation", "return-navigation": "return_navigation",
  "observation-posture": "observation_posture", "return-posture": "return_posture",
};

function destination(state, activity) {
  const execution = state.execution;
  const mission = activity === "mission_navigation";
  const targetId = execution ? execution.target_spot_id : mission ? state.runner?.activeSpotId
    : activity === "return_navigation" ? state.resume_checkpoint?.return_spot_id || state.observation?.spot_id
      : state.observation?.spot_id;
  const route = state.mission_route || [];
  // Mission waypoints and saved observation spots have separate ID namespaces.
  const spots = mission ? route : [state.observation, ...(state.config?.spots || [])];
  let label = targetId ? spots.find(spot => spot && (spot.id || spot.spot_id) === targetId)?.label : "";
  if (!label && targetId && mission && state.runner?.activeSpotId === targetId) label = state.runner.label;
  // A snapshot with no target must not inherit a retained compatibility target.
  if (!execution && !targetId) {
    const index = state.runner?.currentIndex;
    label = mission ? (Number.isInteger(index) && index >= 0 ? route[index]?.label : "") || state.runner?.label
      : state.observation?.label;
  }
  // Saved observation point names often include their internal role as a suffix.
  return typeof label === "string" ? label.replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ").replace(/\s+observation(?:\s+(?:point|spot))?$/i, "").trim() : "";
}

// Read acknowledged execution only; a proposed Qwen decision never starts an action.
export function thoughtAction(state = {}) {
  const execution = state.execution;
  const lifecycle = execution?.lifecycle || state.phase || "idle";
  if (state.error || execution?.error || ["error", "failed"].includes(lifecycle)) {
    return action("Needs attention", "error", "danger");
  }
  if (["stopping", "cancelling"].includes(lifecycle)) return action("Stopping", "stop", "neutral");
  if (["stopped", "interrupted"].includes(lifecycle)) return action("Stopped", "stop", "neutral");
  if (lifecycle === "idle") return action("Ready", "ready", "neutral");
  if (["starting", "loading"].includes(lifecycle)) return action("Getting ready", "ready", "neutral");
  const control = execution?.control || state.control_status || (state.paused || state.phase === "paused" ? "paused" : "running");
  if (execution?.owns_execution === false || control === "idle" || (!execution && state.active === false)) {
    return action("Ready", "ready", "neutral");
  }
  if (control === "paused") return action("Paused", "pause", "neutral");
  if (control === "pausing") return action("Pausing", "pause", "neutral");
  if (control === "cancelling") return action("Stopping", "stop", "neutral");
  if (control === "resuming") return action("Resuming", "resume", "neutral");

  // Legacy snapshots still carry actual runner phases. Mission admission alone
  // is insufficient evidence that a physical task is underway.
  const activity = execution ? execution.activity : RUNNER_ACTIVITIES[state.runner?.phase]
    || ({ positioning: "patrol_navigation", returning: "return_navigation", settling: "observation_posture",
      observing: "observing", watching: "observing", waiting: "observing" })[state.phase];
  if (["patrol_navigation", "mission_navigation", "return_navigation"].includes(activity)) {
    const place = destination(state, activity);
    const returning = activity === "return_navigation";
    return action(`${returning ? "Returning" : "Moving"}${place ? ` to ${place}` : ""}`, returning ? "return" : "move");
  }
  if (["observation_posture", "return_posture"].includes(activity)) return action("Getting ready", "ready");
  if (activity === "observing") return action("Looking around", "look");
  if (activity === "mission_action") {
    const name = state.mission?.mission_name;
    const key = typeof name === "string" ? name.replace(/[\s_-]+/g, "").toLowerCase() : "";
    const label = TASK_LABELS[key] || "Working";
    return action(label, label === "Serving food" ? "serve" : "work");
  }
  return action("Getting ready", "ready", "neutral");
}

const ICONS = { error: FiAlertCircle, ready: FiClock, look: FiEye, move: FiNavigation,
  pause: FiPause, resume: FiPlay, return: FiRotateCcw, stop: FiSquare, work: FiTool, serve: MdOutlineRestaurant };

export default function OperateThoughtAction({ observation = {} }) {
  const { label, kind, tone } = thoughtAction(observation);
  const Icon = ICONS[kind];
  return <div className={`operate-thought-current-action tone-${tone}`} role="status" aria-label="Current action" aria-atomic="true">
    <Icon aria-hidden="true"/><span>{label}</span>
  </div>;
}
