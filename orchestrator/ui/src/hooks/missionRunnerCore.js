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

// Pure, side-effect-free core for the Mission Runner: empty-BT detection and
// the reducer that publishes observable run progress. The async driver that
// sends nav goals and ticks behavior trees lives in useMissionRunner.js.


export const RunnerStatus = {
  IDLE: "idle",
  STARTING: "starting",
  NAVIGATING: "navigating",
  RUNNING_BT: "running-bt",
  ADVANCING: "advancing",
  DONE: "done",
  FAILED: "failed",
  CANCELLED: "cancelled",
};

export const RunnerPhase = {
  NONE: "none",
  NAV_SENT: "nav-sent",
  AWAITING_NAV_RESULT: "awaiting-nav-result",
  ARRIVED: "arrived",
  BT_LOADING: "bt-loading",
  BT_RUNNING: "bt-running",
  BT_DONE: "bt-done",
};

export const WaypointState = {
  PENDING: "pending",
  NAVIGATING: "navigating",
  RUNNING_BT: "running-bt",
  DONE: "done",
  SKIPPED: "skipped",
  FAILED: "failed",
};

// Only BT execution uses polling; Nav2 completion comes from the action result.
export const DEFAULT_RUNNER_CONFIG = {
  btStartTimeoutMs: 5000,
  btTimeoutMs: 300000,
  // Covers the supervisor's maximum 6 h build + 30 min readiness ceilings,
  // with one minute for request/transition overhead. Stop remains cancellable.
  backendTaskTimeoutMs: 23520000,
  pollMs: 250,
};

// Convert a stored spot into the pose sent to Nav2.
export function goalFromSpot(spot) {
  const pose = (spot && spot.pose) || {};
  return {
    x: Number(pose.x ?? 0),
    y: Number(pose.y ?? 0),
    yaw: Number(pose.yaw ?? 0),
  };
}

// A default/empty local BT (childless MainTree or blank) means "navigate only".
// Unparseable XML is treated as non-empty so load_and_run surfaces the error.
export function isEmptyBt(xml) {
  if (!xml || !xml.trim()) return true;
  if (typeof DOMParser === "undefined") return false;
  try {
    const doc = new DOMParser().parseFromString(xml, "text/xml");
    if (doc.getElementsByTagName("parsererror").length) return false;
    const trees = Array.from(doc.getElementsByTagName("BehaviorTree"));
    if (!trees.length) return true;
    const root = doc.documentElement;
    const mainId = root && root.getAttribute("main_tree_to_execute");
    const main = trees.find((tree) => tree.getAttribute("ID") === mainId) || trees[0];
    return !Array.from(main.childNodes).some((node) => node.nodeType === 1);
  } catch (error) {
    return false;
  }
}

// Image pull/build can legitimately take much longer than an ordinary local
// task. Extend the mission-side watchdog only for SendCommand operations that
// may provision a backend; all other trees retain the normal five-minute cap.
// Legacy SendCommand XML has no target and therefore means INFERENCE.
export function requiresBackendTaskTimeout(xml) {
  if (!xml || !xml.trim() || typeof DOMParser === "undefined") return false;
  try {
    const doc = new DOMParser().parseFromString(xml, "text/xml");
    if (doc.getElementsByTagName("parsererror").length) return false;
    return Array.from(doc.getElementsByTagName("SendCommand")).some((node) => {
      const target = String(node.getAttribute("target") || "INFERENCE").trim().toUpperCase();
      const command = String(node.getAttribute("command") || "LOAD").trim().toUpperCase();
      if (target === "INFERENCE") return command === "LOAD";
      return target === "DOCKER" && (command === "START" || command === "RESTART");
    });
  } catch (error) {
    return false;
  }
}

// Build one navigation action. At most two consecutive nav-only waypoints are
// collected; the following BT waypoint is included as the terminal pose.
export function navigationBatchFromIndex(
  spots,
  startIndex,
  resolveBtXml,
  maxEmptyWaypoints = 2,
) {
  if (!Array.isArray(spots) || startIndex < 0 || startIndex >= spots.length) {
    return { indices: [], useThroughPoses: false };
  }

  const xmlAt = (index) => (
    typeof resolveBtXml === "function" ? resolveBtXml(spots[index]) : ""
  );
  if (!isEmptyBt(xmlAt(startIndex))) {
    return { indices: [startIndex], useThroughPoses: false };
  }

  const indices = [];
  let index = startIndex;
  while (
    index < spots.length
    && indices.length < maxEmptyWaypoints
    && isEmptyBt(xmlAt(index))
  ) {
    indices.push(index);
    index += 1;
  }

  if (index < spots.length && !isEmptyBt(xmlAt(index))) {
    indices.push(index);
  }

  return {
    indices,
    useThroughPoses: indices.length >= 2,
  };
}

export function initialRunnerState(spots = []) {
  return {
    status: RunnerStatus.IDLE,
    currentIndex: -1,
    phase: RunnerPhase.NONE,
    reason: "",
    total: spots.length,
    activeIndices: [],
    progress: spots.map((spot) => ({
      id: spot.id,
      label: spot.label || spot.id,
      state: WaypointState.PENDING,
    })),
  };
}

function withProgress(state, index, waypointState) {
  if (index < 0 || index >= state.progress.length) return state.progress;
  return state.progress.map((entry, i) => (
    i === index ? { ...entry, state: waypointState } : entry
  ));
}

function withProgressIndices(state, indices, waypointState) {
  const selected = new Set(indices || []);
  if (!selected.size) return state.progress;
  return state.progress.map((entry, index) => (
    selected.has(index) ? { ...entry, state: waypointState } : entry
  ));
}

// Reducer over runner actions. Callers dispatch coarse lifecycle events; the
// async driver in useMissionRunner drives these in order.
export function missionRunnerReducer(state, action) {
  switch (action.type) {
    case "reset":
      return initialRunnerState(action.spots || []);
    case "start":
      return {
        ...state,
        status: RunnerStatus.STARTING,
        currentIndex: -1,
        phase: RunnerPhase.NONE,
        reason: "",
        activeIndices: [],
        progress: state.progress.map((entry) => ({ ...entry, state: WaypointState.PENDING })),
      };
    case "navigate":
      {
        const indices = action.indices || [action.index];
        return {
          ...state,
          status: RunnerStatus.NAVIGATING,
          currentIndex: action.index,
          activeIndices: indices,
          phase: RunnerPhase.NAV_SENT,
          progress: withProgressIndices(state, indices, WaypointState.NAVIGATING),
        };
      }
    case "phase":
      return { ...state, phase: action.phase };
    case "runBt":
      return {
        ...state,
        status: RunnerStatus.RUNNING_BT,
        currentIndex: action.index,
        activeIndices: [action.index],
        phase: RunnerPhase.BT_LOADING,
        progress: withProgress(state, action.index, WaypointState.RUNNING_BT),
      };
    case "finish":
      {
        const indices = action.indices || [action.index];
        const completed = new Set(indices);
        return {
          ...state,
          phase: RunnerPhase.BT_DONE,
          activeIndices: state.activeIndices.filter((index) => !completed.has(index)),
          progress: withProgressIndices(
            state,
            indices,
            action.skipped ? WaypointState.SKIPPED : WaypointState.DONE,
          ),
        };
      }
    case "advance":
      return { ...state, status: RunnerStatus.ADVANCING };
    case "done":
      return {
        ...state,
        status: RunnerStatus.DONE,
        phase: RunnerPhase.NONE,
        currentIndex: -1,
        activeIndices: [],
      };
    case "fail":
      {
        const indices = action.indices || (action.index >= 0 ? [action.index] : []);
        return {
          ...state,
          status: RunnerStatus.FAILED,
          reason: action.reason || "Mission failed",
          activeIndices: [],
          progress: indices.length
            ? withProgressIndices(state, indices, WaypointState.FAILED)
            : state.progress,
        };
      }
    case "cancel": {
      const resetActive = (
        state.status === RunnerStatus.NAVIGATING
        || state.status === RunnerStatus.RUNNING_BT
      );
      return {
        ...state,
        status: RunnerStatus.CANCELLED,
        currentIndex: -1,
        phase: RunnerPhase.NONE,
        reason: action.reason || "Cancelled",
        activeIndices: [],
        progress: resetActive
          ? withProgressIndices(state, state.activeIndices, WaypointState.PENDING)
          : state.progress,
      };
    }
    default:
      return state;
  }
}

export function isRunnerActive(status) {
  return (
    status === RunnerStatus.STARTING
    || status === RunnerStatus.NAVIGATING
    || status === RunnerStatus.RUNNING_BT
    || status === RunnerStatus.ADVANCING
  );
}
