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

// Record owns a task run and return to its pinned observation, independently of capture.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useMissionRunner from "../../../hooks/useMissionRunner";
import useRecordArrival from "./useRecordArrival";

const TERMINAL = new Set(["idle", "done", "failed", "cancelled"]);
const RETURN_PHASES = new Set(["return-starting", "returning", "return-settling", "return-arrived"]);
export function returnSnapshotForEpisode(episode) {
  const spot = episode?.return_observation_spot;
  if (!spot) return null;
  const point = episode.recording_plan?.observations?.find(value => value.spot_id === spot.id);
  return { ...spot, arrival_bt: spot.arrival_bt ?? point?.arrival_bt ?? "" };
}
function sameReturnTarget(expected, actual) {
  return expected.id === actual.id && ["x", "y", "yaw"].every(key => expected.pose?.[key] === actual.pose?.[key])
    && String(expected.arrival_bt || "").trim() === String(actual.arrival_bt || "").trim();
}

const initial = { phase: "idle", pinned: null, missionSeen: false, pendingPaused: false, reason: "" };

export default function useRecordMissionCycle({ enabled, mapName, localized, currentPose,
  mission, missionRunner, manager, startMission, cancelMissionStart, sendGoal, cancelGoal, onMessage, returnRuntime = {}, autoReturn = false }) {
  const [state, setState] = useState(initial);
  const live = useRef(initial), request = useRef(0), latest = useRef(null);
  const update = useCallback(changes => { const next = { ...live.current, ...changes }; live.current = next; setState(next); }, []);
  latest.current = { enabled, mapName, localized, manager, mission, startMission, cancelMissionStart, missionRunner };
  const orderedSpots = useMemo(() => state.pinned ? [state.pinned.spot] : [], [state.pinned]);
  const returnRunner = useMissionRunner({ ...returnRuntime, orderedSpots, pausableNavigation: true,
    resolveBtXml: spot => spot?.arrival_bt || "",
    getFlags: () => ({ ...returnRuntime.getFlags?.(), navRunning: enabled && localized }), sendGoal, cancelGoal, onMessage });
  const arrivalReady = useRecordArrival({ enabled: state.phase === "return-settling", localized,
    moving: returnRunner.isRunning, pose: currentPose, spot: state.pinned?.spot });

  useEffect(() => {
    const phase = live.current.phase;
    if (!["starting", "mission"].includes(phase)) return;
    if (missionRunner.isRunning) {
      if (phase !== "mission" || !live.current.missionSeen) update({ phase: "mission", missionSeen: true });
    } else if (live.current.missionSeen) {
      if (missionRunner.status === "done") update({ phase: "return-pending" });
      else if (["failed", "cancelled"].includes(missionRunner.status)) update({ phase: missionRunner.status, reason: missionRunner.reason || "" });
    }
  }, [state.phase, missionRunner.isRunning, missionRunner.status, missionRunner.reason, update]);

  useEffect(() => {
    if (live.current.phase !== "returning" || returnRunner.isRunning) return;
    if (returnRunner.status === "done") update({ phase: "return-settling" });
    else if (["failed", "cancelled"].includes(returnRunner.status)) update({ phase: returnRunner.status, reason: returnRunner.reason || "Return navigation failed." });
  }, [state.phase, returnRunner.isRunning, returnRunner.status, returnRunner.reason, update]);
  useEffect(() => {
    if (state.phase === "return-settling" && arrivalReady) update({ phase: "return-arrived" });
  }, [state.phase, arrivalReady, update]);
  useEffect(() => {
    if (state.phase !== "return-settling") return undefined;
    const timer = setTimeout(() => {
      if (live.current.phase === "return-settling") {
        const reason = "Return navigation completed, but the robot pose did not settle at the observation. Check localization.";
        update({ phase: "failed", reason }); onMessage?.(reason);
      }
    }, 10000);
    return () => clearTimeout(timer);
  }, [state.phase, onMessage, update]);
  useEffect(() => {
    if (state.phase !== "starting") return undefined;
    const timer = setTimeout(() => {
      if (live.current.phase !== "starting") return;
      request.current += 1; latest.current.cancelMissionStart?.();
      const reason = "Mission execution did not start. Check the selected route and navigation state, then retry.";
      update({ phase: "failed", reason }); onMessage?.(reason);
    }, 10000);
    return () => clearTimeout(timer);
  }, [state.phase, onMessage, update]);
  useEffect(() => () => { request.current += 1; }, []);

  const start = async (expectedReturnSpot = null, sourceObservationId = latest.current.manager.returnSpotId) => {
    const options = latest.current;
    if (!options.enabled || !options.localized || !TERMINAL.has(live.current.phase)
      || !options.mission?.missionId || !sourceObservationId) return false;
    const token = ++request.current;
    update({ ...initial, phase: "preparing" });
    try {
      const spot = await options.manager.resolveReturnSpot(sourceObservationId, options.mission.missionId);
      if (token !== request.current || !latest.current.enabled || latest.current.mapName !== options.mapName) return false;
      if (expectedReturnSpot && !sameReturnTarget(expectedReturnSpot, spot)) {
        throw new Error("The return observation moved after this episode was created, or its arrival posture changed. Start a new episode before running the mission.");
      }
      update({ phase: "starting", pinned: { mapName: options.mapName, missionId: options.mission.missionId, spot } });
      const started = await options.startMission();
      if (token !== request.current) return false;
      if (started === false) { update({ phase: "failed", reason: "Mission could not start." }); return false; }
      return true;
    } catch (error) {
      if (token === request.current) { update({ phase: "failed", reason: error.message }); onMessage?.(error.message); }
      return false;
    }
  };
  // Restoring a saved return child resumes only its return leg, never its task.
  const restoreReturn = async (spotId, missionId, expectedReturnSpot = null) => {
    if (!TERMINAL.has(live.current.phase)) return false;
    const options = latest.current, token = ++request.current;
    const spot = await options.manager.resolveReturnSpot(spotId, missionId);
    if (token !== request.current || !latest.current.enabled || latest.current.mapName !== options.mapName) return false;
    if (expectedReturnSpot && !sameReturnTarget(expectedReturnSpot, spot)) {
      throw new Error("The saved return observation or arrival posture changed. Start a new recording session.");
    }
    update({ ...initial, phase: "return-pending", pinned: { mapName: options.mapName, missionId, spot } });
    return true;
  };
  const startReturn = async () => {
    const finishedMission = live.current.phase === "mission" && live.current.missionSeen
      && !latest.current.missionRunner.isRunning && latest.current.missionRunner.status === "done";
    if (!(live.current.phase === "return-pending" || finishedMission) || !live.current.pinned) return false;
    const token = request.current;
    update({ phase: "return-starting" });
    // Let the return runner receive the pinned route after restoring an episode.
    await new Promise(resolve => setTimeout(resolve, 0));
    while (live.current.pendingPaused && token === request.current) await new Promise(resolve => setTimeout(resolve, 100));
    if (token !== request.current || live.current.phase !== "return-starting" || !latest.current.enabled || !latest.current.localized) return false;
    if (!returnRunner.start()) { update({ phase: "failed", reason: "Return navigation could not start." }); return false; }
    update({ phase: "returning" });
    return true;
  };
  useEffect(() => {
    if (autoReturn && state.phase === "return-pending") void startReturn();
    // Each pending transition starts once; startReturn synchronously advances phase.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoReturn, state.phase]);
  useEffect(() => {
    if (autoReturn && state.phase === "return-arrived") update({ phase: "done" });
  }, [autoReturn, state.phase, update]);
  const stop = async () => {
    const phase = live.current.phase;
    request.current += 1; latest.current.cancelMissionStart?.();
    update({ phase: "cancelled", pendingPaused: false });
    if (RETURN_PHASES.has(phase)) return returnRunner.stop();
    if (["starting", "mission"].includes(phase)) return latest.current.missionRunner.stop();
    return [];
  };
  const pause = () => {
    if (["return-pending", "return-starting"].includes(live.current.phase)) { update({ pendingPaused: true }); return Promise.resolve(true); }
    if (live.current.phase === "returning" && returnRunner.status === "running-bt") {
      return Promise.reject(new Error("The observation posture is being restored. Use Stop to cancel it."));
    }
    return live.current.phase === "returning" ? returnRunner.pause() : latest.current.missionRunner.pause();
  };
  const resume = () => {
    if (live.current.pendingPaused) { update({ pendingPaused: false }); return; }
    return live.current.phase === "returning" ? returnRunner.resume() : latest.current.missionRunner.resume();
  };
  const resetIdle = () => {
    if (!TERMINAL.has(live.current.phase) || missionRunner.isRunning || returnRunner.isRunning) return false;
    request.current += 1; update(initial);
    return true;
  };
  const returning = ["returning", "return-settling", "return-arrived"].includes(state.phase);
  const missionDone = state.phase === "mission" && state.missionSeen && !missionRunner.isRunning && missionRunner.status === "done";
  const navigatingReturn = state.phase === "returning" && (returnRunner.isRunning || returnRunner.status !== "done");
  const currentRunner = navigatingReturn ? returnRunner : missionRunner;
  return { ...currentRunner, start, stop, pause, resume, restoreReturn, startReturn, resetIdle,
    completeReturn: () => { if (live.current.phase === "return-arrived") update({ phase: "done" }); },
    cyclePhase: state.phase, isRunning: !TERMINAL.has(state.phase), status: state.phase === "mission" ? missionRunner.status : state.phase,
    phase: navigatingReturn ? returnRunner.phase : returning ? "settling" : missionRunner.phase,
    paused: state.pendingPaused || (navigatingReturn ? returnRunner.paused : state.phase === "mission" && missionRunner.paused),
    pausing: currentRunner.pausing, pauseFailed: currentRunner.pauseFailed, reason: state.reason || currentRunner.reason,
    returnPending: state.phase === "return-pending" || missionDone, returning, returnArrived: state.phase === "return-arrived",
    pinnedSpot: state.pinned?.spot || null, pinnedMissionId: state.pinned?.missionId || null,
    returnPhase: navigatingReturn && !["arrived", "bt-loading", "bt-running", "bt-done"].includes(returnRunner.phase) ? "returning" : "settling",
    returnSpotId: !TERMINAL.has(state.phase) ? state.pinned?.spot.id || null : null,
    returnSpotName: !TERMINAL.has(state.phase) ? state.pinned?.spot.label || null : null,
    fromSpotId: null, toSpotId: state.pinned?.spot.id || null,
    currentIndex: returning ? 0 : missionRunner.currentIndex,
  };
}
