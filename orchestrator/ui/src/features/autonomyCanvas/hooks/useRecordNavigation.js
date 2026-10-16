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

// One navigation owner visits saved observation points, independently of capture.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useMissionRunner from "../../../hooks/useMissionRunner";
import useRecordArrival, { nearObservation, observationPose } from "./useRecordArrival";

const initial = { status: "idle", kind: "patrol", route: [], index: 0, from: null, closed: false, reason: "", held: false, patrolPaused: false, dwellRemainingMs: null };
const ACTIVE = new Set(["switching", "preparing", "launching", "moving", "settling", "dwelling"]);
export function assertRecordStopped(result) {
  if (Array.isArray(result)) result.forEach(assertRecordStopped);
  else if (result?.status === "rejected") throw result.reason instanceof Error ? result.reason : new Error("Navigation cancellation was not confirmed.");
  else if (result?.status === "fulfilled") assertRecordStopped(result.value);
  else if (result === false || result?.ok === false || result?.success === false) throw new Error("Navigation cancellation was not confirmed.");
  return result;
}
function savedRoute(manager) {
  const plan = manager.configuration?.plan || {};
  const points = plan.observations || [];
  return points.map(point => {
    const spot = manager.spots?.find(value => value.id === point.spot_id);
    return spot ? { ...spot, arrival_bt: point.arrival_bt || "", dwell_seconds: Number(point.dwell_seconds) || 0, departure_policy: point.departure_policy || "reasoning" } : null;
  }).filter(Boolean);
}

export default function useRecordNavigation({ enabled, mapName, manager, localized, currentPose, sendGoal, cancelGoal, onMessage, runtime = {} }) {
  const [state, setState] = useState(initial);
  const live = useRef(initial), latest = useRef(null), generation = useRef(0), cursor = useRef(0), pending = useRef(null);
  const cleanup = useRef(Promise.resolve()), runnerRef = useRef(null), reached = useRef(null);
  const pausedSnapshot = useRef(null), dwellDeadline = useRef(null);
  const update = useCallback(changes => { live.current = { ...live.current, ...changes }; setState(live.current); }, []);
  latest.current = { enabled, mapName, localized, currentPose, manager, onMessage };
  const target = state.route[state.index];
  const orderedSpots = useMemo(() => target ? [target] : [], [target]);
  const runner = useMissionRunner({ ...runtime, orderedSpots, pausableNavigation: true,
    resolveBtXml: spot => spot?.arrival_bt || "", getFlags: () => ({ ...runtime.getFlags?.(), navRunning: enabled && localized }), sendGoal, cancelGoal, onMessage });
  runnerRef.current = runner;
  const settled = useRecordArrival({ enabled: state.status === "settling", localized,
    moving: runner.isRunning, pose: currentPose, spot: target });

  useEffect(() => {
    if (state.status !== "launching" || runner.isRunning) return;
    const started = runnerRef.current.start();
    update({ status: started ? "moving" : "failed", reason: started ? "" : "Observation navigation could not start." });
    pending.current?.(started); pending.current = null;
  }, [state.status, state.index, state.route, runner.isRunning, update]);
  useEffect(() => {
    if (state.status !== "moving" || runner.isRunning) return;
    if (runner.status === "done") update({ status: "settling" });
    else if (["failed", "cancelled"].includes(runner.status)) update({ status: runner.status, reason: runner.reason || "Observation navigation failed." });
  }, [state.status, runner.isRunning, runner.status, runner.reason, update]);
  useEffect(() => {
    if (state.status !== "settling" || (!settled && currentPose !== undefined)) return;
    reached.current = target?.id || null;
    latest.current.manager.select?.(target?.id);
    update({ status: state.kind === "goto" ? "done" : "dwelling" });
  }, [state.status, state.kind, settled, currentPose, target, update]);
  useEffect(() => {
    if (state.status !== "settling") return undefined;
    const timer = setTimeout(() => {
      if (live.current.status === "settling") update({ status: "failed", reason: "Robot did not settle at the observation. Check localization." });
    }, 10000);
    return () => clearTimeout(timer);
  }, [state.status, update]);
  useEffect(() => {
    if (state.status !== "dwelling" || target?.departure_policy !== "timed") {
      dwellDeadline.current = null;
      return undefined;
    }
    const token = generation.current;
    const duration = state.dwellRemainingMs ?? Math.max(0, target?.dwell_seconds || 0) * 1000;
    dwellDeadline.current = Date.now() + duration;
    const timer = setTimeout(() => {
      if (token !== generation.current || live.current.status !== "dwelling") return;
      if (state.closed && state.route.length === 1) return; // A single-point loop remains observing.
      const next = state.index + 1;
      if (next >= state.route.length && !state.closed) { cursor.current = 0; update({ status: "done" }); return; }
      cursor.current = next % state.route.length;
      update({ status: "launching", index: cursor.current, from: target?.id || null, dwellRemainingMs: null });
    }, duration);
    return () => clearTimeout(timer);
  }, [state.status, state.index, state.route, state.closed, state.dwellRemainingMs, target, update]);

  const stop = useCallback(() => {
    generation.current += 1;
    pending.current?.(false); pending.current = null;
    if (live.current.kind === "patrol" && ACTIVE.has(live.current.status)) cursor.current = live.current.index;
    pausedSnapshot.current = null;
    update({ status: "cancelled", patrolPaused: false, held: ["moving", "settling"].includes(live.current.status) || live.current.held });
    // Always serialize cancellation, including a goal whose response is late.
    cleanup.current = Promise.resolve(runnerRef.current.stop()).then(assertRecordStopped);
    return cleanup.current;
  }, [update]);
  // A snapshot belongs to a specific saved route and map. It never starts motion.
  const getPatrolContinuation = () => {
    const value = live.current;
    if (value.kind !== "patrol" || (!ACTIVE.has(value.status) && !value.patrolPaused)) return null;
    const paused = pausedSnapshot.current;
    return { mapName: latest.current.mapName, route: value.route, closed: value.closed, index: value.index,
      atObservation: value.status === "dwelling" || (value.patrolPaused && paused?.atObservation),
      dwellRemainingMs: value.patrolPaused ? paused?.dwellRemainingMs
        : Math.max(0, (dwellDeadline.current || Date.now()) - Date.now()) };
  };
  const pause = async () => {
    const snapshot = getPatrolContinuation();
    if (!snapshot || live.current.patrolPaused) return false;
    // Invalidate dwell timers and pending starts synchronously, then await cancellation.
    const token = generation.current + 1;
    const result = stop();
    pausedSnapshot.current = snapshot;
    update({ status: "paused", patrolPaused: true });
    await result;
    return token === generation.current;
  };
  const launch = async (spotId = null, continuation = null) => {
    const options = latest.current;
    if (!options.enabled || !options.localized || options.manager.planDirty || options.manager.draft || ACTIVE.has(live.current.status)) return false;
    const token = ++generation.current;
    update({ status: "preparing", reason: "", held: false, patrolPaused: false, dwellRemainingMs: null });
    try {
      await cleanup.current;
      const current = await options.manager.refresh();
      if (token !== generation.current || !latest.current.enabled || latest.current.mapName !== options.mapName) return false;
      const managerNow = { ...options.manager, spots: current.spots, configuration: { plan: current.plan } };
      const route = savedRoute(managerNow);
      const selected = spotId ? route.find(spot => spot.id === spotId) || current.spots.find(spot => spot.id === spotId) : null;
      if ((!spotId && !route.length) || (spotId && !selected)) throw new Error("Save an observation route before starting navigation.");
      const expected = savedRoute(options.manager);
      if (spotId ? JSON.stringify(current.spots.find(spot => spot.id === spotId)) !== JSON.stringify(options.manager.spots?.find(spot => spot.id === spotId)) : JSON.stringify(route) !== JSON.stringify(expected)) throw new Error("Observation route changed. Review it and start again.");
      let index = spotId ? 0 : Math.min(cursor.current, route.length - 1);
      if (continuation) {
        const closed = current.plan?.route_closed !== false;
        if (continuation.mapName !== options.mapName || continuation.closed !== closed
          || JSON.stringify(continuation.route) !== JSON.stringify(route)) {
          throw new Error("Observation route changed. Review it and start again.");
        }
        index = continuation.index;
        if (continuation.atObservation) {
          if (options.currentPose !== undefined && !nearObservation(observationPose(options.currentPose), route[index])) {
            throw new Error("Return to the observation before resuming this patrol.");
          }
          reached.current = route[index]?.id;
          cursor.current = index;
          pausedSnapshot.current = null;
          update({ status: "dwelling", route, index, kind: "patrol", closed,
            from: null, dwellRemainingMs: continuation.dwellRemainingMs });
          return true;
        }
        cursor.current = index;
      }
      pausedSnapshot.current = null;
      const origin = options.currentPose === undefined ? reached.current
        : route.find(spot => nearObservation(observationPose(options.currentPose), spot))?.id || null;
      return await new Promise(resolve => {
        pending.current = resolve;
        update({ status: "launching", kind: spotId ? "goto" : "patrol", route: spotId ? [selected] : route,
          index, from: origin === (spotId || route[index]?.id) ? null : origin, closed: !spotId && current.plan?.route_closed !== false });
      });
    } catch (error) {
      if (token === generation.current) { update({ status: "failed", reason: error.message }); options.onMessage?.(error.message); }
      if (spotId) throw error;
      return false;
    }
  };
  const canStepObservation = direction => {
    const options = latest.current, snapshot = getPatrolContinuation();
    return !!(options.enabled && options.localized && !options.manager.planDirty && !options.manager.draft
      && !options.manager.working && snapshot && snapshot.route.length > 1
      && ["moving", "settling", "dwelling", "paused"].includes(live.current.status)
      && (snapshot.closed || (snapshot.index + direction >= 0 && snapshot.index + direction < snapshot.route.length)));
  };
  const stepObservation = async direction => {
    if (!canStepObservation(direction)) return false;
    const snapshot = getPatrolContinuation();
    const destination = { ...snapshot,
      index: (snapshot.index + direction + snapshot.route.length) % snapshot.route.length,
      atObservation: false, dwellRemainingMs: null };
    // Cancel and acknowledge the old goal before dispatching its replacement.
    // Stop/map changes invalidate this request; repeated clicks during cancellation are ignored.
    const cancellation = stop();
    const token = generation.current;
    update({ status: "switching", held: false });
    try {
      await cancellation;
      if (token !== generation.current) return false;
      update({ status: "cancelled" });
      return launch(null, destination);
    } catch (error) {
      if (token === generation.current) {
        update({ status: "failed", reason: error.message });
        latest.current.onMessage?.(error.message);
      }
      return false;
    }
  };
  useEffect(() => {
    cursor.current = 0; reached.current = null; pausedSnapshot.current = null;
    return () => { generation.current += 1; pending.current?.(false); pending.current = null; };
  }, [mapName]);
  const resetIdle = useCallback(() => {
    if (ACTIVE.has(live.current.status) || runnerRef.current.isRunning) return false;
    generation.current += 1; cursor.current = 0; reached.current = null; pausedSnapshot.current = null;
    update(initial);
    return true;
  }, [update]);
  const isRunning = ACTIVE.has(state.status);
  const observing = ["dwelling", "done", "cancelled", "paused"].includes(state.status) && reached.current === target?.id;
  return { ...runner, start: () => launch(), goToSpot: id => launch(id), stop, pause, resetIdle,
    getPatrolContinuation, skipDwell: () => stepObservation(1), canSkipDwell: canStepObservation(1),
    previousObservation: () => stepObservation(-1), canPreviousObservation: canStepObservation(-1),
    resume: () => launch(null, getPatrolContinuation()),
    resumeAfterMission: continuation => launch(null, continuation), patrolPaused: state.patrolPaused,
    orderedSpots: state.route, currentIndex: state.index, total: state.route.length,
    status: state.held ? "paused" : state.status === "moving" ? runner.status : state.status,
    paused: state.held || runner.paused, held: state.held,
    phase: state.status === "moving" ? runner.phase : state.status,
    isRunning, reason: state.reason || runner.reason, kind: state.kind,
    dwelling: state.status === "dwelling", observationSpotId: observing ? reached.current : null,
    fromSpotId: state.from, toSpotId: target?.id || null, routeClosed: state.closed };
}
