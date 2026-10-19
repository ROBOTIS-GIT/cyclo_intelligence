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

// Frontend-orchestrated Mission Runner. Consecutive nav-only waypoints are
// batched with NavigateThroughPoses; BT endpoints retain the authoritative
// Nav2 SUCCEEDED → local BT ordering. Live signals are read through refs so the
// async loop never sees stale React values. Cancellation aborts the loop and
// best-effort stops the robot.

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import {
  DEFAULT_RUNNER_CONFIG,
  RunnerPhase,
  RunnerStatus,
  goalFromSpot,
  initialRunnerState,
  isEmptyBt,
  isRunnerActive,
  missionRunnerReducer,
  navigationBatchFromIndex,
} from "./missionRunnerCore";
import { withMissionWaypointContext } from "../utils/missionWaypointContext";
import { formatTaskDisplayMessage } from "../utils/taskTerminology";

const normStatus = (value) => String(value || "").trim().toLowerCase();
const taskDisplayMessage = (value) => formatTaskDisplayMessage(value, "Waypoint Task");

// Sleep that rejects as soon as the abort signal fires, so a Stop mid-wait
// unwinds the driver loop immediately instead of after the next tick.
function cancellableSleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

const isAbort = (error) => error && error.name === "AbortError";

function throwIfAborted(signal) {
  if (signal.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

export function useMissionRunner({
  orderedSpots,
  missionWaypoints,
  pausableNavigation = false,
  resolveBtXml,
  btStatusRef,
  btStatusHistoryRef,
  callService,
  sendGoal,
  sendGoals,
  cancelGoal,
  stopBt,
  ensureBtActive,
  releaseBt,
  getFlags,
  onMessage,
  config: configOverride,
} = {}) {
  const config = useMemo(
    () => ({ ...DEFAULT_RUNNER_CONFIG, ...(configOverride || {}) }),
    [configOverride],
  );

  const [state, dispatch] = useReducer(missionRunnerReducer, orderedSpots || [], initialRunnerState);
  const [lifecycleBusy, setLifecycleBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [pausing, setPausing] = useState(false);
  const [pauseFailed, setPauseFailed] = useState(false);
  const pauseFailedRef = useRef(false);
  const pauseRef = useRef(false);
  const pausingRef = useRef(false);
  const navAttemptRef = useRef(null);
  const pauseCleanupRef = useRef(Promise.resolve());

  // Everything the loop needs, read live through refs (no stale closures).
  const spotsRef = useRef(orderedSpots || []);
  const missionWaypointsRef = useRef(missionWaypoints || orderedSpots || []);
  const resolveBtXmlRef = useRef(resolveBtXml);
  const callServiceRef = useRef(callService);
  const sendGoalRef = useRef(sendGoal);
  const sendGoalsRef = useRef(sendGoals);
  const cancelGoalRef = useRef(cancelGoal);
  const stopBtRef = useRef(stopBt);
  const ensureBtActiveRef = useRef(ensureBtActive);
  const releaseBtRef = useRef(releaseBt);
  const getFlagsRef = useRef(getFlags);
  const onMessageRef = useRef(onMessage);
  const configRef = useRef(config);
  const abortRef = useRef(null);
  const isRunningRef = useRef(false);
  const stopCleanupRef = useRef(Promise.resolve());
  const btLoadPromiseRef = useRef(null);
  const btNeedsStopRef = useRef(false);

  useEffect(() => { spotsRef.current = orderedSpots || []; }, [orderedSpots]);
  useEffect(() => { missionWaypointsRef.current = missionWaypoints || orderedSpots || []; }, [missionWaypoints, orderedSpots]);

  // Re-seed the observable progress list when the route changes while idle, so
  // the panel reflects the loaded mission before a run begins. Never clobber a
  // run in flight.
  useEffect(() => {
    if (!isRunningRef.current) {
      dispatch({ type: "reset", spots: orderedSpots || [] });
    }
  }, [orderedSpots]);
  useEffect(() => { resolveBtXmlRef.current = resolveBtXml; }, [resolveBtXml]);
  useEffect(() => { callServiceRef.current = callService; }, [callService]);
  useEffect(() => { sendGoalRef.current = sendGoal; }, [sendGoal]);
  useEffect(() => { sendGoalsRef.current = sendGoals; }, [sendGoals]);
  useEffect(() => { cancelGoalRef.current = cancelGoal; }, [cancelGoal]);
  useEffect(() => { stopBtRef.current = stopBt; }, [stopBt]);
  useEffect(() => { ensureBtActiveRef.current = ensureBtActive; }, [ensureBtActive]);
  useEffect(() => { releaseBtRef.current = releaseBt; }, [releaseBt]);
  useEffect(() => { getFlagsRef.current = getFlags; }, [getFlags]);
  useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
  useEffect(() => { configRef.current = config; }, [config]);

  const emit = useCallback((message) => {
    if (message && typeof onMessageRef.current === "function") onMessageRef.current(message);
  }, []);

  // A navigation-only mission must never stop a tree owned by the standalone
  // workspace. Once this runner has issued /bt/load_and_run, however, it owns
  // that execution and Stop must be ordered after any in-flight load request;
  // otherwise a late load response can restart the tree after Stop returned.
  const stopEngagedBt = useCallback(async () => {
    const pendingLoad = btLoadPromiseRef.current;
    if (pendingLoad) {
      try { await pendingLoad; } catch { /* ownership is still best-effort */ }
    }
    if (!btNeedsStopRef.current || typeof stopBtRef.current !== "function") return;
    await stopBtRef.current();
    btNeedsStopRef.current = false;
  }, []);

  // Keep the newly-loaded tree ticking until it reports a FRESH terminal status.
  // The engine latches the previous run's `completed`, so we only accept a
  // terminal once we've seen `running` (or the status object identity changed).
  const awaitBtTerminal = useCallback(async (signal, baselineSequence) => {
    const cfg = configRef.current;
    const statusAtLoad = btStatusRef.current;
    let sawRunning = false;
    const startDeadline = Date.now() + cfg.btStartTimeoutMs;
    for (;;) {
      const status = normStatus(btStatusRef.current);
      const history = btStatusHistoryRef?.current;
      if (history && history.runningSequence > baselineSequence) sawRunning = true;
      if (history && history.runningSequence > baselineSequence && history.terminalSequence > history.runningSequence) {
        return history.terminalStatus;
      }
      if (status === "running") sawRunning = true;
      const fresh = sawRunning || btStatusRef.current !== statusAtLoad;
      if (fresh && status === "completed") return "completed";
      if (fresh && status === "failed") return "failed";
      if (!sawRunning && Date.now() > startDeadline) return "nostart";
      await cancellableSleep(cfg.pollMs, signal);
    }
  }, [btStatusRef, btStatusHistoryRef]);

  const runWaypointBt = useCallback(async (index, xml, signal) => {
    const spot = spotsRef.current[index];
    const label = (spot && (spot.label || spot.id)) || `Waypoint ${index + 1}`;

    dispatch({ type: "runBt", index });
    // A short tree can publish running and completed before the service reply.
    // Keep the event sequence from before load so neither transition is lost.
    const baselineSequence = btStatusHistoryRef?.current?.sequence ?? 0;
    let loadResult;
    const loadPromise = Promise.resolve().then(() => callServiceRef.current(
      "/bt/load_and_run",
      "interfaces/srv/LoadAndRunTree",
      { tree_xml: withMissionWaypointContext(xml, missionWaypointsRef.current) },
      30000,
    ));
    btNeedsStopRef.current = true;
    btLoadPromiseRef.current = loadPromise;
    try {
      loadResult = await loadPromise;
      throwIfAborted(signal);
    } catch (error) {
      if (signal.aborted || isAbort(error)) throw error;
      dispatch({ type: "fail", reason: `Waypoint Task failed to load at ${label}: ${taskDisplayMessage(error.message || error)}`, index });
      return false;
    } finally {
      if (btLoadPromiseRef.current === loadPromise) btLoadPromiseRef.current = null;
    }
    if (loadResult && loadResult.success === false) {
      btNeedsStopRef.current = false;
      dispatch({ type: "fail", reason: `Waypoint Task was rejected at ${label}: ${taskDisplayMessage(loadResult.message)}`, index });
      return false;
    }
    dispatch({ type: "phase", phase: RunnerPhase.BT_RUNNING });

    // Running trees may loop forever; individual actions own their deadlines.
    const outcome = await awaitBtTerminal(signal, baselineSequence);
    if (outcome === "completed") {
      btNeedsStopRef.current = false;
      dispatch({ type: "finish", index, skipped: false });
      return true;
    }
    const reasonByOutcome = {
      failed: `Waypoint Task failed at ${label}`,
      nostart: `Waypoint Task did not start at ${label}`,
    };
    if (outcome === "failed") {
      btNeedsStopRef.current = false;
    } else if (outcome === "nostart") {
      try { await stopEngagedBt(); } catch (error) { /* best-effort */ }
    }
    dispatch({ type: "fail", reason: reasonByOutcome[outcome] || `Waypoint Task error at ${label}`, index });
    return false;
  }, [awaitBtTerminal, btStatusHistoryRef, stopEngagedBt]);

  // Record keeps each unfinished waypoint pending while its current goal is cancelled.
  const sendPausableGoal = useCallback(async (goal, signal) => {
    while (true) {
      while (pauseRef.current) await cancellableSleep(50, signal);
      throwIfAborted(signal);
      await pauseCleanupRef.current;
      throwIfAborted(signal);
      if (pauseRef.current) continue;
      const attempt = new AbortController();
      const abort = () => attempt.abort();
      signal.addEventListener("abort", abort, { once: true });
      navAttemptRef.current = attempt;
      try {
        const result = await sendGoalRef.current(goal.x, goal.y, goal.yaw, attempt.signal);
        throwIfAborted(signal);
        if (!attempt.signal.aborted) return result;
      } catch (error) {
        throwIfAborted(signal);
        if (!attempt.signal.aborted) throw error;
      } finally {
        signal.removeEventListener("abort", abort);
        if (navAttemptRef.current === attempt) navAttemptRef.current = null;
      }
    }
  }, []);

  const pause = useCallback(async () => {
    if (!pausableNavigation || !isRunningRef.current || pausingRef.current) return;
    pauseRef.current = true; setPaused(true);
    pausingRef.current = true; setPausing(true);
    navAttemptRef.current?.abort();
    const cleanup = Promise.resolve().then(() => cancelGoalRef.current?.());
    pauseCleanupRef.current = cleanup.catch(() => {});
    try { await cleanup; pauseFailedRef.current = false; setPauseFailed(false); }
    catch (error) { pauseFailedRef.current = true; setPauseFailed(true); emit("Pause failed. Retry Pause before resuming."); throw error; }
    finally { pausingRef.current = false; setPausing(false); }
  }, [pausableNavigation, emit]);

  const resume = useCallback(() => {
    if (!isRunningRef.current || pausingRef.current || pauseFailedRef.current) return;
    pauseRef.current = false; setPaused(false);
  }, []);

  const runNavigationBatch = useCallback(async (startIndex, signal) => {
    const spots = spotsRef.current;
    const resolveXml = (spot) => (
      resolveBtXmlRef.current ? resolveBtXmlRef.current(spot) : ""
    );
    const batch = pausableNavigation
      ? { indices: startIndex < spots.length ? [startIndex] : [], useThroughPoses: false }
      : navigationBatchFromIndex(spots, startIndex, resolveXml);
    const indices = batch.indices;
    if (!indices.length) {
      dispatch({
        type: "fail",
        reason: `Unable to build navigation batch at waypoint ${startIndex + 1}`,
        index: startIndex,
      });
      return { ok: false, nextIndex: spots.length };
    }

    const goals = indices.map((index) => goalFromSpot(spots[index]));
    const labels = indices.map((index) => {
      const spot = spots[index];
      return (spot && (spot.label || spot.id)) || `Waypoint ${index + 1}`;
    });
    const batchLabel = labels.join(" → ");

    dispatch({ type: "navigate", index: startIndex, indices });
    dispatch({ type: "phase", phase: RunnerPhase.AWAITING_NAV_RESULT });

    let navigationResult;
    try {
      if (pausableNavigation) {
        navigationResult = await sendPausableGoal(goals[0], signal);
      } else if (batch.useThroughPoses) {
        if (typeof sendGoalsRef.current !== "function") {
          throw new Error("NavigateThroughPoses client is unavailable");
        }
        navigationResult = await sendGoalsRef.current(goals, signal);
      } else {
        const [goal] = goals;
        navigationResult = await sendGoalRef.current(
          goal.x,
          goal.y,
          goal.yaw,
          signal,
        );
      }
      throwIfAborted(signal);
    } catch (error) {
      if (signal.aborted || isAbort(error)) throw error;
      dispatch({
        type: "fail",
        reason: `Navigation request failed at ${batchLabel}: ${error.message || error}`,
        index: startIndex,
        indices,
      });
      return { ok: false, nextIndex: indices[indices.length - 1] + 1 };
    }

    const navigationStatus = String(navigationResult?.status || "UNKNOWN").toUpperCase();
    if (navigationStatus !== "SUCCEEDED") {
      dispatch({
        type: "fail",
        reason: `Navigation ${navigationStatus.toLowerCase()} at ${batchLabel}`,
        index: startIndex,
        indices,
      });
      return { ok: false, nextIndex: indices[indices.length - 1] + 1 };
    }
    dispatch({ type: "phase", phase: RunnerPhase.ARRIVED });

    const endpointIndex = indices[indices.length - 1];
    const endpointXml = resolveXml(spots[endpointIndex]);
    const skippedIndices = indices.filter((index) => isEmptyBt(resolveXml(spots[index])));
    if (skippedIndices.length) {
      dispatch({
        type: "finish",
        index: skippedIndices[0],
        indices: skippedIndices,
        skipped: true,
      });
    }

    if (isEmptyBt(endpointXml)) {
      return { ok: true, nextIndex: endpointIndex + 1 };
    }

    const ok = await runWaypointBt(endpointIndex, endpointXml, signal);
    return { ok, nextIndex: endpointIndex + 1 };
  }, [runWaypointBt, pausableNavigation, sendPausableGoal]);

  const start = useCallback(() => {
    if (isRunningRef.current) return false;
    const spots = spotsRef.current;
    if (!spots.length) {
      emit("No route to run — connect waypoints in Design first");
      return false;
    }
    const flags = getFlagsRef.current ? getFlagsRef.current() : {};
    if (!flags.navRunning) {
      emit("Start navigation before running the mission");
      return false;
    }
    const needsBt = spots.some((spot) => !isEmptyBt(resolveBtXmlRef.current ? resolveBtXmlRef.current(spot) : ""));
    // Without an activation callback the caller owns the BT lifecycle
    // (legacy behavior): refuse to start while the node is down.
    if (needsBt && !flags.btNodeIsUp && !ensureBtActiveRef.current) {
      emit("Activate the Task Engine before running the mission");
      return false;
    }

    pauseRef.current = false; setPaused(false);
    pauseFailedRef.current = false; setPauseFailed(false);
    const controller = new AbortController();
    abortRef.current = controller;
    isRunningRef.current = true;
    stopCleanupRef.current = Promise.resolve();
    btLoadPromiseRef.current = null;
    btNeedsStopRef.current = false;
    setLifecycleBusy(true);
    dispatch({ type: "start" });

    (async () => {
      try {
        // Supervisor "up" only means the launcher process exists. Always let
        // the owner verify ROS service readiness when a mission uses BTs;
        // ensureBtActive remains responsible for avoiding a redundant start.
        if (needsBt && ensureBtActiveRef.current) {
          emit(flags.btNodeIsUp ? "Checking Task Engine" : "Activating Task Engine");
          let activated = false;
          try {
            activated = !!(await ensureBtActiveRef.current());
          } catch {
            activated = false;
          }
          if (controller.signal.aborted) return;
          if (!activated) {
            dispatch({ type: "fail", reason: "Task Engine failed to activate", index: -1 });
            return;
          }
        }
        let index = 0;
        while (index < spotsRef.current.length) {
          if (controller.signal.aborted) return;
          if (index > 0) dispatch({ type: "advance" });
          const result = await runNavigationBatch(index, controller.signal);
          if (!result.ok) return;
          index = result.nextIndex;
        }
        dispatch({ type: "done" });
        emit("Mission complete");
      } catch (error) {
        if (!controller.signal.aborted && !isAbort(error)) {
          dispatch({ type: "fail", reason: taskDisplayMessage(error.message) || "Mission error", index: -1 });
        }
      } finally {
        // A Stop may still be cancelling Nav2 or stopping the current tree.
        // Finish those calls, then release the process, before admitting the
        // next run. Otherwise late cleanup from this controller can cancel or
        // shut down resources already acquired by a newer run.
        try {
          await stopCleanupRef.current;
          await pauseCleanupRef.current;
          await stopEngagedBt().catch(() => {});
          if (needsBt && releaseBtRef.current) {
            await Promise.resolve().then(releaseBtRef.current).catch(() => {});
          }
        } finally {
          if (abortRef.current === controller) {
            abortRef.current = null;
            isRunningRef.current = false;
            setLifecycleBusy(false);
          }
        }
      }
    })();
    return true;
  }, [emit, runNavigationBatch, stopEngagedBt]);

  const stop = useCallback(() => {
    const controller = abortRef.current;
    if (controller) controller.abort();
    pauseRef.current = false; setPaused(false);
    dispatch({ type: "cancel" });
    const cleanup = Promise.allSettled([
      Promise.resolve().then(() => (cancelGoalRef.current ? cancelGoalRef.current() : null)),
      Promise.resolve().then(stopEngagedBt),
    ]);
    stopCleanupRef.current = cleanup;
    cleanup.then((results) => {
      if (results.some((r) => r.status === "rejected")) {
        emit("Stop sent, but the robot may still be executing");
      }
    });
    // Callers that also tear down the navigation runtime must be able to keep
    // their Stop lock engaged until the cancel-all request has settled. If a
    // new runtime starts before this cleanup finishes, the late cancellation
    // can otherwise kill a goal from that new session.
    return cleanup;
  }, [emit, stopEngagedBt]);

  // Abort on unmount, but only if a run is genuinely in flight (guards against
  // React StrictMode's double effect invocation spuriously cancelling).
  useEffect(() => () => {
    if (isRunningRef.current && abortRef.current) {
      abortRef.current.abort();
      isRunningRef.current = false;
      if (cancelGoalRef.current) Promise.resolve().then(cancelGoalRef.current).catch(() => {});
      Promise.resolve().then(stopEngagedBt).catch(() => {});
    }
  }, [stopEngagedBt]);

  const activeSpotId = (
    state.currentIndex >= 0
    && state.activeIndices.includes(state.currentIndex)
    && state.progress[state.currentIndex]
  )
    ? state.progress[state.currentIndex].id
    : "";

  return {
    pause, resume, paused, pausing, pauseFailed,
    status: state.status,
    phase: state.phase,
    currentIndex: state.currentIndex,
    total: state.total,
    progress: state.progress,
    reason: state.reason,
    isRunning: isRunnerActive(state.status) || lifecycleBusy,
    activeSpotId,
    start,
    stop,
    RunnerStatus,
    RunnerPhase,
  };
}

export default useMissionRunner;
