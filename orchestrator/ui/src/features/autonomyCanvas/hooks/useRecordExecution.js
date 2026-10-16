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

// Record navigation, child tasks and their returns share one execution owner.
// Camera capture deliberately has no role in admitting or completing motion.
import { useCallback, useEffect, useRef, useState } from "react";
import { missionsAtObservation } from "./useRecordObservationSpots";
import useRecordNavigation, { assertRecordStopped } from "./useRecordNavigation";
import useRecordMissionCycle from "./useRecordMissionCycle";
import useRecordArrival, { nearObservation, observationPose } from "./useRecordArrival";
import { buildRecordExecutionState, recordExecutionType, recordNavigationActive } from "../lib/recordExecutionState";

export default function useRecordExecution({ enabled, mapName, localized, currentPose, manager,
  mission, missionRunner, startMission, cancelMissionStart, sendGoal, cancelGoal, onMessage, runtime = {} }) {
  const navigation = useRecordNavigation({ enabled, mapName, localized, currentPose, manager,
    sendGoal, cancelGoal, onMessage, runtime });
  const cycle = useRecordMissionCycle({ enabled, mapName, localized, currentPose, mission, missionRunner, manager,
    startMission, cancelMissionStart, sendGoal, cancelGoal, onMessage, returnRuntime: runtime, autoReturn: true });
  const [owner, setOwner] = useState(null);
  const [stopping, setStopping] = useState(false);
  const [cancelFailed, setCancelFailed] = useState(false);
  const [pendingControl, setPendingControl] = useState(null);
  const [controlError, setControlError] = useState(null);
  const controlEpoch = useRef(0), retainedExecution = useRef(null);
  const cancelUnconfirmed = useRef(false);
  const stopCount = useRef(0);
  const activeOwner = useRef(null), latest = useRef(null), generation = useRef(0), cleanup = useRef(Promise.resolve());
  const lastObservation = useRef(null), patrolContinuation = useRef(null);
  const navMoving = navigation.isRunning && !navigation.dwelling;
  const moving = cancelFailed || stopping || navMoving || cycle.isRunning || missionRunner.isRunning;
  const spots = manager.spots || [];
  const planIds = new Set((manager.configuration?.plan?.observations || []).map(point => point.spot_id));
  const reachedId = navigation.observationSpotId || cycle.pinnedSpot?.id;
  const reached = spots.find(spot => spot.id === reachedId && nearObservation(observationPose(currentPose), spot));
  const candidate = reached || spots.find(spot => planIds.has(spot.id) && nearObservation(observationPose(currentPose), spot));
  const poseSettled = useRecordArrival({ enabled, localized, moving, pose: currentPose, spot: candidate });
  const runnerSettled = !moving && nearObservation(observationPose(currentPose), candidate)
    && ((navigation.observationSpotId === candidate?.id && (navigation.dwelling || navigation.status === "done" || (navigation.patrolPaused && !navigation.held)))
      || (owner === "mission" && cycle.status === "done" && cycle.pinnedSpot?.id === candidate?.id));
  const observationReady = enabled && localized && (poseSettled || runnerSettled);
  const observationSpotId = observationReady ? candidate?.id || null : null;
  const missionObservationId = cycle.isRunning ? cycle.pinnedSpot?.id : observationSpotId;
  const linkedMissions = missionsAtObservation(manager.configuration, missionObservationId);
  const missionAllowed = observationReady && linkedMissions.some(item => item.mission_id === mission?.missionId);
  latest.current = { enabled, mapName, localized, currentPose, navigation, cycle, missionRunner, mission, manager, observationReady, observationSpotId, missionAllowed, cancelMissionStart, onMessage };
  useEffect(() => {
    const identity = `${mapName}/${observationSpotId || ""}`;
    if (lastObservation.current === identity) return;
    lastObservation.current = identity;
    if (owner && observationSpotId && !manager.route?.routeMode && !manager.draft && !manager.planDirty
      && manager.selectedId !== observationSpotId) manager.select?.(observationSpotId);
  }, [mapName, observationSpotId, manager, owner]);

  const trackCleanup = useCallback((promise, confirmedStop = null, kind = "cancelling") => {
    const controlToken = ++controlEpoch.current;
    retainedExecution.current = latest.current.executionState;
    setPendingControl({ kind, token: controlToken });
    stopCount.current += 1; setStopping(true);
    return promise.then(result => {
      // Only a successful explicit Stop clears an unconfirmed cancellation.
      if (confirmedStop !== null && confirmedStop === generation.current) {
        cancelUnconfirmed.current = false; setCancelFailed(false); setControlError(null);
      }
      return result;
    }).catch(error => {
      cancelUnconfirmed.current = true; setCancelFailed(true); setControlError(error.message || String(error));
      throw error;
    }).finally(() => {
      stopCount.current -= 1;
      if (!stopCount.current) setStopping(false);
      setPendingControl(current => current?.token === controlToken ? null : current);
    });
  }, []);
  const stop = useCallback(() => {
    patrolContinuation.current = null;
    const stopToken = ++generation.current;
    latest.current.cancelMissionStart?.();
    const current = latest.current;
    // Queue all cancellation before admitting another owner. Both hooks invalidate
    // pending refresh/start continuations synchronously inside stop().
    const navStop = current.navigation.stop();
    const missionStop = current.cycle.stop();
    activeOwner.current = null; setOwner(null);
    cleanup.current = trackCleanup(Promise.all([navStop, missionStop]).then(assertRecordStopped), stopToken);
    return cleanup.current;
  }, [trackCleanup]);
  const launchNavigation = async id => {
    const current = latest.current;
    const blockedReason = stopCount.current ? "Wait for navigation to finish stopping."
      : cancelUnconfirmed.current ? "The previous navigation cancellation was not confirmed. Press Stop again before moving."
      : !current.enabled ? "Load a map in Record before moving."
      : !current.localized ? "Localize the robot before moving to a waypoint."
      : current.manager.planDirty ? "Save the observation route changes before moving."
      : current.manager.route?.routeMode ? "Finish editing the observation route before moving."
      : current.manager.draft ? "Save or cancel the new observation waypoint before moving."
      : current.cycle.isRunning || current.missionRunner.isRunning ? "Stop the running mission before moving to another waypoint." : null;
    if (blockedReason) {
      if (id) throw new Error(blockedReason);
      return false;
    }
    patrolContinuation.current = null;
    const token = ++generation.current;
    // Retargeting first stops the previous goal and awaits its cancellation.
    const navStop = current.navigation.stop();
    cleanup.current = trackCleanup(Promise.all([cleanup.current, navStop]), null, "starting");
    await cleanup.current;
    if (token !== generation.current || !latest.current.enabled || latest.current.mapName !== current.mapName) return false;
    activeOwner.current = id ? "goto" : "patrol"; setOwner(activeOwner.current);
    return id ? latest.current.navigation.goToSpot(id) : latest.current.navigation.start();
  };
  const startChildMission = async () => {
    const current = latest.current;
    if (stopCount.current || cancelUnconfirmed.current || !current.enabled || !current.localized || current.cycle.isRunning || current.missionRunner.isRunning
      || current.manager.planDirty || current.manager.route?.routeMode || current.manager.draft || !current.missionAllowed) return false;
    const snapshot = current.navigation.getPatrolContinuation();
    const sourceIndex = snapshot?.route.findIndex(spot => spot.id === current.observationSpotId) ?? -1;
    const continuation = sourceIndex < 0 ? null : { ...snapshot, index: sourceIndex, atObservation: true };
    const token = ++generation.current;
    await trackCleanup(current.navigation.stop(), null, "starting");
    await cleanup.current;
    if (token !== generation.current || !latest.current.enabled || latest.current.mapName !== current.mapName
      || !latest.current.localized
      || latest.current.mission?.missionId !== current.mission?.missionId
      || !nearObservation(observationPose(latest.current.currentPose), latest.current.manager.spots.find(spot => spot.id === current.observationSpotId))
      || !missionsAtObservation(latest.current.manager.configuration, current.observationSpotId).some(item => item.mission_id === current.mission?.missionId)) return false;
    activeOwner.current = "mission"; setOwner("mission");
    patrolContinuation.current = continuation?.atObservation
      ? { route: continuation, token, seen: false } : null;
    const started = await latest.current.cycle.start(null, current.observationSpotId);
    if (!started && token === generation.current) patrolContinuation.current = null;
    return started;
  };
  const pausePatrol = async () => {
    const current = latest.current;
    if (stopCount.current || cancelUnconfirmed.current || current.cycle.isRunning
      || current.missionRunner.isRunning || !current.navigation.isRunning) return false;
    return trackCleanup(current.navigation.pause(), null, "pausing");
  };
  const resumePatrol = async () => {
    const current = latest.current;
    if (stopCount.current || cancelUnconfirmed.current || !current.enabled || !current.localized
      || current.cycle.isRunning || current.missionRunner.isRunning || !current.navigation.patrolPaused) return false;
    activeOwner.current = "patrol"; setOwner("patrol");
    const controlToken = ++controlEpoch.current;
    setPendingControl({ kind: "resuming", token: controlToken });
    try { return await current.navigation.resume(); }
    finally { setPendingControl(value => value?.token === controlToken ? null : value); }
  };
  const stepObservation = async (previous = false) => {
    const current = latest.current;
    if (stopCount.current || cancelUnconfirmed.current || !current.enabled || !current.localized
      || current.cycle.isRunning || current.missionRunner.isRunning
      || !(previous ? current.navigation.canPreviousObservation : current.navigation.canSkipDwell)) return false;
    return previous ? current.navigation.previousObservation() : current.navigation.skipDwell();
  };
  useEffect(() => {
    const continuation = patrolContinuation.current;
    if (!continuation) return;
    if (!enabled || !localized || continuation.token !== generation.current
      || continuation.route.mapName !== mapName || ["failed", "cancelled"].includes(cycle.cyclePhase)) {
      patrolContinuation.current = null;
      return;
    }
    if (cycle.isRunning) continuation.seen = true;
    if (cycle.cyclePhase !== "done" || !continuation.seen) return;
    // The mission cycle includes the pinned return goal, its BT and pose settling.
    // Consume the intent before awaiting refresh so Stop can invalidate this launch.
    patrolContinuation.current = null;
    if (cancelUnconfirmed.current || !latest.current.cycle.resetIdle()) return;
    activeOwner.current = "patrol"; setOwner("patrol");
    void latest.current.navigation.resumeAfterMission(continuation.route).catch(error => latest.current.onMessage?.(error.message));
  }, [enabled, localized, mapName, cycle.cyclePhase, cycle.isRunning]);
  useEffect(() => {
    if (!enabled && activeOwner.current) void stop().catch(error => latest.current.onMessage?.(error.message));
  }, [enabled, stop]);
  useEffect(() => () => {
    generation.current += 1; patrolContinuation.current = null;
    if (activeOwner.current) {
      latest.current.cancelMissionStart?.();
      void latest.current.navigation.stop().catch(() => {}); void latest.current.cycle.stop().catch(() => {});
    }
  }, [mapName]);

  const resetIdle = () => {
    const current = latest.current;
    if (cancelUnconfirmed.current || stopCount.current || current.navigation.isRunning
      || current.navigation.patrolPaused || current.cycle.isRunning || current.missionRunner.isRunning) return false;
    if (!current.cycle.resetIdle() || !current.navigation.resetIdle()) return false;
    activeOwner.current = null; setOwner(null); lastObservation.current = null; patrolContinuation.current = null;
    return true;
  };
  const returning = cycle.returning || cycle.returnPending || cycle.status === "return-starting";
  const executionState = buildRecordExecutionState({ mapName, planRevision: manager.configuration?.plan?.revision ?? null,
    enabled, navigation, cycle, missionRunner, owner, observationReady, observationSpotId, stopping, cancelFailed,
    pendingControl, controlError, retainedExecution: retainedExecution.current, patrolContinuation: patrolContinuation.current?.route });
  latest.current.executionState = executionState;
  const executionType = recordExecutionType(executionState, navigation.held);
  const runner = cycle.isRunning ? cycle : navigation;
  const navigationActive = enabled && localized && recordNavigationActive(executionState);
  return { ...runner, navigationActive, start: () => launchNavigation(), startPatrol: () => launchNavigation(),
    goToSpot: id => launchNavigation(id), startChildMission, pausePatrol, resumePatrol, skipObservation: () => stepObservation(), previousObservation: () => stepObservation(true), stop, resetIdle,
    executionState, navigation, cycle, missionRunner, owner, stopping, cancelFailed, needsStop: cancelFailed, executionType, observationReady, observationSpotId, linkedMissions,
    isRunning: cancelFailed || stopping || navigation.isRunning || navigation.patrolPaused || cycle.isRunning || missionRunner.isRunning,
    returning, canRunMission: !manager.route?.routeMode && !manager.planDirty && !manager.draft && !cancelFailed && !stopping && missionAllowed && !cycle.isRunning && !missionRunner.isRunning,
    canPreviousObservation: !stopping && !cancelFailed && navigation.canPreviousObservation && !cycle.isRunning && !missionRunner.isRunning,
    canSkipObservation: !stopping && !cancelFailed
      && navigation.canSkipDwell && !cycle.isRunning && !missionRunner.isRunning,
    activeObservation: observationReady ? candidate : null,
  };
}
