// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useRef } from "react";
import { STAGE_NAVIGATE } from "../lib/stages";

// Coordinates Run/Navigate runtime startup and initial-pose convergence. All
// React/session mutations are ports so this hook does not own the shared Nav2
// runtime or the mission document.
export default function useRunNavigationLocalizationController({
  getState,
  runtime,
  commits,
  pose,
  runCommand,
  onMessage,
}) {
  const portsRef = useRef({ getState, runtime, commits, pose, runCommand, onMessage });
  portsRef.current = { getState, runtime, commits, pose, runCommand, onMessage };
  const mountedRef = useRef(true);
  const poseGenerationRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      poseGenerationRef.current += 1;
    };
  }, []);

  const invalidatePoseRequest = useCallback(() => {
    poseGenerationRef.current += 1;
  }, []);

  const localize = useCallback(() => {
    const ports = portsRef.current;
    const state = ports.getState();
    const missionSnapshotRequired = state.stage !== STAGE_NAVIGATE;
    if (state.mapBusy || (missionSnapshotRequired && state.snapshotInvalid)) {
      ports.onMessage(missionSnapshotRequired
        ? "Wait for the selected mission to finish loading"
        : "Wait for the selected map to finish loading");
      return Promise.resolve({ skipped: true, reason: "map-loading" });
    }
    if (state.interactionMode === "initial") {
      invalidatePoseRequest();
      ports.commits.setInteractionMode("view");
      return Promise.resolve({ disarmed: true });
    }

    // Capture both map and stage before entering runCommand. A later picker or
    // stage change must not retarget an already-starting supervisor request.
    const targetMapName = state.targetMapName;
    const targetStage = state.stage === STAGE_NAVIGATE ? STAGE_NAVIGATE : state.runStage;
    const runtimeAlreadyUp = state.running && state.runtimeMode === "run";
    return ports.runCommand("Localize", async () => {
      invalidatePoseRequest();
      ports.commits.setStage(targetStage);
      ports.commits.setRuntimeMode("run");
      ports.commits.setDesignPoseReady(false);
      if (!runtimeAlreadyUp) ports.commits.setRunPoseReady(false);
      ports.commits.setRuntimeOwned(true);
      ports.commits.setShutdownPending(false);
      ports.commits.persistClaim({
        mapName: state.sessionMapName,
        workspaceStage: targetStage,
        navigationRuntimeMode: "run",
        designPoseInitialized: false,
        runRuntimeOwned: true,
        runShutdownPending: false,
        runShutdownRequestedAt: null,
      });

      let startAttempted = false;
      try {
        if (!runtimeAlreadyUp) {
          startAttempted = true;
          await ports.runtime.start("nav", targetMapName);
        }
      } catch (error) {
        if (!ports.runtime.pageExitStopSent()) {
          ports.commits.persistRollback({
            navigationRuntimeMode: "idle",
            runRuntimeOwned: false,
            runShutdownPending: false,
            runShutdownRequestedAt: null,
          });
          if (mountedRef.current) {
            ports.commits.setRuntimeMode("idle");
            ports.commits.setRuntimeOwned(false);
          }
        }
        throw error;
      } finally {
        // A pagehide Stop may race the supervisor start. Preserve ordering by
        // sending a second keepalive Stop after that start has settled.
        if (startAttempted && ports.runtime.pageExitStopSent()) {
          void ports.runtime.stop({ keepalive: true }).catch(() => {});
        }
      }
      if (ports.runtime.pageExitStopSent()) return "Run session stopping";
      if (!mountedRef.current) return { stale: true };
      ports.commits.setInteractionMode("initial");
      return "Click and drag the robot pose on the map";
    });
  }, [invalidatePoseRequest]);

  const estimatePose = useCallback((x, y, yaw) => {
    const ports = portsRef.current;
    const generation = poseGenerationRef.current + 1;
    poseGenerationRef.current = generation;
    ports.commits.setInteractionMode("view");
    return ports.runCommand("Set robot pose", async () => {
      ports.pose.clearCache();
      await ports.pose.publish({ x, y, yaw, frameId: "map" });
      if (!mountedRef.current || poseGenerationRef.current !== generation) {
        return { stale: true };
      }
      ports.onMessage("Localizing robot");
      await ports.pose.waitForConvergence();
      if (!mountedRef.current || poseGenerationRef.current !== generation) {
        return { stale: true };
      }
      ports.commits.setRunPoseReady(true);
      return "Robot localized";
    });
  }, []);

  return { localize, estimatePose, invalidatePoseRequest };
}
