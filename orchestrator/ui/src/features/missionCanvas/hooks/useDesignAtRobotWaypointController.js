// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef } from "react";
import { STAGE_AUTHORING } from "../lib/stages";

function staleDocumentError() {
  const error = new Error("Design document changed during At Robot localization");
  error.code = "STALE_DESIGN_DOCUMENT";
  return error;
}

// Coordinates the temporary Design localization session used by At Robot.
// Waypoint persistence remains in useDesignWaypointController; this hook owns
// only the begin -> pose convergence -> guaranteed runtime-finalize workflow.
export default function useDesignAtRobotWaypointController({
  getState,
  document,
  runtime,
  waypoint,
  ui,
  session,
  runCommand,
  onMessage,
}) {
  const portsRef = useRef({ getState, document, runtime, waypoint, ui, session, runCommand, onMessage });
  portsRef.current = { getState, document, runtime, waypoint, ui, session, runCommand, onMessage };
  const generationRef = useRef(0);
  const beginActiveRef = useRef(false);
  const completeActiveRef = useRef(false);
  const transactionRef = useRef(null);

  const isCurrent = useCallback((transaction) => (
    transactionRef.current === transaction
    && generationRef.current === transaction.generation
    && portsRef.current.document.isLeaseCurrent(transaction.lease)
  ), []);

  const begin = useCallback(() => {
    const ports = portsRef.current;
    const state = ports.getState();
    if (state.mappingActive || state.runActive || state.runnerActive) {
      ports.onMessage("Stop the active navigation session before using At Robot");
      return Promise.resolve({ skipped: true, reason: "runtime-active" });
    }
    if (!state.documentReady || !state.mapPath) {
      ports.onMessage("Load a map before creating a waypoint");
      return Promise.resolve({ skipped: true, reason: "document-not-ready" });
    }
    if (beginActiveRef.current || completeActiveRef.current || transactionRef.current) {
      return Promise.resolve({ skipped: true, reason: "operation-active" });
    }

    const transaction = {
      generation: generationRef.current + 1,
      lease: ports.document.captureLease(),
      mapName: state.mapName,
      mapPath: state.mapPath,
    };
    generationRef.current = transaction.generation;
    transactionRef.current = transaction;
    beginActiveRef.current = true;
    ports.ui.prepareBegin();

    let command;
    try {
      command = ports.runCommand("At Robot", async () => {
        let started = false;
        let cleaned = false;
        const cleanupStartedRuntime = async () => {
          if (!started || cleaned) return;
          cleaned = true;
          await ports.runtime.cleanupStartedRuntime?.(transaction);
        };
        try {
          await ports.runtime.startLocalization(transaction.mapName);
          started = true;
          await ports.runtime.configureAmcl();
          if (!isCurrent(transaction)) {
            try { await cleanupStartedRuntime(); } catch { /* best effort for a stale document */ }
            if (transactionRef.current === transaction) transactionRef.current = null;
            return { stale: true };
          }
          ports.ui.setRuntimeMode("localization");
          ports.ui.setRuntimeOwned(false);
          ports.ui.setShutdownPending(false);
          ports.session.save({
            mapName: transaction.mapName,
            workspaceStage: STAGE_AUTHORING,
            designMapPath: transaction.mapPath,
            navigationRuntimeMode: "localization",
            designPoseInitialized: false,
            runRuntimeOwned: false,
            runShutdownPending: false,
            runShutdownRequestedAt: null,
          });
          ports.ui.setInteractionMode("initial");
          return "Click and drag the robot pose on the map";
        } catch (error) {
          const current = isCurrent(transaction);
          if (started) {
            try { await cleanupStartedRuntime(); } catch { /* preserve the original failure */ }
          }
          if (!current) {
            if (transactionRef.current === transaction) transactionRef.current = null;
            return { stale: true };
          }
          if (transactionRef.current === transaction) transactionRef.current = null;
          throw error;
        }
      });
    } catch (error) {
      beginActiveRef.current = false;
      if (transactionRef.current === transaction) transactionRef.current = null;
      return Promise.reject(error);
    }
    return Promise.resolve(command).finally(() => {
      beginActiveRef.current = false;
    });
  }, [isCurrent]);

  const completeAtPose = useCallback((x, y, yaw) => {
    const ports = portsRef.current;
    const state = ports.getState();
    if (state.mappingActive || state.runActive || state.runnerActive) {
      ports.ui.setInteractionMode("view");
      ports.onMessage("Stop the active navigation session before using At Robot");
      return Promise.resolve({ skipped: true, reason: "runtime-active" });
    }
    const transaction = transactionRef.current;
    if (!transaction || !isCurrent(transaction)) {
      return Promise.resolve({ skipped: true, reason: "stale-document" });
    }
    if (completeActiveRef.current) {
      return Promise.resolve({ skipped: true, reason: "operation-active" });
    }
    completeActiveRef.current = true;
    ports.ui.setInteractionMode("view");
    ports.ui.setWaypointOptionsOpen(false);

    let operation;
    try {
      operation = ports.waypoint.createAtRobot({
        resolvePose: async () => {
          if (!isCurrent(transaction)) throw staleDocumentError();
          ports.runtime.clearPoseCache();
          await ports.runtime.publishInitialPose({
            x, y, yaw, frameId: "map", mapName: transaction.mapName,
          });
          if (!isCurrent(transaction)) throw staleDocumentError();
          ports.ui.setDesignPoseReady(true);
          ports.session.save({ designPoseInitialized: true });
          const localizedPose = await ports.runtime.waitForConvergence();
          if (!isCurrent(transaction)) throw staleDocumentError();
          return ports.runtime.poseCoordinates(localizedPose);
        },
        finalize: async ({ documentCurrent }) => {
          if (!documentCurrent || !isCurrent(transaction)) return;
          ports.runtime.invalidateStatus();
          try {
            await ports.runtime.stop();
          } finally {
            // Runtime ownership must never remain stuck merely because the
            // supervisor response was lost after it stopped successfully.
            ports.ui.setRuntimeMode("idle");
            ports.ui.setDesignPoseReady(false);
            ports.ui.setRuntimeOwned(false);
            ports.ui.setShutdownPending(false);
            ports.session.save({
              mapName: transaction.mapName,
              workspaceStage: STAGE_AUTHORING,
              designMapPath: transaction.mapPath,
              navigationRuntimeMode: "idle",
              designPoseInitialized: false,
              runRuntimeOwned: false,
              runShutdownPending: false,
              runShutdownRequestedAt: null,
            });
          }
        },
        resultMessage: (initialized) => `Created ${initialized.label} at robot`,
      });
    } catch (error) {
      completeActiveRef.current = false;
      if (transactionRef.current === transaction) transactionRef.current = null;
      return Promise.reject(error);
    }
    return Promise.resolve(operation).finally(() => {
      completeActiveRef.current = false;
      if (transactionRef.current === transaction) transactionRef.current = null;
    });
  }, [isCurrent]);

  const cancelPending = useCallback(() => {
    const state = portsRef.current.getState();
    if (
      state.stage !== STAGE_AUTHORING
      || state.interactionMode !== "initial"
      || !state.designLocalizationActive
    ) {
      return false;
    }
    generationRef.current += 1;
    transactionRef.current = null;
    void portsRef.current.runtime.stopShared();
    return true;
  }, []);

  return { begin, completeAtPose, cancelPending };
}
