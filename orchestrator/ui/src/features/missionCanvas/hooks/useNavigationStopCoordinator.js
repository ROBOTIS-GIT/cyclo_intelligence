// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef } from "react";

function invokeAsPromise(action) {
  try {
    return Promise.resolve(action());
  } catch (error) {
    return Promise.reject(error);
  }
}

export default function useNavigationStopCoordinator({
  getDefaultClearRunSnapshot,
  invalidate,
  runner,
  navigation,
  snapshot,
  pose,
  commits,
  session,
}) {
  const portsRef = useRef({
    getDefaultClearRunSnapshot,
    invalidate,
    runner,
    navigation,
    snapshot,
    pose,
    commits,
    session,
  });
  portsRef.current = {
    getDefaultClearRunSnapshot,
    invalidate,
    runner,
    navigation,
    snapshot,
    pose,
    commits,
    session,
  };

  return useCallback(async (options = {}) => {
    const ports = portsRef.current;
    const clearRunSnapshot = options.clearRunSnapshot === undefined
      ? ports.getDefaultClearRunSnapshot()
      : options.clearRunSnapshot;

    ports.invalidate.mapSelection();
    ports.invalidate.poseRequest();
    const runnerCleanup = invokeAsPromise(ports.runner.stop);
    ports.invalidate.navGoal();
    ports.commits.setInteractionView();

    if (clearRunSnapshot) {
      ports.commits.hideMap();
      ports.snapshot.cancelAndClear();
      ports.commits.setRunPoseReady(false);
      ports.session.save({ runMissionName: "" });
    }

    const [, navigationStopResult] = await Promise.allSettled([
      runnerCleanup,
      invokeAsPromise(ports.navigation.stop),
    ]);
    if (navigationStopResult.status === "rejected") {
      throw navigationStopResult.reason;
    }

    ports.pose.clearCache();
    ports.pose.resetMappingSync();
    ports.commits.setStatusIdle();
    ports.commits.setRuntimeModeIdle();
    ports.commits.setDesignPoseReady(false);
    ports.commits.setRunPoseReady(false);
    ports.commits.setRuntimeOwned(false);
    ports.commits.setShutdownPending(false);
    ports.session.save({
      navigationRuntimeMode: "idle",
      designPoseInitialized: false,
      runRuntimeOwned: false,
      runShutdownPending: false,
      runShutdownRequestedAt: null,
      ...(clearRunSnapshot ? { runMissionName: "" } : {}),
    });
    return navigationStopResult.value;
  }, []);
}
