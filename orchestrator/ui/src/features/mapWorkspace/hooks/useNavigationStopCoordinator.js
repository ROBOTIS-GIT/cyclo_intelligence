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
  autonomy,
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
    autonomy,
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
    autonomy,
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
    // Explicit UI teardown must release the server session before touching its
    // Nav2/BT resources. A failed Stop leaves the current tab and map intact.
    if (ports.autonomy?.needsStop && !(await ports.autonomy.stop())) {
      throw new Error("Autonomy stop is not confirmed. Retry Stop before leaving this session.");
    }
    const clearRunSnapshot = options.clearRunSnapshot === undefined
      ? ports.getDefaultClearRunSnapshot()
      : options.clearRunSnapshot;

    ports.invalidate.mapSelection();
    ports.invalidate.poseRequest();
    // Collect's second Stop follows an acknowledged motion cancellation.
    // Repeating cancel-all while Nav2 is going down waits for an absent service.
    const runnerCleanup = options.runnerAlreadyStopped
      ? Promise.resolve()
      : invokeAsPromise(ports.runner.stop);
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
