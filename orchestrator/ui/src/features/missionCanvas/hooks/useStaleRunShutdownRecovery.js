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

import { useEffect, useRef } from "react";
import { stopNavigation } from "../../../utils/navigationApi";
import {
  initialNavigationRuntimeMode,
  initialRunRuntimeOwned,
  initialRunShutdownPending,
  readMissionSession,
  recentRunShutdownMarker,
  saveMissionSession,
} from "../lib/session";

// Session-only half of the Mission Canvas run lifecycle, for workspaces that
// do not mount the mission runtime (the Action Canvas). It mirrors what the
// mission workspace does on mount and on page exit:
//  - normalize the persisted ownership markers (an expired shutdown marker
//    drops ownership; a legacy Run session gains it) so a later page exit only
//    stops a runtime this tab still owns;
//  - confirm a Run shutdown that was still pending when the document unloaded;
//  - on page exit, stop an owned Run runtime with a keepalive request and
//    leave the retry marker behind for the next load.
export default function useStaleRunShutdownRecovery(enabled = true) {
  // Survives StrictMode's simulated remount, so the confirmation POST is sent
  // once (the mission workspace keeps the same promise in a ref).
  const confirmationRef = useRef(null);
  const exitStopSentRef = useRef(false);

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    const session = readMissionSession();
    const shutdownPending = initialRunShutdownPending(session);
    const savedRequestedAt = Number(session.runShutdownRequestedAt);
    saveMissionSession({
      navigationRuntimeMode: initialNavigationRuntimeMode(session),
      runRuntimeOwned: initialRunRuntimeOwned(session),
      runShutdownPending: shutdownPending,
      runShutdownRequestedAt: shutdownPending
        ? (
          Number.isFinite(savedRequestedAt) && savedRequestedAt > 0
            ? savedRequestedAt
            : Date.now()
        )
        : null,
    });
    if (shutdownPending) {
      if (!confirmationRef.current) {
        confirmationRef.current = stopNavigation();
      }
      confirmationRef.current
        .then(() => {
          if (cancelled) return;
          saveMissionSession({
            navigationRuntimeMode: "idle",
            designPoseInitialized: false,
            runRuntimeOwned: false,
            runShutdownPending: false,
            runShutdownRequestedAt: null,
          });
        })
        .catch(() => {
          // The marker stays in the session, so the next load (or a page
          // exit) retries the stop.
        });
    }
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return undefined;
    const handlePageHide = (event) => {
      if (event.persisted === true || exitStopSentRef.current) return;
      const savedSession = readMissionSession();
      const ownsRunRuntime = (
        recentRunShutdownMarker(savedSession) || savedSession.runRuntimeOwned === true
      );
      if (!ownsRunRuntime) return;
      exitStopSentRef.current = true;
      saveMissionSession({
        navigationRuntimeMode: "idle",
        designPoseInitialized: false,
        runRuntimeOwned: true,
        runShutdownPending: true,
        runShutdownRequestedAt: Date.now(),
      });
      void stopNavigation({ keepalive: true }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, [enabled]);
}
