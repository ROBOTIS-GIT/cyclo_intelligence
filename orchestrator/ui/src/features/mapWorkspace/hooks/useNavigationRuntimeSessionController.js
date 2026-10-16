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

import { useCallback, useEffect, useRef } from "react";

// Owns the browser/session shell around the shared navigation runtime. Runtime
// Stop orchestration remains outside because Mapping, Design localization,
// Navigate and Run have different document-cleanup responsibilities.
export default function useNavigationRuntimeSessionController({
  enabled = true,
  pollIntervalMs = 10000,
  state,
  status,
  session,
  runtime,
  commits,
  onMessage,
}) {
  const mountedRef = useRef(true);
  const pollOwnerRef = useRef(null);
  const pollGenerationRef = useRef(0);
  const exitStopSentRef = useRef(false);
  const confirmationRef = useRef(null);
  const portsRef = useRef({ state, status, session, runtime, commits, onMessage });
  portsRef.current = { state, status, session, runtime, commits, onMessage };

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      pollGenerationRef.current += 1;
      pollOwnerRef.current = null;
    };
  }, []);

  const invalidateStatus = useCallback(() => {
    pollGenerationRef.current += 1;
    pollOwnerRef.current = null;
  }, []);

  const refreshStatus = useCallback(async () => {
    const ports = portsRef.current;
    if (
      !enabled
      || ports.state.shutdownPending
      || pollOwnerRef.current
      || document.visibilityState === "hidden"
    ) {
      return { skipped: true };
    }
    const generation = pollGenerationRef.current + 1;
    const owner = {};
    pollGenerationRef.current = generation;
    pollOwnerRef.current = owner;
    try {
      const nextStatus = await ports.status.get();
      if (!mountedRef.current || pollGenerationRef.current !== generation) {
        return { stale: true };
      }
      const mode = ports.status.modeOf(nextStatus);
      ports.commits.setStatus(nextStatus);
      if (mode) {
        ports.commits.setRuntimeMode(mode);
        if (mode === "idle") ports.commits.setDesignPoseReady(false);
        if (mode === "mapping" || mode === "localization") {
          ports.commits.setRuntimeOwned(false);
        }
      }
      return { status: nextStatus, mode };
    } catch (error) {
      return { error };
    } finally {
      if (pollOwnerRef.current === owner) pollOwnerRef.current = null;
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return undefined;
    void refreshStatus();
    const interval = window.setInterval(refreshStatus, pollIntervalMs);
    const handleVisibility = () => {
      if (document.visibilityState === "visible") void refreshStatus();
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [enabled, pollIntervalMs, refreshStatus]);

  // Invalidate a poll that began before shutdown ownership changed. It must
  // not restore a stale backend "run" mode while Stop confirmation is pending.
  useEffect(() => {
    if (state.shutdownPending) invalidateStatus();
  }, [invalidateStatus, state.shutdownPending]);

  useEffect(() => {
    if (!enabled || state.serverOwned || !state.shutdownPending) {
      if (!state.shutdownPending) confirmationRef.current = null;
      return undefined;
    }
    let cancelled = false;
    const ports = portsRef.current;
    if (!confirmationRef.current) confirmationRef.current = ports.runtime.stop();
    confirmationRef.current
      .then(() => {
        if (cancelled) return;
        ports.commits.confirmStopped();
        ports.session.save({
          navigationRuntimeMode: "idle",
          designPoseInitialized: false,
          runRuntimeOwned: false,
          runShutdownPending: false,
          runShutdownRequestedAt: null,
        });
      })
      .catch((error) => {
        if (cancelled) return;
        ports.onMessage(error instanceof Error
          ? `Failed to stop the previous Run session: ${error.message}`
          : "Failed to stop the previous Run session");
      });
    return () => { cancelled = true; };
  }, [enabled, state.serverOwned, state.shutdownPending]);

  useEffect(() => {
    if (!enabled) return undefined;
    const handlePageHide = (event) => {
      if (event.persisted === true || exitStopSentRef.current) return;
      // Navigation belongs to the server while autonomy is running.
      try { if (localStorage.getItem("autonomy.serverOwned") === "true") return; } catch (_) {}
      const ports = portsRef.current;
      if (ports.state.serverOwned) return;
      const saved = ports.session.read();
      const current = ports.state;
      const knownNonRunRuntime = (
        current.status?.is_up === true
        && current.runtimeMode !== "run"
        && saved.navigationRuntimeMode !== "run"
      );
      const ownsRunRuntime = (
        ports.session.recentShutdownMarker(saved)
        || (!knownNonRunRuntime && saved.runRuntimeOwned === true)
        || (
          !knownNonRunRuntime
          && saved.runRuntimeOwned === undefined
          && (current.runtimeOwned || current.shutdownPending)
        )
      );
      if (!ownsRunRuntime) return;
      exitStopSentRef.current = true;
      ports.session.save({
        navigationRuntimeMode: "idle",
        designPoseInitialized: false,
        runRuntimeOwned: true,
        runShutdownPending: true,
        runShutdownRequestedAt: Date.now(),
      });
      void ports.runtime.stop({ keepalive: true }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, [enabled]);

  useEffect(() => {
    if (!enabled || exitStopSentRef.current) return;
    const ports = portsRef.current;
    const current = ports.state;
    const saved = ports.session.read();
    const requestedAt = Number(saved.runShutdownRequestedAt);
    ports.session.save({
      mapName: current.mapName,
      workspaceStage: current.stage,
      designMapPath: current.designMapPath,
      navigationRuntimeMode: current.runtimeMode,
      designPoseInitialized: current.designPoseReady,
      runRuntimeOwned: current.runtimeOwned,
      runShutdownPending: current.shutdownPending,
      runShutdownRequestedAt: current.shutdownPending
        ? (Number.isFinite(requestedAt) && requestedAt > 0 ? requestedAt : Date.now())
        : null,
      missionName: current.missionName,
      runMissionName: current.runMissionName,
    });
  }, [
    enabled,
    state.designMapPath,
    state.designPoseReady,
    state.mapName,
    state.missionName,
    state.runMissionName,
    state.runtimeMode,
    state.runtimeOwned,
    state.shutdownPending,
    state.stage,
  ]);

  return {
    refreshStatus,
    invalidateStatus,
    isPageExitStopSent: () => exitStopSentRef.current,
  };
}
