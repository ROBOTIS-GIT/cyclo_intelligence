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

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useOperateModel from "./useOperateModel";
import { executionOwnsRobot, latestExecutionStatus } from "../lib/executionState";

import { autonomyRequest } from "./autonomyApi";
import useObservationPlan from "./useObservationPlan";
export { autonomyRequest } from "./autonomyApi";

// The browser is a client of the server session; unmount never cancels execution.
export default function useAutonomyOperation({ enabled, mapName, busy = false }) {
  const choice = useOperateModel(mapName);
  const [state, setState] = useState({ phase: "idle", active: false, needsStop: false, events: [] });
  const stateRef = useRef(state);
  const statusSignatureRef = useRef(null);
  const acceptStatus = useCallback(next => {
    const accepted = latestExecutionStatus(stateRef.current, next);
    if (accepted !== stateRef.current) {
      // Compare the whole response: model/result fields can change without an
      // execution revision. Do this only on receipt, never on map renders.
      const signature = JSON.stringify(accepted);
      if (signature === statusSignatureRef.current) return stateRef.current;
      statusSignatureRef.current = signature;
      stateRef.current = accepted;
      setState(accepted);
      try { localStorage.setItem("autonomy.serverOwned", String(executionOwnsRobot(accepted))); } catch (_) {}
    }
    return accepted;
  }, []);
  const [working, setWorking] = useState(false);
  const [starting, setStarting] = useState(false);
  const locked = executionOwnsRobot(state) || starting || working;
  const config = useObservationPlan({ mapName, editable: true, locked: locked || busy });
  const draft = config.draft, dirty = config.dirty;
  const pendingStart = useRef(null);
  const commandEpoch = useRef(0);
  const [error, setError] = useState(null);
  const [pollError, setPollError] = useState(null);
  const latest = useRef({});
  latest.current = { enabled, mapName, choice, state, draft, dirty, busy: busy || config.working };
  useEffect(() => {
    let cancelled = false, timer;
    async function poll() {
      try {
        const epoch = commandEpoch.current;
        const next = await autonomyRequest("/autonomy/status");
        if (cancelled) return;
        setPollError(null);
        if (epoch === commandEpoch.current && !pendingStart.current) {
          acceptStatus(next);
        }
      } catch (failure) { if (!cancelled) setPollError(failure.message); }
      if (!cancelled) timer = setTimeout(poll, 750);
    }
    void poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [acceptStatus]);
  const perform = useCallback(async action => {
    ++commandEpoch.current;
    setWorking(true); setError(null);
    try { return await action(); }
    catch (failure) { setError(failure.message); return false; }
    finally { ++commandEpoch.current; setWorking(false); }
  }, []);
  const start = () => {
    const v = latest.current;
    if (pendingStart.current || executionOwnsRobot(v.state) || !v.enabled || v.busy || v.dirty || !v.choice.selected || !v.draft?.observations.length) return Promise.resolve(false);
    setStarting(true);
    acceptStatus({ ...stateRef.current, needsStop: true });
    try { localStorage.setItem("autonomy.serverOwned", "true"); } catch (_) {}
    const request = perform(async () => {
      const next = await autonomyRequest("/autonomy/start", "POST", { map_name: v.mapName,
        model_id: v.choice.selected.id, localized: true, plan_revision: v.draft.revision });
      acceptStatus(next);
      return true;
    });
    pendingStart.current = request;
    void request.finally(() => { pendingStart.current = null; setStarting(false); });
    return request;
  };
  const stop = () => perform(async () => {
    // Serialize an explicit Stop after this browser's in-flight Start. Otherwise
    // the server could accept Start only after navigation has been torn down.
    if (pendingStart.current) { await pendingStart.current; setWorking(true); }
    let next = acceptStatus(await autonomyRequest("/autonomy/stop", "POST"));
    const deadline = Date.now() + 180000;
    while (executionOwnsRobot(next) && (next.execution?.lifecycle === "stopping" || next.phase === "stopping") && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 400));
      next = acceptStatus(await autonomyRequest("/autonomy/status"));
    }
    if (executionOwnsRobot(next)) throw new Error(next.error || "Autonomy stop is not confirmed. Retry Stop.");
    try { localStorage.setItem("autonomy.serverOwned", "false"); } catch (_) {}
    return true;
  });
  const events = useMemo(() => (state.events || []).map(row => ({
    ...row, time: new Date(row.time_ms).toLocaleTimeString(),
  })), [state.events]);
  return { ...state, needsStop: Boolean(executionOwnsRobot(state) || starting), error: error || pollError || config.error || state.error, working, enabled,
    start, stop, cancelMission: stop,
    pause: () => perform(async () => {
      acceptStatus(await autonomyRequest("/autonomy/pause", "POST")); return true;
    }),
    resume: () => perform(async () => {
      acceptStatus(await autonomyRequest("/autonomy/resume", "POST")); return true;
    }),
    sendInstruction: text => perform(async () => {
      acceptStatus(await autonomyRequest("/autonomy/instruction", "POST", { text })); return true;
    }),
    modelChoice: { ...choice, locked, select: id => { if (!locked) choice.select(id); } },
    canStart: Boolean(enabled && !busy && !config.working && !dirty && draft?.observations.length && choice.selected && !choice.loading && !choice.error && !locked),
    events,
    config,
  };
}
