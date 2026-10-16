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

import readApiResponse from "../../../utils/readApiResponse";
import { useCallback, useEffect, useRef, useState } from "react";

async function request(path, signal, body) {
  const response = await fetch(`/api/reasoning/${path}`, {
    signal,
    ...(body !== undefined ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
  });
  return readApiResponse(response, "Qwen request failed");
}

export default function useQwenObservation(enabled, collection = {}, options = {}) {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const collectionRef = useRef(collection);
  collectionRef.current = collection;
  const [view, setView] = useState({ active: false, phase: "idle", result: null, error: null, needsStop: false });
  const session = useRef({ id: 0, controller: null, timer: null, active: false, startRequest: null, stopRequest: null });
  const cancel = useCallback(() => {
    const current = session.current;
    current.id += 1;
    current.active = false;
    current.controller?.abort();
    clearTimeout(current.timer);
  }, []);
  const pause = useCallback(() => {
    cancel();
    setView(previous => ({ ...previous, active: false, phase: previous.active ? "paused" : previous.phase }));
  }, [cancel]);
  const stop = useCallback(() => {
    if (session.current.stopRequest) return session.current.stopRequest;
    cancel();
    session.current.readySession = null;
    setView(previous => ({ ...previous, active: false, phase: "stopping", error: null, needsStop: true }));
    // Do not abort the short start request: its server acknowledgement must
    // precede Stop, even if the user clicks Stop during container startup.
    const operation = (async () => {
      try {
        await session.current.startRequest?.catch(() => {});
        await request("stop", undefined, {});
        setView(previous => ({ ...previous, active: false, phase: "stopped", error: null, needsStop: false }));
        return true;
      } catch (error) {
        setView(previous => ({ ...previous, active: false, phase: "error", error: error.message, needsStop: true }));
        return false;
      } finally {
        session.current.stopRequest = null;
      }
    })();
    session.current.stopRequest = operation;
    return operation;
  }, [cancel]);
  useEffect(() => {
    setView(previous => ({ ...previous, result: null, error: null }));
  }, [options.contextKey]);
  useEffect(() => cancel, [cancel]);
  useEffect(() => { if (!enabled) pause(); }, [enabled, pause]);

  const start = useCallback(async () => {
    if (!enabled || session.current.active || session.current.stopRequest) return;
    cancel();
    const current = session.current;
    const id = current.id;
    const controller = new AbortController();
    current.controller = controller;
    current.active = true;
    let previousStamp = null;
    let serverSession = null;
    const startPayload = optionsRef.current.startPayload || {};
    const payloadKey = JSON.stringify(startPayload);
    const retained = optionsRef.current.keepLoadedSession && current.readySession?.payloadKey === payloadKey
      ? current.readySession : null;
    if (retained) serverSession = retained.id;
    const valid = () => session.current.id === id && !controller.signal.aborted;
    const update = values => { if (valid()) setView(previous => ({ ...previous, ...values })); };
    const schedule = (fn, ms) => { if (valid()) current.timer = setTimeout(fn, ms); };
    const fail = error => {
      if (!valid()) return;
      current.active = false;
      update({ active: false, phase: "error", error: error.message });
    };
    async function observe() {
      if (!valid()) return;
      update({ phase: "observing", error: null });
      try {
        const mode = optionsRef.current;
        const body = mode.endpoint === "decide"
          ? { ...mode.payload, previous_frame_stamp: previousStamp, ...(serverSession ? { expected_session_id: serverSession } : {}) }
          : { previous_frame_stamp: previousStamp, ...collectionRef.current, context: { ...collectionRef.current.context, ...mode.getContext?.(), client_sampled_at_ms: Date.now() } };
        const result = await request(mode.endpoint || "observe", controller.signal, body);
        if (!valid()) return;
        if (mode.contextKey !== optionsRef.current.contextKey) {
          schedule(observe, 0); return; // The operator/mission context changed during inference.
        }
        previousStamp = result.frame_stamp;
        update({ phase: result.error ? "waiting" : "watching", result, error: result.error || null });
        if (optionsRef.current.onResult) await optionsRef.current.onResult(result);
        schedule(observe, result.error ? 3000 : 200);
      } catch (error) {
        if (error.status === 409 || error.status === 503) {
          update({ phase: "waiting", error: error.message });
          schedule(observe, 3000);
        } else fail(error);
      }
    }
    async function poll() {
      if (!valid()) return;
      try {
        const status = await request("status", controller.signal);
        if (!valid()) return;
        if (retained && ["stopped", "stopping", "idle"].includes(status.phase)) throw new Error("Qwen session ended. Stop and run again.");
        if (status.phase === "error") throw new Error(status.error || "Qwen preparation failed");
        if (serverSession && status.session_id !== serverSession) throw new Error("Reasoning session changed. Stop and run again.");
        if (status.phase === "ready") {
          if (startPayload.model_id && status.selected_model_id !== startPayload.model_id) throw new Error("The selected model was not loaded");
          current.readySession = serverSession ? { id: serverSession, payloadKey } : null;
          update({ loadedModel: status });
          await observe(); return;
        }
        update({ phase: status.phase });
        schedule(poll, 1500);
      } catch (error) { fail(error); }
    }
    setView({ active: true, phase: "starting", result: null, error: null, needsStop: true });
    try {
      if (retained) { await poll(); return; }
      current.startRequest = Promise.resolve(current.startRequest).catch(() => {}).then(() => request("start", undefined, startPayload));
      const started = await current.startRequest;
      serverSession = started.session_id;
      if (valid()) await poll();
    } catch (error) { fail(error); }
  }, [cancel, enabled]);

  return { ...view, start, pause, stop };
}
