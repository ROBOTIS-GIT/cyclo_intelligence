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

// Recording owns storage only. Movement is controlled by the map execution bar.
import { useEffect, useRef, useState } from "react";
import { resolveInitialRobotType } from "../../tasks/taskSlice";
import { captureScopeMatches, episodeFieldsForSession, episodeFieldsForPreparation } from "../utils/recordCapture";

export default function useCollectRecording({ draft, session }) {
  const [captureMode, setCaptureMode] = useState("continuous");
  const [intervalMs, setIntervalMs] = useState(1000);
  const [recordError, setRecordError] = useState("");
  const [captureTransition, setCaptureTransition] = useState(false);
  const [singleArmed, setSingleArmed] = useState(false);
  const [singlePending, setSinglePending] = useState(false);
  const singleInFlight = useRef(false);
  const previousMap = useRef(session?.mapName);
  const captureFields = useRef(null), pending = useRef(false), request = useRef(0);
  const latest = useRef(null);
  latest.current = { draft, session, captureMode, intervalMs };
  useEffect(() => () => { ++request.current; latest.current.draft.stopRecording?.(); }, []);
  // The recording session follows execution. Its current segment never gates motion.
  const captureType = session ? (session.ready ? session.type : null) : draft.episode?.episode_type || "observation";
  const awaitingTarget = !!(draft.active && session && !session.ready);
  const preparation = (options = {}) => episodeFieldsForSession(session, options)
    || (session && episodeFieldsForPreparation(session, "patrol", options))
    || episodeFieldsForPreparation(session, "observation", options);
  const targetReady = !!preparation();
  const executionReady = draft.active && !captureTransition
    && !recordError && (draft.recording || singleArmed);
  const fields = () => episodeFieldsForSession(latest.current.session, {
    robot_type: resolveInitialRobotType() || "unknown",
    capture_mode: latest.current.captureMode, capture_interval_ms: latest.current.intervalMs,
  });
  const stopCapture = () => {
    ++request.current; setSingleArmed(false); setSinglePending(false); draft.stopRecording();
  };
  useEffect(() => {
    if (previousMap.current === session?.mapName) return;
    previousMap.current = session?.mapName;
    setSingleArmed(false); setSinglePending(false);
    ++request.current; latest.current.draft.stopRecording?.();
  }, [session?.mapName]);
  useEffect(() => {
    if (!draft.active) { setSingleArmed(false); setSinglePending(false); }
  }, [draft.active]);
  async function ensureSegment() {
    const current = latest.current;
    if (!current.session) return true;
    const next = fields();
    if (!next) return null; // Transition/settling: wait for a verified execution report.
    if (captureScopeMatches(current.draft.episode, next, current.session.returning)) return true;
    setCaptureTransition(true);
    try {
      const source = current.draft.episode;
      const useReturn = current.session.returning && source?.episode_type === "mission"
        && source.recording_mission_id === current.session.missionId
        && source.return_observation_spot_id === current.session.returnSpotId;
      const child = useReturn
        ? await current.draft.startReturnEpisode({ keepRecording: true })
        : await current.draft.startSegment(next);
      if (!child) throw new Error("Could not save the next capture segment. Earlier frames are preserved.");
      const after = fields();
      return captureScopeMatches(child, after, latest.current.session?.returning) ? true : null;
    } catch (error) { setRecordError(error.message); return false; }
    finally { setCaptureTransition(false); }
  }
  async function startEpisode() {
    if (pending.current || !targetReady || draft.active) return false;
    setRecordError(""); setSingleArmed(false); setSinglePending(false);
    const options = { robot_type: resolveInitialRobotType() || "unknown",
      capture_mode: captureMode, capture_interval_ms: intervalMs };
    pending.current = true;
    try {
      return await draft.start(preparation(options));
    } finally { pending.current = false; }
  }
  async function captureSingle() {
    if (pending.current || singleInFlight.current || !latest.current.draft.active) return;
    singleInFlight.current = true;
    const token = ++request.current;
    setRecordError(""); setSingleArmed(true);
    try {
      const ready = await ensureSegment();
      if (token !== request.current) return;
      if (ready === null) { setSinglePending(true); return; }
      setSinglePending(false);
      if (ready !== true) { setSingleArmed(false); return; }
      const ok = await latest.current.draft.capture(captureFields.current);
      if (ok === false) setSingleArmed(false);
    } finally { singleInFlight.current = false; }
  }
  useEffect(() => {
    if (!singlePending || !draft.active || draft.busy || !session?.ready) return;
    void captureSingle();
  });
  async function startCapture() {
    if (pending.current || !latest.current.draft.active) return;
    setRecordError(""); setSingleArmed(false); setSinglePending(false);
    // The pre-frame boundary is serialized by the draft loop with capture/save.
    await draft.startRecording(intervalMs, () => captureFields.current, ensureSegment);
  }
  async function finalize(discard = false) {
    if (pending.current) return;
    if (discard && !window.confirm("Delete this episode and all its images, labels, and masks? This cannot be undone.")) return;
    pending.current = true; stopCapture(); setCaptureTransition(true); setRecordError("");
    try { if (discard) await draft.discardEpisode(); else await draft.finish({ allowEmpty: true }); }
    catch (error) { setRecordError(error.message); }
    finally { pending.current = false; setCaptureTransition(false); }
  }
  return { captureMode, setCaptureMode, intervalMs, setIntervalMs, captureFields, recordError,
    captureTransition, captureType, targetReady,
    awaitingTarget, singlePending, executionReady,
    startEpisode, captureSingle, startCapture, stopCapture,
    resumeEpisode: async id => {
      setRecordError("");
      // Loading data must never move the robot or select a different route.
      stopCapture();
      await draft.load(id);
    },
    saveEpisode: () => finalize(), discardCapture: () => finalize(true),
  };
}
