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

import { useState } from "react";
import { MdNoteAdd, MdPhotoCamera, MdPause, MdSave } from "react-icons/md";
import CollectEpisodeDeleteControl from "./CollectEpisodeDeleteControl";
import "./CollectCaptureControls.css";

export default function CollectCaptureControls({
  draft,
  recording,
  mapName,
  mapLoaded,
  recordMission,
  recordObservationSpotId,
  recordSession,
  disabled,
}) {
  const [singleRequestPending, setSingleRequestPending] = useState(false);
  const { episode, active, busy, recording: capturing } = draft;
  const {
    captureMode,
    setCaptureMode,
    intervalMs,
    setIntervalMs,
    captureTransition,
    recordError,
    startCapture,
    startEpisode,
    captureSingle,
    saveEpisode,
  } = recording;
  const captureType = recording.captureType ?? (recordSession?.ready ? recordSession.type : null);
  const captureTypeLabel = { observation: "Observation", patrol: "Patrol", mission: "Mission" }[captureType] || "Awaiting execution";
  const frames = episode?.observations?.length || 0;
  const targetReady = recording.targetReady ?? !recordSession;
  const capturePending = capturing || recording.singlePending || singleRequestPending;
  const toggleCapture = async () => {
    if (capturePending) {
      recording.stopCapture();
    } else if (captureMode === "single") {
      // The hook also exposes queued frames; keep an in-flight single request stoppable.
      setSingleRequestPending(true);
      try { await captureSingle(); }
      finally { setSingleRequestPending(false); }
    } else {
      await startCapture();
    }
  };
  const locked = busy || captureTransition;
  const openEpisodes = draft.episodes.filter(ep => !ep.ended_at && ep.map_name === mapName);
  return (
    <section className="collect-capture-controls collect-capture-compact" aria-label="Episode capture">
      <div className="collect-capture-heading collect-capture-summary-heading">
        <span aria-label="Recording type">{captureTypeLabel}</span>
        <span role="status">
          {active
            ? `${capturePending ? "● " : ""}${episode?.capture_segment === "mission_return" ? "Return · " : ""}${frames} frames`
            : episode
              ? `${frames} frames saved`
              : ""}
        </span>
      </div>
      {!episode && openEpisodes.length > 0 && (
        <label>
          Resume episode
          <select
            value=""
            disabled={locked || disabled}
            onChange={(e) => recording.resumeEpisode(e.target.value)}
          >
            <option value="">Select an open episode</option>
            {openEpisodes.map((ep) => (
              <option key={ep.id} value={ep.id}>
                {ep.capture_segment === "mission_return" ? "Return · " : ""}{ep.storage_path?.split("/").pop() || ep.scenario}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="collect-capture-options">
        <label>
          Capture mode
          <select value={captureMode} disabled={locked || disabled || capturePending}
            onChange={(e) => setCaptureMode(e.target.value)}>
            <option value="single">Single frame</option>
            <option value="continuous">Continuous</option>
          </select>
        </label>
        {captureMode === "continuous" && <label>
          Interval
          <select aria-label="Capture interval" value={intervalMs}
            disabled={locked || disabled || capturePending || captureMode !== "continuous"}
            onChange={(e) => setIntervalMs(Number(e.target.value))}>
            <option value={500}>0.5 sec</option>
            <option value={1000}>1 sec</option>
            <option value={2000}>2 sec</option>
            <option value={5000}>5 sec</option>
          </select>
        </label>}
      </div>
      <div className="collect-capture-buttons" role="group" aria-label="Episode actions">
        <button className="collect-capture-start" type="button" aria-label="Start episode" title="Start episode"
          disabled={active || locked || disabled || !mapLoaded || !targetReady}
          onClick={startEpisode}>
          <MdNoteAdd size={18} aria-hidden="true" />
        </button>
        <button type="button" aria-label={capturePending ? "Stop capture" : "Capture"}
          title={capturePending ? "Pause capture" : captureMode === "single" ? "Capture single frame" : "Start continuous capture"}
          aria-pressed={!!capturePending} className={capturePending ? "collect-capture-active" : ""}
          disabled={!capturePending && (!active || locked || disabled || !mapLoaded)}
          onClick={toggleCapture}>
          {capturePending ? <MdPause size={18} aria-hidden="true" /> : <MdPhotoCamera size={18} aria-hidden="true" />}
        </button>
        <button type="button" aria-label="Save episode" title="Save episode" disabled={!active || captureTransition} onClick={saveEpisode}>
          <MdSave size={18} aria-hidden="true" />
        </button>
        <CollectEpisodeDeleteControl draft={draft} mapName={mapName}
          disabled={captureTransition || disabled} />
      </div>
      {recordSession?.returning && <p role="status" className="text-xs">Returning to {recordSession.returnSpotName || "observation"}</p>}
      {recordSession && !targetReady && !active && <p role="status" className="text-xs">Save an observation route to prepare capture.</p>}
      {active && capturePending && recording.awaitingTarget && <p role="status" className="text-xs">
        {recordSession?.waitingReason
          ? `Capture waiting · ${recordSession.waitingReason}`
          : "Waiting for execution"}
      </p>}
      {(recordError || draft.error) && <p role="alert">{recordError || draft.error}</p>}
    </section>
  );
}
