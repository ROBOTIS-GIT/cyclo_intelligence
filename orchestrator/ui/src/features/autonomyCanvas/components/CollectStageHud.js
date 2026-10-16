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

import { useContext, useRef, useState } from "react";
import { CollectCaptureContext } from "./CollectCaptureContext";
import RecordAuthoringToolbar from "./RecordAuthoringToolbar";
import RecordMapEditOverlay from "./RecordMapEditOverlay";
import RecordRouteEditor from "./RecordRouteEditor";
import RecordControlPanel from "./RecordControlPanel";
import RecordMissionControl from "./RecordMissionControl";
import { MdMyLocation, MdPlayArrow, MdPause, MdSkipNext, MdSkipPrevious, MdStop } from "react-icons/md";

export default function CollectStageHud({ onLocalize, localizing, localized = false, localizeDisabled,
  missionNames = [], selectedMission = "", missionLoading, missionDisabled, onMissionChange,
  observations, waypointAnchor, mapName, observationName, executionLabel, canStartMission, onRunMission,
  clickNavigationDisabled, observationSelectionDisabled, onGoToObservation,
  shutdownPending, onStopRoute, onStart, startDisabled, patrolRunning = false, patrolPaused = false, onPausePatrol, onResumePatrol,
  canSkipObservation = false, onSkipObservation, canPreviousObservation = false, onPreviousObservation, routeActive, authoringDisabled, stopping: runtimeStopping, onMessage }) {
  const captureSession = useContext(CollectCaptureContext);
  const commandPending = useRef(false), stopPending = useRef(false), commandVersion = useRef(0);
  const [stopping, setStopping] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const perform = async (action, interrupt = false) => {
    if (interrupt ? stopPending.current : commandPending.current) return;
    const token = ++commandVersion.current;
    commandPending.current = true;
    if (interrupt) { stopPending.current = true; setStopping(true); }
    setPending(true); setError("");
    try { if (await action?.() === false) throw new Error("The command could not be completed."); }
    catch (failure) { if (token === commandVersion.current) { setError(failure.message); onMessage?.(failure.message); } }
    finally {
      if (interrupt) { stopPending.current = false; setStopping(false); }
      if (token === commandVersion.current) { commandPending.current = false; setPending(false); }
    }
  };
  const locked = pending || runtimeStopping || shutdownPending;
  const icon = (label, Icon, action, disabled, className = "", pressed) => <button type="button" aria-label={label} title={label}
    onClick={action} disabled={disabled} className={`record-control-icon ${className}`} aria-pressed={pressed}><Icon size={18}/></button>;
  const editing = !!observations?.route?.routeMode;
  const authoringLocked = locked || routeActive || authoringDisabled;
  return <>
    <div className={`record-map-tools${editing ? " is-editing" : observations?.draft ? " is-creating" : ""}`}>
      <div className="record-movement-controls" role="toolbar" aria-label="Movement controls">
        {icon("Localize",MdMyLocation,onLocalize,localizeDisabled || locked || editing || !!observations?.draft,"",!!localizing)}
        {icon("Previous observation",MdSkipPrevious,()=>perform(onPreviousObservation),!localized || !canPreviousObservation || locked)}
        {icon(patrolRunning ? "Pause patrol" : patrolPaused ? "Resume patrol" : "Start patrol",patrolRunning ? MdPause : MdPlayArrow,
          ()=>perform(patrolRunning ? onPausePatrol : patrolPaused ? onResumePatrol : onStart),!localized || startDisabled || locked,"record-patrol-button")}
        {icon("Next observation",MdSkipNext,()=>perform(onSkipObservation),!localized || !canSkipObservation || locked)}
        {icon("Stop",MdStop,()=>perform(onStopRoute,true),stopping || runtimeStopping || shutdownPending || !routeActive,"record-stop-button")}
      </div>
      {observations && <><span className="record-tool-separator" aria-hidden="true" />
        <RecordAuthoringToolbar manager={observations} disabled={authoringLocked}/></>}
    </div>
    {editing && <div className="record-route-bar-position"><RecordRouteEditor manager={observations} disabled={authoringLocked}/></div>}
    <RecordMapEditOverlay manager={observations} anchor={waypointAnchor} disabled={authoringLocked}
      selectionDisabled={observationSelectionDisabled}
      goDisabled={!localized || clickNavigationDisabled || locked || !!observations?.draft || observations?.planDirty || editing}
      onGoToObservation={id => perform(() => onGoToObservation?.(id))}/>

    <RecordControlPanel error={error}
    capture={<>
      {captureSession?.controls}
      <RecordMissionControl {...{missionNames,selectedMission,onMissionChange}}
        canStartMission={canStartMission} onRunMission={() => perform(onRunMission)}
        loading={missionLoading} disabled={missionDisabled || locked}/>
    </>}
    footer={<div className="record-movement-status"><span title={observationName}>{observationName || (localized ? "Localized" : "Not localized")}</span><span>{executionLabel || "Idle"}</span></div>}/>
  </>;
}
