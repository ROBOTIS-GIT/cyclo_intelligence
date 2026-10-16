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

import { MdSave } from "react-icons/md";
import { episodeCategory, categoryLabel } from "./episodeCategory";
import { frameMatchesEpisode } from "../utils/reasoningDataset";
import { DECISION_CHOICES, SELECTION_TASK, PATROL_TASK, decisionNeedsMission } from "../utils/reasoningLabelContract";

export default function CollectReasoningFields({draft,editor,onSave,compact=false,extraActions}) {
  const {active,selectedFrame}=draft;
  const {decision,setDecision,missionId,setMissionId,reason,setReason,saving,maskBusy,
    receipt,loadedFrame,busy,missions,reasonPresets=[],selectedLabel,frameMission,invalidDecisionContext,
    missionRequired,deletedSelectedMission,selectedMissionUnavailable,
    supervision=false,reasoningTask=SELECTION_TASK,
    decisionChoices=DECISION_CHOICES[SELECTION_TASK],unsupportedSavedLabel=false}=editor;
  const patrol = reasoningTask === PATROL_TASK || episodeCategory(draft.episode) === "patrol";
  const missionEpisode = episodeCategory(draft.episode) === "mission";
  const missingMissionContext = editor.episodeFrameValid === false ||
    (draft.episode && !frameMatchesEpisode(draft.episode, selectedFrame)) ||
    (missionEpisode && !supervision && !patrol);
  const selection = !supervision && !patrol && !missionEpisode;
  const canChooseMission = selection && decision === "start";
  const missionPlaceholder = canChooseMission ? "Select a mission" : "No mission required";
  const decisionTitle = patrol ? "Movement action" : selection ? "Expected decision" : "Mission action";
  const frameMissionName = missions.find(m => m.mission_id === frameMission?.mission_id)?.mission_name
    || (deletedSelectedMission?.mission_id === frameMission?.mission_id ? deletedSelectedMission?.mission_name : null)
    || frameMission?.mission_id;
  const matchingReasonPresets = reasonPresets.filter((preset) => (
    preset.decision === decision &&
    (!preset.reasoning_task || preset.reasoning_task === reasoningTask) &&
    (!decisionNeedsMission(decision, reasoningTask) || preset.mission_id === missionId)
  ));
  const selectedPreset = matchingReasonPresets.some((preset) => preset.reason === reason)
    ? reason
    : "";
  return <div className="collect-reasoning-fields">
      {!active && selectedFrame && (
        <form
          className={`collect-label-form${compact ? " collect-label-form-compact" : ""}${selection ? "" : " collect-supervision-form"}`}
          onSubmit={(e) => {
            e.preventDefault();
            if (!missingMissionContext) onSave();
          }}
        >
          {!compact && <strong>{patrol ? "Patrol supervision" : selection ? "Observation · Mission selection" : "Mission supervision"}</strong>}
          {missingMissionContext && <p className="operate-sidebar-note" role="status">This frame has no valid {categoryLabel(draft.episode)} context. Select a frame captured during the matching activity.</p>}
          {unsupportedSavedLabel && <p className="operate-sidebar-note" role="status">This saved label does not match the recorded task. Select a valid decision to review it.</p>}
          <label className="collect-field collect-reasoning-decision">
            {decisionTitle}
            <select
              value={missingMissionContext ? "" : decision}
              disabled={busy || maskBusy || missingMissionContext}
              onChange={(e) => {
                const nextDecision = e.target.value;
                setDecision(nextDecision);
                if (!supervision && nextDecision !== "start") setMissionId("");
              }}
            >
              {missingMissionContext && <option value="">No valid execution context</option>}
              {!missingMissionContext && !decisionChoices.includes(decision) && <option value={decision} disabled>{decision ? `${decision} (legacy)` : "Select a decision"}</option>}
              {!missingMissionContext && decisionChoices.map((value) => (

                <option
                  key={value}
                  value={value}
                  disabled={
                    value === "start" && !!frameMission
                  }
                >
                  {value === "next_observation" ? "Next observation" : value === "cancel" ? "Cancel and return to patrol" : value.charAt(0).toUpperCase() + value.slice(1)}
                </option>
              ))}
            </select>
          </label>
          {selection && <label className="collect-field collect-reasoning-mission">
            Mission
            <select
              value={missionId}
              disabled={busy || maskBusy || missingMissionContext || !canChooseMission}
              onChange={(e) => setMissionId(e.target.value)}
            >
              <option value="">{missionPlaceholder}</option>
              {deletedSelectedMission && (
                <option value={deletedSelectedMission.mission_id} disabled>
                  {deletedSelectedMission.mission_name} (deleted)
                </option>
              )}
              {missions.map((m) => (
                <option key={m.mission_id} value={m.mission_id}>
                  {m.mission_name}
                </option>
              ))}
            </select>
          </label>}
          {!missions.length && !deletedSelectedMission && canChooseMission && (
            <p className="operate-sidebar-note">
              No missions are available for this frame.
            </p>
          )}
          {selection && !missionRequired && (
            <p className="operate-sidebar-note">No mission is required for this decision.</p>
          )}
          {frameMission && <p className="operate-sidebar-note">
            Active mission: {frameMissionName}{frameMission.action ? ` · ${frameMission.action}` : ""}
            {selectedFrame.context?.runtime_source === "operator_supplied_preview" ? " (test context)" : ""}
          </p>}
          <label className="collect-field collect-reasoning-preset">
            Reason preset
            <select
              value={selectedPreset}
              disabled={busy || maskBusy || missingMissionContext || !matchingReasonPresets.length}
              onChange={(e) => {
                if (e.target.value) setReason(e.target.value);
              }}
            >
              <option value="">
                {matchingReasonPresets.length ? "Write a custom reason" : "No saved reasons yet"}
              </option>
              {matchingReasonPresets.map((preset) => (
                <option
                  key={`${preset.decision}/${preset.mission_id}/${preset.reason}`}
                  value={preset.reason}
                >
                  {preset.reason} · {preset.count} {preset.count === 1 ? "label" : "labels"}
                </option>
              ))}
            </select>
          </label>
          <label className="collect-field collect-reasoning-reason">
            Reason
            <textarea
              rows={3}
              value={reason}
              disabled={busy || maskBusy || missingMissionContext}
              onChange={(e) => setReason(e.target.value)}
              maxLength={1000}
              placeholder={patrol ? "Explain why patrol should continue or pause." : selection ? "Describe the visible scene and explain your decision." : "Explain why the current action should continue, pause, or be canceled."}
            />
          </label>
          <div className="collect-actions">
            <button
              type="submit"
              className="collect-primary"
              aria-label={saving ? "Saving…" : "Save"}
              title={saving ? "Saving…" : "Save"}
              disabled={
                missingMissionContext ||
                invalidDecisionContext ||
                selectedMissionUnavailable ||
                loadedFrame !== selectedFrame.frame_id ||
                busy ||
                maskBusy ||
                !reason.trim() ||
                (missionRequired && !missionId)
              }
            >
              {compact ? <MdSave size={18}/> : saving ? "Saving…" : "Save"}
            </button>
            {extraActions}
          </div>
          {missionRequired && !missionId && (
            <p className="operate-sidebar-note">{supervision ? "Supervision requires an active mission captured by the server." : "Select a mission above."}</p>
          )}
          {selectedMissionUnavailable && (
            <p className="operate-sidebar-note">
              Selected mission is no longer available in this frame's candidates. Select an available mission.
            </p>
          )}
          <p role="status" className="operate-sidebar-note">
            {receipt || (selectedLabel ? `Saved label · ${selectedLabel.review_status}` : "")}
          </p>
        </form>
      )}

  </div>;
}
