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

import { FiTrash2, FiX } from "react-icons/fi";
import { RecordToolButton } from "./RecordAuthoringControls";
import "./RecordMapEditor.css";

export default function RecordObservationEditor({ manager, disabled = false, onClose }) {
  const route = manager.route, editor = manager.routeEditor;
  const spot = manager.spots?.find(value => value.id === manager.selectedId);
  if (!route?.routeMode || !editor?.plan || !spot) return null;
  const includedPoint = editor.plan.observations?.find(point => point.spot_id === spot.id);
  const point = editor.pointFor?.(spot.id) || includedPoint;
  const missionIds = point?.mission_ids || [];
  const missions = editor.missions || [];
  const excluded = new Set(editor.plan.excluded_mission_ids || []);
  const visibleMissions = missions.filter(mission => !excluded.has(mission.mission_id) || missionIds.includes(mission.mission_id));
  const unavailableIds = missionIds.filter(id => !missions.some(mission => mission.mission_id === id));
  const locked = disabled || editor.locked || manager.working || manager.blocked;
  function close() { manager.select(""); onClose?.(); }
  function setMission(id, checked) {
    editor.updatePoint(spot.id, { mission_ids: checked ? [...new Set([...missionIds, id])] : missionIds.filter(value => value !== id) });
  }
  function remove() {
    if (locked || !window.confirm(`Delete waypoint "${spot.label}"? It will be removed when you save the route.`)) return;
    void Promise.resolve().then(() => editor.removeWaypoint(spot.id)).then(result => { if (result !== false) close(); }).catch(() => {});
  }

  return <section className="record-observation-editor" aria-label="Observation waypoint settings"
    onKeyDown={event => { if (event.key === "Escape") { event.stopPropagation(); close(); } }}>
    <div className="record-observation-editor-heading">
      <strong>Waypoint</strong>
      <RecordToolButton label="Close waypoint settings" onClick={close}><FiX aria-hidden="true" /></RecordToolButton>
    </div>
    <fieldset className="record-observation-editor-fields" disabled={locked}>
      <label className="record-observation-field">Name
        <input aria-label="Observation name" maxLength={128} value={spot.label || ""}
          onChange={event => editor.updateSpot(spot.id, { label: event.target.value })} />
      </label>
      <div className="record-observation-editor-row">
        <label className="record-observation-field">Move to next
          <select aria-label="Observation departure" disabled={!includedPoint}
            value={point?.departure_policy || "reasoning"}
            onChange={event => editor.updatePoint(spot.id, { departure_policy: event.target.value })}>
            <option value="reasoning">By decision</option><option value="timed">After timer</option>
          </select>
        </label>
        {point?.departure_policy === "timed" && <label className="record-observation-field record-observation-dwell">Dwell (seconds)
          <input aria-label="Dwell seconds" type="number" min="2" max="3600" step="any"
            disabled={!includedPoint} value={point?.dwell_seconds ?? 15}
            onChange={event => editor.updatePoint(spot.id, { dwell_seconds: Number(event.target.value) })} />
        </label>}
        <label className="record-observation-checkbox record-observation-include">
          <input type="checkbox" checked={!!includedPoint}
            onChange={event => route.setIncluded(spot.id, event.target.checked)} />Include in route
        </label>
      </div>
      {!includedPoint && <span className="record-map-editor-note">Include in route to edit departure and missions.</span>}
      <div className="record-observation-missions" role="group" aria-label="Linked missions">
        <div className="record-observation-editor-heading"><strong>Linked missions</strong><span className="record-map-editor-note">Optional</span></div>
        <div className="record-observation-mission-list">
          {visibleMissions.map(mission => <label className="record-observation-checkbox" key={mission.mission_id}>
            <input type="checkbox" disabled={!includedPoint} checked={missionIds.includes(mission.mission_id)}
              onChange={event => setMission(mission.mission_id, event.target.checked)} />
            <span>{mission.mission_name || mission.mission_id}{excluded.has(mission.mission_id) ? " (excluded)" : ""}</span>
          </label>)}
          {unavailableIds.map(id => <label className="record-observation-checkbox" key={id}>
            <input type="checkbox" disabled={!includedPoint} checked onChange={() => setMission(id, false)} />
            <span>Unavailable mission ({id})</span>
          </label>)}
          {!visibleMissions.length && !unavailableIds.length && <span className="record-map-editor-note">No missions available. This waypoint can stay unlinked.</span>}
        </div>
      </div>
      <div className="record-observation-editor-footer">
        <span className="record-map-editor-note">Changes saved with route.</span>
        <RecordToolButton label="Delete waypoint" onClick={remove}><FiTrash2 aria-hidden="true" /></RecordToolButton>
      </div>
    </fieldset>
  </section>;
}
