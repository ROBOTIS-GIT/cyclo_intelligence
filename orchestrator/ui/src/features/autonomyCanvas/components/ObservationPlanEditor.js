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

import "./ObservationPlan.css";
import { observationSpots } from "../lib/observationSpots";
import { FiPlus, FiMinus, FiArrowUp, FiArrowDown, FiSave, FiRefreshCw } from "react-icons/fi";

export default function ObservationPlanEditor({ config, disabled = false }) {
  if (!config?.draft) return null;
  const { draft, missions = [], locked, edit } = config;
  const items = draft.observations;
  const spots = observationSpots({ ...config, plan: { observations: [...(config.plan?.observations || []), ...draft.observations] } });
  const update = (index, value) => edit({ observations: items.map((p, i) => i === index ? { ...p, ...value } : p) });
  const move = (index, delta) => {
    const next = [...items]; [next[index], next[index + delta]] = [next[index + delta], next[index]];
    edit({ observations: next });
  };
  const available = spots.filter(s => !items.some(p => p.spot_id === s.id));
  return <fieldset className="observation-plan" disabled={locked || disabled}>
    <legend>Observation plan</legend>
    <div className="observation-plan-toolbar">
      <select aria-label="Observation mode" value={draft.mode} onChange={e => edit({ mode: e.target.value })}>
        <option value="fixed">Fixed observation</option><option value="patrol">Patrol</option>
      </select>
      <button type="button" title="Add saved spot" aria-label="Add saved observation spot" disabled={!available.length} onClick={() => edit({ observations: [...items, { spot_id: available[0].id, departure_policy: "reasoning", dwell_seconds: 15, mission_ids: [], arrival_bt: "" }] })}><FiPlus /></button>
      <button type="button" title="Refresh spots and missions" aria-label="Refresh observation plan" onClick={() => { void config.refresh().catch(() => {}); }}><FiRefreshCw /></button>
      <button type="button" title="Save observation plan" aria-label="Save observation plan" disabled={!config.dirty} onClick={config.save}><FiSave /></button>
    </div>
    {draft.mode === "patrol" && <label className="observation-plan-dwell">Route behavior
      <select aria-label="Observation route behavior" value={draft.route_closed === false ? "one-way" : "closed-loop"}
        onChange={event => edit({ route_closed: event.target.value === "closed-loop" })}>
        <option value="closed-loop">Closed loop</option><option value="one-way">One-way</option>
      </select>
    </label>}
    {items.map((item, index) => <div className="observation-plan-point" key={item.spot_id}>
      <div className="observation-plan-toolbar">
        <span>{index + 1}</span>
        <select aria-label={`Observation spot ${index + 1}`} value={item.spot_id} onChange={e => update(index, { spot_id: e.target.value, arrival_bt: "" })}>
          {!spots.some(s => s.id === item.spot_id) && <option value={item.spot_id}>Deleted spot</option>}
          {spots.filter(s => s.id === item.spot_id || !items.some(p => p.spot_id === s.id)).map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
        <button type="button" aria-label={`Move observation ${index + 1} up`} disabled={!index} onClick={() => move(index, -1)}><FiArrowUp /></button>
        <button type="button" aria-label={`Move observation ${index + 1} down`} disabled={index === items.length - 1} onClick={() => move(index, 1)}><FiArrowDown /></button>
        <button type="button" aria-label={`Remove observation ${index + 1} from plan`} title="Remove from plan" onClick={() => edit({ observations: items.filter((_, i) => i !== index) })}><FiMinus /></button>
      </div>
      {draft.mode === "patrol" && <label className="observation-plan-dwell">Move to next
        <select aria-label={`Observation departure ${index + 1}`} value={item.departure_policy || "reasoning"}
          onChange={e => update(index, { departure_policy: e.target.value })}>
          <option value="reasoning">By decision</option><option value="timed">After timer</option>
        </select>
      </label>}
      {draft.mode === "patrol" && item.departure_policy === "timed" && <label className="observation-plan-dwell">Dwell (seconds)<input aria-label={`Dwell seconds ${index + 1}`} type="number" min="2" max="3600" value={item.dwell_seconds} onChange={e => update(index, { dwell_seconds: Number(e.target.value) })} /></label>}
      <details><summary>{item.mission_ids.length ? `${item.mission_ids.length} missions` : "No missions"}</summary>
        {missions.filter(m => !draft.excluded_mission_ids.includes(m.mission_id)).map(m => <label className="observation-plan-mission" key={m.mission_id}>
          <input type="checkbox" checked={item.mission_ids.includes(m.mission_id)} onChange={e => update(index, { mission_ids: e.target.checked ? [...item.mission_ids, m.mission_id] : item.mission_ids.filter(id => id !== m.mission_id) })} />{m.mission_name}
        </label>)}
        {item.mission_ids.filter(id => !missions.some(m => m.mission_id === id)).map(id => <label className="observation-plan-mission" key={id}>
          <input type="checkbox" checked onChange={() => update(index, { mission_ids: item.mission_ids.filter(value => value !== id) })} />Unavailable mission
        </label>)}
      </details>
    </div>)}
    {draft.mode === "fixed" && items.length > 1 && <p className="operate-sidebar-note">Fixed observation uses the first spot.</p>}
    {!items.length && <p className="operate-sidebar-note">Create an observation spot, then add it to this plan.</p>}
    {config.error && <p role="alert" className="operate-sidebar-note">{config.error}</p>}
    {config.dirty && <p className="operate-sidebar-note">Save changes before starting.</p>}
  </fieldset>;
}
