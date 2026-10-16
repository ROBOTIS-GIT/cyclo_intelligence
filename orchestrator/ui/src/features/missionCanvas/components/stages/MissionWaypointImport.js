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

import { useEffect, useState } from "react";
import { getNavigationMission } from "../../../../utils/navigationMissionsApi";
import { missionRequestName } from "../../../mapWorkspace/lib/missionNames";
import { WaypointOptionButton } from "../../../mapWorkspace/components/primitives";

export default function MissionWaypointImport({ mapName, missionName, missionNames = [], disabled, onImport }) {
  const [open, setOpen] = useState(false);
  const [source, setSource] = useState("");
  const [waypoints, setWaypoints] = useState([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const names = missionNames.filter((name) => name !== missionName);

  useEffect(() => {
    let current = true;
    setWaypoints([]);
    setSelected("");
    setError("");
    setLoading(Boolean(open && source));
    if (open && source) {
      getNavigationMission(mapName, missionRequestName(source)).then((mission) => {
        if (!current) return;
        const items = mission?.waypoints || [];
        setWaypoints(items);
        if (!items.length) setError("This mission has no saved waypoints.");
      }).catch((failure) => {
        if (current) setError(failure.message || "Unable to load waypoints.");
      }).finally(() => { if (current) setLoading(false); });
    }
    return () => { current = false; };
  }, [mapName, source, open]);

  const fieldStyle = { backgroundColor: "var(--mc-surface)", color: "var(--mc-text)", border: "1px solid var(--mc-border)", borderRadius: 8, fontFamily: "inherit" };
  return (
    <div>
      <WaypointOptionButton active={open} disabled={disabled} onClick={() => setOpen(!open)}>From Mission</WaypointOptionButton>
      {open && (
        <div className="absolute left-0 top-full mt-2 w-72 flex flex-col gap-3 p-3 text-[12.5px]" role="group" aria-label="Import waypoint from mission"
          style={{ ...fieldStyle, boxShadow: "var(--mc-shadow)" }}>
          <label className="flex flex-col gap-1">
            <span className="text-[12.5px] font-semibold" style={{ color: "var(--mc-text-subtle)" }}>Source mission</span>
            <select aria-label="Source mission" className="w-full h-8 px-2 text-[13px] font-medium" style={{ ...fieldStyle, backgroundColor: "var(--mc-surface-2)", borderColor: "var(--mc-border-strong)" }} value={source} disabled={disabled}
              onChange={(event) => { setWaypoints([]); setSelected(""); setSource(event.target.value); }}>
              <option value="">Select a mission</option>
              {names.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {!names.length && <span>No other saved missions on this map.</span>}
          <label className="flex flex-col gap-1">
            <span className="text-[12.5px] font-semibold" style={{ color: "var(--mc-text-subtle)" }}>Waypoint</span>
            <select aria-label="Source waypoint" className="w-full h-8 px-2 text-[13px] font-medium" style={{ ...fieldStyle, backgroundColor: "var(--mc-surface-2)", borderColor: "var(--mc-border-strong)" }} value={selected}
              disabled={disabled || loading || !waypoints.length} onChange={(event) => setSelected(event.target.value)}>
              <option value="">{loading ? "Loading…" : "Select a waypoint"}</option>
              {waypoints.map((waypoint, index) => <option key={waypoint.id || index} value={String(index)}>{waypoint.label || waypoint.id}</option>)}
            </select>
          </label>
          <span style={{ color: "var(--mc-text-muted)" }}>Copy name, position and heading. Actions are not copied.</span>
          {error && <span role="alert">{error}</span>}
          <button type="button" className="w-full h-8 text-[12.5px] font-semibold disabled:opacity-40" style={{ ...fieldStyle, backgroundColor: "var(--mc-accent)", color: "var(--mc-accent-fg)" }}
            disabled={disabled || loading || selected === "" || !waypoints[Number(selected)]}
            onClick={async () => {
              try { await onImport(waypoints[Number(selected)], mapName); }
              catch (failure) { setError(failure.message || "Unable to import waypoint."); }
            }}>Add waypoint</button>
        </div>
      )}
    </div>
  );
}
