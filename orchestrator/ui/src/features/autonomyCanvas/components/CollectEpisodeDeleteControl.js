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

import { useEffect, useRef, useState } from "react";
import { MdDeleteOutline } from "react-icons/md";
import { EPISODE_TYPES, episodeListDetails } from "./episodeListDetails";
import "./CollectEpisodeDeleteControl.css";

export default function CollectEpisodeDeleteControl({ draft, mapName, disabled }) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState([]);
  const [deleted, setDeleted] = useState([]);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef(false);
  useEffect(() => { setOpen(false); setFilter("all"); setSelected([]); setDeleted([]); setError(""); }, [mapName]);
  const episodes = draft.episodes.filter(ep => ep.map_name === mapName);
  if (draft.episode?.map_name === mapName && !episodes.some(ep => ep.id === draft.episode.id)) episodes.unshift(draft.episode);
  const spotNames = {};
  [...episodes].sort((a,b) => (a.started_at || 0) - (b.started_at || 0)).forEach(ep => {
    [ep.observation_spot, ep.return_observation_spot].forEach(spot => { if (spot?.id && spot.label) spotNames[spot.id] = spot.label; });
  });
  const rows = episodes.filter(ep => !deleted.includes(ep.id)).map(ep => ({...episodeListDetails(ep, spotNames), episode:ep}));
  const visible = rows.filter(row => filter === "all" || row.type === filter);
  const chosen = visible.filter(row => selected.includes(row.episode.id));
  const groups = [...new Set(visible.map(row => row.group))];
  // Keep controls stable between writes while continuous capture is active.
  const locked = disabled || deleting || draft.recording || draft.busy;
  const toggle = id => setSelected(ids => ids.includes(id) ? ids.filter(value => value !== id) : [...ids, id]);
  async function removeSelected() {
    if (locked || pending.current || !chosen.length) return;
    if (!window.confirm(`Delete ${chosen.length} selected episodes and all their images, labels, and masks?\n${chosen.map(row => row.name).join("\n")}\nThis cannot be undone.`)) return;
    pending.current = true;
    setDeleting(true); setError("");
    const failures = [];
    try {
      for (const row of chosen) {
        try {
          if (await draft.deleteEpisode(row.episode.id) === false) throw new Error("Delete failed");
          setDeleted(ids => [...ids, row.episode.id]);
          setSelected(ids => ids.filter(id => id !== row.episode.id));
        } catch (err) { failures.push(`${row.name}: ${err.message || "Delete failed"}`); }
      }
      if (failures.length) setError(failures.join("; "));
    } finally { pending.current = false; setDeleting(false); }
  }
  return <div className="collect-episode-delete">
    <button type="button" className="collect-capture-delete" aria-label="Delete episode" title="Delete episodes"
      aria-expanded={open} disabled={locked} onClick={() => setOpen(value => !value)}>
      <MdDeleteOutline size={18} aria-hidden="true" />
    </button>
    {open && <div className="collect-episode-delete-list" role="region" aria-label="Saved episodes">
      <div className="episode-type-filters" role="group" aria-label="Episode type">
        {Object.entries({all:"All", ...EPISODE_TYPES}).map(([value,label]) =>
          <button key={value} type="button" aria-pressed={filter === value} disabled={deleting}
            onClick={() => { setFilter(value); setSelected([]); setError(""); }}>{label}</button>)}
      </div>
      <div className="episode-selection-actions">
        <label><input type="checkbox" aria-label="Select all visible episodes" disabled={locked || !visible.length}
          checked={visible.length > 0 && chosen.length === visible.length}
          ref={node => { if(node) node.indeterminate = chosen.length > 0 && chosen.length < visible.length; }}
          onChange={event => setSelected(event.target.checked ? visible.map(row => row.episode.id) : [])}/>Select all</label>
        <button type="button" disabled={locked || !chosen.length} onClick={removeSelected}>
          <MdDeleteOutline aria-hidden="true"/>{deleting ? "Deleting…" : `Delete (${chosen.length})`}
        </button>
      </div>
      {groups.map(group => <section key={group} className="episode-delete-group">
        <strong>{group}</strong>
        {visible.filter(row => row.group === group).map(row => <label key={row.episode.id} className="episode-delete-row">
          <input type="checkbox" aria-label={`Select ${row.name}`} disabled={locked}
            checked={selected.includes(row.episode.id)} onChange={() => toggle(row.episode.id)}/>
          <span><span className="episode-delete-name">{row.name}</span>
            <small>{row.frames} frames{!row.episode.ended_at ? " · Open" : ""}</small>
            {row.detail && <small>{row.detail}</small>}</span>
        </label>)}
      </section>)}
      {!visible.length && <span>No episodes for this filter.</span>}
      {error && <p role="alert">{error}</p>}
    </div>}
  </div>;
}
