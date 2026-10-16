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
import { categoryLabel, episodeCategory, EPISODE_TYPES, TYPE_LABELS } from "./episodeCategory";
import { MdRefresh, MdDeleteOutline } from "react-icons/md";
import { Panel } from "../../mapWorkspace/components/primitives";

export default function CollectEpisodeLibrary({ draft, disabled, masks = false, reviewMode = false, combined = false, onDelete }) {
  const [map, setMap] = useState("");
  const [spot, setSpot] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState(() => episodeCategory(draft.episode || draft.episodes.find(ep => ep.ended_at)));
  const selectedEpisode = draft.episode;
  useEffect(() => { if (selectedEpisode?.ended_at) setCategory(episodeCategory(selectedEpisode)); }, [selectedEpisode]);
  const episodes = draft.episodes.filter(ep => ep.ended_at);
  const maps = [...new Set(episodes.map(ep => ep.map_name).filter(Boolean))].sort();
  const mapFilter = maps.includes(map) ? map : "";
  const spotOf = ep => ep.observation_spot_id || ep.observation_spot?.id || "";
  const spotKey = ep => JSON.stringify([ep.map_name || "", spotOf(ep)]);
  const spotLabel = ep => ep.observation_spot?.label || spotOf(ep) || "Unassigned";
  const observations = episodes.filter(ep => episodeCategory(ep) === "observation" && (!mapFilter || ep.map_name === mapFilter));
  const groups = [...new Map(observations.map(ep => [spotKey(ep), {
    key: spotKey(ep), label: `${mapFilter ? "" : `${ep.map_name || "No map"} / `}${spotLabel(ep)}`,
  }])).values()].sort((a, b) => a.label.localeCompare(b.label));
  const spotFilter = groups.some(group => group.key === spot) ? spot : "";
  const nameOf = ep => ep.storage_path?.split("/").filter(Boolean).pop() || ep.scenario || ep.id;
  const contextOf = ep => [ep.map_name || "No map", categoryLabel(ep), episodeCategory(ep) === "observation" && spotOf(ep) ? spotLabel(ep) : null, ep.capture_segment === "mission_return" ? `${ep.return_mission_name || "Mission"} return` : ep.recording_mission_name].filter(Boolean).join(" / ");
  const legacyContextOf = ep => [ep.map_name || "No map", ep.capture_segment === "mission_return" ? `${ep.return_mission_name || "Mission"} return` : ep.recording_mission_name].filter(Boolean).join(" / ");
  const visible = episodes.filter(ep => episodeCategory(ep) === category && (!mapFilter || ep.map_name === mapFilter) &&
    (category !== "observation" || !spotFilter || spotKey(ep) === spotFilter) &&
    `${spotLabel(ep)} ${nameOf(ep)} ${ep.map_name || ""} ${episodeCategory(ep)} ${ep.recording_mission_name || ""} ${ep.scenario || ""}`.toLowerCase().includes(query.trim().toLowerCase()));
  const countFor = type => episodes.filter(ep => episodeCategory(ep) === type && (!mapFilter || ep.map_name === mapFilter)).length;
  function chooseCategory(next) {
    if (next === category) return;
    setCategory(next);
    const first = episodes.find(ep => episodeCategory(ep) === next && (!mapFilter || ep.map_name === mapFilter));
    if (first?.id !== draft.episode?.id) draft.load(first?.id || "");
  }
  const locked = disabled || draft.busy || draft.active;
  return <Panel title="Episodes" className="collect-episode-library">
    <div className="collect-library-tools">
      <input aria-label="Search episodes" placeholder="Search episodes" value={query}
        onChange={e => setQuery(e.target.value)} />
      <button type="button" title="Refresh episodes" aria-label="Refresh episodes"
        disabled={locked} onClick={draft.refreshEpisodes}><MdRefresh size={18} /></button>
      {onDelete && <button type="button" title="Delete selected episode" aria-label="Delete selected episode"
        disabled={locked || !draft.episode?.ended_at} onClick={onDelete}><MdDeleteOutline size={18} /></button>}
    </div>
    <select aria-label="Filter episodes by map" value={mapFilter} onChange={e => { setMap(e.target.value); setSpot(""); }}>
      <option value="">All maps</option>
      {maps.map(name => <option key={name} value={name}>{name}</option>)}
    </select>
    <div className="collect-library-categories" role="group" aria-label="Episode type">
      {EPISODE_TYPES.map(type => <button key={type} type="button" aria-pressed={category === type}
        disabled={locked} onClick={() => chooseCategory(type)}>{TYPE_LABELS[type]} <span>{countFor(type)}</span></button>)}
    </div>
    {category === "observation" && <select aria-label="Filter episodes by observation point" value={spotFilter}
      onChange={e => setSpot(e.target.value)}>
      <option value="">All observation points</option>
      {groups.map(group => <option key={group.key} value={group.key}>
        {group.label} ({observations.filter(ep => spotKey(ep) === group.key).length})
      </option>)}
    </select>}
    <span className="collect-library-count">{visible.length} / {countFor(category)} {category} episodes</span>
    <div className="collect-library-list" role="list" aria-label="Saved episodes">
      {visible.map(ep => {
        const selected = draft.episode?.id === ep.id;
        const eligible = ep.reasoning_frame_count ?? ep.frame_count;
        const approved = ep.reasoning_approved_count ?? Math.max(0, (ep.frame_count || 0) - (ep.unlabeled_count ?? ep.frame_count ?? 0));
        const unlabeled = ep.reasoning_frame_count != null ? eligible - approved : ep.unlabeled_count;
        const complete = eligible > 0 && unlabeled === 0;
        return <div key={ep.id} role="listitem">
          <button type="button" className="collect-library-item" aria-pressed={selected}
            disabled={locked} onClick={() => { if (!selected) draft.load(ep.id); }}>
            <strong>{nameOf(ep)}</strong>
            <span>{contextOf(ep)}</span>
            {ep.scenario && ep.scenario.split("/").map(s => s.trim()).join(" / ") !== legacyContextOf(ep) && <span className="collect-library-scenario" title={ep.scenario}>{ep.scenario}</span>}
            <span className="collect-library-progress">
              <span>{ep.frame_count ?? "—"} frames</span>
              <span>{combined ? `${ep.reasoning_approved_count != null || ep.unlabeled_count != null ? approved : "—"} decisions · ${ep.mask_labeled_count ?? 0} masks` : masks ? (reviewMode ? `${ep.mask_reviewed_count ?? 0} reviewed · ${ep.mask_pending_count ?? 0} pending` : `${ep.mask_labeled_count ?? ep.mask_reviewed_count ?? 0} labeled`) : complete ? "Labeled" : unlabeled != null ? `${unlabeled} unlabeled` : "Not reviewed"}</span>
            </span>
          </button>
        </div>;
      })}
      {!visible.length && <p className="operate-sidebar-note">{episodes.length ? `No ${category} episodes match this filter.` : "No saved episodes. Save an episode in Record to begin."}</p>}
    </div>
    {draft.error && <p role="alert" className="operate-sidebar-note">{draft.error}</p>}
  </Panel>;
}
