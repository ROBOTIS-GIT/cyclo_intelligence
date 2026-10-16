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

export default function LabelerDatasetOptions({maps=[],mapName,onMapChange,disabled,kind,target,onTargetChange,hasSelectedEpisode=false}) {
  return <div className="labeler-dataset-options">
    <label>Map<select aria-label={`${kind} map`} value={mapName||""} disabled={disabled||!maps.length}
      onChange={event=>onMapChange(event.target.value)}>
      {!maps.length&&<option value="">No saved dataset map</option>}
      {maps.map(map=><option value={map} key={map}>{map}</option>)}
    </select></label>
    {onTargetChange&&<label>Target<select aria-label="Auto-label target" value={target} disabled={disabled}
      onChange={event=>onTargetChange(event.target.value)}>
      <option value="episode" disabled={!hasSelectedEpisode}>Selected episode</option>
      <option value="map">All compatible episodes on this map</option>
    </select></label>}
  </div>;
}
