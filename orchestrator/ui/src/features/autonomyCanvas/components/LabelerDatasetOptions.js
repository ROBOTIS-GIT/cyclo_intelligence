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
