import React, { useEffect, useState } from 'react';
import { getPgmFiles } from '../../../utils/navigationApi';
import { getNavigationMission, getNavigationMissions, subscribeNavigationMissionChanges } from '../../../utils/navigationMissionsApi';
import { mapNameFromPgmPath } from '../../mapWorkspace/lib/missionNames';

// Each request belongs to the current selection; late responses cannot replace it.
export default function ReturnWaypointFields({ params, onChange }) {
  const { map_name: map = '', mission_name: mission = '', waypoint = '' } = params;
  const [maps, setMaps] = useState([]);
  const [missions, setMissions] = useState([]);
  const [waypoints, setWaypoints] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => subscribeNavigationMissionChanges(() => setRevision(n => n + 1)), []);
  useEffect(() => {
    let current = true;
    setError(''); setLoading(true); setMissions([]); setWaypoints([]);
    async function load() {
      const inventory = await getPgmFiles();
      if (!current) return;
      setMaps([...new Set((inventory.files || []).map(f => mapNameFromPgmPath(f.path)))]);
      if (!map) return;
      const list = await getNavigationMissions(map);
      if (!current) return;
      setMissions(list.missions || []);
      if (!mission || !(list.missions || []).includes(mission)) return;
      const document = await getNavigationMission(map, mission);
      if (current) setWaypoints(document.waypoints || []);
    }
    load().catch(e => { if (current) setError(e.message || 'Could not load destinations'); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [map, mission, revision]);
  const change = (key, value) => {
    onChange(key, value);
    if (key === 'map_name') onChange('mission_name', '');
    if (key !== 'waypoint') onChange('waypoint', '');
  };
  const fields = [
    ['map_name', map, maps.map(v => [v, v]), false, 'Select a map'],
    ['mission_name', mission, missions.map(v => [v, v]), !map || loading, 'Select a mission'],
    ['waypoint', waypoint, waypoints.map(w => [w.id, w.label || w.id]), !mission || loading, 'Select a waypoint'],
  ];
  return <>
    {fields.map(([key, value, options, disabled, placeholder]) => {
      // Preserve existing XML names until the user explicitly chooses an ID.
      const known = options.some(([id]) => id === value);
      return <div key={key}>
        <label htmlFor={'return-' + key} className="block text-xs font-medium mb-1 font-mono text-[var(--mc-text-muted)]">{key}</label>
        <select id={'return-' + key} value={value} disabled={disabled}
          onChange={e => change(key, e.target.value)}
          className="w-full px-2 py-1.5 border border-[var(--mc-border-strong)] rounded-lg text-sm bg-[var(--mc-surface)] text-[var(--mc-text)] disabled:opacity-50">
          <option value="">{placeholder}</option>
          {value && !known && <option value={value}>{value}</option>}
          {options.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
        </select>
      </div>;
    })}
    {error && <p role="alert" className="text-xs text-red-500">{error}</p>}
  </>;
}
