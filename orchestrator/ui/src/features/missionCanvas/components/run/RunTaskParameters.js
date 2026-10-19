import React, { useEffect, useMemo, useState } from 'react';
import { MdTune, MdClose, MdSave } from 'react-icons/md';
import { useBTNodeCatalog } from '../../../../hooks/useBTNodeCatalog';
import { getNavigationMission, getNavigationMissionBtFile, saveNavigationMissionBtFile } from '../../../../utils/navigationMissionsApi';
import { localBtPathForSpot } from '../../../mapWorkspace/lib/missionBtFiles';
import BtParameterPanel from '../../../btEditor/parameters/BtParameterPanel';
import { parameterNodes, updateParameter } from './localBtParameters';

function Editor({ mapName, missionName, disabled, onClose, onSaved }) {
  const { catalog = [] } = useBTNodeCatalog();
  const [mission, setMission] = useState(null);
  const [spotId, setSpotId] = useState('');
  const [xml, setXml] = useState('');
  const [revision, setRevision] = useState(null);
  const [selected, setSelected] = useState('');
  const [busy, setBusy] = useState(true);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    let active = true;
    getNavigationMission(mapName, missionName).then(m => {
      if (!active) return;
      setMission(m); setSpotId(m.waypoints?.[0]?.id || '');
      if (!m.waypoints?.length) setBusy(false);
    }).catch(e => { if (active) { setError(e.message); setBusy(false); } });
    return () => { active = false; };
  }, [mapName, missionName]);
  const spot = mission?.waypoints?.find(w => w.id === spotId);
  const path = spot ? localBtPathForSpot(spot) : '';
  useEffect(() => {
    if (!path) return undefined;
    let active = true;
    setBusy(true); setError(''); setXml(''); setSelected(''); setDirty(false); setSaved(false);
    getNavigationMissionBtFile(mapName, path, missionName).then(file => {
      if (!active) return;
      if (!file.exists) throw new Error('No saved waypoint task');
      parameterNodes(file.content); // validate before enabling editing
      setXml(file.content); setRevision(file.revision);
    }).catch(e => { if (active) setError(e.message); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [mapName, missionName, path]);
  const nodes = useMemo(() => xml ? parameterNodes(xml, catalog) : [], [xml, catalog]);
  const close = () => { if (!busy && (!dirty || window.confirm('Discard unsaved parameter changes?'))) onClose(); };
  const save = async () => {
    if (disabled || busy || !dirty) return;
    setBusy(true); setError('');
    try {
      const result = await saveNavigationMissionBtFile(mapName, path, xml, missionName, {waypointId: spotId, expectedRevision: revision});
      setRevision(result.revision); setDirty(false);
      await onSaved();
      setSaved(true);
    } catch (e) { setError(e.message || 'Could not save task parameters'); }
    finally { setBusy(false); }
  };
  return <div className="fixed inset-0 z-[100] bg-black/40 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Waypoint task parameters">
    <div className="w-[680px] max-w-[95vw] h-[70vh] flex flex-col rounded-xl border shadow-xl" style={{background:'var(--mc-surface-2)', borderColor:'var(--mc-border)', color:'var(--mc-text)'}}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--mc-border)]">
        <span className="text-sm font-semibold">Waypoint task parameters</span>
        <button title="Close" aria-label="Close" disabled={busy} onClick={close}><MdClose size={20}/></button>
      </div>
      <div className="relative flex-1 min-h-0">
        <div className="absolute inset-y-0 left-0 right-[320px] p-3 overflow-y-auto">
          <label className="text-xs block mb-1" htmlFor="run-parameter-waypoint">Waypoint</label>
          <select id="run-parameter-waypoint" className="w-full rounded-lg border p-2 text-sm bg-[var(--mc-surface)] border-[var(--mc-border)]" value={spotId} disabled={busy || disabled}
            onChange={e => { if (!dirty || window.confirm('Discard unsaved parameter changes?')) setSpotId(e.target.value); }}>
            {(mission?.waypoints || []).map(w => <option key={w.id} value={w.id}>{w.label || w.id}</option>)}
          </select>
          <div className="grid gap-1 mt-3">{nodes.map(n => <button key={n.id} disabled={busy || disabled} onClick={() => setSelected(n.id)}
            className="text-left rounded-lg px-2 py-2 text-xs border" style={{borderColor: selected === n.id ? 'var(--mc-accent)' : 'var(--mc-border)', background:'var(--mc-surface)'}}>{n.data.label}<span className="block text-[var(--mc-text-subtle)]">{n.data.nodeType}</span></button>)}</div>
        </div>
        <fieldset disabled={busy || disabled} className="contents">
          <BtParameterPanel nodes={nodes} selectedNodeId={selected} onClose={() => setSelected('')} allowNameEdit={false}
            onParamChange={(id, key, value) => { if (!busy && !disabled) { setXml(prev => updateParameter(prev, id, key, value)); setDirty(true); setSaved(false); } }} />
        </fieldset>
      </div>
      <div className="px-4 py-3 border-t border-[var(--mc-border)] flex items-center gap-3">
        <span className="text-xs flex-1" role="status">{error || (disabled ? 'Stop the mission before editing.' : busy ? 'Loading…' : saved ? 'Saved. Ready for the next run.' : 'Select a task node to edit its parameters.')}</span>
        <button onClick={save} disabled={busy || disabled || !dirty} className="flex items-center gap-1 px-3 py-2 rounded-lg text-xs border border-[var(--mc-border)] disabled:opacity-40"><MdSave/>Save</button>
      </div>
    </div>
  </div>;
}

export default function RunTaskParameters({ mapName, missionName, disabled, onSaved }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [mapName, missionName]);
  return <>
    <button type="button" title="Edit waypoint task parameters" aria-label="Edit waypoint task parameters" disabled={disabled || !mapName} onClick={() => setOpen(true)}
      className="p-1.5 rounded-lg border border-[var(--mc-border)] hover:bg-[var(--mc-surface-hover)] disabled:opacity-40"><MdTune size={18}/></button>
    {open && <Editor mapName={mapName} missionName={missionName} disabled={disabled} onClose={() => setOpen(false)} onSaved={onSaved}/>}
  </>;
}
