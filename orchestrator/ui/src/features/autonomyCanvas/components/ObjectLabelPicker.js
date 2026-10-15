import { useEffect, useId, useState } from "react";
import { collectionRequest } from "../../../utils/collectionApi";

export default function ObjectLabelPicker({ episodeId, regions, value, onChange, disabled, onNewInstance, compact = false }) {
  const listId = useId();
  const [saved, setSaved] = useState([]);
  const [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    setSaved([]); setError("");
    collectionRequest(`episodes/${episodeId}/segmentation/labels`)
      .then(result => { if (alive) setSaved(result.labels || []); })
      .catch(err => { if (alive) setError(err.message); });
    return () => { alive = false; };
  }, [episodeId]);
  const labels = [...new Set([...saved, ...regions.filter(r=>r.review_status === "approved").map(r=>r.object_label)])].sort();
  if (compact) return <div className="sam-object-label-input">
    <input aria-label="Object name" title={error || "Object name"} list={listId} maxLength={120}
      value={value} disabled={disabled} onChange={e=>onChange(e.target.value)} placeholder="Object name" />
    <datalist id={listId}>{labels.map(label=><option key={label} value={label}/>)}</datalist>
  </div>;
  return <div className="sam-object-label-picker">
    <div className="sam2-toolbar">
      <label className="collect-field">Object name<input maxLength={120} value={value} disabled={disabled}
        onChange={e=>onChange(e.target.value)} placeholder="e.g. wooden_stick" /></label>
      {onNewInstance && <button type="button" disabled={disabled} onClick={onNewInstance}>New instance</button>}
    </div>
    {!!labels.length && <div className="sam2-toolbar" role="group" aria-label="Object labels">
      {labels.map(label=><button type="button" key={label} aria-pressed={value === label}
        disabled={disabled} onClick={()=>onChange(label)}>{label}</button>)}
    </div>}
    {error && <p role="alert">{error}</p>}
  </div>;
}
