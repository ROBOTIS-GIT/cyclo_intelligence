import { FiCrosshair, FiMapPin, FiSave, FiX } from "react-icons/fi";
import { RecordToolButton } from "./RecordAuthoringControls";

export default function RecordWaypointEditor({ manager, disabled = false }) {
  const { draft } = manager;
  if (!draft) return null;
  return <fieldset className="record-waypoint-editor" disabled={disabled || manager.working || manager.blocked} aria-label="Observation waypoint editor">
    <label>Observation name<input aria-label="Observation name" maxLength={128} value={draft.label}
      onChange={event => manager.setDraft({ ...draft, label: event.target.value })} /></label>
    <div className="record-authoring-heading">
      <div className="record-authoring-actions">
        <RecordToolButton label="Observation on map" onClick={manager.onMap}><FiMapPin /></RecordToolButton>
        <RecordToolButton label="Observation at robot" disabled={!manager.canUseRobot} onClick={manager.atRobot}><FiCrosshair /></RecordToolButton>
      </div>
      <label className="record-waypoint-yaw">Yaw °<input type="number" aria-label="Observation yaw" step="1" disabled={!draft.pose}
        value={Number.isFinite(draft.pose?.yaw) ? Math.round(draft.pose.yaw * 180 / Math.PI) : ""}
        onChange={event => manager.setDraft({ ...draft, pose: { ...draft.pose, yaw: event.target.value === "" ? null : Number(event.target.value) * Math.PI / 180 } })} /></label>
    </div>
    <div className="record-authoring-heading">
      <span className="record-authoring-note">{draft.pose ? "Position set" : "Choose a position"}</span>
      <div className="record-authoring-actions">
        <RecordToolButton label="Save observation spot" disabled={!draft.label.trim() || !draft.pose || !Number.isFinite(draft.pose.yaw)} onClick={manager.save}><FiSave /></RecordToolButton>
        <RecordToolButton label="Cancel observation edit" onClick={manager.cancel}><FiX /></RecordToolButton>
      </div>
    </div>
  </fieldset>;
}
