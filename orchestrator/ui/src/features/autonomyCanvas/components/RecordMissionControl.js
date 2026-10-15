import { MdPlayArrow } from "react-icons/md";

export default function RecordMissionControl({ missionNames = [], selectedMission,
  loading, disabled, canStartMission, onMissionChange, onRunMission }) {
  if (!missionNames.length) return null;
  return <section className="record-mission-control">

    <label className="flex flex-col gap-1 text-xs font-semibold">
      Mission
      <div className="flex items-center gap-1">
        <select aria-label="Collect map mission" value={selectedMission || ""} disabled={disabled || loading}
          onChange={event => onMissionChange(event.target.value)} className="w-full min-w-0 h-8 px-2 text-[13px] font-medium"
          style={{borderRadius:8,fontFamily:"inherit",color:"var(--mc-text)",background:"var(--mc-surface-2)",border:"1px solid var(--mc-border-strong)"}}>
          <option value="">{loading ? "Loading…" : "Select a mission"}</option>
          {missionNames.map(name => <option key={name} value={name}>{name}</option>)}
        </select>
        <button type="button" aria-label="Run child mission" title="Run child mission" disabled={disabled || loading || !canStartMission || !selectedMission}
          onClick={onRunMission} className="h-8 w-8 shrink-0 inline-flex items-center justify-center disabled:opacity-45"
          style={{borderRadius:9,background:"var(--mc-success)",color:"var(--mc-accent-fg)"}}><MdPlayArrow size={19}/></button>
      </div>
    </label>
  </section>;
}
