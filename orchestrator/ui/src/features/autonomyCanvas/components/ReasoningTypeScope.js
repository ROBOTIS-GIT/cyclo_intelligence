import { EPISODE_TYPES, TYPE_LABELS } from "../utils/reasoningDataset";

export default function ReasoningTypeScope({value,onChange,counts={},disabled}) {
  return <fieldset className="reasoning-type-scope" disabled={disabled}>
    <legend>Training types</legend>
    {EPISODE_TYPES.map(type=><label key={type}>
      <input type="checkbox" checked={value.includes(type)} onChange={event=>onChange(previous=>event.target.checked
        ? EPISODE_TYPES.filter(item=>item===type||previous.includes(item)) : previous.filter(item=>item!==type))}/>
      <span>{TYPE_LABELS[type]}</span><small>{counts[type]||0} labels</small>
    </label>)}
  </fieldset>;
}
