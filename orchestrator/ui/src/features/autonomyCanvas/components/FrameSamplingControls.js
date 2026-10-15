import {MdShuffle} from "react-icons/md";

// Shared sampling controls for SAM and Qwen labeling.
export default function FrameSamplingControls({count, onCountChange, max, disabled, canSample, sampleCount, onSample}) {
  const label=sampleCount ? "Resample frames" : "Sample frames";
  return <div className="frame-sampling-controls" aria-label="Training frame sampling">
    <label>Samples<input aria-label="Sample count" type="number" min="1" max={max || 1} step="1" value={count}
      title={max > 0 ? `1–${max} frames` : "No frames"} placeholder="Count"
      disabled={disabled || !max} onChange={event=>onCountChange(event.target.value)}/></label>
    <button type="button" aria-label={label} title={label} disabled={disabled || !canSample} onClick={onSample}><MdShuffle size={19}/></button>
  </div>;
}
