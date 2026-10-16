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
