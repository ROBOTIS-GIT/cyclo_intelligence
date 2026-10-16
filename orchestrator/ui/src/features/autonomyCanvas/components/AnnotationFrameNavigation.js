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

import FrameRangeDeleteButton from "./FrameRangeDeleteButton";

export default function AnnotationFrameNavigation({ draft, frames, index, disabled, deletionBlocked, progress }) {
  return <div className="collect-actions collect-frame-navigation">
    <button type="button" aria-label="Previous frame" disabled={disabled || index <= 0}
      onClick={() => draft.selectFrame(frames[index - 1])}>←</button>
    <span className="labeling-frame-position" aria-label="Current frame">{index >= 0 ? index + 1 : "–"} / {frames.length}</span>
    <button type="button" aria-label="Next frame" disabled={disabled || !frames.length || index >= frames.length - 1}
      onClick={() => draft.selectFrame(frames[index + 1])}>→</button>
    {progress && <span className="labeling-frame-progress" role="status" aria-label="Labeling progress">{progress}</span>}
    <FrameRangeDeleteButton episode={draft.episode} disabled={disabled || deletionBlocked || !draft.deleteFrames} onDelete={draft.deleteFrames}/>
  </div>;
}
