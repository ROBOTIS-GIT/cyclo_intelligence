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

import {MdFactCheck, MdFilterList} from "react-icons/md";
import "./LabelingWorkflowBar.css";

// Model-job controls stay mounted while the editor switches between all frames, samples and review.
export default function LabelingWorkflowBar({sampling, actions, feedback,
  frameScope, onFrameScopeChange, sampleCount=0, disabled=false, pendingReviewCount=0}) {
  const review=frameScope==="review";
  return <div className="labeling-workflow">
    <div className="labeling-workflow-bar" role="group" aria-label="Labeling tools">
      <fieldset className="labeling-inline-sampling" disabled={disabled || review} aria-label="Sample labeling">
        {sampling}
      </fieldset>
      <button type="button" className="labeling-sample-filter" aria-label={frameScope === "samples" ? "Show all frames" : "Show sampled frames"}
        aria-pressed={frameScope === "samples"} title={frameScope === "samples" ? "Show all frames" : "Show sampled frames"}
        disabled={disabled || review || !sampleCount}
        onClick={()=>onFrameScopeChange(frameScope === "samples" ? "all" : "samples")}>
        <MdFilterList size={19}/>
      </button>
      <span className="labeling-tool-divider" aria-hidden="true"/>
      {actions}
      <span className="labeling-tool-divider" aria-hidden="true"/>
      <button type="button" className="labeling-review-button" aria-label="Review" aria-pressed={review}
        disabled={disabled || (!review && !pendingReviewCount)}
        title={review ? "Exit map review" : pendingReviewCount ? `Review ${pendingReviewCount} generated frames on this map` : "No generated labels to review"}
        onClick={()=>onFrameScopeChange(review ? "all" : "review")}>
        <MdFactCheck size={19}/>
        <span className="labeling-review-count" aria-label="Pending reviews">{pendingReviewCount}</span>
      </button>
    </div>
    {feedback}
  </div>;
}
