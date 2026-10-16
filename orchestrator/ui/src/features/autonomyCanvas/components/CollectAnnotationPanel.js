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

import { useEffect, useState } from "react";
import LabelingWorkflowBar from "./LabelingWorkflowBar";
import useLabelingFrameScope from "./useLabelingFrameScope";
import useGeneratedReview from "./useGeneratedReview";
import useLabelingJobRefresh from "../hooks/useLabelingJobRefresh";
import SamWorkflowControls from "./SamWorkflowControls";
import Sam2Annotation from "./Sam2Annotation";
import AnnotationFrameStage from "./AnnotationFrameStage";
import AnnotationFrameNavigation from "./AnnotationFrameNavigation";
import { currentSampleSelection } from "./sampleSelection";

export default function CollectAnnotationPanel({ draft, editor, maskDirty = false,
  setMaskDirty, onBusyChange, onRunningChange, selectedModelId }) {
  const { episode, active, selectedFrame } = draft;
  const { version, onJobChange } = useLabelingJobRefresh(draft, maskDirty);
  const [workflowBusy, setWorkflowBusy] = useState(false);
  const [workflowRunning, setWorkflowRunning] = useState(false);
  const [maskMissionId, setMaskMissionId] = useState("");
  useEffect(() => { onBusyChange?.(workflowBusy); }, [workflowBusy, onBusyChange]);
  useEffect(() => { onRunningChange?.(workflowRunning); }, [workflowRunning, onRunningChange]);
  const { maskEditor, setMaskBusy, setLoadedFrame, busy, observations, reviewBusy: editorReviewBusy } = editor;
  const workflowLocked = editorReviewBusy || maskDirty || workflowBusy || workflowRunning;
  const labeled = id => {
    const regions = (episode?.regions || []).filter(region => region.frame_id === id);
    return regions.length ? regions.every(region => ["approved", "rejected"].includes(region.review_status))
      : observations.find(frame => frame.frame_id === id)?.mask_review?.status === "empty";
  };
  const generatedIds = observations.filter(frame => frame.auto_labeling?.sam?.status === "generated" ||
    (episode?.regions || []).some(region => region.frame_id === frame.frame_id && region.annotator === "sam-auto"))
    .map(frame => frame.frame_id);
  const sampleIds = currentSampleSelection(episode?.sam_sample)?.frame_ids || [];
  const scope = useLabelingFrameScope({ episodeId: episode?.id, sampleIds, frames: observations, isReviewed: labeled, generatedIds });
  const review = useGeneratedReview({ draft, kind: "sam", scope, disabled: workflowLocked || busy || editor.saving });
  const reviewBusy = workflowLocked || review.busy;
  const frames = scope.visibleFrames;
  useEffect(() => {
    if (active || reviewBusy) return;
    if ((frames.length && !frames.some(frame => frame.frame_id === selectedFrame?.frame_id)) ||
        (!frames.length && selectedFrame)) draft.selectFrame(frames[0] || null);
  }, [active, selectedFrame, reviewBusy, frames, draft]);
  const index = frames.findIndex(frame => frame.frame_id === selectedFrame?.frame_id);
  const controls = <AnnotationFrameNavigation draft={draft} frames={frames} index={index}
    disabled={reviewBusy} deletionBlocked={busy || editor.saving || editor.frameDeletionBlocked}
    progress={scope.progress && (review.progress || scope.progress)}/>;
  function renderMaskLayout({ imageEditor, objectControls, instanceControls }) {
    return <AnnotationFrameStage compact episode={episode} frame={selectedFrame} controls={controls} tools={objectControls} footer={instanceControls}>
      {imageEditor}
    </AnnotationFrameStage>;
  }
  const empty = <section className="sam2-annotation" aria-label="SAM2 segmentation">
    <div className="sam2-toolbar"><strong>Object mask</strong></div>
    <div className="sam2-empty" role="status">
      <strong>{episode && !frames.length ? (scope.frameScope === "review" ? "All generated labels reviewed" : "No frames match this filter") : "No saved frame selected"}</strong>
      <p>{!episode ? "Select an episode from the list to begin labeling."
        : scope.frameScope === "review" && !frames.length ? "Turn off Review to return to sample labeling."
        : observations.length ? "Select a saved frame to begin click segmentation."
        : active ? "Capture frames in Data Collection, then save the episode to begin labeling."
        : "This episode has no saved frames. Start a new episode and capture a frame."}</p>
    </div>
  </section>;
  return <div className="collect-dataset-editor collect-mask-panel">
    <div className="annotate-mask-workspace">
      <SamWorkflowControls episode={episode} episodes={draft.episodes} selectedModelId={selectedModelId}
        disabled={editorReviewBusy || maskDirty} onBusyChange={setWorkflowBusy} onRunningChange={setWorkflowRunning}
        onJobChange={onJobChange}
        onSampled={async result => {
          await draft.refresh?.(); await draft.refreshEpisodes?.();
          scope.setFrameScope("samples");
          draft.selectFrame(result.observations.find(frame => result.sam_sample.frame_ids.includes(frame.frame_id)) || null);
        }}
        renderLayout={slots => <LabelingWorkflowBar {...slots} frameScope={scope.frameScope} onFrameScopeChange={review.changeScope}
          sampleCount={sampleIds.length} pendingReviewCount={review.pendingReviewCount} disabled={reviewBusy}
          feedback={<>{slots.feedback}{review.error && <p role="alert">{review.error}</p>}</>}/>}/>
      {!active && selectedFrame ? <Sam2Annotation ref={maskEditor} disabled={busy || workflowBusy || workflowRunning}
        key={`${episode.id}/${selectedFrame.frame_id}/${version}`} episode={episode} frame={selectedFrame}
        missionId={maskMissionId} onMissionChange={setMaskMissionId} onBusyChange={setMaskBusy} onFrameLoaded={setLoadedFrame}
        onMaskSaved={async ({ advance = false } = {}) => {
          await draft.refresh?.(); await draft.refreshEpisodes?.();
          if (advance && index >= 0 && frames[index + 1]) draft.selectFrame(frames[index + 1]);
        }} standaloneSave onDirtyChange={setMaskDirty} renderLayout={renderMaskLayout}
        reviewMode={scope.frameScope === "review" || (episode?.regions || []).some(region => region.frame_id === selectedFrame.frame_id && region.review_status === "draft")}/>
        : renderMaskLayout({ imageEditor: empty })}
    </div>
  </div>;
}
