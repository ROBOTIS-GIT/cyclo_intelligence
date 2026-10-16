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
import { MdRestartAlt } from "react-icons/md";
import LabelingWorkflowBar from "./LabelingWorkflowBar";
import useLabelingFrameScope from "./useLabelingFrameScope";
import useGeneratedReview from "./useGeneratedReview";
import useLabelingJobRefresh from "../hooks/useLabelingJobRefresh";
import AnnotationFrameStage from "./AnnotationFrameStage";
import AnnotationFrameNavigation from "./AnnotationFrameNavigation";
import CollectReasoningFields from "./CollectReasoningFields";
import QwenWorkflowControls from "./QwenWorkflowControls";
import { frameMatchesEpisode } from "../utils/reasoningDataset";
import { currentSampleSelection } from "./sampleSelection";

export default function CollectReasoningPanel({ draft, editor, onBusyChange, onRunningChange, selectedModelId }) {
  const [workflowBusy, setWorkflowBusy] = useState(false);
  const [workflowRunning, setWorkflowRunning] = useState(false);
  const { episode, selectedFrame, active } = draft;
  const observations = episode?.observations || [];
  const labels = episode?.annotations || [];
  const sampleIds = currentSampleSelection(episode?.qwen_sample)?.frame_ids || [];
  const approved = id => labels.some(label => label.frame_ids[0] === id && label.review_status === "approved");
  const validFrames = observations.filter(frame => frameMatchesEpisode(episode, frame));
  const generatedIds = validFrames.filter(frame => frame.auto_labeling?.qwen?.status === "generated" ||
    labels.some(label => label.frame_ids[0] === frame.frame_id && label.annotator?.startsWith("qwen:")))
    .map(frame => frame.frame_id);
  const scope = useLabelingFrameScope({ episodeId: episode?.id, sampleIds, frames: validFrames, isReviewed: approved, generatedIds });
  const blocked = editor.busy || editor.saving || editor.maskBusy || editor.labelDirty || active || workflowBusy || workflowRunning;
  const review = useGeneratedReview({ draft, kind: "qwen", scope, disabled: blocked });
  const frames = scope.visibleFrames;
  const index = frames.findIndex(frame => frame.frame_id === selectedFrame?.frame_id);
  const locked = blocked || review.busy;
  useEffect(() => {
    if (!locked && ((frames.length && !frames.some(frame => frame.frame_id === selectedFrame?.frame_id)) ||
        (!frames.length && selectedFrame))) draft.selectFrame(frames[0] || null);
  }, [selectedFrame, locked, frames, draft]);
  useEffect(() => { onBusyChange?.(workflowBusy); }, [workflowBusy, onBusyChange]);
  useEffect(() => { onRunningChange?.(workflowRunning); }, [workflowRunning, onRunningChange]);
  const { onJobChange } = useLabelingJobRefresh(draft, editor.labelDirty);
  async function save() {
    if (await editor.save("approved")) {
      const next = frames[index + 1];
      if (next) draft.selectFrame(next);
    }
  }
  const reset = <button type="button" className="reasoning-reset-action" aria-label="Discard changes" title="Reset label"
    disabled={!editor.labelDirty || editor.busy || workflowBusy} onClick={editor.resetLabel}><MdRestartAlt size={18}/></button>;
  const fields = <CollectReasoningFields compact draft={draft}
    editor={{ ...editor, busy: editor.busy || workflowBusy || workflowRunning }} onSave={save} extraActions={reset}/>;
  const controls = <>
    <AnnotationFrameNavigation draft={draft} frames={frames} index={index} disabled={locked}
      deletionBlocked={editor.frameDeletionBlocked} progress={scope.progress && (review.progress || scope.progress)}/>
    {validFrames.length < observations.length && <p className="operate-sidebar-note">
      {observations.length - validFrames.length} frames without matching execution context are excluded from reasoning labels.
    </p>}
  </>;
  return <div className="collect-dataset-editor collect-reasoning-panel">
    <QwenWorkflowControls onJobChange={onJobChange} selectedModelId={selectedModelId} episode={episode} episodes={draft.episodes}
      disabled={editor.busy || editor.saving || editor.maskBusy || editor.labelDirty || active}
      onBusyChange={setWorkflowBusy} onRunningChange={setWorkflowRunning}
      onSampled={async result => {
        await draft.refresh(); await draft.refreshEpisodes?.();
        scope.setFrameScope("samples");
        draft.selectFrame(result.observations.find(frame => result.qwen_sample.frame_ids.includes(frame.frame_id)) || null);
      }} renderLayout={slots => <div className="annotate-mask-workspace annotate-reasoning-workspace">
        <LabelingWorkflowBar {...slots} frameScope={scope.frameScope} onFrameScopeChange={review.changeScope}
          sampleCount={sampleIds.length} pendingReviewCount={review.pendingReviewCount} disabled={locked}
          feedback={<>{slots.feedback}{review.error && <p role="alert">{review.error}</p>}</>}/>
        <AnnotationFrameStage compact episode={episode} frame={selectedFrame} controls={controls}
          tools={<section className="reasoning-label-tools" aria-label="Frame labeling">{fields}</section>}>
          {selectedFrame && !active ? <img className="collect-reasoning-image" key={`${episode.id}/${selectedFrame.frame_id}`}
            alt="Saved frame for reasoning annotation" src={`/api/collection/episodes/${episode.id}/images/${selectedFrame.frame_id}`}
            onLoad={() => editor.setLoadedFrame(selectedFrame.frame_id)} onError={() => editor.setLoadedFrame(null)}/>
            : <div className="sam2-empty">{episode && !frames.length
              ? (scope.frameScope === "review" ? "All generated labels reviewed." : "No frames match this filter.")
              : "Select a saved episode and frame to label."}</div>}
        </AnnotationFrameStage>
      </div>}/>
  </div>;
}
