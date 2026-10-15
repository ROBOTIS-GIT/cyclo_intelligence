import {useEffect, useState} from "react";

// Saved sampling metadata never limits browsing until the operator selects it.
export default function useLabelingFrameScope({episodeId, sampleIds=[], frames, isReviewed, generatedIds=[]}) {
  const [selection, setSelection] = useState(null);
  useEffect(()=>{
    setSelection(value=>value?.scope === "review" && value.episodeId === null ? value : null);
  },[episodeId]);
  const current = selection && (selection.episodeId === null || selection.episodeId === episodeId);
  const frameScope = current && selection.scope === "review" ? "review"
    : current && selection.scope === "samples" && sampleIds.length ? "samples" : "all";
  const review = frameScope === "review", samples = frameScope === "samples";
  const setFrameScope = (scope, targetEpisodeId=episodeId) => setSelection({episodeId:targetEpisodeId, scope});
  const ids = new Set(review ? generatedIds : sampleIds);
  const generated = new Set(generatedIds);
  const pendingReviewCount = frames.filter(frame => generated.has(frame.frame_id) && !isReviewed(frame.frame_id)).length;
  const progressFrames = review || samples ? frames.filter(frame => ids.has(frame.frame_id)) : frames;
  const completed = progressFrames.filter(frame => isReviewed(frame.frame_id)).length;
  const visibleFrames = review ? progressFrames.filter(frame => !isReviewed(frame.frame_id)) : progressFrames;
  return {frameScope, setFrameScope, visibleFrames, pendingReviewCount,
    progress: `${completed}/${progressFrames.length} ${review ? "reviewed" : "labeled"}`};
}
