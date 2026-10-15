import {frameMatchesEpisode} from "../utils/reasoningDataset";
import {currentSampleSelection} from "./sampleSelection";

export function approvedQwenSamples(episode) {
  if(typeof episode.qwen_sample_approved_count==="number")return episode.qwen_sample_approved_count;
  const sampled=new Set(currentSampleSelection(episode.qwen_sample)?.frame_ids||[]);
  const valid=new Set((episode.observations||[]).filter(frame=>frameMatchesEpisode(episode,frame)).map(frame=>frame.frame_id));
  return new Set((episode.annotations||[]).filter(label=>label.review_status==="approved"
    && sampled.has(label.frame_ids?.[0]) && valid.has(label.frame_ids?.[0])).map(label=>label.frame_ids[0])).size;
}

export function approvedSamSamples(episode) {
  if(typeof episode.sam_sample_trainable_instance_count==="number")return episode.sam_sample_trainable_instance_count;
  if(typeof episode.sam_sample_approved_instance_count==="number")return episode.sam_sample_approved_instance_count;
  const sampled=new Set(currentSampleSelection(episode.sam_sample)?.frame_ids||[]);
  const pending=new Set((episode.regions||[]).filter(region=>region.review_status==="draft").map(region=>region.frame_id));
  return (episode.regions||[]).filter(region=>sampled.has(region.frame_id)
    && region.review_status==="approved" && !pending.has(region.frame_id)).length;
}
