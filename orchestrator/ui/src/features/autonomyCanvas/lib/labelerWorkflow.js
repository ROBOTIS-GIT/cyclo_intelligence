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

import { approvedQwenSamples, approvedSamSamples } from "../components/labelerSampleCounts";
import { currentSampleSelection } from "../components/sampleSelection";
import { episodeCategory } from "../components/episodeCategory";
import { frameMatchesEpisode } from "../utils/reasoningDataset";

export function modelSupportsEpisode(model, episode) {
  const types = model?.episode_types?.length ? model.episode_types : model?.episode_type ? [model.episode_type] : [];
  return !types.length || types.includes(episodeCategory(episode));
}

export const LABELERS = {
  qwen: {
    title: "Qwen workflow", sampleKey: "qwen_sample", countLabel: "frames",
    isModel: model => model.purpose !== "reasoning" && (!model.status || model.status === "completed"),
    supports: modelSupportsEpisode,
    trainedModel: (model, job) => model.model_name === job.model_name && model.map_name === job.map_name,
    frames: episode => (episode?.observations || []).filter(frame => frameMatchesEpisode(episode, frame)),
    approvedSamples: approvedQwenSamples,
    currentEpisode: episode => ({ ...episode, qwen_sample_approved_count: undefined }),
  },
  segmentation: {
    title: "SAM workflow", sampleKey: "sam_sample", countLabel: "instances",
    isModel: model => model.strategy === "sam_multiclass_instances_v1",
    supports: () => true,
    trainedModel: (model, job) => model.id === job.id,
    frames: episode => episode?.observations || [],
    approvedSamples: approvedSamSamples,
    currentEpisode: episode => ({ ...episode, sam_sample_trainable_instance_count: undefined, sam_sample_approved_instance_count: undefined }),
  },
};

export function labelerSampleSummary(config, episode) {
  const frames = config.frames(episode);
  const sampled = currentSampleSelection(episode?.[config.sampleKey])?.frame_ids || [];
  return { max: frames.length || (config === LABELERS.segmentation && !episode?.observations ? episode?.frame_count || 0 : 0), sampled: sampled.length };
}
