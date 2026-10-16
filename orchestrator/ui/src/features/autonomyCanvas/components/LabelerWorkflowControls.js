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
import LabelerActions from "./LabelerActions";
import LabelerFeedback from "./LabelerFeedback";
import FrameSamplingControls from "./FrameSamplingControls";
import useLabelerDataset from "./useLabelerDataset";
import useLabelerWorkflow from "../hooks/useLabelerWorkflow";
import { LABELERS, labelerSampleSummary } from "../lib/labelerWorkflow";
import { jobRunning } from "./trainingJobStatus";

// SAM and Qwen expose the same compact workflow; only data eligibility and API family vary.
export default function LabelerWorkflowControls({ family, episode, episodes, disabled,
  onSampled, onJobChange, onBusyChange, onRunningChange, selectedModelId, renderLayout }) {
  const config = LABELERS[family];
  const workflow = useLabelerWorkflow({ family, onSampled, onJobChange, onBusyChange, onRunningChange });
  const { state, connectionError, working, running, pendingTraining, run, cancelTraining, cancelling, deleteModel } = workflow;
  const [count, setCount] = useState("");
  const [modelId, setModelId] = useState("");
  const [scope, setScope] = useState("episode");
  useEffect(() => { setCount(""); setModelId(selectedModelId || ""); }, [episode?.id, selectedModelId]);
  const dataset = useLabelerDataset({ episode, episodes,
    models: (state?.models || []).filter(config.isModel) });
  const { maps, mapName, setMap, mapEpisodes, selectedEpisode } = dataset;
  const generationTarget = selectedEpisode ? scope : "map";
  const models = (state?.models || []).filter(model => config.isModel(model) && model.map_name === mapName &&
    (generationTarget === "map" || !selectedEpisode || config.supports(model, selectedEpisode)))
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const selected = models.find(model => model.id === modelId) || models[0];
  useEffect(() => {
    const job = state?.training;
    if (jobRunning(job)) pendingTraining.current = job.id;
    if (!pendingTraining.current || job?.id !== pendingTraining.current || job.status !== "completed") return;
    const trained = (state.models || []).find(model => config.isModel(model) && config.trainedModel(model, job) &&
      model.map_name === mapName && (!selectedEpisode || config.supports(model, selectedEpisode)));
    if (trained) { setModelId(trained.id); pendingTraining.current = null; }
  }, [state, config, mapName, selectedEpisode, pendingTraining]);

  const sample = labelerSampleSummary(config, episode);
  const requestedCount = Number(count);
  const validCount = count.trim() !== "" && Number.isInteger(requestedCount) && requestedCount >= 1 && requestedCount <= sample.max;
  const approvedCount = mapEpisodes.reduce((sum, item) => sum + config.approvedSamples(
    item.id === episode?.id ? config.currentEpisode(item) : item), 0);
  const targets = generationTarget === "map" ? mapEpisodes : selectedEpisode ? [selectedEpisode] : [];
  const generationEpisodes = targets.filter(item => config.supports(selected, item));
  const locked = disabled || working || running;
  const sampling = <FrameSamplingControls count={count} onCountChange={setCount} max={sample.max}
    disabled={locked} canSample={!!episode?.ended_at && validCount} sampleCount={sample.sampled}
    onSample={() => run(`episodes/${episode.id}/${family}/sample`, { count: requestedCount })}/>;
  const actions = <LabelerActions models={models} selected={selected} onSelect={setModelId}
    maps={maps} mapName={mapName} onMapChange={setMap} generationTarget={generationTarget} onTargetChange={setScope}
    trainingSummary={`${approvedCount} approved sampled ${config.countLabel} on this map`}
    trainingHint={locked ? "Finish the current edit or job before starting fine-tuning." : approvedCount < 2 ? `At least 2 approved sampled ${config.countLabel} are required.` : ""}
    trainingJob={state?.training} batchJob={state?.batch} onCancelTrain={cancelTraining} cancelling={cancelling}
    disabled={locked} episodeId={selectedEpisode?.id}
    canTrain={!!state && !connectionError && !!mapName && approvedCount >= 2} canGenerate={generationEpisodes.length > 0}
    onTrain={() => run(`${family}/auto-train`, { map_name: mapName })}
    onGenerate={() => run(`${family}/auto-label`, { model_id: selected?.id,
      episode_ids: generationEpisodes.map(item => item.id), replace_unreviewed: false })}
    onDelete={async id => { if (await deleteModel(id)) setModelId(""); }}/>;
  const feedback = <LabelerFeedback family={family} workflow={workflow}/>;
  return <section className="qwen-workflow-layout" aria-label={config.title}>
    {renderLayout ? renderLayout({ sampling, actions, feedback }) : <>{sampling}{actions}{feedback}</>}
  </section>;
}
