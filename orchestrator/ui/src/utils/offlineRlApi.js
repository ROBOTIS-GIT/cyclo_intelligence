// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

const OFFLINE_RL_API_BASE = '/api/offline-rl';
const IMITATION_LEARNING_API_BASE = '/api/imitation-learning';
const FLOW_SDE_PPO_API_BASE = '/api/flow-sde-ppo';
const RLT_STAGE1_API_BASE = '/api/rlt-stage1';
const RLT_STAGE2_API_BASE = '/api/rlt-stage2';

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (_error) {
    return { detail: text };
  }
}

async function requireOk(response, action) {
  const data = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(
      data.detail || data.message || `${action} failed (${response.status})`
    );
  }
  return data;
}

export async function startOfflineRLTraining(request) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'Offline RL start');
}

export async function getOfflineRLStatus() {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'Offline RL status');
}

export async function stopOfflineRLTraining(jobId) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Offline RL stop');
}

export async function cancelOfflineRLTraining(jobId) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Offline RL cancel');
}

export async function startACTTD3CriticWarmup(request) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/critic-warmup/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'ACT-TD3 critic warm-up start');
}

export async function getACTTD3CriticWarmupStatus() {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/critic-warmup/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'ACT-TD3 critic warm-up status');
}

export async function stopACTTD3CriticWarmup(jobId) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/critic-warmup/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'ACT-TD3 critic warm-up stop');
}

export async function startImitationLearningTraining(request) {
  const response = await fetch(`${IMITATION_LEARNING_API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'Imitation Learning start');
}

export async function getImitationLearningStatus() {
  const response = await fetch(`${IMITATION_LEARNING_API_BASE}/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'Imitation Learning status');
}

export async function stopImitationLearningTraining(jobId) {
  const response = await fetch(`${IMITATION_LEARNING_API_BASE}/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Imitation Learning stop');
}

export async function startRLTStage1Training(request) {
  const response = await fetch(`${RLT_STAGE1_API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'RLT Stage 1 start');
}

export async function getRLTStage1Status() {
  const response = await fetch(`${RLT_STAGE1_API_BASE}/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'RLT Stage 1 status');
}

export async function stopRLTStage1Training(jobId) {
  const response = await fetch(`${RLT_STAGE1_API_BASE}/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'RLT Stage 1 stop');
}

export async function startRLTStage2Training(request) {
  const response = await fetch(`${RLT_STAGE2_API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'RLT Stage 2 start');
}

export async function getRLTStage2Status() {
  const response = await fetch(`${RLT_STAGE2_API_BASE}/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'RLT Stage 2 status');
}

export async function stopRLTStage2Training(jobId) {
  const response = await fetch(`${RLT_STAGE2_API_BASE}/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'RLT Stage 2 stop');
}

export async function startFlowSDEPPOTraining(request) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'Flow-SDE PPO start');
}

export async function getFlowSDEPPOStatus() {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'Flow-SDE PPO status');
}

export async function stopFlowSDEPPOTraining(jobId) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Flow-SDE PPO stop');
}

export async function submitFlowSDEPPOOutcome(jobId, outcome) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/outcome`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, outcome }),
  });
  return requireOk(response, 'Flow-SDE PPO outcome');
}

export async function startFlowSDEPPOPolicyRollout(request) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/rollout/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'Flow-SDE PPO rollout start');
}

export async function getFlowSDEPPOPolicyRolloutStatus() {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/rollout/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'Flow-SDE PPO rollout status');
}

export async function stopFlowSDEPPOPolicyRollout(jobId) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/rollout/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Flow-SDE PPO rollout stop');
}

export async function submitFlowSDEPPOPolicyRolloutOutcome(jobId, outcome) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/rollout/outcome`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, outcome }),
  });
  return requireOk(response, 'Flow-SDE PPO rollout outcome');
}

export async function startFlowSDEPPOUpdate(rolloutBundle) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/update/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rollout_bundle: rolloutBundle }),
  });
  return requireOk(response, 'Flow-SDE PPO update start');
}

export async function getFlowSDEPPOUpdateStatus() {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/update/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'Flow-SDE PPO update status');
}

export async function stopFlowSDEPPOUpdate(jobId) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/update/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Flow-SDE PPO update stop');
}

export async function startFlowSDEPPOValueWarmup(request) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/value-warmup/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'Flow-SDE PPO value warm-up start');
}

export async function getFlowSDEPPOValueWarmupStatus() {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/value-warmup/status`, {
    cache: 'no-store',
  });
  return requireOk(response, 'Flow-SDE PPO value warm-up status');
}

export async function stopFlowSDEPPOValueWarmup(jobId) {
  const response = await fetch(`${FLOW_SDE_PPO_API_BASE}/value-warmup/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return requireOk(response, 'Flow-SDE PPO value warm-up stop');
}

export async function getOfflineRLDatasetInfo(datasetPath) {
  const query = new URLSearchParams({ dataset_path: datasetPath });
  const response = await fetch(`${OFFLINE_RL_API_BASE}/dataset?${query}`, {
    cache: 'no-store',
  });
  return requireOk(response, 'LeRobot dataset inspection');
}

export async function getOfflineRLDatasetEpisodeData(datasetPath, episodeIndex) {
  const query = new URLSearchParams({
    dataset_path: datasetPath,
    episode_index: String(episodeIndex),
  });
  const response = await fetch(
    `${OFFLINE_RL_API_BASE}/dataset/episode-data?${query}`,
    { cache: 'no-store' }
  );
  return requireOk(response, 'LeRobot episode data');
}

export async function getOfflineRLDatasets(rootPath = '') {
  const query = new URLSearchParams();
  if (String(rootPath || '').trim()) {
    query.set('root_path', String(rootPath).trim());
  }
  const suffix = query.toString() ? `?${query}` : '';
  const response = await fetch(`${OFFLINE_RL_API_BASE}/datasets${suffix}`, {
    cache: 'no-store',
  });
  return requireOk(response, 'LeRobot dataset inventory');
}

export async function reserveOfflineRLDataEpoch(request) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/data-epochs/reserve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return requireOk(response, 'Data Epoch reservation');
}

export async function deleteOfflineRLDatasetEpisodes(datasetPath, episodeIndices) {
  const response = await fetch(`${OFFLINE_RL_API_BASE}/dataset/delete-episodes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_path: datasetPath,
      episode_indices: episodeIndices,
    }),
  });
  return requireOk(response, 'LeRobot episode deletion');
}
