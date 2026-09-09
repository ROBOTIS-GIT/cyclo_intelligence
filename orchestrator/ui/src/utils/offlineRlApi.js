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

async function getJson(url, action) {
  return requireOk(await fetch(url, { cache: 'no-store' }), action);
}

async function postJson(url, payload, action) {
  return requireOk(await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }), action);
}

export async function startOfflineRLTraining(request) {
  return postJson(`${OFFLINE_RL_API_BASE}/start`, request, 'Offline RL start');
}

export async function getOfflineRLStatus() {
  return getJson(`${OFFLINE_RL_API_BASE}/status`, 'Offline RL status');
}

export async function stopOfflineRLTraining(jobId) {
  return postJson(`${OFFLINE_RL_API_BASE}/stop`, { job_id: jobId }, 'Offline RL stop');
}

export async function cancelOfflineRLTraining(jobId) {
  return postJson(`${OFFLINE_RL_API_BASE}/cancel`, { job_id: jobId }, 'Offline RL cancel');
}

export async function startACTTD3CriticWarmup(request) {
  return postJson(`${OFFLINE_RL_API_BASE}/critic-warmup/start`, request, 'ACT-TD3 critic warm-up start');
}

export async function getACTTD3CriticWarmupStatus() {
  return getJson(`${OFFLINE_RL_API_BASE}/critic-warmup/status`, 'ACT-TD3 critic warm-up status');
}

export async function stopACTTD3CriticWarmup(jobId) {
  return postJson(`${OFFLINE_RL_API_BASE}/critic-warmup/stop`, { job_id: jobId }, 'ACT-TD3 critic warm-up stop');
}

export async function startImitationLearningTraining(request) {
  return postJson(`${IMITATION_LEARNING_API_BASE}/start`, request, 'Imitation Learning start');
}

export async function getImitationLearningStatus() {
  return getJson(`${IMITATION_LEARNING_API_BASE}/status`, 'Imitation Learning status');
}

export async function stopImitationLearningTraining(jobId) {
  return postJson(`${IMITATION_LEARNING_API_BASE}/stop`, { job_id: jobId }, 'Imitation Learning stop');
}

export async function startRLTStage1Training(request) {
  return postJson(`${RLT_STAGE1_API_BASE}/start`, request, 'RLT Stage 1 start');
}

export async function getRLTStage1Status() {
  return getJson(`${RLT_STAGE1_API_BASE}/status`, 'RLT Stage 1 status');
}

export async function stopRLTStage1Training(jobId) {
  return postJson(`${RLT_STAGE1_API_BASE}/stop`, { job_id: jobId }, 'RLT Stage 1 stop');
}

export async function startRLTStage2Training(request) {
  return postJson(`${RLT_STAGE2_API_BASE}/start`, request, 'RLT Stage 2 start');
}

export async function getRLTStage2Status() {
  return getJson(`${RLT_STAGE2_API_BASE}/status`, 'RLT Stage 2 status');
}

export async function stopRLTStage2Training(jobId) {
  return postJson(`${RLT_STAGE2_API_BASE}/stop`, { job_id: jobId }, 'RLT Stage 2 stop');
}

export async function startFlowSDEPPOPolicyRollout(request) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/rollout/start`, request, 'Flow-SDE PPO rollout start');
}

export async function getFlowSDEPPOPolicyRolloutStatus() {
  return getJson(`${FLOW_SDE_PPO_API_BASE}/rollout/status`, 'Flow-SDE PPO rollout status');
}

export async function stopFlowSDEPPOPolicyRollout(jobId) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/rollout/stop`, { job_id: jobId }, 'Flow-SDE PPO rollout stop');
}

export async function submitFlowSDEPPOPolicyRolloutOutcome(jobId, outcome) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/rollout/outcome`, { job_id: jobId, outcome }, 'Flow-SDE PPO rollout outcome');
}

export async function startFlowSDEPPOUpdate(rolloutBundle) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/update/start`, { rollout_bundle: rolloutBundle }, 'Flow-SDE PPO update start');
}

export async function getFlowSDEPPOUpdateStatus() {
  return getJson(`${FLOW_SDE_PPO_API_BASE}/update/status`, 'Flow-SDE PPO update status');
}

export async function stopFlowSDEPPOUpdate(jobId) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/update/stop`, { job_id: jobId }, 'Flow-SDE PPO update stop');
}

export async function startFlowSDEPPOValueWarmup(request) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/value-warmup/start`, request, 'Flow-SDE PPO value warm-up start');
}

export async function getFlowSDEPPOValueWarmupStatus() {
  return getJson(`${FLOW_SDE_PPO_API_BASE}/value-warmup/status`, 'Flow-SDE PPO value warm-up status');
}

export async function stopFlowSDEPPOValueWarmup(jobId) {
  return postJson(`${FLOW_SDE_PPO_API_BASE}/value-warmup/stop`, { job_id: jobId }, 'Flow-SDE PPO value warm-up stop');
}

export async function getOfflineRLDatasetInfo(datasetPath) {
  const query = new URLSearchParams({ dataset_path: datasetPath });
  return getJson(`${OFFLINE_RL_API_BASE}/dataset?${query}`, 'LeRobot dataset inspection');
}

export async function getOfflineRLDatasetEpisodeData(datasetPath, episodeIndex) {
  const query = new URLSearchParams({
    dataset_path: datasetPath,
    episode_index: String(episodeIndex),
  });
  return getJson(`${OFFLINE_RL_API_BASE}/dataset/episode-data?${query}`, 'LeRobot episode data');
}

export async function getOfflineRLDatasets(rootPath = '') {
  const query = new URLSearchParams();
  if (String(rootPath || '').trim()) {
    query.set('root_path', String(rootPath).trim());
  }
  const suffix = query.toString() ? `?${query}` : '';
  return getJson(`${OFFLINE_RL_API_BASE}/datasets${suffix}`, 'LeRobot dataset inventory');
}

export async function reserveOfflineRLDataEpoch(request) {
  return postJson(`${OFFLINE_RL_API_BASE}/data-epochs/reserve`, request, 'Data Epoch reservation');
}

export async function deleteOfflineRLDatasetEpisodes(datasetPath, episodeIndices) {
  return postJson(`${OFFLINE_RL_API_BASE}/dataset/delete-episodes`, {
      dataset_path: datasetPath,
      episode_indices: episodeIndices,
    }, 'LeRobot episode deletion');
}
