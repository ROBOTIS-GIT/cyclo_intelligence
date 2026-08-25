// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

const OFFLINE_RL_API_BASE = '/api/offline-rl';
const IMITATION_LEARNING_API_BASE = '/api/imitation-learning';
const FLOW_SDE_PPO_API_BASE = '/api/flow-sde-ppo';

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
