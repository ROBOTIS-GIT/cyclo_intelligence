// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

const OFFLINE_RL_API_BASE = '/api/offline-rl';

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
