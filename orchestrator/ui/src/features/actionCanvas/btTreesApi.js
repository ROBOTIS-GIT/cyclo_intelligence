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

// Saved Action Canvas trees are user data owned by the supervisor API
// (CYCLO_BT_TREES_DIR); the UI never touches file paths directly.
const API_BASE = '/api/bt/trees';

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function detailMessage(data, fallback) {
  const detail = data?.detail;
  if (detail && typeof detail === 'object') return detail.message || fallback;
  return detail || data?.message || fallback;
}

async function request(path, init) {
  const response = await fetch(`${API_BASE}${path}`, init);
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const error = new Error(detailMessage(data, `Request failed (${response.status})`));
    error.status = response.status;
    error.detail = data?.detail;
    throw error;
  }
  return data;
}

export function listBtTrees() {
  return request('');
}

export function readBtTree(name) {
  return request(`/${encodeURIComponent(name)}`);
}

export function saveBtTree({ filename, content, overwrite = false }) {
  return request('', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, content, overwrite }),
  });
}
