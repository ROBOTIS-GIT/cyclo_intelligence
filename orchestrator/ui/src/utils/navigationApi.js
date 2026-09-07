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

const API_BASE = '/api/navigation';

async function request(path, init) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      // Keep the HTTP status for a non-JSON error response.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined;
  return response.json();
}

export function getServiceStatus() {
  return request('/status');
}

export function startNavigation(mode, mapName = 'map') {
  return request('/start', {
    method: 'POST',
    body: JSON.stringify({
      mode: mode === 'map' ? 'map' : mode === 'localize' ? 'localize' : 'nav',
      map_name: mapName,
    }),
  });
}

export function stopNavigation({ keepalive = false } = {}) {
  return request('/stop', {
    method: 'POST',
    ...(keepalive ? { keepalive: true } : {}),
  });
}

export function saveNavigationMap(mapName = 'map') {
  return request('/save-map', {
    method: 'POST',
    body: JSON.stringify({ map_name: mapName }),
  });
}

export function getPgmFiles() {
  return request('/maps/pgm-files');
}

export function getPgmImage(path) {
  return request(`/maps/pgm?path=${encodeURIComponent(path)}`);
}

export function deletePgmMap(path) {
  return request(`/maps/pgm?path=${encodeURIComponent(path)}`, {
    method: 'DELETE',
  });
}

export function savePgmImage(
  path,
  width,
  height,
  maxval,
  pixelsBase64
) {
  return request('/maps/pgm/save', {
    method: 'POST',
    body: JSON.stringify({
      path,
      width,
      height,
      maxval,
      pixels_base64: pixelsBase64,
    }),
  });
}

export function getMapAnnotations(path) {
  return request(`/maps/annotations?path=${encodeURIComponent(path)}`);
}

export function saveMapAnnotations(path, annotations) {
  return request('/maps/annotations/save', {
    method: 'POST',
    body: JSON.stringify({
      path,
      annotations,
    }),
  });
}

export function sendNavigateToPoseGoal(goal) {
  return request('/goal', {
    method: 'POST',
    body: JSON.stringify(goal),
  });
}

export function sendNavigateToPoseGoalAndWait(goal, signal) {
  return request('/goal/wait', {
    method: 'POST',
    body: JSON.stringify(goal),
    signal,
  });
}

export function sendNavigateThroughPosesGoalsAndWait(goals, signal) {
  return request('/goals/wait', {
    method: 'POST',
    body: JSON.stringify(goals),
    signal,
  });
}

export function cancelNavigateToPoseGoal() {
  return request('/cancel', { method: 'POST' });
}

export function sendInitialPoseEstimate({ x, y, yaw, frameId = 'map', mapName }) {
  return request('/initial-pose', {
    method: 'POST',
    body: JSON.stringify({
      x,
      y,
      yaw,
      frame_id: frameId,
      ...(mapName ? { map_name: mapName } : {}),
    }),
  });
}

export function requestNoMotionUpdate() {
  return request('/nomotion-update', { method: 'POST' });
}

export function requestGlobalLocalization() {
  return request('/global-localization', { method: 'POST' });
}

export function configureDesignLocalizationAmcl() {
  return request('/amcl/design-localization-params', { method: 'POST' });
}

export function getServiceLogs({ tail = 300, cursor } = {}) {
  const params = new URLSearchParams({ tail: String(tail) });
  if (cursor !== undefined && cursor !== null) {
    params.set('cursor', String(cursor));
  }
  return request(`/logs?${params.toString()}`);
}

export function clearServiceLogs() {
  return request('/logs', { method: 'DELETE' });
}
