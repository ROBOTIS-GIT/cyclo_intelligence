// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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
      mode: mode === 'map' ? 'map' : 'nav',
      map_name: mapName,
    }),
  });
}

export function stopNavigation() {
  return request('/stop', { method: 'POST' });
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

export function sendNavigateToPoseGoal(goal) {
  return request('/goal', {
    method: 'POST',
    body: JSON.stringify(goal),
  });
}

export function cancelNavigateToPoseGoal() {
  return request('/cancel', { method: 'POST' });
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
