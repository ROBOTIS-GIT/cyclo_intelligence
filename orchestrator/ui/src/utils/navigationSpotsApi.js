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

const API_BASE = '/api/navigation/spots';

async function request(path = '', init) {
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
      // Keep the HTTP status for non-JSON errors.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined;
  return response.json();
}

export function getNavigationSpots(mapName = 'map') {
  return request(`?map_name=${encodeURIComponent(mapName)}`);
}

export function createNavigationSpot(spot) {
  return request('', {
    method: 'POST',
    body: JSON.stringify(spot),
  });
}

export function updateNavigationSpot(spotId, patch) {
  return request(`/${encodeURIComponent(spotId)}`, {
    method: 'PATCH',
    body: JSON.stringify(patch),
  });
}

export function deleteNavigationSpot(spotId, mapName = 'map') {
  return request(
    `/${encodeURIComponent(spotId)}?map_name=${encodeURIComponent(mapName)}`,
    { method: 'DELETE' },
  );
}
