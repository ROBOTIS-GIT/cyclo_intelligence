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

const API_BASE = '/api/navigation/missions';

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

function missionQuery(missionName) {
  return missionName ? `?mission_name=${encodeURIComponent(missionName)}` : "";
}

export function getNavigationMissions(mapName) {
  return request(`?map_name=${encodeURIComponent(mapName)}`);
}

export function getNavigationMission(mapName, missionName = "") {
  return request(`/${encodeURIComponent(mapName)}${missionQuery(missionName)}`);
}

export function saveNavigationMission(mapName, mission, missionName = "") {
  return request(`/${encodeURIComponent(mapName)}${missionQuery(missionName)}`, {
    method: 'POST',
    body: JSON.stringify(mission),
  });
}

export function deleteNavigationMission(mapName, missionName, options = {}) {
  const expectedRevision = options?.expectedRevision ?? options?.expected_revision;
  return request(
    `/${encodeURIComponent(mapName)}?mission_name=${encodeURIComponent(missionName)}`
      + `${Number.isInteger(expectedRevision)
        ? `&expected_revision=${encodeURIComponent(expectedRevision)}`
        : ''}`,
    { method: 'DELETE' },
  );
}

export function renameNavigationMission(mapName, missionName, newName, options = {}) {
  const expectedRevision = options?.expectedRevision ?? options?.expected_revision;
  return request(`/${encodeURIComponent(mapName)}/rename`, {
    method: 'POST',
    body: JSON.stringify({
      mission_name: missionName,
      new_name: newName,
      ...(Number.isInteger(expectedRevision)
        ? { expected_revision: expectedRevision }
        : {}),
    }),
  });
}

export function duplicateNavigationMission(mapName, missionName, newName, options = {}) {
  const expectedRevision = options?.expectedRevision ?? options?.expected_revision;
  return request(`/${encodeURIComponent(mapName)}/duplicate`, {
    method: 'POST',
    body: JSON.stringify({
      mission_name: missionName,
      new_name: newName,
      ...(Number.isInteger(expectedRevision)
        ? { expected_revision: expectedRevision }
        : {}),
    }),
  });
}

export function getNavigationMissionBtFile(mapName, path, missionName = "") {
  return request(
    `/${encodeURIComponent(mapName)}/bt?path=${encodeURIComponent(path)}${missionName ? `&mission_name=${encodeURIComponent(missionName)}` : ""}`,
  );
}

export function saveNavigationMissionBtFile(
  mapName,
  path,
  content,
  missionName = "",
  options = {},
) {
  const waypointId = options?.waypointId || options?.waypoint_id || "";
  const expectedRevision = options?.expectedRevision ?? options?.expected_revision;
  return request(`/${encodeURIComponent(mapName)}/bt${missionQuery(missionName)}`, {
    method: 'PUT',
    body: JSON.stringify({
      path,
      content,
      ...(waypointId ? { waypoint_id: waypointId } : {}),
      ...(Number.isInteger(expectedRevision)
        ? { expected_revision: expectedRevision }
        : {}),
    }),
  });
}

export function setNavigationMissionDefaultBtFile(
  mapName,
  waypointId,
  path,
  missionName = "",
  options = {},
) {
  const expectedRevision = options?.expectedRevision ?? options?.expected_revision;
  return request(
    `/${encodeURIComponent(mapName)}/bt/default${missionQuery(missionName)}`,
    {
      method: 'PUT',
      body: JSON.stringify({
        waypoint_id: waypointId,
        path,
        ...(Number.isInteger(expectedRevision)
          ? { expected_revision: expectedRevision }
          : {}),
      }),
    },
  );
}

export function deleteNavigationMissionBtFile(
  mapName,
  path,
  missionName = "",
  options = {},
) {
  const expectedRevision = options?.expectedRevision ?? options?.expected_revision;
  return request(
    `/${encodeURIComponent(mapName)}/bt?path=${encodeURIComponent(path)}`
      + `${missionName ? `&mission_name=${encodeURIComponent(missionName)}` : ""}`
      + `${Number.isInteger(expectedRevision)
        ? `&expected_revision=${encodeURIComponent(expectedRevision)}`
        : ""}`,
    { method: 'DELETE' },
  );
}
