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

import { getNavigationMissionBtFile } from "../../../utils/navigationMissionsApi";
import { getNavigationSpots } from "../../../utils/navigationSpotsApi";
import { formatTaskDisplayMessage } from "../../../utils/taskTerminology";
import { DEFAULT_MAP_NAME, missionRequestName } from "./missionNames";

export function taskDisplayMessage(value) {
  return formatTaskDisplayMessage(value, "Waypoint Task");
}

export function navigationRuntimeModeFromStatus(status) {
  if (status?.is_up === false) return "idle";
  if (!status?.is_up) return "";
  if (status.mode === "map" || status.mode === "mapping") return "mapping";
  if (status.mode === "localize" || status.mode === "localization") return "localization";
  if (status.mode === "nav" || status.mode === "run") return "run";
  return "";
}

export async function fetchLegacySpotsForMap(targetMapName) {
  const normalizedMapName = String(targetMapName || "").trim() || DEFAULT_MAP_NAME;
  const result = await getNavigationSpots(normalizedMapName);
  return result.spots || [];
}

export async function loadMissionBtFileOrDefault(
  targetMapName,
  targetMissionName,
  path,
  fallback,
  expectedRevision,
) {
  const response = await getNavigationMissionBtFile(
    targetMapName,
    path,
    missionRequestName(targetMissionName),
  );
  if (
    Number.isInteger(expectedRevision)
    && Number.isInteger(response?.revision)
    && response.revision !== expectedRevision
  ) {
    throw new Error(
      `Mission changed while loading ${path}; reload the mission before editing or running it`,
    );
  }
  if (response?.exists && typeof response.content === "string") {
    return response.content;
  }
  return fallback;
}
