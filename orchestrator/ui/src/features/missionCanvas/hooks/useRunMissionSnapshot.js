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

import { useCallback, useRef, useState } from "react";
import { getNavigationMission } from "../../../utils/navigationMissionsApi";
import {
  buildGlobalMissionXml,
  missionBtFileDefaultsForRunSpots,
} from "../lib/missionBtFiles";
import {
  missionStepSpotsFromMissionFlow,
  normalizeMissionFlow,
} from "../lib/missionFlow";
import {
  orderedMissionSpots,
  spotsFromMissionWaypoints,
} from "../lib/missionSpots";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../lib/missionNames";

function normalizedName(value, fallback) {
  return String(value || "").trim() || fallback;
}

export function emptyRunMissionSnapshot({
  mapName = "",
  missionName = "",
} = {}) {
  return {
    mapName,
    missionName,
    catalog: { mapName: "", names: [] },
    spots: [],
    btFiles: {},
    flowNodes: [],
    flowEdges: [],
    invalid: true,
  };
}

// Owns the Run tab's read-only mission document. A load is assembled entirely
// off-state and committed with one setState only after the manifest and every
// runtime BT file agree on one revision. Generation checks prevent a slower
// selection from overwriting a newer one.
export default function useRunMissionSnapshot({
  initialMissionName = DEFAULT_MISSION_NAME,
  loadLegacySpotsForMap,
  loadMissionBtFileOrDefault,
} = {}) {
  const loadGenerationRef = useRef(0);
  const [snapshot, setSnapshot] = useState(() => emptyRunMissionSnapshot({
    missionName: normalizedName(initialMissionName, DEFAULT_MISSION_NAME),
  }));

  const load = useCallback(async (
    targetMapName,
    targetMissionName,
    { catalogNames } = {},
  ) => {
    const mapName = normalizedName(targetMapName, DEFAULT_MAP_NAME);
    const missionName = normalizedName(targetMissionName, DEFAULT_MISSION_NAME);
    const generation = loadGenerationRef.current + 1;
    loadGenerationRef.current = generation;

    let exists = false;
    let spots = [];
    try {
      const mission = await getNavigationMission(
        mapName,
        missionRequestName(missionName),
      );
      exists = Boolean(mission?.exists);

      let flow;
      let btFiles;
      if (exists) {
        spots = spotsFromMissionWaypoints(mapName, mission.waypoints);
        flow = normalizeMissionFlow(spots, mission.metadata?.mission_flow);

        // Run consumes the selected local_bt for each waypoint. Alternate
        // authoring files are intentionally not fetched here.
        const defaults = missionBtFileDefaultsForRunSpots(spots);
        const globalPath = mission.global_bt || "global.xml";
        const expectedRevision = Number.isInteger(mission.revision) ? mission.revision : 0;
        defaults[globalPath] = buildGlobalMissionXml(
          missionStepSpotsFromMissionFlow(spots, flow.nodes, flow.edges),
        );
        const entries = await Promise.all(Object.entries(defaults).map(async ([path, fallback]) => [
          path,
          await loadMissionBtFileOrDefault(
            mapName,
            missionName,
            path,
            fallback,
            expectedRevision,
          ),
        ]));
        btFiles = Object.fromEntries(entries);
      } else {
        spots = await loadLegacySpotsForMap(mapName, { apply: false });
        flow = normalizeMissionFlow(orderedMissionSpots(spots));
        btFiles = missionBtFileDefaultsForRunSpots(spots);
      }

      if (loadGenerationRef.current !== generation) {
        return { exists, loadedDesign: false, spotCount: spots.length, stale: true };
      }

      setSnapshot((current) => ({
        mapName,
        missionName,
        catalog: Array.isArray(catalogNames)
          ? { mapName, names: catalogNames }
          : current.catalog.mapName === mapName
            ? current.catalog
            : { mapName: "", names: [] },
        spots,
        btFiles,
        flowNodes: flow.nodes,
        flowEdges: flow.edges,
        invalid: false,
      }));
      return { exists, loadedDesign: false, spotCount: spots.length };
    } catch (error) {
      // Cancellation makes every late outcome inert, including a rejected
      // request. The caller that initiated the newer load owns any UI error.
      if (loadGenerationRef.current !== generation) {
        return { exists, loadedDesign: false, spotCount: spots.length, stale: true };
      }
      throw error;
    }
  }, [loadLegacySpotsForMap, loadMissionBtFileOrDefault]);

  const clear = useCallback((identity = {}) => {
    setSnapshot(emptyRunMissionSnapshot(identity));
  }, []);

  const invalidate = useCallback(() => {
    loadGenerationRef.current += 1;
  }, []);

  const cancelAndClear = useCallback((identity = {}) => {
    invalidate();
    setSnapshot(emptyRunMissionSnapshot(identity));
  }, [invalidate]);

  return {
    snapshot,
    load,
    clear,
    invalidate,
    cancelAndClear,
  };
}
