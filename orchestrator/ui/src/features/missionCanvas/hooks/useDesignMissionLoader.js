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

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getNavigationMission,
  getNavigationMissionBtFile,
} from "../../../utils/navigationMissionsApi";
import { getNavigationSpots } from "../../../utils/navigationSpotsApi";
import {
  buildGlobalMissionXml,
  localBtPathsForSpot,
  missionBtFileDefaultsForSpots,
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
import { savedBehaviorNodesForMap } from "../lib/designStore";

function normalizedName(value, fallback) {
  return String(value || "").trim() || fallback;
}

function normalizedIdentity({ mapName, missionName } = {}) {
  return {
    mapName: normalizedName(mapName, DEFAULT_MAP_NAME),
    missionName: normalizedName(missionName, DEFAULT_MISSION_NAME),
  };
}

async function defaultLoadLegacySpots(mapName) {
  const result = await getNavigationSpots(mapName);
  return Array.isArray(result?.spots) ? result.spots : [];
}

// The manifest revision is the read transaction boundary. Every XML response
// must come from that same generation; otherwise returning even one fallback
// would let a later Save overwrite a newer server document.
export async function loadDesignMissionBtFileOrDefault(
  mapName,
  missionName,
  path,
  fallback,
  expectedRevision,
) {
  const response = await getNavigationMissionBtFile(
    mapName,
    path,
    missionRequestName(missionName),
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

async function loadBtFilesForManifest({
  mapName,
  missionName,
  mission,
  spots,
  flow,
  loadBtFile,
}) {
  const defaults = missionBtFileDefaultsForSpots(spots);
  const globalPath = mission.global_bt || "global.xml";
  const expectedRevision = Number.isInteger(mission.revision) ? mission.revision : 0;
  defaults[globalPath] = buildGlobalMissionXml(
    missionStepSpotsFromMissionFlow(spots, flow.nodes, flow.edges),
  );
  const entries = await Promise.all(
    Object.entries(defaults).map(async ([path, fallback]) => [
      path,
      await loadBtFile(
        mapName,
        missionName,
        path,
        fallback,
        expectedRevision,
      ),
    ]),
  );
  return Object.fromEntries(entries);
}

// Assembles a complete, side-effect-free Design snapshot. It deliberately
// returns data instead of accepting React setters: the caller has one atomic
// commit point and a failed assembly cannot partially replace its document.
export async function loadDesignMissionSnapshot(
  request,
  {
    getMission = getNavigationMission,
    loadBtFile = loadDesignMissionBtFileOrDefault,
    loadLegacySpots = defaultLoadLegacySpots,
    loadLegacyBehaviorNodes = savedBehaviorNodesForMap,
  } = {},
) {
  const identity = normalizedIdentity(request);
  const loadLegacyDesign = request?.loadLegacyDesign === true;
  const mission = await getMission(
    identity.mapName,
    missionRequestName(identity.missionName),
  );
  const exists = Boolean(mission?.exists);

  let spots;
  let flow;
  let btFiles;
  let behaviorNodesPatch = null;
  let loadedLegacyDesign = false;

  if (exists) {
    spots = spotsFromMissionWaypoints(identity.mapName, mission.waypoints);
    flow = normalizeMissionFlow(spots, mission.metadata?.mission_flow);
    btFiles = await loadBtFilesForManifest({
      ...identity,
      mission,
      spots,
      flow,
      loadBtFile,
    });
  } else {
    const legacyResult = await loadLegacySpots(identity.mapName, { apply: false });
    spots = Array.isArray(legacyResult)
      ? legacyResult
      : Array.isArray(legacyResult?.spots)
        ? legacyResult.spots
        : [];
    flow = normalizeMissionFlow(orderedMissionSpots(spots));
    btFiles = missionBtFileDefaultsForSpots(spots);
    if (loadLegacyDesign) {
      const savedNodes = await loadLegacyBehaviorNodes(identity.mapName);
      if (Array.isArray(savedNodes)) {
        behaviorNodesPatch = savedNodes;
        loadedLegacyDesign = true;
      }
    }
  }

  const revision = Number.isInteger(mission?.revision) ? mission.revision : 0;
  const persistedBtFiles = exists ? { ...btFiles } : {};
  const persistedLocalBtPaths = exists
    ? [...new Set(spots.flatMap((spot) => localBtPathsForSpot(spot)))]
    : [];
  const snapshot = {
    identity,
    spots,
    behaviorNodesPatch,
    flowNodes: flow.nodes,
    flowEdges: flow.edges,
    btFiles,
    baseline: {
      revision,
      persistedBtFiles,
      persistedLocalBtPaths,
    },
  };

  return {
    exists,
    loadedDesign: loadedLegacyDesign,
    spotCount: spots.length,
    snapshot,
  };
}

function staleResult(result = {}) {
  return {
    exists: result.exists ?? false,
    loadedDesign: result.loadedDesign ?? false,
    spotCount: result.spotCount ?? 0,
    stale: true,
  };
}

function loaderState(phase, identity, error = null) {
  return { phase, identity, error };
}

// Owns only request lifecycle and the Design document epoch. Editable content
// remains caller-owned until it explicitly commits a successful snapshot.
// Leases let the remaining async editor commands share the same same-identity
// reload guard without exposing the mutable generation ref.
export default function useDesignMissionLoader({
  initialMapName = DEFAULT_MAP_NAME,
  initialMissionName = DEFAULT_MISSION_NAME,
  getMission = getNavigationMission,
  loadBtFile = loadDesignMissionBtFileOrDefault,
  loadLegacySpots = defaultLoadLegacySpots,
  loadLegacyBehaviorNodes = savedBehaviorNodesForMap,
} = {}) {
  const initialIdentityRef = useRef(null);
  if (initialIdentityRef.current === null) {
    initialIdentityRef.current = normalizedIdentity({
      mapName: initialMapName,
      missionName: initialMissionName,
    });
  }
  const generationRef = useRef(0);
  const identityRef = useRef(initialIdentityRef.current);
  const mountedRef = useRef(true);
  const [state, setState] = useState(() => (
    loaderState("idle", initialIdentityRef.current)
  ));

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
    };
  }, []);

  const captureLease = useCallback(() => Object.freeze({
    generation: generationRef.current,
    ...identityRef.current,
  }), []);

  const isCurrent = useCallback((lease) => (
    mountedRef.current
    && Number.isInteger(lease?.generation)
    && lease.generation === generationRef.current
    && lease.mapName === identityRef.current.mapName
    && lease.missionName === identityRef.current.missionName
  ), []);

  const invalidate = useCallback((nextIdentity = identityRef.current) => {
    const identity = normalizedIdentity(nextIdentity);
    generationRef.current += 1;
    identityRef.current = identity;
    if (mountedRef.current) setState(loaderState("idle", identity));
    return Object.freeze({ generation: generationRef.current, ...identity });
  }, []);

  const clearError = useCallback(() => {
    if (!mountedRef.current) return;
    setState((current) => (
      current.error ? loaderState("idle", current.identity) : current
    ));
  }, []);

  const load = useCallback(async (request = {}) => {
    const identity = normalizedIdentity(request);
    const lease = invalidate(identity);
    if (!mountedRef.current) return staleResult();
    setState(loaderState("loading", identity));
    try {
      const result = await loadDesignMissionSnapshot(request, {
        getMission,
        loadBtFile,
        loadLegacySpots,
        loadLegacyBehaviorNodes,
      });
      if (!isCurrent(lease)) return staleResult(result);
      setState(loaderState("idle", identity));
      return { ...result, lease };
    } catch (error) {
      if (!isCurrent(lease)) return staleResult();
      setState(loaderState("error", identity, error));
      throw error;
    }
  }, [
    getMission,
    invalidate,
    isCurrent,
    loadBtFile,
    loadLegacyBehaviorNodes,
    loadLegacySpots,
  ]);

  return {
    ...state,
    load,
    invalidate,
    captureLease,
    isCurrent,
    clearError,
  };
}
