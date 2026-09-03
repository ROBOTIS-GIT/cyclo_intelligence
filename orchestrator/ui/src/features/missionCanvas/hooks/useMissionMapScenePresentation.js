// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import { useMemo, useRef } from "react";
import { LAYER_DEFINITIONS, STAGE_LAYER_IDS } from "../lib/layers";
import { deriveMissionRouteView } from "../lib/missionRouteView";
import { spotForMapDisplay } from "../lib/missionSpots";
import { STAGE_AUTHORING, STAGE_RUN } from "../lib/stages";

function normalizedMapIdentity(value) {
  const leaf = String(value || "").trim().split(/[\\/]/).pop() || "";
  return leaf.replace(/\.[^.]+$/, "");
}

function useIdentityScopedMap(expectedIdentity, sample) {
  const cacheRef = useRef({ identity: "", map: null });
  const expected = normalizedMapIdentity(expectedIdentity);
  const sampleIdentity = normalizedMapIdentity(sample?.identity);

  // MapEditor intentionally retains its previous image while a replacement is
  // loading. Only associate a grid with the document after the loaded image's
  // identity agrees with the requested map; otherwise legacy-cell projection
  // can briefly use the previous map's origin and resolution.
  if (sample?.map && expected && sampleIdentity === expected) {
    cacheRef.current = { identity: expected, map: sample.map };
  }

  return cacheRef.current.identity === expected ? cacheRef.current.map : null;
}

export function deriveMissionRoutePresentation({
  runSessionActive,
  designDocumentReady,
  runRouteView,
  designRouteView,
}) {
  return {
    renderedRouteView: runSessionActive ? runRouteView : designRouteView,
    designPanelRouteSpots: designDocumentReady ? (designRouteView?.treeSpots || []) : [],
    designPanelRouteClosed: designDocumentReady && !!designRouteView?.closed,
  };
}

export default function useMissionMapScenePresentation({
  workspaceStage,
  designMapName,
  runtimeMapName,
  designMapSample,
  runMapSample,
  designMapAvailable,
  designMapBusy,
  designMissionLoadPhase,
  designMissionLoadError,
  displayedMap,
  runSessionActive,
  designSpots,
  runSpots,
  designBehaviorNodes,
  runBehaviorNodes,
  runMissionFlowNodes,
  runMissionFlowEdges,
  activeLayers,
  setLayersByStage,
}) {
  const designMissionMap = useIdentityScopedMap(designMapName, designMapSample);
  const runMissionMap = useIdentityScopedMap(runtimeMapName, runMapSample);

  const designDocumentReady = (
    designMapAvailable
    && !designMapBusy
    && designMissionLoadPhase === "idle"
    && !designMissionLoadError
  );
  const missionOverlayActive = (
    (workspaceStage === STAGE_RUN && !!displayedMap)
    || (workspaceStage === STAGE_AUTHORING && designDocumentReady)
  );

  const designVisibleSpots = useMemo(
    () => (designSpots || []).map((spot) => spotForMapDisplay(spot, designMissionMap)),
    [designMissionMap, designSpots],
  );
  const runVisibleSpots = useMemo(
    () => (runSpots || []).map((spot) => spotForMapDisplay(spot, runMissionMap)),
    [runMissionMap, runSpots],
  );
  const runRouteView = useMemo(() => deriveMissionRouteView({
    spots: runVisibleSpots,
    flowNodes: runMissionFlowNodes || [],
    flowEdges: runMissionFlowEdges || [],
  }), [runMissionFlowEdges, runMissionFlowNodes, runVisibleSpots]);

  const renderedBehaviorNodes = runSessionActive ? runBehaviorNodes : designBehaviorNodes;
  const renderedVisibleSpots = runSessionActive ? runVisibleSpots : designVisibleSpots;
  const designPanelSpots = designDocumentReady ? (designSpots || []) : [];
  const designPanelBehaviorNodes = designDocumentReady ? (designBehaviorNodes || []) : [];

  const layerToggles = useMemo(() => (
    (STAGE_LAYER_IDS[workspaceStage] || []).map((id) => ({
      id,
      label: LAYER_DEFINITIONS[id],
      checked: !!activeLayers?.[id],
      onChange: (checked) => {
        setLayersByStage((current) => ({
          ...current,
          [workspaceStage]: {
            ...current[workspaceStage],
            [id]: checked,
          },
        }));
      },
    }))
  ), [activeLayers, setLayersByStage, workspaceStage]);

  return {
    designMissionMap,
    runMissionMap,
    designDocumentReady,
    missionOverlayActive,
    designVisibleSpots,
    runVisibleSpots,
    runRouteView,
    renderedBehaviorNodes,
    renderedVisibleSpots,
    designPanelSpots,
    designPanelBehaviorNodes,
    layerToggles,
  };
}
