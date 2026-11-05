// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import React from "react";
import { MapViewer } from "../../../../components/navigation/MapViewer";
import { EDITOR_BRUSH_RING_COLORS } from "../../lib/theme";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_NAVIGATE,
  STAGE_RUN,
} from "../../lib/stages";
import { LayersPopover } from "../mapChrome";
import DesignStageHud from "../stages/DesignStageHud";
import MapEditStageOverlay from "../stages/MapEditStageOverlay";
import MappingStageHud from "../stages/MappingStageHud";
import { NavigationStageHud } from "../stages/NavigationStage";
import { RunStageHud } from "../stages/RunStage";

export default function MissionCanvasSceneSurface({
  stage,
  scene,
  mission,
  editors,
  interaction,
  bt,
  layers,
  hud,
}) {
  const { mapEditor, designMapEditor, runDisplayMapEditor } = editors;
  const mappingEditorActive = stage.mappingEditorActive;
  const authoring = stage.id === STAGE_AUTHORING;

  return (
    <section
      className="min-h-0 overflow-hidden relative"
      style={{ backgroundColor: "var(--mc-surface)", borderRight: "1px solid var(--mc-border)" }}
    >
      <MapViewer
        map={scene.displayedMap}
        globalCostmap={mappingEditorActive ? null : layers.needsGlobalCostmap ? scene.globalCostmap : null}
        localCostmap={mappingEditorActive ? null : layers.needsLocalCostmap ? scene.localCostmap : null}
        scan={mappingEditorActive ? null : layers.needsScan ? scene.scan : null}
        scanPose={
          mappingEditorActive
            ? null
            : stage.mappingTopicsActive
              ? scene.mappingPoseSync.scanPose
              : stage.runTopicsActive
                ? scene.runPoseSync.scanPose
                : null
        }
        pose={mappingEditorActive
          ? null
          : (stage.designLocalizationActive || stage.navigationTopicsActive)
            ? scene.currentPose
            : null}
        goalPose={stage.id === STAGE_NAVIGATE ? scene.navGoalPose : null}
        showGoalPose={stage.id === STAGE_NAVIGATE && !!scene.navGoalPose && scene.navGoalStatus !== "reached"}
        plan={mappingEditorActive ? null : layers.needsPlan ? scene.plan : null}
        footprint={mappingEditorActive ? null : layers.needsRobotModel ? scene.footprint : null}
        tf={mappingEditorActive ? null : (layers.needsTf || layers.needsRobotModel) ? scene.displayTf : null}
        spots={mission.overlayActive ? mission.renderedVisibleSpots : []}
        selectedSpotId={mission.overlayActive && authoring ? mission.selectedSpotId : ""}
        activeWaypointId={stage.id === STAGE_RUN ? mission.activeWaypointId : ""}
        missionFollowRobot={mission.followRobot}
        behaviorNodes={mission.overlayActive ? mission.renderedBehaviorNodes : []}
        selectedBehaviorNodeId={mission.overlayActive ? mission.selectedBehaviorNodeId : ""}
        behaviorPreviewNode={mission.overlayActive ? mission.behaviorPreviewNode : null}
        missionRouteOrder={mission.overlayActive ? mission.renderedRouteView.order : []}
        missionRouteClosed={mission.overlayActive && mission.renderedRouteView.closed}
        missionRouteMode={authoring && mission.routeMode}
        selectedMissionRouteSourceId={mission.routeSourceId}
        mapAnnotations={
          mappingEditorActive
            ? mapEditor.annotations
            : authoring && stage.designMapActive && layers.active.mapAreas
              ? designMapEditor.annotations
              : stage.runFamily && mission.mapLoaded && layers.active.mapAreas
                ? runDisplayMapEditor.annotations
                : []
        }
        selectedMapAnnotationId={mappingEditorActive ? mapEditor.selectedAnnotationId : ""}
        mapRefined={false}
        editorBrush={
          mappingEditorActive && mapEditor.map && EDITOR_BRUSH_RING_COLORS[mapEditor.tool]
            ? {
              sizeCells: mapEditor.brushSize,
              color: EDITOR_BRUSH_RING_COLORS[mapEditor.tool],
            }
            : null
        }
        btLayer={authoring ? bt.waypointLayer : bt.runLayer}
        showMap={mappingEditorActive ? true : layers.active.map}
        showGlobalCostmap={mappingEditorActive ? false : layers.needsGlobalCostmap}
        showLocalCostmap={mappingEditorActive ? false : layers.needsLocalCostmap}
        showScan={mappingEditorActive ? false : layers.needsScan}
        showGlobalPlan={mappingEditorActive ? false : layers.needsPlan}
        showTf={mappingEditorActive ? false : layers.needsTf && layers.active.tf}
        showRobotModel={mappingEditorActive ? false : layers.needsRobotModel}
        interactionDisabled={
          !!interaction.busy
          || (authoring && !mission.documentReady)
          || (mappingEditorActive && mapEditor.busy)
          || (stage.designMapActive && designMapEditor.busy)
        }
        interactionMode={mappingEditorActive ? "view" : interaction.mode}
        editorActive={mappingEditorActive && !!mapEditor.map && mapEditor.tool !== "view"}
        editorPaintOnDrag
        editorAreaSelection={mappingEditorActive && mapEditor.tool === "label_marker"}
        fitContainer
        viewKey={mappingEditorActive
          ? `mission-editor:${mapEditor.selectedPath || "none"}`
          : authoring
            ? stage.designMapActive
              ? `mission-design:${designMapEditor.selectedPath || mission.designMapPath || "none"}`
              : "mission-design:none"
            : `mission:${mission.mapName}:${scene.displayedMap ? "ready" : "wait"}`}
        waitingLabel={mappingEditorActive
          ? "Load a map"
          : authoring
            ? stage.designMapActive ? "Loading selected map" : "Load a map"
            : stage.running
              ? "Waiting for /map"
              : runDisplayMapEditor.busy ? "Loading map" : "Load a mission to view the map"}
        onSpotClick={authoring && mission.documentReady && !bt.waypointLayer
          ? (mission.routeMode ? interaction.onRouteSpotClick : interaction.onSpotClick)
          : undefined}
        onBehaviorNodeClick={authoring && mission.documentReady && !bt.waypointLayer
          ? interaction.onBehaviorNodeClick
          : undefined}
        onMissionRouteSpotClick={authoring && mission.documentReady
          ? interaction.onRouteSpotClick
          : undefined}
        onMissionRouteMapClick={authoring && mission.documentReady
          ? interaction.onRouteMapClick
          : undefined}
        onSpotPoseChange={authoring && mission.documentReady && !mission.routeMode && !bt.waypointLayer
          ? interaction.onSpotPoseChange
          : undefined}
        onBehaviorNodePoseChange={authoring && mission.documentReady && !mission.routeMode && !bt.waypointLayer
          ? interaction.onBehaviorNodePoseChange
          : undefined}
        onEditorMapPoint={mapEditor.tool === "extend_area" || mapEditor.tool === "erase_area"
          ? mapEditor.editAreaAtMapPoint
          : mapEditor.editAtMapPoint}
        onEditorMapArea={mapEditor.tool === "label_marker"
          ? mapEditor.placeAnnotationAtMapArea
          : undefined}
        onMapClick={bt.waypointLayer ? undefined : interaction.onMapClick}
        onMapPose={bt.waypointLayer ? undefined : interaction.onMapPose}
        onBtLayerClose={authoring ? interaction.onBtLayerClose : undefined}
      />

      {authoring && !bt.waypointLayer && <DesignStageHud {...hud.design} />}
      {stage.id === STAGE_MAPPING && <MappingStageHud {...hud.mapping} />}
      {stage.id === STAGE_MAP_EDIT && <MapEditStageOverlay {...hud.mapEdit} />}
      {stage.id === STAGE_RUN && <RunStageHud {...hud.run} />}
      {stage.id === STAGE_NAVIGATE && <NavigationStageHud {...hud.navigate} />}
      {!bt.activeLayer && !mappingEditorActive && <LayersPopover layerToggles={layers.toggles} />}
    </section>
  );
}
