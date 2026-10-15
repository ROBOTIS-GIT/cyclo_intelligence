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
// Author: Howon Kim, Seongwoo Kim

"use client";

import { EMPTY_LIST } from './mapViewer/constants';
import { useRef, useState } from 'react';
import { useSelectedSpotScreenPosition } from './useSelectedSpotScreenPosition';
import { usesSg2TopDownMarker } from './robotMapMarker';
import * as THREE from 'three';
import { useMapRendererScene } from './mapViewer/useMapRendererScene';
import { useEditorBrushLayer } from './mapViewer/useEditorBrushLayer';
import { useMapViewport } from './mapViewer/useMapViewport';
import { useRobotSensorLayers } from './mapViewer/useRobotSensorLayers';
import { useMapNavigationLayers } from './mapViewer/useMapNavigationLayers';
import { useWaypointLayers } from './mapViewer/useWaypointLayers';
import { useMapPointerInteractions } from './mapViewer/useMapPointerInteractions';
import { WaypointBtFocusLayer } from './mapViewer/WaypointBtFocusLayer';
// Own the shared scene references once; layer hooks only update their own groups.
// Canvas features continue to use this stable public prop contract.
export function MapViewer({
  map,
  globalCostmap,
  localCostmap,
  scan,
  scanPose = null,
  pose,
  plan,
  goalPose,
  footprint,
  tf,
  showMap,
  showGlobalCostmap,
  showLocalCostmap,
  showScan,
  showGlobalPlan,
  showGoalPose,
  showTf,
  showRobotModel,
  showFootprint = showRobotModel,
  renderProfile = "default",
  robotType = "",
  interactionDisabled,
  interactionMode,
  editorActive,
  editorPaintOnDrag = true,
  editorAreaSelection = false,
  editorBrush = null,
  mapRefined = true,
  viewKey,
  waitingLabel = "Waiting for /map",
  fitContainer = false,
  spots = EMPTY_LIST,
  selectedSpotId = "",
  activeWaypointId = "",
  navigationDestination = null,
  missionFollowRobot = false,
  behaviorNodes = EMPTY_LIST,
  selectedBehaviorNodeId = "",
  behaviorPreviewNode = null,
  waypointColors = null,
  missionRouteOverlay = null,
  missionRouteOrder = EMPTY_LIST,
  missionRouteClosed = false,
  missionRouteMode = false,
  selectedMissionRouteSourceId = "",
  mapAnnotations = EMPTY_LIST,
  selectedMapAnnotationId = "",
  btLayer = null,
  onBtLayerClose,
  onSpotClick,
  onSelectedSpotScreenPositionChange,
  onRobotScreenPositionChange,
  onBehaviorNodeClick,
  onMissionRouteSpotClick,
  onMissionRouteMapClick,
  onSpotPoseChange,
  onBehaviorNodePoseChange,
  onEditorMapPoint,
  onEditorMapArea,
  onMapClick,
  onMapTap,
  onMapPose
}) {
  const btSpotId = btLayer?.spot?.id;
  const btSpotPose = btLayer?.spot?.pose;
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const {
    selectedSpotMarkerRef,
    publishSelectedSpotScreenPosition
  } = useSelectedSpotScreenPosition({
    cameraRef,
    containerRef,
    onChange: onSelectedSpotScreenPositionChange
  });
  const {
    selectedSpotMarkerRef: robotMarkerRef,
    publishSelectedSpotScreenPosition: publishRobotScreenPosition
  } = useSelectedSpotScreenPosition({
    cameraRef,
    containerRef,
    onChange: onRobotScreenPositionChange
  });
  const sg2MarkerRef = useRef(null);
  const showSg2TopDown = usesSg2TopDownMarker({
    renderProfile,
    robotType,
    showFootprint
  });
  const renderProfileRef = useRef(renderProfile);
  renderProfileRef.current = renderProfile;
  const controlsRef = useRef(null);
  const layersRef = useRef(null);
  const mapLayerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const fitMapKeyRef = useRef(null);
  const viewRollRef = useRef(0);
  const viewRotateDragRef = useRef(null);
  const btFocusKeyRef = useRef(null);
  const followRobotRef = useRef(false);
  const followPoseRef = useRef(null);
  const lastMotionPoseRef = useRef(null);
  const renderActiveUntilRef = useRef(0);
  const renderInteractionActiveRef = useRef(false);
  const renderInteractionUntilRef = useRef(0);
  // Keep active-waypoint pulses above the 10fps idle cadence: autonomy
  // uses 30fps; the default profile keeps the existing 60fps animation.
  const pulseActiveRef = useRef(false);
  pulseActiveRef.current = Boolean(activeWaypointId || missionRouteOverlay?.activeWaypointId);
  const latestFootprintRef = useRef(null);
  const tfSyncedFootprintRef = useRef(null);
  const [dragPreviewPose, setDragPreviewPose] = useState(null);
  const [nodeDragPreview, setNodeDragPreview] = useState(null);
  const [editorAreaPreview, setEditorAreaPreview] = useState(null);
  const [mapDragActive, setMapDragActive] = useState(false);
  const [viewerError, setViewerError] = useState(null);
  // Freeze each LaserScan in display coordinates until the next scan or map geometry change.
  const scanProjectionRef = useRef(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const pointerRef = useRef(new THREE.Vector2());
  const pointerDownRef = useRef(null);
  const backgroundTapRef = useRef(null);
  const editorPaintPointerRef = useRef(null);
  const editorMovePendingRef = useRef(null);
  const editorMoveRafRef = useRef(null);
  const editorAreaDragRef = useRef(null);
  const editorBrushLayerRef = useRef(null);
  // Split layer groups so high-frequency signals (TF pose, scan) never force
  // a rebuild of the expensive layers (costmap pixel planes, marker sprites).
  const costmapLayerRef = useRef(null);
  const globalCostmapTextureRef = useRef(null);
  const navPathLayerRef = useRef(null);
  const liveLayerRef = useRef(null);
  const scanLayerRef = useRef(null);
  const nodeDragRef = useRef(null);
  useMapRendererScene({
    containerRef,
    rendererRef,
    sceneRef,
    cameraRef,
    renderProfileRef,
    mapLayerRef,
    layersRef,
    editorBrushLayerRef,
    costmapLayerRef,
    navPathLayerRef,
    liveLayerRef,
    scanLayerRef,
    controlsRef,
    renderInteractionActiveRef,
    renderActiveUntilRef,
    renderInteractionUntilRef,
    animationFrameRef,
    followPoseRef,
    followRobotRef,
    pulseActiveRef,
    publishSelectedSpotScreenPosition,
    publishRobotScreenPosition,
    setViewerError,
    selectedSpotMarkerRef,
    robotMarkerRef,
    globalCostmapTextureRef,
    renderProfile
  });

  // Editor brush ring — geometry rebuilt only when the brush spec changes.
  useEditorBrushLayer({
    editorBrushLayerRef,
    map,
    editorBrush,
    rendererRef,
    cameraRef,
    pointerRef,
    raycasterRef
  });

  useMapViewport({
    rendererRef,
    controlsRef,
    interactionDisabled,
    editorActive,
    editorAreaSelection,
    interactionMode,
    mapDragActive,
    cameraRef,
    map,
    btSpotPose,
    btFocusKeyRef,
    viewKey,
    btSpotId,
    viewRollRef,
    followRobotRef,
    missionFollowRobot,
    renderActiveUntilRef,
    pose,
    followPoseRef,
    lastMotionPoseRef,
    dragPreviewPose,
    nodeDragPreview,
    editorAreaPreview,
    renderInteractionActiveRef
  });


  useRobotSensorLayers({
    latestFootprintRef,
    footprint,
    tfSyncedFootprintRef,
    tf,
    sceneRef,
    showSg2TopDown,
    sg2MarkerRef,
    robotMarkerRef,
    liveLayerRef,
    map,
    pose,
    showLocalCostmap,
    localCostmap,
    scan,
    scanPose,
    showScan,
    scanProjectionRef,
    scanLayerRef,
    showTf,
    showFootprint
  });

  // The map is a static layer. Keep its texture alive while TF, pose, scan,
  // plans, and robot markers update in the dynamic layer below.
  useMapNavigationLayers({
    mapLayerRef,
    showMap,
    map,
    mapRefined,
    costmapLayerRef,
    showGlobalCostmap,
    globalCostmap,
    globalCostmapTextureRef,
    navPathLayerRef,
    showGlobalPlan,
    plan,
    showGoalPose,
    goalPose
  });
  useWaypointLayers({
    sceneRef,
    layersRef,
    cameraRef,
    controlsRef,
    selectedSpotMarkerRef,
    map,
    waypointColors,
    dragPreviewPose,
    interactionMode,
    behaviorPreviewNode,
    spots,
    mapAnnotations,
    selectedMapAnnotationId,
    editorAreaPreview,
    nodeDragPreview,
    selectedSpotId,
    activeWaypointId,
    navigationDestination,
    missionRouteOrder,
    missionRouteClosed,
    selectedMissionRouteSourceId,
    missionRouteOverlay,
    behaviorNodes,
    selectedBehaviorNodeId,
    viewKey,
    fitMapKeyRef,
    btSpotPose,
    viewRollRef,
    publishSelectedSpotScreenPosition,
    btSpotId
  });




  useMapPointerInteractions({
    rendererRef,
    cameraRef,
    controlsRef,
    map,
    pointerRef,
    raycasterRef,
    layersRef,
    onSpotClick,
    onBehaviorNodeClick,
    spots,
    onSpotPoseChange,
    behaviorNodes,
    selectedBehaviorNodeId,
    onBehaviorNodePoseChange,
    pose,
    onEditorMapPoint,
    editorMoveRafRef,
    editorMovePendingRef,
    backgroundTapRef,
    interactionDisabled,
    editorActive,
    interactionMode,
    viewRotateDragRef,
    viewRollRef,
    renderInteractionActiveRef,
    pointerDownRef,
    setDragPreviewPose,
    editorAreaSelection,
    onEditorMapArea,
    editorAreaDragRef,
    setEditorAreaPreview,
    editorPaintOnDrag,
    editorPaintPointerRef,
    missionRouteMode,
    onMissionRouteSpotClick,
    nodeDragRef,
    onMissionRouteMapClick,
    onMapClick,
    onMapTap,
    setMapDragActive,
    setNodeDragPreview,
    renderActiveUntilRef,
    onMapPose,
    selectedSpotId
  });
  return <div className={`relative border rounded-md min-h-0 overflow-hidden ${fitContainer ? "h-full w-full" : ""}`} style={{
    ...(fitContainer ? {
      height: "100%",
      width: "100%"
    } : {
      aspectRatio: "1 / 1"
    }),
    backgroundColor: "var(--vscode-editor-background)",
    borderColor: "var(--vscode-panel-border)"
  }}>
      <div ref={containerRef} className="h-full w-full" />
      {viewerError && <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm pointer-events-none" style={{
      color: "var(--vscode-descriptionForeground)"
    }}>
          Map viewer unavailable: {viewerError}
        </div>}
      {!viewerError && showMap && !map && <div className="absolute inset-0 flex items-center justify-center text-sm pointer-events-none" style={{
      color: "var(--vscode-descriptionForeground)"
    }}>
          {waitingLabel}
        </div>}
      <WaypointBtFocusLayer layer={btLayer} onClose={onBtLayerClose} />
    </div>;
}
export { mapRenderIntervalMs } from "./mapViewer/cameraGeometry";
export { updateGlobalCostmapTexture } from "./mapViewer/occupancyTexture";
export { mapAreaPreviewCellIndices } from "./mapViewer/gridGeometry";
