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

import { useEffect } from 'react';
import { disposeObject } from './sceneObjects';
import { gridMeta } from './gridGeometry';
import { markerPalette } from './constants';
import { yawFromPose } from '../../../utils/navigationTf';
import { makeSpotMarker, makePoseMarker } from './waypointMarkers';
import { makeBehaviorNodeMarker } from './robotLayers';
import { makeMapAnnotationRegion, makeEditorAreaPreview } from './annotationRegions';
import * as THREE from 'three';
import { makeMissionRouteLine, makeMissionRouteBadge } from './routeMarkers';
import { addRouteOverlay } from '../addRouteOverlay';
import { createDestinationMapMarker } from '../destinationMapMarker';
import { fitCameraToMap } from './cameraGeometry';
export function useWaypointLayers({
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
}) {
  useEffect(() => {
    const scene = sceneRef.current;
    const layers = layersRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!scene || !layers || !camera || !controls) return;
    selectedSpotMarkerRef.current = null;
    disposeObject(layers);
    layers.clear();
    const meta = gridMeta(map);
    const waypointScale = meta?.resolution && (meta.resolution > 0) ? meta.resolution : 1;
    const waypointPalette = waypointColors ? {
      ...markerPalette,
      ...waypointColors
    } : markerPalette;
    if ((dragPreviewPose === null) || (dragPreviewPose === void 0) ? void 0 : dragPreviewPose.position) {
      const previewX = Number(dragPreviewPose.position.x ?? 0);
      const previewY = Number(dragPreviewPose.position.y ?? 0);
      const previewYaw = yawFromPose(dragPreviewPose);
      if (interactionMode === "spot") {
        layers.add(makeSpotMarker({
          id: "__waypoint_preview__",
          label: "Waypoint",
          pose: {
            x: previewX,
            y: previewY,
            yaw: previewYaw
          }
        }, true, waypointScale, false, waypointPalette));
      } else if ((interactionMode === "behavior") && behaviorPreviewNode) {
        layers.add(makeBehaviorNodeMarker({
          id: "__behavior_preview__",
          tag: behaviorPreviewNode.tag,
          label: behaviorPreviewNode.label || behaviorPreviewNode.tag,
          category: behaviorPreviewNode.category || "action",
          pose: {
            x: previewX,
            y: previewY,
            yaw: previewYaw
          }
        }, true));
      } else {
        layers.add(makePoseMarker(dragPreviewPose, interactionMode === "initial" ? 0x5b8266 : 0xc96442, 0.2));
      }
    }
    const spotById = new Map(spots.map(spot => [spot.id, spot]));
    const coveredAnnotationCells = new Set();
    mapAnnotations.forEach(annotation => {
      const marker = makeMapAnnotationRegion(annotation, map, coveredAnnotationCells, annotation.id === selectedMapAnnotationId);
      if (marker) layers.add(marker);
    });
    if (editorAreaPreview) {
      const preview = makeEditorAreaPreview(editorAreaPreview, map, coveredAnnotationCells);
      if (preview) layers.add(preview);
    }
    spots.forEach(spot => {
      const preview = ((nodeDragPreview === null || nodeDragPreview === void 0 ? void 0 : nodeDragPreview.type) === "spot") && (nodeDragPreview.id === spot.id) ? nodeDragPreview : null;
      const marker = makeSpotMarker(preview ? {
        ...spot,
        pose: {
          ...spot.pose,
          x: preview.x,
          y: preview.y,
          yaw: preview.yaw
        }
      } : spot, spot.id === selectedSpotId, waypointScale, spot.id === activeWaypointId, waypointPalette, (navigationDestination?.scope === "mission") || (spot.id !== navigationDestination?.id));
      if (spot.id === selectedSpotId) selectedSpotMarkerRef.current = marker;
      if (marker) layers.add(marker);
    });
    if (missionRouteOrder.length >= 2) {
      const routePoints = [...missionRouteOrder].sort((a, b) => a.order - b.order).map(({
        id
      }) => spotById.get(id)).filter(spot => (spot === null) || (spot === void 0) ? void 0 : spot.pose).map(spot => new THREE.Vector3(Number(spot.pose.x ?? 0), Number(spot.pose.y ?? 0), 0.02));
      if (missionRouteClosed && (routePoints.length > 1)) {
        routePoints.push(routePoints[0].clone());
      }
      const routeLine = makeMissionRouteLine(routePoints, waypointPalette.idle, waypointScale);
      if (routeLine) layers.add(routeLine);
    }
    missionRouteOrder.forEach(({
      id,
      order
    }) => {
      const spot = spotById.get(id);
      const badge = makeMissionRouteBadge(spot, order, id === selectedMissionRouteSourceId, waypointScale, waypointPalette);
      if (badge) layers.add(badge);
    });
    addRouteOverlay(layers, missionRouteOverlay, waypointScale, markerPalette, {
      marker: (spot, ...args) => makeSpotMarker(spot, ...args, (navigationDestination?.scope !== "mission") || (spot.id !== navigationDestination?.id)),
      badge: makeMissionRouteBadge,
      line: makeMissionRouteLine
    });
    const destinationMarker = createDestinationMapMarker(navigationDestination, waypointScale);
    if (destinationMarker) layers.add(destinationMarker);
    behaviorNodes.forEach(node => {
      const preview = ((nodeDragPreview === null || nodeDragPreview === void 0 ? void 0 : nodeDragPreview.type) === "behaviorNode") && (nodeDragPreview.id === node.id) ? nodeDragPreview : null;
      const marker = makeBehaviorNodeMarker(preview ? {
        ...node,
        pose: {
          ...node.pose,
          x: preview.x,
          y: preview.y,
          yaw: preview.yaw
        }
      } : node, node.id === selectedBehaviorNodeId);
      if (marker) layers.add(marker);
    });
    const fitKey = (viewKey !== null) && (viewKey !== void 0) ? viewKey : "default";
    if (meta && (fitMapKeyRef.current !== fitKey) && !btSpotPose) {
      fitCameraToMap(camera, controls, meta, viewRollRef.current);
      fitMapKeyRef.current = fitKey;
    }
    publishSelectedSpotScreenPosition();
  }, [publishSelectedSpotScreenPosition, selectedSpotMarkerRef, dragPreviewPose, editorAreaPreview, nodeDragPreview, interactionMode, missionRouteOrder, missionRouteOverlay, waypointColors, missionRouteClosed, mapAnnotations, selectedMapAnnotationId, selectedMissionRouteSourceId, behaviorPreviewNode, map, behaviorNodes, btSpotId, btSpotPose, selectedBehaviorNodeId, selectedSpotId, activeWaypointId, navigationDestination, spots, viewKey, sceneRef, layersRef, cameraRef, controlsRef, fitMapKeyRef, viewRollRef]);
}
